//
//  File.swift
//  Llama2
//
//  Created by Ronald Mannak on 6/22/25.
//

import Foundation

// MARK: - ProbIndex Structure

/// Structure used when sorting probabilities during top-p sampling
struct ProbIndex {
    let prob: Float
    let index: Int
}

// MARK: - Sampler

struct Sampler {
    private let temperature: Float
    private let topP: Float
    private var rng: any RandomNumberGenerator
    private var probIndexBuffer: [ProbIndex]
    
    init(temperature: Float, topp: Float, seed: UInt64, vocabSize: Int = 32000) {
        self.temperature = temperature
        self.topP = topp
        // Use seeded RNG if seed is provided, otherwise use system RNG
        if seed == 0 {
            self.rng = SystemRandomNumberGenerator()
        } else {
            self.rng = SeededRandomNumberGenerator(seed: seed)
        }
        // Pre-allocate buffer for top-p sampling
        self.probIndexBuffer = Array(repeating: ProbIndex(prob: 0.0, index: 0), count: vocabSize)
    }
    
    /// Sample the token given the logits and hyperparameters
    /// - Parameter logits: Array of logits from the model
    /// - Returns: Index of the sampled token
    mutating func sample(logits: [Float]) -> Int {
        let vocabSize = logits.count
        
        if temperature == 0.0 {
            // Greedy argmax sampling: take the token with the highest probability
            return sampleArgmax(probabilities: logits, n: vocabSize)
        } else {
            // Apply temperature to the logits
            var temperatureAdjustedLogits = logits
            for i in 0..<vocabSize {
                temperatureAdjustedLogits[i] /= temperature
            }
            
            // Apply softmax to get probabilities
            softmax(values: &temperatureAdjustedLogits, size: vocabSize)
            
            // Generate random value for sampling
            let coin = Float.random(in: 0..<1, using: &rng)
            
            // Sample from the distribution
            if topP <= 0 || topP >= 1 {
                // Simply sample from the predicted probability distribution
                return sampleMult(probabilities: temperatureAdjustedLogits, n: vocabSize, coin: coin)
            } else {
                // Top-p (nucleus) sampling, clamping the least likely tokens to zero
                return sampleTopp(probabilities: temperatureAdjustedLogits, n: vocabSize, topp: topP, coin: coin)
            }
        }
    }
    
    // MARK: - Sampling Methods
    
    /// Return the index that has the highest probability
    /// - Parameters:
    ///   - probabilities: Array of probabilities
    ///   - n: Size of the array
    /// - Returns: Index of the maximum probability
    private func sampleArgmax(probabilities: [Float], n: Int) -> Int {
        var maxIndex = 0
        var maxProb = probabilities[0]
        
        for i in 1..<n {
            if probabilities[i] > maxProb {
                maxIndex = i
                maxProb = probabilities[i]
            }
        }
        return maxIndex
    }
    
    /// Sample index from probabilities (they must sum to 1!)
    /// - Parameters:
    ///   - probabilities: Array of probabilities that sum to 1
    ///   - n: Size of the array
    ///   - coin: Random number in [0, 1)
    /// - Returns: Sampled index
    private func sampleMult(probabilities: [Float], n: Int, coin: Float) -> Int {
        var cdf: Float = 0.0
        
        for i in 0..<n {
            cdf += probabilities[i]
            if coin < cdf {
                return i
            }
        }
        return n - 1 // In case of rounding errors
    }
    
    /// Top-p sampling (or "nucleus sampling") samples from the smallest set of
    /// tokens that exceed probability topp
    /// - Parameters:
    ///   - probabilities: Array of probabilities
    ///   - n: Size of the array
    ///   - topp: Top-p threshold
    ///   - coin: Random number in [0, 1)
    /// - Returns: Sampled index
    private mutating func sampleTopp(probabilities: [Float], n: Int, topp: Float, coin: Float) -> Int {
        var n0 = 0
        
        // Quicksort indices in descending order of probabilities
        // Values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        let cutoff = (1.0 - topp) / Float(n - 1)
        
        for i in 0..<n {
            if probabilities[i] >= cutoff {
                probIndexBuffer[n0] = ProbIndex(prob: probabilities[i], index: i)
                n0 += 1
            }
        }
        
        // Sort by probability in descending order
        probIndexBuffer[0..<n0].sort { $0.prob > $1.prob }
        
        // Truncate the list where cumulative probability exceeds topp
        var cumulativeProb: Float = 0.0
        var lastIdx = n0 - 1 // In case of rounding errors consider all elements
        
        for i in 0..<n0 {
            cumulativeProb += probIndexBuffer[i].prob
            if cumulativeProb > topp {
                lastIdx = i
                break // We've exceeded topp by including lastIdx
            }
        }
        
        // Sample from the truncated list
        let r = coin * cumulativeProb
        var cdf: Float = 0.0
        
        for i in 0...lastIdx {
            cdf += probIndexBuffer[i].prob
            if r < cdf {
                return probIndexBuffer[i].index
            }
        }
        
        return probIndexBuffer[lastIdx].index // In case of rounding errors
    }
}

// MARK: - Seeded Random Number Generator

/// A deterministic random number generator using a simple linear congruential generator
/// This provides reproducible results when the same seed is used
struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64
    
    /// Creates a new seeded random number generator
    /// - Parameter seed: The seed value. Use 0 for non-deterministic behavior
    init(seed: UInt64) {
        self.state = seed == 0 ? UInt64.random(in: 0...UInt64.max) : seed
    }
    
    /// Generates the next random number using Linear Congruential Generator (LCG)
    /// 
    /// Uses the formula: next_state = (current_state * multiplier + increment) mod 2^64
    /// The magic numbers are carefully chosen constants from the MMIX LCG by Donald Knuth:
    /// - 6364136223846793005: Multiplier constant (a) - provides maximum period length
    /// - 1442695040888963407: Increment constant (c) - ensures good statistical properties
    /// 
    /// These constants have been extensively tested and provide:
    /// - Maximum period length (2^64)
    /// - Good distribution properties
    /// - No obvious patterns or correlations
    /// - Fast computation (single multiplication + addition)
    /// 
    /// Note: This is NOT cryptographically secure, but suitable for ML sampling applications.
    /// - Returns: Next 64-bit random number
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}

/*
 Port this C code to this Swift project. Create stub functions for the tokenizer, sampler, and transformer and other custom objects or functions. Make sure the code conforms to modern Swift 6.1 standards and follows Swift-idiomatic patterns including automatic memory management, value types where appropriate, and Swift's built-in data structures like Data for file handling. Avoid manual memory management and low-level system calls in favor of Swift's safe, automatic memory management.
 
 // ----------------------------------------------------------------------------
 // The Sampler, which takes logits and returns a sampled token
 // sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

 typedef struct {
     float prob;
     int index;
 } ProbIndex; // struct used when sorting probabilities during top-p sampling

 typedef struct {
     int vocab_size;
     ProbIndex* probindex; // buffer used in top-p sampling
     float temperature;
     float topp;
     unsigned long long rng_state;
 } Sampler;

 int sample_argmax(float* probabilities, int n) {
     // return the index that has the highest probability
     int max_i = 0;
     float max_p = probabilities[0];
     for (int i = 1; i < n; i++) {
         if (probabilities[i] > max_p) {
             max_i = i;
             max_p = probabilities[i];
         }
     }
     return max_i;
 }

 int sample_mult(float* probabilities, int n, float coin) {
     // sample index from probabilities (they must sum to 1!)
     // coin is a random number in [0, 1), usually from random_f32()
     float cdf = 0.0f;
     for (int i = 0; i < n; i++) {
         cdf += probabilities[i];
         if (coin < cdf) {
             return i;
         }
     }
     return n - 1; // in case of rounding errors
 }

 int compare(const void* a, const void* b) {
     ProbIndex* a_ = (ProbIndex*) a;
     ProbIndex* b_ = (ProbIndex*) b;
     if (a_->prob > b_->prob) return -1;
     if (a_->prob < b_->prob) return 1;
     return 0;
 }

 int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
     // top-p sampling (or "nucleus sampling") samples from the smallest set of
     // tokens that exceed probability topp. This way we never sample tokens that
     // have very low probabilities and are less likely to go "off the rails".
     // coin is a random number in [0, 1), usually from random_f32()

     int n0 = 0;
     // quicksort indices in descending order of probabilities
     // values smaller than (1 - topp) / (n - 1) cannot be part of the result
     // so for efficiency we crop these out as candidates before sorting
     const float cutoff = (1.0f - topp) / (n - 1);
     for (int i = 0; i < n; i++) {
         if (probabilities[i] >= cutoff) {
             probindex[n0].index = i;
             probindex[n0].prob = probabilities[i];
             n0++;
         }
     }
     qsort(probindex, n0, sizeof(ProbIndex), compare);

     // truncate the list where cumulative probability exceeds topp
     float cumulative_prob = 0.0f;
     int last_idx = n0 - 1; // in case of rounding errors consider all elements
     for (int i = 0; i < n0; i++) {
         cumulative_prob += probindex[i].prob;
         if (cumulative_prob > topp) {
             last_idx = i;
             break; // we've exceeded topp by including last_idx
         }
     }

     // sample from the truncated list
     float r = coin * cumulative_prob;
     float cdf = 0.0f;
     for (int i = 0; i <= last_idx; i++) {
         cdf += probindex[i].prob;
         if (r < cdf) {
             return probindex[i].index;
         }
     }
     return probindex[last_idx].index; // in case of rounding errors
 }

 void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
     sampler->vocab_size = vocab_size;
     sampler->temperature = temperature;
     sampler->topp = topp;
     sampler->rng_state = rng_seed;
     // buffer only used with nucleus sampling; may not need but it's ~small
     sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
 }

 void free_sampler(Sampler* sampler) {
     free(sampler->probindex);
 }

 unsigned int random_u32(unsigned long long *state) {
     // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
     *state ^= *state >> 12;
     *state ^= *state << 25;
     *state ^= *state >> 27;
     return (*state * 0x2545F4914F6CDD1Dull) >> 32;
 }
 float random_f32(unsigned long long *state) { // random float32 in [0,1)
     return (random_u32(state) >> 8) / 16777216.0f;
 }

 int sample(Sampler* sampler, float* logits) {
     // sample the token given the logits and some hyperparameters
     int next;
     if (sampler->temperature == 0.0f) {
         // greedy argmax sampling: take the token with the highest probability
         next = sample_argmax(logits, sampler->vocab_size);
     } else {
         // apply the temperature to the logits
         for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
         // apply softmax to the logits to get the probabilities for next token
         softmax(logits, sampler->vocab_size);
         // flip a (float) coin (this is our source of entropy for sampling)
         float coin = random_f32(&sampler->rng_state);
         // we sample from this distribution to get the next token
         if (sampler->topp <= 0 || sampler->topp >= 1) {
             // simply sample from the predicted probability distribution
             next = sample_mult(logits, sampler->vocab_size, coin);
         } else {
             // top-p (nucleus) sampling, clamping the least likely tokens to zero
             next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
         }
     }
     return next;
 }
 */
