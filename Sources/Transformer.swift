//
//  File.swift
//  Llama2
//
//  Created by Ronald Mannak on 6/22/25.
//

import Foundation

// MARK: - Transformer model

struct Config {
    let dim: Int // transformer dimension
    let hiddenDim: Int // for ffn layers
    let numLayers: Int // number of layers
    let numHeads: Int // number of query heads
    let numKvHeads: Int // number of key/value heads (can be < query heads because of multiquery)
    let vocabSize: Int // vocabulary size, usually 256 (byte-level)
    let seqLen: Int // max sequence length
}

extension Config {
    // Convenience initializer for backward compatibility
    init(vocabSize: Int, seqLen: Int, embeddingDim: Int = 4096, numLayers: Int = 32, numHeads: Int = 32) {
        self.dim = embeddingDim
        self.hiddenDim = embeddingDim * 4 // Common ratio for FFN
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.numKvHeads = numHeads // Default to same as query heads
        self.vocabSize = vocabSize
        self.seqLen = seqLen
    }
}

struct TransformerWeights {
    // token embedding table
    let tokenEmbeddingTable: [Float] // (vocab_size, dim)
    // weights for rmsnorms
    let rmsAttWeight: [Float] // (layer, dim) rmsnorm weights
    let rmsFfnWeight: [Float] // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    let wq: [Float] // (layer, dim, n_heads * head_size)
    let wk: [Float] // (layer, dim, n_kv_heads * head_size)
    let wv: [Float] // (layer, dim, n_kv_heads * head_size)
    let wo: [Float] // (layer, n_heads * head_size, dim)
    // weights for ffn
    let w1: [Float] // (layer, hidden_dim, dim)
    let w2: [Float] // (layer, dim, hidden_dim)
    let w3: [Float] // (layer, hidden_dim, dim)
    // final rmsnorm
    let rmsFinalWeight: [Float] // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    let wcls: [Float]? // (vocab_size, dim) or nil if shared weights
}

// MARK: - Weight Mapping and Checkpoint Reading

extension TransformerWeights {
    /// Maps binary data to weight arrays, equivalent to C's memory_map_weights
    static func mapFromData(_ data: Data, config: Config, sharedWeights: Bool) throws -> TransformerWeights {
        let headSize = config.dim / config.numHeads
        
        // Convert Data to [Float] for easier manipulation
        let floatCount = data.count / MemoryLayout<Float>.size
        let floats = data.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self).prefix(floatCount))
        }
        
        var ptr = 0
        
        // Helper function to safely extract array slice
        func extractArray(_ count: Int) throws -> [Float] {
            guard ptr + count <= floats.count else {
                throw Llama2Error.invalidParameter("Insufficient data for weight matrix")
            }
            let result = Array(floats[ptr..<(ptr + count)])
            ptr += count
            return result
        }
        
        // Map weights in the same order as C code
        let tokenEmbeddingTable = try extractArray(config.vocabSize * config.dim)
        let rmsAttWeight = try extractArray(config.numLayers * config.dim)
        let wq = try extractArray(config.numLayers * config.dim * (config.numHeads * headSize))
        let wk = try extractArray(config.numLayers * config.dim * (config.numKvHeads * headSize))
        let wv = try extractArray(config.numLayers * config.dim * (config.numKvHeads * headSize))
        let wo = try extractArray(config.numLayers * (config.numHeads * headSize) * config.dim)
        let rmsFfnWeight = try extractArray(config.numLayers * config.dim)
        let w1 = try extractArray(config.numLayers * config.dim * config.hiddenDim)
        let w2 = try extractArray(config.numLayers * config.hiddenDim * config.dim)
        let w3 = try extractArray(config.numLayers * config.dim * config.hiddenDim)
        let rmsFinalWeight = try extractArray(config.dim)
        
        // Skip RoPE frequency tables (freq_cis_real and freq_cis_imag)
        ptr += config.seqLen * headSize / 2 // freq_cis_real
        ptr += config.seqLen * headSize / 2 // freq_cis_imag
        
        // Handle classifier weights
        let wcls: [Float]?
        if sharedWeights {
            wcls = nil // Use token embedding table for shared weights
        } else {
            wcls = try extractArray(config.vocabSize * config.dim)
        }
        
        return TransformerWeights(
            tokenEmbeddingTable: tokenEmbeddingTable,
            rmsAttWeight: rmsAttWeight,
            rmsFfnWeight: rmsFfnWeight,
            wq: wq,
            wk: wk,
            wv: wv,
            wo: wo,
            w1: w1,
            w2: w2,
            w3: w3,
            rmsFinalWeight: rmsFinalWeight,
            wcls: wcls
        )
    }
}

struct RunState {
    // current wave of activations
    var x: [Float] // activation at current time stamp (dim,)
    var xb: [Float] // same, but inside a residual branch (dim,)
    var xb2: [Float] // an additional buffer just for convenience (dim,)
    var hb: [Float] // buffer for hidden dimension in the ffn (hidden_dim,)
    var hb2: [Float] // buffer for hidden dimension in the ffn (hidden_dim,)
    var q: [Float] // query (dim,)
    var k: [Float] // key (dim,)
    var v: [Float] // value (dim,)
    var att: [Float] // buffer for scores/attention values (n_heads, seq_len)
    var logits: [Float] // output logits
    // kv cache
    var keyCache: [Float] // (layer, seq_len, dim)
    var valueCache: [Float] // (layer, seq_len, dim)
    
    init(config: Config) {
        let dim = config.dim
        let hiddenDim = config.hiddenDim
        let numLayers = config.numLayers
        let numHeads = config.numHeads
        let seqLen = config.seqLen
        let kvDim = (dim * config.numKvHeads) / numHeads // KV cache dimension
        
        self.x = Array(repeating: 0.0, count: dim)
        self.xb = Array(repeating: 0.0, count: dim)
        self.xb2 = Array(repeating: 0.0, count: dim)
        self.hb = Array(repeating: 0.0, count: hiddenDim)
        self.hb2 = Array(repeating: 0.0, count: hiddenDim)
        self.q = Array(repeating: 0.0, count: dim)
        self.k = Array(repeating: 0.0, count: dim)
        self.v = Array(repeating: 0.0, count: dim)
        self.att = Array(repeating: 0.0, count: numHeads * seqLen)
        self.logits = Array(repeating: 0.0, count: config.vocabSize)
        self.keyCache = Array(repeating: 0.0, count: numLayers * seqLen * kvDim)
        self.valueCache = Array(repeating: 0.0, count: numLayers * seqLen * kvDim)
    }
}

class Transformer {
    let config: Config // the hyperparameters of the architecture (the blueprint)
    let weights: TransformerWeights // the weights of the model
    var state: RunState // buffers for the "wave" of activations in the forward pass
    
    init(checkpointPath: String) throws {
        
        // Sanity check
        guard FileManager.default.fileExists(atPath: checkpointPath) else {
            throw Llama2Error.fileNotFound(checkpointPath)
        }
        let (config, weights) = try Self.readCheckpoint(from: checkpointPath)
                
        self.config = config
        self.weights = weights
        self.state = RunState(config: config)
    }
    
    /// Reads a checkpoint file and returns the config and weights
    private static func readCheckpoint(from path: String) throws -> (config: Config, weights: TransformerWeights) {
        let fileURL = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: fileURL)
        
        // Read config from the beginning of the file
        let configSize = MemoryLayout<Config>.size
        guard data.count >= configSize else {
            throw Llama2Error.invalidParameter("File too small to contain config")
        }
        
        let configData = data.prefix(configSize)
        let config = configData.withUnsafeBytes { bytes in
            bytes.load(as: Config.self)
        }
        
        // Check for shared weights (negative vocab size indicates unshared weights)
        let sharedWeights = config.vocabSize > 0
        let actualVocabSize = abs(config.vocabSize)
        
        // Create a new config with the corrected vocab size
        let correctedConfig = Config(
            dim: config.dim,
            hiddenDim: config.hiddenDim,
            numLayers: config.numLayers,
            numHeads: config.numHeads,
            numKvHeads: config.numKvHeads,
            vocabSize: actualVocabSize,
            seqLen: config.seqLen
        )
        
        // Get the weight data (everything after the config)
        let weightData = data.dropFirst(configSize)
        
        // Map the weights from the weight data
        let weights = try TransformerWeights.mapFromData(weightData, config: correctedConfig, sharedWeights: sharedWeights)
        
        return (correctedConfig, weights)
    }
    
    /// Forward pass through the transformer
    /// - Parameters:
    ///   - token: Input token ID
    ///   - pos: Position in the sequence
    /// - Returns: Logits array for next token prediction
    func forward(token: Int, pos: Int) -> [Float] {
        // Convenience variables
        let dim = config.dim
        let kvDim = (config.dim * config.numKvHeads) / config.numHeads
        let kvMul = config.numHeads / config.numKvHeads // integer multiplier of the kv sharing in multiquery
        let hiddenDim = config.hiddenDim
        let headSize = dim / config.numHeads
        
        // Copy the token embedding into x
        let tokenOffset = token * dim
        for i in 0..<dim {
            state.x[i] = weights.tokenEmbeddingTable[tokenOffset + i]
        }
        
        // Forward all the layers
        for l in 0..<config.numLayers {
            // Attention rmsnorm
            let rmsAttOffset = l * dim
            rmsnorm(output: &state.xb, input: state.x, weight: Array(weights.rmsAttWeight[rmsAttOffset..<(rmsAttOffset + dim)]), size: dim)
            
            // Key and value point to the kv cache
            let loff = l * config.seqLen * kvDim // kv cache layer offset for convenience
            let kOffset = loff + pos * kvDim
            let vOffset = loff + pos * kvDim
            
            // qkv matmuls for this position
            let wqOffset = l * dim * dim
            let wkOffset = l * dim * kvDim
            let wvOffset = l * dim * kvDim
            
            matmul(output: &state.q, input: state.xb, weights: Array(weights.wq[wqOffset..<(wqOffset + dim * dim)]), n: dim, d: dim)
            matmul(output: &state.k, input: state.xb, weights: Array(weights.wk[wkOffset..<(wkOffset + dim * kvDim)]), n: dim, d: kvDim)
            matmul(output: &state.v, input: state.xb, weights: Array(weights.wv[wvOffset..<(wvOffset + dim * kvDim)]), n: dim, d: kvDim)
            
            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for i in stride(from: 0, to: dim, by: 2) {
                let headDim = i % headSize
                let freq = 1.0 / pow(10000.0, Double(headDim) / Double(headSize))
                let val = Double(pos) * freq
                let fcr = cos(val)
                let fci = sin(val)
                let rotn = i < kvDim ? 2 : 1 // how many vectors? 2 = q & k, 1 = q only
                
                for v in 0..<rotn {
                    if v == 0 {
                        // Rotate query vector
                        let v0 = state.q[i]
                        let v1 = state.q[i + 1]
                        state.q[i] = Float(Double(v0) * fcr - Double(v1) * fci)
                        state.q[i + 1] = Float(Double(v0) * fci + Double(v1) * fcr)
                    } else {
                        // Rotate key vector
                        let v0 = state.k[i]
                        let v1 = state.k[i + 1]
                        state.k[i] = Float(Double(v0) * fcr - Double(v1) * fci)
                        state.k[i + 1] = Float(Double(v0) * fci + Double(v1) * fcr)
                    }
                }
            }
            
            // Store k and v in the cache
            for i in 0..<kvDim {
                state.keyCache[kOffset + i] = state.k[i]
                state.valueCache[vOffset + i] = state.v[i]
            }
            
            // Multihead attention. iterate over all heads
            for h in 0..<config.numHeads {
                // Get the query vector for this head
                let qOffset = h * headSize
                let q = Array(state.q[qOffset..<(qOffset + headSize)])
                
                // Attention scores for this head
                let attOffset = h * config.seqLen
                var att = Array(state.att[attOffset..<(attOffset + config.seqLen)])
                
                // Iterate over all timesteps, including the current one
                for t in 0...pos {
                    // Get the key vector for this head and at this timestep
                    let kCacheOffset = loff + t * kvDim + (h / kvMul) * headSize
                    let k = Array(state.keyCache[kCacheOffset..<(kCacheOffset + headSize)])
                    
                    // Calculate the attention score as the dot product of q and k
                    var score: Float = 0.0
                    for i in 0..<headSize {
                        score += q[i] * k[i]
                    }
                    score /= sqrt(Float(headSize))
                    
                    // Save the score to the attention buffer
                    att[t] = score
                }
                
                // Softmax the scores to get attention weights, from 0..pos inclusively
                softmax(values: &att, size: pos + 1)
                
                // Update the attention buffer
                for t in 0...pos {
                    state.att[attOffset + t] = att[t]
                }
                
                // Weighted sum of the values, store back into xb
                let xbOffset = h * headSize
                for i in 0..<headSize {
                    state.xb[xbOffset + i] = 0.0
                }
                
                for t in 0...pos {
                    // Get the value vector for this head and at this timestep
                    let vCacheOffset = loff + t * kvDim + (h / kvMul) * headSize
                    let v = Array(state.valueCache[vCacheOffset..<(vCacheOffset + headSize)])
                    
                    // Get the attention weight for this timestep
                    let a = att[t]
                    
                    // Accumulate the weighted value into xb
                    for i in 0..<headSize {
                        state.xb[xbOffset + i] += a * v[i]
                    }
                }
            }
            
            // Final matmul to get the output of the attention
            let woOffset = l * dim * dim
            matmul(output: &state.xb2, input: state.xb, weights: Array(weights.wo[woOffset..<(woOffset + dim * dim)]), n: dim, d: dim)
            
            // Residual connection back into x
            for i in 0..<dim {
                state.x[i] += state.xb2[i]
            }
            
            // FFN rmsnorm
            let rmsFfnOffset = l * dim
            rmsnorm(output: &state.xb, input: state.x, weight: Array(weights.rmsFfnWeight[rmsFfnOffset..<(rmsFfnOffset + dim)]), size: dim)
            
            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // First calculate self.w1(x) and self.w3(x)
            let w1Offset = l * dim * hiddenDim
            let w3Offset = l * dim * hiddenDim
            
            matmul(output: &state.hb, input: state.xb, weights: Array(weights.w1[w1Offset..<(w1Offset + dim * hiddenDim)]), n: dim, d: hiddenDim)
            matmul(output: &state.hb2, input: state.xb, weights: Array(weights.w3[w3Offset..<(w3Offset + dim * hiddenDim)]), n: dim, d: hiddenDim)
            
            // SwiGLU non-linearity
            for i in 0..<hiddenDim {
                var val = state.hb[i]
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= (1.0 / (1.0 + exp(-val)))
                // Elementwise multiply with w3(x)
                val *= state.hb2[i]
                state.hb[i] = val
            }
            
            // Final matmul to get the output of the ffn
            let w2Offset = l * hiddenDim * dim
            matmul(output: &state.xb, input: state.hb, weights: Array(weights.w2[w2Offset..<(w2Offset + hiddenDim * dim)]), n: hiddenDim, d: dim)
            
            // Residual connection
            for i in 0..<dim {
                state.x[i] += state.xb[i]
            }
        }
        
        // Final rmsnorm
        rmsnorm(output: &state.x, input: state.x, weight: weights.rmsFinalWeight, size: dim)
        
        // Classifier into logits
        if let wcls = weights.wcls {
            // Use separate classifier weights
            matmul(output: &state.logits, input: state.x, weights: wcls, n: dim, d: config.vocabSize)
        } else {
            // Use shared weights (token embedding table)
            matmul(output: &state.logits, input: state.x, weights: weights.tokenEmbeddingTable, n: dim, d: config.vocabSize)
        }
        
        return state.logits
    }
}


/* Original C code

 typedef struct {
     int dim; // transformer dimension
     int hidden_dim; // for ffn layers
     int n_layers; // number of layers
     int n_heads; // number of query heads
     int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
     int vocab_size; // vocabulary size, usually 256 (byte-level)
     int seq_len; // max sequence length
 } Config;

 typedef struct {
     // token embedding table
     float* token_embedding_table;    // (vocab_size, dim)
     // weights for rmsnorms
     float* rms_att_weight; // (layer, dim) rmsnorm weights
     float* rms_ffn_weight; // (layer, dim)
     // weights for matmuls. note dim == n_heads * head_size
     float* wq; // (layer, dim, n_heads * head_size)
     float* wk; // (layer, dim, n_kv_heads * head_size)
     float* wv; // (layer, dim, n_kv_heads * head_size)
     float* wo; // (layer, n_heads * head_size, dim)
     // weights for ffn
     float* w1; // (layer, hidden_dim, dim)
     float* w2; // (layer, dim, hidden_dim)
     float* w3; // (layer, hidden_dim, dim)
     // final rmsnorm
     float* rms_final_weight; // (dim,)
     // (optional) classifier weights for the logits, on the last layer
     float* wcls;
 } TransformerWeights;

 typedef struct {
     // current wave of activations
     float *x; // activation at current time stamp (dim,)
     float *xb; // same, but inside a residual branch (dim,)
     float *xb2; // an additional buffer just for convenience (dim,)
     float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
     float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
     float *q; // query (dim,)
     float *k; // key (dim,)
     float *v; // value (dim,)
     float *att; // buffer for scores/attention values (n_heads, seq_len)
     float *logits; // output logits
     // kv cache
     float* key_cache;   // (layer, seq_len, dim)
     float* value_cache; // (layer, seq_len, dim)
 } RunState;


 float* forward(Transformer* transformer, int token, int pos) {

     // a few convenience variables
     Config* p = &transformer->config;
     TransformerWeights* w = &transformer->weights;
     RunState* s = &transformer->state;
     float *x = s->x;
     int dim = p->dim;
     int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
     int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
     int hidden_dim =  p->hidden_dim;
     int head_size = dim / p->n_heads;

     // copy the token embedding into x
     float* content_row = w->token_embedding_table + token * dim;
     memcpy(x, content_row, dim*sizeof(*x));

     // forward all the layers
     for(unsigned long long l = 0; l < p->n_layers; l++) {

         // attention rmsnorm
         rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

         // key and value point to the kv cache
         int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
         s->k = s->key_cache + loff + pos * kv_dim;
         s->v = s->value_cache + loff + pos * kv_dim;

         // qkv matmuls for this position
         matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
         matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
         matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

         // RoPE relative positional encoding: complex-valued rotate q and k in each head
         for (int i = 0; i < dim; i+=2) {
             int head_dim = i % head_size;
             float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
             float val = pos * freq;
             float fcr = cosf(val);
             float fci = sinf(val);
             int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
             for (int v = 0; v < rotn; v++) {
                 float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                 float v0 = vec[i];
                 float v1 = vec[i+1];
                 vec[i]   = v0 * fcr - v1 * fci;
                 vec[i+1] = v0 * fci + v1 * fcr;
             }
         }

         // multihead attention. iterate over all heads
         int h;
         #pragma omp parallel for private(h)
         for (h = 0; h < p->n_heads; h++) {
             // get the query vector for this head
             float* q = s->q + h * head_size;
             // attention scores for this head
             float* att = s->att + h * p->seq_len;
             // iterate over all timesteps, including the current one
             for (int t = 0; t <= pos; t++) {
                 // get the key vector for this head and at this timestep
                 float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                 // calculate the attention score as the dot product of q and k
                 float score = 0.0f;
                 for (int i = 0; i < head_size; i++) {
                     score += q[i] * k[i];
                 }
                 score /= sqrtf(head_size);
                 // save the score to the attention buffer
                 att[t] = score;
             }

             // softmax the scores to get attention weights, from 0..pos inclusively
             softmax(att, pos + 1);

             // weighted sum of the values, store back into xb
             float* xb = s->xb + h * head_size;
             memset(xb, 0, head_size * sizeof(float));
             for (int t = 0; t <= pos; t++) {
                 // get the value vector for this head and at this timestep
                 float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                 // get the attention weight for this timestep
                 float a = att[t];
                 // accumulate the weighted value into xb
                 for (int i = 0; i < head_size; i++) {
                     xb[i] += a * v[i];
                 }
             }
         }

         // final matmul to get the output of the attention
         matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

         // residual connection back into x
         for (int i = 0; i < dim; i++) {
             x[i] += s->xb2[i];
         }

         // ffn rmsnorm
         rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

         // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
         // first calculate self.w1(x) and self.w3(x)
         matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
         matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

         // SwiGLU non-linearity
         for (int i = 0; i < hidden_dim; i++) {
             float val = s->hb[i];
             // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
             val *= (1.0f / (1.0f + expf(-val)));
             // elementwise multiply with w3(x)
             val *= s->hb2[i];
             s->hb[i] = val;
         }

         // final matmul to get the output of the ffn
         matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

         // residual connection
         for (int i = 0; i < dim; i++) {
             x[i] += s->xb[i];
         }
     }

     // final rmsnorm
     rmsnorm(x, x, w->rms_final_weight, dim);

     // classifier into logits
     matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
     return s->logits;
 }
 */
