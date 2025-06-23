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
    
    func forward(tokens: [Int]) -> [Float] {
        // TODO: Implement actual transformer forward pass
        return Array(repeating: 0.0, count: config.vocabSize)
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
}


/*
 Port this C code to this Swift project. Create stub functions for the tokenizer, sampler, and transformer and other custom objects or functions. Make sure the code conforms to modern Swift 6.1 standards and follows Swift-idiomatic patterns including automatic memory management, value types where appropriate, and Swift's built-in data structures like Data for file handling. Avoid manual memory management and low-level system calls in favor of Swift's safe, automatic memory management.
 


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
