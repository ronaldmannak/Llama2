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
    
    init(
        tokenEmbeddingTable: [Float] = [],
        rmsAttWeight: [Float] = [],
        rmsFfnWeight: [Float] = [],
        wq: [Float] = [],
        wk: [Float] = [],
        wv: [Float] = [],
        wo: [Float] = [],
        w1: [Float] = [],
        w2: [Float] = [],
        w3: [Float] = [],
        rmsFinalWeight: [Float] = [],
        wcls: [Float]? = nil
    ) {
        self.tokenEmbeddingTable = tokenEmbeddingTable
        self.rmsAttWeight = rmsAttWeight
        self.rmsFfnWeight = rmsFfnWeight
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.rmsFinalWeight = rmsFinalWeight
        self.wcls = wcls
    }
}

// MARK: - Weight Mapping and Checkpoint Reading

extension TransformerWeights {
    /// Maps binary data to weight arrays, equivalent to C's memory_map_weights
    static func mapFromData(_ data: Data, config: Config, sharedWeights: Bool) -> TransformerWeights {
        let headSize = config.dim / config.numHeads
        let kvDim = (config.dim * config.numKvHeads) / config.numHeads
        
        // Convert Data to [Float] for easier manipulation
        let floatCount = data.count / MemoryLayout<Float>.size
        let floats = data.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self).prefix(floatCount))
        }
        
        var ptr = 0
        
        // Skip the config header (already read)
        ptr += MemoryLayout<Config>.size / MemoryLayout<Float>.size
        
        // Map weights in the same order as C code
        let tokenEmbeddingTable = Array(floats[ptr..<(ptr + config.vocabSize * config.dim)])
        ptr += config.vocabSize * config.dim
        
        let rmsAttWeight = Array(floats[ptr..<(ptr + config.numLayers * config.dim)])
        ptr += config.numLayers * config.dim
        
        let wq = Array(floats[ptr..<(ptr + config.numLayers * config.dim * (config.numHeads * headSize))])
        ptr += config.numLayers * config.dim * (config.numHeads * headSize)
        
        let wk = Array(floats[ptr..<(ptr + config.numLayers * config.dim * (config.numKvHeads * headSize))])
        ptr += config.numLayers * config.dim * (config.numKvHeads * headSize)
        
        let wv = Array(floats[ptr..<(ptr + config.numLayers * config.dim * (config.numKvHeads * headSize))])
        ptr += config.numLayers * config.dim * (config.numKvHeads * headSize)
        
        let wo = Array(floats[ptr..<(ptr + config.numLayers * (config.numHeads * headSize) * config.dim)])
        ptr += config.numLayers * (config.numHeads * headSize) * config.dim
        
        let rmsFfnWeight = Array(floats[ptr..<(ptr + config.numLayers * config.dim)])
        ptr += config.numLayers * config.dim
        
        let w1 = Array(floats[ptr..<(ptr + config.numLayers * config.dim * config.hiddenDim)])
        ptr += config.numLayers * config.dim * config.hiddenDim
        
        let w2 = Array(floats[ptr..<(ptr + config.numLayers * config.hiddenDim * config.dim)])
        ptr += config.numLayers * config.hiddenDim * config.dim
        
        let w3 = Array(floats[ptr..<(ptr + config.numLayers * config.dim * config.hiddenDim)])
        ptr += config.numLayers * config.dim * config.hiddenDim
        
        let rmsFinalWeight = Array(floats[ptr..<(ptr + config.dim)])
        ptr += config.dim
        
        // Skip RoPE frequency tables (freq_cis_real and freq_cis_imag)
        ptr += config.seqLen * headSize / 2 // freq_cis_real
        ptr += config.seqLen * headSize / 2 // freq_cis_imag
        
        // Handle classifier weights
        let wcls: [Float]?
        if sharedWeights {
            wcls = nil // Use token embedding table for shared weights
        } else {
            wcls = Array(floats[ptr..<(ptr + config.vocabSize * config.dim)])
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
    
    init(config: Config, weights: TransformerWeights = TransformerWeights()) {
        self.config = config
        self.weights = weights
        self.state = RunState(config: config)
    }
    
    func forward(tokens: [Int]) -> [Float] {
        // TODO: Implement actual transformer forward pass
        return Array(repeating: 0.0, count: config.vocabSize)
    }
}


/*
 Port this C code to this Swift project. Create stub functions for the tokenizer, sampler, and transformer and other custom objects or functions. Make sure the code conforms to modern Swift 6.1 standards and follows Swift-idiomatic patterns including automatic memory management, value types where appropriate, and Swift's built-in data structures like Data for file handling. Avoid manual memory management and low-level system calls in favor of Swift's safe, automatic memory management.
 
 // ----------------------------------------------------------------------------
 // neural net blocks; the dynamics of the Transformer

 void rmsnorm(float* o, float* x, float* weight, int size) {
     // calculate sum of squares
     float ss = 0.0f;
     for (int j = 0; j < size; j++) {
         ss += x[j] * x[j];
     }
     ss /= size;
     ss += 1e-5f;
     ss = 1.0f / sqrtf(ss);
     // normalize and scale
     for (int j = 0; j < size; j++) {
         o[j] = weight[j] * (ss * x[j]);
     }
 }

 void softmax(float* x, int size) {
     // find max value (for numerical stability)
     float max_val = x[0];
     for (int i = 1; i < size; i++) {
         if (x[i] > max_val) {
             max_val = x[i];
         }
     }
     // exp and sum
     float sum = 0.0f;
     for (int i = 0; i < size; i++) {
         x[i] = expf(x[i] - max_val);
         sum += x[i];
     }
     // normalize
     for (int i = 0; i < size; i++) {
         x[i] /= sum;
     }
 }

 void matmul(float* xout, float* x, float* w, int n, int d) {
     // W (d,n) @ x (n,) -> xout (d,)
     // by far the most amount of time is spent inside this little function
     int i;
     #pragma omp parallel for private(i)
     for (i = 0; i < d; i++) {
         float val = 0.0f;
         for (int j = 0; j < n; j++) {
             val += w[i * n + j] * x[j];
         }
         xout[i] = val;
     }
 }

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
