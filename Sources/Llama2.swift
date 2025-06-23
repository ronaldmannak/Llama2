// The Swift Programming Language
// https://docs.swift.org/swift-book
// 
// Swift Argument Parser
// https://swiftpackageindex.com/apple/swift-argument-parser/documentation

import ArgumentParser
import Foundation

// MARK: - Error Types

enum Llama2Error: Error, LocalizedError {
    case invalidMode(String)
    case invalidParameter(String)
    case invalidTemperature(Float)
    case invalidTopP(Float)
    case invalidSteps(Int)
    case fileNotFound(String)
    
    var errorDescription: String? {
        switch self {
        case .invalidMode(let mode):
            return "Invalid mode: '\(mode)'. Valid modes are: generate, chat"
        case .invalidParameter(let param):
            return "Invalid parameter: \(param)"
        case .invalidTemperature(let temp):
            return "Temperature must be >= 0.0, got: \(temp)"
        case .invalidTopP(let topp):
            return "Top-p must be between 0.0 and 1.0, got: \(topp)"
        case .invalidSteps(let steps):
            return "Steps must be >= 0, got: \(steps)"
        case .fileNotFound(let path):
            return "File not found: \(path)"
        }
    }
}

// MARK: - Enums

enum Mode: String, CaseIterable, ExpressibleByArgument {
    case generate, chat
    
    static var allValueStrings: [String] {
        allCases.map { $0.rawValue }
    }
}

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

struct Tokenizer {
    private let vocab: [String: Int]
    private let merges: [String: String]
    
    init(vocab: [String: Int] = [:], merges: [String: String] = [:]) {
        self.vocab = vocab
        self.merges = merges
    }
    
    func tokenize(_ text: String) -> [Int] {
        // TODO: Implement actual tokenization
        return text.components(separatedBy: " ").compactMap { vocab[$0] }
    }
    
    func detokenize(_ tokens: [Int]) -> String {
        // TODO: Implement actual detokenization
        return tokens.compactMap { token in
            vocab.first { $0.value == token }?.key
        }.joined(separator: " ")
    }
}

struct Sampler {
    private let temperature: Float
    private let topP: Float
    private var rng: RandomNumberGenerator
    
    init(temperature: Float, topp: Float, seed: UInt64) {
        self.temperature = temperature
        self.topP = topp
        self.rng = RandomNumberGenerator(seed: seed)
    }
    
    mutating func sample(logits: [Float]) -> Int {
        // TODO: Implement actual sampling with temperature and top-p
        return Int.random(in: 0..<logits.count, using: &rng)
    }
}

// MARK: - Random Number Generator

struct RandomNumberGenerator: Swift.RandomNumberGenerator {
    private var state: UInt64
    
    init(seed: UInt64) {
        self.state = seed
    }
    
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}

// MARK: - Engine

final class Llama2Engine {
    private let transformer: Transformer
    private let tokenizer: Tokenizer
    private var sampler: Sampler
    
    init(transformer: Transformer, tokenizer: Tokenizer, sampler: Sampler) {
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.sampler = sampler
    }
    
    func generate(prompt: String?, steps: Int) throws -> String {
        let inputTokens = prompt.map { tokenizer.tokenize($0) } ?? []
        var outputTokens: [Int] = []
        
        for _ in 0..<steps {
            let allTokens = inputTokens + outputTokens
            let logits = transformer.forward(tokens: allTokens)
            let nextToken = sampler.sample(logits: logits)
            outputTokens.append(nextToken)
        }
        
        return tokenizer.detokenize(outputTokens)
    }
    
    func chat(prompt: String?, systemPrompt: String?, steps: Int) throws -> String {
        let systemTokens = systemPrompt.map { tokenizer.tokenize($0) } ?? []
        let promptTokens = prompt.map { tokenizer.tokenize($0) } ?? []
        
        // TODO: Implement proper chat formatting
        let fullPrompt = [systemPrompt, prompt].compactMap { $0 }.joined(separator: "\n")
        return try generate(prompt: fullPrompt, steps: steps)
    }
}

// MARK: - Parameter Validation

struct GenerationParameters {
    let temperature: Float
    let topP: Float
    let seed: UInt64
    let steps: Int
    let prompt: String?
    let systemPrompt: String?
    
    init(
        temperature: Float,
        topP: Float,
        seed: UInt64,
        steps: Int,
        prompt: String?,
        systemPrompt: String?
    ) throws {
        // Validate temperature
        guard temperature >= 0.0 else {
            throw Llama2Error.invalidTemperature(temperature)
        }
        
        // Validate topp
        guard topP >= 0.0 && topP <= 1.0 else {
            throw Llama2Error.invalidTopP(topP)
        }
        
        // Validate steps
        guard steps >= 0 else {
            throw Llama2Error.invalidSteps(steps)
        }
        
        self.temperature = temperature
        self.topP = topP
        self.seed = seed == 0 ? UInt64(Date().timeIntervalSince1970) : seed
        self.steps = steps
        self.prompt = prompt
        self.systemPrompt = systemPrompt
    }
}

// MARK: - CLI

@main
struct Llama2: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "A Swift implementation of Llama2 text generation",
        discussion: """
        Example: Llama2 model.bin -n 256 -i "Once upon a time"
        """
    )
    
    @Argument(help: "Path to the model checkpoint file (e.g., out/model.bin)")
    var checkpointPath: String
    
    @Option(name: .customShort("t"), help: "Temperature in [0,inf], default 1.0")
    var temperature: Float = 1.0
    
    @Option(name: .customShort("p"), help: "P value in top-p (nucleus) sampling in [0,1], default 0.9")
    var topP: Float = 0.9
    
    @Option(name: .customShort("s"), help: "Random seed, default time(NULL)")
    var seed: UInt64 = 0
    
    @Option(name: .customShort("n"), help: "Number of steps to run for, default 256. 0 = max_seq_len")
    var steps: Int = 256
    
    @Option(name: .customShort("i"), help: "Input prompt")
    var prompt: String?
    
    @Option(name: .customShort("z"), help: "Optional path to custom tokenizer")
    var tokenizerPath: String = "tokenizer.bin"
    
    @Option(name: .customShort("m"), help: "Mode: generate|chat, default: generate")
    var mode: Mode = .generate
    
    @Option(name: .customShort("y"), help: "Optional system prompt in chat mode")
    var systemPrompt: String?
    
    func run() throws {
        // Validate and prepare parameters
        let params = try GenerationParameters(
            temperature: temperature,
            topP: topP,
            seed: seed,
            steps: steps,
            prompt: prompt,
            systemPrompt: systemPrompt
        )
        
        // Validate files exist
        guard FileManager.default.fileExists(atPath: checkpointPath) else {
            throw Llama2Error.fileNotFound(checkpointPath)
        }
        guard FileManager.default.fileExists(atPath: tokenizerPath) else {
            throw Llama2Error.fileNotFound(tokenizerPath)
        }
        
        // Read checkpoint file
        let (config, weights) = try readCheckpoint(from: checkpointPath)
        
        // Create components with actual model data
        let transformer = Transformer(config: config, weights: weights)
        let tokenizer = Tokenizer()
        let sampler = Sampler(temperature: params.temperature, topp: params.topP, seed: params.seed)
        
        // Create engine
        let engine = Llama2Engine(
            transformer: transformer,
            tokenizer: tokenizer,
            sampler: sampler
        )
        
        // Determine actual steps
        let actualSteps = params.steps == 0 ? transformer.config.seqLen : params.steps
        
        // Generate output
        let output: String
        switch mode {
        case .generate:
            output = try engine.generate(prompt: params.prompt, steps: actualSteps)
        case .chat:
            output = try engine.chat(prompt: params.prompt, systemPrompt: params.systemPrompt, steps: actualSteps)
        }
        
        print(output)
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

/// Reads a checkpoint file and returns the config and weights
func readCheckpoint(from path: String) throws -> (config: Config, weights: TransformerWeights) {
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
    
    // Map the weights from the remaining data
    let weights = TransformerWeights.mapFromData(data, config: correctedConfig, sharedWeights: sharedWeights)
    
    return (correctedConfig, weights)
}
