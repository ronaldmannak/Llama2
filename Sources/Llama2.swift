// The Swift Programming Language
// https://docs.swift.org/swift-book
// 
// Swift Argument Parser
// https://swiftpackageindex.com/apple/swift-argument-parser/documentation

import ArgumentParser
import Foundation

// MARK: - Enums

enum Mode: String, CaseIterable, ExpressibleByArgument {
    case generate, chat
    
    static var allValueStrings: [String] {
        allCases.map { $0.rawValue }
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
        return ""
//        let inputTokens = prompt.map { tokenizer.tokenize($0) } ?? []
//        var outputTokens: [Int] = []
//        
//        for _ in 0..<steps {
//            let allTokens = inputTokens + outputTokens
//            let logits = transformer.forward(tokens: allTokens)
//            let nextToken = sampler.sample(logits: logits)
//            outputTokens.append(nextToken)
//        }
//        
//        return tokenizer.detokenize(outputTokens)
    }
    
    func chat(prompt: String?, systemPrompt: String?, steps: Int) throws -> String {
        return ""
//        let systemTokens = systemPrompt.map { tokenizer.tokenize($0) } ?? []
//        let promptTokens = prompt.map { tokenizer.tokenize($0) } ?? []
//        
//        // TODO: Implement proper chat formatting
//        let fullPrompt = [systemPrompt, prompt].compactMap { $0 }.joined(separator: "\n")
//        return try generate(prompt: fullPrompt, steps: steps)
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
    var checkpointPath: String = Bundle.module.path(forResource: "Resources/stories260K", ofType: "bin") ?? ""
    
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
    var tokenizerPath: String = Bundle.module.path(forResource: "Resources/tokenizer", ofType: "json") ?? "uh-oh"
    
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
        
        // Handle tokenizer path - try bundled resource first, then fallback to file system
        guard FileManager.default.fileExists(atPath: tokenizerPath) else {
            throw Llama2Error.fileNotFound("tokenizer.json not found in bundled resources or at specified path: \(tokenizerPath)")
        }
        
        // Read checkpoint file
//        let (config, weights) = try readCheckpoint(from: checkpointPath)
        
        // Create components with actual model data
        let tokenizer = try Tokenizer(tokenizerPath: tokenizerPath)
//        let transformer = Transformer(config: config, weights: weights)
        let sampler = Sampler(temperature: params.temperature, topp: params.topP, seed: params.seed)
        /*
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
         */
        
        print(tokenizer.config.model.type)
        print(tokenizer.config.model.unkToken)
        print(tokenizer.vocab.count)
//        print(tokenizer.vocab)
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

/*
 Port this C code to this Swift project. Create stub functions for the tokenizer, sampler, and transformer and other custom objects or functions. Make sure the code conforms to modern Swift 6.1 standards and follows Swift-idiomatic patterns including automatic memory management, value types where appropriate, and Swift's built-in data structures like Data for file handling. Avoid manual memory management and low-level system calls in favor of Swift's safe, automatic memory management.
 */
