//
//  File.swift
//  Llama2
//
//  Created by Ronald Mannak on 6/23/25.
//

import Foundation
import ArgumentParser

enum Llama2Error: Error, LocalizedError {
    case invalidMode(String)
    case invalidParameter(String)
    case invalidTemperature(Float)
    case invalidTopP(Float)
    case invalidSteps(Int)
    case fileNotFound(String)
    case unsupportedTokenizer(String)
    
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
        case .unsupportedTokenizer(let string):
            return "Unsupported tokenizer: Expected BPE, got: \(string)"
        }
    }
}

// MARK: - Tokenizer JSON Model (copied from Tokenizer.swift, minimal set)

struct AddedToken: Codable {
    let id: Int
    let content: String
    let singleWord: Bool
    let lstrip: Bool
    let rstrip: Bool
    let normalized: Bool
    let special: Bool
}

struct BPEModel: Codable {
    let type: String
    let dropout: String?
    let unkToken: String
    let continuingSubwordPrefix: String?
    let endOfWordSuffix: String?
    let fuseUnk: Bool
    let byteFallback: Bool
    let vocab: [String: Int]
    let merges: [String]
}

struct TokenizerConfig: Codable {
    let version: String
    let truncation: String?
    let padding: String?
    let addedTokens: [AddedToken]
    let normalizer: String? // Simplified for now
    let preTokenizer: String?
    let postProcessor: String?
    let decoder: String?
    let model: BPEModel
}

// MARK: - Binary Tokenizer Structure

struct BinaryTokenizer {
    let vocabSize: Int
    let maxTokenLength: Int32
    let vocab: [String]
    let vocabScores: [Float]
    let bytePieces: [String]
    
    init(binPath: String) throws {
        // Get the path to the binary tokenizer file
        
        // Read the binary file
        let data = try Data(contentsOf: URL(fileURLWithPath: binPath))
        
        print("Binary file size: \(data.count) bytes")
        
        // Parse according to C specification
        var offset = 0
        
        // Read max_token_length (4 bytes)
        guard offset + 4 <= data.count else {
            throw Llama2Error.invalidParameter("File too small to contain max_token_length")
        }
        
        // Use explicit byte copying to ensure proper alignment
        var maxTokenLength: Int32 = 0
        data.copyBytes(to: withUnsafeMutableBytes(of: &maxTokenLength) { $0 }, from: offset..<(offset + 4))
        offset += 4
        
        print("Max token length: \(maxTokenLength)")
        
        // For now, we'll assume vocab_size is 512 (based on filename tok512.bin)
        // In a real implementation, this should be read from the file or passed as parameter
        let vocabSize = 512
        
        // Read vocab_scores and vocab strings
        var vocab: [String] = []
        var vocabScores: [Float] = []
        
        for i in 0..<vocabSize {
            // Read vocab score (4 bytes)
            guard offset + 4 <= data.count else {
                throw Llama2Error.invalidParameter("File too small to contain vocab score \(i)")
            }
            
            // Use explicit byte copying to ensure proper alignment
            var score: Float = 0.0
            data.copyBytes(to: withUnsafeMutableBytes(of: &score) { $0 }, from: offset..<(offset + 4))
            vocabScores.append(score)
            offset += 4
            
            // Read string length (4 bytes)
            guard offset + 4 <= data.count else {
                throw Llama2Error.invalidParameter("File too small to contain string length \(i)")
            }
            
            var len: Int32 = 0
            data.copyBytes(to: withUnsafeMutableBytes(of: &len) { $0 }, from: offset..<(offset + 4))
            offset += 4
            
            // Read the string
            guard offset + Int(len) <= data.count else {
                throw Llama2Error.invalidParameter("File too small to contain string \(i)")
            }
            
            let stringData = data[offset..<(offset + Int(len))]
            if let string = String(data: stringData, encoding: .utf8) {
                vocab.append(string)
            } else {
                throw Llama2Error.invalidParameter("Failed to decode string \(i) as UTF-8")
            }
            offset += Int(len)
        }
        
        // Initialize byte pieces (same as C code)
        var bytePieces: [String] = []
        for i in 0..<256 {
            if let scalar = UnicodeScalar(i) {
                bytePieces.append(String(scalar))
            } else {
                bytePieces.append("")
            }
        }
        
        // Store the parsed data
        self.vocabSize = vocabSize
        self.maxTokenLength = maxTokenLength
        self.vocab = vocab
        self.vocabScores = vocabScores
        self.bytePieces = bytePieces
    }
    
    // MARK: - Debug Information
    
    func printDebugInfo() {
        print("Binary Tokenizer Debug Info:")
        print("  Vocab Size: \(vocabSize)")
        print("  Max Token Length: \(maxTokenLength)")
        print("  File Size: \(vocab.count) tokens loaded")
        print("  First 10 tokens:")
        for i in 0..<min(10, vocab.count) {
            print("    [\(i)] Score: \(vocabScores[i]), Token: \"\(vocab[i])\"")
        }
        print("  Last 10 tokens:")
        let start = max(0, vocab.count - 10)
        for i in start..<vocab.count {
            print("    [\(i)] Score: \(vocabScores[i]), Token: \"\(vocab[i])\"")
        }
    }
    
    // MARK: - Export to HuggingFace Tokenizer JSON
    
    func exportToTokenizerJSON(outputPath: String, merges: [String] = []) throws {
        // Build vocab dictionary: token string -> index
        var vocabDict: [String: Int] = [:]
        for (i, token) in vocab.enumerated() {
            vocabDict[token] = i
        }
        
        // Build addedTokens (empty for now)
        let addedTokens: [AddedToken] = []
        
        // Build BPEModel
        let bpeModel = BPEModel(
            type: "BPE",
            dropout: nil,
            unkToken: "<unk>",
            continuingSubwordPrefix: nil,
            endOfWordSuffix: nil,
            fuseUnk: false,
            byteFallback: false,
            vocab: vocabDict,
            merges: merges
        )
        
        // Build TokenizerConfig
        let config = TokenizerConfig(
            version: "1.0",
            truncation: nil,
            padding: nil,
            addedTokens: addedTokens,
            normalizer: nil,
            preTokenizer: nil,
            postProcessor: nil,
            decoder: nil,
            model: bpeModel
        )
        
        // Encode as JSON
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let jsonData = try encoder.encode(config)
        
        // Write to file
        try jsonData.write(to: URL(fileURLWithPath: outputPath))
        print("Exported tokenizer JSON to \(outputPath)")
    }
}

// MARK: - Main Function for Testing

func testBinaryTokenizer(binPath: String) {
    do {
        let tokenizer = try BinaryTokenizer(binPath: binPath)
        tokenizer.printDebugInfo()
    } catch {
        print("Error reading binary tokenizer: \(error)")
    }
}

// MARK: - Main Entry Point

@main
struct TokenizerConverter: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Convert binary tokenizer to JSON format",
        discussion: "Reads a binary tokenizer file and converts it to JSON format compatible with our tokenizer"
    )
    
    @Option(
        name: .shortAndLong,
        help: "Path to the binary tokenizer file (defaults to bundled tok512.bin)"
    )
    var binPath: String?
    
    @Option(
        name: .shortAndLong,
        help: "Path to output tokenizer.json (defaults to ./tokenizer.json)"
    )
    var outputPath: String?
    
    func run() throws {
        let filePath: String
        
        if let providedPath = binPath {
            // Use the provided path
            filePath = providedPath
        } else {
            // Use the default bundled file
            guard let defaultPath = Bundle.module.path(forResource: "Resources/tok512", ofType: "bin") else {
                throw Llama2Error.fileNotFound("Binary tokenizer file not found: tok512.bin")
            }
            filePath = defaultPath
        }
        
        print("Testing binary tokenizer reading...")
        print("Using file: \(filePath)")
        testBinaryTokenizer(binPath: filePath)
        
        // Export to tokenizer.json
        let outPath = outputPath ?? "tokenizer.json"
        let tokenizer = try BinaryTokenizer(binPath: filePath)
        try tokenizer.exportToTokenizerJSON(outputPath: outPath)
        print("Tokenizer JSON export complete.")
    }
}
