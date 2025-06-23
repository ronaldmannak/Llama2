//
//  Tokenizer.swift
//  Llama2
//
//  Created by Ronald Mannak on 6/22/25.
//

import Foundation

/// Represents a token with its string representation and ID
struct TokenIndex {
    let str: String
    let id: Int
}

/// Added token information from the tokenizer JSON
struct AddedToken: Codable {
    let id: Int
    let content: String
    let singleWord: Bool
    let lstrip: Bool
    let rstrip: Bool
    let normalized: Bool
    let special: Bool
}

/// Normalizer configuration
struct Normalizer: Codable {
    let type: String
    let normalizers: [NormalizerConfig]?
    let prepend: String?
    let pattern: PatternConfig?
    let content: String?
}

struct NormalizerConfig: Codable {
    let type: String
    let prepend: String?
    let pattern: PatternConfig?
    let content: String?
}

struct PatternConfig: Codable {
    let string: String?
}

/// Post processor configuration
struct PostProcessor: Codable {
    let type: String
    let single: [ProcessingStep]?
    let pair: [ProcessingStep]?
    let specialTokens: [String: SpecialTokenInfo]?
}

struct ProcessingStep: Codable {
    let specialToken: SpecialTokenStep?
    let sequence: SequenceStep?
    
    enum CodingKeys: String, CodingKey {
        case specialToken = "SpecialToken"
        case sequence = "Sequence"
    }
}

struct SpecialTokenStep: Codable {
    let id: String
    let typeId: Int
}

struct SequenceStep: Codable {
    let id: String
    let typeId: Int
}

struct SpecialTokenInfo: Codable {
    let id: String
    let ids: [Int]
    let tokens: [String]
}

/// Decoder configuration
struct Decoder: Codable {
    let type: String
    let decoders: [DecoderConfig]?
    let pattern: PatternConfig?
    let content: String?
    let start: Int?
    let stop: Int?
}

struct DecoderConfig: Codable {
    let type: String
    let pattern: PatternConfig?
    let content: String?
    let start: Int?
    let stop: Int?
}

/// BPE model configuration
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

/// Complete tokenizer configuration
struct TokenizerConfig: Codable {
    let version: String
    let truncation: String?
    let padding: String?
    let addedTokens: [AddedToken]
    let normalizer: Normalizer?
    let preTokenizer: String?
    let postProcessor: PostProcessor?
    let decoder: Decoder?
    let model: BPEModel
}

/// Byte Pair Encoding (BPE) Tokenizer for converting between strings and tokens.
/// Unlike Llama2.c, which leverages a binary file format for its tokenizer,
/// this implementation uses Hugging Face's `tokenizer.json` for educational purposes.
struct Tokenizer {
    private let config: TokenizerConfig
    private let vocab: [String: Int]
    private let reverseVocab: [Int: String]
    private let merges: [String]
    private var sortedVocab: [TokenIndex]?
    private let bytePieces: [String]
    
    /// Initialize tokenizer from a JSON file path
    init(tokenizerPath: String) throws {
        // Read and decode the JSON file with automatic snake_case to camelCase conversion
        let data = try Data(contentsOf: URL(fileURLWithPath: tokenizerPath))
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.config = try decoder.decode(TokenizerConfig.self, from: data)
        
        // Initialize vocabulary
        self.vocab = config.model.vocab
        self.merges = config.model.merges
        
        // Create reverse vocabulary for efficient lookup
        var reverseVocab: [Int: String] = [:]
        for (token, id) in vocab {
            reverseVocab[id] = token
        }
        self.reverseVocab = reverseVocab
        
        // Initialize byte pieces for single-byte strings
        var bytePieces: [String] = []
        for i in 0..<256 {
            if let scalar = UnicodeScalar(i) {
                bytePieces.append(String(scalar))
            } else {
                bytePieces.append("")
            }
        }
        self.bytePieces = bytePieces
    }
    
    /// Decode a token to its string representation
    func decode(prevToken: Int, token: Int) -> String {
        guard let piece = reverseVocab[token] else {
            return config.model.unkToken
        }
        
        // Following BOS (1) token, sentencepiece decoder strips any leading whitespace
        if prevToken == 1 && piece.hasPrefix(" ") {
            return String(piece.dropFirst())
        }
        
        // Handle raw byte tokens that look like '<0x01>'
        if piece.hasPrefix("<0x") && piece.hasSuffix(">") {
            let hexString = String(piece.dropFirst(3).dropLast(1))
            if let byteValue = UInt8(hexString, radix: 16) {
                return bytePieces[Int(byteValue)]
            }
        }
        
        return piece
    }
    
    /// Safely print a piece, handling raw byte tokens
    func safePrint(_ piece: String) {
        guard !piece.isEmpty else { return }
        
        if piece.count == 1 {
            let byteValue = piece.utf8.first ?? 0
            // Only print printable characters or whitespace
            if !(byteValue >= 32 && byteValue <= 126) && byteValue != 9 && byteValue != 10 && byteValue != 13 {
                return // Bad byte, don't print it
            }
        }
        
        print(piece, terminator: "")
    }
    
    /// Efficiently find the perfect match for str in vocab, return its index or -1 if not found
    private func strLookup(_ str: String, sortedVocab: [TokenIndex]) -> Int {
        let token = TokenIndex(str: str, id: -1)
        
        // Binary search for the token
        var left = 0
        var right = sortedVocab.count - 1
        
        while left <= right {
            let mid = (left + right) / 2
            let comparison = str.compare(sortedVocab[mid].str)
            
            if comparison == .orderedSame {
                return sortedVocab[mid].id
            } else if comparison == .orderedAscending {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        
        return -1
    }
    
    /// Encode a string into tokens using BPE
    mutating func encode(text: String, bos: Bool = false, eos: Bool = false) -> [Int] {
        guard !text.isEmpty else {
            return []
        }
        
        // Lazy initialization of sorted vocabulary
        if sortedVocab == nil {
            sortedVocab = vocab.map { (token, id) in
                TokenIndex(str: token, id: id)
            }.sorted { $0.str < $1.str }
        }
        
        guard let sortedVocab = sortedVocab else { return [] }
        
        var tokens: [Int] = []
        
        // Add optional BOS token
        if bos {
            if let bosId = vocab["<s>"] {
                tokens.append(bosId)
            }
        }
        
        // Apply normalizer if present (simplified - just prepend space if needed)
        var normalizedText = text
        if config.normalizer?.type == "Sequence" {
            // Simple implementation: prepend space if text doesn't start with space
            if !normalizedText.hasPrefix(" ") && !normalizedText.hasPrefix("▁") {
                normalizedText = " " + normalizedText
            }
        }
        
        // Initial tokenization: split into characters/subwords
        var wordTokens: [String] = []
        var currentToken = ""
        
        for char in normalizedText {
            let charStr = String(char)
            if char == " " || char == "▁" {
                if !currentToken.isEmpty {
                    wordTokens.append(currentToken)
                    currentToken = ""
                }
            } else {
                currentToken += charStr
            }
        }
        if !currentToken.isEmpty {
            wordTokens.append(currentToken)
        }
        
        // Tokenize each word using BPE
        for word in wordTokens {
            var wordTokens = tokenizeWord(word, sortedVocab: sortedVocab)
            tokens.append(contentsOf: wordTokens)
        }
        
        // Add optional EOS token
        if eos {
            if let eosId = vocab["</s>"] {
                tokens.append(eosId)
            }
        }
        
        return tokens
    }
    
    /// Tokenize a single word using BPE merges
    private func tokenizeWord(_ word: String, sortedVocab: [TokenIndex]) -> [Int] {
        var tokens: [String] = []
        
        // Start with individual characters
        for char in word {
            tokens.append(String(char))
        }
        
        // Apply BPE merges
        var changed = true
        while changed {
            changed = false
            
            for merge in merges {
                let parts = merge.components(separatedBy: " ")
                guard parts.count == 2 else { continue }
                
                let first = parts[0]
                let second = parts[1]
                
                for i in 0..<(tokens.count - 1) {
                    if tokens[i] == first && tokens[i + 1] == second {
                        let merged = first + second
                        if vocab[merged] != nil {
                            tokens[i] = merged
                            tokens.remove(at: i + 1)
                            changed = true
                            break
                        }
                    }
                }
                if changed { break }
            }
        }
        
        // Convert tokens to IDs
        return tokens.compactMap { vocab[$0] }
    }
    
    /// Tokenize text into tokens
    mutating func tokenize(_ text: String) -> [Int] {
        return encode(text: text, bos: true, eos: true)
    }
    
    /// Detokenize tokens back to text
    func detokenize(_ tokens: [Int]) -> String {
        var result = ""
        var prevToken = 0
        
        for token in tokens {
            let piece = decode(prevToken: prevToken, token: token)
            result += piece
            prevToken = token
        }
        
        // Apply decoder if present (simplified)
        if config.decoder?.type == "Sequence" {
            // Simple implementation: replace ▁ with space and strip leading space
            result = result.replacingOccurrences(of: "▁", with: " ")
            if result.hasPrefix(" ") {
                result = String(result.dropFirst())
            }
        }
        
        return result
    }
    
    /// Get vocabulary size
    var vocabularySize: Int {
        return vocab.count
    }
    
    /// Get unknown token ID
    var unknownTokenId: Int {
        return vocab[config.model.unkToken] ?? 0
    }
    
    /// Get BOS token ID
    var bosTokenId: Int {
        return vocab["<s>"] ?? 1
    }
    
    /// Get EOS token ID
    var eosTokenId: Int {
        return vocab["</s>"] ?? 2
    }
}

/// Errors that can occur during tokenizer operations
enum TokenizerError: Error {
    case invalidFileFormat(String)
    case fileNotFound
    case decodingError
    case jsonDecodingError(Error)
}
