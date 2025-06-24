//
//  File.swift
//  Llama2
//
//  Created by Ronald Mannak on 6/22/25.
//

import Foundation
// MARK: - Error Types

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
