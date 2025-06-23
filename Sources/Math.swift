//
//  File.swift
//  Llama2
//
//  Created by Ronald Mannak on 6/23/25.
//

import Foundation

// MARK: - Neural Network Math Functions

/// RMS (Root Mean Square) normalization
/// - Parameters:
///   - output: Output array to store normalized values
///   - input: Input array to normalize
///   - weight: Weight array for scaling
///   - size: Size of the arrays
func rmsnorm(output: inout [Float], input: [Float], weight: [Float], size: Int) {
    // Calculate sum of squares
    let sumOfSquares = input.prefix(size).reduce(0.0) { $0 + $1 * $1 }
    let ss = 1.0 / sqrt(sumOfSquares / Float(size) + 1e-5)
    
    // Normalize and scale
    for j in 0..<size {
        output[j] = weight[j] * (ss * input[j])
    }
}

/// Softmax function for numerical stability
/// - Parameters:
///   - values: Array of values to apply softmax to (modified in place)
///   - size: Size of the array
func softmax(values: inout [Float], size: Int) {
    // Find max value for numerical stability
    let maxVal = values.prefix(size).max() ?? 0.0
    
    // Apply exp and calculate sum
    var sum: Float = 0.0
    for i in 0..<size {
        values[i] = exp(values[i] - maxVal)
        sum += values[i]
    }
    
    // Normalize
    for i in 0..<size {
        values[i] /= sum
    }
}

/// Matrix multiplication: W (d,n) @ x (n,) -> xout (d,)
/// - Parameters:
///   - output: Output array (d elements)
///   - input: Input vector (n elements)
///   - weights: Weight matrix (d x n elements, stored row-major)
///   - n: Number of input dimensions
///   - d: Number of output dimensions
func matmul(output: inout [Float], input: [Float], weights: [Float], n: Int, d: Int) {
    // W (d,n) @ x (n,) -> xout (d,)
    // This is the most computationally intensive function
    for i in 0..<d {
        var val: Float = 0.0
        for j in 0..<n {
            val += weights[i * n + j] * input[j]
        }
        output[i] = val
    }
}

// MARK: - Convenience Overloads

/// RMS normalization with automatic output array creation
/// - Parameters:
///   - input: Input array to normalize
///   - weight: Weight array for scaling
/// - Returns: Normalized output array
func rmsnorm(input: [Float], weight: [Float]) -> [Float] {
    let size = min(input.count, weight.count)
    var output = Array(repeating: 0.0 as Float, count: size)
    rmsnorm(output: &output, input: input, weight: weight, size: size)
    return output
}

/// Softmax with automatic array creation
/// - Parameter values: Array of values to apply softmax to
/// - Returns: Softmax probabilities
func softmax(values: [Float]) -> [Float] {
    var result = values
    softmax(values: &result, size: result.count)
    return result
}

/// Matrix multiplication with automatic output array creation
/// - Parameters:
///   - input: Input vector (n elements)
///   - weights: Weight matrix (d x n elements, stored row-major)
///   - n: Number of input dimensions
///   - d: Number of output dimensions
/// - Returns: Output vector (d elements)
func matmul(input: [Float], weights: [Float], n: Int, d: Int) -> [Float] {
    var output = Array(repeating: 0.0 as Float, count: d)
    matmul(output: &output, input: input, weights: weights, n: n, d: d)
    return output
}
