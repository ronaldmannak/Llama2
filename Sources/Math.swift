//
//  File.swift
//  Llama2
//
//  Created by Ronald Mannak on 6/23/25.
//

import Foundation

// MARK: - Neural Network Math Functions

/// RMS (Root Mean Square) normalization - Core Implementation
/// 
/// This is the performance-optimized version that modifies an existing output buffer.
/// Use this version for:
/// - Performance-critical loops (e.g., transformer inference)
/// - Large arrays where memory allocation overhead matters
/// - Memory-constrained environments
/// - When you need to reuse the same buffer repeatedly
///
/// - Parameters:
///   - output: Output array to store normalized values (modified in place)
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

/// Softmax function for numerical stability - Core Implementation
/// 
/// This is the performance-optimized version that modifies the input array in place.
/// Use this version for:
/// - Performance-critical loops
/// - When you don't need to preserve the original input values
/// - Large arrays where memory allocation overhead matters
///
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

/// Matrix multiplication: W (d,n) @ x (n,) -> xout (d,) - Core Implementation
/// 
/// This is the performance-optimized version that writes to an existing output buffer.
/// Use this version for:
/// - Performance-critical loops (most computationally intensive function)
/// - Large matrices where memory allocation overhead matters
/// - When you need to reuse the same output buffer repeatedly
///
/// - Parameters:
///   - output: Output array (d elements) - will be overwritten
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

// MARK: - Convenience Wrappers

/// RMS normalization - Convenience Wrapper
/// 
/// This is the Swift-idiomatic version that returns a new array.
/// Use this version for:
/// - One-off operations and prototyping
/// - Higher-level APIs where code clarity matters
/// - Small arrays where allocation overhead is negligible
/// - When you need to preserve the original input
///
/// **Performance Note**: Creates a new array each call. For performance-critical loops,
/// use the inout version instead.
///
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

/// Softmax - Convenience Wrapper
/// 
/// This is the Swift-idiomatic version that returns a new array.
/// Use this version for:
/// - One-off operations and prototyping
/// - When you need to preserve the original input values
/// - Small arrays where allocation overhead is negligible
///
/// **Performance Note**: Creates a new array each call. For performance-critical loops,
/// use the inout version instead.
///
/// - Parameter values: Array of values to apply softmax to
/// - Returns: Softmax probabilities
func softmax(values: [Float]) -> [Float] {
    var result = values
    softmax(values: &result, size: result.count)
    return result
}

/// Matrix multiplication - Convenience Wrapper
/// 
/// This is the Swift-idiomatic version that returns a new array.
/// Use this version for:
/// - One-off operations and prototyping
/// - Higher-level APIs where code clarity matters
/// - Small matrices where allocation overhead is negligible
///
/// **Performance Note**: Creates a new array each call. For performance-critical loops,
/// use the inout version instead.
///
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

/* original C code

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

*/
