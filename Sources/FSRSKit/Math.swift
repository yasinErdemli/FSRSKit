#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif

// The C library's math, under names of their own: `ModelNumber` has `exp()`,
// `log()` and `pow(_:)` methods, which would hide these inside its types.

/// e to the power of `x`.
@inline(__always) func exponential(_ x: Double) -> Double { exp(x) }
/// The natural logarithm of `x`.
@inline(__always) func logarithm(_ x: Double) -> Double { log(x) }
/// `x` to the power of `y`.
@inline(__always) func power(_ x: Double, _ y: Double) -> Double { pow(x, y) }
/// The cosine of `x`, in radians.
@inline(__always) func cosine(_ x: Double) -> Double { cos(x) }
