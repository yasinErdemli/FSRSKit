import Foundation

/// A number the model computes with: `Double` to schedule, or ``Tracked`` to
/// train, which records each operation so it can be differentiated.
///
/// The model is written once, generic over this, so scheduling and training
/// can never disagree about the formulas.
protocol ModelNumber {
    /// A constant: its derivatives are zero.
    init(_ constant: Double)
    /// The number itself, without derivatives.
    var value: Double { get }

    static func + (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self
    static func / (lhs: Self, rhs: Self) -> Self
    static func + (lhs: Self, rhs: Double) -> Self
    static func * (lhs: Self, rhs: Double) -> Self
    static prefix func - (operand: Self) -> Self

    func exp() -> Self
    func log() -> Self
    /// The number to a power, for a positive number.
    func pow(_ exponent: Self) -> Self

    // Clamping as the reference does: outside the bounds, the result is a
    // constant, so its derivatives are zero there.
    func clamped(to range: ClosedRange<Double>) -> Self
    func clamped(atLeast lowerBound: Double) -> Self
    func clamped(atMost upperBound: Double) -> Self
}

extension ModelNumber {
    static func - (lhs: Self, rhs: Double) -> Self { lhs + -rhs }
    static func - (lhs: Double, rhs: Self) -> Self { -rhs + lhs }
    static func + (lhs: Double, rhs: Self) -> Self { rhs + lhs }
    static func * (lhs: Double, rhs: Self) -> Self { rhs * lhs }
    static func / (lhs: Self, rhs: Double) -> Self { lhs * (1 / rhs) }
    static func / (lhs: Double, rhs: Self) -> Self { Self(lhs) / rhs }
}

/// The smaller of two numbers by value, or the first on a tie, as the reference picks.
func smaller<Number: ModelNumber>(_ first: Number, _ second: Number) -> Number {
    second.value < first.value ? second : first
}

/// The larger of two numbers by value, or the first on a tie, as the reference picks.
func larger<Number: ModelNumber>(_ first: Number, _ second: Number) -> Number {
    second.value > first.value ? second : first
}

extension Double: ModelNumber {
    var value: Double { self }

    func exp() -> Double { Foundation.exp(self) }
    func log() -> Double { Foundation.log(self) }
    func pow(_ exponent: Double) -> Double { Foundation.pow(self, exponent) }

    func clamped(to range: ClosedRange<Double>) -> Double {
        Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }

    func clamped(atLeast lowerBound: Double) -> Double { Swift.max(self, lowerBound) }
    func clamped(atMost upperBound: Double) -> Double { Swift.min(self, upperBound) }
}
