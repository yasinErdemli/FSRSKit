import Foundation

/// Reverse-mode differentiation: the tape records every operation of a
/// forward pass with its local derivatives, and one sweep backward then gives
/// the derivative of the result by every parameter.
///
/// That costs a few operations per operation recorded, however many
/// parameters there are, which is why training runs the model on
/// ``Tracked`` numbers instead of carrying 34 derivatives forward.
///
/// A tape is used by one thread at a time: the optimizer gives each parallel
/// chunk of a batch its own. Its nodes live in a raw buffer, because recording
/// happens hundreds of times per review and an array in a class would pay for
/// a runtime exclusivity check each time.
final class Tape: @unchecked Sendable {
    /// One operation: the nodes it was computed from, and its derivative by each.
    struct Node {
        var first: Int32
        var second: Int32
        var firstDerivative: Double
        var secondDerivative: Double
    }

    /// The recorded nodes, which ``Tracked`` numbers point to.
    struct Storage {
        var nodes: UnsafeMutablePointer<Node>
        var count = 0
        var capacity: Int

        /// Records an operation computed from up to two nodes (-1 for none).
        @inline(__always)
        mutating func record(_ first: Int32, _ firstDerivative: Double, _ second: Int32, _ secondDerivative: Double) -> Int32 {
            if count == capacity { grow() }
            nodes[count] = Node(first: first, second: second, firstDerivative: firstDerivative, secondDerivative: secondDerivative)
            count += 1
            return Int32(count - 1)
        }

        private mutating func grow() {
            let larger = UnsafeMutablePointer<Node>.allocate(capacity: capacity * 2)
            larger.moveInitialize(from: nodes, count: count)
            nodes.deallocate()
            nodes = larger
            capacity *= 2
        }
    }

    let storage: UnsafeMutablePointer<Storage>
    private var adjoints: UnsafeMutablePointer<Double>
    private var adjointCapacity: Int

    init() {
        storage = .allocate(capacity: 1)
        storage.initialize(to: Storage(nodes: .allocate(capacity: 4096), capacity: 4096))
        adjointCapacity = 4096
        adjoints = .allocate(capacity: adjointCapacity)
    }

    deinit {
        storage.pointee.nodes.deallocate()
        storage.deinitialize(count: 1)
        storage.deallocate()
        adjoints.deallocate()
    }

    /// The parameters, as the tape's first variables. Clears the tape first.
    func parameters(_ values: [Double]) -> [Tracked] {
        storage.pointee.count = 0
        return values.map { Tracked(value: $0, index: storage.pointee.record(-1, 0, -1, 0), tape: storage) }
    }

    /// Adds the derivatives of `output` by the parameters to `gradient`.
    func backward(from output: Tracked, into gradient: inout [Double]) {
        guard output.index >= 0 else { return }
        let count = storage.pointee.count
        if count > adjointCapacity {
            adjoints.deallocate()
            adjointCapacity = max(count, adjointCapacity * 2)
            adjoints = .allocate(capacity: adjointCapacity)
        }
        adjoints.update(repeating: 0, count: count)
        adjoints[Int(output.index)] = 1
        let nodes = storage.pointee.nodes
        // The parameters are the first nodes, and have nothing to pass back to.
        for index in stride(from: Int(output.index), through: gradient.count, by: -1) {
            let adjoint = adjoints[index]
            guard adjoint != 0 else { continue }
            let node = nodes[index]
            if node.first >= 0 { adjoints[Int(node.first)] += adjoint * node.firstDerivative }
            if node.second >= 0 { adjoints[Int(node.second)] += adjoint * node.secondDerivative }
        }
        for index in gradient.indices {
            gradient[index] += adjoints[index]
        }
    }
}

/// A number recorded on a ``Tape``, or a constant if it has none.
struct Tracked: ModelNumber {
    var value: Double
    /// Its node on the tape, or -1 for a constant.
    var index: Int32
    /// The tape's nodes. The tape outlives every number recorded on it.
    var tape: UnsafeMutablePointer<Tape.Storage>?

    init(_ constant: Double) {
        value = constant
        index = -1
        tape = nil
    }

    init(value: Double, index: Int32, tape: UnsafeMutablePointer<Tape.Storage>?) {
        self.value = value
        self.index = index
        self.tape = tape
    }

    /// The result of an operation on one number, with its derivative.
    @inline(__always)
    private func unary(_ value: Double, _ derivative: Double) -> Tracked {
        guard let tape else { return Tracked(value) }
        return Tracked(value: value, index: tape.pointee.record(index, derivative, -1, 0), tape: tape)
    }

    /// The result of an operation on two numbers, with its derivative by each.
    @inline(__always)
    private static func binary(_ lhs: Tracked, _ rhs: Tracked, _ value: Double, _ lhsDerivative: Double, _ rhsDerivative: Double) -> Tracked {
        guard let tape = lhs.tape ?? rhs.tape else { return Tracked(value) }
        return Tracked(value: value, index: tape.pointee.record(lhs.index, lhsDerivative, rhs.index, rhsDerivative), tape: tape)
    }

    static func + (lhs: Tracked, rhs: Tracked) -> Tracked {
        binary(lhs, rhs, lhs.value + rhs.value, 1, 1)
    }

    static func - (lhs: Tracked, rhs: Tracked) -> Tracked {
        binary(lhs, rhs, lhs.value - rhs.value, 1, -1)
    }

    static func * (lhs: Tracked, rhs: Tracked) -> Tracked {
        binary(lhs, rhs, lhs.value * rhs.value, rhs.value, lhs.value)
    }

    static func / (lhs: Tracked, rhs: Tracked) -> Tracked {
        let quotient = lhs.value / rhs.value
        return binary(lhs, rhs, quotient, 1 / rhs.value, -quotient / rhs.value)
    }

    static func + (lhs: Tracked, rhs: Double) -> Tracked {
        lhs.unary(lhs.value + rhs, 1)
    }

    static func * (lhs: Tracked, rhs: Double) -> Tracked {
        lhs.unary(lhs.value * rhs, rhs)
    }

    static prefix func - (operand: Tracked) -> Tracked {
        operand.unary(-operand.value, -1)
    }

    func exp() -> Tracked {
        let result = Foundation.exp(value)
        return unary(result, result)
    }

    func log() -> Tracked {
        unary(Foundation.log(value), 1 / value)
    }

    func pow(_ exponent: Tracked) -> Tracked {
        let logarithm = Foundation.log(value)
        let result = Foundation.exp(exponent.value * logarithm)
        return Self.binary(self, exponent, result, exponent.value * result / value, result * logarithm)
    }

    func clamped(to range: ClosedRange<Double>) -> Tracked {
        if value < range.lowerBound { return Tracked(range.lowerBound) }
        if value > range.upperBound { return Tracked(range.upperBound) }
        return self
    }

    func clamped(atLeast lowerBound: Double) -> Tracked {
        value < lowerBound ? Tracked(lowerBound) : self
    }

    func clamped(atMost upperBound: Double) -> Tracked {
        value > upperBound ? Tracked(upperBound) : self
    }
}

extension Model<Tracked> {
    /// The model with every parameter as a variable on a tape. Clears the tape.
    init(_ parameters: Parameters, on tape: Tape) {
        w = tape.parameters(parameters.values)
    }
}
