/// The 34 numbers that tune FSRS-7 to a learner.
///
/// The defaults suit an average learner. ``Optimizer`` fits them to one
/// learner's own reviews, which is where most of FSRS's accuracy comes from.
///
/// What each group does, by index:
/// - 0–3: the stability after a first review rated again, hard, good or easy
/// - 4–6: the difficulty after a first review, and how much each rating changes it
/// - 7–14: how the slow trace's stability changes at a review: how fast it
///   grows, how that slows as it gets larger and grows with forgetting, the
///   stability left after a lapse, and the penalty for hard and bonus for easy
/// - 15–22: the same eight for the fast trace
/// - 23–33: the shape of the forgetting curve: each trace's decay, how much
///   recall is left after one stability, and how much each trace counts
///
/// Values are always clipped into the ranges the model is trained in, so any
/// `Parameters` is safe to schedule with.
public struct Parameters: Sendable, Hashable {
    /// How many numbers FSRS-7 has.
    public static let count = 34

    /// The parameters for an average learner, from the reference implementation.
    public static let `default` = Parameters(clipping: [
        0.1104, 2.2395, 3.9221, 11.7841, 6.1686, 0.6457, 3.6807, 1.9795, 0.0, 1.3826, 0.7024, 0.5999,
        0.8146, 0.6398, 1.0, 1.3207, 0.6707, 3.8668, 0.4416, 0.0934, 1.8631, 0.6162, 1.0869, 0.1567,
        0.0801, 0.2421, 0.9464, 0.1433, 0.7145, 0.0, 0.5667, 0.3734, 0.5333, 0.3048,
    ])

    /// The values, in the order described above.
    public let values: [Double]

    /// Parameters from 34 values, clipped into the model's ranges.
    ///
    /// - Throws: ``Error`` if there aren't 34 values, or one isn't a finite number.
    public init(_ values: [Double]) throws(Error) {
        guard values.count == Self.count else { throw .wrongCount(values.count) }
        if let index = values.firstIndex(where: { !$0.isFinite }) { throw .notFinite(index: index) }
        self.init(clipping: values)
    }

    init(clipping values: [Double]) {
        var values = values
        Self.clip(&values)
        self.values = values
    }

    /// The value at an index from 0 to 33.
    public subscript(index: Int) -> Double { values[index] }

    /// Why a list of numbers can't be parameters.
    public enum Error: Swift.Error, Hashable {
        /// There weren't 34 values.
        case wrongCount(Int)
        /// The value at this index was infinite or not a number.
        case notFinite(index: Int)
    }

    /// Clips values into the ranges the reference trains in
    /// (`parameter_clipper_v7.rs`). The four initial stabilities also stay in
    /// order, so a better first rating never means a shorter memory.
    static func clip(_ w: inout [Double]) {
        let smallest = Bounds.stability.lowerBound
        let largest = Bounds.largestInitialStability
        w[0] = clamp(w[0], smallest, largest / 2)
        w[1] = clamp(w[1], w[0], largest)
        w[2] = clamp(w[2], w[1], largest)
        w[3] = clamp(w[3], w[2], largest)
        for (index, range) in ranges {
            w[index] = clamp(w[index], range.lowerBound, range.upperBound)
        }
        // The slow trace keeps at least as much recall after one stability as the fast one.
        w[26] = clamp(w[26], max(w[25], 0.5), 0.99)
    }

    /// The range of every parameter after the initial stabilities, except 26,
    /// whose lower bound depends on 25.
    private static let ranges: [(Int, ClosedRange<Double>)] = [
        (4, 1...10), (5, 0.001...4), (6, 0.1...4),
        (7, 0...4), (8, 0...1.2), (9, 0.3...3), (10, 0.01...1.5), (11, 0.1...1),
        (12, 0...3.5), (13, 0...1), (14, 1...7),
        (15, 0...4), (16, 0...2), (17, 0.5...6), (18, 0.001...1.5), (19, 0.001...1),
        (20, 0...5), (21, 0...1), (22, 1...7),
        (23, 0.01...0.25), (24, 0.01...0.95), (25, 0.2...0.85),
        (27, 0.01...1), (28, 0.1...1), (29, 0...0.9), (30, 0.1...1.1),
        (31, 0...1), (32, 0...0.6), (33, 0...0.6),
    ]

    /// Clamps like the reference's `clamp_safe`: a value that isn't finite
    /// becomes the lower bound, and bounds given in the wrong order are swapped.
    private static func clamp(_ value: Double, _ low: Double, _ high: Double) -> Double {
        let (low, high) = low <= high ? (low, high) : (high, low)
        guard value.isFinite else { return low }
        return min(max(value, low), high)
    }
}

extension Parameters: Codable {
    /// Decodes parameters from a plain list of 34 numbers.
    public init(from decoder: any Decoder) throws {
        let values = try decoder.singleValueContainer().decode([Double].self)
        try self.init(values)
    }

    /// Encodes parameters as a plain list of 34 numbers.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(values)
    }
}
