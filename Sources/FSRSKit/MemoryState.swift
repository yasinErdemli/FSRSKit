/// How well something is remembered after its last review.
///
/// FSRS-7 keeps two traces of a memory: a fast one that fades within hours or
/// days, and a slow one that holds for weeks or years. Both are measured as
/// stabilities, in days. Difficulty, from 1 to 10, is how hard the item is for
/// this learner, and slows how fast the slow trace grows.
public struct MemoryState: Sendable, Hashable, Codable {
    /// The slow trace's stability, in days: longer means remembered for longer.
    public var stability: Double
    /// The fast trace's stability, in days.
    public var fastStability: Double
    /// How hard the item is for the learner, from 1 (easy) to 10 (hard).
    public var difficulty: Double

    /// A memory state with the given stabilities and difficulty.
    public init(stability: Double, fastStability: Double, difficulty: Double) {
        self.stability = stability
        self.fastStability = fastStability
        self.difficulty = difficulty
    }
}

/// The ranges the model keeps its states in, as the reference does.
enum Bounds {
    static let stability = 0.0001...36_500.0
    static let difficulty = 1.0...10.0
    /// The shortest interval: one second, in days.
    static let shortestInterval = 1.0 / 86_400
    /// The most an initial stability can be when parameters are clipped.
    static let largestInitialStability = 100.0
}
