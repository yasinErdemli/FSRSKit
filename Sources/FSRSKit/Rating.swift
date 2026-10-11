/// How a review went: whether the learner remembered, and how easily.
public enum Rating: Int, Sendable, Hashable, Codable, CaseIterable {
    /// Forgotten.
    case again = 1
    /// Remembered, with difficulty.
    case hard = 2
    /// Remembered.
    case good = 3
    /// Remembered without effort.
    case easy = 4

    /// Whether the learner remembered: every rating but `again`.
    public var isRecall: Bool { self != .again }
}
