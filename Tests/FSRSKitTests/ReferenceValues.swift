import Foundation
@testable import FSRSKit

/// What the Rust reference computed (Tools/reference), decoded from `Reference/`.
struct ReferenceValues: Decodable {
    struct Curve: Decodable {
        var parameters: String
        var state: MemoryState
        var elapsed: [Double]
        var retrievability: [Double]
        var desiredRetention: [Double]
        var intervals: [Double]
    }

    struct Outcome: Decodable {
        var state: MemoryState
        var interval: Double
    }

    struct NextStatesCase: Decodable {
        var parameters: String
        var state: MemoryState?
        var elapsed: Double
        var desiredRetention: Double
        var outcomes: [String: Outcome]
    }

    struct Replay: Decodable {
        var parameters: String
        var card: Int
        var state: MemoryState
    }

    struct Scores: Decodable {
        var logLoss: Double
        var rmseBins: Double
    }

    struct Optimized: Decodable {
        var parameters: [Double]
        var evaluation: Scores
    }

    var commit: String
    var parameterSets: [String: [Double]]
    var curves: [Curve]
    var nextStates: [NextStatesCase]
    var memoryStates: [Replay]
    var scoredReviews: Int
    var evaluations: [String: Scores]
    var optimized: [String: Optimized]

    static let shared: ReferenceValues = load("reference")

    /// The simulated reviews the reference evaluated and optimized on, one
    /// history per card.
    static let histories: [[Review]] = {
        struct Dataset: Decodable { var cards: [[[Double]]] }
        let dataset: Dataset = load("dataset")
        return dataset.cards.map { card in
            card.map { review in
                Review(
                    date: Date(timeIntervalSinceReferenceDate: review[0] * 86_400),
                    rating: Rating(rawValue: Int(review[1]))!
                )
            }
        }
    }()

    /// The scheduler for a named parameter set.
    func fsrs(_ name: String) throws -> FSRS {
        FSRS(parameters: try Parameters(parameterSets[name]!))
    }

    private static func load<Value: Decodable>(_ name: String) -> Value {
        let url = Bundle.module.url(forResource: "Reference", withExtension: nil)!
            .appendingPathComponent("\(name).json")
        return try! JSONDecoder().decode(Value.self, from: Data(contentsOf: url))
    }
}

/// Whether two numbers agree within a relative tolerance, or an absolute one near zero.
func agree(_ a: Double, _ b: Double, relative: Double, absolute: Double = 1e-6) -> Bool {
    abs(a - b) <= max(absolute, relative * max(abs(a), abs(b)))
}

func agree(_ a: MemoryState, _ b: MemoryState, relative: Double) -> Bool {
    agree(a.stability, b.stability, relative: relative)
        && agree(a.fastStability, b.fastStability, relative: relative)
        && agree(a.difficulty, b.difficulty, relative: relative)
}
