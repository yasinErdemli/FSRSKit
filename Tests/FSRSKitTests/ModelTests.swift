import Foundation
import Testing
@testable import FSRSKit

@Suite("The model")
struct ModelTests {
    let fsrs = FSRS()
    let states = [
        MemoryState(stability: 0.3, fastStability: 0.2, difficulty: 8),
        MemoryState(stability: 4, fastStability: 3, difficulty: 5),
        MemoryState(stability: 90, fastStability: 30, difficulty: 2),
    ]

    @Test("Recall starts near certain and falls as time passes")
    func recallFalls() {
        for state in states {
            #expect(fsrs.retrievability(of: state, elapsedDays: 0) > 0.999)
            var last = 1.0
            for elapsed in stride(from: 0.01, through: 1000, by: 7.3) {
                let recall = fsrs.retrievability(of: state, elapsedDays: elapsed)
                #expect(recall < last)
                last = recall
            }
        }
    }

    @Test("At the interval, recall is the desired retention")
    func intervalHitsRetention() {
        for state in states {
            for retention in [0.75, 0.85, 0.9, 0.95, 0.98] {
                let interval = fsrs.interval(for: state, desiredRetention: retention)
                #expect(abs(fsrs.retrievability(of: state, elapsedDays: interval) - retention) < 1e-3)
            }
        }
    }

    @Test("Wanting to remember more means reviewing sooner")
    func higherRetentionIsSooner() {
        for state in states {
            let intervals = [0.7, 0.8, 0.9, 0.95].map { fsrs.interval(for: state, desiredRetention: $0) }
            #expect(intervals == intervals.sorted(by: >))
        }
    }

    @Test("A better rating never brings the next review sooner")
    func betterRatingIsLater() {
        for state in [nil] + states.map(Optional.some) {
            let next = fsrs.nextStates(after: state, elapsedDays: 3, desiredRetention: 0.9)
            let intervals = Rating.allCases.map { next[$0].interval }
            #expect(intervals == intervals.sorted())
        }
    }

    @Test("Forgetting lowers stability, and remembering after a long gap raises it")
    func lapsesAndGrowth() {
        let state = states[1]
        #expect(fsrs.nextState(after: state, elapsedDays: 4, rating: .again).stability < state.stability)
        #expect(fsrs.nextState(after: state, elapsedDays: 4, rating: .good).stability > state.stability)
    }

    @Test("Reviews replay in date order, whatever order they come in")
    func replayOrder() {
        let start = Date(timeIntervalSinceReferenceDate: 0)
        let reviews = [
            Review(date: start, rating: .good),
            Review(date: start.addingTimeInterval(600), rating: .again),
            Review(date: start.addingTimeInterval(86_400 * 2), rating: .good),
        ]
        #expect(fsrs.memoryState(after: reviews) == fsrs.memoryState(after: reviews.reversed()))
        #expect(fsrs.memoryState(after: []) == nil)
    }

    @Test("The due date is the interval after the last review")
    func dueDate() {
        let last = Date(timeIntervalSinceReferenceDate: 1_000_000)
        let due = fsrs.dueDate(for: states[1], lastReview: last, desiredRetention: 0.9)
        #expect(abs(fsrs.retrievability(of: states[1], lastReview: last, at: due) - 0.9) < 1e-3)
    }
}

@Suite("Parameters")
struct ParametersTests {
    @Test("Parameters need 34 finite values")
    func validation() {
        #expect(throws: Parameters.Error.wrongCount(3)) { try Parameters([1, 2, 3]) }
        var values = Parameters.default.values
        values[5] = .nan
        #expect(throws: Parameters.Error.notFinite(index: 5)) { try Parameters(values) }
    }

    @Test("Values are clipped into range, with the initial stabilities in order")
    func clipping() throws {
        var values = Parameters.default.values
        values[0] = 80
        values[1] = 0.01
        values[23] = 0.5
        values[26] = 0.3
        let parameters = try Parameters(values)
        #expect(parameters[0] == 50)
        #expect(parameters[1] == 50)
        #expect(parameters[23] == 0.25)
        #expect(parameters[26] == 0.5)
    }

    @Test("The defaults are already in range")
    func defaults() throws {
        #expect(try Parameters(Parameters.default.values) == .default)
    }

    @Test("Parameters encode as a plain list and decode back")
    func coding() throws {
        let data = try JSONEncoder().encode(Parameters.default)
        #expect(try JSONDecoder().decode([Double].self, from: data) == Parameters.default.values)
        #expect(try JSONDecoder().decode(Parameters.self, from: data) == .default)
    }
}

@Suite("The optimizer")
struct OptimizerTests {
    @Test("With too few reviews it keeps the starting parameters")
    func tooFewReviews() throws {
        let histories = Array(ReferenceValues.histories.prefix(3))
        #expect(try Optimizer().optimize(histories) == .default)
    }

    @Test("It stops when its task is cancelled")
    func cancellation() async {
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try Optimizer().optimize(ReferenceValues.histories)
        }
        await #expect(throws: CancellationError.self) { try await task.value }
    }

    @Test("The same seed gives the same parameters")
    func repeatable() throws {
        var optimizer = Optimizer()
        optimizer.epochs = 1
        let histories = Array(ReferenceValues.histories.prefix(200))
        #expect(try optimizer.optimize(histories) == optimizer.optimize(histories))
    }
}
