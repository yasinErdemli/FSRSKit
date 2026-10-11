import Foundation
import Testing
@testable import FSRSKit

/// FSRSKit against the Rust reference, fsrs-rs, on the same inputs. The
/// reference computes in 32-bit floats and FSRSKit in 64-bit, so they agree
/// to about five digits.
@Suite("Agreement with the reference")
struct ReferenceTests {
    let reference = ReferenceValues.shared

    @Test("The chance of recall matches the reference's")
    func retrievability() throws {
        for curve in reference.curves {
            let fsrs = try reference.fsrs(curve.parameters)
            for (elapsed, expected) in zip(curve.elapsed, curve.retrievability) {
                let actual = fsrs.retrievability(of: curve.state, elapsedDays: elapsed)
                #expect(
                    agree(actual, expected, relative: 1e-5),
                    "\(curve.parameters), \(curve.state), \(elapsed) days: \(actual), expected \(expected)"
                )
            }
        }
    }

    @Test("Intervals match the reference's")
    func intervals() throws {
        for curve in reference.curves {
            let fsrs = try reference.fsrs(curve.parameters)
            for (retention, expected) in zip(curve.desiredRetention, curve.intervals) {
                let actual = fsrs.interval(for: curve.state, desiredRetention: retention)
                #expect(
                    agree(actual, expected, relative: 1e-3),
                    "\(curve.parameters), \(curve.state), retention \(retention): \(actual), expected \(expected)"
                )
            }
        }
    }

    @Test("The states after each rating match the reference's")
    func nextStates() throws {
        for testCase in reference.nextStates {
            let fsrs = try reference.fsrs(testCase.parameters)
            let next = fsrs.nextStates(
                after: testCase.state, elapsedDays: testCase.elapsed, desiredRetention: testCase.desiredRetention
            )
            for rating in Rating.allCases {
                let expected = testCase.outcomes["\(rating)"]!
                let actual = next[rating]
                #expect(
                    agree(actual.state, expected.state, relative: 1e-4),
                    "\(testCase.parameters), \(String(describing: testCase.state)), \(testCase.elapsed) days, \(rating): \(actual.state), expected \(expected.state)"
                )
                #expect(
                    agree(actual.interval, expected.interval, relative: 1e-3),
                    "\(testCase.parameters), \(String(describing: testCase.state)), \(testCase.elapsed) days, \(rating): \(actual.interval), expected \(expected.interval)"
                )
            }
        }
    }

    @Test("Replaying a card's reviews gives the reference's memory state")
    func replay() throws {
        for replay in reference.memoryStates {
            let fsrs = try reference.fsrs(replay.parameters)
            let state = try #require(fsrs.memoryState(after: ReferenceValues.histories[replay.card]))
            #expect(
                agree(state, replay.state, relative: 1e-3),
                "\(replay.parameters), card \(replay.card): \(state), expected \(replay.state)"
            )
        }
    }

    @Test("Evaluating parameters gives the reference's scores", arguments: ["default", "learner"])
    func evaluation(parameters: String) throws {
        let expected = reference.evaluations[parameters]!
        let evaluation = try #require(try reference.fsrs(parameters).evaluate(ReferenceValues.histories))
        #expect(agree(evaluation.logLoss, expected.logLoss, relative: 1e-4))
        #expect(agree(evaluation.rmseBins, expected.rmseBins, relative: 1e-3))
    }

    @Test("Optimizing predicts as well as the reference's optimizer", arguments: [false, true])
    func optimizing(penalties: Bool) throws {
        let expected = reference.optimized[penalties ? "penalties" : "plain"]!.evaluation
        var optimizer = Optimizer()
        optimizer.schedulePenalties = penalties
        let parameters = try optimizer.optimize(ReferenceValues.histories)
        let evaluation = try #require(FSRS(parameters: parameters).evaluate(ReferenceValues.histories))
        // The batches come in a different random order, so the parameters
        // differ a little; how well they predict shouldn't.
        #expect(evaluation.logLoss <= expected.logLoss + 0.001, "\(evaluation.logLoss), reference \(expected.logLoss)")
        #expect(evaluation.rmseBins <= expected.rmseBins + 0.005, "\(evaluation.rmseBins), reference \(expected.rmseBins)")
    }
}
