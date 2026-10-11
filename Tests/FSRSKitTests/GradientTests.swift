import Foundation
import Testing
@testable import FSRSKit

/// The optimizer's derivatives against finite differences: nudge each
/// parameter a little each way and see how the loss moves.
@Suite("Gradients")
struct GradientTests {
    /// Parameters away from their bounds, so a nudge either way isn't clipped.
    let parameters = try! Parameters(ReferenceValues.shared.parameterSets["learner"]!)

    @Test("The loss's gradient matches finite differences")
    func lossGradient() throws {
        var optimizer = Optimizer()
        optimizer.batchSize = 200
        let batch = try #require(optimizer.makeBatches(TrainingData(Array(ReferenceValues.histories.prefix(120)))).last)
        let gradient = Optimizer.lossGradient(batch, parameters, tapes: [Tape(), Tape(), Tape()])
        try expectFiniteDifferences(gradient) { Optimizer.loss(batch, $0) }
    }

    @Test("The scheduling penalty's gradient matches finite differences")
    func penaltyGradient() throws {
        let gradient = SchedulePenalty.penalty(parameters).gradient
        try expectFiniteDifferences(gradient) { SchedulePenalty.penalty($0).value }
    }

    private func expectFiniteDifferences(_ gradient: [Double], of function: (Parameters) -> Double) throws {
        for index in 0..<Parameters.count {
            let step = 1e-6 * max(abs(parameters[index]), 1)
            var up = parameters.values
            var down = parameters.values
            up[index] += step
            down[index] -= step
            let upParameters = try Parameters(up)
            let downParameters = try Parameters(down)
            // Skip parameters a nudge would push out of their range.
            guard upParameters.values == up, downParameters.values == down else { continue }
            let estimate = (function(upParameters) - function(downParameters)) / (2 * step)
            #expect(
                agree(gradient[index], estimate, relative: 1e-4, absolute: 1e-5),
                "parameter \(index): \(gradient[index]), finite differences \(estimate)"
            )
        }
    }
}
