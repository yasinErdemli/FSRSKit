/// How well parameters predict a learner's reviews: lower is better for both.
///
/// Compare the evaluations of the current and the newly optimized parameters
/// on the same reviews, and keep the new ones only if their log loss is lower.
public struct Evaluation: Sendable, Hashable {
    /// How far the predicted chances of recall were from what happened, on
    /// average (binary cross-entropy). The main measure.
    public var logLoss: Double
    /// How far predicted recall was from actual recall, among reviews grouped
    /// by interval, review count and lapses (the benchmark's RMSE (bins)).
    public var rmseBins: Double
}

extension FSRS {
    /// How well these parameters predict the reviews, one history per card.
    /// Recent reviews count more, as in the reference. `nil` if no card has
    /// more than one review.
    public func evaluate(_ histories: [[Review]]) -> Evaluation? {
        let data = TrainingData(histories)
        guard !data.targets.isEmpty else { return nil }
        let predictions = data.predictions(Model(parameters))
        let last = Double(max(data.targets.count - 1, 1))

        var loss = 0.0
        var totalWeight = 0.0
        var bins: [Bin: (predicted: Double, actual: Double, count: Double, weight: Double)] = [:]
        for (index, target) in data.targets.enumerated() {
            let weight = 0.25 + 0.75 * power(Double(index) / last, 3)
            let actual = target.recalled ? 1.0 : 0.0
            let predicted = predictions[index]
            let clamped = predicted.clamped(to: 0.0001...0.9999)
            loss -= weight * (actual * logarithm(clamped) + (1 - actual) * logarithm(1 - clamped))
            totalWeight += weight

            let card = data.cards[target.card]
            let bin = Bin(
                elapsed: card.elapsed[target.row],
                longTermReviews: card.longTermCounts[target.row],
                lapses: card.lapseCounts[target.row]
            )
            var sums = bins[bin, default: (0, 0, 0, 0)]
            sums.predicted += predicted
            sums.actual += actual
            sums.count += 1
            sums.weight += weight
            bins[bin] = sums
        }

        var squares = 0.0
        var binWeight = 0.0
        for sums in bins.values {
            let difference = sums.predicted / sums.count - sums.actual / sums.count
            squares += difference * difference * sums.weight
            binWeight += sums.weight
        }
        return Evaluation(logLoss: loss / totalWeight, rmseBins: (squares / binWeight).squareRoot())
    }
}

/// A group of reviews for RMSE (bins), by interval, review count and lapses,
/// each on a roughly logarithmic scale (the reference's `r_matrix_index`).
private struct Bin: Hashable {
    var interval: Int
    var length: Int
    var lapses: Int

    init(elapsed: Double, longTermReviews: Int, lapses: Int) {
        func bucket(_ value: Double, base: Double, scale: Double) -> Double {
            scale * power(base, (logarithm(value) / logarithm(base)).rounded(.down))
        }
        interval = Int((bucket(elapsed, base: 3.62, scale: 2.48) * 100).rounded())
        length = Int(bucket(Double(longTermReviews + 1), base: 1.89, scale: 1.99).rounded())
        self.lapses = lapses == 0 ? 0 : Int(bucket(Double(lapses), base: 1.73, scale: 1.65).rounded())
    }
}
