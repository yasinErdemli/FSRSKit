/// The reference's scheduling penalties for FSRS-7 (`training_v7.rs`),
/// optional extra terms in the training loss.
///
/// Fitting only for prediction can give parameters that schedule badly: huge
/// jumps between intervals at low retention, and intervals of seconds at very
/// high retention. These terms follow a card answered good ten times and
/// penalize both, at the cost of slightly worse predictions. Anki trains
/// FSRS-7 with them on.
enum SchedulePenalty {
    private static let reviews = 10
    private static let growthWeight = 0.5
    private static let shortWeight = 0.0015
    private static let growthRetention = 0.90
    private static let shortRetentions = [0.99]
    /// Ten minutes, as a rate per day: shorter average intervals are penalized.
    private static let shortRate = 86_400.0 / 600
    private static let longestInterval = 36_500.0

    /// The penalty for the parameters, and its gradient.
    static func penalty(_ parameters: Parameters, tape: Tape = Tape()) -> (value: Double, gradient: [Double]) {
        let model = Model(parameters, on: tape)
        let growth = intervalGrowth(model, values: parameters.values)
        let short = shortIntervals(model, values: parameters.values)
        let penalty = growth * growthWeight + short * shortWeight
        var gradient = [Double](repeating: 0, count: Parameters.count)
        tape.backward(from: penalty, into: &gradient)
        return (penalty.value, gradient)
    }

    /// The largest ratio between consecutive intervals of a day or more, squared.
    private static func intervalGrowth(_ model: Model<Tracked>, values: [Double]) -> Tracked {
        var state = model.initialState(after: 3)
        var previous: Tracked?
        var largest: Tracked?
        for _ in 0..<reviews {
            let interval = nextInterval(model, values: values, state: state, retention: growthRetention)
            if let previous, previous.value >= 1 {
                let ratio = interval / previous
                if ratio.value > (largest?.value ?? -.infinity) { largest = ratio }
            }
            previous = interval
            state = model.state(after: state, elapsed: interval, rating: 3)
        }
        return largest.map { $0 * $0 } ?? Tracked(0)
    }

    /// How far the average interval under a day falls below ten minutes, as a rate.
    private static func shortIntervals(_ model: Model<Tracked>, values: [Double]) -> Tracked {
        var total = Tracked(0)
        var counted = 0
        for retention in shortRetentions {
            var state = model.initialState(after: 3)
            var sum = Tracked(0)
            var count = 0
            for _ in 0..<reviews {
                let interval = nextInterval(model, values: values, state: state, retention: retention)
                if interval.value < 1 {
                    sum = sum + interval
                    count += 1
                }
                state = model.state(after: state, elapsed: interval, rating: 3)
            }
            guard count > 0 else { continue }
            let average = (sum / Double(count)).clamped(atLeast: Bounds.shortestInterval)
            total = total + (1 / average).clamped(atLeast: shortRate) - shortRate
            counted += 1
        }
        return counted == 0 ? Tracked(0) : total / Double(counted)
    }

    /// The interval to a retention, differentiable: Newton's method finds it
    /// in plain numbers, and one implicit-function step at the answer gives
    /// its derivatives (the reference's `next_interval_dual`).
    private static func nextInterval(
        _ model: Model<Tracked>,
        values: [Double],
        state: Model<Tracked>.State,
        retention: Double
    ) -> Tracked {
        let interval = newtonInterval(
            values,
            stability: state.stability.value,
            fastStability: state.fastStability.value,
            difficulty: state.difficulty.value,
            retention: retention
        )
        let (recall, slope) = rawRecall(model, after: interval, of: state)
        let slopeByLog = min(slope * interval, -1e-9)
        let logBounds = logarithm(Bounds.shortestInterval)...logarithm(longestInterval)
        return (logarithm(interval) - (recall - retention) / slopeByLog).clamped(to: logBounds).exp()
    }

    /// Newton's method for the interval, in plain numbers (the reference's
    /// `penalty_newton_interval`).
    private static func newtonInterval(
        _ w: [Double],
        stability: Double,
        fastStability: Double,
        difficulty: Double,
        retention: Double
    ) -> Double {
        let logBounds = logarithm(Bounds.shortestInterval)...logarithm(longestInterval)
        let intervalBounds = Bounds.shortestInterval...longestInterval
        let fastDecay = -(w[23] * power(fastStability, w[33] - 0.3)).clamped(to: 0.01...0.95)
        let fastFactor = exponential(min(logarithm(max(w[25], 1e-9)) / fastDecay, 60)) - 1
        let fastRate = fastFactor / fastStability
        let slowDecay = -w[24].clamped(to: 0.01...0.95)
        let slowFactor = power(max(w[26], 1e-9), 1 / slowDecay) - 1
        let timescale = exponential((difficulty - 5) * (w[32] - 0.3))
        let slowRate = slowFactor * timescale / stability
        let fastWeight = w[27] * power(fastStability, -w[29])
        let slowWeight = w[28] * power(stability, w[30]) * exponential((difficulty - 5) * (w[31] - 0.5))
        let weights = fastWeight + slowWeight

        var logInterval = logarithm(max(stability, 1e-10))
        for _ in 0..<7 {
            logInterval = logInterval.clamped(to: logBounds)
            let interval = exponential(logInterval).clamped(to: intervalBounds)
            let fastBase = max(fastRate * interval + 1, 1e-9)
            let slowBase = max(slowRate * interval + 1, 1e-9)
            let recall = (fastWeight * power(fastBase, fastDecay) + slowWeight * power(slowBase, slowDecay)) / weights
            let fastSlope = fastDecay * power(fastBase, fastDecay - 1) * fastRate
            let slowSlope = slowDecay * power(slowBase, slowDecay - 1) * slowRate
            let slope = (fastWeight * fastSlope + slowWeight * slowSlope) / weights
            logInterval -= (recall - retention) / min(slope * interval, -1e-12)
        }
        return exponential(logInterval.clamped(to: logBounds)).clamped(to: intervalBounds)
    }

    /// The two traces' mixed recall, without the model's final rescaling, as a
    /// dual number, and its slope per day as a plain one (the reference's
    /// `penalty_raw_mixture`).
    private static func rawRecall(
        _ model: Model<Tracked>,
        after interval: Double,
        of state: Model<Tracked>.State
    ) -> (Tracked, Double) {
        let w = model.w
        let (stability, fastStability, difficulty) = (state.stability, state.fastStability, state.difficulty)
        let fastDecay = -(w[23] * fastStability.pow(w[33] - 0.3)).clamped(to: 0.01...0.95)
        let fastFactor = (w[25].log() / fastDecay).clamped(atMost: 60).exp() - 1
        let fastRate = fastFactor / fastStability
        let fastBase = (fastRate * interval + 1).clamped(atLeast: 1e-9)
        let fast = fastBase.pow(fastDecay)
        let slowDecay = -w[24].clamped(to: 0.01...0.95)
        let slowFactor = w[26].pow(1 / slowDecay) - 1
        let timescale = ((difficulty - 5) * (w[32] - 0.3)).exp()
        let slowRate = slowFactor * timescale / stability
        let slowBase = (slowRate * interval + 1).clamped(atLeast: 1e-9)
        let slow = slowBase.pow(slowDecay)
        let fastWeight = w[27] * fastStability.pow(-w[29])
        let slowWeight = w[28] * stability.pow(w[30]) * ((difficulty - 5) * (w[31] - 0.5)).exp()
        let weights = (fastWeight + slowWeight).clamped(atLeast: 1e-9)
        let recall = ((fastWeight * fast + slowWeight * slow) / weights).clamped(to: 0...1)

        let fastSlope = fastDecay.value * power(fastBase.value, fastDecay.value - 1) * fastRate.value
        let slowSlope = slowDecay.value * power(slowBase.value, slowDecay.value - 1) * slowRate.value
        let slope = min((fastWeight.value * fastSlope + slowWeight.value * slowSlope) / weights.value, 0)
        return (recall, slope)
    }
}
