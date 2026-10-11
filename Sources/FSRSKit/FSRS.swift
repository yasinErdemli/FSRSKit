import Foundation

/// The FSRS-7 scheduler for one learner: how well they remember an item, and
/// when to review it next.
///
/// Times are in days and may be fractional, so reviews minutes apart, in the
/// same lesson, count as what they are. Use ``Review`` and the `Date` methods
/// to work with dates instead.
///
/// ```swift
/// let fsrs = FSRS()
/// let state = fsrs.memoryState(after: reviews)!
/// let due = fsrs.dueDate(for: state, lastReview: reviews.last!.date, desiredRetention: 0.9)
/// ```
public struct FSRS: Sendable, Hashable {
    /// The learner's parameters.
    public var parameters: Parameters

    /// A scheduler with the given parameters: the defaults, or ones fitted by ``Optimizer``.
    public init(parameters: Parameters = .default) {
        self.parameters = parameters
    }

    private var model: Model<Double> { Model(parameters) }

    /// The memory state after a first review.
    public func initialState(after rating: Rating) -> MemoryState {
        MemoryState(model.initialState(after: rating.rawValue))
    }

    /// The memory state after a review.
    ///
    /// - Parameters:
    ///   - state: The state before the review, or `nil` if this is the first.
    ///   - elapsedDays: The days since the last review. Ignored for the first.
    ///   - rating: How the review went.
    public func nextState(after state: MemoryState?, elapsedDays: Double, rating: Rating) -> MemoryState {
        guard let state else { return initialState(after: rating) }
        return MemoryState(model.state(after: .init(state), elapsed: elapsedDays, rating: rating.rawValue))
    }

    /// The chance, from 0 to 1, that the learner remembers the item
    /// `elapsedDays` after its last review.
    public func retrievability(of state: MemoryState, elapsedDays: Double) -> Double {
        model.retrievability(after: elapsedDays, of: .init(state))
    }

    /// The days after the last review until the chance of recall falls to
    /// `desiredRetention`: when to review next. Fractional, and can be under a day.
    ///
    /// - Parameter desiredRetention: The chance of recall to review at, from 0
    ///   to 1. Higher means more reviews and less forgetting; 0.9 is usual.
    public func interval(for state: MemoryState, desiredRetention: Double) -> Double {
        let target = desiredRetention.clamped(to: 0.0001...0.9999)
        guard target < 0.9999 else { return 0 }
        let state = Model<Double>.State(
            stability: state.stability.clamped(to: Bounds.stability),
            fastStability: state.fastStability.clamped(to: Bounds.stability),
            difficulty: state.difficulty.clamped(to: Bounds.difficulty)
        )
        // Newton's method on the logarithm of the interval, as the reference
        // does; bisection if it doesn't land within 0.001 of the target.
        let logBounds = Foundation.log(Bounds.shortestInterval)...Foundation.log(Bounds.stability.upperBound)
        var logInterval = Foundation.log(max(state.stability, state.fastStability, Bounds.shortestInterval))
        for _ in 0..<7 {
            logInterval = logInterval.clamped(to: logBounds)
            let interval = Foundation.exp(logInterval).clamped(to: Bounds.shortestInterval...Bounds.stability.upperBound)
            let (recall, slope) = recallAndSlope(after: interval, of: state)
            let slopeByLog = min(slope * interval, -1e-12)
            logInterval -= ((recall - target) / slopeByLog).clamped(to: -4...4)
            guard logInterval.isFinite else { return bisectedInterval(for: state, target: target) }
        }
        let interval = Foundation.exp(logInterval).clamped(to: 0...Bounds.stability.upperBound)
        let recall = model.retrievability(after: interval, of: state)
        guard recall.isFinite, abs(recall - target) <= 1e-3 else {
            return bisectedInterval(for: state, target: target)
        }
        return interval
    }

    /// The memory state and next interval for each rating the next review could get.
    ///
    /// - Parameters:
    ///   - state: The state now, or `nil` for an item never reviewed.
    ///   - elapsedDays: The days since the last review. Ignored for a new item.
    ///   - desiredRetention: The chance of recall to schedule the next review at.
    public func nextStates(after state: MemoryState?, elapsedDays: Double, desiredRetention: Double) -> NextStates {
        func outcome(_ rating: Rating) -> NextStates.Outcome {
            let next = nextState(after: state, elapsedDays: elapsedDays, rating: rating)
            return .init(state: next, interval: interval(for: next, desiredRetention: desiredRetention))
        }
        return NextStates(again: outcome(.again), hard: outcome(.hard), good: outcome(.good), easy: outcome(.easy))
    }

    /// The chance of recall and how fast it's falling, per day, `elapsed` days
    /// after the last review (the reference's
    /// `fsrs7_forgetting_curve_and_derivative_scalar`).
    private func recallAndSlope(after elapsed: Double, of state: Model<Double>.State) -> (Double, Double) {
        let w = parameters.values
        let elapsed = max(elapsed, 0)
        let stability = state.stability
        let fastStability = state.fastStability
        let difficulty = state.difficulty

        let fastDecay = -(w[23] * Foundation.pow(fastStability, w[33] - 0.3)).clamped(to: 0.01...0.95)
        let fastFactor = Foundation.exp(min(Foundation.log(w[25]) / fastDecay, 60)) - 1
        let fastBase = 1 + fastFactor * (elapsed / fastStability)
        let fast = Foundation.pow(fastBase, fastDecay)
        let fastSlope = fastDecay * Foundation.pow(fastBase, fastDecay - 1) * fastFactor / fastStability

        let slowDecay = -w[24].clamped(to: 0.01...0.95)
        let slowFactor = Foundation.pow(w[26], 1 / slowDecay) - 1
        let timescale = Foundation.exp((difficulty - 5) * (w[32] - 0.3))
        let slowBase = 1 + slowFactor * timescale * (elapsed / stability)
        let slow = Foundation.pow(slowBase, slowDecay)
        let slowSlope = slowDecay * Foundation.pow(slowBase, slowDecay - 1) * slowFactor * timescale / stability

        let fastWeight = w[27] * Foundation.pow(fastStability, -w[29])
        let slowWeight = w[28] * Foundation.pow(stability, w[30]) * Foundation.exp((difficulty - 5) * (w[31] - 0.5))
        let weights = max(fastWeight + slowWeight, 1e-9)
        let recall = (fastWeight * fast + slowWeight * slow) / weights
        let slope = (fastWeight * fastSlope + slowWeight * slowSlope) / weights
        return (recall * (1 - 2e-5) + 1e-5, slope * (1 - 2e-5))
    }

    /// The interval by bisection, for when Newton's method doesn't converge.
    private func bisectedInterval(for state: Model<Double>.State, target: Double) -> Double {
        var low = 0.0
        var high = max(state.stability, state.fastStability, 1)
        while model.retrievability(after: high, of: state) > target, high < Bounds.stability.upperBound {
            high = min(high * 2, Bounds.stability.upperBound)
        }
        for _ in 0..<50 {
            let middle = (low + high) / 2
            if model.retrievability(after: middle, of: state) > target {
                low = middle
            } else {
                high = middle
            }
        }
        return ((low + high) / 2).clamped(to: 0...Bounds.stability.upperBound)
    }
}

/// What the next review would do, for each rating it could get.
public struct NextStates: Sendable, Hashable {
    /// The memory state after a review, and the days until the one after it.
    public struct Outcome: Sendable, Hashable {
        /// The memory state after the review.
        public var state: MemoryState
        /// The days from this review to the next, at the desired retention.
        public var interval: Double
    }

    /// If the learner forgets.
    public var again: Outcome
    /// If the learner remembers with difficulty.
    public var hard: Outcome
    /// If the learner remembers.
    public var good: Outcome
    /// If the learner remembers without effort.
    public var easy: Outcome

    /// The outcome for a rating.
    public subscript(rating: Rating) -> Outcome {
        switch rating {
        case .again: again
        case .hard: hard
        case .good: good
        case .easy: easy
        }
    }
}
