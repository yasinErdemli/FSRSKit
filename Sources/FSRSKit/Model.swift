/// FSRS-7's memory model, written once for scheduling (`Double`) and training
/// (``Tracked``).
///
/// A port of the reference's `model_v7.rs` and `analytic_v7.rs` (fsrs-rs, see
/// NOTICE.md). Comments name the reference's terms where they differ from ours.
struct Model<Number: ModelNumber> {
    /// The 34 parameters, as numbers of the kind the model computes with.
    let w: [Number]

    /// A memory state in the model's number type.
    struct State {
        var stability: Number
        var fastStability: Number
        var difficulty: Number
    }

    /// The state after a first review.
    func initialState(after rating: Int) -> State {
        let stability = w[rating - 1]
        return State(
            stability: stability.clamped(to: Bounds.stability),
            fastStability: (stability * 0.8).clamped(to: Bounds.stability),
            difficulty: initialDifficulty(after: rating).clamped(to: Bounds.difficulty)
        )
    }

    /// The difficulty after a first review, before clamping.
    func initialDifficulty(after rating: Int) -> Number {
        w[4] - (w[5] * Double(rating - 1)).exp() + 1
    }

    /// The chance of recall `elapsed` days after the last review: a weighted
    /// mix of the fast and the slow trace's forgetting curves, each a power
    /// law. Kept just inside 0 to 1, as the reference does.
    func retrievability(after elapsed: Number, of state: State) -> Number {
        let elapsed = elapsed.clamped(atLeast: 0)
        let stability = state.stability.clamped(to: Bounds.stability)
        let fastStability = state.fastStability.clamped(to: Bounds.stability)
        let difficulty = state.difficulty.clamped(to: Bounds.difficulty)

        let fast = fastRecall(after: elapsed, fastStability: fastStability)

        let slowDecay = -w[24].clamped(to: 0.01...0.95)
        let slowFactor = w[26].pow(1 / slowDecay) - 1
        // Harder items fade faster: difficulty scales the slow trace's time.
        let timescale = ((difficulty - 5) * (w[32] - 0.3)).exp()
        let slow = (elapsed / stability * slowFactor * timescale + 1).pow(slowDecay)

        let fastWeight = w[27] * fastStability.pow(-w[29])
        let slowWeight = w[28] * stability.pow(w[30]) * ((difficulty - 5) * (w[31] - 0.5)).exp()
        let recall = (fastWeight * fast + slowWeight * slow) / (fastWeight + slowWeight)
        return recall * (1 - 2e-5) + 1e-5
    }

    /// The fast trace's forgetting curve alone. Its decay depends on its own
    /// stability, so short memories fall off differently from long ones.
    func fastRecall(after elapsed: Number, fastStability: Number) -> Number {
        let elapsed = elapsed.clamped(atLeast: 0)
        let fastStability = fastStability.clamped(to: Bounds.stability)
        let decay = -(w[23] * fastStability.pow(w[33] - 0.3)).clamped(to: 0.01...0.95)
        let factor = (w[25].log() / decay).clamped(atMost: 60).exp() - 1
        return (elapsed / fastStability * factor + 1).pow(decay)
    }

    /// A trace's stability after a review (the reference's
    /// `stability_for_set`). The trace's eight parameters start at `first`:
    /// 7 for the slow trace, 15 for the fast one. Not clamped.
    func stability(
        after stability: Number,
        difficulty: Number,
        recall: Number,
        rating: Int,
        parametersFrom first: Int
    ) -> Number {
        // After a lapse the memory is rebuilt from what's left; a trace never
        // grows from forgetting.
        let afterLapse = w[first + 3]
            * ((stability + 1).pow(w[first + 4]) - 1)
            * ((1 - recall) * w[first + 5]).exp()
        let lapsed = smaller(stability, afterLapse)
        guard rating > 1 else { return lapsed }

        // Remembering grows the trace, most when the memory was nearly lost,
        // when it was still weak, and when the item is easy.
        var growth = (w[first] - 1.5).exp()
            * (11 - difficulty)
            * stability.pow(-w[first + 1])
            * (((1 - recall) * w[first + 2]).exp() - 1)
        if rating == 2 { growth = growth * w[first + 6] }
        if rating == 4 { growth = growth * w[first + 7] }
        return larger(lapsed, stability * (growth + 1))
    }

    /// The difficulty after a review: again and hard raise it, easy lowers
    /// it, it changes less near 10, and it drifts slowly back toward an easy
    /// item's difficulty. Forgetting raises it less when recall was already low.
    func nextDifficulty(_ difficulty: Number, rating: Int, recall: Number) -> Number {
        var change = -w[6] * (Double(rating) - 3)
        if rating == 1 { change = change * (recall + 0.1) }
        let damped = difficulty + (10 - difficulty) * change * (1.0 / 9)
        return (initialDifficulty(after: 4) * 0.01 + damped * 0.99).clamped(to: Bounds.difficulty)
    }

    /// The state after a review `elapsed` days after the last one.
    func state(after state: State, elapsed: Number, rating: Int) -> State {
        let stability = state.stability.clamped(to: Bounds.stability)
        let difficulty = state.difficulty.clamped(to: Bounds.difficulty)
        let fastStability = state.fastStability.clamped(to: Bounds.stability)
        let elapsed = elapsed.clamped(atLeast: 0)

        let current = State(stability: stability, fastStability: fastStability, difficulty: difficulty)
        let recall = retrievability(after: elapsed, of: current)
        let slow = self.stability(
            after: stability, difficulty: difficulty, recall: recall, rating: rating, parametersFrom: 7
        )
        let fastRecall = fastRecall(after: elapsed, fastStability: fastStability)
        var fast = self.stability(
            after: fastStability, difficulty: difficulty, recall: fastRecall, rating: rating,
            parametersFrom: 15
        )
        // After a lapse the fast trace starts below the slow one.
        if rating == 1 { fast = smaller(fast, slow * 0.8) }
        return State(
            stability: slow.clamped(to: Bounds.stability),
            fastStability: fast.clamped(to: Bounds.stability),
            difficulty: nextDifficulty(difficulty, rating: rating, recall: recall)
        )
    }
}

extension Model<Double> {
    init(_ parameters: Parameters) {
        w = parameters.values
    }
}

extension Model<Double>.State {
    init(_ state: MemoryState) {
        self.init(stability: state.stability, fastStability: state.fastStability, difficulty: state.difficulty)
    }
}

extension MemoryState {
    init(_ state: Model<Double>.State) {
        self.init(stability: state.stability, fastStability: state.fastStability, difficulty: state.difficulty)
    }
}
