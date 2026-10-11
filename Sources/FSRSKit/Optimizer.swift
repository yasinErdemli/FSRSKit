import Foundation

/// Fits FSRS-7's parameters to one learner's reviews.
///
/// Most of FSRS's accuracy comes from this: the defaults suit an average
/// learner, and fitted parameters suit this one. It's a port of the reference
/// optimizer (fsrs-rs): Adam with a cosine learning-rate schedule, recent
/// reviews weighted more, a pull toward the starting parameters so a small
/// history can't drag them far, and optionally ``schedulePenalties``.
///
/// ```swift
/// let fitted = try Optimizer().optimize(histories, startingFrom: current)
/// let better = FSRS(parameters: fitted).evaluate(histories)!.logLoss
///     < FSRS(parameters: current).evaluate(histories)!.logLoss
/// ```
///
/// It blocks the calling thread while it works, and spreads the work over the
/// device's cores: on an M4 Max, a hundred thousand reviews take half a
/// second. Call it from a background task; it stops with `CancellationError`
/// when that task is cancelled.
public struct Optimizer: Sendable {
    /// How many times to go through the reviews.
    public var epochs = 9
    /// The learning rate at the start; it falls along a cosine to zero.
    public var learningRate = 0.0118
    /// About how many scored reviews each step of the optimizer looks at.
    public var batchSize = 512
    /// Reviews with longer histories than this are left out of training.
    public var longestHistory = 1024
    /// Seeds the order of the batches, so a run can be repeated exactly.
    public var seed: UInt64 = 2023
    /// Whether to add the scheduling penalties: they keep intervals from
    /// jumping wildly at low retention, or shrinking to seconds at very high
    /// retention, for slightly worse predictions. Anki trains FSRS-7 with them.
    public var schedulePenalties = true

    /// An optimizer with the reference's settings.
    public init() {}

    /// Why optimizing failed.
    public enum Error: Swift.Error, Hashable {
        /// A parameter became infinite or not a number.
        case diverged
    }

    /// Parameters fitted to the reviews, one history per card.
    ///
    /// With fewer than 64 reviews to score, or only first reviews after a
    /// day, there's nothing to learn from, and it returns `initial`.
    ///
    /// - Parameters:
    ///   - histories: Each card's reviews. Every review after a card's first
    ///     is scored, same-day ones included.
    ///   - initial: Where to start, and what to stay near.
    ///   - progress: Called after each step with the fraction done, 0 to 1.
    public func optimize(
        _ histories: [[Review]],
        startingFrom initial: Parameters = .default,
        progress: ((Double) -> Void)? = nil
    ) throws -> Parameters {
        let data = TrainingData(histories)
        let count = data.targets.count
        guard count >= 64 else { return initial }
        let onlyFirstReviews = data.targets.allSatisfy { data.cards[$0.card].longTermCounts[$0.row] == 1 }
        guard !onlyFirstReviews else { return initial }

        let batches = makeBatches(data)
        let scored = batches.reduce(0) { $0 + $1.count }
        var parameters = initial.values
        var adam = Adam()
        var schedule = CosineSchedule(steps: Double(batches.count * epochs), start: learningRate)
        var random = SplitMix64(seed: seed)
        var order = Array(batches.indices)
        let tapes = (0..<Self.chunks).map { _ in Tape() }
        let steps = Double(batches.count * epochs)
        var step = 0.0

        for _ in 0..<epochs {
            Self.shuffle(&order, using: &random)
            for batchIndex in order {
                try Task.checkCancellation()
                let batch = batches[batchIndex]
                let current = Parameters(clipping: parameters)
                var gradient = Self.lossGradient(batch, current, tapes: tapes)

                // The pull toward the starting parameters, and the penalties,
                // are spread over the batches in proportion to their size.
                let share = Double(batch.count) / Double(scored)
                for index in 0..<Parameters.count {
                    let deviation = parameters[index] - initial[index]
                    let spread = Self.parameterSpread[index]
                    gradient[index] += 2 * deviation / (spread * spread) * Self.pullWeight * share
                }
                if schedulePenalties {
                    let penalty = SchedulePenalty.penalty(current, tape: tapes[0])
                    if penalty.value.isFinite {
                        for index in 0..<Parameters.count where penalty.gradient[index].isFinite {
                            gradient[index] += penalty.gradient[index] * share
                        }
                    }
                }

                adam.step(&parameters, gradient: gradient, learningRate: schedule.next())
                Parameters.clip(&parameters)
                step += 1
                progress?(step / steps)
            }
        }

        guard parameters.allSatisfy(\.isFinite) else { throw Error.diverged }
        Self.smoothInitialStabilities(&parameters)
        return Parameters(clipping: parameters)
    }

    /// How many parts each batch is split into, to work on in parallel.
    static let chunks = 8

    /// How strongly the parameters are pulled toward where they started.
    private static let pullWeight = 0.3333

    /// How far each parameter typically varies between learners: the pull is
    /// weaker on those that vary more. The initial stabilities aren't pulled.
    private static let parameterSpread: [Double] = [
        9999, 9999, 9999, 9999, 0.523, 0.2528, 0.4329, 0.2966, 0.2139, 0.2889, 0.1862, 0.175,
        0.3812, 0.3013, 0.9104, 0.3234, 0.2448, 0.3273, 0.1842, 0.1735, 0.4608, 0.311, 0.864, 0.0418,
        0.2596, 0.0798, 0.0682, 0.1282, 0.1397, 0.1407, 0.1489, 0.2, 0.15, 0.15,
    ]

    /// One card's reviews up to its last scored one, with the scored ones
    /// marked by whether they were remembered and how much they count.
    struct TrainingCard {
        var elapsed: [Double]
        var ratings: [Int]
        /// The weight of each review's score, or 0 if it isn't scored.
        var weights: [Double]
        var recalled: [Bool]
    }

    /// Cards trained on together in one step.
    struct Batch {
        var cards: [TrainingCard]
        /// How many reviews are scored in the batch.
        var count: Int
    }

    /// Weighs the scored reviews by recency, and groups them by card into
    /// batches of about `batchSize` scored reviews, cards with similar
    /// history lengths together (the reference's `build_windowed_batches`).
    func makeBatches(_ data: TrainingData) -> [Batch] {
        let count = data.targets.count
        var cards: [Int: TrainingCard] = [:]
        for (index, target) in data.targets.enumerated() where target.row < longestHistory {
            // The reference's FSRS-7 recency weights: the newest reviews count
            // fifteen times as much as the oldest.
            let weight = 0.0667 + 0.9333 * Self.power11_25(Double(index) / Double(count))
            let source = data.cards[target.card]
            var card = cards[target.card] ?? TrainingCard(
                elapsed: [], ratings: [],
                weights: Array(repeating: 0, count: source.ratings.count),
                recalled: Array(repeating: false, count: source.ratings.count)
            )
            card.weights[target.row] = weight
            card.recalled[target.row] = target.recalled
            cards[target.card] = card
        }

        var trimmed: [(index: Int, card: TrainingCard)] = []
        for (index, var card) in cards {
            let source = data.cards[index]
            let length = (card.weights.lastIndex { $0 != 0 } ?? 0) + 1
            card.elapsed = Array(source.elapsed[..<length])
            card.ratings = Array(source.ratings[..<length])
            card.weights = Array(card.weights[..<length])
            card.recalled = Array(card.recalled[..<length])
            trimmed.append((index, card))
        }
        trimmed.sort { ($0.card.ratings.count, $0.index) < ($1.card.ratings.count, $1.index) }

        var batches: [Batch] = []
        var current = Batch(cards: [], count: 0)
        for (_, card) in trimmed {
            let scored = card.weights.count(where: { $0 != 0 })
            if !current.cards.isEmpty, current.count + scored > batchSize {
                batches.append(current)
                current = Batch(cards: [], count: 0)
            }
            current.cards.append(card)
            current.count += scored
        }
        if !current.cards.isEmpty { batches.append(current) }
        return batches
    }

    /// The gradient of the batch's loss: the weighted binary cross-entropy of
    /// every scored review, summed.
    ///
    /// The cards are split into one chunk per tape, worked on in parallel, and
    /// summed in order. The number of chunks is fixed, not the number of
    /// cores, so a seed gives the same parameters on every device.
    static func lossGradient(_ batch: Batch, _ parameters: Parameters, tapes: [Tape]) -> [Double] {
        let chunks = min(tapes.count, batch.cards.count)
        let partials = Partials(chunks: chunks)
        defer { partials.deallocate() }
        DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
            let cards = batch.cards.count * chunk / chunks..<batch.cards.count * (chunk + 1) / chunks
            partials.store(lossGradient(batch.cards[cards], parameters, tape: tapes[chunk]), chunk: chunk)
        }
        return partials.sum()
    }

    /// The gradient of some cards' loss, one tape pass per card.
    private static func lossGradient(_ cards: ArraySlice<TrainingCard>, _ parameters: Parameters, tape: Tape) -> [Double] {
        var gradient = [Double](repeating: 0, count: Parameters.count)
        for card in cards {
            let model = Model(parameters, on: tape)
            var loss = Tracked(0)
            var state = model.initialState(after: card.ratings[0])
            for row in 1..<card.ratings.count {
                let elapsed = Tracked(card.elapsed[row])
                if card.weights[row] != 0 {
                    let recall = model.retrievability(after: elapsed, of: state).clamped(to: 1e-5...(1 - 1e-5))
                    let likelihood = card.recalled[row] ? recall : 1 - recall
                    loss = loss + likelihood.log() * -card.weights[row]
                }
                if row + 1 < card.ratings.count {
                    state = model.state(after: state, elapsed: elapsed, rating: card.ratings[row])
                }
            }
            tape.backward(from: loss, into: &gradient)
        }
        return gradient
    }

    /// The batch's loss, the sum the gradient is of.
    static func loss(_ batch: Batch, _ parameters: Parameters) -> Double {
        let model = Model<Double>(parameters)
        var loss = 0.0
        for card in batch.cards {
            var state = model.initialState(after: card.ratings[0])
            for row in 1..<card.ratings.count {
                if card.weights[row] != 0 {
                    let recall = model.retrievability(after: card.elapsed[row], of: state).clamped(to: 1e-5...(1 - 1e-5))
                    loss -= card.weights[row] * Foundation.log(card.recalled[row] ? recall : 1 - recall)
                }
                if row + 1 < card.ratings.count {
                    state = model.state(after: state, elapsed: card.elapsed[row], rating: card.ratings[row])
                }
            }
        }
        return loss
    }

    /// Keeps the four initial stabilities in order after training: a better
    /// first rating never gives a shorter memory (the reference's
    /// `smooth_initial_stabilities_fsrs7`, with all four known).
    static func smoothInitialStabilities(_ parameters: inout [Double]) {
        for index in 0..<4 {
            parameters[index] = parameters[index].clamped(to: 0.0001...Bounds.largestInitialStability)
            if index > 0 { parameters[index] = max(parameters[index], parameters[index - 1]) }
        }
    }

    /// x^11.25 for x from 0 to 1, as the reference computes it.
    private static func power11_25(_ x: Double) -> Double {
        let x2 = x * x
        let x4 = x2 * x2
        let x8 = x4 * x4
        return x8 * x2 * x * x.squareRoot().squareRoot()
    }

    /// Fisher–Yates, with our own generator so a seed gives the same order everywhere.
    private static func shuffle(_ order: inout [Int], using random: inout SplitMix64) {
        guard order.count > 1 else { return }
        for index in stride(from: order.count - 1, to: 0, by: -1) {
            let other = Int(random.next() % UInt64(index + 1))
            order.swapAt(index, other)
        }
    }
}

/// Each parallel chunk's gradient, in a buffer each chunk writes only its
/// own part of, so the chunks can share it.
private struct Partials: @unchecked Sendable {
    private let chunks: Int
    private let values: UnsafeMutablePointer<Double>

    init(chunks: Int) {
        self.chunks = chunks
        values = .allocate(capacity: chunks * Parameters.count)
    }

    func store(_ gradient: [Double], chunk: Int) {
        for index in 0..<Parameters.count {
            values[chunk * Parameters.count + index] = gradient[index]
        }
    }

    /// The chunks' gradients added up, in chunk order.
    func sum() -> [Double] {
        var gradient = [Double](repeating: 0, count: Parameters.count)
        for chunk in 0..<chunks {
            for index in 0..<Parameters.count {
                gradient[index] += values[chunk * Parameters.count + index]
            }
        }
        return gradient
    }

    func deallocate() {
        values.deallocate()
    }
}

/// Adam, with the reference's settings.
private struct Adam {
    private let beta1 = 0.70
    private let beta2 = 0.98
    private let epsilon = 1e-8
    private var first = [Double](repeating: 0, count: Parameters.count)
    private var second = [Double](repeating: 0, count: Parameters.count)
    private var time = 0.0

    mutating func step(_ parameters: inout [Double], gradient: [Double], learningRate: Double) {
        time += 1
        let firstCorrection = 1 - Foundation.pow(beta1, time)
        let secondCorrection = 1 - Foundation.pow(beta2, time)
        for index in 0..<Parameters.count {
            let g = gradient[index]
            first[index] = first[index] * beta1 + g * (1 - beta1)
            second[index] = second[index] * beta2 + g * g * (1 - beta2)
            parameters[index] -= learningRate * (first[index] / firstCorrection)
                / ((second[index] / secondCorrection).squareRoot() + epsilon)
        }
    }
}

/// A learning rate that falls along half a cosine, from its start to zero
/// over the given steps (PyTorch's `CosineAnnealingLR`, as the reference
/// computes it).
private struct CosineSchedule {
    let steps: Double
    let start: Double
    private var count = -1.0
    private var rate: Double

    init(steps: Double, start: Double) {
        self.steps = steps
        self.start = start
        rate = start
    }

    mutating func next() -> Double {
        count += 1
        if count == 0 {
            rate = start
        } else if (count - 1 - steps).truncatingRemainder(dividingBy: 2 * steps) == 0 {
            rate = start * (1 - cos(.pi / steps)) / 2
        } else {
            rate = (1 + cos(.pi * count / steps)) / (1 + cos(.pi * (count - 1) / steps)) * rate
        }
        return rate
    }
}

/// A small, fast random number generator with a fixed algorithm, so the same
/// seed shuffles the same way on every platform.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
