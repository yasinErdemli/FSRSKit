import Foundation

/// A learner's reviews arranged for training and evaluation: each card's
/// reviews as days since the one before, and every review the model can be
/// scored on, in date order.
///
/// As in the reference benchmark, every review after a card's first is
/// scored, same-day ones included, and a review at the very same moment as
/// the one before it is dropped.
struct TrainingData {
    /// One card's reviews, in date order.
    struct Card {
        /// The days since the review before, for each review; 0 for the first.
        var elapsed: [Double] = []
        /// Each review's rating, 1 to 4.
        var ratings: [Int] = []
        /// For each review, how many reviews up to and including it came a
        /// day or more after the one before (the reference's long-term reviews).
        var longTermCounts: [Int] = []
        /// For each review, how many reviews before it were long-term lapses.
        var lapseCounts: [Int] = []
    }

    /// A review the model is scored on: the chance of recall it predicted from
    /// the card's earlier reviews, against whether the learner remembered.
    struct Target {
        var card: Int
        var row: Int
        var recalled: Bool
        var date: Date
    }

    var cards: [Card] = []
    /// Every review after a card's first, in date order.
    var targets: [Target] = []

    init(_ histories: [[Review]]) {
        for history in histories {
            var card = Card()
            var lastDate: Date?
            var longTerm = 0
            var lapses = 0
            for review in history.sorted(by: { $0.date < $1.date }) {
                let elapsed = lastDate.map { days(from: $0, to: review.date) } ?? 0
                if lastDate != nil, elapsed == 0 { continue }
                let row = card.ratings.count
                card.lapseCounts.append(lapses)
                if elapsed >= 1 {
                    longTerm += 1
                    if review.rating == .again { lapses += 1 }
                }
                card.elapsed.append(elapsed)
                card.ratings.append(review.rating.rawValue)
                card.longTermCounts.append(longTerm)
                if row > 0 {
                    targets.append(Target(card: cards.count, row: row, recalled: review.rating.isRecall, date: review.date))
                }
                lastDate = review.date
            }
            cards.append(card)
        }
        targets.sort { ($0.date, $0.card, $0.row) < ($1.date, $1.card, $1.row) }
    }

    /// The model's chance of recall for every target, in `targets` order,
    /// walking each card's reviews once.
    func predictions(_ model: Model<Double>) -> [Double] {
        var byCard = [[Double]](repeating: [], count: cards.count)
        for (index, card) in cards.enumerated() where card.ratings.count > 1 {
            var recalls = [Double](repeating: 0, count: card.ratings.count)
            var state = model.initialState(after: card.ratings[0])
            for row in 1..<card.ratings.count {
                recalls[row] = model.retrievability(after: card.elapsed[row], of: state)
                state = model.state(after: state, elapsed: card.elapsed[row], rating: card.ratings[row])
            }
            byCard[index] = recalls
        }
        return targets.map { byCard[$0.card][$0.row] }
    }
}
