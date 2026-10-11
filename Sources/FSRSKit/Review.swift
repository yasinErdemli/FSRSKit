import Foundation

/// One review of an item: when it happened and how it went.
public struct Review: Sendable, Hashable, Codable {
    /// When the learner answered.
    public var date: Date
    /// How the review went.
    public var rating: Rating

    /// A review at a date with a rating.
    public init(date: Date, rating: Rating) {
        self.date = date
        self.rating = rating
    }
}

/// Seconds in a day: the model's unit of time.
let secondsPerDay = 86_400.0

/// The days from one date to a later one, fractional, and never negative.
func days(from start: Date, to end: Date) -> Double {
    max(end.timeIntervalSince(start) / secondsPerDay, 0)
}

extension FSRS {
    /// The memory state after an item's reviews, replayed in date order, or
    /// `nil` if there are none.
    ///
    /// Replaying the whole history is how FSRS is meant to be used: keep the
    /// reviews, not the state, and the state can always be rebuilt with new
    /// parameters.
    public func memoryState(after reviews: some Sequence<Review>) -> MemoryState? {
        var state: MemoryState?
        var lastDate: Date?
        for review in reviews.sorted(by: { $0.date < $1.date }) {
            let elapsed = lastDate.map { days(from: $0, to: review.date) } ?? 0
            state = nextState(after: state, elapsedDays: elapsed, rating: review.rating)
            lastDate = review.date
        }
        return state
    }

    /// The chance, from 0 to 1, that the learner remembers the item at `date`.
    public func retrievability(of state: MemoryState, lastReview: Date, at date: Date) -> Double {
        retrievability(of: state, elapsedDays: days(from: lastReview, to: date))
    }

    /// When the chance of recall falls to `desiredRetention`: when to review next.
    public func dueDate(for state: MemoryState, lastReview: Date, desiredRetention: Double) -> Date {
        lastReview.addingTimeInterval(interval(for: state, desiredRetention: desiredRetention) * secondsPerDay)
    }
}
