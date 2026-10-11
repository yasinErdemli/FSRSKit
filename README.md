# FSRSKit

[![CI](https://github.com/yasinErdemli/FSRSKit/actions/workflows/ci.yml/badge.svg)](https://github.com/yasinErdemli/FSRSKit/actions/workflows/ci.yml)

**FSRS-7 for Swift: the most accurate spaced-repetition scheduler an app can run for each learner, with the optimizer that fits it to them.**

FSRSKit tells you how likely a learner is to remember something right now, and when to review it so they don't forget. It's a port of [fsrs-rs](https://github.com/open-spaced-repetition/fsrs-rs), the reference implementation of [FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm), the algorithm in Anki. It's pure Swift with no dependencies, and runs on every Apple platform and on Linux.

- **FSRS-7**, the newest version. It models a fast and a slow trace of each memory, and takes time in fractional days, so reviews minutes apart, in the same lesson, are predicted as well as reviews weeks apart.
- **The optimizer** fits FSRS's 34 parameters to one learner's own reviews, on the device. That's where most of FSRS's accuracy comes from, and no other Swift library has it.
- **The same answers as the reference**, checked by tests on every curve, interval, state, evaluation and optimization against values fsrs-rs computes.

## Why FSRS-7

The [open spaced-repetition benchmark](https://github.com/open-spaced-repetition/srs-benchmark) predicts the reviews of 10,000 Anki users with each algorithm. Log loss, lower is better:

| Algorithm | Reviews a day or more apart | Every review, same-day ones included |
|---|---|---|
| FSRS-7, optimized, recent reviews weighted more | **0.337** | **0.318** |
| FSRS-6, optimized | 0.346 | 0.384 |
| FSRS-5, optimized | 0.356 | 0.457 |
| FSRS-7 with default parameters | 0.362 | 0.340 |
| Duolingo's half-life regression | 0.469 | 0.705 |

Older versions fall apart on same-day reviews; FSRS-7 was built for them. Among algorithms that can schedule reviews, only neural networks predict better: a two-million-parameter model trained across thousands of users with features specific to Anki, and small recurrent networks that are about 1% better but give no stability or difficulty to schedule with. FSRS-7 is the best an app can fit to one learner and explain.

## Installing

```swift
.package(url: "https://github.com/yasinErdemli/FSRSKit", from: "0.1.0"),
```

Then add `"FSRSKit"` to your target's dependencies. FSRSKit needs Swift 6.2, and iOS 16, macOS 13, tvOS 16, watchOS 9, visionOS 1 or Linux.

FSRS-7 is on fsrs-rs's main branch but not yet in one of its releases, and its default parameters can still change. FSRSKit stays at 0.x until fsrs-rs releases FSRS-7, so a minor version may change the defaults or the optimizer's results.

## Scheduling

Keep each item's reviews, and work out its state from them when you need it:

```swift
import FSRSKit

let fsrs = FSRS()                       // default parameters, or ones you've fitted
let reviews = [
    Review(date: lessonStart, rating: .good),
    Review(date: lessonStart.addingTimeInterval(600), rating: .good),
    Review(date: threeDaysLater, rating: .hard),
]

let state = fsrs.memoryState(after: reviews)!
let recall = fsrs.retrievability(of: state, lastReview: threeDaysLater, at: .now)
let due = fsrs.dueDate(for: state, lastReview: threeDaysLater, desiredRetention: 0.9)
```

- **`desiredRetention`** is the chance of recall to review at. 0.9 is usual: higher means more reviews and less forgetting.
- **Intervals can be under a day.** After a lapse, or early in learning, FSRS-7 may ask for the next review in minutes or hours. Round them however your app works.
- **`nextStates(after:elapsedDays:desiredRetention:)`** gives the state and interval for each rating, to show on answer buttons.
- **Replay rather than store.** A memory state is cheap to rebuild from the reviews, and rebuilding means new parameters apply to the whole history.

## Fitting to a learner

```swift
let histories: [[Review]] = …           // one array per item
let current = learner.parameters        // .default at first

let fitted = try Optimizer().optimize(histories, startingFrom: current)
if let new = FSRS(parameters: fitted).evaluate(histories),
   let old = FSRS(parameters: current).evaluate(histories),
   new.logLoss < old.logLoss {
    learner.parameters = fitted
}
```

- **When:** every few weeks, or after every thousand or so new reviews. With fewer than 64 scorable reviews it returns the starting parameters.
- **How long:** it blocks the calling thread and uses all cores. On an M4 Max, 100,000 reviews take half a second; fsrs-rs, with hand-written NEON kernels, takes an eighth of a second. Run it in a background task; it stops with `CancellationError` when the task is cancelled.
- **Schedule penalties** are on by default, as in Anki: they keep intervals from jumping wildly at low retention or shrinking to seconds at very high retention, for slightly worse predictions. Turn them off with `optimizer.schedulePenalties = false`.
- **The result is repeatable:** the same reviews and `seed` give the same parameters on every device.
- **`Parameters` is `Codable`**, as a plain list of 34 numbers.

## How it matches the reference

`Tools/reference` is a small Rust program that runs fsrs-rs, at the commit pinned in its `Cargo.toml`, on a simulated year of reviews: 600 items from a learner unlike the default one, with lesson steps, lapses and late reviews. It writes everything fsrs-rs computes to `Tests/FSRSKitTests/Reference`, and the tests check FSRSKit against it:

| | Agreement |
|---|---|
| Recall, next states, replayed states | within 0.01% (fsrs-rs computes in 32-bit floats, FSRSKit in 64-bit) |
| Intervals | within 0.1% |
| Evaluation | the same log loss and RMSE |
| Optimizing | log loss 0.3002 against fsrs-rs's 0.3004, from 0.3057 with the defaults |

The optimizer's gradients are also checked against finite differences. To move to a newer fsrs-rs, change the commit in `Tools/reference/Cargo.toml`, run `Tools/make-reference.sh` (it needs Rust), and the tests show what changed.

## How it works

The model is written once, generic over the kind of number it computes with: `Double` to schedule, and a number recorded on a tape to train. The tape records each operation's local derivatives on the forward pass over a card's reviews, and one sweep backward gives the gradient by all 34 parameters (reverse-mode automatic differentiation). So the formulas the optimizer trains are exactly the formulas the scheduler uses.

## License

FSRSKit is under the BSD 3-Clause License: see [LICENSE](LICENSE). It contains code ported from fsrs-rs, under fsrs-rs's own BSD 3-Clause License: see [NOTICE.md](NOTICE.md). An app that ships FSRSKit must include both notices in its credits.
