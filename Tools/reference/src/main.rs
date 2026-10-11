//! Writes the values FSRSKit's tests compare against, computed by fsrs-rs at
//! the commit pinned in Cargo.toml.
//!
//!     cargo run --release -- <output folder>
//!
//! It writes `dataset.json`, a year of simulated reviews, and `reference.json`:
//! forgetting curves, intervals, next states, replayed memory states,
//! evaluations and optimized parameters.

use fsrs::{
    ComputeParametersInput, DEFAULT_PARAMETERS, FSRS, FSRSItem, FSRSReview, MemoryState, NextStates,
    compute_parameters,
};
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;

const COMMIT: &str = "0a5737406a0c5c7a5259b144738cbe4a0b649d76";

/// The same generator as FSRSKit's tests use, so a seed means the same everywhere.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number from 0 up to 1.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A learner unlike the default one: slower to forget at first, and with a
/// different curve, so the optimizer has something to find.
fn learner_parameters() -> Vec<f32> {
    let mut w = DEFAULT_PARAMETERS.to_vec();
    for index in 0..4 {
        w[index] *= 1.5;
    }
    w[4] += 0.5;
    w[6] *= 0.8;
    w[7] *= 1.1;
    w[9] *= 1.2;
    w[10] *= 0.8;
    w[15] *= 1.1;
    w[23] = 0.2;
    w[24] = 0.1;
    w[27] *= 1.3;
    w[30] *= 0.9;
    w
}

/// Values outside the model's ranges, to check that both sides clip them alike.
fn raw_parameters() -> Vec<f32> {
    let mut w = DEFAULT_PARAMETERS.to_vec();
    w[0] = 80.0;
    w[1] = 0.01;
    w[4] = 12.0;
    w[23] = 0.5;
    w[26] = 0.3;
    w[29] = -1.0;
    w
}

fn state_json(state: MemoryState) -> Value {
    json!({
        "stability": state.stability,
        "fastStability": state.stability_fast,
        "difficulty": state.difficulty,
    })
}

fn next_states_json(states: &NextStates) -> Value {
    let outcome = |item: &fsrs::ItemState| json!({ "state": state_json(item.memory), "interval": item.interval });
    json!({
        "again": outcome(&states.again),
        "hard": outcome(&states.hard),
        "good": outcome(&states.good),
        "easy": outcome(&states.easy),
    })
}

fn pick(states: &NextStates, rating: u32) -> MemoryState {
    match rating {
        1 => states.again.memory,
        2 => states.hard.memory,
        3 => states.good.memory,
        _ => states.easy.memory,
    }
}

/// One simulated card: the true memory follows the learner's parameters, and
/// the schedule follows the defaults, as an app would before optimizing.
struct Card {
    reviews: Vec<(f64, u32)>,
    truth: MemoryState,
    scheduled: MemoryState,
    time: f64,
}

impl Card {
    fn review(&mut self, gap: f64, truth: &FSRS, scheduler: &FSRS, random: &mut SplitMix64) -> u32 {
        let elapsed = gap as f32;
        let recall = truth.current_retrievability(self.truth, elapsed) as f64;
        let rating = if random.unit() < recall {
            match random.unit() {
                r if r < 0.12 => 2,
                r if r < 0.92 => 3,
                _ => 4,
            }
        } else {
            1
        };
        let next = truth.next_states_with_elapsed_days(Some(self.truth), 0.9, elapsed).unwrap();
        self.truth = pick(&next, rating);
        let next = scheduler.next_states_with_elapsed_days(Some(self.scheduled), 0.9, elapsed).unwrap();
        self.scheduled = pick(&next, rating);
        self.time += gap;
        self.reviews.push((self.time, rating));
        rating
    }
}

fn simulate(truth: &FSRS, scheduler: &FSRS) -> Vec<Vec<(f64, u32)>> {
    let mut random = SplitMix64(20_261_011);
    let minute = 1.0 / 1440.0;
    let mut cards = Vec::new();
    for _ in 0..600 {
        let start = random.unit() * 200.0;
        let first = match random.unit() {
            r if r < 0.25 => 1,
            r if r < 0.35 => 2,
            r if r < 0.90 => 3,
            _ => 4,
        };
        let mut card = Card {
            reviews: vec![(start, first)],
            truth: pick(&truth.next_states(None, 0.9, 0).unwrap(), first),
            scheduled: pick(&scheduler.next_states(None, 0.9, 0).unwrap(), first),
            time: start,
        };
        // Lesson steps: a hard first answer is asked twice more that day.
        let steps = if first <= 2 { 2 } else if random.unit() < 0.5 { 1 } else { 0 };
        for _ in 0..steps {
            let gap = (5.0 + random.unit() * 25.0) * minute;
            card.review(gap, truth, scheduler, &mut random);
        }
        while card.reviews.len() < 40 {
            let interval = scheduler.next_interval_for_state(card.scheduled, 0.9) as f64;
            let mut gap = interval.max(10.0 * minute) * (0.8 + 0.8 * random.unit());
            if random.unit() < 0.1 {
                gap *= 3.0;
            }
            if card.time + gap > 400.0 {
                break;
            }
            if card.review(gap, truth, scheduler, &mut random) == 1 {
                let gap = (10.0 + random.unit() * 20.0) * minute;
                card.review(gap, truth, scheduler, &mut random);
            }
        }
        cards.push(card.reviews);
    }
    cards
}

/// Every review after a card's first, as an item with the reviews before it,
/// in time order, with each item's card.
fn training_items(cards: &[Vec<(f64, u32)>]) -> (Vec<FSRSItem>, Vec<i64>) {
    let mut items = Vec::new();
    for (card_index, card) in cards.iter().enumerate() {
        let reviews: Vec<FSRSReview> = card
            .iter()
            .enumerate()
            .map(|(index, &(time, rating))| FSRSReview {
                rating,
                delta_t: if index == 0 { 0.0 } else { (time - card[index - 1].0) as f32 },
            })
            .collect();
        for index in 1..reviews.len() {
            if reviews[index].delta_t > 0.0 {
                let item = FSRSItem { reviews: reviews[..=index].to_vec() };
                items.push((card[index].0, card_index, index, item));
            }
        }
    }
    items.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let card_ids = items.iter().map(|item| item.1 as i64).collect();
    (items.into_iter().map(|item| item.3).collect(), card_ids)
}

fn main() {
    let output = PathBuf::from(std::env::args().nth(1).expect("usage: fsrskit-reference <output folder>"));
    let parameter_sets = [
        ("default", DEFAULT_PARAMETERS.to_vec()),
        ("learner", learner_parameters()),
        ("raw", raw_parameters()),
    ];

    let states = [
        MemoryState { stability: 0.5, stability_fast: 0.4, difficulty: 7.0 },
        MemoryState { stability: 3.9, stability_fast: 3.1, difficulty: 3.5 },
        MemoryState { stability: 20.0, stability_fast: 8.0, difficulty: 5.0 },
        MemoryState { stability: 150.0, stability_fast: 60.0, difficulty: 2.0 },
        MemoryState { stability: 0.01, stability_fast: 0.008, difficulty: 9.5 },
    ];
    let minute = 1.0 / 1440.0;
    let elapsed: [f32; 11] = [0.0, minute, 10.0 * minute, 0.1, 0.5, 1.0, 2.5, 7.0, 30.0, 100.0, 365.0];
    let retentions: [f32; 6] = [0.7, 0.8, 0.85, 0.9, 0.95, 0.99];
    let next_elapsed: [f32; 5] = [5.0 * minute, 0.5, 1.0, 3.0, 20.0];

    let mut curves = Vec::new();
    let mut next_states = Vec::new();
    for (name, parameters) in &parameter_sets {
        let fsrs = FSRS::new(parameters).unwrap();
        next_states.push(json!({
            "parameters": name,
            "state": Value::Null,
            "elapsed": 0.0,
            "desiredRetention": 0.9,
            "outcomes": next_states_json(&fsrs.next_states(None, 0.9, 0).unwrap()),
        }));
        for state in states {
            curves.push(json!({
                "parameters": name,
                "state": state_json(state),
                "elapsed": elapsed,
                "retrievability": elapsed.iter().map(|&t| fsrs.current_retrievability(state, t)).collect::<Vec<_>>(),
                "desiredRetention": retentions,
                "intervals": retentions.iter().map(|&r| fsrs.next_interval_for_state(state, r)).collect::<Vec<_>>(),
            }));
            for &days in &next_elapsed {
                next_states.push(json!({
                    "parameters": name,
                    "state": state_json(state),
                    "elapsed": days,
                    "desiredRetention": 0.9,
                    "outcomes": next_states_json(&fsrs.next_states_with_elapsed_days(Some(state), 0.9, days).unwrap()),
                }));
            }
        }
    }

    let truth = FSRS::new(&learner_parameters()).unwrap();
    let scheduler = FSRS::new(&DEFAULT_PARAMETERS).unwrap();
    let cards = simulate(&truth, &scheduler);
    let (items, card_ids) = training_items(&cards);

    let mut memory_states = Vec::new();
    for (name, parameters) in &parameter_sets[..2] {
        let fsrs = FSRS::new(parameters).unwrap();
        for (index, card) in cards.iter().enumerate().take(40) {
            let reviews = card
                .iter()
                .enumerate()
                .map(|(i, &(time, rating))| FSRSReview {
                    rating,
                    delta_t: if i == 0 { 0.0 } else { (time - card[i - 1].0) as f32 },
                })
                .collect();
            let state = fsrs.memory_state(FSRSItem { reviews }, None).unwrap();
            memory_states.push(json!({ "parameters": name, "card": index, "state": state_json(state) }));
        }
    }

    let evaluate = |parameters: &[f32]| {
        let evaluation = FSRS::new(parameters)
            .unwrap()
            .evaluate_with_card_ids(items.clone(), card_ids.clone(), |_| true)
            .unwrap();
        json!({ "logLoss": evaluation.log_loss, "rmseBins": evaluation.rmse_bins })
    };
    let optimize = |penalties: bool| {
        let parameters = compute_parameters(ComputeParametersInput {
            train_set: items.clone(),
            card_ids: Some(card_ids.clone()),
            enable_sched_penalties: penalties,
            ..Default::default()
        })
        .unwrap();
        let evaluation = evaluate(&parameters);
        json!({ "parameters": parameters, "evaluation": evaluation })
    };

    let reference = json!({
        "commit": COMMIT,
        "parameterSets": parameter_sets.iter().map(|(name, w)| (name.to_string(), json!(w))).collect::<serde_json::Map<_, _>>(),
        "curves": curves,
        "nextStates": next_states,
        "memoryStates": memory_states,
        "scoredReviews": items.len(),
        "evaluations": {
            "default": evaluate(&DEFAULT_PARAMETERS),
            "learner": evaluate(&learner_parameters()),
        },
        "optimized": {
            "plain": optimize(false),
            "penalties": optimize(true),
        },
    });

    fs::create_dir_all(&output).unwrap();
    let dataset = json!({ "cards": cards });
    fs::write(output.join("dataset.json"), serde_json::to_string(&dataset).unwrap()).unwrap();
    fs::write(output.join("reference.json"), serde_json::to_string_pretty(&reference).unwrap()).unwrap();
    println!("{} cards, {} scored reviews", cards.len(), items.len());
}
