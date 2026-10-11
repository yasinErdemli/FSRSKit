// swift-tools-version: 6.2
import PackageDescription

// The same strict settings for the library and its tests.
let swiftSettings: [SwiftSetting] = [
    .enableUpcomingFeature("InferIsolatedConformances"),
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
    .enableUpcomingFeature("MemberImportVisibility"),
]

let package = Package(
    name: "FSRSKit",
    // Nothing here needs a recent OS: the model is arithmetic, and the
    // optimizer's derivatives are SIMD vectors, which every Apple OS has.
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
        .tvOS(.v16),
        .watchOS(.v9),
        .visionOS(.v1),
    ],
    products: [
        // FSRS-7: memory states, recall, intervals, and the optimizer that
        // fits the parameters to one learner's reviews.
        .library(name: "FSRSKit", targets: ["FSRSKit"]),
    ],
    targets: [
        .target(name: "FSRSKit", swiftSettings: swiftSettings),
        .testTarget(
            name: "FSRSKitTests",
            dependencies: ["FSRSKit"],
            // Values from the Rust reference, made by Tools/reference.
            resources: [.copy("Reference")],
            swiftSettings: swiftSettings
        ),
    ]
)
