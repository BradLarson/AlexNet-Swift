// swift-tools-version:4.0

import PackageDescription

let package = Package(
    name: "alexnettest",
    products: [
        .executable(name: "AlexNet", targets: ["AlexNet-Swift"]),
    ],
    dependencies: [
    ],
    targets: [
        .target(
            name: "AlexNet-Swift",
            dependencies: [],
			path: "AlexNet"),
    ]
)