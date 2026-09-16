// swift-tools-version: 5.10
// Music Agent — native macOS shell application.
//
// Builds the AppKit/WebKit shell used to launch and host the local
// Music Agent web runtime.
import PackageDescription

let package = Package(
    name: "MusicAgent",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "MusicAgent",
            path: "Sources/MusicAgent",
            linkerSettings: [.linkedFramework("Security")]
        )
    ]
)
