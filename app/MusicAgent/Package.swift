// swift-tools-version: 5.10
// Music Agent — macOS shell application (P20 Native App Shell V1).
//
// Slice A scope: prove this CLT-only machine can take an AppKit/WebKit GUI
// executable from SwiftPM to a hand-assembled .app bundle. No Python child
// process, no WebShell, no HTTP — those land in slices B and C.
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
