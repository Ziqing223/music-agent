import Foundation

private final class StubStore: DeepSeekCredentialReading {
    let value: String?
    private(set) var readCount = 0

    init(value: String?) {
        self.value = value
    }

    func read() throws -> String? {
        readCount += 1
        return value
    }
}

private func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    guard condition() else {
        FileHandle.standardError.write(Data("FAIL: \(message)\n".utf8))
        exit(1)
    }
}

@main
struct KeychainCredentialContractProbe {
    static func main() throws {
        let environmentStore = StubStore(value: "keychain-value")
        let environmentResult = try DeepSeekCredentialResolver.resolve(
            environment: ["DEEPSEEK_API_KEY": "environment-value"],
            store: environmentStore)
        require(environmentResult == .available(ResolvedDeepSeekCredential(
            value: "environment-value", source: .environment)),
                "environment credential must win")
        require(environmentStore.readCount == 0,
                "environment precedence must not read Keychain")

        let keychainResult = try DeepSeekCredentialResolver.resolve(
            environment: [:], store: StubStore(value: "keychain-value"))
        require(keychainResult == .available(ResolvedDeepSeekCredential(
            value: "keychain-value", source: .keychain)),
                "Keychain fallback must resolve")
        let missingResult = try DeepSeekCredentialResolver.resolve(
            environment: [:], store: StubStore(value: nil))
        require(missingResult == .missing,
                "missing credentials must remain missing")

        let normalized = try DeepSeekCredentialInput.normalize("  valid-key  \n")
        require(normalized == "valid-key",
                "surrounding whitespace must be trimmed")
        for malformed in ["   ", "key with space", "\"quoted\""] {
            do {
                _ = try DeepSeekCredentialInput.normalize(malformed)
                require(false, "malformed credential was accepted")
            } catch is DeepSeekCredentialInputError {
                // Expected; never print the rejected value.
            }
        }

        let runtime = ShellRuntime(repoRoot: "/repo",
                                   pythonExecutable: "/repo/.venv/bin/python",
                                   databasePath: "/db")
        let command = ShellCommand.build(
            runtime: runtime,
            port: 1234,
            inherited: ["UNRELATED": "kept", "DEEPSEEK_API_KEY": "old-value"],
            deepSeekAPIKey: "resolved-value")
        require(command.environment["DEEPSEEK_API_KEY"] == "resolved-value",
                "resolved credential must replace inherited child value")
        require(command.environment["UNRELATED"] == "kept",
                "unrelated child environment must remain intact")

        let store = KeychainCredentialStore(
            service: "Music Agent Tests \(UUID().uuidString)",
            account: "DEEPSEEK_API_KEY")
        defer { try? store.delete() }
        let initiallyStored = try store.read()
        require(initiallyStored == nil, "temporary Keychain item must start absent")
        try store.save("first-value")
        let firstStored = try store.read()
        require(firstStored == "first-value", "Keychain create/read failed")
        try store.save("replacement-value")
        let replacementStored = try store.read()
        require(replacementStored == "replacement-value", "Keychain update failed")
        try store.delete()
        let deletedValue = try store.read()
        require(deletedValue == nil, "Keychain delete failed")

        print("PASS: Keychain credential contracts")
    }
}
