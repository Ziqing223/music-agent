import Foundation

private func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    guard condition() else {
        FileHandle.standardError.write(Data("FAIL: \(message)\n".utf8))
        exit(1)
    }
}

@main
struct ShellCommandContractProbe {
    static func main() throws {
        let configuredIdentity =
            "agt_55555555-5555-4555-8555-555555555555:full"
        let configuredRuntime = ShellRuntime(
            repoRoot: "/repo",
            pythonExecutable: "/repo/.venv/bin/python",
            databasePath: "/store/music_agent.db",
            agentClientEntry: configuredIdentity)
        let configuredCommand = ShellCommand.build(
            runtime: configuredRuntime,
            port: 43127,
            inherited: [:])
        let configuredArgv = [configuredCommand.executablePath]
            + configuredCommand.arguments
        require(configuredArgv == [
            "/repo/.venv/bin/python",
            "-m", "music_agent.cli", "web",
            "--db", "/store/music_agent.db",
            "--no-browser",
            "--port", "43127",
            "--agent-client", configuredIdentity,
        ], "configured identity must appear unchanged in the final Web argv")

        let unconfiguredRuntime = ShellRuntime(
            repoRoot: "/repo",
            pythonExecutable: "/repo/.venv/bin/python",
            databasePath: "/store/music_agent.db")
        let unconfiguredCommand = ShellCommand.build(
            runtime: unconfiguredRuntime,
            port: 43128,
            inherited: [:])
        let unconfiguredArgv = [unconfiguredCommand.executablePath]
            + unconfiguredCommand.arguments
        require(unconfiguredArgv == [
            "/repo/.venv/bin/python",
            "-m", "music_agent.cli", "web",
            "--db", "/store/music_agent.db",
            "--no-browser",
            "--port", "43128",
        ], "missing identity must not append or manufacture a full client")

        let runtimeArguments = [
            "/repo/.venv/bin/python", "-m", "music_agent", "run",
            "--db", "/store/music_agent.db",
            "--refresh-interval", "900",
            "--agent-client", configuredIdentity,
        ]
        require(
            RuntimeLaunchConfiguration.configuredAgentClient(
                programArguments: runtimeArguments,
                databasePath: "/store/music_agent.db") == configuredIdentity,
            "matching Runtime argv must preserve its configured identity")
        require(
            RuntimeLaunchConfiguration.configuredAgentClient(
                programArguments: runtimeArguments,
                databasePath: "/other/music_agent.db") == nil,
            "a Runtime configured for another database must not grant identity")
        require(
            RuntimeLaunchConfiguration.configuredAgentClient(
                programArguments: runtimeArguments + [
                    "--agent-client",
                    "agt_66666666-6666-4666-8666-666666666666:read_only",
                ],
                databasePath: "/store/music_agent.db") == nil,
            "multiple Runtime identities must fail closed")

        let temporaryHome = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temporaryHome) }
        let launchAgents = temporaryHome
            .appendingPathComponent("Library/LaunchAgents")
        try FileManager.default.createDirectory(
            at: launchAgents, withIntermediateDirectories: true)
        let plist: [String: Any] = [
            "Label": RuntimeLaunchConfiguration.label,
            "ProgramArguments": runtimeArguments,
        ]
        let data = try PropertyListSerialization.data(
            fromPropertyList: plist, format: .xml, options: 0)
        try data.write(
            to: launchAgents.appendingPathComponent(
                "\(RuntimeLaunchConfiguration.label).plist"))
        require(
            RuntimeLaunchConfiguration.configuredAgentClient(
                home: temporaryHome.path,
                databasePath: "/store/music_agent.db") == configuredIdentity,
            "the matching installed Runtime plist must feed Web launch config")

        let repoRoot = temporaryHome.appendingPathComponent("repo")
        let packageDirectory = repoRoot.appendingPathComponent("src/music_agent")
        let virtualEnvironment = repoRoot.appendingPathComponent(".venv/bin")
        try FileManager.default.createDirectory(
            at: packageDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: virtualEnvironment, withIntermediateDirectories: true)
        _ = FileManager.default.createFile(
            atPath: repoRoot.appendingPathComponent("pyproject.toml").path,
            contents: Data())
        let python = virtualEnvironment.appendingPathComponent("python")
        _ = FileManager.default.createFile(atPath: python.path, contents: Data())
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: python.path)
        let database = temporaryHome.appendingPathComponent("music_agent.db")
        _ = FileManager.default.createFile(atPath: database.path, contents: Data())
        let integratedArguments = [
            python.path, "-m", "music_agent", "run",
            "--db", database.path,
            "--agent-client", configuredIdentity,
        ]
        let integratedPlist: [String: Any] = [
            "Label": RuntimeLaunchConfiguration.label,
            "ProgramArguments": integratedArguments,
        ]
        let integratedData = try PropertyListSerialization.data(
            fromPropertyList: integratedPlist, format: .xml, options: 0)
        try integratedData.write(
            to: launchAgents.appendingPathComponent(
                "\(RuntimeLaunchConfiguration.label).plist"))

        let resolvedRuntime: ShellRuntime
        switch ShellController.resolveRuntime(
            executableURL: nil,
            environment: [
                "MUSIC_AGENT_REPO_ROOT": repoRoot.path,
                "MUSIC_AGENT_DB": database.path,
            ],
            home: temporaryHome.path) {
        case .success(let runtime):
            resolvedRuntime = runtime
        case .failure(let failure):
            FileHandle.standardError.write(
                Data("FAIL: integrated runtime resolution: \(failure)\n".utf8))
            exit(1)
        }
        let resolvedCommand = ShellCommand.build(
            runtime: resolvedRuntime, port: 43129, inherited: [:])
        require(
            [resolvedCommand.executablePath] + resolvedCommand.arguments == [
                python.path,
                "-m", "music_agent.cli", "web",
                "--db", database.path,
                "--no-browser",
                "--port", "43129",
                "--agent-client", configuredIdentity,
            ],
            "resolved LaunchAgent identity must reach the final process argv")

        print("PASS: Web agent-client launch command contracts")
    }
}
