// Music Agent — WebShell process lifecycle (P20 Slice B).
//
// Owns exactly one child process: the existing Python WebShell
// (`python -m music_agent.cli web`), spawned without a Terminal, with
// `--no-browser`, against the real durable store. Responsibilities:
//
//   • resolve/validate repo root, <repo>/.venv/bin/python and the database
//     BEFORE spawning — missing any of the three lands in a loud error state
//     (we never create a database);
//   • pick a free 127.0.0.1 port and hand it to `--port`;
//   • poll the existing GET /healthz until HTTP 200 + {"ok": true}, observing
//     the child the whole time (early exit fails immediately, hard timeout
//     fails the run) — and only then record the ready URL;
//   • continuously drain stdout/stderr into a bounded tail (never letting a
//     full pipe stall Python);
//   • shut down its OWN child on request: POST /api/shutdown, then SIGTERM,
//     then SIGKILL — never anything else. The independent run authority (if
//     any) is not ours and is never signalled.
//
// Attach/embed semantics stay entirely inside web_shell.py: this layer only
// passes `--db <db> --no-browser --port <P>` plus the one agent-client entry
// already configured for the matching default Runtime LaunchAgent, and lets
// the shell probe <db>.agent.sock the way it already does (attach when served,
// else embed). Missing or ambiguous Runtime identity stays missing here; this
// launcher never manufactures or upgrades a client policy.
//
// Foundation-only on purpose: the whole file compiles stand-alone with
// swiftc, so the lifecycle logic can be exercised by a headless probe.
import Darwin
import Foundation

// MARK: - Resolved runtime inputs

struct ShellRuntime: Equatable {
    let repoRoot: String
    let pythonExecutable: String
    let databasePath: String
    let agentClientEntry: String?

    init(repoRoot: String, pythonExecutable: String, databasePath: String,
         agentClientEntry: String? = nil) {
        self.repoRoot = repoRoot
        self.pythonExecutable = pythonExecutable
        self.databasePath = databasePath
        self.agentClientEntry = agentClientEntry
    }
}

/// Read-only projection of the existing default Runtime LaunchAgent config.
/// The installed plist is already the durable source of the Runtime's argv;
/// this type only extracts one identity when that argv names the same database.
enum RuntimeLaunchConfiguration {
    static let label = "com.musicagent.runtime"

    static func configuredAgentClient(home: String,
                                      databasePath: String) -> String? {
        let plistPath = URL(fileURLWithPath: home)
            .appendingPathComponent("Library/LaunchAgents/\(label).plist")
        guard let data = try? Data(contentsOf: plistPath),
              let object = try? PropertyListSerialization.propertyList(
                  from: data, options: [], format: nil),
              let plist = object as? [String: Any],
              plist["Label"] as? String == label,
              let arguments = plist["ProgramArguments"] as? [String]
        else { return nil }
        return configuredAgentClient(
            programArguments: arguments,
            databasePath: databasePath)
    }

    static func configuredAgentClient(programArguments: [String],
                                      databasePath: String) -> String? {
        let databaseValues = values(after: "--db", in: programArguments)
        guard databaseValues.count == 1,
              (databaseValues[0] as NSString).standardizingPath
                == (databasePath as NSString).standardizingPath
        else { return nil }

        let clientValues = values(after: "--agent-client", in: programArguments)
        guard clientValues.count == 1, !clientValues[0].isEmpty else { return nil }
        return clientValues[0]
    }

    private static func values(after flag: String,
                               in arguments: [String]) -> [String] {
        arguments.indices.compactMap { index in
            guard arguments[index] == flag,
                  arguments.indices.contains(index + 1)
            else { return nil }
            return arguments[index + 1]
        }
    }
}

/// Outcome of path resolution/validation — surfaced verbatim in the window.
enum ShellFailure: Error, Equatable {
    case repoNotFound(checked: [String])
    case repoMarkerMissing(String)
    case pythonNotFound(String)
    case databaseNotFound(String)
    case credentialRequired
    case credentialStoreFailed(String)
    case noFreePort
    case spawnFailed(String)
    case childExitedBeforeReady(status: Int32)
    case childExitedAfterReady(status: Int32)
    case readinessTimeout(seconds: Int)

    /// Short human-readable reason for the dev status view (no formal UI).
    var reasonText: String {
        switch self {
        case .repoNotFound(let checked):
            let locations = checked.isEmpty ? "无可用候选位置" : checked.joined(separator: "；")
            return "Music Agent 开发运行环境未找到（已检查：\(locations)）"
        case .repoMarkerMissing(let path):
            return "仓库校验失败：\(path)"
        case .pythonNotFound(let path):
            return "找不到 Python：\(path)"
        case .databaseNotFound(let path):
            return "找不到数据库：\(path)（不自动创建）"
        case .credentialRequired:
            return "需要 DeepSeek API Key 才能启动服务。"
        case .credentialStoreFailed(let detail):
            return "无法读取或保存 DeepSeek 凭据：\(detail)"
        case .noFreePort:
            return "无法分配本地端口"
        case .spawnFailed(let detail):
            return "WebShell 启动失败：\(detail)"
        case .childExitedBeforeReady(let status):
            return "WebShell 在就绪前退出（状态码 \(status)）"
        case .childExitedAfterReady(let status):
            return "服务已退出（状态码 \(status)）"
        case .readinessTimeout(let seconds):
            return "等待服务就绪超时（\(seconds) 秒）"
        }
    }
}

// MARK: - Command construction (pure, probe-tested)

struct ShellCommand: Equatable {
    let executablePath: String
    let arguments: [String]
    let environment: [String: String]
    let workingDirectory: String
    let port: Int

    /// The real launch command: `.venv/bin/python -m music_agent.cli web
    /// --db <db> --no-browser --port <P> [--agent-client ID:POLICY]` under
    /// repo root with PYTHONPATH pointed at <repo>/src and unbuffered stdio.
    /// Inherits the app's own environment (DEEPSEEK_API_KEY rides along when
    /// the launcher had it; its absence is never a spawn blocker here — the
    /// shell decides).
    static func build(runtime: ShellRuntime, port: Int,
                      inherited: [String: String],
                      deepSeekAPIKey: String? = nil) -> ShellCommand {
        var env = inherited
        env["PYTHONPATH"] = (runtime.repoRoot as NSString).appendingPathComponent("src")
        env["PYTHONUNBUFFERED"] = "1"
        if let deepSeekAPIKey, !deepSeekAPIKey.isEmpty {
            env["DEEPSEEK_API_KEY"] = deepSeekAPIKey
        }
        var arguments = [
            "-m", "music_agent.cli", "web",
            "--db", runtime.databasePath,
            "--no-browser",
            "--port", String(port),
        ]
        if let agentClientEntry = runtime.agentClientEntry,
           !agentClientEntry.isEmpty {
            arguments.append(contentsOf: ["--agent-client", agentClientEntry])
        }
        return ShellCommand(
            executablePath: runtime.pythonExecutable,
            arguments: arguments,
            environment: env,
            workingDirectory: runtime.repoRoot,
            port: port
        )
    }
}

// MARK: - Free-port allocation

enum PortPicker {
    /// Bind 127.0.0.1:0, read the assigned port, release it. The child is the
    /// authority on the final bind — a race after release is retried upstream.
    static func pick() -> Int? {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        defer { close(fd) }

        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")

        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { return nil }

        var boundAddr = sockaddr_in()
        var boundLen = socklen_t(MemoryLayout<sockaddr_in>.size)
        let got = withUnsafeMutablePointer(to: &boundAddr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &boundLen)
            }
        }
        guard got == 0 else { return nil }
        let port = Int(UInt16(bigEndian: boundAddr.sin_port))
        return port == 0 ? nil : port
    }
}

// MARK: - Healthz parsing (pure, probe-tested)

enum Healthz {
    /// GET /healthz is ready only when HTTP 200 AND the JSON body says ok=true.
    static func parse(_ data: Data) -> Bool {
        guard let object = try? JSONSerialization.jsonObject(with: data),
              let body = object as? [String: Any] else { return false }
        return (body["ok"] as? Bool) == true
    }
}

// MARK: - Bounded log tail

final class LogTail {
    /// Ring of the most recent child output lines (bounded memory: the
    /// WebShell must never be able to balloon the app by logging forever).
    /// Pipe output arrives in arbitrary chunks, so an in-progress fragment is
    /// buffered and committed per newline — a trailing partial only appears
    /// in `snapshot`, never as a committed line.
    private let capacity: Int
    private var lines: [String] = []
    private var pending = ""
    private let lock = NSLock()

    init(capacity: Int = 200) {
        self.capacity = max(capacity, 1)
    }

    func append(_ data: Data) {
        append(String(decoding: data, as: UTF8.self))
    }

    func append(_ chunk: String) {
        lock.lock()
        defer { lock.unlock() }
        pending += chunk
        while let newline = pending.firstIndex(of: "\n") {
            var text = String(pending[..<newline])
            pending.removeSubrange(...newline)
            if text.hasSuffix("\r") { text.removeLast() }
            lines.append(text)
            if lines.count > capacity {
                lines.removeFirst(lines.count - capacity)
            }
        }
    }

    func snapshot() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return pending.isEmpty ? lines : lines + [pending]
    }
}

// MARK: - Lifecycle state

enum ShellState: Equatable {
    case idle
    case resolving
    case starting(port: Int, attempt: Int)
    case waitingReady(port: Int)
    case ready(url: URL, port: Int)
    case stopping
    case stopped
    case failed(ShellFailure)
}

struct ShellTiming: Equatable {
    var pollInterval: TimeInterval = 0.15
    var readinessTimeout: TimeInterval = 30
    var maxPortAttempts: Int = 3
    var postShutdownGrace: TimeInterval = 5
    var terminateGrace: TimeInterval = 3

    static let production = ShellTiming()
}

// MARK: - Controller

final class ShellController {
    // Injectable seams (probe construction): the child is spawned through
    // `spawn`, readiness polls happen through `healthz`, graceful shutdown
    // goes through `requestShutdown`. Production wiring below.
    typealias Spawn = (ShellCommand) throws -> ChildProcess
    typealias HealthzCheck = (_ url: URL, _ complete: @escaping (Bool) -> Void) -> Void
    typealias ShutdownRequest = (_ url: URL, _ complete: @escaping () -> Void) -> Void

    private let runtime: ShellRuntime
    private let deepSeekAPIKey: String?
    private let timing: ShellTiming
    private let spawn: Spawn
    private let healthz: HealthzCheck
    private let requestShutdown: ShutdownRequest

    let logTail = LogTail(capacity: 200)

    /// Delivered on an arbitrary background queue; the UI hops to main.
    var onStateChange: ((ShellState) -> Void)?

    private let queue = DispatchQueue(label: "musicagent.web-shell.lifecycle", qos: .userInitiated)
    private var state: ShellState = .idle { didSet { publishState() } }
    private var child: ChildProcess?
    private var childHasExited = false
    private var attemptNumber = 1
    private var readyURL: URL?
    private var readinessTimer: DispatchSourceTimer?

    init(runtime: ShellRuntime,
         deepSeekAPIKey: String? = nil,
         timing: ShellTiming = .production,
         spawn: @escaping Spawn = ShellController.systemSpawn,
         healthz: @escaping HealthzCheck = ShellController.systemHealthz,
         requestShutdown: @escaping ShutdownRequest = ShellController.systemShutdownRequest) {
        self.runtime = runtime
        self.deepSeekAPIKey = deepSeekAPIKey
        self.timing = timing
        self.spawn = spawn
        self.healthz = healthz
        self.requestShutdown = requestShutdown
    }

    var currentState: ShellState {
        queue.sync { state }
    }

    /// The ready shell URL (authoritative only after `.ready`).
    var shellURL: URL? {
        queue.sync { readyURL }
    }

    // MARK: Start

    func start() {
        queue.async { [weak self] in
            guard let self else { return }
            guard case .idle = self.state else { return }
            self.state = .resolving
            self.validateAndLaunch()
        }
    }

    private func validateAndLaunch() {
        let manager = FileManager.default

        // 1. Repo root: `src/music_agent/cli.py` is the marker.
        var isDirectory: ObjCBool = false
        guard manager.fileExists(atPath: runtime.repoRoot, isDirectory: &isDirectory), isDirectory.boolValue else {
            return setFailed(.repoNotFound(checked: [runtime.repoRoot]))
        }
        let marker = (runtime.repoRoot as NSString).appendingPathComponent("src/music_agent/cli.py")
        guard manager.fileExists(atPath: marker) else {
            return setFailed(.repoMarkerMissing(marker))
        }

        // 2. Python interpreter inside the repo venv.
        guard manager.isExecutableFile(atPath: runtime.pythonExecutable) else {
            return setFailed(.pythonNotFound(runtime.pythonExecutable))
        }

        // 3. Database must already exist — never auto-create.
        guard manager.fileExists(atPath: runtime.databasePath) else {
            return setFailed(.databaseNotFound(runtime.databasePath))
        }

        launchAttempt(1)
    }

    private func launchAttempt(_ attempt: Int) {
        guard attempt <= timing.maxPortAttempts else {
            return setFailed(.childExitedBeforeReady(status: -1))
        }
        guard let port = PortPicker.pick() else {
            return setFailed(.noFreePort)
        }

        let command = ShellCommand.build(runtime: runtime, port: port,
                                         inherited: ProcessInfo.processInfo.environment,
                                         deepSeekAPIKey: deepSeekAPIKey)
        let process: ChildProcess
        do {
            process = try spawn(command)
        } catch {
            return setFailed(.spawnFailed(String(describing: error)))
        }
        child = process
        childHasExited = false
        attemptNumber = attempt
        readyURL = nil

        process.onExit = { [weak self] status in
            self?.queue.async { self?.childDidExit(status: status) }
        }
        process.onOutput = { [weak self] data in
            self?.logTail.append(data)
        }

        state = .starting(port: port, attempt: attempt)
        beginReadiness(port: port)
    }

    private func beginReadiness(port: Int) {
        state = .waitingReady(port: port)
        let url = URL(string: "http://127.0.0.1:\(port)/healthz")!

        readinessTimer?.cancel()
        let timer = DispatchSource.makeTimerSource(queue: queue)
        let deadline = Date().addingTimeInterval(timing.readinessTimeout)
        timer.schedule(deadline: .now() + timing.pollInterval, repeating: timing.pollInterval, leeway: .milliseconds(20))
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            if Date() >= deadline {
                self.readinessTimer?.cancel()
                self.setFailed(.readinessTimeout(seconds: Int(self.timing.readinessTimeout)))
                return
            }
            self.healthz(url) { [weak self] isOk in
                guard let self, isOk else { return }
                self.queue.async { self.becomeReady(url: url) }
            }
        }
        readinessTimer = timer
        timer.resume()
    }

    private func becomeReady(url: URL) {
        guard case .waitingReady = state, !childHasExited else { return }
        readinessTimer?.cancel()
        // Record the shell root (the healthz URL we just verified, minus its
        // path) as the authoritative ready URL for display and shutdown.
        var shellURL = url
        if var components = URLComponents(url: url, resolvingAgainstBaseURL: false) {
            components.path = "/"
            if let root = components.url { shellURL = root }
        }
        readyURL = shellURL
        state = .ready(url: shellURL, port: url.port ?? 0)
    }

    /// Child exit observed anywhere in the lifecycle: before ready it fails
    /// the current attempt (bounded retry with a fresh port absorbs a port
    /// steal or a failed bind); after ready it surfaces as an error state.
    private func childDidExit(status: Int32) {
        readinessTimer?.cancel()
        childHasExited = true
        switch state {
        case .waitingReady, .starting:
            if attemptNumber < timing.maxPortAttempts {
                launchAttempt(attemptNumber + 1)
            } else {
                setFailed(.childExitedBeforeReady(status: status))
            }
        case .ready:
            setFailed(.childExitedAfterReady(status: status))
        case .stopping, .stopped:
            childHasExited = true
        default:
            break
        }
    }

    // MARK: Stop / shutdown

    /// App-quit path: graceful — POST /api/shutdown, wait, SIGTERM, wait,
    /// SIGKILL. Own child only. Synchronous; bounded by the timing values.
    func stopAndWait() {
        let semaphore = DispatchSemaphore(value: 0)
        stop { semaphore.signal() }
        let cap = timing.postShutdownGrace + timing.terminateGrace + 2
        _ = semaphore.wait(timeout: .now() + cap)
    }

    func stop(completion: @escaping () -> Void) {
        queue.async { [weak self] in
            guard let self else { return completion() }
            switch self.state {
            case .stopping, .stopped:
                return completion()
            default:
                break
            }
            // NOTE: .failed falls through on purpose — a readiness timeout
            // (or spawn error) can still leave a live child that must die.
            self.state = .stopping
            self.readinessTimer?.cancel()

            // Best-effort graceful door when we know the URL: the ready URL
            // is the shell root, so swap its path for the shutdown endpoint.
            if let ready = self.readyURL, !self.childHasExited,
               var components = URLComponents(url: ready, resolvingAgainstBaseURL: false) {
                components.path = "/api/shutdown"
                if let url = components.url {
                    let group = DispatchGroup()
                    group.enter()
                    self.requestShutdown(url) { group.leave() }
                    _ = group.wait(timeout: .now() + 2)
                }
            }

            self.stopChild { [weak self] in
                self?.queue.async {
                    self?.state = .stopped
                    completion()
                }
            }
        }
    }

    /// SIGTERM → (grace) → SIGKILL escalation for our own child only. Bounded
    /// strong captures: the chain is short and completion must always fire,
    /// even if the controller is released mid-quit.
    private func stopChild(completion: @escaping () -> Void) {
        guard let process = child else {
            childHasExited = true
            return completion()
        }
        let grace = timing.postShutdownGrace
        let terminateGrace = timing.terminateGrace
        process.waitForExit(timeout: grace) { exited in
            if exited { return completion() }
            process.terminate()
            process.waitForExit(timeout: terminateGrace) { exited in
                if !exited {
                    process.forceKill()
                }
                completion()
            }
        }
    }

    /// Last-resort force kill (used by stopAndWait timeout and the UI).
    func escalateKill() {
        queue.async { [weak self] in
            guard let self, !self.childHasExited else { return }
            self.child?.forceKill()
        }
    }

    /// Resolve the standard dev layout; validation happens in the controller.
    static func resolveRuntime() -> Result<ShellRuntime, ShellFailure> {
        resolveRuntime(executableURL: Bundle.main.executableURL,
                       environment: ProcessInfo.processInfo.environment,
                       home: FileManager.default.homeDirectoryForCurrentUser.path)
    }

    /// Development runtime resolution is independent of the launcher's
    /// current directory: an explicit override wins, then the standard
    /// checkout under ~/Documents, then bundle ancestry supports running
    /// the uninstalled app from the repository. The older override remains
    /// accepted for compatibility with existing local diagnostics.
    static func resolveRuntime(executableURL: URL?, environment: [String: String],
                               home: String) -> Result<ShellRuntime, ShellFailure> {
        let manager = FileManager.default
        var checked: [String] = []
        var seen = Set<String>()

        func validatedRepo(_ path: String?) -> String? {
            guard let path, !path.isEmpty else { return nil }
            let root = URL(fileURLWithPath: path).standardizedFileURL
            guard seen.insert(root.path).inserted else { return nil }
            checked.append(root.path)

            var isDirectory: ObjCBool = false
            guard manager.fileExists(atPath: root.path, isDirectory: &isDirectory),
                  isDirectory.boolValue,
                  manager.fileExists(atPath: root.appendingPathComponent("pyproject.toml").path),
                  manager.isExecutableFile(atPath: root.appendingPathComponent(".venv/bin/python").path)
            else { return nil }

            isDirectory = false
            let packageDirectory = root.appendingPathComponent("src/music_agent").path
            guard manager.fileExists(atPath: packageDirectory, isDirectory: &isDirectory),
                  isDirectory.boolValue else { return nil }
            return root.path
        }

        var repoRoot = validatedRepo(environment["MUSIC_AGENT_REPO_ROOT"])
        if repoRoot == nil {
            repoRoot = validatedRepo(environment["MUSIC_AGENT_REPO"])
        }
        if repoRoot == nil {
            repoRoot = validatedRepo(
                URL(fileURLWithPath: home)
                    .appendingPathComponent("Documents/music-agent-core")
                    .path)
        }
        if repoRoot == nil, let start = executableURL?.deletingLastPathComponent() {
            var cursor = start
            while true {
                if let resolved = validatedRepo(cursor.path) {
                    repoRoot = resolved
                    break
                }
                // NSString semantics: the filesystem root's parent is itself,
                // so this is a stable fixpoint — URL.deletingLastPathComponent
                // degrades to "/..", "/../..", … and would walk forever.
                let parentPath = (cursor.path as NSString).deletingLastPathComponent
                if parentPath == cursor.path { break }
                cursor = URL(fileURLWithPath: parentPath)
            }
        }
        guard let repoRoot else { return .failure(.repoNotFound(checked: checked)) }
        let db = environment["MUSIC_AGENT_DB"]
            ?? (home as NSString).appendingPathComponent("MusicAgent/music_agent.db")
        let agentClientEntry = RuntimeLaunchConfiguration.configuredAgentClient(
            home: home, databasePath: db)
        return .success(ShellRuntime(
            repoRoot: repoRoot,
            pythonExecutable: (repoRoot as NSString).appendingPathComponent(".venv/bin/python"),
            databasePath: db,
            agentClientEntry: agentClientEntry
        ))
    }

    // MARK: - Production seams

    static func systemSpawn(_ command: ShellCommand) throws -> ChildProcess {
        let process = ChildProcess(command: command)
        try process.launch()
        return process
    }

    /// URLSession poll of GET /healthz: HTTP 200 + parsed ok=true only.
    static func systemHealthz(_ url: URL, _ complete: @escaping (Bool) -> Void) {
        let request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData,
                                 timeoutInterval: 2)
        URLSession.shared.dataTask(with: request) { data, response, _ in
            guard let status = (response as? HTTPURLResponse)?.statusCode, status == 200,
                  let data else { return complete(false) }
            complete(Healthz.parse(data))
        }.resume()
    }

    static func systemShutdownRequest(_ url: URL, _ complete: @escaping () -> Void) {
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData,
                                 timeoutInterval: 3)
        request.httpMethod = "POST"
        URLSession.shared.dataTask(with: request) { _, _, _ in complete() }.resume()
    }

    // MARK: - State plumbing

    private func setFailed(_ failure: ShellFailure) {
        if case .failed = state {
            readinessTimer?.cancel()
            return
        }
        readinessTimer?.cancel()
        state = .failed(failure)
    }

    private func publishState() {
        let snapshot = state
        onStateChange?(snapshot)
    }
}

// MARK: - Child process wrapper

final class ChildProcess {
    private let process: Process
    private let lock = NSLock()
    private var launched = false
    private var exited = false

    var onExit: ((Int32) -> Void)?
    var onOutput: ((Data) -> Void)?

    init(command: ShellCommand) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: command.executablePath)
        process.arguments = command.arguments
        process.environment = command.environment
        process.currentDirectoryURL = URL(fileURLWithPath: command.workingDirectory)
        self.process = process
    }

    func launch() throws {
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        for pipe in [stdoutPipe, stderrPipe] {
            pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                let data = handle.availableData
                if data.isEmpty { return }
                self?.onOutput?(data)
            }
        }

        process.terminationHandler = { [weak self] process in
            if let self {
                self.lock.lock()
                self.exited = true
                self.lock.unlock()
            }
            self?.onExit?(process.terminationStatus)
        }

        try process.run()
        lock.lock()
        launched = true
        lock.unlock()
    }

    var hasExited: Bool {
        lock.lock()
        defer { lock.unlock() }
        return exited
    }

    private var wasLaunched: Bool {
        lock.lock()
        defer { lock.unlock() }
        return launched
    }

    /// Block until exit or `timeout` seconds; reports whether it exited.
    func waitForExit(timeout: TimeInterval, completion: @escaping (Bool) -> Void) {
        if !wasLaunched {
            return completion(true)
        }
        DispatchQueue.global(qos: .background).async { [weak self] in
            guard let self else { return completion(true) }
            let deadline = Date().addingTimeInterval(timeout)
            while !self.hasExited && Date() < deadline {
                Thread.sleep(forTimeInterval: 0.05)
            }
            completion(self.hasExited)
        }
    }

    func terminate() {
        lock.lock()
        defer { lock.unlock() }
        guard launched, !exited else { return }
        process.terminate()
    }

    func forceKill() {
        lock.lock()
        defer { lock.unlock() }
        guard launched, !exited else { return }
        kill(process.processIdentifier, SIGKILL)
    }
}
