// Music Agent — application delegate (P20 Slice A/B/C).
//
// Lifecycle contract: one window at launch, brought frontmost; closing the
// last window terminates the app for real (no window-less process lingering
// in the Dock). Slice B adds the WebShell: resolved and started in the
// background right after the window appears, and shut down for real before
// the app leaves the process — the app never leaks an owned WebShell child.
// Slice C: the ready state hands the shell root URL to the window (existing
// Web UI), and a graceful post-ready exit 0 of our own child (the page's
// 退出应用 button) ends the app through the very same terminate lifecycle.
import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var windowController: MainWindowController?
    private var shell: ShellController?
    private let credentialStore = KeychainCredentialStore()

    func applicationDidFinishLaunching(_ notification: Notification) {
        let controller = MainWindowController()
        controller.showWindow(nil)
        windowController = controller
        NSApp.activate(ignoringOtherApps: true)

        // Resolve synchronously (cheap filesystem checks): any missing
        // input must fail loudly into the window, never silently.
        switch ShellController.resolveRuntime() {
        case .failure(let failure):
            controller.updateStatus(.failed(failure))
        case .success(let runtime):
            resolveCredentialAndStart(runtime: runtime, controller: controller)
        }
    }

    private func resolveCredentialAndStart(runtime: ShellRuntime,
                                           controller: MainWindowController) {
        do {
            switch try DeepSeekCredentialResolver.resolve(
                environment: ProcessInfo.processInfo.environment,
                store: credentialStore) {
            case .available(let credential):
                startShell(runtime: runtime, credential: credential.value,
                           controller: controller)
            case .missing:
                guard let credential = promptForDeepSeekCredential(
                    title: "设置 DeepSeek API Key",
                    message: "凭据将安全保存到 macOS Keychain，仅用于 Music Agent 的 WebShell。")
                else {
                    controller.updateStatus(.failed(.credentialRequired))
                    return
                }
                try credentialStore.save(credential)
                startShell(runtime: runtime, credential: credential,
                           controller: controller)
            }
        } catch {
            controller.updateStatus(.failed(.credentialStoreFailed(error.localizedDescription)))
        }
    }

    private func startShell(runtime: ShellRuntime, credential: String,
                            controller: MainWindowController) {
        let shell = ShellController(runtime: runtime, deepSeekAPIKey: credential)
        shell.onStateChange = { [weak controller] state in
            DispatchQueue.main.async {
                controller?.updateStatus(state)
                // The web UI carries its own 退出应用 button (POST
                // /api/shutdown): the WebShell then exits 0 and the
                // page's server is gone. Follow the standard terminate
                // lifecycle instead of leaving a dead window —
                // applicationShouldTerminate re-verifies the (already
                // dead) child in microseconds, and status 0 after ready
                // is the clean-shutdown signature; any non-zero exit
                // stays an on-screen error, never an app kill.
                if case .failed(.childExitedAfterReady(status: 0)) = state {
                    NSApp.terminate(nil)
                }
            }
        }
        self.shell = shell
        shell.start()
    }

    private func promptForDeepSeekCredential(title: String,
                                             message: String) -> String? {
        var validationMessage: String?
        while true {
            let alert = NSAlert()
            alert.messageText = title
            alert.informativeText = validationMessage ?? message
            alert.alertStyle = validationMessage == nil ? .informational : .warning
            alert.addButton(withTitle: "保存")
            alert.addButton(withTitle: "取消")

            let field = NSSecureTextField(frame: NSRect(x: 0, y: 0, width: 360, height: 24))
            field.placeholderString = "DeepSeek API Key"
            alert.accessoryView = field
            alert.window.initialFirstResponder = field
            alert.window.makeFirstResponder(field)

            guard alert.runModal() == .alertFirstButtonReturn else { return nil }
            do {
                return try DeepSeekCredentialInput.normalize(field.stringValue)
            } catch {
                validationMessage = error.localizedDescription
            }
        }
    }

    @objc func updateDeepSeekAPIKey(_ sender: Any?) {
        guard let credential = promptForDeepSeekCredential(
            title: "更新 DeepSeek API Key",
            message: "输入新凭据。保存后请重新启动 Music Agent 以应用。")
        else { return }

        do {
            try credentialStore.save(credential)
            let alert = NSAlert()
            alert.messageText = "DeepSeek API Key 已保存"
            if let environmentValue = ProcessInfo.processInfo.environment["DEEPSEEK_API_KEY"],
               !environmentValue.isEmpty {
                alert.informativeText = "当前进程仍优先使用环境变量。重新启动后也会继续遵循环境变量优先级。"
            } else {
                alert.informativeText = "请重新启动 Music Agent 以使用新凭据。"
            }
            alert.addButton(withTitle: "好")
            alert.runModal()
        } catch {
            let alert = NSAlert(error: error)
            alert.messageText = "无法保存 DeepSeek API Key"
            alert.runModal()
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    /// View-menu entry (P20 C.2): the only explicit Main ⇄ Mini control.
    /// The v1 UI deliberately hides the topbar, so there is no web button
    /// to reuse; a menu item + ⇧⌘M is the minimal entry. The item label
    /// follows the resulting form so the user always reads the NEXT action.
    @objc func toggleWindowForm(_ sender: Any?) {
        guard let controller = windowController else { return }
        controller.toggleWindowForm()
        if let item = sender as? NSMenuItem {
            item.title = controller.form == .mini ? "Main Player" : "Mini Player"
        }
    }

    // Owned-WebShell shutdown gate: POST /api/shutdown → SIGTERM → SIGKILL,
    // bounded by the controller's timing budget, before the app actually
    // leaves. The independent run authority is never touched from here.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        shell?.stopAndWait()
        return .terminateNow
    }
}
