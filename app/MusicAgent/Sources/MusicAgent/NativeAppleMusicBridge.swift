import AppKit
import WebKit

/// Web -> Native boundary for the one external-app action the native shell owns.
///
/// The WebShell/backend remains authoritative for track identity and returns a validated
/// song-level Apple Music URL through a side-effect-free native resolve endpoint. CLI/MCP
/// and non-native surfaces keep their existing tool execution path; inside the native app,
/// this bridge is the sole external-app handoff authority. It re-validates the narrow URL
/// shape, asks LaunchServices from the
/// foreground native process to open it specifically with Music.app, and reports that
/// completion back to the page. Python command acceptance is never treated as native UI
/// success.
final class NativeAppleMusicBridge: NSObject, WKScriptMessageHandler {
    static let handlerName = "appleMusicOpen"

    weak var webView: WKWebView?

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage
    ) {
        guard message.name == Self.handlerName, message.frameInfo.isMainFrame else {
            return
        }
        guard let body = message.body as? [String: Any],
              let requestID = body["request_id"] as? String,
              !requestID.isEmpty,
              let rawURL = body["client_url"] as? String,
              let clientURL = Self.validatedSongURL(rawURL) else {
            reply(requestID: (message.body as? [String: Any])?["request_id"] as? String ?? "",
                  ok: false,
                  message: "Apple Music 打开请求无效。")
            return
        }

        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true
        configuration.requiresUniversalLinks = true
        NSWorkspace.shared.open(
            clientURL,
            configuration: configuration
        ) { [weak self] application, error in
            if let error {
                self?.reply(
                    requestID: requestID,
                    ok: false,
                    message: "Music.app 打开失败：\(error.localizedDescription)"
                )
                return
            }
            guard application?.bundleIdentifier == "com.apple.Music" else {
                self?.reply(
                    requestID: requestID,
                    ok: false,
                    message: "Apple Music Universal Link 未由 Music.app 接管。"
                )
                return
            }
            self?.reply(requestID: requestID, ok: true, message: nil)
        }
    }

    /// Native side fail-closed check. Python owns identity derivation; Swift only accepts
    /// the exact song-level URL shape that the backend is allowed to hand across.
    private static func validatedSongURL(_ rawValue: String) -> URL? {
        guard let components = URLComponents(string: rawValue),
              components.scheme?.lowercased() == "https",
              components.host?.lowercased() == "music.apple.com",
              components.query == nil,
              components.fragment == nil else {
            return nil
        }
        let parts = components.path.split(separator: "/")
        guard parts.count == 3,
              parts[0].count == 2,
              parts[0].allSatisfy({ $0.isLetter }),
              parts[1] == "song",
              !parts[2].isEmpty,
              parts[2].allSatisfy({ $0.isNumber }) else {
            return nil
        }
        return components.url
    }

    private func reply(requestID: String, ok: Bool, message: String?) {
        guard !requestID.isEmpty else { return }
        var payload: [String: Any] = [
            "request_id": requestID,
            "ok": ok,
        ]
        if let message {
            payload["message"] = message
        }
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let json = String(data: data, encoding: .utf8) else {
            return
        }
        DispatchQueue.main.async { [weak self] in
            self?.webView?.evaluateJavaScript(
                "window.__musicAgentNativeAppleMusicResult && "
                    + "window.__musicAgentNativeAppleMusicResult(\(json));"
            )
        }
    }
}
