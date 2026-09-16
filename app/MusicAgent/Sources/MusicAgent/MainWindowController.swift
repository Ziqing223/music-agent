// Music Agent — native Main / Mini window forms (P20 Slice C recovery).
//
// AppKit is the only window-form/frame authority; WebKit owns layout inside
// that final form. Main is a resizable 660x880 fixed canvas (80%...100%).
// Mini is one fixed 420x66 player pill. Switching uses a short content fade
// around a direct final-frame change — NSWindow.frame is never continuously
// morphed through aspect ratios that match neither product form.
//
// The sole Native → Web contract is data-native-form="main" | "mini".
// Full Screen remains disabled because the former exit-fullscreen frame
// restoration path crashed in AppKit.
import AppKit
import WebKit

/// The two explicit product forms. The page never infers a native form from
/// viewport width.
enum WindowForm {
    case main
    case mini

    static let mainSize = NSSize(width: 660, height: 880)
    static let mainMinSize = NSSize(width: 528, height: 704)
    static let miniSize = NSSize(width: 420, height: 66)

    var webValue: String { self == .main ? "main" : "mini" }

    var nativeStyleMask: NSWindow.StyleMask {
        switch self {
        case .main:
            return [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView]
        case .mini:
            return [.borderless]
        }
    }
}

/// Main's opaque product underlay. Mini keeps this tone inside the Web
/// document itself; its Native carrier is transparent and draws no second
/// rounded surface around the player.
enum NativeChrome {
    static let underlayColor = NSColor(srgbRed: 0x0B / 255.0,
                                       green: 0x0C / 255.0,
                                       blue: 0x1A / 255.0,
                                       alpha: 1.0)
    static let miniCornerRadius: CGFloat = 26
}

/// A transparent native drag surface above the WKWebView. Main uses a
/// fixed top band (the card's top strip is empty by design); Mini uses a
/// full-window sieve whose hitTest hands the transport-controls rect to the
/// WKWebView below — only the title/artist area and the pill's own empty
/// strip drag the window (there is no halo left to drag; P20 round 4), and
/// interactive web content keeps every click. Window dragging via
/// performDrag never swallows clicks outside these regions.
final class DragRegionView: NSView {
    /// Rects computed for the given bounds size (this view's coordinates)
    /// fall through to the views BELOW — interactive web content lives
    /// there. nil = the whole view is one drag region.
    var fallthroughRectProvider: ((NSSize) -> NSRect?)?

    override func hitTest(_ point: NSPoint) -> NSView? {
        if let rect = fallthroughRectProvider?(bounds.size), rect.contains(point) {
            return nil
        }
        return super.hitTest(point)
    }

    override func mouseDown(with event: NSEvent) {
        window?.performDrag(with: event)
    }
}

/// Mini is borderless so AppKit contributes no second rounded frame, but it
/// remains a normal key/main product window: Web controls, menu shortcuts,
/// and the existing native drag regions keep the same activation semantics.
final class ProductWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

final class MainWindowController: NSWindowController, WKNavigationDelegate {
    private let statusField = NSTextField(wrappingLabelWithString: "启动中…")
    private let appleMusicBridge = NativeAppleMusicBridge()

    /// The live Web UI surface — nil until `.ready`. Internal so the
    /// headless probe can assert the gating (created/loaded only on ready).
    private(set) var webView: WKWebView?

    /// Current product form. Internal so the headless probe asserts the
    /// toggle contract.
    private(set) var form: WindowForm = .main

    /// Main frame size last chosen by the user, restored after Mini.
    private var mainContentSize = WindowForm.mainSize
    private var isTransitioning = false

    /// Native drag regions (probe-visible).
    private(set) var mainDragBand: DragRegionView?
    private(set) var miniDragRegion: DragRegionView?

    private let fadeOutDuration: TimeInterval = 0.06
    private let fadeInDuration: TimeInterval = 0.10

    init() {
        let window = ProductWindow(
            contentRect: NSRect(origin: .zero, size: WindowForm.mainSize),
            styleMask: WindowForm.main.nativeStyleMask,
            backing: .buffered,
            defer: false
        )
        // Native chrome (P20 C.1): the Web UI's rounded canvas IS the app
        // window — no white bar, no centered title. Content extends under
        // the titlebar area and the real system traffic lights float over
        // the card's own empty top-left strip. Standard NSWindow lifecycle
        // and standard window buttons throughout (nothing re-drawn).
        window.title = "Music Agent"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.titlebarSeparatorStyle = .none
        // Fixed 3:4 product geometry (P20 C.1, consolidated): exactly two
        // window forms exist — the 660x880 Main canvas and the compact
        // Mini pill. Main stays resizable, but every edge/corner drag is
        // re-proportioned by AppKit AND clamped to the 528x704..660x880
        // scale band, so the canvas — the card IS the window — only ever
        // scales as one unit between S=0.8 and S=1.0. Mini is fixed-size.
        window.contentAspectRatio = WindowForm.mainSize
        window.minSize = WindowForm.mainMinSize
        window.maxSize = WindowForm.mainSize
        // Never expose the default system surface underneath WebKit.
        window.backgroundColor = NativeChrome.underlayColor
        // P20 C.2 fullscreen exclusion: macOS Full Screen is NOT a V1
        // window form (only Main / Mini exist). The default green-button
        // behavior is fullscreen participation, and the fullscreen exit
        // frame-restore path is what crashed
        // (_adjustNeedsDisplayRegionForNewFrame) against this window's
        // chrome + aspect geometry. .fullScreenNone removes the ability
        // entirely — green reverts to a plain zoom, which AppKit keeps
        // inside the aspect/clamp contract — and .managed keeps the usual
        // Spaces/Exposé participation. No fullscreen state machine.
        window.collectionBehavior = [.managed, .fullScreenNone]
        window.center()
        super.init(window: window)
        buildContent()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) is not supported")
    }

    private func buildContent() {
        guard let window, let contentView = window.contentView else { return }
        statusField.frame = contentView.bounds.insetBy(dx: 32, dy: 32)
        statusField.autoresizingMask = [.width, .height]
        statusField.alignment = .center
        statusField.font = .systemFont(ofSize: 15)
        statusField.textColor = .secondaryLabelColor
        contentView.addSubview(statusField)
        // The WKWebView is deliberately NOT built here: its creation and
        // load are gated on the ready state (Slice C), so a startup failure
        // can never surface as a blank web view.
    }

    // MARK: window forms (P20 C.2 consolidated, round 3)

    /// Main ⇄ Mini uses a short content fade around one direct frame switch.
    /// Input during the hidden swap is ignored instead of creating a second
    /// geometry queue.
    func toggleWindowForm() {
        guard !isTransitioning, let window else { return }
        let target: WindowForm = form == .main ? .mini : .main
        if target == .mini {
            mainContentSize = window.frame.size
        }
        form = target
        isTransitioning = true

        guard let webView else {
            applyWindowAndWebForm(target) { [weak self] in
                self?.isTransitioning = false
            }
            return
        }

        NSAnimationContext.runAnimationGroup({ context in
            context.duration = fadeOutDuration
            context.timingFunction = CAMediaTimingFunction(name: .easeOut)
            webView.animator().alphaValue = 0
        }, completionHandler: { [weak self] in
            self?.applyWindowAndWebForm(target) {
                self?.revealWebView()
            }
        })
    }

    /// Switch directly between final frames while WebKit is hidden. There
    /// is no animated intermediate aspect ratio for either form to occupy.
    private func applyWindowAndWebForm(_ target: WindowForm,
                                       completion: @escaping () -> Void) {
        guard let window else {
            completion()
            return
        }
        let sourceRect = window.frame

        if target == .mini {
            // Compact form (P20 round 4): fixed at the pill's border box —
            // zero halo. User resizing is removed for the duration (growing
            // the window could only re-create the empty band this form
            // dropped) and zoom has no meaning under a fixed size (button
            // disabled). The traffic lights belong to the Main chrome and
            // are hidden: the pill fills the whole window and no top strip
            // is reserved for them — they return, unchanged, on the way
            // back.
            window.contentAspectRatio = .zero
            window.minSize = WindowForm.miniSize
            window.maxSize = WindowForm.miniSize
            window.standardWindowButton(.zoomButton)?.isEnabled = false
            setTrafficLights(hidden: true)
            window.styleMask = target.nativeStyleMask
        } else {
            window.styleMask = target.nativeStyleMask
            window.minSize = WindowForm.mainMinSize
            window.maxSize = WindowForm.mainSize
            window.contentAspectRatio = WindowForm.mainSize
            window.standardWindowButton(.zoomButton)?.isEnabled = true
            setTrafficLights(hidden: false)
        }

        // Mini has exactly one rounded visual surface: the Web player. The
        // Mini's borderless NSWindow frame/content views and WKWebView are
        // clear, unmasked carriers with no border or shadow of their own.
        // Main restores the original titled/full-size-content frame inside
        // this same hidden atomic form swap; WebView lifecycle is unchanged.
        // The Web root/ambient material remains unchanged and supplies the player's
        // backdrop input, so the desktop never participates in compositing.
        // Main restores its original opaque Native/Web underlay.
        let isMini = target == .mini
        let nativeBackingColor = isMini ? NSColor.clear : NativeChrome.underlayColor
        window.isOpaque = target != .mini
        window.backgroundColor = nativeBackingColor
        webView?.underPageBackgroundColor = nativeBackingColor
        for view in [window.contentView?.superview, window.contentView, webView] {
            view?.wantsLayer = true
            view?.layer?.backgroundColor = nativeBackingColor.cgColor
            view?.layer?.isOpaque = !isMini
            view?.layer?.cornerRadius = 0
            view?.layer?.cornerCurve = .continuous
            view?.layer?.masksToBounds = false
        }
        applyWebViewOutputClip(for: target)
        window.hasShadow = target != .mini
        window.invalidateShadow()

        let targetSize = target == .mini ? WindowForm.miniSize : mainContentSize
        let targetRect = NSRect(
            x: sourceRect.minX,
            y: sourceRect.maxY - targetSize.height,
            width: targetSize.width,
            height: targetSize.height)

        window.setFrame(targetRect, display: true, animate: false)
        // Auto Layout must publish the target WKWebView viewport BEFORE the
        // web form changes. Otherwise Main can briefly compute against the
        // old 420x66 Mini viewport (or vice versa) and reveal a clipped
        // target projection.
        window.contentView?.layoutSubtreeIfNeeded()
        webView?.layoutSubtreeIfNeeded()
        mainDragBand?.isHidden = target != .main
        miniDragRegion?.isHidden = target != .mini

        applyFormToWeb(target) {
            // One display turn lets WebKit commit the forced target layout
            // before the hidden view starts fading back in.
            DispatchQueue.main.async {
                window.contentView?.layoutSubtreeIfNeeded()
                completion()
            }
        }
    }

    private func revealWebView() {
        guard let webView else {
            isTransitioning = false
            return
        }
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = fadeInDuration
            context.timingFunction = CAMediaTimingFunction(name: .easeIn)
            webView.animator().alphaValue = 1
        }, completionHandler: { [weak self] in
            if let self {
                // WebKit commits a new compositing tree during the form
                // change. Re-assert the clip once at the existing
                // transition completion boundary; no polling is needed.
                self.applyWebViewOutputClip(for: self.form)
            }
            self?.isTransitioning = false
        })
    }

    /// Mini clips the complete WebKit composited output to the player
    /// silhouette. This layer is a clip only: it has no fill, stroke or
    /// shadow. Main removes the clip and restores ordinary rectangular
    /// WebView output.
    private func applyWebViewOutputClip(for target: WindowForm) {
        guard let webView, let layer = webView.layer else { return }
        let isMini = target == .mini
        layer.mask = nil
        layer.cornerRadius = isMini ? NativeChrome.miniCornerRadius : 0
        layer.cornerCurve = .continuous
        layer.masksToBounds = isMini
        if isMini {
            webView.underPageBackgroundColor = .clear
            layer.backgroundColor = NSColor.clear.cgColor
            layer.isOpaque = false
            layer.borderWidth = 0
            layer.shadowOpacity = 0
        }
    }

    /// Mini hides the system traffic lights, Main shows them (P20 round 4):
    /// the compact pill fills the whole window and must not grow a strip
    /// to host them. They stay the same real AppKit standard buttons — the
    /// flag only withdraws them from the frame, nothing is re-drawn and
    /// nothing about Main changes. Idempotent, so repeated toggles cannot
    /// drift the visibility state.
    private func setTrafficLights(hidden: Bool) {
        window?.standardWindowButton(.closeButton)?.isHidden = hidden
        window?.standardWindowButton(.miniaturizeButton)?.isHidden = hidden
        window?.standardWindowButton(.zoomButton)?.isHidden = hidden
    }

    /// Native → Web form contract: one explicit attribute. Reading layout
    /// in the same script forces the target projection before completion;
    /// no Web measurement feeds back into native window sizing.
    private func applyFormToWeb(_ target: WindowForm? = nil,
                                completion: (() -> Void)? = nil) {
        let value = (target ?? form).webValue
        guard let webView else {
            completion?()
            return
        }
        webView.evaluateJavaScript("""
            document.documentElement.dataset.nativeForm = '\(value)';
            (function () {
              var stage = document.querySelector('.viewport-stage');
              var shell = document.querySelector('.shell');
              return [stage ? stage.getBoundingClientRect().width : 0,
                      shell ? shell.getBoundingClientRect().height : 0];
            })();
            """) { _, _ in
            DispatchQueue.main.async { completion?() }
        }
    }

    /// A late load (never normal in this shell) re-asserts the CURRENT form
    /// over the boot user script's 'main' default.
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        applyFormToWeb()
    }

    /// Dev status text for the current lifecycle state; on `.ready` the
    /// status overlay steps aside and the existing Web UI appears.
    func updateStatus(_ state: ShellState) {
        statusField.stringValue = statusText(state)
        switch state {
        case .ready(let url, _):
            showWebView(for: url)
        default:
            statusField.isHidden = false
        }
    }

    private func showWebView(for url: URL) {
        guard let window, let contentView = window.contentView else { return }
        if webView == nil {
            let config = WKWebViewConfiguration()
            // Boot-time form truth, BEFORE first paint: the native shell
            // owns the form ('main' at launch), so a reload can never
            // flash the browser-mode letterbox. didFinish then re-asserts
            // the actual current form.
            let formScript = WKUserScript(
                source: "document.documentElement.dataset.nativeForm = '\(form.webValue)';",
                injectionTime: .atDocumentStart, forMainFrameOnly: true)
            config.userContentController.addUserScript(formScript)
            // P21 native handoff: Web owns the user gesture and backend identity
            // resolution; this one message channel hands only the validated
            // song-level client URL to the foreground AppKit process.
            config.userContentController.add(
                appleMusicBridge, name: NativeAppleMusicBridge.handlerName)
            let webView = WKWebView(frame: .zero, configuration: config)
            appleMusicBridge.webView = webView
            // Match the window and page root while WebKit repaints a live
            // Main resize or the hidden direct form switch.
            webView.underPageBackgroundColor = NativeChrome.underlayColor
            webView.translatesAutoresizingMaskIntoConstraints = false
            webView.navigationDelegate = self
            webView.wantsLayer = true
            webView.layerContentsRedrawPolicy = .duringViewResize
            contentView.addSubview(webView, positioned: .below, relativeTo: statusField)
            NSLayoutConstraint.activate([
                webView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
                webView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
                webView.topAnchor.constraint(equalTo: contentView.topAnchor),
                webView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
            ])
            window.makeFirstResponder(webView)
            self.webView = webView
            installDragRegions(in: contentView, above: webView)
        }
        // Native App Mode is an explicit query flag, never user-agent
        // sniffing, and the load target keeps the ready URL's origin but
        // uses the concrete /index.html document path: the WebShell's GET
        // routing matches path == "/" or path.startswith("/index.html") on
        // the RAW path INCLUDING the query (?native=1), so the bare root
        // plus query would fall through to the unknown-endpoint branch.
        // Only path + query change — scheme, host and port come straight
        // from the ready URL handed over by ShellController (same origin,
        // no hardcoded port, no stdout parsing).
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        components?.path = "/index.html"
        components?.queryItems = [URLQueryItem(name: "native", value: "1")]
        webView?.load(URLRequest(url: components?.url ?? url))
        statusField.isHidden = true
    }

    private func installDragRegions(in contentView: NSView, above webView: WKWebView) {
        // Main: a 52px full-width band across the top. The card's top strip
        // is design-empty — no content starts above Y≈8% of the canvas
        // (≥56px even at the smallest S=0.8) — so the band drags the window
        // without ever eating a click. The real traffic lights live in the
        // NSThemeFrame layer above the content view, so they stay fully
        // clickable through the band.
        let band = DragRegionView(frame: NSRect(
            x: 0,
            y: contentView.bounds.maxY - 52,
            width: contentView.bounds.width,
            height: 52))
        band.autoresizingMask = [.width, .minYMargin]
        contentView.addSubview(band, positioned: .above, relativeTo: webView)
        mainDragBand = band
        // Mini: title/artist space drags; the fixed right-side controls
        // rect falls through to WebKit so prev/play-pause/next stay live.
        let sieve = DragRegionView(frame: contentView.bounds)
        sieve.autoresizingMask = [.width, .height]
        sieve.isHidden = true
        sieve.fallthroughRectProvider = { boundsSize in
            // CSS pins three 36px buttons, two 8px gaps and a 10px right
            // inset. The 140px region includes a small click cushion.
            let width = min(140, boundsSize.width)
            return NSRect(x: boundsSize.width - width, y: 7,
                          width: width, height: max(0, boundsSize.height - 14))
        }
        contentView.addSubview(sieve, positioned: .above, relativeTo: webView)
        miniDragRegion = sieve
    }

    private func statusText(_ state: ShellState) -> String {
        switch state {
        case .idle, .resolving:
            return "启动中…"
        case .starting(let port, _), .waitingReady(let port):
            return "启动中…\n127.0.0.1:\(port)"
        case .ready(let url, _):
            return "服务已就绪\n\(url.absoluteString)"
        case .stopping:
            return "正在退出…"
        case .stopped:
            return "已退出"
        case .failed(let failure):
            return "服务启动失败：\(failure.reasonText)"
        }
    }
}
