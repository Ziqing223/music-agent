// Music Agent — application entry point (P20 Slice A).
//
// A plain NSApplication.run() under activation policy .regular: the app gets
// a Dock presence and a frontmost window when launched via `open`, and never
// touches a Terminal. The delegate owns the single window.
import AppKit

let app = NSApplication.shared
let appDelegate = AppDelegate()
app.setActivationPolicy(.regular)
app.delegate = appDelegate

// Minimal standard Application menu: a programmatically created NSApplication
// has NO main menu, so ⌘Q would have no key equivalent and the app could not
// be quit from the keyboard. The item targets the standard
// NSApplication.terminate(_:), which flows through the delegate's termination
// lifecycle (ShellController.stopAndWait() included) — never a raw exit().
let mainMenu = NSMenu()
let appMenuItem = NSMenuItem()
mainMenu.addItem(appMenuItem)
let appMenu = NSMenu()
appMenuItem.submenu = appMenu
let credentialItem = NSMenuItem(title: "DeepSeek API Key…",
                                action: #selector(AppDelegate.updateDeepSeekAPIKey(_:)),
                                keyEquivalent: "")
credentialItem.target = appDelegate
appMenu.addItem(credentialItem)
appMenu.addItem(.separator())
appMenu.addItem(NSMenuItem(title: "Quit Music Agent",
                           action: #selector(NSApplication.terminate(_:)),
                           keyEquivalent: "q"))

// Standard responder-chain editing commands. The API-key prompts use an
// NSSecureTextField, so clipboard shortcuts must flow through AppKit rather
// than reading or handling the pasteboard directly.
let editMenuItem = NSMenuItem(title: "Edit", action: nil, keyEquivalent: "")
mainMenu.addItem(editMenuItem)
let editMenu = NSMenu(title: "Edit")
editMenuItem.submenu = editMenu

func addResponderItem(title: String, action: Selector, keyEquivalent: String) {
    let item = NSMenuItem(title: title, action: action, keyEquivalent: keyEquivalent)
    item.target = nil
    editMenu.addItem(item)
}

addResponderItem(title: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
addResponderItem(title: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
addResponderItem(title: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
addResponderItem(title: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")

// P20 C.2: Mini Player is the second product window form, reached from a
// View menu item rather than by shrinking the window — the v1 UI has no
// web button for it (the topbar is deliberately hidden). ⇧⌘M (not ⌘M,
// which is the system Minimize command). The action lives on the app
// delegate, which forwards to the single window controller.
let viewMenuItem = NSMenuItem(title: "View", action: nil, keyEquivalent: "")
mainMenu.addItem(viewMenuItem)
let viewMenu = NSMenu(title: "View")
viewMenuItem.submenu = viewMenu
let miniToggle = NSMenuItem(title: "Mini Player",
                            action: #selector(AppDelegate.toggleWindowForm(_:)),
                            keyEquivalent: "M")
miniToggle.keyEquivalentModifierMask = [.command, .shift]
miniToggle.target = appDelegate
viewMenu.addItem(miniToggle)
app.mainMenu = mainMenu

app.run()
