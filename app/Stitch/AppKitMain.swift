import AppKit
import UniformTypeIdentifiers

@main
enum StitchMain {
    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.regular)
        let delegate = AppController()
        app.delegate = delegate
        withExtendedLifetime(delegate) {
            app.run()
        }
    }
}

@MainActor
final class AppController: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private let session = SessionModel()
    private var ui: MainViewController!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.mainMenu = Self.makeMainMenu()

        ui = MainViewController(session: session)

        let contentRect = NSRect(x: 0, y: 0, width: 560, height: 440)
        window = NSWindow(
            contentRect: contentRect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Stitch"
        window.minSize = NSSize(width: 420, height: 360)
        window.contentView = ui.view
        window.setContentSize(contentRect.size)
        window.center()
        window.isReleasedWhenClosed = false
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private static func makeMainMenu() -> NSMenu {
        let mainMenu = NSMenu()

        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)
        let appMenu = NSMenu(title: "Stitch")
        appMenuItem.submenu = appMenu
        appMenu.addItem(withTitle: "About Stitch", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Hide Stitch", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        let hideOthers = appMenu.addItem(withTitle: "Hide Others", action: #selector(NSApplication.hideOtherApplications(_:)), keyEquivalent: "h")
        hideOthers.keyEquivalentModifierMask = [.command, .option]
        appMenu.addItem(withTitle: "Show All", action: #selector(NSApplication.unhideAllApplications(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit Stitch", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

        let fileMenuItem = NSMenuItem()
        mainMenu.addItem(fileMenuItem)
        let fileMenu = NSMenu(title: "File")
        fileMenuItem.submenu = fileMenu
        fileMenu.addItem(withTitle: "Open…", action: #selector(MainViewController.openClicked), keyEquivalent: "o")

        let editMenuItem = NSMenuItem()
        mainMenu.addItem(editMenuItem)
        let editMenu = NSMenu(title: "Edit")
        editMenuItem.submenu = editMenu
        editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")

        let windowMenuItem = NSMenuItem()
        mainMenu.addItem(windowMenuItem)
        let windowMenu = NSMenu(title: "Window")
        windowMenuItem.submenu = windowMenu
        windowMenu.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        windowMenu.addItem(withTitle: "Zoom", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        windowMenu.addItem(.separator())
        windowMenu.addItem(withTitle: "Bring All to Front", action: #selector(NSApplication.arrangeInFront(_:)), keyEquivalent: "")
        NSApp.windowsMenu = windowMenu

        return mainMenu
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        let tiffs = urls.filter { SessionModel.isTIFF($0) }
        if !tiffs.isEmpty {
            session.openImages(tiffs)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@MainActor
final class MainViewController: NSViewController {
    private let session: SessionModel
    private let titleLabel = NSTextField(labelWithString: "Stitch")
    private let subtitleLabel = NSTextField(
        labelWithString: "Planar-stitch overlapping flatbed TIFF scans (CD booklets / digipaks)."
    )
    private let dropBox = NSBox()
    private let dropLabel = NSTextField(wrappingLabelWithString: "Drop two or more TIFFs here")
    private let statusLabel = NSTextField(wrappingLabelWithString: "Waiting for scans…")
    private let detailLabel = NSTextField(wrappingLabelWithString: "")
    private let spinner = NSProgressIndicator()
    private let openButton = NSButton(title: "Open TIFFs…", target: nil, action: nil)
    private let againButton = NSButton(title: "Stitch again", target: nil, action: nil)

    init(session: SessionModel) {
        self.session = session
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError() }

    override func loadView() {
        let root = DropView(frame: NSRect(x: 0, y: 0, width: 560, height: 440)) { [weak self] urls in
            self?.session.openImages(urls)
        }
        root.wantsLayer = true
        root.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        view = root
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        titleLabel.font = .systemFont(ofSize: 28, weight: .semibold)
        subtitleLabel.textColor = .secondaryLabelColor
        subtitleLabel.font = .systemFont(ofSize: 13)

        dropBox.boxType = .custom
        dropBox.titlePosition = .noTitle
        dropBox.borderWidth = 2
        dropBox.cornerRadius = 12
        dropBox.borderColor = NSColor.separatorColor
        dropBox.fillColor = NSColor.controlBackgroundColor
        dropBox.contentViewMargins = NSSize(width: 16, height: 16)
        dropBox.translatesAutoresizingMaskIntoConstraints = false

        dropLabel.alignment = .center
        dropLabel.font = .systemFont(ofSize: 14, weight: .medium)
        dropLabel.translatesAutoresizingMaskIntoConstraints = false
        dropBox.addSubview(dropLabel)

        statusLabel.font = .systemFont(ofSize: 16, weight: .semibold)
        detailLabel.textColor = .secondaryLabelColor
        detailLabel.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        detailLabel.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        spinner.style = .spinning
        spinner.controlSize = .small
        spinner.isDisplayedWhenStopped = false

        openButton.target = self
        openButton.action = #selector(openClicked)
        againButton.target = self
        againButton.action = #selector(againClicked)

        let spacer = NSView()
        spacer.setContentHuggingPriority(.defaultLow, for: .horizontal)
        let buttonRow = NSStackView(views: [openButton, againButton, spacer])
        buttonRow.orientation = .horizontal
        buttonRow.spacing = 10
        buttonRow.distribution = .fill

        let stack = NSStackView(views: [
            titleLabel, subtitleLabel, dropBox, statusLabel, detailLabel, spinner, buttonRow
        ])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 14
        stack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(stack)

        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24),
            stack.topAnchor.constraint(equalTo: view.topAnchor, constant: 24),
            stack.bottomAnchor.constraint(lessThanOrEqualTo: view.bottomAnchor, constant: -24),

            dropBox.heightAnchor.constraint(equalToConstant: 140),
            dropBox.widthAnchor.constraint(equalTo: stack.widthAnchor),
            dropLabel.centerXAnchor.constraint(equalTo: dropBox.centerXAnchor),
            dropLabel.centerYAnchor.constraint(equalTo: dropBox.centerYAnchor),
            dropLabel.leadingAnchor.constraint(greaterThanOrEqualTo: dropBox.leadingAnchor, constant: 16),
            dropLabel.trailingAnchor.constraint(lessThanOrEqualTo: dropBox.trailingAnchor, constant: -16),

            buttonRow.widthAnchor.constraint(equalTo: stack.widthAnchor),
        ])

        Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
        refresh()
    }

    private func refresh() {
        if session.imageURLs.count >= 2 {
            let names = session.imageURLs.map(\.lastPathComponent).joined(separator: "\n")
            dropLabel.stringValue = "\(session.imageURLs.count) files (filename order):\n\(names)"
        } else if session.imageURLs.count == 1 {
            dropLabel.stringValue = "Need one more TIFF…\n\(session.imageURLs[0].lastPathComponent)"
        } else {
            dropLabel.stringValue = "Drop two or more TIFFs here"
        }

        if session.isBusy {
            spinner.startAnimation(nil)
            statusLabel.stringValue = "Stitching…"
            detailLabel.stringValue = "Affine planar match + gain compensation + multiband blend."
        } else {
            spinner.stopAnimation(nil)
            if let err = session.errorText {
                statusLabel.stringValue = "Error"
                detailLabel.stringValue = err
            } else if let result = session.result {
                statusLabel.stringValue = "Stitched"
                var lines: [String] = []
                if let msg = result.message { lines.append(msg) }
                if let out = result.outputPath { lines.append("Out: \(out)") }
                if let w = result.width, let h = result.height {
                    lines.append("Size: \(w)×\(h)")
                }
                detailLabel.stringValue = lines.joined(separator: "\n")
            } else {
                statusLabel.stringValue = "Waiting for scans…"
                detailLabel.stringValue = "Overlapping flatbed panels only — planar motion, no panorama skew."
            }
        }

        againButton.isEnabled = session.imageURLs.count >= 2 && !session.isBusy
    }

    @objc func openClicked() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.tiff]
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK {
            session.openImages(panel.urls)
        }
    }

    @objc private func againClicked() {
        Task { await session.runStitch() }
    }
}

final class DropView: NSView {
    private let onDrop: ([URL]) -> Void

    init(frame: NSRect, onDrop: @escaping ([URL]) -> Void) {
        self.onDrop = onDrop
        super.init(frame: frame)
        registerForDraggedTypes([.fileURL])
        wantsLayer = true
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError() }

    override var intrinsicContentSize: NSSize {
        NSSize(width: 560, height: 440)
    }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        tiffURLs(from: sender).count >= 1 ? .copy : []
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        let urls = tiffURLs(from: sender)
        guard !urls.isEmpty else { return false }
        onDrop(urls)
        return true
    }

    private func tiffURLs(from sender: NSDraggingInfo) -> [URL] {
        let urls = sender.draggingPasteboard.readObjects(forClasses: [NSURL.self]) as? [URL] ?? []
        return urls.filter { SessionModel.isTIFF($0) }
    }
}
