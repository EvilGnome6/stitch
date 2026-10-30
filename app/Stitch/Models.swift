import Foundation

struct EngineResult: Codable, Equatable {
    var ok: Bool
    var status: String
    var message: String?
    var inputPaths: [String]?
    var outputPath: String?
    var width: Int?
    var height: Int?
    var count: Int?
    var tiff: Bool?
}

@MainActor
final class SessionModel {
    var imageURLs: [URL] = []
    var result: EngineResult?
    var isBusy = false
    var errorText: String?

    private let engine = EngineClient()

    func openImages(_ urls: [URL]) {
        let tiffs = urls.filter { Self.isTIFF($0) }.sorted {
            $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending
        }
        guard tiffs.count >= 2 else {
            imageURLs = tiffs
            result = nil
            errorText = tiffs.isEmpty
                ? "Drop at least two TIFF scans."
                : "Need at least two overlapping TIFFs (got \(tiffs.count))."
            return
        }
        imageURLs = tiffs
        result = nil
        errorText = nil
        Task { await runStitch() }
    }

    func runStitch() async {
        guard imageURLs.count >= 2 else { return }
        isBusy = true
        errorText = nil
        defer { isBusy = false }
        do {
            let (res, _) = try await engine.stitch(inputURLs: imageURLs)
            result = res
        } catch {
            errorText = error.localizedDescription
        }
    }

    static func isTIFF(_ url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        return ext == "tif" || ext == "tiff"
    }
}
