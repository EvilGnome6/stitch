import Foundation

enum EngineClientError: LocalizedError {
    case engineMissing(String)
    case badJSON(String)
    case nonZeroExit(Int, String)

    var errorDescription: String? {
        switch self {
        case .engineMissing(let path):
            return "stitch-engine not found at \(path)"
        case .badJSON(let detail):
            return "Could not parse engine JSON: \(detail)"
        case .nonZeroExit(let code, let stderr):
            return "Engine exited \(code). \(stderr)"
        }
    }
}

struct EngineClient {
    func stitch(inputURLs: [URL], outputURL: URL? = nil) async throws -> (EngineResult, String) {
        let engineURL = try resolveEngine()
        let process = Process()
        process.executableURL = engineURL
        var args = ["--json"]
        if let outputURL {
            args += ["-o", outputURL.path]
        }
        args += inputURLs.map(\.path)
        process.arguments = args
        process.currentDirectoryURL = engineURL.deletingLastPathComponent()

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        try process.run()
        process.waitUntilExit()

        let outData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errData = stderr.fileHandleForReading.readDataToEndOfFile()
        let errText = String(data: errData, encoding: .utf8) ?? ""

        do {
            let result = try JSONDecoder().decode(EngineResult.self, from: outData)
            if process.terminationStatus != 0 || result.ok == false {
                let msg = result.message ?? errText
                throw EngineClientError.nonZeroExit(Int(process.terminationStatus), msg)
            }
            return (result, errText)
        } catch let error as EngineClientError {
            throw error
        } catch {
            if process.terminationStatus != 0 {
                throw EngineClientError.nonZeroExit(Int(process.terminationStatus), errText)
            }
            let preview = String(data: outData, encoding: .utf8) ?? "<binary>"
            throw EngineClientError.badJSON("\(error.localizedDescription) — stdout: \(preview.prefix(400))")
        }
    }

    private func resolveEngine() throws -> URL {
        if let resources = Bundle.main.resourceURL {
            let bundled = resources.appendingPathComponent("engine/stitch-engine")
            if FileManager.default.isExecutableFile(atPath: bundled.path) {
                return bundled
            }
        }

        // Dev fallback: engine built by scripts/sync-stitch-engine.sh / setup.sh
        let dev = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Stitch/
            .deletingLastPathComponent() // app/
            .deletingLastPathComponent() // repo root
            .appendingPathComponent("engine/stitch-engine")
        if FileManager.default.isExecutableFile(atPath: dev.path) {
            return dev
        }

        throw EngineClientError.engineMissing(dev.path)
    }
}
