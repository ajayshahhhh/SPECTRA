import ARKit
import UIKit
import CoreImage
import Foundation

enum CaptureManager {

    /// Saves an RGB PNG and a raw float32 depth binary to the app's Documents directory.
    /// Returns a short status string for display.
    nonisolated static func capture(frame: ARFrame) async -> String {
        let result = performCapture(frame: frame)
        return result
    }

    // MARK: - Private

    nonisolated private static func performCapture(frame: ARFrame) -> String {
        guard let sceneDepth = frame.sceneDepth else {
            return "No LiDAR data on this frame"
        }

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let stamp = timestampString()

        // ── 1. RGB PNG ──────────────────────────────────────────────────
        let rgbURL = docs.appendingPathComponent("capture_\(stamp).png")
        let ciCtx = CIContext()
        let ci = CIImage(cvPixelBuffer: frame.capturedImage)
            .oriented(CGImagePropertyOrientation.right)   // landscape sensor → portrait
        guard let cg = ciCtx.createCGImage(ci, from: ci.extent) else {
            return "Failed to render RGB image"
        }
        guard let pngData = UIImage(cgImage: cg).pngData() else {
            return "Failed to encode PNG"
        }
        do {
            try pngData.write(to: rgbURL, options: .atomic)
        } catch {
            return "RGB save error: \(error.localizedDescription)"
        }

        // ── 2. Depth binary ─────────────────────────────────────────────
        let depthMap = sceneDepth.depthMap
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let bpr = CVPixelBufferGetBytesPerRow(depthMap)
        guard let baseAddr = CVPixelBufferGetBaseAddress(depthMap) else {
            return "Cannot access depth buffer"
        }

        // Build confidence lookup
        var confs = [UInt8]()
        if let confBuf = sceneDepth.confidenceMap {
            CVPixelBufferLockBaseAddress(confBuf, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(confBuf, .readOnly) }
            let cw = CVPixelBufferGetWidth(confBuf)
            let ch = CVPixelBufferGetHeight(confBuf)
            let cbr = CVPixelBufferGetBytesPerRow(confBuf)
            if cw == width, ch == height, let ca = CVPixelBufferGetBaseAddress(confBuf) {
                confs = [UInt8](repeating: 0, count: width * height)
                for row in 0..<height {
                    let src = ca.advanced(by: row * cbr).bindMemory(to: UInt8.self, capacity: width)
                    for col in 0..<width { confs[row * width + col] = src[col] }
                }
            }
        }
        let hasConf = confs.count == width * height

        // Float32 array, 0 where confidence is low or value is invalid
        var floats = [Float32](repeating: 0, count: width * height)
        for row in 0..<height {
            let src = baseAddr.advanced(by: row * bpr).bindMemory(to: Float32.self, capacity: width)
            for col in 0..<width {
                let idx = row * width + col
                let d = src[col]
                guard d > 0, d.isFinite else { continue }
                if hasConf, confs[idx] == 0 { continue }   // zero out low-confidence
                floats[idx] = d
            }
        }

        // Raw binary: numpy compatible via np.fromfile(f, dtype=np.float32).reshape(H, W)
        let binURL = docs.appendingPathComponent("capture_\(stamp)_depth_\(width)x\(height).bin")
        let binData = floats.withUnsafeBytes { Data($0) }
        do {
            try binData.write(to: binURL, options: .atomic)
        } catch {
            return "RGB saved; depth error: \(error.localizedDescription)"
        }

        return "Saved \(stamp)"
    }

    nonisolated private static func timestampString() -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd_HHmmss"
        return fmt.string(from: Date())
    }
}
