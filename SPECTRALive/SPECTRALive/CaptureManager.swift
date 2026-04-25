import ARKit
import UIKit
import CoreImage
import Foundation

struct CaptureResult {
    var message: String
    var urls: [URL]
}

enum CaptureManager {

    /// Saves an RGB PNG and a raw float32 depth binary to the app's Documents directory.
    nonisolated static func capture(frame: ARFrame) async -> CaptureResult {
        return performCapture(frame: frame)
    }

    // MARK: - Private

    nonisolated private static func performCapture(frame: ARFrame) -> CaptureResult {
        guard frame.sceneDepth != nil else {
            return CaptureResult(message: "No LiDAR data on this frame", urls: [])
        }

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let stamp = timestampString()

        let ciCtx = CIContext()
        let ci = CIImage(cvPixelBuffer: frame.capturedImage)
            .oriented(CGImagePropertyOrientation.right)
        guard let cg = ciCtx.createCGImage(ci, from: ci.extent) else {
            return CaptureResult(message: "Failed to render RGB image", urls: [])
        }

        guard let depthResult = DepthProcessor.process(frame: frame) else {
            return CaptureResult(message: "Failed to process depth", urls: [])
        }

        let rgbImage = UIImage(cgImage: cg)
        let size = rgbImage.size
        let renderer = UIGraphicsImageRenderer(size: size)
        let composite = renderer.image { _ in
            rgbImage.draw(in: CGRect(origin: .zero, size: size))
            depthResult.colorImage.draw(in: CGRect(origin: .zero, size: size), blendMode: .normal, alpha: 0.5)
        }

        guard let pngData = composite.pngData() else {
            return CaptureResult(message: "Failed to encode image", urls: [])
        }

        let url = docs.appendingPathComponent("capture_\(stamp).png")
        do {
            try pngData.write(to: url, options: .atomic)
        } catch {
            return CaptureResult(message: "Save error: \(error.localizedDescription)", urls: [])
        }

        return CaptureResult(message: "Saved \(stamp)", urls: [url])
    }

    nonisolated private static func timestampString() -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd_HHmmss"
        return fmt.string(from: Date())
    }
}
