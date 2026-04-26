import ARKit
import UIKit
import CoreImage
import Foundation
import Photos

struct CaptureResult {
    var message: String
    var urls: [URL]
}

enum CaptureManager {

    /// Saves an RGB PNG to the camera roll and Documents directory.
    nonisolated static func capture(frame: ARFrame) async -> CaptureResult {
        return await performCapture(frame: frame)
    }

    // MARK: - Private

    nonisolated private static func performCapture(frame: ARFrame) async -> CaptureResult {
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
        let composite = renderer.image { ctx in
            rgbImage.draw(in: CGRect(origin: .zero, size: size))
            depthResult.colorImage.draw(in: CGRect(origin: .zero, size: size), blendMode: .normal, alpha: 0.5)
            drawColorScale(in: ctx.cgContext, imageSize: size,
                           minDepth: depthResult.minDepth, maxDepth: depthResult.maxDepth)
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

        // Save to camera roll
        let photoSaved = await saveToCameraRoll(image: composite)
        let message = photoSaved ? "Saved to Photos & Files" : "Saved to Files (Photos permission denied)"

        return CaptureResult(message: message, urls: [url])
    }

    nonisolated private static func saveToCameraRoll(image: UIImage) async -> Bool {
        let status = PHPhotoLibrary.authorizationStatus(for: .addOnly)

        if status == .notDetermined {
            let newStatus = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
            if newStatus != .authorized && newStatus != .limited {
                return false
            }
        } else if status != .authorized && status != .limited {
            return false
        }

        do {
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAsset(from: image)
            }
            return true
        } catch {
            print("Error saving to camera roll: \(error)")
            return false
        }
    }

    nonisolated private static func drawColorScale(in ctx: CGContext, imageSize: CGSize,
                                                      minDepth: Float, maxDepth: Float) {
        let scale = imageSize.height / 844
        let barWidth: CGFloat = 16 * scale
        let barHeight: CGFloat = 160 * scale
        let margin: CGFloat = 12 * scale
        let barX = imageSize.width - margin - barWidth
        let barY = imageSize.height * 0.35

        let colors: [CGColor] = [
            UIColor.red.cgColor, UIColor.yellow.cgColor,
            UIColor.green.cgColor, UIColor.cyan.cgColor, UIColor.blue.cgColor
        ]
        let locations: [CGFloat] = [0, 0.25, 0.5, 0.75, 1.0]
        guard let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(),
                                        colors: colors as CFArray,
                                        locations: locations) else { return }

        let barRect = CGRect(x: barX, y: barY, width: barWidth, height: barHeight)
        let cornerRadius: CGFloat = 4 * scale
        let barPath = UIBezierPath(roundedRect: barRect, cornerRadius: cornerRadius)
        ctx.saveGState()
        barPath.addClip()
        ctx.drawLinearGradient(gradient,
                               start: CGPoint(x: barX, y: barY),
                               end: CGPoint(x: barX, y: barY + barHeight),
                               options: [])
        ctx.restoreGState()

        ctx.setStrokeColor(UIColor.white.withAlphaComponent(0.6).cgColor)
        ctx.setLineWidth(1.5 * scale)
        barPath.stroke()

        let fontSize: CGFloat = 12 * scale
        let font = UIFont.monospacedSystemFont(ofSize: fontSize, weight: .medium)
        let shadow = NSShadow()
        shadow.shadowColor = UIColor.black.withAlphaComponent(0.8)
        shadow.shadowOffset = CGSize(width: 0, height: 1 * scale)
        shadow.shadowBlurRadius = 3 * scale
        let attrs: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: UIColor.white,
            .shadow: shadow
        ]

        let mid = (minDepth + maxDepth) / 2
        let labels: [(String, CGFloat)] = [
            (String(format: "%.1fm", minDepth), barY),
            (String(format: "%.1fm", mid), barY + barHeight / 2),
            (String(format: "%.1fm", maxDepth), barY + barHeight)
        ]

        let labelGap: CGFloat = 6 * scale
        for (text, centerY) in labels {
            let nsText = text as NSString
            let textSize = nsText.size(withAttributes: attrs)
            let textX = barX - labelGap - textSize.width
            let textY = centerY - textSize.height / 2
            nsText.draw(at: CGPoint(x: textX, y: textY), withAttributes: attrs)
        }
    }

    nonisolated private static func timestampString() -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd_HHmmss"
        return fmt.string(from: Date())
    }
}
