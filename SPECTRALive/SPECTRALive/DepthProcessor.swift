import ARKit
import UIKit
import CoreVideo

struct DepthResult {
    let colorImage: UIImage
    let centerDistance: Float?
}

enum DepthProcessor {

    /// Produces a heatmap UIImage (portrait orientation) and center distance from an ARFrame.
    /// Returns nil if no sceneDepth is available or the depth range is degenerate.
    nonisolated static func process(frame: ARFrame) -> DepthResult? {
        guard let sceneDepth = frame.sceneDepth else { return nil }

        let depthMap = sceneDepth.depthMap
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
        guard let baseAddr = CVPixelBufferGetBaseAddress(depthMap) else { return nil }

        // Flatten with stride to handle any row padding
        var depths = [Float32](repeating: 0, count: width * height)
        for row in 0..<height {
            let src = baseAddr.advanced(by: row * bytesPerRow)
                .bindMemory(to: Float32.self, capacity: width)
            for col in 0..<width {
                depths[row * width + col] = src[col]
            }
        }

        // Confidence map: UInt8 where 0=low, 1=medium, 2=high
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
                    for col in 0..<width {
                        confs[row * width + col] = src[col]
                    }
                }
            }
        }
        let hasConf = confs.count == width * height

        // Auto-scale: min/max over high-confidence, positive, finite values
        var minD: Float = .greatestFiniteMagnitude
        var maxD: Float = -.greatestFiniteMagnitude
        for i in 0..<depths.count {
            let d = depths[i]
            guard d > 0, d.isFinite else { continue }
            if hasConf, confs[i] == 0 { continue }
            if d < minD { minD = d }
            if d > maxD { maxD = d }
        }
        guard maxD > minD else { return nil }
        let range = maxD - minD

        // Center distance: average of 5x5 patch, excluding low-confidence pixels
        let cx = width / 2, cy = height / 2
        var distSum: Float = 0
        var distCount = 0
        for dy in -2...2 {
            for dx in -2...2 {
                let x = cx + dx, y = cy + dy
                guard x >= 0, x < width, y >= 0, y < height else { continue }
                let idx = y * width + x
                let d = depths[idx]
                guard d > 0, d.isFinite else { continue }
                if hasConf, confs[idx] == 0 { continue }
                distSum += d
                distCount += 1
            }
        }
        let centerDist: Float? = distCount > 0 ? distSum / Float(distCount) : nil

        // Build RGBA heatmap — alpha=0 for no-data, 255 for valid (SwiftUI applies .opacity(0.5))
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<depths.count {
            let d = depths[i]
            let valid = d > 0 && d.isFinite && (!hasConf || confs[i] > 0)
            guard valid else { continue }
            let t = (d - minD) / range      // 0 = close (red), 1 = far (blue)
            let (r, g, b) = heatColor(t: t)
            let base = i * 4
            rgba[base]     = r
            rgba[base + 1] = g
            rgba[base + 2] = b
            rgba[base + 3] = 255
        }

        guard let image = makePortraitImage(rgba, width: width, height: height) else { return nil }
        return DepthResult(colorImage: image, centerDistance: centerDist)
    }

    // MARK: - Helpers

    /// t=0 → red (hue 0°), t=1 → blue (hue 240°) via HSV with s=1, v=1
    nonisolated private static func heatColor(t: Float) -> (UInt8, UInt8, UInt8) {
        let t = min(1, max(0, t))
        let hue = t * (240.0 / 360.0)
        let sector = hue * 6
        let i = Int(sector) % 6
        let f = sector - Float(Int(sector))
        let q = 1 - f
        let r, g, b: Float
        switch i {
        case 0: (r, g, b) = (1,   f,   0)
        case 1: (r, g, b) = (q,   1,   0)
        case 2: (r, g, b) = (0,   1,   f)
        case 3: (r, g, b) = (0,   q,   1)
        case 4: (r, g, b) = (f,   0,   1)
        default:(r, g, b) = (1,   0,   q)
        }
        return (UInt8(min(255, r * 255)),
                UInt8(min(255, g * 255)),
                UInt8(min(255, b * 255)))
    }

    /// Creates a UIImage from an RGBA byte array, rotated to portrait orientation.
    /// LiDAR depth buffers are landscape (wider than tall); .right rotates 90° CW for portrait display.
    nonisolated private static func makePortraitImage(_ pixels: [UInt8], width: Int, height: Int) -> UIImage? {
        var mutable = pixels
        return mutable.withUnsafeMutableBytes { ptr in
            guard let ctx = CGContext(
                data: ptr.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ), let cg = ctx.makeImage() else { return nil }
            return UIImage(cgImage: cg, scale: 1.0, orientation: .right)
        }
    }
}
