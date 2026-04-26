import ARKit
import UIKit
import CoreVideo
import Accelerate

struct DepthResult {
    let colorImage: UIImage
    let centerDistance: Float?
    let minDepth: Float
    let maxDepth: Float
}

enum DepthProcessor {

    // Turbo-inspired HSV LUT: red (near) → yellow → green → cyan → blue (far)
    nonisolated static let lut: [(UInt8, UInt8, UInt8)] = {
        (0..<256).map { i in
            let t = Float(i) / 255.0
            let hue = t * (240.0 / 360.0)
            let sector = hue * 6
            let si = Int(sector) % 6
            let f = sector - Float(Int(sector))
            let q = 1 - f
            let r, g, b: Float
            switch si {
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
    }()

    // MARK: - Live LiDAR entry point

    nonisolated static func process(frame: ARFrame) -> DepthResult? {
        guard let sceneDepth = frame.sceneDepth else { return nil }

        let depthMap = sceneDepth.depthMap
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width  = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let bpr    = CVPixelBufferGetBytesPerRow(depthMap)
        guard let baseAddr = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
        let count = width * height

        var depths = [Float](repeating: 0, count: count)
        depths.withUnsafeMutableBufferPointer { dst in
            for row in 0..<height {
                memcpy(dst.baseAddress!.advanced(by: row * width),
                       baseAddr.advanced(by: row * bpr),
                       width * MemoryLayout<Float>.size)
            }
        }

        var confs = [UInt8]()
        if let confBuf = sceneDepth.confidenceMap {
            CVPixelBufferLockBaseAddress(confBuf, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(confBuf, .readOnly) }
            let cw = CVPixelBufferGetWidth(confBuf), ch = CVPixelBufferGetHeight(confBuf)
            let cbr = CVPixelBufferGetBytesPerRow(confBuf)
            if cw == width, ch == height, let ca = CVPixelBufferGetBaseAddress(confBuf) {
                confs = [UInt8](repeating: 0, count: count)
                confs.withUnsafeMutableBufferPointer { dst in
                    for row in 0..<height {
                        memcpy(dst.baseAddress!.advanced(by: row * width),
                               ca.advanced(by: row * cbr), width)
                    }
                }
            }
        }
        let hasConf = confs.count == count

        // Mask invalid pixels to NaN so colorize skips them
        for i in 0..<count {
            let d = depths[i]
            if d <= 0 || !d.isFinite || (hasConf && confs[i] == 0) {
                depths[i] = .nan
            }
        }

        return colorize(depths: depths, width: width, height: height)
    }

    // MARK: - Shared colorize (used by both LiDAR and SPECTRANet)
    // Invalid pixels should be NaN or ≤ 0; they render as transparent.

    nonisolated static func colorize(depths: [Float], width: Int, height: Int) -> DepthResult? {
        let count = width * height
        guard count == depths.count else { return nil }

        var minD: Float = .greatestFiniteMagnitude
        var maxD: Float = -.greatestFiniteMagnitude
        for d in depths {
            guard !d.isNaN, d > 0 else { continue }
            if d < minD { minD = d }
            if d > maxD { maxD = d }
        }
        guard maxD > minD else { return nil }

        // Center distance (5×5 sample)
        let cx = width / 2, cy = height / 2
        var distSum: Float = 0, distCount = 0
        for dy in -2...2 {
            for dx in -2...2 {
                let x = cx + dx, y = cy + dy
                guard x >= 0, x < width, y >= 0, y < height else { continue }
                let d = depths[y * width + x]
                guard !d.isNaN, d > 0 else { continue }
                distSum += d; distCount += 1
            }
        }
        let centerDist: Float? = distCount > 0 ? distSum / Float(distCount) : nil

        // Normalize valid pixels to 0..255
        let range = maxD - minD
        var negMin = -minD
        var scale = 255.0 / range as Float
        var normalized = [Float](repeating: 0, count: count)
        vDSP_vsadd(depths, 1, &negMin, &normalized, 1, vDSP_Length(count))
        vDSP_vsmul(normalized, 1, &scale, &normalized, 1, vDSP_Length(count))

        // LUT → RGBA
        var rgba = [UInt8](repeating: 0, count: count * 4)
        for i in 0..<count {
            let n = normalized[i]
            guard !n.isNaN, depths[i] > 0 else { continue }
            let idx = min(255, max(0, Int(n)))
            let (r, g, b) = lut[idx]
            rgba[i*4] = r; rgba[i*4+1] = g; rgba[i*4+2] = b; rgba[i*4+3] = 255
        }

        guard let image = makePortraitImage(rgba, width: width, height: height) else { return nil }
        return DepthResult(colorImage: image, centerDistance: centerDist, minDepth: minD, maxDepth: maxD)
    }

    // MARK: - Image creation

    nonisolated static func makePortraitImage(_ pixels: [UInt8], width: Int, height: Int) -> UIImage? {
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
