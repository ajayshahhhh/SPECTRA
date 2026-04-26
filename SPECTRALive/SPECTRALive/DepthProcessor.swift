import ARKit
import UIKit
import CoreImage
import CoreVideo
import Accelerate

struct DepthResult {
    let colorImage: UIImage
    let centerDistance: Float?
    let minDepth: Float
    let maxDepth: Float
}

enum DepthProcessor {

    // HSV LUT: red (near) → yellow → green → cyan → blue (far), avoiding magenta wrap
    nonisolated static let lut: [(UInt8, UInt8, UInt8)] = {
        (0..<256).map { i in
            let t = Float(i) / 255.0
            let hue = t * (210.0 / 360.0)  // Cap at 210° to avoid blue→magenta transition
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

    nonisolated private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

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

        let minD: Float = 0.5
        let maxD: Float = 5.0  // 5m+ = blue
        guard depths.contains(where: { !$0.isNaN && $0 > 0 }) else { return nil }

        let cx = width / 2, cy = height / 2
        var distSum: Float = 0, distCount = 0
        for dy in -5...5 {
            for dx in -5...5 {
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

    // MARK: - Camera recolor heatmap (fuses camera luminance with depth color)

    nonisolated static func recolorCamera(frame: ARFrame) -> DepthResult? {
        guard let sceneDepth = frame.sceneDepth else { return nil }

        let depthMap = sceneDepth.depthMap
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        let dW = CVPixelBufferGetWidth(depthMap)
        let dH = CVPixelBufferGetHeight(depthMap)
        let dBpr = CVPixelBufferGetBytesPerRow(depthMap)
        guard let dBase = CVPixelBufferGetBaseAddress(depthMap) else {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            return nil
        }
        var rawDepths = [Float](repeating: 0, count: dW * dH)
        rawDepths.withUnsafeMutableBufferPointer { dst in
            for row in 0..<dH {
                memcpy(dst.baseAddress!.advanced(by: row * dW),
                       dBase.advanced(by: row * dBpr),
                       dW * MemoryLayout<Float>.size)
            }
        }
        CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)

        let outW = 320, outH = 240
        let depths = resampleFloat(rawDepths, srcW: dW, srcH: dH, dstW: outW, dstH: outH)
        let minD: Float = 0.5, maxD: Float = 5.0  // 5m+ = blue

        let cx = outW / 2, cy = outH / 2
        var distSum: Float = 0, distN = 0
        for dy in -5...5 { for dx in -5...5 {
            let x = cx + dx, y = cy + dy
            guard x >= 0, x < outW, y >= 0, y < outH else { continue }
            let d = depths[y * outW + x]
            guard d > 0 && d.isFinite else { continue }
            distSum += d; distN += 1
        }}
        let centerDist: Float? = distN > 0 ? distSum / Float(distN) : nil

        guard let image = recolorCameraImage(
            capturedImage: frame.capturedImage, depths: depths, outW: outW, outH: outH
        ) else { return nil }
        return DepthResult(colorImage: image, centerDistance: centerDist, minDepth: minD, maxDepth: maxD)
    }

    // MARK: - Camera recolor core (shared by LiDAR and SPECTRANet)

    nonisolated static func recolorCameraImage(
        capturedImage: CVPixelBuffer,
        depths: [Float],
        outW: Int,
        outH: Int
    ) -> UIImage? {
        let count = outW * outH
        guard depths.count == count else { return nil }

        let minD: Float = 0.5, maxD: Float = 5.0, range = maxD - minD  // 5m+ = blue

        let ci = CIImage(cvPixelBuffer: capturedImage)
        let sx = CGFloat(outW) / ci.extent.width
        let sy = CGFloat(outH) / ci.extent.height
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        var camBuf: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: outW,
            kCVPixelBufferHeightKey as String: outH,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        CVPixelBufferCreate(nil, outW, outH, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &camBuf)
        guard let camBuf else { return nil }
        ciContext.render(scaled, to: camBuf)

        CVPixelBufferLockBaseAddress(camBuf, .readOnly)
        guard let camBase = CVPixelBufferGetBaseAddress(camBuf) else {
            CVPixelBufferUnlockBaseAddress(camBuf, .readOnly)
            return nil
        }
        let camBpr = CVPixelBufferGetBytesPerRow(camBuf)

        var rgba = [UInt8](repeating: 0, count: count * 4)
        for y in 0..<outH {
            let row = camBase.advanced(by: y * camBpr).assumingMemoryBound(to: UInt8.self)
            for x in 0..<outW {
                let i = y * outW + x
                let d = depths[i]
                let cb = Float(row[x * 4])
                let cg = Float(row[x * 4 + 1])
                let cr = Float(row[x * 4 + 2])
                let lum = (0.299 * cr + 0.587 * cg + 0.114 * cb) / 255.0

                if d > 0.1 && d.isFinite {
                    let t = max(0, min(1, (d - minD) / range))
                    let idx = min(255, max(0, Int(t * 255)))
                    let (lr, lg, lb) = lut[idx]
                    let b = 0.3 + 0.7 * lum
                    rgba[i * 4]     = UInt8(min(255, Float(lr) * b))
                    rgba[i * 4 + 1] = UInt8(min(255, Float(lg) * b))
                    rgba[i * 4 + 2] = UInt8(min(255, Float(lb) * b))
                    rgba[i * 4 + 3] = 255
                } else {
                    let gray = UInt8(max(0, min(255, lum * 60)))
                    rgba[i * 4] = gray
                    rgba[i * 4 + 1] = gray
                    rgba[i * 4 + 2] = gray
                    rgba[i * 4 + 3] = 255
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(camBuf, .readOnly)

        return makePortraitImage(rgba, width: outW, height: outH)
    }

    // MARK: - Blend pre-colorized heatmap with camera luminance

    nonisolated static func blendHeatmapWithCamera(
        heatmap: UIImage,
        capturedImage: CVPixelBuffer
    ) -> UIImage? {
        guard let heatCG = heatmap.cgImage else { return nil }
        let outW = heatCG.width
        let outH = heatCG.height
        let count = outW * outH

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var heatPixels = [UInt8](repeating: 0, count: count * 4)
        guard let heatCtx = CGContext(
            data: &heatPixels,
            width: outW, height: outH,
            bitsPerComponent: 8, bytesPerRow: outW * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        heatCtx.draw(heatCG, in: CGRect(x: 0, y: 0, width: outW, height: outH))

        let ci = CIImage(cvPixelBuffer: capturedImage)
        let sx = CGFloat(outW) / ci.extent.width
        let sy = CGFloat(outH) / ci.extent.height
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        var camBuf: CVPixelBuffer?
        CVPixelBufferCreate(nil, outW, outH, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary,
                            &camBuf)
        guard let camBuf else { return nil }
        ciContext.render(scaled, to: camBuf)

        CVPixelBufferLockBaseAddress(camBuf, .readOnly)
        guard let camBase = CVPixelBufferGetBaseAddress(camBuf) else {
            CVPixelBufferUnlockBaseAddress(camBuf, .readOnly)
            return nil
        }
        let camBpr = CVPixelBufferGetBytesPerRow(camBuf)

        var rgba = [UInt8](repeating: 0, count: count * 4)
        for y in 0..<outH {
            let camRow = camBase.advanced(by: y * camBpr).assumingMemoryBound(to: UInt8.self)
            for x in 0..<outW {
                let i = y * outW + x
                let cb = Float(camRow[x * 4])
                let cg = Float(camRow[x * 4 + 1])
                let cr = Float(camRow[x * 4 + 2])
                let lum = (0.299 * cr + 0.587 * cg + 0.114 * cb) / 255.0

                let hr = Float(heatPixels[i * 4])
                let hg = Float(heatPixels[i * 4 + 1])
                let hb = Float(heatPixels[i * 4 + 2])

                let b = 0.3 + 0.7 * lum
                rgba[i * 4]     = UInt8(min(255, hr * b))
                rgba[i * 4 + 1] = UInt8(min(255, hg * b))
                rgba[i * 4 + 2] = UInt8(min(255, hb * b))
                rgba[i * 4 + 3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(camBuf, .readOnly)

        return makePortraitImage(rgba, width: outW, height: outH)
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

    nonisolated private static func resampleFloat(
        _ src: [Float], srcW: Int, srcH: Int, dstW: Int, dstH: Int
    ) -> [Float] {
        var result = [Float](repeating: 0, count: dstH * dstW)
        src.withUnsafeBytes { srcRaw in
            result.withUnsafeMutableBytes { dstRaw in
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcRaw.baseAddress!),
                                      height: vImagePixelCount(srcH), width: vImagePixelCount(srcW),
                                      rowBytes: srcW * MemoryLayout<Float>.size)
                var d = vImage_Buffer(data: dstRaw.baseAddress!,
                                      height: vImagePixelCount(dstH), width: vImagePixelCount(dstW),
                                      rowBytes: dstW * MemoryLayout<Float>.size)
                vImageScale_PlanarF(&s, &d, nil, vImage_Flags(kvImageHighQualityResampling))
            }
        }
        return result
    }
}
