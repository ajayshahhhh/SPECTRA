import ARKit
import CoreML
import CoreImage
import Accelerate
import UIKit

struct SPECTRANetResult {
    let depth: DepthResult
    let edge: EdgeDepthResult?
    let recoloredImage: UIImage?
}

enum SPECTRANetProcessor {

    nonisolated static let modelH: Int = 768
    nonisolated static let modelW: Int = 1024
    nonisolated static let depthMax: Float = 10.0
    nonisolated static let depthMin: Float = 0.3

    nonisolated private static let imagenetMean: (Float, Float, Float) = (0.485, 0.456, 0.406)
    nonisolated private static let imagenetStd:  (Float, Float, Float) = (0.229, 0.224, 0.225)

    nonisolated private static let ciContext = CIContext(options: [
        .useSoftwareRenderer: false,
        .workingColorSpace: CGColorSpaceCreateDeviceRGB()
    ])

    nonisolated private static let emaAlpha: Float = 0.6
    nonisolated(unsafe) private static var prevDepths: [Float]?

    nonisolated(unsafe) private static let sharedMLModel: MLModel? = {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        guard let url = Bundle.main.url(forResource: "spectranet_depth", withExtension: "mlmodelc") else { return nil }
        return try? MLModel(contentsOf: url, configuration: cfg)
    }()

    // MARK: - Main entry point

    nonisolated static func process(frame: ARFrame) -> SPECTRANetResult? {
        guard let sceneDepth = frame.sceneDepth,
              let model = sharedMLModel else { return nil }

        let H = modelH, W = modelW, count = H * W

        // Read depth + confidence at native LiDAR resolution
        let depthMap = sceneDepth.depthMap
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        let lW = CVPixelBufferGetWidth(depthMap)
        let lH = CVPixelBufferGetHeight(depthMap)
        let lBpr = CVPixelBufferGetBytesPerRow(depthMap)
        guard let lBase = CVPixelBufferGetBaseAddress(depthMap) else {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            return nil
        }
        var loDepth = [Float](repeating: 0, count: lH * lW)
        loDepth.withUnsafeMutableBufferPointer { dst in
            for row in 0..<lH {
                memcpy(dst.baseAddress!.advanced(by: row * lW),
                       lBase.advanced(by: row * lBpr),
                       lW * MemoryLayout<Float>.size)
            }
        }
        CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)

        var confLow = [UInt8](repeating: 0, count: lH * lW)
        if let cb = sceneDepth.confidenceMap {
            CVPixelBufferLockBaseAddress(cb, .readOnly)
            let cbpr = CVPixelBufferGetBytesPerRow(cb)
            if let ca = CVPixelBufferGetBaseAddress(cb) {
                confLow.withUnsafeMutableBufferPointer { dst in
                    for row in 0..<lH {
                        memcpy(dst.baseAddress!.advanced(by: row * lW),
                               ca.advanced(by: row * cbpr), lW)
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(cb, .readOnly)
        }

        // ── Validity gate ─────────────────────────────────────────────
        // If fewer than 1% of pixels have high-confidence LiDAR, the model
        // has no real anchor and will hallucinate a depth (usually ~0.5–1 m).
        // Return nil so the overlay shows nothing rather than wrong depth.
        var validCount = 0
        for c in confLow where c == 2 { validCount += 1 }
        guard Float(validCount) / Float(lH * lW) >= 0.01 else {
            prevDepths = nil  // Reset temporal smoothing buffer
            return nil
        }

        // Zero out non-high-confidence depth before bicubic upsample
        for i in 0..<lH * lW where confLow[i] < 1 { loDepth[i] = 0 }

        // ── Depth inputs ──────────────────────────────────────────────
        let bicubicFlat = vImageResampleFloat(loDepth, srcW: lW, srcH: lH, dstW: W, dstH: H)

        var bicubicNorm = [Float](repeating: 0, count: count)
        var zero: Float = 0, dmax = depthMax, invMax: Float = 1.0 / depthMax
        bicubicFlat.withUnsafeBufferPointer { src in
            bicubicNorm.withUnsafeMutableBufferPointer { dst in
                vDSP_vclip(src.baseAddress!, 1, &zero, &dmax, dst.baseAddress!, 1, vDSP_Length(count))
            }
        }
        vDSP_vsmul(bicubicNorm, 1, &invMax, &bicubicNorm, 1, vDSP_Length(count))

        let confFloat = confLow.map { $0 == 2 ? Float(1) : Float(0) }
        let confHigh = vImageResampleFloat(confFloat, srcW: lW, srcH: lH, dstW: W, dstH: H)

        guard let bicubicArr = try? MLMultiArray(
                shape: [1, 1, NSNumber(value: H), NSNumber(value: W)], dataType: .float32),
              let confArr = try? MLMultiArray(
                shape: [1, 1, NSNumber(value: H), NSNumber(value: W)], dataType: .float32)
        else { return nil }

        memcpy(bicubicArr.dataPointer, bicubicNorm, count * MemoryLayout<Float>.size)
        memcpy(confArr.dataPointer, confHigh, count * MemoryLayout<Float>.size)

        // ── RGB input (vImage channel split — avoids per-pixel Swift loop) ──
        guard let rgbArr = makeRGBArray(from: frame.capturedImage, H: H, W: W) else { return nil }

        // ── Inference ─────────────────────────────────────────────────
        guard let input = try? MLDictionaryFeatureProvider(dictionary: [
            "rgb": MLFeatureValue(multiArray: rgbArr),
            "bicubic_norm": MLFeatureValue(multiArray: bicubicArr),
            "conf_hi": MLFeatureValue(multiArray: confArr)
        ]),
        let output = try? model.prediction(from: input),
        let predNorm = output.featureValue(for: "pred_norm")?.multiArrayValue else { return nil }

        var depths = [Float](repeating: 0, count: count)
        if predNorm.dataType == .float32 {
            memcpy(&depths, predNorm.dataPointer, count * MemoryLayout<Float>.size)
        } else if predNorm.dataType == .float16 {
            var fp16 = [Float16](repeating: 0, count: count)
            memcpy(&fp16, predNorm.dataPointer, count * MemoryLayout<Float16>.size)
            vDSP.convertElements(of: fp16, to: &depths)
        } else {
            for i in 0..<count { depths[i] = predNorm[i].floatValue }
        }

        // Denormalize + clamp
        var dmin = depthMin
        depths.withUnsafeMutableBufferPointer { ptr in
            vDSP_vsmul(ptr.baseAddress!, 1, &dmax, ptr.baseAddress!, 1, vDSP_Length(count))
        }
        var clamped = depths
        depths.withUnsafeBufferPointer { src in
            clamped.withUnsafeMutableBufferPointer { dst in
                vDSP_vclip(src.baseAddress!, 1, &dmin, &dmax, dst.baseAddress!, 1, vDSP_Length(count))
            }
        }
        for i in 0..<count where confHigh[i] < 0.1 { clamped[i] = 0 }

        // Temporal smoothing only for pixels with valid current depth
        if let prev = prevDepths, prev.count == count {
            let alpha = emaAlpha
            for i in 0..<count {
                // Only blend if current pixel has valid depth, otherwise use current (0)
                if clamped[i] > 0 {
                    clamped[i] = prev[i] * (1 - alpha) + clamped[i] * alpha
                }
                // If clamped[i] == 0, keep it at 0 (don't preserve old values)
            }
        }
        prevDepths = clamped

        guard let depthResult = DepthProcessor.colorize(depths: clamped, width: W, height: H) else {
            return nil
        }

        // Skip edge detection for lower latency
        let edgeResult: EdgeDepthResult? = nil

        // Reduced resolution for faster recoloring: 480×360 instead of 640×480
        let rcW = 480, rcH = 360
        let rcDepths = vImageResampleFloat(clamped, srcW: W, srcH: H, dstW: rcW, dstH: rcH)
        let recolored = DepthProcessor.recolorCameraImage(
            capturedImage: frame.capturedImage,
            depths: rcDepths, outW: rcW, outH: rcH
        )

        return SPECTRANetResult(depth: depthResult, edge: edgeResult, recoloredImage: recolored)
    }

    // MARK: - RGB preprocessing via vImage (fast channel split + vDSP normalize)

    nonisolated private static func makeRGBArray(from pixelBuffer: CVPixelBuffer, H: Int, W: Int) -> MLMultiArray? {
        // Render landscape capturedImage → W×H BGRA pixel buffer
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        let sx = CGFloat(W) / ci.extent.width
        let sy = CGFloat(H) / ci.extent.height
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        var outBuf: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: W,
            kCVPixelBufferHeightKey as String: H,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        CVPixelBufferCreate(nil, W, H, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &outBuf)
        guard let outBuf else { return nil }
        ciContext.render(scaled, to: outBuf)

        CVPixelBufferLockBaseAddress(outBuf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(outBuf, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(outBuf) else { return nil }
        let bpr = CVPixelBufferGetBytesPerRow(outBuf)

        let count = H * W

        // Deinterleave BGRA → four UInt8 planes via vImage
        var bBytes = [UInt8](repeating: 0, count: count)
        var gBytes = [UInt8](repeating: 0, count: count)
        var rBytes = [UInt8](repeating: 0, count: count)
        var aBytes = [UInt8](repeating: 0, count: count)

        bBytes.withUnsafeMutableBufferPointer { bp in
        gBytes.withUnsafeMutableBufferPointer { gp in
        rBytes.withUnsafeMutableBufferPointer { rp in
        aBytes.withUnsafeMutableBufferPointer { ap in
            var src = vImage_Buffer(data: base, height: vImagePixelCount(H),
                                    width: vImagePixelCount(W), rowBytes: bpr)
            var bDst = vImage_Buffer(data: bp.baseAddress!, height: vImagePixelCount(H),
                                     width: vImagePixelCount(W), rowBytes: W)
            var gDst = vImage_Buffer(data: gp.baseAddress!, height: vImagePixelCount(H),
                                     width: vImagePixelCount(W), rowBytes: W)
            var rDst = vImage_Buffer(data: rp.baseAddress!, height: vImagePixelCount(H),
                                     width: vImagePixelCount(W), rowBytes: W)
            var aDst = vImage_Buffer(data: ap.baseAddress!, height: vImagePixelCount(H),
                                     width: vImagePixelCount(W), rowBytes: W)
            vImageConvert_ARGB8888toPlanar8(&src, &bDst, &gDst, &rDst, &aDst, vImage_Flags(kvImageNoFlags))
        }}}}

        // UInt8 → Float [0,1]
        var rPlane = [Float](repeating: 0, count: count)
        var gPlane = [Float](repeating: 0, count: count)
        var bPlane = [Float](repeating: 0, count: count)
        var inv255: Float = 1.0 / 255.0
        vDSP_vfltu8(rBytes, 1, &rPlane, 1, vDSP_Length(count))
        vDSP_vsmul(rPlane, 1, &inv255, &rPlane, 1, vDSP_Length(count))
        vDSP_vfltu8(gBytes, 1, &gPlane, 1, vDSP_Length(count))
        vDSP_vsmul(gPlane, 1, &inv255, &gPlane, 1, vDSP_Length(count))
        vDSP_vfltu8(bBytes, 1, &bPlane, 1, vDSP_Length(count))
        vDSP_vsmul(bPlane, 1, &inv255, &bPlane, 1, vDSP_Length(count))

        // ImageNet normalize: (x − mean) / std  ≡  x * (1/std) + (−mean/std)
        let (mr, mg, mb) = imagenetMean
        let (sr, sg, sb) = imagenetStd
        var rs: Float = 1/sr, rb: Float = -mr/sr
        var gs: Float = 1/sg, gb: Float = -mg/sg
        var bs: Float = 1/sb, bb: Float = -mb/sb
        rPlane.withUnsafeMutableBufferPointer { p in vDSP_vsmsa(p.baseAddress!, 1, &rs, &rb, p.baseAddress!, 1, vDSP_Length(count)) }
        gPlane.withUnsafeMutableBufferPointer { p in vDSP_vsmsa(p.baseAddress!, 1, &gs, &gb, p.baseAddress!, 1, vDSP_Length(count)) }
        bPlane.withUnsafeMutableBufferPointer { p in vDSP_vsmsa(p.baseAddress!, 1, &bs, &bb, p.baseAddress!, 1, vDSP_Length(count)) }

        // Pack CHW: [R plane | G plane | B plane]
        guard let array = try? MLMultiArray(
            shape: [1, 3, NSNumber(value: H), NSNumber(value: W)], dataType: .float32
        ) else { return nil }
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        memcpy(ptr,                   rPlane, count * MemoryLayout<Float>.size)
        memcpy(ptr.advanced(by: count),   gPlane, count * MemoryLayout<Float>.size)
        memcpy(ptr.advanced(by: count*2), bPlane, count * MemoryLayout<Float>.size)
        return array
    }

    // MARK: - vImage float resampling

    nonisolated private static func vImageResampleFloat(_ src: [Float], srcW: Int, srcH: Int, dstW: Int, dstH: Int) -> [Float] {
        var result = [Float](repeating: 0, count: dstH * dstW)
        src.withUnsafeBytes { srcRaw in
            result.withUnsafeMutableBytes { dstRaw in
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcRaw.baseAddress!),
                                      height: vImagePixelCount(srcH), width: vImagePixelCount(srcW),
                                      rowBytes: srcW * MemoryLayout<Float>.size)
                var d = vImage_Buffer(data: dstRaw.baseAddress!,
                                      height: vImagePixelCount(dstH), width: vImagePixelCount(dstW),
                                      rowBytes: dstW * MemoryLayout<Float>.size)
                vImageScale_PlanarF(&s, &d, nil, vImage_Flags(kvImageNoFlags))  // Bilinear (faster than bicubic)
            }
        }
        return result
    }
}
