import ARKit
import CoreML
import CoreImage
import Accelerate
import UIKit

enum SPECTRANetProcessor {

    // Half the training resolution — 4× fewer pixels, ~3–4× faster on Neural Engine
    static let modelH: Int = 384
    static let modelW: Int = 512
    static let depthMax: Float = 10.0
    static let depthMin: Float = 0.5

    private static let imagenetMean: (Float, Float, Float) = (0.485, 0.456, 0.406)
    private static let imagenetStd:  (Float, Float, Float) = (0.229, 0.224, 0.225)

    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    private static let sharedModel: spectranet_depth? = {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        return try? spectranet_depth(configuration: cfg)
    }()

    // MARK: - Main entry point

    nonisolated static func process(frame: ARFrame) -> DepthResult? {
        guard let sceneDepth = frame.sceneDepth,
              let model = sharedModel else { return nil }

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
        guard Float(validCount) / Float(lH * lW) >= 0.01 else { return nil }

        // Zero out non-high-confidence depth before bicubic upsample
        for i in 0..<lH * lW where confLow[i] < 2 { loDepth[i] = 0 }

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

        var confFloat = confLow.map { $0 == 2 ? Float(1) : Float(0) }
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
        guard let output = try? model.prediction(rgb: rgbArr, bicubic_norm: bicubicArr, conf_hi: confArr) else { return nil }

        var depths = [Float](repeating: 0, count: count)
        let predNorm = output.pred_norm
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

        return DepthProcessor.colorize(depths: clamped, width: W, height: H)
    }

    // MARK: - RGB preprocessing via vImage (fast channel split + vDSP normalize)

    private static func makeRGBArray(from pixelBuffer: CVPixelBuffer, H: Int, W: Int) -> MLMultiArray? {
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
        var rPlane = [Float](repeating: 0, count: count)
        var gPlane = [Float](repeating: 0, count: count)
        var bPlane = [Float](repeating: 0, count: count)
        var aPlane = [Float](repeating: 0, count: count)  // discarded

        // vImageConvert_BGRA8888toPlanarF splits interleaved BGRA → 4 float planes in [0,1]
        var maxF = Pixel_FFFF(1, 1, 1, 1)
        var minF = Pixel_FFFF(0, 0, 0, 0)

        rPlane.withUnsafeMutableBufferPointer { rp in
        gPlane.withUnsafeMutableBufferPointer { gp in
        bPlane.withUnsafeMutableBufferPointer { bp in
        aPlane.withUnsafeMutableBufferPointer { ap in
            var src = vImage_Buffer(data: base, height: vImagePixelCount(H),
                                    width: vImagePixelCount(W), rowBytes: bpr)
            var rBuf = vImage_Buffer(data: rp.baseAddress!, height: vImagePixelCount(H),
                                     width: vImagePixelCount(W), rowBytes: W * MemoryLayout<Float>.size)
            var gBuf = vImage_Buffer(data: gp.baseAddress!, height: vImagePixelCount(H),
                                     width: vImagePixelCount(W), rowBytes: W * MemoryLayout<Float>.size)
            var bBuf = vImage_Buffer(data: bp.baseAddress!, height: vImagePixelCount(H),
                                     width: vImagePixelCount(W), rowBytes: W * MemoryLayout<Float>.size)
            var aBuf = vImage_Buffer(data: ap.baseAddress!, height: vImagePixelCount(H),
                                     width: vImagePixelCount(W), rowBytes: W * MemoryLayout<Float>.size)
            // BGRA → destB, destG, destR, destA (channel order matches argument order)
            vImageConvert_BGRA8888toPlanarF(&src, &bBuf, &gBuf, &rBuf, &aBuf, &maxF, &minF, 0)
        }}}}

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

    private static func vImageResampleFloat(_ src: [Float], srcW: Int, srcH: Int, dstW: Int, dstH: Int) -> [Float] {
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
