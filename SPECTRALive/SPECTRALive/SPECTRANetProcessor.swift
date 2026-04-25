import ARKit
import CoreML
import CoreImage
import Accelerate
import UIKit

enum SPECTRANetProcessor {

    static let modelH: Int = 768
    static let modelW: Int = 1024
    static let depthMax: Float = 10.0
    static let depthMin: Float = 0.5

    private static let imagenetMean: (Float, Float, Float) = (0.485, 0.456, 0.406)
    private static let imagenetStd:  (Float, Float, Float) = (0.229, 0.224, 0.225)

    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // Loaded once on first use; .all = prefer Neural Engine for max fps
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

        guard let rgbArray    = makeRGBArray(from: frame.capturedImage, H: H, W: W),
              let (bicubicArr, confArr) = makeDepthArrays(
                  depthMap: sceneDepth.depthMap,
                  confidenceMap: sceneDepth.confidenceMap,
                  H: H, W: W)
        else { return nil }

        guard let output = try? model.prediction(
            rgb: rgbArray,
            bicubic_norm: bicubicArr,
            conf_hi: confArr
        ) else { return nil }

        // Read pred_norm (1,1,H,W) — model weights are fp16 but CoreML typically
        // returns float32; handle float16 output as fallback.
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

        var dmax = depthMax, dmin = depthMin
        depths.withUnsafeMutableBufferPointer { ptr in
            let base = ptr.baseAddress!
            vDSP_vsmul(base, 1, &dmax, base, 1, vDSP_Length(count))
        }
        // clamp into a separate buffer to avoid Swift aliasing rule
        var clamped = depths
        depths.withUnsafeBufferPointer { src in
            clamped.withUnsafeMutableBufferPointer { dst in
                vDSP_vclip(src.baseAddress!, 1, &dmin, &dmax,
                           dst.baseAddress!, 1, vDSP_Length(count))
            }
        }

        return DepthProcessor.colorize(depths: clamped, width: W, height: H)
    }

    // MARK: - RGB preprocessing
    // capturedImage is landscape (native sensor orientation). Resize to W×H (landscape),
    // ImageNet-normalize, pack into CHW MLMultiArray.

    private static func makeRGBArray(from pixelBuffer: CVPixelBuffer, H: Int, W: Int) -> MLMultiArray? {
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
        let pixels = base.assumingMemoryBound(to: UInt8.self)

        guard let array = try? MLMultiArray(
            shape: [1, 3, NSNumber(value: H), NSNumber(value: W)],
            dataType: .float32
        ) else { return nil }

        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let (mr, mg, mb) = imagenetMean
        let (sr, sg, sb) = imagenetStd
        let rOff = 0, gOff = H * W, bOff = H * W * 2

        for y in 0..<H {
            let row = pixels.advanced(by: y * bpr)
            let yOff = y * W
            for x in 0..<W {
                let p = row.advanced(by: x * 4)
                // BGRA: p[0]=B  p[1]=G  p[2]=R  p[3]=A
                ptr[rOff + yOff + x] = (Float(p[2]) / 255.0 - mr) / sr
                ptr[gOff + yOff + x] = (Float(p[1]) / 255.0 - mg) / sg
                ptr[bOff + yOff + x] = (Float(p[0]) / 255.0 - mb) / sb
            }
        }
        return array
    }

    // MARK: - Depth + confidence preprocessing

    private static func makeDepthArrays(
        depthMap: CVPixelBuffer,
        confidenceMap: CVPixelBuffer?,
        H: Int, W: Int
    ) -> (MLMultiArray, MLMultiArray)? {

        // Read low-res depth (192×256 float32, landscape)
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        let lW = CVPixelBufferGetWidth(depthMap)
        let lH = CVPixelBufferGetHeight(depthMap)
        let bpr = CVPixelBufferGetBytesPerRow(depthMap)
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            return nil
        }
        var loDepth = [Float](repeating: 0, count: lH * lW)
        loDepth.withUnsafeMutableBufferPointer { dst in
            for row in 0..<lH {
                memcpy(dst.baseAddress!.advanced(by: row * lW),
                       base.advanced(by: row * bpr),
                       lW * MemoryLayout<Float>.size)
            }
        }
        CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)

        // Read confidence (192×256 uint8)
        var confLow = [UInt8](repeating: 0, count: lH * lW)
        if let cb = confidenceMap {
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

        // Zero out non-high-confidence pixels (spec: use only confidence==2)
        for i in 0..<lH * lW where confLow[i] < 2 { loDepth[i] = 0 }

        // Bicubic upsample depth from (lH×lW) to (H×W) using vImage PlanarF
        let bicubicFlat = vImageResampleFloat(loDepth, srcW: lW, srcH: lH, dstW: W, dstH: H)

        // bicubic_norm = clamp(bicubic, 0, depthMax) / depthMax
        let count = H * W
        var bicubicNorm = [Float](repeating: 0, count: count)
        var zero: Float = 0, dmax = depthMax
        var invMax: Float = 1.0 / depthMax
        bicubicFlat.withUnsafeBufferPointer { src in
            bicubicNorm.withUnsafeMutableBufferPointer { dst in
                vDSP_vclip(src.baseAddress!, 1, &zero, &dmax, dst.baseAddress!, 1, vDSP_Length(count))
            }
        }
        vDSP_vsmul(bicubicNorm, 1, &invMax, &bicubicNorm, 1, vDSP_Length(count))

        // conf_hi = bilinear upsample of (confLow==2) mask
        var confFloat = confLow.map { $0 == 2 ? Float(1) : Float(0) }
        let confHigh = vImageResampleFloat(confFloat, srcW: lW, srcH: lH, dstW: W, dstH: H)

        // Pack into MLMultiArrays
        guard let bicubicArr = try? MLMultiArray(
                shape: [1, 1, NSNumber(value: H), NSNumber(value: W)], dataType: .float32),
              let confArr = try? MLMultiArray(
                shape: [1, 1, NSNumber(value: H), NSNumber(value: W)], dataType: .float32)
        else { return nil }

        memcpy(bicubicArr.dataPointer, bicubicNorm, count * MemoryLayout<Float>.size)
        memcpy(confArr.dataPointer, confHigh, count * MemoryLayout<Float>.size)

        return (bicubicArr, confArr)
    }

    // MARK: - vImage float resampling (supports bicubic via kvImageHighQualityResampling)

    private static func vImageResampleFloat(
        _ src: [Float], srcW: Int, srcH: Int, dstW: Int, dstH: Int
    ) -> [Float] {
        var result = [Float](repeating: 0, count: dstH * dstW)
        src.withUnsafeBytes { srcRaw in
            result.withUnsafeMutableBytes { dstRaw in
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcRaw.baseAddress!),
                    height: vImagePixelCount(srcH), width: vImagePixelCount(srcW),
                    rowBytes: srcW * MemoryLayout<Float>.size)
                var dstBuf = vImage_Buffer(
                    data: dstRaw.baseAddress!,
                    height: vImagePixelCount(dstH), width: vImagePixelCount(dstW),
                    rowBytes: dstW * MemoryLayout<Float>.size)
                vImageScale_PlanarF(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageHighQualityResampling))
            }
        }
        return result
    }
}
