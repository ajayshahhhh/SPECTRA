import ARKit
import CoreImage
import UIKit
import Compression

enum SPECTRANetProcessor {

    // Half the training resolution — 4× fewer pixels, ~3–4× faster on Neural Engine
    static let modelH: Int = 384
    static let modelW: Int = 512

    static let serverURL = URL(string: "ws://10.30.131.25:8000/ws")!

    nonisolated private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // Persistent WebSocket — opened once, reused for every frame
    private static var socket: URLSessionWebSocketTask? = nil
    private static let session = URLSession(configuration: .default)

    private static func getSocket() -> URLSessionWebSocketTask {
        if let s = socket, s.state == .running { return s }
        let s = session.webSocketTask(with: serverURL)
        s.resume()
        socket = s
        return s
    }

    // MARK: - Main entry point

    nonisolated static func process(frame: ARFrame) async -> DepthResult? {
        guard let sceneDepth = frame.sceneDepth else { return nil }

        guard let jpegData = makeRGBJPEG(from: frame.capturedImage, H: modelH, W: modelW) else { return nil }

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
        guard let (depthBytes, confBytes, lH, lW) = extractDepthConf(
            depthMap: sceneDepth.depthMap,
            confidenceMap: sceneDepth.confidenceMap)
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

    // MARK: - RGB → JPEG

    nonisolated private static func makeRGBJPEG(from pixelBuffer: CVPixelBuffer, H: Int, W: Int) -> Data? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        let sx = CGFloat(W) / ci.extent.width
        let sy = CGFloat(H) / ci.extent.height
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        guard let cgImg = ciContext.createCGImage(scaled, from: scaled.extent) else { return nil }
        return UIImage(cgImage: cgImg).jpegData(compressionQuality: 0.4)
    }

    // MARK: - Extract raw bytes from ARKit depth/confidence buffers

    nonisolated private static func extractDepthConf(
        depthMap: CVPixelBuffer,
        confidenceMap: CVPixelBuffer?
    ) -> (depthBytes: Data, confBytes: Data, lH: Int, lW: Int)? {

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        let lW  = CVPixelBufferGetWidth(depthMap)
        let lH  = CVPixelBufferGetHeight(depthMap)
        let bpr = CVPixelBufferGetBytesPerRow(depthMap)
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            return nil
        }
        var depthFlat = [Float](repeating: 0, count: lH * lW)
        depthFlat.withUnsafeMutableBufferPointer { dst in
            for row in 0..<lH {
                memcpy(dst.baseAddress!.advanced(by: row * lW),
                       base.advanced(by: row * bpr),
                       lW * MemoryLayout<Float>.size)
            }
        }
        CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)

        var confFlat = [UInt8](repeating: 0, count: lH * lW)
        if let cb = confidenceMap {
            CVPixelBufferLockBaseAddress(cb, .readOnly)
            let cbpr = CVPixelBufferGetBytesPerRow(cb)
            if let ca = CVPixelBufferGetBaseAddress(cb) {
                confFlat.withUnsafeMutableBufferPointer { dst in
                    for row in 0..<lH {
                        memcpy(dst.baseAddress!.advanced(by: row * lW),
                               ca.advanced(by: row * cbpr), lW)
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(cb, .readOnly)
        }

        let depthBytes = depthFlat.withUnsafeBufferPointer { Data(buffer: $0) }
        let confBytes  = confFlat.withUnsafeBufferPointer { Data(buffer: $0) }
        return (depthBytes, confBytes, lH, lW)
    }

    // MARK: - zlib compression using Apple's Compression framework

    nonisolated private static func compress(_ data: Data) -> Data? {
        data.withUnsafeBytes { src -> Data? in
            guard let base = src.baseAddress else { return nil }
            let bound = compression_encode_scratch_buffer_size(COMPRESSION_ZLIB)
            let scratch = UnsafeMutablePointer<UInt8>.allocate(capacity: bound)
            defer { scratch.deallocate() }
            let dstSize = data.count + 1024
            let dst = UnsafeMutablePointer<UInt8>.allocate(capacity: dstSize)
            defer { dst.deallocate() }
            let n = compression_encode_buffer(
                dst, dstSize,
                base.assumingMemoryBound(to: UInt8.self), data.count,
                scratch, COMPRESSION_ZLIB)
            guard n > 0 else { return nil }
            return Data(bytes: dst, count: n)
        }
    }

    // MARK: - WebSocket send/receive

    // Binary frame layout sent to server:
    //   [4B uint32 lH][4B uint32 lW][4B uint32 jpeg_len]
    //   [jpeg_bytes][zlib_depth_bytes][zlib_conf_bytes]
    //
    // Binary reply from server:
    //   [4B uint32 jpeg_len][4B float32 center][4B float32 min_d][4B float32 max_d]
    //   [jpeg_bytes]

    private static func sendFrame(
        jpeg: Data, depthZ: Data, confZ: Data, lH: Int, lW: Int
    ) async -> DepthResult? {
        var msg = Data()

        func appendUInt32(_ v: UInt32) {
            var b = v.bigEndian
            msg.append(contentsOf: withUnsafeBytes(of: &b) { Array($0) })
        }

        appendUInt32(UInt32(lH))
        appendUInt32(UInt32(lW))
        appendUInt32(UInt32(jpeg.count))
        msg += jpeg
        msg += depthZ
        msg += confZ

        let ws = getSocket()
        guard (try? await ws.send(.data(msg))) != nil else { return nil }

        guard case .data(let reply) = try? await ws.receive(),
              reply.count > 16
        else { return nil }

        let jpegLen = reply.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self).bigEndian }
        let center  = reply.withUnsafeBytes { Float(bitPattern: $0.load(fromByteOffset: 4,  as: UInt32.self).bigEndian) }
        let minD    = reply.withUnsafeBytes { Float(bitPattern: $0.load(fromByteOffset: 8,  as: UInt32.self).bigEndian) }
        let maxD    = reply.withUnsafeBytes { Float(bitPattern: $0.load(fromByteOffset: 12, as: UInt32.self).bigEndian) }

        let jpegData = reply.subdata(in: 16..<(16 + Int(jpegLen)))
        guard let cgImg = UIImage(data: jpegData)?.cgImage else { return nil }
        let colorImage = UIImage(cgImage: cgImg, scale: 1.0, orientation: .right)

        return DepthResult(colorImage: colorImage,
                           centerDistance: center,
                           minDepth: minD,
                           maxDepth: maxD)
    }
}
