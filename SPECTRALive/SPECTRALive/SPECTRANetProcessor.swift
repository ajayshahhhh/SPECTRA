import ARKit
import CoreImage
import UIKit

struct SPECTRANetResult {
    let depth: DepthResult
    let edge: EdgeDepthResult?
    let recoloredImage: UIImage?
}
import Compression

enum SPECTRANetProcessor {

    nonisolated static let modelH: Int = 768
    nonisolated static let modelW: Int = 1024

    nonisolated static let serverURL = URL(string: "ws://10.30.131.25:8000/ws")!

    nonisolated nonisolated private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    nonisolated private static let imagenetMean: (Float, Float, Float) = (0.485, 0.456, 0.406)
    nonisolated private static let imagenetStd:  (Float, Float, Float) = (0.229, 0.224, 0.225)

    nonisolated private static let ciContext = CIContext(options: [
        .useSoftwareRenderer: false,
        .workingColorSpace: CGColorSpaceCreateDeviceRGB()
    ])
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
    nonisolated private static let emaAlpha: Float = 0.3
    nonisolated(unsafe) private static var prevDepths: [Float]?

    nonisolated(unsafe) private static let sharedMLModel: MLModel? = {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        guard let url = Bundle.main.url(forResource: "spectranet_depth", withExtension: "mlmodelc") else { return nil }
        return try? MLModel(contentsOf: url, configuration: cfg)
    }()
    static let modelH: Int = 768
    static let modelW: Int = 1024

    // ← Set this to your GX10's local IP address before building
    static let serverURL = URL(string: "http://10.30.131.25:8000/infer")!

    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Main entry point

    nonisolated static func process(frame: ARFrame) -> SPECTRANetResult? {
        guard let sceneDepth = frame.sceneDepth,
              let model = sharedMLModel else { return nil }
    nonisolated static func process(frame: ARFrame) async -> DepthResult? {
        guard let sceneDepth = frame.sceneDepth else { return nil }

        guard let jpegData = makeRGBJPEG(from: frame.capturedImage, H: modelH, W: modelW) else { return nil }

        // Read depth + confidence at native LiDAR resolution
        let depthMap = sceneDepth.depthMap
    nonisolated static func process(frame: ARFrame) async -> DepthResult? {
        guard let sceneDepth = frame.sceneDepth else { return nil }

        guard
            let jpegData = makeRGBJPEG(from: frame.capturedImage, H: modelH, W: modelW),
            let (depthBytes, confBytes, lH, lW) = extractDepthConf(
                depthMap: sceneDepth.depthMap,
                confidenceMap: sceneDepth.confidenceMap)
        else { return nil }

        return await postInfer(jpeg: jpegData, depthBytes: depthBytes,
                               confBytes: confBytes, lH: lH, lW: lW)
    }

    // MARK: - RGB → JPEG

    private static func makeRGBJPEG(from pixelBuffer: CVPixelBuffer, H: Int, W: Int) -> Data? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        let sx = CGFloat(W) / ci.extent.width
        let sy = CGFloat(H) / ci.extent.height
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        guard let cgImg = ciContext.createCGImage(scaled, from: scaled.extent) else { return nil }
        return UIImage(cgImage: cgImg).jpegData(compressionQuality: 0.4)
    }

    // MARK: - Extract raw bytes from ARKit depth/confidence buffers

    private static func extractDepthConf(
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

    // MARK: - HTTP multipart POST → colorized JPEG + depth stats in headers

    private static func postInfer(
        jpeg: Data, depthBytes: Data, confBytes: Data, lH: Int, lW: Int
    ) async -> DepthResult? {
        let boundary = "SPECTRABoundary_\(UUID().uuidString.prefix(8))"
        var body = Data()

        func appendFile(_ name: String, _ data: Data, filename: String, mime: String) {
            body += "--\(boundary)\r\n".utf8Data
            body += "Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n".utf8Data
            body += "Content-Type: \(mime)\r\n\r\n".utf8Data
            body += data
            body += "\r\n".utf8Data
        }

        func appendField(_ name: String, _ value: String) {
            body += "--\(boundary)\r\n".utf8Data
            body += "Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".utf8Data
            body += value.utf8Data
            body += "\r\n".utf8Data
        }

        appendFile("rgb",   jpeg,       filename: "rgb.jpg",   mime: "image/jpeg")
        appendFile("depth", depthBytes, filename: "depth.bin", mime: "application/octet-stream")
        appendFile("conf",  confBytes,  filename: "conf.bin",  mime: "application/octet-stream")
        appendField("lH", "\(lH)")
        appendField("lW", "\(lW)")
        body += "--\(boundary)--\r\n".utf8Data

        var request = URLRequest(url: serverURL, timeoutInterval: 10)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)",
                         forHTTPHeaderField: "Content-Type")
        request.httpBody = body

        guard let (data, response) = try? await URLSession.shared.data(for: request),
              let http = response as? HTTPURLResponse,
              http.statusCode == 200,
              let cgImg = UIImage(data: data)?.cgImage
        guard let (depthBytes, confBytes, lH, lW) = extractDepthConf(
            depthMap: sceneDepth.depthMap,
            confidenceMap: sceneDepth.confidenceMap)
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
        // Server returns a landscape JPEG (1024×768); apply .right to display as portrait
        let colorImage = UIImage(cgImage: cgImg, scale: 1.0, orientation: .right)

        let center = http.value(forHTTPHeaderField: "X-Center-Distance").flatMap(Float.init)
        let minD   = Float(http.value(forHTTPHeaderField: "X-Min-Depth")  ?? "") ?? 0.5
        let maxD   = Float(http.value(forHTTPHeaderField: "X-Max-Depth")  ?? "") ?? 10.0

        return DepthResult(colorImage: colorImage, centerDistance: center,
                           minDepth: minD, maxDepth: maxD)
    }
}

// MARK: - Helpers

private extension String {
    var utf8Data: Data { Data(utf8) }
}
