import ARKit
import CoreImage
import UIKit

// ZeticMLange only available on real devices (not simulator)
#if !targetEnvironment(simulator)
import ZeticMLange
#endif

enum SPECTRANetBackend {
    case gx10Server
    #if !targetEnvironment(simulator)
    case zeticMLange
    #endif
}

enum SPECTRANetProcessor {

    // GX10 server accepts lower resolution for faster transfer
    static let gx10_H: Int = 256
    static let gx10_W: Int = 320

    // ZeticMLange model requires full resolution (trained at 768×1024)
    static let zeticH: Int = 768
    static let zeticW: Int = 1024

    // ImageNet normalization constants (from SPECTRA training)
    static let IMAGENET_MEAN: [Float] = [0.485, 0.456, 0.406]
    static let IMAGENET_STD: [Float] = [0.229, 0.224, 0.225]

    // Depth normalization constants (matching server)
    static let DEPTH_MAX: Float = 10.0
    static let DEPTH_MIN: Float = 0.5

    // ← Set this to your GX10's local IP address before building
    static let serverURL = URL(string: "http://10.30.131.25:8000/infer")!

    // ZeticMLange model (lazy loaded, device-only)
    #if !targetEnvironment(simulator)
    nonisolated(unsafe) private static var zeticModel: ZeticMLangeModel?
    nonisolated(unsafe) private static var zeticModelLoading = false
    #endif

    nonisolated private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Main entry point

    nonisolated static func process(frame: ARFrame, backend: SPECTRANetBackend = .gx10Server) async -> DepthResult? {
        guard let sceneDepth = frame.sceneDepth else { return nil }

        // Extract async properties first
        let capturedImage = frame.capturedImage
        let depthMap = sceneDepth.depthMap
        let confidenceMap = sceneDepth.confidenceMap

        switch backend {
        case .gx10Server:
            guard
                let jpegData = makeRGBJPEG(from: capturedImage, H: gx10_H, W: gx10_W),
                let (depthBytes, confBytes, lH, lW) = extractDepthConf(
                    depthMap: depthMap,
                    confidenceMap: confidenceMap)
            else { return nil }
            return await postInfer(jpeg: jpegData, depthBytes: depthBytes,
                                   confBytes: confBytes, lH: lH, lW: lW)
        #if !targetEnvironment(simulator)
        case .zeticMLange:
            // Zetic needs full resolution (768×1024)
            guard
                let jpegData = makeRGBJPEG(from: capturedImage, H: zeticH, W: zeticW),
                let (depthBytes, confBytes, lH, lW) = extractDepthConf(
                    depthMap: depthMap,
                    confidenceMap: confidenceMap)
            else { return nil }
            return await zeticInfer(capturedImage: capturedImage, jpegData: jpegData,
                                    depthBytes: depthBytes, confBytes: confBytes,
                                    lH: lH, lW: lW)
        #endif
        }
    }

    // MARK: - RGB → JPEG

    nonisolated private static func makeRGBJPEG(from pixelBuffer: CVPixelBuffer, H: Int, W: Int) -> Data? {
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        let sx = CGFloat(W) / ci.extent.width
        let sy = CGFloat(H) / ci.extent.height
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        guard let cgImg = ciContext.createCGImage(scaled, from: scaled.extent) else { return nil }
        return UIImage(cgImage: cgImg).jpegData(compressionQuality: 0.85)
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

    // MARK: - Extract RGB tensor with ImageNet normalization

    nonisolated private static func extractRGBTensor(from cgImage: CGImage) -> Data? {
        let width = cgImage.width
        let height = cgImage.height

        // Create bitmap context to extract pixels (RGBA format)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixels = [UInt8](repeating: 0, count: width * height * 4)

        guard let context = CGContext(
            data: &pixels,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert RGBA uint8 → RGB float32 [CHW layout]
        var rgbArray = [Float](repeating: 0, count: 3 * height * width)

        for c in 0..<3 {  // R, G, B channels
            let channelOffset = c * height * width
            for y in 0..<height {
                for x in 0..<width {
                    let pixelIdx = (y * width + x) * 4
                    let rgbIdx = channelOffset + y * width + x

                    // uint8 [0,255] → float32 [0,1]
                    let pixelValue = Float(pixels[pixelIdx + c]) / 255.0

                    // Apply ImageNet normalization: (x - mean) / std
                    rgbArray[rgbIdx] = (pixelValue - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
                }
            }
        }

        return rgbArray.withUnsafeBytes { Data($0) }
    }

    // MARK: - Upsample depth/confidence to target resolution

    nonisolated private static func upsampleDepthConf(
        depthBytes: Data, confBytes: Data, srcH: Int, srcW: Int, dstH: Int, dstW: Int
    ) -> (depthData: Data, confData: Data)? {
        // Simple bilinear upsampling
        let srcDepth = depthBytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let srcConf = confBytes.withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) }

        var dstDepth = [Float](repeating: 0, count: dstH * dstW)
        var dstConf = [Float](repeating: 0, count: dstH * dstW)  // Float32, not uint8!

        let scaleY = Float(srcH) / Float(dstH)
        let scaleX = Float(srcW) / Float(dstW)

        for dy in 0..<dstH {
            for dx in 0..<dstW {
                let sy = Float(dy) * scaleY
                let sx = Float(dx) * scaleX

                let sy0 = Int(sy)
                let sx0 = Int(sx)
                let sy1 = min(sy0 + 1, srcH - 1)
                let sx1 = min(sx0 + 1, srcW - 1)

                let fy = sy - Float(sy0)
                let fx = sx - Float(sx0)

                // Bilinear interpolation for depth
                let d00 = srcDepth[sy0 * srcW + sx0]
                let d01 = srcDepth[sy0 * srcW + sx1]
                let d10 = srcDepth[sy1 * srcW + sx0]
                let d11 = srcDepth[sy1 * srcW + sx1]

                let d0 = d00 * (1 - fx) + d01 * fx
                let d1 = d10 * (1 - fx) + d11 * fx
                let depth = d0 * (1 - fy) + d1 * fy
                // Normalize depth: (depth / DEPTH_MAX).clamp(0, 1)
                dstDepth[dy * dstW + dx] = min(max(depth / DEPTH_MAX, 0.0), 1.0)

                // Binary confidence mask: 1.0 if confidence >= 2, else 0.0
                let srcIdx = sy0 * srcW + sx0
                dstConf[dy * dstW + dx] = srcConf[srcIdx] >= 2 ? 1.0 : 0.0
            }
        }

        let depthData = dstDepth.withUnsafeBytes { Data($0) }
        let confData = dstConf.withUnsafeBytes { Data($0) }
        return (depthData, confData)
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
        else { return nil }

        // Server returns a landscape JPEG (1024×768); apply .right to display as portrait
        let colorImage = UIImage(cgImage: cgImg, scale: 1.0, orientation: .right)

        let center = http.value(forHTTPHeaderField: "X-Center-Distance").flatMap(Float.init)
        let minD   = Float(http.value(forHTTPHeaderField: "X-Min-Depth")  ?? "") ?? 0.5
        let maxD   = Float(http.value(forHTTPHeaderField: "X-Max-Depth")  ?? "") ?? 10.0

        return DepthResult(colorImage: colorImage, centerDistance: center,
                           minDepth: minD, maxDepth: maxD)
    }

    // MARK: - ZeticMLange on-device inference (device-only)

    #if !targetEnvironment(simulator)
    private static func zeticInfer(
        capturedImage: CVPixelBuffer,
        jpegData: Data,
        depthBytes: Data,
        confBytes: Data,
        lH: Int,
        lW: Int
    ) async -> DepthResult? {
        // Load model if needed
        if zeticModel == nil && !zeticModelLoading {
            zeticModelLoading = true
            do {
                zeticModel = try ZeticMLangeModel(
                    personalKey: "dev_af0d70e3dfe742c3bb6ece4b540f4919",
                    name: "Linjfeng/SPECTRA",
                    version: 1,
                    modelMode: ModelMode.RUN_AUTO,
                    onDownload: { progress in
                        print("[Zetic] Download progress: \(Int(progress * 100))%")
                    }
                )
                print("[Zetic] Model loaded successfully")
            } catch {
                print("[Zetic] Failed to load model: \(error)")
                zeticModelLoading = false
                return nil
            }
            zeticModelLoading = false
        }

        guard let model = zeticModel else {
            print("[Zetic] Model not available")
            return nil
        }

        // Prepare input tensors
        do {
            // Extract RGB tensor with ImageNet normalization
            guard let uiImage = UIImage(data: jpegData),
                  let cgImage = uiImage.cgImage,
                  let rgbData = extractRGBTensor(from: cgImage) else {
                print("[Zetic] Failed to extract RGB tensor")
                return nil
            }

            // Upsample depth/confidence to match RGB resolution
            guard let (upsampledDepth, upsampledConf) = upsampleDepthConf(
                depthBytes: depthBytes, confBytes: confBytes,
                srcH: lH, srcW: lW, dstH: zeticH, dstW: zeticW
            ) else {
                print("[Zetic] Failed to upsample depth/confidence")
                return nil
            }

            print("[Zetic] Input tensors: RGB[\(zeticH)×\(zeticW)], Depth[\(lH)×\(lW)→\(zeticH)×\(zeticW)]")

            // Model expects: rgb, bicubic_depth, conf_hi (in that order)
            let inputs: [Tensor] = [
                // 1. RGB tensor (768×1024)
                Tensor(data: rgbData, dataType: BuiltinDataType.float32, shape: [1, 3, zeticH, zeticW]),
                // 2. Depth tensor (upsampled to 768×1024, normalized)
                Tensor(data: upsampledDepth, dataType: BuiltinDataType.float32, shape: [1, 1, zeticH, zeticW]),
                // 3. Binary confidence mask (1.0 where confidence >= 2, else 0.0)
                Tensor(data: upsampledConf, dataType: BuiltinDataType.float32, shape: [1, 1, zeticH, zeticW])
            ]

            // Run inference with correct label
            let outputs = try model.run(inputs: inputs)

            // Parse output (assuming output is a colorized depth map)
            guard let outputTensor = outputs.first else { return nil }

            // 1. Validate shape: expecting [1, 1, H, W]
            guard outputTensor.shape.count == 4,
                  outputTensor.shape[0] == 1,
                  outputTensor.shape[1] == 1
            else {
                print("[Zetic] Unexpected output shape: \(outputTensor.shape)")
                return nil
            }

            let height = outputTensor.shape[2]
            let width = outputTensor.shape[3]
            let expectedCount = height * width

            print("[Zetic] Output shape: [1, 1, \(height), \(width)]")

            // 2. Extract float array from tensor data
            let floatArray = DataUtils.dataToFloatArray(outputTensor.data)

            guard floatArray.count == expectedCount else {
                print("[Zetic] Data size mismatch: got \(floatArray.count), expected \(expectedCount)")
                return nil
            }

            // 3. Denormalize: [0, 1] → meters (matching server behavior)
            var depthsInMeters = floatArray.map { normalized in
                let meters = normalized * DEPTH_MAX
                return (meters >= DEPTH_MIN && meters.isFinite) ? meters : Float.nan
            }

            // 4. Colorize using existing pipeline (reuses DepthProcessor.colorize)
            guard let result = DepthProcessor.colorize(
                depths: depthsInMeters,
                width: width,
                height: height
            ) else {
                print("[Zetic] Colorization failed")
                return nil
            }

            print("[Zetic] Success - center: \(result.centerDistance?.description ?? "nil")m, " +
                  "range: [\(result.minDepth), \(result.maxDepth)]m")

            return result

        } catch {
            print("[Zetic] Inference failed: \(error)")
            return nil
        }
    }
    #endif
}

// MARK: - Helpers

private extension String {
    var utf8Data: Data { Data(utf8) }
}
