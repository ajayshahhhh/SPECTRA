import ARKit
import CoreImage
import UIKit

enum SPECTRANetProcessor {

    static let modelH: Int = 256
    static let modelW: Int = 320

    // ← Set this to your GX10's local IP address before building
    static let serverURL = URL(string: "http://10.30.131.25:8000/infer")!

    nonisolated private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Main entry point

    nonisolated static func process(frame: ARFrame) async -> DepthResult? {
        guard let sceneDepth = frame.sceneDepth else { return nil }

        // Extract async properties first
        let capturedImage = frame.capturedImage
        let depthMap = sceneDepth.depthMap
        let confidenceMap = sceneDepth.confidenceMap

        guard
            let jpegData = makeRGBJPEG(from: capturedImage, H: modelH, W: modelW),
            let (depthBytes, confBytes, lH, lW) = extractDepthConf(
                depthMap: depthMap,
                confidenceMap: confidenceMap)
        else { return nil }

        return await postInfer(jpeg: jpegData, depthBytes: depthBytes,
                               confBytes: confBytes, lH: lH, lW: lW)
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
}

// MARK: - Helpers

private extension String {
    var utf8Data: Data { Data(utf8) }
}
