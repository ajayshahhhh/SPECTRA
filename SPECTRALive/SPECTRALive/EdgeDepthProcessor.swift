import ARKit
import Vision
import UIKit
import Accelerate

struct EdgeDepthResult {
    let overlayImage: UIImage          // depth-colored edges + object boxes, .right orientation
    let centerDistance: Float?
    let minDepth: Float?
    let maxDepth: Float?
    let detections: [EdgeDetection]    // for floating SwiftUI labels
}

struct EdgeDetection {
    let label: String                  // e.g. "Face", "Person"
    let depth: Float?                  // center depth in meters
    /// Normalized position in portrait display space (0,0=top-left, 1,1=bottom-right)
    let portraitCenter: CGPoint
}

enum EdgeDepthProcessor {

    // Drawing canvas in landscape (matches depth map orientation).
    // UIImage returned with .right orientation so it displays as portrait.
    nonisolated static let canvasW: CGFloat = 1024
    nonisolated static let canvasH: CGFloat = 768

    // MARK: - LiDAR entry point

    nonisolated static func process(frame: ARFrame) -> EdgeDepthResult? {
        guard let sceneDepth = frame.sceneDepth else { return nil }

        let (depths, dW, dH) = readFloatBuffer(sceneDepth.depthMap)
        let confs = readUInt8Buffer(sceneDepth.confidenceMap, count: dW * dH)

        var filteredDepths = depths
        for i in 0..<filteredDepths.count where confs[i] < 2 { filteredDepths[i] = 0 }

        return process(capturedImage: frame.capturedImage, depths: filteredDepths, dW: dW, dH: dH)
    }

    // MARK: - Core entry point (accepts any depth source)

    nonisolated static func process(
        capturedImage: CVPixelBuffer,
        depths: [Float], dW: Int, dH: Int,
        detectObjects: Bool = true
    ) -> EdgeDepthResult? {
        var minD: Float = .greatestFiniteMagnitude, maxD: Float = 0
        for d in depths where d > 0 && d.isFinite {
            if d < minD { minD = d }
            if d > maxD { maxD = d }
        }
        guard maxD > minD else { return nil }

        let cx = dW/2, cy = dH/2
        var distSum: Float = 0, distN = 0
        for dy in -2...2 { for dx in -2...2 {
            let x = cx+dx, y = cy+dy
            guard x >= 0, x < dW, y >= 0, y < dH else { continue }
            let d = depths[y*dW+x]
            guard d > 0 && d.isFinite else { continue }
            distSum += d; distN += 1
        }}
        let centerDist: Float? = distN > 0 ? distSum/Float(distN) : nil

        let handler = VNImageRequestHandler(cvPixelBuffer: capturedImage, options: [:])

        let contourReq = VNDetectContoursRequest()
        contourReq.contrastAdjustment = 2.0
        contourReq.detectsDarkOnLight = true
        contourReq.maximumImageDimension = 512

        var requests: [VNRequest] = [contourReq]
        var faceReq: VNDetectFaceRectanglesRequest?
        var humanReq: VNDetectHumanRectanglesRequest?
        if detectObjects {
            let fr = VNDetectFaceRectanglesRequest()
            let hr = VNDetectHumanRectanglesRequest()
            faceReq = fr
            humanReq = hr
            requests.append(contentsOf: [fr, hr])
        }

        try? handler.perform(requests)

        let contourObs = contourReq.results?.first as? VNContoursObservation
        let faces: [VNFaceObservation] = faceReq?.results ?? []
        let humans: [VNDetectedObjectObservation] = humanReq?.results ?? []

        var detections: [EdgeDetection] = []

        let size = CGSize(width: canvasW, height: canvasH)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false

        let overlay = UIGraphicsImageRenderer(size: size, format: format).image { ctx in
            let cgCtx = ctx.cgContext

            if let obs = contourObs {
                drawContours(obs, ctx: cgCtx, canvas: size,
                             depths: depths, dW: dW, dH: dH,
                             minD: minD, maxD: maxD)
            }

            for face in faces {
                if let det = drawDetectionBox(face.boundingBox, label: "Face", accentColor: .systemYellow,
                                              ctx: cgCtx, canvas: size,
                                              depths: depths, dW: dW, dH: dH) {
                    detections.append(det)
                }
            }

            for human in humans {
                if let det = drawDetectionBox(human.boundingBox, label: "Person", accentColor: .systemGreen,
                                              ctx: cgCtx, canvas: size,
                                              depths: depths, dW: dW, dH: dH) {
                    detections.append(det)
                }
            }
        }

        let portraitImage = UIImage(cgImage: overlay.cgImage!, scale: 1.0, orientation: .right)

        return EdgeDepthResult(overlayImage: portraitImage,
                               centerDistance: centerDist,
                               minDepth: minD, maxDepth: maxD,
                               detections: detections)
    }

    // MARK: - Contour drawing (depth-colored edges)

    private static func drawContours(
        _ obs: VNContoursObservation,
        ctx: CGContext,
        canvas: CGSize,
        depths: [Float], dW: Int, dH: Int,
        minD: Float, maxD: Float
    ) {
        ctx.setLineWidth(1.5)
        ctx.setLineCap(.round)

        // Keep only contours with enough points to be meaningful
        let all = obs.topLevelContours
        let contours = all.filter { $0.pointCount >= 12 }

        for contour in contours.prefix(200) {
            let pts = contour.normalizedPoints
            guard pts.count >= 2 else { continue }

            var prev: CGPoint? = nil

            // Sample every 2nd point for performance (~halves draw calls)
            for i in stride(from: 0, to: pts.count, by: 2) {
                let p = pts[i]
                // Vision landscape → canvas (top-left origin)
                let cx = CGFloat(p.x) * canvas.width
                let cy = (1.0 - CGFloat(p.y)) * canvas.height
                let cur = CGPoint(x: cx, y: cy)

                // Depth at this point in the LiDAR map
                let dX = min(dW-1, max(0, Int(p.x * Float(dW))))
                let dY = min(dH-1, max(0, Int((1-p.y) * Float(dH))))
                let depth = depths[dY * dW + dX]

                if depth > 0 && depth.isFinite, let prev {
                    let t = max(0, min(1, (depth - minD) / (maxD - minD)))
                    let (r, g, b) = DepthProcessor.lut[Int(t * 255)]
                    ctx.setStrokeColor(CGColor(red: CGFloat(r)/255,
                                               green: CGFloat(g)/255,
                                               blue: CGFloat(b)/255,
                                               alpha: 0.9))
                    ctx.move(to: prev)
                    ctx.addLine(to: cur)
                    ctx.strokePath()
                }
                // Break the chain at missing-depth boundaries so we don't draw
                // lines across unrelated scene regions
                prev = (depth > 0 && depth.isFinite) ? cur : nil
            }
        }
    }

    // MARK: - Object bounding box drawing

    @discardableResult
    private static func drawDetectionBox(
        _ visionBox: CGRect,       // Vision normalized, bottom-left origin
        label: String,
        accentColor: UIColor,
        ctx: CGContext,
        canvas: CGSize,
        depths: [Float], dW: Int, dH: Int
    ) -> EdgeDetection? {

        // Convert Vision rect (bottom-left) to canvas rect (top-left)
        let bx = visionBox.origin.x * canvas.width
        let by = (1.0 - visionBox.maxY) * canvas.height
        let bw = visionBox.width  * canvas.width
        let bh = visionBox.height * canvas.height
        let drawRect = CGRect(x: bx, y: by, width: bw, height: bh)

        // Depth at center of the box
        let vcx = Float(visionBox.midX), vcy = Float(visionBox.midY)
        let centerDX = min(dW-1, max(0, Int(vcx * Float(dW))))
        let centerDY = min(dH-1, max(0, Int((1-vcy) * Float(dH))))
        let centerDepth: Float? = {
            let d = depths[centerDY * dW + centerDX]
            return d > 0 && d.isFinite ? d : nil
        }()

        // ── Box outline ───────────────────────────────────────────────
        ctx.setLineWidth(2.0)
        ctx.setStrokeColor(accentColor.withAlphaComponent(0.9).cgColor)
        ctx.stroke(drawRect)

        // ── Corner dots colored by depth ──────────────────────────────
        let corners: [(Float, Float)] = [
            (Float(visionBox.minX), Float(visionBox.minY)),
            (Float(visionBox.maxX), Float(visionBox.minY)),
            (Float(visionBox.minX), Float(visionBox.maxY)),
            (Float(visionBox.maxX), Float(visionBox.maxY)),
        ]
        for (vx, vy) in corners {
            let edX = min(dW-1, max(0, Int(vx * Float(dW))))
            let edY = min(dH-1, max(0, Int((1-vy) * Float(dH))))
            let d   = depths[edY * dW + edX]
            guard d > 0 && d.isFinite else { continue }

            let t = max(0, min(1, (d - 0.5) / 9.5))
            let (r, g, b) = DepthProcessor.lut[Int(t * 255)]
            let dotX = CGFloat(vx) * canvas.width
            let dotY = (1.0 - CGFloat(vy)) * canvas.height

            // Colored depth dot at corner
            ctx.setFillColor(CGColor(red: CGFloat(r)/255, green: CGFloat(g)/255,
                                     blue: CGFloat(b)/255, alpha: 1.0))
            ctx.fillEllipse(in: CGRect(x: dotX-5, y: dotY-5, width: 10, height: 10))

            // Corner depth text
            let str = String(format: "%.1fm", d) as NSString
            let attrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.monospacedSystemFont(ofSize: 12, weight: .semibold),
                .foregroundColor: UIColor.white,
            ]
            str.draw(at: CGPoint(x: dotX + 7, y: dotY - 9), withAttributes: attrs)
        }

        // ── Label above box ───────────────────────────────────────────
        let depthSuffix = centerDepth.map { String(format: " %.1fm", $0) } ?? ""
        let labelStr = "\(label)\(depthSuffix)" as NSString
        let labelAttrs: [NSAttributedString.Key: Any] = [
            .font: UIFont.monospacedSystemFont(ofSize: 14, weight: .bold),
            .foregroundColor: UIColor.white,
        ]
        // Background pill for readability
        let textSize = labelStr.size(withAttributes: labelAttrs)
        let pillRect = CGRect(x: bx, y: by - textSize.height - 6,
                              width: textSize.width + 10, height: textSize.height + 4)
        ctx.setFillColor(accentColor.withAlphaComponent(0.75).cgColor)
        ctx.fillEllipse(in: CGRect(x: pillRect.minX, y: pillRect.minY,
                                   width: pillRect.height, height: pillRect.height))
        ctx.fill(pillRect.insetBy(dx: pillRect.height/2, dy: 0))
        ctx.fillEllipse(in: CGRect(x: pillRect.maxX-pillRect.height, y: pillRect.minY,
                                   width: pillRect.height, height: pillRect.height))
        labelStr.draw(at: CGPoint(x: pillRect.minX + 5, y: pillRect.minY + 2),
                      withAttributes: labelAttrs)

        // Build EdgeDetection for SwiftUI label overlay.
        // Portrait space: landscape (vx, vy) with bottom-left Vision origin →
        // portrait portrait (px, py) where the canvas is rotated 90° CW.
        // Landscape → portrait via .right rotation: px = 1-vy, py = vx
        let portraitX = 1.0 - CGFloat(vcy)
        let portraitY = CGFloat(vcx)

        return EdgeDetection(label: label,
                             depth: centerDepth,
                             portraitCenter: CGPoint(x: portraitX, y: portraitY))
    }

    // MARK: - Buffer helpers

    nonisolated private static func readFloatBuffer(_ buf: CVPixelBuffer) -> ([Float], Int, Int) {
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf)
        let bpr = CVPixelBufferGetBytesPerRow(buf)
        guard let base = CVPixelBufferGetBaseAddress(buf) else { return ([], w, h) }
        var out = [Float](repeating: 0, count: w * h)
        out.withUnsafeMutableBufferPointer { dst in
            for row in 0..<h {
                memcpy(dst.baseAddress!.advanced(by: row * w),
                       base.advanced(by: row * bpr), w * MemoryLayout<Float>.size)
            }
        }
        return (out, w, h)
    }

    nonisolated private static func readUInt8Buffer(_ buf: CVPixelBuffer?, count: Int) -> [UInt8] {
        guard let buf else { return [UInt8](repeating: 2, count: count) }
        CVPixelBufferLockBaseAddress(buf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
        let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf)
        let bpr = CVPixelBufferGetBytesPerRow(buf)
        guard let base = CVPixelBufferGetBaseAddress(buf) else { return [UInt8](repeating: 0, count: count) }
        var out = [UInt8](repeating: 0, count: w * h)
        out.withUnsafeMutableBufferPointer { dst in
            for row in 0..<h {
                memcpy(dst.baseAddress!.advanced(by: row * w),
                       base.advanced(by: row * bpr), w)
            }
        }
        return out
    }
}
