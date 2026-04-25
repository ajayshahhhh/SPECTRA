import SwiftUI
import RealityKit
import ARKit

struct ARViewContainer: UIViewRepresentable {
    let model: ARSessionModel

    func makeCoordinator() -> Coordinator {
        Coordinator(model: model)
    }

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        arView.session.delegate = context.coordinator
        arView.renderOptions = []

        let config = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics = .sceneDepth
        } else {
            model.captureMessage = "LiDAR not available on this device"
        }
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}

// MARK: - Coordinator

final class Coordinator: NSObject, ARSessionDelegate {
    // nonisolated(unsafe): written from ARKit thread, read on main for capture
    private let model: ARSessionModel
    nonisolated(unsafe) private var lastProcessTime: CFAbsoluteTime = 0
    private let processInterval: CFAbsoluteTime = 1.0 / 15.0

    init(model: ARSessionModel) {
        self.model = model
    }

    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        model.latestFrame = frame
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastProcessTime >= processInterval else { return }
        lastProcessTime = now
        guard let result = DepthProcessor.recolorCamera(frame: frame) else { return }
        let image = result.colorImage
        let trackingNormal = frame.camera.trackingState == .normal
        let dist = trackingNormal ? result.centerDistance : nil
        let minD = result.minDepth
        let maxD = result.maxDepth
        Task { @MainActor [model = self.model] in
            model.depthImage = image
            model.centerDistance = dist
            model.minDepth = minD
            model.maxDepth = maxD
        }
    }

    nonisolated func session(_ session: ARSession, didFailWithError error: Error) {
        let msg = error.localizedDescription
        Task { @MainActor [model = self.model] in
            model.captureMessage = "AR Error: \(msg)"
        }
    }
}
