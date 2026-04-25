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
    nonisolated(unsafe) private let model: ARSessionModel

    init(model: ARSessionModel) {
        self.model = model
    }

    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        model.latestFrame = frame
        guard let result = DepthProcessor.process(frame: frame) else { return }
        let image = result.colorImage
        let dist = result.centerDistance
        Task { @MainActor [model = self.model] in
            model.depthImage = image
            model.centerDistance = dist
        }
    }

    nonisolated func session(_ session: ARSession, didFailWithError error: Error) {
        let msg = error.localizedDescription
        Task { @MainActor [model = self.model] in
            model.captureMessage = "AR Error: \(msg)"
        }
    }
}
