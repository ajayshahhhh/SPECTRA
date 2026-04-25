import SwiftUI
import Combine
import ARKit
import RealityKit

enum DemoDepthMode: String, CaseIterable {
    case liveDepth = "Live Depth"
    case spectraNet = "SPECTRANet"
}

// MARK: - Session model

@MainActor
final class DemoSessionModel: ObservableObject {
    @Published var depthImage: UIImage?
    @Published var centerDistance: Float?
    @Published var minDepth: Float?
    @Published var maxDepth: Float?
    @Published var lastInferenceMs: Int?
    @Published var depthMode: DemoDepthMode = .liveDepth
    nonisolated(unsafe) var latestFrame: ARFrame?
}

// MARK: - AR container (drives camera display + depth processing)

struct DemoARViewContainer: UIViewRepresentable {
    let model: DemoSessionModel

    func makeCoordinator() -> DemoCoordinator { DemoCoordinator(model: model) }

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        arView.session.delegate = context.coordinator
        arView.renderOptions = []
        let config = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics = .sceneDepth
        }
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        context.coordinator.depthMode = model.depthMode
    }
}

final class DemoCoordinator: NSObject, ARSessionDelegate {
    private let model: DemoSessionModel
    nonisolated(unsafe) var depthMode: DemoDepthMode = .liveDepth
    nonisolated(unsafe) private var lastProcessTime: CFAbsoluteTime = 0
    nonisolated(unsafe) private var processing = false

    init(model: DemoSessionModel) { self.model = model }

    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        model.latestFrame = frame
        let now = CFAbsoluteTimeGetCurrent()
        let interval: CFAbsoluteTime = depthMode == .liveDepth ? 1.0 / 15.0 : 0.5
        guard now - lastProcessTime >= interval else { return }
        guard !processing else { return }
        processing = true
        lastProcessTime = now

        let mode = depthMode
        let trackingNormal = frame.camera.trackingState == .normal
        let t0 = now

        Task.detached(priority: .userInitiated) { [model = self.model, coordinator = self] in
            var depthImg: UIImage?
            var centerDist: Float?
            var minD: Float?
            var maxD: Float?

            switch mode {
            case .liveDepth:
                if let result = DepthProcessor.recolorCamera(frame: frame) {
                    if let cg = result.colorImage.cgImage {
                        depthImg = UIImage(cgImage: cg, scale: 1.0, orientation: .up)
                    }
                    centerDist = trackingNormal ? result.centerDistance : nil
                    minD = result.minDepth
                    maxD = result.maxDepth
                }
            case .spectraNet:
                if let result = SPECTRANetProcessor.process(frame: frame) {
                    if let recolored = result.recoloredImage, let cg = recolored.cgImage {
                        depthImg = UIImage(cgImage: cg, scale: 1.0, orientation: .up)
                    }
                    centerDist = trackingNormal ? result.depth.centerDistance : nil
                    minD = result.depth.minDepth
                    maxD = result.depth.maxDepth
                }
            }

            let ms = Int((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            coordinator.processing = false

            let finalImg = depthImg
            let finalDist = centerDist
            let finalMin = minD
            let finalMax = maxD
            await MainActor.run {
                model.depthImage = finalImg
                model.centerDistance = finalDist
                model.minDepth = finalMin
                model.maxDepth = finalMax
                model.lastInferenceMs = ms
            }
        }
    }

    nonisolated func session(_ session: ARSession, didFailWithError error: Error) {}
}

// MARK: - Demo View (landscape two-pane)

struct DemoView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var model = DemoSessionModel()

    var body: some View {
        GeometryReader { geo in
            ZStack {
                HStack(spacing: 0) {
                    // Left pane: raw camera feed
                    DemoARViewContainer(model: model)
                        .frame(width: geo.size.width / 2)
                        .clipped()

                    Rectangle()
                        .fill(.white.opacity(0.4))
                        .frame(width: 2)

                    // Right pane: depth colormap
                    ZStack {
                        Color.black
                        if let img = model.depthImage {
                            Image(uiImage: img)
                                .resizable()
                                .scaledToFill()
                                .frame(width: geo.size.width / 2 - 2, height: geo.size.height)
                                .clipped()
                        }
                    }
                    .frame(width: geo.size.width / 2 - 2, height: geo.size.height)
                    .clipped()
                }

                // HUD overlay
                VStack(spacing: 0) {
                    // Top bar: pane labels + branding
                    HStack {
                        paneLabel("CAMERA", icon: "camera.fill")
                        Spacer()
                        Text("SPECTRA")
                            .font(.system(size: 14, weight: .bold, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.5))
                        Spacer()
                        paneLabel(
                            model.depthMode == .liveDepth ? "LiDAR DEPTH" : "SPECTRANet",
                            icon: model.depthMode == .liveDepth ? "scope" : "brain"
                        )
                    }
                    .padding(.horizontal, 12)
                    .padding(.top, 12)

                    Spacer()

                    // Bottom: distance + controls
                    VStack(spacing: 8) {
                        if let dist = model.centerDistance {
                            Text(String(format: "%.3f m", dist))
                                .font(.system(size: 18, weight: .bold, design: .monospaced))
                                .foregroundStyle(.white)
                                .shadow(color: .black.opacity(0.8), radius: 3)
                        }

                        HStack(spacing: 16) {
                            Button { dismiss() } label: {
                                Image(systemName: "xmark")
                                    .font(.system(size: 16, weight: .semibold))
                                    .foregroundStyle(.white)
                                    .frame(width: 36, height: 36)
                                    .background(.white.opacity(0.2), in: Circle())
                            }

                            Picker("Mode", selection: $model.depthMode) {
                                ForEach(DemoDepthMode.allCases, id: \.self) {
                                    Text($0.rawValue).tag($0)
                                }
                            }
                            .pickerStyle(.segmented)
                            .frame(width: 220)

                            if let ms = model.lastInferenceMs {
                                Text("\(ms)ms")
                                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.7))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 3)
                                    .background(.black.opacity(0.4), in: Capsule())
                            } else {
                                Spacer().frame(width: 50)
                            }
                        }
                    }
                    .padding(.bottom, 16)
                }

                // Color legend (right pane, bottom-right)
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        colorLegend
                            .padding(.trailing, 8)
                            .padding(.bottom, 80)
                    }
                }
            }
        }
        .ignoresSafeArea()
        .statusBarHidden()
        .onAppear {
            AppDelegate.orientationLock = .landscape
            if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene {
                scene.requestGeometryUpdate(.iOS(interfaceOrientations: .landscape))
                scene.windows.first?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
            }
        }
        .onDisappear {
            AppDelegate.orientationLock = .all
            if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene {
                scene.requestGeometryUpdate(.iOS(interfaceOrientations: .all))
                scene.windows.first?.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations()
            }
        }
    }

    // MARK: - Components

    private func paneLabel(_ text: String, icon: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))
            Text(text)
                .font(.system(size: 10, weight: .bold, design: .monospaced))
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.black.opacity(0.5), in: Capsule())
    }

    private var colorLegend: some View {
        let barH: CGFloat = 80
        return HStack(alignment: .center, spacing: 3) {
            VStack(alignment: .trailing, spacing: 0) {
                Text("near"); Spacer(); Text("far")
            }
            .font(.system(size: 8, weight: .medium, design: .monospaced))
            .foregroundStyle(.white.opacity(0.7))
            .frame(height: barH)

            LinearGradient(
                colors: [.red, .yellow, .green, .cyan, .blue],
                startPoint: .top, endPoint: .bottom
            )
            .frame(width: 8, height: barH)
            .clipShape(RoundedRectangle(cornerRadius: 3))
            .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(.white.opacity(0.4), lineWidth: 1))
        }
        .shadow(color: .black.opacity(0.6), radius: 2)
    }
}
