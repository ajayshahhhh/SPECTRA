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

    // Backend selection now comes from AppStorage
    var spectraNetBackend: SPECTRANetBackend {
        let useZetic: Bool
        if UserDefaults.standard.object(forKey: "useZetic") == nil {
            useZetic = true  // Default to Zetic
        } else {
            useZetic = UserDefaults.standard.bool(forKey: "useZetic")
        }
        #if !targetEnvironment(simulator)
        return useZetic ? .zeticMLange : .gx10Server
        #else
        return .gx10Server
        #endif
    }
}

// MARK: - AR container (drives camera display + depth processing)

struct DemoARViewContainer: UIViewRepresentable {
    let model: DemoSessionModel
    let depthMode: DemoDepthMode

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
        context.coordinator.depthMode = depthMode
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        print("[DemoToggle] updateUIView called — container.depthMode=\(depthMode), coordinator.depthMode=\(context.coordinator.depthMode)")
        if context.coordinator.depthMode != depthMode {
            print("[DemoToggle] → mode CHANGED, updating coordinator to \(depthMode)")
            context.coordinator.depthMode = depthMode
            context.coordinator.resetProcessing()
        }
        // Backend is now read dynamically from UserDefaults in the session method
    }
}

final class DemoCoordinator: NSObject, ARSessionDelegate {
    private let model: DemoSessionModel
    nonisolated(unsafe) var depthMode: DemoDepthMode = .liveDepth
    nonisolated(unsafe) private var lastProcessTime: CFAbsoluteTime = 0
    nonisolated(unsafe) private var processing = false
    nonisolated(unsafe) private var lastMode: DemoDepthMode = .liveDepth

    init(model: DemoSessionModel) { self.model = model }

    // Read backend from UserDefaults
    nonisolated private var spectraNetBackend: SPECTRANetBackend {
        let useZetic: Bool
        if UserDefaults.standard.object(forKey: "useZetic") == nil {
            useZetic = true  // Default to Zetic
        } else {
            useZetic = UserDefaults.standard.bool(forKey: "useZetic")
        }
        #if !targetEnvironment(simulator)
        return useZetic ? .zeticMLange : .gx10Server
        #else
        return .gx10Server
        #endif
    }

    nonisolated func resetProcessing() {
        lastProcessTime = 0
        processing = false
    }

    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        model.latestFrame = frame

        // Clear image immediately when mode changes
        if depthMode != lastMode {
            print("DemoCoordinator: Mode changed detected: \(lastMode) -> \(depthMode)")
            lastMode = depthMode
            Task { @MainActor [model = self.model] in
                model.depthImage = nil
                model.centerDistance = nil
            }
        }

        let now = CFAbsoluteTimeGetCurrent()
        let interval: CFAbsoluteTime = 1.0 / 60.0
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

            print("[DemoToggle] Processing frame with mode=\(mode), coordinator.depthMode=\(coordinator.depthMode)")
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
                let backend = coordinator.spectraNetBackend
                if let result = await SPECTRANetProcessor.process(frame: frame, backend: backend) {
                    let blended = DepthProcessor.blendHeatmapWithCamera(
                        heatmap: result.colorImage,
                        capturedImage: frame.capturedImage
                    )
                    print("[DemoView] Blending result: \(blended != nil ? "✓ success" : "✗ failed, using raw heatmap")")
                    let source = blended ?? result.colorImage
                    if let cg = source.cgImage {
                        depthImg = UIImage(cgImage: cg, scale: 1.0, orientation: .up)
                        print("[DemoView] Final image: \(cg.width)×\(cg.height)")
                    } else {
                        print("[DemoView] ✗ WARNING: source has no CGImage!")
                    }
                    centerDist = trackingNormal ? result.centerDistance : nil
                    minD = result.minDepth
                    maxD = result.maxDepth
                }
            }

            let ms = Int((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            coordinator.processing = false

            // Only update if mode hasn't changed since processing started
            guard coordinator.depthMode == mode else { return }

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
                    DemoARViewContainer(model: model, depthMode: model.depthMode)
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

                // Crosshairs for both panes
                GeometryReader { geo in
                    HStack(spacing: 0) {
                        // Left pane crosshair
                        crosshair
                            .frame(width: geo.size.width / 2, height: geo.size.height)

                        Rectangle()
                            .fill(.clear)
                            .frame(width: 2)

                        // Right pane crosshair
                        crosshair
                            .frame(width: geo.size.width / 2 - 2, height: geo.size.height)
                    }
                }
                .allowsHitTesting(false)

                // HUD overlay
                VStack(spacing: 0) {
                    // Top bar: branding only
                    HStack {
                        Spacer()
                        Text("SPECTRA")
                            .font(.system(size: 14, weight: .bold, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.5))
                        Spacer()
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
                            .onChange(of: model.depthMode) { oldVal, newVal in
                                print("[DemoToggle] Picker changed: \(oldVal) → \(newVal)")
                            }

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

    private var crosshair: some View {
        let size: CGFloat = 28, thick: CGFloat = 2, gap: CGFloat = 6
        return ZStack {
            HStack(spacing: gap * 2) {
                Rectangle().frame(width: size, height: thick)
                Rectangle().frame(width: size, height: thick)
            }
            VStack(spacing: gap * 2) {
                Rectangle().frame(width: thick, height: size)
                Rectangle().frame(width: thick, height: size)
            }
            Circle().frame(width: 4, height: 4)
        }
        .foregroundStyle(.white.opacity(0.8))
        .shadow(color: .black.opacity(0.6), radius: 2)
        .allowsHitTesting(false)
    }
}
