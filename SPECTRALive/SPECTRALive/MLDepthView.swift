import SwiftUI
import Combine
import ARKit
import RealityKit

// MARK: - Session model

@MainActor
final class MLDepthSessionModel: ObservableObject {
    @Published var depthImage: UIImage?
    @Published var centerDistance: Float?
    @Published var minDepth: Float?
    @Published var maxDepth: Float?
    @Published var captureMessage: String?
    @Published var capturedURLs: [URL] = []
    @Published var isProcessing = false
    @Published var lastInferenceMs: Int?
    nonisolated(unsafe) var latestFrame: ARFrame?
}

// MARK: - AR container for ML mode

struct MLARViewContainer: UIViewRepresentable {
    let model: MLDepthSessionModel

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        arView.session.delegate = context.coordinator
        arView.renderOptions = []
        let config = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics = .sceneDepth
        } else {
            Task { @MainActor in
                model.captureMessage = "LiDAR not available on this device"
            }
        }
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    final class Coordinator: NSObject, ARSessionDelegate {
        private let model: MLDepthSessionModel
        // Prevents overlapping requests — next frame fires as soon as the previous completes
        nonisolated(unsafe) private var isInFlight = false

        init(model: MLDepthSessionModel) { self.model = model }

        nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
            model.latestFrame = frame
            guard !isInFlight else { return }
            isInFlight = true

            Task { @MainActor [model = self.model] in model.isProcessing = true }

            let t0 = CFAbsoluteTimeGetCurrent()
            Task.detached(priority: .userInitiated) { [model = self.model] in
                let result = await SPECTRANetProcessor.process(frame: frame)
                let ms = Int((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                await MainActor.run {
                    if let r = result {
                        let blended = DepthProcessor.blendHeatmapWithCamera(
                            heatmap: r.colorImage,
                            capturedImage: frame.capturedImage
                        )
                        model.depthImage     = blended ?? r.colorImage
                        model.centerDistance = r.centerDistance
                        model.minDepth       = r.minDepth
                        model.maxDepth       = r.maxDepth
                    }
                    model.isProcessing    = false
                    model.lastInferenceMs = ms
                }
                self.isInFlight = false
            }
        }

        nonisolated func session(_ session: ARSession, didFailWithError error: Error) {
            let msg = error.localizedDescription
            Task { @MainActor [model = self.model] in
                model.captureMessage = "AR Error: \(msg)"
            }
        }
    }
}

// MARK: - ML Depth View

struct MLDepthView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var model = MLDepthSessionModel()
    @State private var isCapturing = false
    @State private var showShareSheet = false
    @State private var processingPulse = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // ARKit session runs hidden — needed for LiDAR depth data
            MLARViewContainer(model: model)
                .ignoresSafeArea()
                .opacity(0)

            // Full-screen heatmap (no camera feed blended in)
            if let img = model.depthImage {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFill()
                    .ignoresSafeArea()
                    .clipped()
                    .allowsHitTesting(false)
            } else {
                VStack(spacing: 12) {
                    ProgressView().tint(.white).scaleEffect(1.4)
                    Text("Waiting for first frame…")
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.6))
                }
            }

            // Crosshair
            crosshair

            // Spinning ring while inference runs
            Circle()
                .strokeBorder(.white.opacity(processingPulse ? 0.5 : 0.0), lineWidth: 1.5)
                .frame(width: 52, height: 52)
                .animation(
                    model.isProcessing
                        ? .easeInOut(duration: 0.5).repeatForever(autoreverses: true)
                        : .default,
                    value: processingPulse
                )
                .onChange(of: model.isProcessing) { _, running in
                    processingPulse = running
                }
                .allowsHitTesting(false)

            // HUD
            VStack(spacing: 0) {
                // Top row: mode badge (left) + inference time (right)
                HStack(alignment: .top) {
                    modeBadge
                    Spacer()
                    VStack(alignment: .trailing, spacing: 6) {
                        if model.depthImage == nil {
                            loadingBadge
                        }
                        if let ms = model.lastInferenceMs {
                            infoBadge("\(ms) ms")
                        }
                    }
                }
                .padding(.top, 56)
                .padding(.horizontal, 16)

                distanceLabel.padding(.top, 10)
                Spacer()
                HStack {
                    Spacer()
                    colorScaleKey
                }
                .padding(.trailing, 12)
                .padding(.bottom, 8)
                HStack(spacing: 40) {
                    backButton
                    captureButton
                    shareButton
                        .disabled(model.capturedURLs.isEmpty)
                        .opacity(model.capturedURLs.isEmpty ? 0.3 : 1)
                }
                .padding(.bottom, 48)
            }

            // Toast
            if let msg = model.captureMessage {
                VStack {
                    Text(msg)
                        .font(.caption)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(.black.opacity(0.7), in: Capsule())
                        .padding(.top, 110)
                    Spacer()
                }
                .transition(.opacity)
                .animation(.easeInOut(duration: 0.2), value: model.captureMessage)
            }
        }
        .sheet(isPresented: $showShareSheet) {
            ShareSheet(items: model.capturedURLs)
        }
    }

    // MARK: - Badges

    private var modeBadge: some View {
        HStack(spacing: 5) {
            Image(systemName: "brain")
                .font(.system(size: 11, weight: .semibold))
            Text("SPECTRANet")
                .font(.system(size: 11, weight: .bold, design: .monospaced))
        }
        .foregroundStyle(.black)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(.white, in: Capsule())
    }

    private var loadingBadge: some View {
        HStack(spacing: 5) {
            ProgressView().scaleEffect(0.7).tint(.white)
            Text("loading model…")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
        }
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(.white.opacity(0.12), in: Capsule())
    }

    private func infoBadge(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .medium, design: .monospaced))
            .foregroundStyle(.white.opacity(0.75))
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(.white.opacity(0.12), in: Capsule())
    }

    // MARK: - HUD elements (mirror ContentView)

    @ViewBuilder
    private var distanceLabel: some View {
        let text = model.centerDistance.map { String(format: "%.3f m", $0) } ?? "— m"
        Text(text)
            .font(.system(size: 30, weight: .bold, design: .monospaced))
            .foregroundStyle(.white)
            .shadow(color: .black.opacity(0.8), radius: 3, x: 0, y: 1)
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

    private var colorScaleKey: some View {
        let barH: CGFloat = 120
        let minD = model.minDepth, maxD = model.maxDepth
        return HStack(alignment: .center, spacing: 4) {
            VStack(alignment: .trailing, spacing: 0) {
                if let minD, let maxD {
                    let mid = (minD + maxD) / 2
                    Text(String(format: "%.1fm", minD))
                    Spacer()
                    Text(String(format: "%.1fm", mid))
                    Spacer()
                    Text(String(format: "%.1fm", maxD))
                } else {
                    Text("—"); Spacer(); Text("—"); Spacer(); Text("—")
                }
            }
            .font(.system(size: 9, weight: .medium, design: .monospaced))
            .foregroundStyle(.white)
            .frame(height: barH)

            LinearGradient(
                colors: [.red, .yellow, .green, .cyan, .blue],
                startPoint: .top, endPoint: .bottom
            )
            .frame(width: 12, height: barH)
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(.white.opacity(0.6), lineWidth: 1))
        }
        .shadow(color: .black.opacity(0.7), radius: 3, x: 0, y: 1)
    }

    // MARK: - Buttons

    private var captureButton: some View {
        Button {
            guard !isCapturing, let frame = model.latestFrame else { return }
            isCapturing = true
            Task {
                let result = await CaptureManager.capture(frame: frame)
                model.captureMessage = result.message
                model.capturedURLs = result.urls
                isCapturing = false
                if !result.urls.isEmpty { showShareSheet = true }
                try? await Task.sleep(for: .seconds(3))
                if model.captureMessage == result.message { model.captureMessage = nil }
            }
        } label: {
            ZStack {
                Circle().fill(isCapturing ? Color.gray : Color.white).frame(width: 68, height: 68)
                Circle().strokeBorder(.white, lineWidth: 3).frame(width: 80, height: 80)
            }
        }
        .disabled(isCapturing)
    }

    private var backButton: some View {
        Button { dismiss() } label: {
            Image(systemName: "chevron.left")
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 44, height: 44)
                .background(.white.opacity(0.2), in: Circle())
        }
    }

    private var shareButton: some View {
        Button { showShareSheet = true } label: {
            Image(systemName: "square.and.arrow.up")
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 44, height: 44)
                .background(.white.opacity(0.2), in: Circle())
        }
    }
}
