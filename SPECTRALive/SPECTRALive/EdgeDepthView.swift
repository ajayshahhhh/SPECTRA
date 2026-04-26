import SwiftUI
import Combine
import ARKit
import RealityKit

// MARK: - Session model

@MainActor
final class EdgeDepthSessionModel: ObservableObject {
    @Published var overlayImage: UIImage?
    @Published var centerDistance: Float?
    @Published var minDepth: Float?
    @Published var maxDepth: Float?
    @Published var detections: [EdgeDetection] = []
    @Published var captureMessage: String?
    @Published var capturedURLs: [URL] = []
    nonisolated(unsafe) var latestFrame: ARFrame?
}

// MARK: - AR container

struct EdgeARViewContainer: UIViewRepresentable {
    let model: EdgeDepthSessionModel

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }

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

    func updateUIView(_ uiView: ARView, context: Context) {}

    final class Coordinator: NSObject, ARSessionDelegate {
        private let model: EdgeDepthSessionModel
        nonisolated(unsafe) private var lastProcessTime: CFAbsoluteTime = 0
        nonisolated(unsafe) private var processing = false
        private let processInterval: CFAbsoluteTime = 1.0 / 60.0

        init(model: EdgeDepthSessionModel) { self.model = model }

        nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
            model.latestFrame = frame
            let now = CFAbsoluteTimeGetCurrent()
            guard now - lastProcessTime >= processInterval else { return }
            guard !processing else { return }
            processing = true
            lastProcessTime = now

            let trackingNormal = frame.camera.trackingState == .normal
            Task.detached(priority: .userInitiated) { [model = self.model, coordinator = self] in
                let result = EdgeDepthProcessor.process(frame: frame)
                coordinator.processing = false
                await MainActor.run {
                    if let r = result {
                        model.overlayImage    = r.overlayImage
                        model.centerDistance  = trackingNormal ? r.centerDistance : nil
                        model.minDepth        = r.minDepth
                        model.maxDepth        = r.maxDepth
                        model.detections      = r.detections
                    }
                }
            }
        }

        nonisolated func session(_ session: ARSession, didFailWithError error: Error) {
            let msg = error.localizedDescription
            Task { @MainActor [model = self.model] in model.captureMessage = "AR Error: \(msg)" }
        }
    }
}

// MARK: - Edge Depth View

struct EdgeDepthView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var model = EdgeDepthSessionModel()
    @State private var isCapturing = false
    @State private var showShareSheet = false

    var body: some View {
        ZStack {
            // Live camera feed
            EdgeARViewContainer(model: model).ignoresSafeArea()

            // Depth-colored edge overlay (transparent background)
            if let img = model.overlayImage {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFill()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipped()
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
            }

            // Crosshair
            crosshair

            // HUD
            VStack(spacing: 0) {
                HStack(alignment: .top) {
                    Spacer()
                    if !model.detections.isEmpty {
                        detectionCountBadge(model.detections.count)
                            .fixedSize()
                    }
                }
                .padding(.top, 56)
                .padding(.horizontal, 16)

                distanceLabel.padding(.top, 10)
                Spacer()

                HStack(spacing: 40) {
                    backButton
                    captureButton
                    shareButton
                        .disabled(model.capturedURLs.isEmpty)
                        .opacity(model.capturedURLs.isEmpty ? 0.3 : 1)
                }
                .padding(.bottom, 48)
            }
            .overlay(alignment: .bottomTrailing) {
                colorLegend
                    .padding(.trailing, 6)
                    .padding(.bottom, 140)
            }

            // Toast
            if let msg = model.captureMessage {
                VStack {
                    Text(msg)
                        .font(.caption).foregroundStyle(.white)
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
            Image(systemName: "lines.measurement.horizontal")
                .font(.system(size: 11, weight: .semibold))
            Text("EDGE DEPTH")
                .font(.system(size: 11, weight: .bold, design: .monospaced))
        }
        .foregroundStyle(.black)
        .padding(.horizontal, 10).padding(.vertical, 5)
        .background(.white, in: Capsule())
        .fixedSize()
    }

    private func detectionCountBadge(_ n: Int) -> some View {
        Text("\(n) object\(n == 1 ? "" : "s")")
            .font(.system(size: 10, weight: .medium, design: .monospaced))
            .foregroundStyle(.white.opacity(0.8))
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(.white.opacity(0.15), in: Capsule())
    }

    // MARK: - HUD elements

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

    // Compact color legend: shows near/far mapping
    private var colorLegend: some View {
        let barH: CGFloat = 100
        let minD = model.minDepth, maxD = model.maxDepth
        return HStack(alignment: .center, spacing: 4) {
            VStack(alignment: .trailing, spacing: 0) {
                if let minD, let maxD {
                    Text(String(format: "%.1fm", minD))
                    Spacer()
                    Text(String(format: "%.1fm", maxD))
                } else {
                    Text("near"); Spacer(); Text("far")
                }
            }
            .font(.system(size: 9, weight: .medium, design: .monospaced))
            .foregroundStyle(.white)
            .frame(height: barH)

            LinearGradient(
                colors: [.red, .yellow, .green, .cyan, .blue],
                startPoint: .top, endPoint: .bottom
            )
            .frame(width: 10, height: barH)
            .clipShape(RoundedRectangle(cornerRadius: 3))
            .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(.white.opacity(0.5), lineWidth: 1))
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
