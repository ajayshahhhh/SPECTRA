import SwiftUI
import ARKit

struct ContentView: View {
    @StateObject private var model = ARSessionModel()
    @State private var isCapturing = false

    var body: some View {
        ZStack {
            // ── Camera feed (ARView) ──────────────────────────────────
            ARViewContainer(model: model)
                .ignoresSafeArea()

            // ── Depth heatmap overlay ─────────────────────────────────
            if let depthImage = model.depthImage {
                Image(uiImage: depthImage)
                    .resizable()
                    .scaledToFill()
                    .ignoresSafeArea()
                    .opacity(0.5)
                    .clipped()
                    .allowsHitTesting(false)
            }

            // ── HUD ───────────────────────────────────────────────────
            VStack {
                distanceLabel
                    .padding(.top, 60)
                Spacer()
                captureButton
                    .padding(.bottom, 48)
            }

            // ── Toast ─────────────────────────────────────────────────
            if let msg = model.captureMessage {
                VStack {
                    Text(msg)
                        .font(.caption)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.black.opacity(0.7), in: Capsule())
                        .padding(.top, 110)
                    Spacer()
                }
                .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.2), value: model.captureMessage)
    }

    // MARK: - Subviews

    @ViewBuilder
    private var distanceLabel: some View {
        let text = model.centerDistance.map { String(format: "%.2f m", $0) } ?? "— m"
        Text(text)
            .font(.system(size: 30, weight: .bold, design: .monospaced))
            .foregroundStyle(.white)
            .shadow(color: .black.opacity(0.8), radius: 3, x: 0, y: 1)
    }

    private var captureButton: some View {
        Button {
            guard !isCapturing, let frame = model.latestFrame else { return }
            isCapturing = true
            Task {
                let msg = await CaptureManager.capture(frame: frame)
                model.captureMessage = msg
                isCapturing = false
                try? await Task.sleep(for: .seconds(3))
                if model.captureMessage == msg {
                    model.captureMessage = nil
                }
            }
        } label: {
            ZStack {
                Circle()
                    .fill(isCapturing ? Color.gray : Color.white)
                    .frame(width: 68, height: 68)
                Circle()
                    .strokeBorder(.white, lineWidth: 3)
                    .frame(width: 80, height: 80)
            }
        }
        .disabled(isCapturing)
    }
}

#Preview {
    ContentView()
}
