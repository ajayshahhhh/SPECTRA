import SwiftUI
import ARKit

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

struct ContentView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var model = ARSessionModel()
    @State private var isCapturing = false
    @State private var showShareSheet = false

    var body: some View {
        ZStack {
            // ── Camera feed (ARView) ──────────────────────────────────
            ARViewContainer(model: model)
                .ignoresSafeArea()

            // ── Depth heatmap overlay ─────────────────────────────────
            if let depthImage = model.depthImage {
                Image(uiImage: depthImage)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFill()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipped()
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
            }

            // ── Crosshair ────────────────────────────────────────────
            crosshair

            // ── HUD ───────────────────────────────────────────────────
            VStack {
                distanceLabel
                    .padding(.top, 70)
                Spacer()
                HStack {
                    Spacer()
                    colorScaleKey
                }
                .padding(.trailing, 16)
                .padding(.bottom, 12)
                HStack(spacing: 50) {
                    backButton
                    captureButton
                    shareButton
                        .disabled(model.capturedURLs.isEmpty)
                        .opacity(model.capturedURLs.isEmpty ? 0.3 : 1)
                }
                .padding(.bottom, 60)
            }
            .overlay(alignment: .bottomTrailing) {
                colorScaleKey
                    .padding(.trailing, 16)
                    .padding(.bottom, 180)
            }

            // ── Toast ─────────────────────────────────────────────────
            if let msg = model.captureMessage {
                VStack {
                    Text(msg)
                        .font(.system(size: 18, weight: .medium))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 22)
                        .padding(.vertical, 12)
                        .background(.black.opacity(0.75), in: Capsule())
                        .padding(.top, 140)
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

    // MARK: - Subviews

    private var lidarBadge: some View {
        HStack(spacing: 5) {
            Image(systemName: "scope")
                .font(.system(size: 11, weight: .semibold))
            Text("RAW LiDAR")
                .font(.system(size: 11, weight: .bold, design: .monospaced))
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(.white.opacity(0.18), in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.35), lineWidth: 1))
        .fixedSize()
    }

    @ViewBuilder
    private var distanceLabel: some View {
        let text = model.centerDistance.map { String(format: "%.3f m", $0) } ?? "— m"
        Text(text)
            .font(.system(size: 52, weight: .bold, design: .monospaced))
            .foregroundStyle(.white)
            .shadow(color: .black.opacity(0.8), radius: 5, x: 0, y: 2)
    }

    private var crosshair: some View {
        let size: CGFloat = 42
        let thickness: CGFloat = 3.5
        let gap: CGFloat = 10
        return ZStack {
            // Horizontal lines
            HStack(spacing: gap * 2) {
                Rectangle().frame(width: size, height: thickness)
                Rectangle().frame(width: size, height: thickness)
            }
            // Vertical lines
            VStack(spacing: gap * 2) {
                Rectangle().frame(width: thickness, height: size)
                Rectangle().frame(width: thickness, height: size)
            }
            // Center dot
            Circle().frame(width: 7, height: 7)
        }
        .foregroundStyle(.white.opacity(0.8))
        .shadow(color: .black.opacity(0.6), radius: 4, x: 0, y: 0)
        .allowsHitTesting(false)
    }

    private var colorScaleKey: some View {
        let barHeight: CGFloat = 170
        let minD = model.minDepth
        let maxD = model.maxDepth
        return HStack(alignment: .center, spacing: 8) {
            VStack(alignment: .trailing, spacing: 0) {
                if let minD, let maxD {
                    let mid = (minD + maxD) / 2
                    Text(String(format: "%.1fm", minD))
                    Spacer()
                    Text(String(format: "%.1fm", mid))
                    Spacer()
                    Text(String(format: "%.1fm", maxD))
                } else {
                    Text("—")
                    Spacer()
                    Text("—")
                    Spacer()
                    Text("—")
                }
            }
            .font(.system(size: 15, weight: .medium, design: .monospaced))
            .foregroundStyle(.white)
            .fixedSize(horizontal: true, vertical: false)
            .frame(height: barHeight)

            LinearGradient(
                colors: [.red, .yellow, .green, .cyan, .blue],
                startPoint: .top,
                endPoint: .bottom
            )
            .frame(width: 18, height: barHeight)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(.white.opacity(0.6), lineWidth: 2)
            )
        }
        .shadow(color: .black.opacity(0.7), radius: 5, x: 0, y: 2)
    }

    private var captureButton: some View {
        Button {
            guard !isCapturing, let frame = model.latestFrame else { return }
            isCapturing = true
            Task {
                let result = await CaptureManager.capture(frame: frame)
                model.captureMessage = result.message
                model.capturedURLs = result.urls
                isCapturing = false
                try? await Task.sleep(for: .seconds(3))
                if model.captureMessage == result.message {
                    model.captureMessage = nil
                }
            }
        } label: {
            ZStack {
                Circle()
                    .fill(isCapturing ? Color.gray : Color.white)
                    .frame(width: 95, height: 95)
                Circle()
                    .strokeBorder(.white, lineWidth: 5)
                    .frame(width: 112, height: 112)
            }
        }
        .disabled(isCapturing)
    }

    private var backButton: some View {
        Button { dismiss() } label: {
            Image(systemName: "chevron.left")
                .font(.system(size: 32, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 64, height: 64)
                .background(.white.opacity(0.2), in: Circle())
        }
    }

    private var shareButton: some View {
        Button {
            showShareSheet = true
        } label: {
            Image(systemName: "square.and.arrow.up")
                .font(.system(size: 32, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 64, height: 64)
                .background(.white.opacity(0.2), in: Circle())
        }
    }
}

#Preview {
    ContentView()
}
