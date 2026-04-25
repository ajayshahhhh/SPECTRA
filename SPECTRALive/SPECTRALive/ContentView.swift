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
                HStack {
                    lidarBadge
                    Spacer()
                }
                .padding(.top, 56)
                .padding(.leading, 24)
                .padding(.trailing, 16)
                distanceLabel
                    .padding(.top, 4)
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
            .overlay(alignment: .bottomTrailing) {
                colorScaleKey
                    .padding(.trailing, 12)
                    .padding(.bottom, 140)
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
    }

    @ViewBuilder
    private var distanceLabel: some View {
        let text = model.centerDistance.map { String(format: "%.3f m", $0) } ?? "— m"
        Text(text)
            .font(.system(size: 30, weight: .bold, design: .monospaced))
            .foregroundStyle(.white)
            .shadow(color: .black.opacity(0.8), radius: 3, x: 0, y: 1)
    }

    private var crosshair: some View {
        let size: CGFloat = 28
        let thickness: CGFloat = 2
        let gap: CGFloat = 6
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
            Circle().frame(width: 4, height: 4)
        }
        .foregroundStyle(.white.opacity(0.8))
        .shadow(color: .black.opacity(0.6), radius: 2, x: 0, y: 0)
        .allowsHitTesting(false)
    }

    private var colorScaleKey: some View {
        let barHeight: CGFloat = 120
        let minD = model.minDepth
        let maxD = model.maxDepth
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
                    Text("—")
                    Spacer()
                    Text("—")
                    Spacer()
                    Text("—")
                }
            }
            .font(.system(size: 9, weight: .medium, design: .monospaced))
            .foregroundStyle(.white)
            .fixedSize(horizontal: true, vertical: false)
            .frame(height: barHeight)

            LinearGradient(
                colors: [.red, .yellow, .green, .cyan, .blue],
                startPoint: .top,
                endPoint: .bottom
            )
            .frame(width: 12, height: barHeight)
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .overlay(
                RoundedRectangle(cornerRadius: 4)
                    .strokeBorder(.white.opacity(0.6), lineWidth: 1)
            )
        }
        .shadow(color: .black.opacity(0.7), radius: 3, x: 0, y: 1)
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
                if !result.urls.isEmpty {
                    showShareSheet = true
                }
                try? await Task.sleep(for: .seconds(3))
                if model.captureMessage == result.message {
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
        Button {
            showShareSheet = true
        } label: {
            Image(systemName: "square.and.arrow.up")
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 44, height: 44)
                .background(.white.opacity(0.2), in: Circle())
        }
    }
}

#Preview {
    ContentView()
}
