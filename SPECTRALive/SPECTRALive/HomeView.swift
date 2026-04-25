import SwiftUI

enum AppDestination: Hashable {
    case liveDepth
    case mlDepth
    case demo
}

struct HomeView: View {
    @State private var path = NavigationPath()

    var body: some View {
        NavigationStack(path: $path) {
            VStack(spacing: 32) {
                VStack(spacing: 4) {
                    Text("SPECTRA")
                        .font(.system(size: 42, weight: .bold, design: .monospaced))
                        .foregroundStyle(.white)
                    Text("Depth Intelligence")
                        .font(.system(size: 13, weight: .medium, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.4))
                }

                VStack(spacing: 16) {
                    modeButton(
                        title: "Live Depth",
                        subtitle: "Raw LiDAR heatmap",
                        icon: "camera.fill",
                        tint: .cyan
                    ) { path.append(AppDestination.liveDepth) }

                    modeButton(
                        title: "SPECTRANet",
                        subtitle: "AI-enhanced depth + edge detection",
                        icon: "brain",
                        tint: .purple
                    ) { path.append(AppDestination.mlDepth) }

                    Text("Supports 0.3–5m depth range")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.35))

                    modeButton(
                        title: "Demo",
                        subtitle: "Side-by-side camera + depth comparison",
                        icon: "rectangle.split.2x1",
                        tint: .orange
                    ) { path.append(AppDestination.demo) }
                }
                .padding(.horizontal, 28)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.black)
            .navigationDestination(for: AppDestination.self) { dest in
                switch dest {
                case .liveDepth:
                    ContentView()
                        .navigationBarBackButtonHidden(true)
                        .toolbar(.hidden, for: .navigationBar)
                case .mlDepth:
                    MLDepthView()
                        .navigationBarBackButtonHidden(true)
                        .toolbar(.hidden, for: .navigationBar)
                case .demo:
                    DemoView()
                        .navigationBarBackButtonHidden(true)
                        .toolbar(.hidden, for: .navigationBar)
                }
            }
        }
    }

    private func modeButton(
        title: String, subtitle: String,
        icon: String, tint: Color,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 16) {
                Image(systemName: icon)
                    .font(.system(size: 24))
                    .foregroundStyle(tint)
                    .frame(width: 40)
                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(.white)
                    Text(subtitle)
                        .font(.system(size: 13))
                        .foregroundStyle(.white.opacity(0.55))
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.3))
            }
            .padding(18)
            .background(.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(tint.opacity(0.3), lineWidth: 1)
            )
        }
    }
}
