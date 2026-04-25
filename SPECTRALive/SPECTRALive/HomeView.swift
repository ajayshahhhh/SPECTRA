import SwiftUI

enum AppDestination: Hashable {
    case liveDepth
    case mlDepth
}

struct HomeView: View {
    @State private var path = NavigationPath()

    var body: some View {
        NavigationStack(path: $path) {
            VStack(spacing: 32) {
                Text("SPECTRA")
                    .font(.system(size: 42, weight: .bold, design: .monospaced))
                    .foregroundStyle(.white)

                VStack(spacing: 20) {
                    modeButton(title: "Live Depth", subtitle: "LiDAR depth mapping", icon: "camera.fill") {
                        path.append(AppDestination.liveDepth)
                    }
                    modeButton(title: "SPECTRANet", subtitle: "Dense AI-enhanced depth", icon: "brain") {
                        path.append(AppDestination.mlDepth)
                    }
                }
                .padding(.horizontal, 32)
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
                }
            }
        }
    }

    private func modeButton(title: String, subtitle: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 16) {
                Image(systemName: icon)
                    .font(.system(size: 28))
                    .frame(width: 44)
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.system(size: 20, weight: .semibold))
                    Text(subtitle)
                        .font(.system(size: 14))
                        .foregroundStyle(.white.opacity(0.6))
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.4))
            }
            .foregroundStyle(.white)
            .padding(20)
            .background(.white.opacity(0.1), in: RoundedRectangle(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .strokeBorder(.white.opacity(0.2), lineWidth: 1)
            )
        }
    }

}
