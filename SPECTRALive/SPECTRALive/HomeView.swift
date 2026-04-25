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
                    modeButton(title: "ML Depth", subtitle: "Model-trained LiDAR", icon: "brain") {
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
                    MLDepthPlaceholderView()
                        .navigationBarBackButtonHidden(true)
                        .toolbar(.hidden, for: .navigationBar)
                        .overlay(alignment: .topLeading) {
                            backButton
                        }
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

    private var backButton: some View {
        Button { path.removeLast() } label: {
            Image(systemName: "chevron.left")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 40, height: 40)
                .background(.black.opacity(0.5), in: Circle())
        }
        .padding(.top, 54)
        .padding(.leading, 16)
    }
}

struct MLDepthPlaceholderView: View {
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 16) {
                Image(systemName: "brain")
                    .font(.system(size: 48))
                    .foregroundStyle(.white.opacity(0.3))
                Text("Coming Soon")
                    .font(.system(size: 24, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.white)
                Text("Model-trained LiDAR depth estimation")
                    .font(.system(size: 14))
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
    }
}
