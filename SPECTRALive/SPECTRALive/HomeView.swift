import SwiftUI

enum AppDestination: Hashable {
    case liveDepth
    case mlDepth
    case demo
}

struct HomeView: View {
    @State private var path = NavigationPath()
    @State private var pulseScale: CGFloat = 1.0
    @AppStorage("useZetic") private var useZetic = true

    // Color palette matching the HTML design
    let bgDark = Color(hex: "0d0c0a")
    let bgCard = Color(hex: "131210")
    let amber = Color(hex: "d4a843")
    let amberDim = Color(hex: "8a6e2a")
    let cream = Color(hex: "f0ead8")
    let creamDim = Color(hex: "887f6a")
    let cardBorder = Color(hex: "252320")
    let red = Color(hex: "c45c3a")

    var body: some View {
        NavigationStack(path: $path) {
            ZStack {
                bgDark.ignoresSafeArea()

                // Subtle orb glow effect matching landing page
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [amber.opacity(0.055), .clear],
                            center: .center,
                            startRadius: 0,
                            endRadius: 280
                        )
                    )
                    .frame(width: 560, height: 560)
                    .scaleEffect(pulseScale)
                    .opacity(0.8)
                    .onAppear {
                        withAnimation(.easeInOut(duration: 5).repeatForever(autoreverses: true)) {
                            pulseScale = 1.04
                        }
                    }

                VStack(spacing: 0) {
                    Spacer()

                    // Header
                    VStack(spacing: 18) {
                        Text("SPECTRA")
                            .font(.system(size: 72, weight: .black, design: .serif))
                            .foregroundStyle(cream)
                            .tracking(-0.02 * 72)

                        Text("SPARSE-TO-DENSE DEPTH COMPLETION")
                            .font(.system(size: 13, weight: .medium, design: .monospaced))
                            .foregroundStyle(amber)
                            .tracking(0.2 * 13)

                        // Backend toggle switch
                        #if !targetEnvironment(simulator)
                        HStack(spacing: 14) {
                            Text("Backend:")
                                .font(.system(size: 15, weight: .medium, design: .monospaced))
                                .foregroundStyle(creamDim)

                            HStack(spacing: 0) {
                                // GX10 side
                                Text("GX10")
                                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                                    .foregroundStyle(useZetic ? creamDim : .black)
                                    .frame(width: 80, height: 38)
                                    .background(useZetic ? Color.clear : Color.orange)
                                    .contentShape(Rectangle())
                                    .onTapGesture {
                                        withAnimation(.easeInOut(duration: 0.2)) {
                                            useZetic = false
                                        }
                                    }

                                // Zetic side
                                Text("Zetic")
                                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                                    .foregroundStyle(useZetic ? .black : creamDim)
                                    .frame(width: 80, height: 38)
                                    .background(useZetic ? Color.green : Color.clear)
                                    .contentShape(Rectangle())
                                    .onTapGesture {
                                        withAnimation(.easeInOut(duration: 0.2)) {
                                            useZetic = true
                                        }
                                    }
                            }
                            .background(cardBorder)
                            .clipShape(Capsule())
                            .overlay(Capsule().strokeBorder(cardBorder.opacity(0.5), lineWidth: 1))
                        }
                        .padding(.top, 8)
                        #endif
                    }
                    .padding(.bottom, 52)

                    // Mode cards - Live Depth and SPECTRANet in one row
                    VStack(spacing: 16) {
                        HStack(alignment: .top, spacing: 16) {
                            modeCard(
                                label: "Live Depth",
                                description: "Raw LiDAR heatmap",
                                icon: "scope",
                                borderColor: amber,
                                destination: .liveDepth
                            )

                            modeCard(
                                label: "SPECTRANet",
                                description: "AI-enhanced depth",
                                icon: "brain",
                                borderColor: Color(hex: "c47bdb"),
                                destination: .mlDepth,
                                titleSize: 23
                            )
                        }

                        // Demo card spanning full width
                        modeCard(
                            label: "Demo",
                            description: "Side-by-side comparison view",
                            icon: "rectangle.split.2x1",
                            borderColor: red,
                            destination: .demo
                        )

                        // Scope note
                        HStack(spacing: 0) {
                            Text("SCOPE: ")
                                .font(.system(size: 13, weight: .medium, design: .monospaced))
                                .foregroundStyle(amber)
                            Text("Indoor · 0.5–10m · CoreML")
                                .font(.system(size: 13, weight: .light, design: .monospaced))
                                .foregroundStyle(creamDim)
                        }
                        .padding(.vertical, 10)
                    }
                    .padding(.horizontal, 72)

                    Spacer()

                    // Footer
                    VStack(spacing: 6) {
                        Text("LA HACKS 2026")
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundStyle(creamDim)
                            .tracking(0.15 * 12)
                    }
                    .padding(.bottom, 32)
                }
            }
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

    private func modeCard(
        label: String,
        description: String,
        icon: String,
        borderColor: Color,
        destination: AppDestination,
        titleSize: CGFloat = 28
    ) -> some View {
        Button {
            path.append(destination)
        } label: {
            VStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.system(size: 34, weight: .medium))
                    .foregroundStyle(borderColor)

                VStack(spacing: 6) {
                    Text(label)
                        .font(.system(size: titleSize, weight: .semibold, design: .serif))
                        .foregroundStyle(cream)
                    Text(description)
                        .font(.system(size: 18, weight: .light))
                        .foregroundStyle(creamDim)
                        .lineLimit(2)
                        .multilineTextAlignment(.center)
                        .minimumScaleFactor(0.7)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 24)
            .padding(.vertical, 28)
            .background(bgCard)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .strokeBorder(cardBorder, lineWidth: 1)
            )
            .overlay(
                Rectangle()
                    .fill(borderColor)
                    .frame(height: 3)
                    .clipShape(RoundedRectangle(cornerRadius: 16)),
                alignment: .top
            )
        }
    }
}

// Hex color extension
extension Color {
    init(hex: String) {
        let scanner = Scanner(string: hex)
        var rgbValue: UInt64 = 0
        scanner.scanHexInt64(&rgbValue)

        let r = Double((rgbValue & 0xFF0000) >> 16) / 255.0
        let g = Double((rgbValue & 0x00FF00) >> 8) / 255.0
        let b = Double((rgbValue & 0x0000FF)) / 255.0

        self.init(red: r, green: g, blue: b)
    }
}
