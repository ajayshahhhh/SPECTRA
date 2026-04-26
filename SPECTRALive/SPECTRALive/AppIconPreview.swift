import SwiftUI

/// Preview view to generate the SPECTRA app icon
/// Screenshot this at 1024x1024 to export as app icon
struct AppIconPreview: View {
    let bgDark = Color(red: 0.05, green: 0.047, blue: 0.039)
    let amber = Color(red: 0.831, green: 0.659, blue: 0.263)
    let amberDim = Color(red: 0.541, green: 0.431, blue: 0.165)

    @State private var animateWaves = false

    var body: some View {
        ZStack {
            // Dark background
            bgDark

            // Concentric circles - LiDAR depth waves
            ForEach(0..<6) { index in
                Circle()
                    .strokeBorder(
                        amber.opacity(waveOpacity(for: index)),
                        lineWidth: waveLineWidth(for: index)
                    )
                    .frame(width: waveSize(for: index), height: waveSize(for: index))
                    .scaleEffect(animateWaves ? 1.0 : 0.95)
                    .animation(
                        .easeInOut(duration: 3.0)
                            .repeatForever(autoreverses: true)
                            .delay(Double(index) * 0.15),
                        value: animateWaves
                    )
            }

            // Central dot - LiDAR origin point
            Circle()
                .fill(
                    RadialGradient(
                        colors: [amber, amber.opacity(0.8)],
                        center: .center,
                        startRadius: 0,
                        endRadius: 35
                    )
                )
                .frame(width: 70, height: 70)
                .shadow(color: amber.opacity(0.6), radius: 20, x: 0, y: 0)

            // Subtle glow effect
            Circle()
                .fill(
                    RadialGradient(
                        colors: [amber.opacity(0.15), .clear],
                        center: .center,
                        startRadius: 0,
                        endRadius: 400
                    )
                )
                .frame(width: 800, height: 800)
                .blendMode(.plusLighter)
        }
        .frame(width: 1024, height: 1024)
        .onAppear {
            animateWaves = true
        }
    }

    // Calculate size for each concentric wave
    private func waveSize(for index: Int) -> CGFloat {
        let baseSize: CGFloat = 200
        let increment: CGFloat = 120
        return baseSize + (CGFloat(index) * increment)
    }

    // Calculate opacity - outer waves are dimmer
    private func waveOpacity(for index: Int) -> Double {
        let maxOpacity = 0.9
        let minOpacity = 0.15
        let step = (maxOpacity - minOpacity) / 5.0
        return maxOpacity - (Double(index) * step)
    }

    // Calculate line width - outer waves are thinner
    private func waveLineWidth(for index: Int) -> CGFloat {
        let maxWidth: CGFloat = 8
        let minWidth: CGFloat = 2
        let step = (maxWidth - minWidth) / 5.0
        return maxWidth - (CGFloat(index) * step)
    }
}

#Preview {
    AppIconPreview()
}
