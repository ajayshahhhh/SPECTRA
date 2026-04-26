//
//  SPECTRALiveApp.swift
//  SPECTRALive
//
//  Created by Ajay Shah on 4/25/26.
//

import SwiftUI

class AppDelegate: NSObject, UIApplicationDelegate {
    static var orientationLock: UIInterfaceOrientationMask = .all

    func application(
        _ application: UIApplication,
        supportedInterfaceOrientationsFor window: UIWindow?
    ) -> UIInterfaceOrientationMask {
        AppDelegate.orientationLock
    }
}

// Loading screen view
private struct LoadingView: View {
    @State private var isRotating = false
    @State private var pulseScale: CGFloat = 1.0

    var body: some View {
        ZStack {
            Color(red: 0.051, green: 0.047, blue: 0.039).ignoresSafeArea()

            // Subtle orb glow effect
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            Color(red: 0.831, green: 0.659, blue: 0.263).opacity(0.08),
                            .clear
                        ],
                        center: .center,
                        startRadius: 0,
                        endRadius: 280
                    )
                )
                .frame(width: 560, height: 560)
                .scaleEffect(pulseScale)
                .opacity(0.9)
                .onAppear {
                    withAnimation(.easeInOut(duration: 4).repeatForever(autoreverses: true)) {
                        pulseScale = 1.06
                    }
                }

            VStack(spacing: 40) {
                // SPECTRA title
                Text("SPECTRA")
                    .font(.system(size: 80, weight: .black, design: .serif))
                    .foregroundStyle(Color(red: 0.941, green: 0.918, blue: 0.847))
                    .tracking(-0.02 * 80)

                // Spinning loading circle
                ZStack {
                    // Outer ring with gradient
                    Circle()
                        .trim(from: 0, to: 0.7)
                        .stroke(
                            AngularGradient(
                                colors: [
                                    Color(red: 0.831, green: 0.659, blue: 0.263),
                                    Color(red: 0.831, green: 0.659, blue: 0.263).opacity(0.8),
                                    Color(red: 0.831, green: 0.659, blue: 0.263).opacity(0.4),
                                    Color(red: 0.831, green: 0.659, blue: 0.263).opacity(0.1),
                                    .clear
                                ],
                                center: .center
                            ),
                            style: StrokeStyle(lineWidth: 4, lineCap: .round)
                        )
                        .frame(width: 60, height: 60)
                        .rotationEffect(.degrees(isRotating ? 360 : 0))
                        .onAppear {
                            withAnimation(.linear(duration: 1.2).repeatForever(autoreverses: false)) {
                                isRotating = true
                            }
                        }

                    // Inner pulsing dot
                    Circle()
                        .fill(Color(red: 0.831, green: 0.659, blue: 0.263))
                        .frame(width: 12, height: 12)
                        .opacity(0.6)
                        .scaleEffect(pulseScale * 0.8)
                }

                // Subtle loading text
                Text("Loading...")
                    .font(.system(size: 14, weight: .medium, design: .monospaced))
                    .foregroundStyle(Color(red: 0.831, green: 0.659, blue: 0.263).opacity(0.6))
                    .tracking(0.2 * 14)
            }
        }
    }
}

@main
struct SPECTRALiveApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @State private var isLoading = true

    var body: some Scene {
        WindowGroup {
            ZStack {
                if isLoading {
                    LoadingView()
                        .transition(.opacity)
                } else {
                    HomeView()
                        .transition(.opacity)
                }
            }
            .animation(.easeInOut(duration: 0.5), value: isLoading)
            .onAppear {
                // Show loading screen for 2 seconds
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    isLoading = false
                }
            }
        }
    }
}
