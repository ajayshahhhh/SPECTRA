import ARKit
import Combine
import UIKit

@MainActor
final class ARSessionModel: ObservableObject {
    @Published var depthImage: UIImage?
    @Published var centerDistance: Float?
    @Published var captureMessage: String?
    // Written from ARKit background thread, read on main — guarded by caller logic
    nonisolated(unsafe) var latestFrame: ARFrame?
}

