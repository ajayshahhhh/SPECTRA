import ARKit
import Combine
import UIKit

@MainActor
final class ARSessionModel: ObservableObject {
    @Published var depthImage: UIImage?
    @Published var centerDistance: Float?
    @Published var minDepth: Float?
    @Published var maxDepth: Float?
    @Published var captureMessage: String?
    @Published var capturedURLs: [URL] = []
    // Written from ARKit background thread, read on main — guarded by caller logic
    nonisolated(unsafe) var latestFrame: ARFrame?
}

