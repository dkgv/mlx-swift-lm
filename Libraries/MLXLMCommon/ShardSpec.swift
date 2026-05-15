import Foundation

/// Describes the slice of a model that a single device is responsible for in a distributed
/// pipeline-parallel inference run.
///
/// The factory uses this to build a key filter so that `loadWeights` only assigns and
/// materialises the tensors this device actually needs. Non-shard parameters remain as
/// lazy, un-evaluated MLXArrays and never consume device memory.
public struct ShardSpec: Sendable {
    /// First transformer layer owned by this device (inclusive).
    public let startLayer: Int
    /// First transformer layer NOT owned by this device (exclusive).
    public let endLayer: Int
    /// Rank of this device in the pipeline (0 = coordinator).
    public let deviceRank: Int
    /// Total number of devices in the pipeline.
    public let worldSize: Int

    public init(startLayer: Int, endLayer: Int, deviceRank: Int, worldSize: Int) {
        self.startLayer = startLayer
        self.endLayer = endLayer
        self.deviceRank = deviceRank
        self.worldSize = worldSize
    }

    /// Coordinator owns the embedding matrix and the first block of layers.
    public var ownsEmbed: Bool { deviceRank == 0 }

    /// Tail worker owns the final norm and LM-head projection.
    public var ownsLMHead: Bool { deviceRank == worldSize - 1 }
}
