// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The expert slices a ``SwitchLinear`` needs for one forward pass, stacked in
/// roster order.
///
/// `weight` is shaped `[roster.count, ...]` — the same layout as the full
/// stacked parameter, but holding only the experts the router actually picked.
/// `scales` and `biases` are present exactly when the underlying checkpoint is
/// quantized.
public struct StackedExpertWeights {
    public var weight: MLXArray
    public var scales: MLXArray?
    public var biases: MLXArray?

    public init(weight: MLXArray, scales: MLXArray? = nil, biases: MLXArray? = nil) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
    }
}

/// Supplies a MoE layer's expert weights on demand, so the full stacked
/// parameter never has to be resident.
///
/// A `[256, 512, 256]` 4-bit expert stack costs 134 MB per projection per
/// layer; a shard holding twenty layers cannot hold them all on a phone or
/// tablet. A provider hands back only the experts a given forward pass routed
/// to, and is free to cache across passes.
///
/// ### Threading
///
/// ``stackedExperts(path:experts:)`` is called synchronously from the model's
/// forward pass. Implementations must not block on an actor or a queue that the
/// calling thread could already own, and must be safe to call concurrently.
///
/// ### Failure
///
/// The hot path cannot throw. Implementations are expected to validate that
/// every byte they will ever need is present via ``preflight()``, which the
/// runtime calls once after loading; after that, a read failure is an
/// unrecoverable I/O fault and may trap.
public protocol ExpertWeightProviding: AnyObject, Sendable {
    /// Confirms every expert range this provider can be asked for is readable.
    func preflight() throws

    /// Stacked weights for `experts`, in the order given.
    ///
    /// - Parameters:
    ///   - path: The module path of the requesting ``SwitchLinear``, e.g.
    ///     `language_model.model.layers.7.mlp.switch_mlp.gate_proj`.
    ///   - experts: Globally-numbered expert ids, ascending and unique.
    func stackedExperts(path: String, experts: [Int]) -> StackedExpertWeights
}

/// The distinct experts one forward pass routes to, and the routing indices
/// renumbered to address them.
///
/// The router emits ids in `0 ..< numExperts`, but a streamed ``SwitchLinear``
/// only materialises the ones in use, so its stacked weight is indexed
/// `0 ..< experts.count`. `compactIndices` is the original index array with each
/// id replaced by its position in `experts`.
///
/// `experts` is ascending, so the remapping is monotonic: indices that were
/// sorted before compaction are still sorted after, and callers may keep
/// passing `sortedIndices: true` to the gather.
public struct ExpertRoster {
    public let experts: [Int]
    public let compactIndices: MLXArray

    public init(experts: [Int], compactIndices: MLXArray) {
        self.experts = experts
        self.compactIndices = compactIndices
    }

    /// Builds a roster from a router's index array.
    ///
    /// This reads `indices` back to the host, which forces an evaluation and a
    /// GPU sync. That is unavoidable: which bytes to load is a data-dependent
    /// decision, and nothing downstream can be enqueued until it is made.
    public static func make(from indices: MLXArray) -> ExpertRoster {
        let flat = indices.asType(.int32).asArray(Int32.self)
        let experts = Array(Set(flat)).sorted()

        var position = [Int32: Int32](minimumCapacity: experts.count)
        for (compact, expert) in experts.enumerated() {
            position[expert] = Int32(compact)
        }

        let compact = flat.map { position[$0]! }
        return ExpertRoster(
            experts: experts.map(Int.init),
            compactIndices: MLXArray(compact, indices.shape).asType(indices.dtype)
        )
    }
}
