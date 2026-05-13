import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum PipelineShardCacheKind: Sendable {
    case kv
    case mamba
}

public final class PipelineShardLayer {
    public let module: Module
    public let cacheKind: PipelineShardCacheKind

    private let callBody: (
        MLXArray,
        MLXFast.ScaledDotProductAttentionMaskMode,
        MLXArray?,
        (any KVCache)?
    ) -> MLXArray

    public init(
        module: Module,
        cacheKind: PipelineShardCacheKind,
        call: @escaping (
            MLXArray,
            MLXFast.ScaledDotProductAttentionMaskMode,
            MLXArray?,
            (any KVCache)?
        ) -> MLXArray
    ) {
        self.module = module
        self.cacheKind = cacheKind
        self.callBody = call
    }

    public func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: (any KVCache)?
    ) -> MLXArray {
        callBody(x, attentionMask, ssmMask, cache)
    }
}

public protocol PipelineShardableLanguageModel: LanguageModel {
    var pipelineShardEmbedTokens: Embedding { get }
    var pipelineShardLayers: [PipelineShardLayer] { get }
    var pipelineShardNorm: RMSNorm { get }
    var pipelineShardLMHead: Linear? { get }

    func pipelineShardSSMMask(hiddenStates: MLXArray, cache: (any KVCache)?) -> MLXArray?
}

public extension PipelineShardableLanguageModel {
    func pipelineShardSSMMask(hiddenStates: MLXArray, cache: (any KVCache)?) -> MLXArray? {
        createSSMMask(h: hiddenStates, cache: cache as? MambaCache)
    }
}
