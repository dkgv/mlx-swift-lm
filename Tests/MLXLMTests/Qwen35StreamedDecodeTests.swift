// Copyright © 2026 Apple Inc.
//
// A streamed MoE reads its router's picks back to the host mid-forward. That
// eval traps inside a trace, so no compiled decode path — the block's own, the
// enclosing layer's, or the model's segment schedule — may be taken by a layer
// whose experts stream. These tests decode with streaming enabled, which is
// what a shard does after prefill.

import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Serves slices out of the stacked weights captured before streaming was
/// enabled, so the streamed path must reproduce the resident path exactly.
private final class ResidentSliceProvider: ExpertWeightProviding, @unchecked Sendable {
    private let weights: [String: MLXArray]

    init(weights: [String: MLXArray]) {
        self.weights = weights
    }

    func preflight() throws {}

    func stackedExperts(path: String, experts: [Int]) -> StackedExpertWeights {
        StackedExpertWeights(weight: weights[path]![MLXArray(experts.map(Int32.init))])
    }
}

final class Qwen35StreamedDecodeTests: XCTestCase {

    /// Layer 0 GDN, layer 1 full attention, MoE mlp in both — every compiled
    /// decode path the model installs is reachable from this config.
    private func tinyMoEConfiguration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 16,
                "shared_expert_intermediate_size": 16
            }
            """
        return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    /// Points every MoE block in `model` at a provider serving its own experts.
    /// Weights are captured first: enabling streaming drops the resident stack.
    private func enableStreaming(on model: Qwen35TextModel) {
        for (index, block) in model.modules().compactMap({ $0 as? Qwen35SparseMoeBlock })
            .enumerated()
        {
            let path = "layers.\(index).mlp.switch_mlp"
            let glu = block.switchMLP
            let weights = [
                "\(path).gate_proj": copy(of: glu.gateProj.weight),
                "\(path).up_proj": copy(of: glu.upProj.weight),
                "\(path).down_proj": copy(of: glu.downProj.weight),
            ]
            glu.enableExpertStreaming(provider: ResidentSliceProvider(weights: weights), path: path)
        }
    }

    /// `update(parameters:)` rewrites a parameter in place, so a provider that
    /// held the module's own array would be serving the placeholder that
    /// replaced it. The copy is taken while the real stack is still there.
    private func copy(of weight: MLXArray) -> MLXArray {
        MLXArray(weight.asArray(Float.self), weight.shape)
    }

    private func decode(_ model: Qwen35TextModel, tokens: [Int32]) -> MLXArray {
        let cache = model.newCache(parameters: nil)
        var logits = MLXArray()
        for token in tokens {
            logits = model(MLXArray([token]).reshaped(1, 1), cache: cache)
            eval(logits)
        }
        return logits
    }

    private func prefill(_ model: Qwen35TextModel, tokens: [Int32]) -> MLXArray {
        let logits = model(
            MLXArray(tokens).reshaped(1, tokens.count), cache: model.newCache(parameters: nil))
        eval(logits)
        return logits
    }

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    /// Two decode steps: the first would install the compiled closures, the
    /// second would replay them. Both have to stay uncompiled while streaming —
    /// before the guards, the roster's readback trapped inside the trace.
    func testStreamedDecodeMatchesResidentDecode() throws {
        let model = Qwen35TextModel(try tinyMoEConfiguration())
        let tokens: [Int32] = [1, 2]

        let resident = decode(model, tokens: tokens)
        enableStreaming(on: model)
        let streamed = decode(model, tokens: tokens)

        XCTAssertLessThan(
            maxDifference(streamed, resident), 1e-4,
            "streamed decode diverged from the resident path")
    }

    /// The multi-token path was never traced, so it is the control: streaming
    /// must not change it either.
    func testStreamedPrefillMatchesResidentPrefill() throws {
        let model = Qwen35TextModel(try tinyMoEConfiguration())
        let tokens: [Int32] = [1, 2]

        let resident = prefill(model, tokens: tokens)
        enableStreaming(on: model)
        let streamed = prefill(model, tokens: tokens)

        XCTAssertLessThan(
            maxDifference(streamed, resident), 1e-4,
            "streamed prefill diverged from the resident path")
    }

    func testStreamedLayersReportStreamingMLP() throws {
        let model = Qwen35TextModel(try tinyMoEConfiguration())
        let blocks = model.modules().compactMap { $0 as? Qwen35SparseMoeBlock }
        XCTAssertFalse(blocks.isEmpty, "expected MoE blocks in the config")
        XCTAssertFalse(blocks.contains { $0.switchMLP.isStreaming })

        enableStreaming(on: model)

        XCTAssertTrue(blocks.allSatisfy { $0.switchMLP.isStreaming })
    }
}
