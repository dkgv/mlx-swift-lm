// Copyright © 2026 Apple Inc.
//
// Pins streamed experts equal to resident experts. A shard that streams
// materialises only the experts a pass routes to and renumbers the gather to
// match; if the compaction and the slicing ever disagree, the model keeps
// running and silently multiplies by the wrong expert, so equality against the
// resident path is the only thing that catches it.

import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

/// Serves slices out of a full stacked weight held in memory. The real provider
/// reads them off disk; what is under test here is the roster/renumbering
/// contract, which is identical either way.
private final class ArrayBackedProvider: ExpertWeightProviding, @unchecked Sendable {
    let weights: [String: StackedExpertWeights]
    private(set) var requestedRosters: [[Int]] = []

    init(weights: [String: StackedExpertWeights]) {
        self.weights = weights
    }

    func preflight() throws {}

    func stackedExperts(path: String, experts: [Int]) -> StackedExpertWeights {
        requestedRosters.append(experts)
        let full = weights[path]!
        let indices = MLXArray(experts.map(Int32.init))
        return StackedExpertWeights(
            weight: full.weight[indices],
            scales: full.scales.map { $0[indices] },
            biases: full.biases.map { $0[indices] }
        )
    }
}

final class SwitchGLUStreamingTests: XCTestCase {

    private let inputDims = 32
    private let hiddenDims = 64
    private let numExperts = 16

    private func makeGLU() -> SwitchGLU {
        SwitchGLU(inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
    }

    /// Copies each projection's stacked parameters out of `glu`, so a provider
    /// can serve exactly what the resident path would have gathered.
    private func provider(for glu: SwitchGLU) -> ArrayBackedProvider {
        var weights: [String: StackedExpertWeights] = [:]
        for (name, linear) in [
            ("gate_proj", glu.gateProj), ("up_proj", glu.upProj), ("down_proj", glu.downProj),
        ] {
            let quantized = linear as? QuantizedSwitchLinear
            weights[name] = StackedExpertWeights(
                weight: linear.weight,
                scales: quantized?.scales,
                biases: quantized?.biases
            )
        }
        return ArrayBackedProvider(weights: weights)
    }

    private func enableStreaming(on glu: SwitchGLU, using provider: ArrayBackedProvider) {
        for (name, linear) in [
            ("gate_proj", glu.gateProj), ("up_proj", glu.upProj), ("down_proj", glu.downProj),
        ] {
            linear.expertProvider = provider
            linear.expertPath = name
        }
    }

    // MARK: - Roster

    func testRosterIsUniqueAscendingAndRenumbersIndices() {
        let indices = MLXArray([Int32(7), 3, 7, 0, 3], [1, 5])
        let roster = ExpertRoster.make(from: indices)

        XCTAssertEqual(roster.experts, [0, 3, 7])
        XCTAssertEqual(roster.compactIndices.asArray(Int32.self), [2, 1, 2, 0, 1])
        XCTAssertEqual(roster.compactIndices.shape, indices.shape)
        XCTAssertEqual(roster.compactIndices.dtype, indices.dtype)
    }

    /// The gather is told `sortedIndices: true` on the large path, which is only
    /// valid if compaction preserves order. It does because the roster is
    /// ascending, so the remap is monotonic.
    func testCompactionPreservesSortedOrder() {
        let sorted = MLXArray([Int32(1), 1, 4, 9, 9, 12])
        let roster = ExpertRoster.make(from: sorted)
        let compact = roster.compactIndices.asArray(Int32.self)

        XCTAssertEqual(compact, compact.sorted())
    }

    func testRosterHandlesUInt32RouterIndices() {
        let indices = MLXArray([UInt32(5), 2, 5])
        let roster = ExpertRoster.make(from: indices)

        XCTAssertEqual(roster.experts, [2, 5])
        XCTAssertEqual(roster.compactIndices.dtype, .uint32)
        XCTAssertEqual(roster.compactIndices.asArray(Int32.self), [1, 0, 1])
    }

    // MARK: - Equivalence

    func testStreamedMatchesResidentUnsorted() {
        // Under 64 indices, so SwitchGLU skips gatherSort.
        assertStreamedMatchesResident(tokens: 4, topK: 2, quantized: false)
    }

    func testStreamedMatchesResidentSorted() {
        // Over 64 indices, so the sorted gather path runs.
        assertStreamedMatchesResident(tokens: 48, topK: 4, quantized: false)
    }

    func testStreamedMatchesResidentQuantized() {
        assertStreamedMatchesResident(tokens: 48, topK: 4, quantized: true)
    }

    /// A pass that routes to every expert must still agree — that is the worst
    /// case for the roster (no compaction to do) and the one prefill hits.
    func testStreamedMatchesResidentWhenAllExpertsRouted() {
        assertStreamedMatchesResident(tokens: numExperts, topK: numExperts, quantized: true)
    }

    private func assertStreamedMatchesResident(
        tokens: Int, topK: Int, quantized: Bool,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        MLXRandom.seed(0)
        let glu = makeGLU()
        if quantized {
            quantize(model: glu, groupSize: 32, bits: 4)
        }

        let x = MLXRandom.normal([1, tokens, inputDims])
        let indices = MLXRandom.randInt(0 ..< numExperts, [1, tokens, topK]).asType(.uint32)

        let resident = glu(x, indices)
        eval(resident)

        let provider = provider(for: glu)
        enableStreaming(on: glu, using: provider)
        let streamed = glu(x, indices)
        eval(streamed)

        XCTAssertEqual(streamed.shape, resident.shape, file: file, line: line)
        let maxDelta = (streamed - resident).abs().max().item(Float.self)
        XCTAssertEqual(maxDelta, 0, accuracy: 1e-6, file: file, line: line)

        // All three projections share one roster: three requests, same ids.
        XCTAssertEqual(provider.requestedRosters.count, 3, file: file, line: line)
        XCTAssertEqual(
            Set(provider.requestedRosters.map { $0 }).count, 1,
            "projections disagreed on the roster", file: file, line: line)
    }

    // MARK: - Installation

    /// The whole point: after installing a provider the stacked parameters are
    /// gone. If they stay, a shard still pays full expert residency and the
    /// device still runs out of memory.
    func testEnableExpertStreamingDropsResidentStack() {
        let glu = makeGLU()
        quantize(model: glu, groupSize: 32, bits: 4)
        let provider = provider(for: glu)

        XCTAssertEqual(glu.gateProj.weight.dim(0), numExperts)

        glu.enableExpertStreaming(provider: provider, path: "switch_mlp")

        for linear in [glu.gateProj, glu.upProj, glu.downProj] {
            XCTAssertEqual(linear.weight.dim(0), 1, "stacked weight was retained")
            XCTAssertEqual(
                (linear as? QuantizedSwitchLinear)?.scales.dim(0), 1,
                "stacked scales were retained")
        }
        XCTAssertTrue(glu.isStreaming)
    }

    func testEnableExpertStreamingNamesEachProjection() {
        let glu = makeGLU()
        glu.enableExpertStreaming(
            provider: provider(for: glu), path: "language_model.model.layers.3.mlp.switch_mlp")

        XCTAssertEqual(
            glu.gateProj.expertPath, "language_model.model.layers.3.mlp.switch_mlp.gate_proj")
        XCTAssertEqual(
            glu.upProj.expertPath, "language_model.model.layers.3.mlp.switch_mlp.up_proj")
        XCTAssertEqual(
            glu.downProj.expertPath, "language_model.model.layers.3.mlp.switch_mlp.down_proj")
    }

    /// Without a provider nothing changes — the resident path must not start
    /// paying for roster construction or a host round-trip.
    func testNonStreamingGLUReportsNotStreaming() {
        let glu = makeGLU()
        XCTAssertFalse(glu.isStreaming)

        enableStreaming(on: glu, using: provider(for: glu))
        XCTAssertTrue(glu.isStreaming)
    }
}
