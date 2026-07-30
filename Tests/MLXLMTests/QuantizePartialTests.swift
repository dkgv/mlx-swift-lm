import MLX
import MLXLMCommon
import MLXNN
import XCTest

private class ShardTestBlock: Module {
    @ModuleInfo var proj: Linear

    init(dimensions: Int) {
        self.proj = Linear(dimensions, dimensions, bias: false)
        super.init()
    }
}

private class ShardTestModel: Module {
    @ModuleInfo var layers: [ShardTestBlock]

    init(layerCount: Int, dimensions: Int) {
        self.layers = (0 ..< layerCount).map { _ in ShardTestBlock(dimensions: dimensions) }
        super.init()
    }
}

final class QuantizePartialTests: XCTestCase {
    private static let dimensions = 64

    private func layerIndex(in path: String) -> Int? {
        path.split(separator: ".").dropFirst().first.flatMap { Int($0) }
    }

    /// The replacement tree `MLXNN.quantize` would build for a shard owning `range`.
    private func quantizationUpdates(
        for model: ShardTestModel, range: Range<Int>
    ) -> ModuleChildren {
        let updates =
            model
            .leafModules()
            .flattened()
            .compactMap { (path, m) -> (String, Module)? in
                guard let layer = layerIndex(in: path), range.contains(layer),
                    let quantized = quantizeSingle(layer: m, groupSize: 64, bits: 4)
                else {
                    return nil
                }
                return (path, quantized)
            }
        return ModuleChildren.unflattened(updates)
    }

    private func quantize(_ model: ShardTestModel, range: Range<Int>) {
        quantizePartial(model: model) { path, _ in
            guard let layer = layerIndex(in: path), range.contains(layer) else { return nil }
            return (groupSize: 64, bits: 4, mode: .affine)
        }
    }

    func testShardAboveLayerZeroLeavesLeadingHoles() {
        let model = ShardTestModel(layerCount: 4, dimensions: Self.dimensions)

        // A shard owning layers 2..<4 quantizes nothing at index 0, so the array it produces
        // starts with a hole. This is what `MLXNN.quantize` would hand to `update(modules:)`
        // inside a `try!`, which turns the error below into a trap.
        XCTAssertThrowsError(
            try model.update(
                modules: quantizationUpdates(for: model, range: 2 ..< 4), verify: .none))
    }

    func testQuantizePartialAppliesShardAboveLayerZero() {
        let model = ShardTestModel(layerCount: 4, dimensions: Self.dimensions)

        quantize(model, range: 2 ..< 4)

        XCTAssertFalse(model.layers[0].proj is QuantizedLinear)
        XCTAssertFalse(model.layers[1].proj is QuantizedLinear)
        XCTAssertTrue(model.layers[2].proj is QuantizedLinear)
        XCTAssertTrue(model.layers[3].proj is QuantizedLinear)
    }

    func testQuantizePartialAppliesShardStartingAtLayerZero() {
        let model = ShardTestModel(layerCount: 4, dimensions: Self.dimensions)

        quantize(model, range: 0 ..< 2)

        XCTAssertTrue(model.layers[0].proj is QuantizedLinear)
        XCTAssertTrue(model.layers[1].proj is QuantizedLinear)
        XCTAssertFalse(model.layers[2].proj is QuantizedLinear)
        XCTAssertFalse(model.layers[3].proj is QuantizedLinear)
    }

    func testQuantizePartialMatchesQuantizeForAWholeModel() {
        let model = ShardTestModel(layerCount: 4, dimensions: Self.dimensions)

        quantize(model, range: 0 ..< 4)

        XCTAssertTrue(model.layers.allSatisfy { $0.proj is QuantizedLinear })
    }
}
