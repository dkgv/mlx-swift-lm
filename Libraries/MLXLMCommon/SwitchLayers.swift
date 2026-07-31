import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, up in
    MLXNN.silu(gate) * up
}

public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        // One roster for all three projections: they route identically, so
        // compacting once keeps their stacked weights on a shared numbering.
        let roster = isStreaming ? ExpertRoster.make(from: idx) : nil
        let gatherIdx = roster?.compactIndices ?? idx

        let xUp = upProj(x, gatherIdx, sortedIndices: doSort, roster: roster)
        let xGate = gateProj(x, gatherIdx, sortedIndices: doSort, roster: roster)
        let activated =
            if let activationProduct {
                activationProduct(xGate, xUp)
            } else {
                activation(xGate) * xUp
            }
        x = downProj(
            activated,
            gatherIdx,
            sortedIndices: doSort,
            roster: roster)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }

    /// Whether this block's experts are supplied on demand rather than held
    /// resident. See ``SwitchLinear/expertProvider``.
    public var isStreaming: Bool {
        gateProj.expertProvider != nil
    }

    /// Switches all three projections to demand-loaded experts.
    ///
    /// - Parameters:
    ///   - provider: Serves the slices for every projection.
    ///   - path: This block's module path, e.g.
    ///     `language_model.model.layers.7.mlp.switch_mlp`. Each projection is
    ///     registered under `path` plus its own key.
    public func enableExpertStreaming(provider: any ExpertWeightProviding, path: String) {
        gateProj.enableExpertStreaming(provider: provider, path: "\(path).gate_proj")
        upProj.enableExpertStreaming(provider: provider, path: "\(path).up_proj")
        downProj.enableExpertStreaming(provider: provider, path: "\(path).down_proj")
    }
}

// MARK: - FusedGateUpSwitchGLU

/// SwitchGLU variant for models that ship a single fused `gate_up_proj` weight
/// of shape `[numExperts, 2*hiddenDims, inputDims]` instead of separate
/// `gate_proj` / `up_proj`. Used by Gemma 4 26B MoE.
public class FusedGateUpSwitchGLU: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: 2 * hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: 2 * hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let gateUp = gateUpProj(x, idx, sortedIndices: doSort)
        let parts = MLX.split(gateUp, parts: 2, axis: -1)
        let activated =
            if let activationProduct {
                activationProduct(parts[0], parts[1])
            } else {
                activation(parts[0]) * parts[1]
            }
        x = downProj(
            activated,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - SwitchLinear

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    /// Supplies expert slices per forward pass instead of holding the full
    /// stack resident. When set, ``weight`` (and, on the quantized subclass,
    /// ``scales``/``biases``) is ignored and may be a placeholder — a shard
    /// that streams never loads the real parameter at all.
    ///
    /// Set together with ``expertPath``; both are required for streaming.
    public var expertProvider: (any ExpertWeightProviding)?

    /// This layer's module path, which the provider uses to locate the
    /// checkpoint tensor backing it.
    public var expertPath: String?

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false,
        roster: ExpertRoster? = nil
    ) -> MLXArray {
        let weight = streamedWeights(roster: roster)?.weight ?? self.weight
        let weightT = weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    /// Switches this layer to demand-loaded experts and drops the resident
    /// stack.
    ///
    /// The stacked parameters are replaced with single-expert placeholders. A
    /// shard that calls this before its first forward pass never materialises
    /// the real ones: they are lazy until evaluated, and nothing evaluates them
    /// once the streaming path is taking over the gather.
    ///
    /// - Parameters:
    ///   - provider: Serves the slices. Preflighted by the caller.
    ///   - path: This layer's module path, e.g.
    ///     `language_model.model.layers.7.mlp.switch_mlp.gate_proj`.
    public func enableExpertStreaming(provider: any ExpertWeightProviding, path: String) {
        self.expertProvider = provider
        self.expertPath = path
        releaseResidentExperts()
    }

    /// Swaps the stacked parameters for one-expert stand-ins. Shapes past the
    /// expert axis are preserved so anything inspecting the layer still sees a
    /// coherent module; the values are never read.
    func releaseResidentExperts() {
        var placeholders = [("weight", Self.oneExpert(like: weight))]
        if let bias {
            placeholders.append(("bias", Self.oneExpert(like: bias)))
        }
        _ = update(parameters: ModuleParameters.unflattened(placeholders))
    }

    static func oneExpert(like array: MLXArray) -> MLXArray {
        MLXArray.zeros([1] + array.shape.dropFirst(), dtype: array.dtype)
    }

    /// The slices for `roster`, or `nil` when this layer holds its experts
    /// resident.
    func streamedWeights(roster: ExpertRoster?) -> StackedExpertWeights? {
        guard let expertProvider, let expertPath, let roster else { return nil }
        return expertProvider.stackedExperts(path: expertPath, experts: roster.experts)
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false,
        roster: ExpertRoster? = nil
    ) -> MLXArray {
        let streamed = streamedWeights(roster: roster)
        var result = MLX.gatherQuantizedMM(
            x,
            streamed?.weight ?? self.weight,
            scales: streamed?.scales ?? self.scales,
            biases: streamed?.biases ?? self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    override func releaseResidentExperts() {
        super.releaseResidentExperts()
        var placeholders = [("scales", Self.oneExpert(like: scales))]
        if let biases {
            placeholders.append(("biases", Self.oneExpert(like: biases)))
        }
        _ = update(parameters: ModuleParameters.unflattened(placeholders))
    }
}
