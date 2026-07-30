import Foundation
import MLX
import MLXNN

/// Quantize the sub-modules of a module according to a filter, tolerating a model where only
/// part of a module array is replaced.
///
/// ``MLXNN/quantize(model:filter:apply:)`` derives its replacement tree from the paths it
/// touched, and `NestedItem.unflattened` pads index gaps with holes. When a pipeline shard owns
/// a slice of the layers, only that slice is quantized, so a shard starting above layer zero
/// hands `Module.update(modules:)` a `layers` array whose leading entries are holes.
/// `update(modules:)` decides how to apply an array by inspecting its first element and reports
/// a leading hole as an unexpected structure — from inside a `try!`, so it traps rather than
/// throws. A shard starting at layer zero escapes this only because its array ends early and the
/// missing tail is zipped away.
///
/// This variant fills those holes with empty dictionaries before updating. An empty dictionary
/// means "descend into this child and change nothing", so untouched layers keep the modules they
/// were built with.
///
/// - Parameters:
///   - model: model to quantize
///   - filter: filter receiving path and module -- return a tuple of
///     `(groupSize: Int, bits: Int, mode: QuantizationMode)` or `nil` to skip quantization
///   - apply: function to attempt the quantization -- the default implementation will quantize
///     ``MLXNN/Linear`` and ``MLXNN/Embedding`` layers
/// ### See Also
/// - ``ShardSpec``
/// - ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:keyFilter:skipEval:)``
public func quantizePartial(
    model: Module,
    filter: (String, Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)?,
    apply: (Module, Int, Int, QuantizationMode) -> Module? = quantizeSingle(
        layer:groupSize:bits:mode:)
) {
    let updates =
        model
        .leafModules()
        .flattened()
        .compactMap { (path, m) -> (String, Module)? in
            if let (groupSize, bits, mode) = filter(path, m) {
                if let quantized = apply(m, groupSize, bits, mode) {
                    return (path, quantized)
                }
            }

            return nil
        }

    model.update(modules: densifyingModuleArrays(ModuleChildren.unflattened(updates)))
}

/// Replace holes in module arrays with empty dictionaries so `Module.update(modules:)` sees a
/// dense array and recurses into every element.
private func densifyingModuleArrays(_ children: ModuleChildren) -> ModuleChildren {
    guard case .dictionary(let values) = densifyingModuleArrays(children.asItem()) else {
        return children
    }
    return ModuleChildren(values: values)
}

private func densifyingModuleArrays(_ item: NestedItem<String, Module>) -> NestedItem<
    String, Module
> {
    switch item {
    case .array(let items):
        let mapped = items.map(densifyingModuleArrays)

        // Only arrays that recurse into children may be filled. An array of `.value`s is a
        // direct replacement, where a hole means "keep the original" and must be preserved.
        let recursesIntoChildren = mapped.contains {
            if case .dictionary = $0 { return true }
            return false
        }
        guard recursesIntoChildren else { return .array(mapped) }

        return .array(
            mapped.map { entry in
                if case .none = entry { return .dictionary([:]) }
                return entry
            })

    case .dictionary(let values):
        return .dictionary(values.mapValues(densifyingModuleArrays))

    case .none, .value:
        return item
    }
}
