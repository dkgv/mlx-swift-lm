// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

private struct SafetensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

package func safetensorWeightURLs(in modelDirectory: URL) throws -> [URL] {
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    if FileManager.default.fileExists(atPath: indexURL.path) {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        return Set(index.weightMap.values)
            .sorted()
            .map { modelDirectory.appendingPathComponent($0) }
    }

    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    return enumerator.compactMap { item -> URL? in
        guard let url = item as? URL, url.pathExtension == "safetensors" else {
            return nil
        }
        return url
    }
}

/// The weight files a load should read.
///
/// A shard only materialises the weight files its own layers live in, so `model.safetensors.index.json`
/// will name files this device never fetched. Those are skipped for a shard load. A full load keeps
/// every entry, so a genuinely missing file still surfaces as an open failure rather than turning
/// into a confusing missing-parameter error later.
package func weightURLsToLoad(in modelDirectory: URL, isShardLoad: Bool) throws -> [URL] {
    let urls = try safetensorWeightURLs(in: modelDirectory)
    guard isShardLoad else { return urls }
    return urls.filter { FileManager.default.fileExists(atPath: $0.path) }
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads model weight `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
///
/// - Parameters:
///   - keyFilter: When provided, only weight keys satisfying the predicate are assigned to
///     the model. Non-matching keys are discarded after loading; the corresponding
///     parameters remain as lazy, un-evaluated init tensors. Pass `nil` (the default) for
///     a full-model load.
///   - skipEval: When `true`, the final `eval(model)` is omitted. The caller is responsible
///     for evaluating whichever parameters it needs. Use with `keyFilter` so that only the
///     relevant parameters are ever materialised.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    keyFilter: ((String) -> Bool)? = nil,
    skipEval: Bool = false
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    for url in try weightURLsToLoad(in: modelDirectory, isShardLoad: keyFilter != nil) {
        let (w, m) = try loadArraysAndMetadata(url: url)
        for (key, value) in w {
            weights[key] = value
        }
        if metadata.isEmpty {
            metadata = m
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // drop keys outside this shard before quantizing and updating the model
    if let keyFilter {
        weights = weights.filter { keyFilter($0.key) }
    }

    // quantize if needed
    //
    // `quantizePartial` rather than `quantize`: with a key filter only this shard's layers have
    // scales, so only they are quantized, and a shard starting above layer zero would otherwise
    // hand `update(modules:)` a `layers` array with leading holes and trap.
    if quantization != nil || perLayerQuantization != nil {
        quantizePartial(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights; when a filter is active not all model keys will be present,
    // so relax verification to only check that every key in the dict matched a parameter
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: keyFilter == nil ? [.all] : .noUnusedKeys)

    if !skipEval {
        eval(model)
    }
}
