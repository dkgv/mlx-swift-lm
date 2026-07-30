// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLXLMCommon

final class LoadWeightsTests: XCTestCase {

    func testLoadWeightsUsesSafetensorsIndexWeightMapWhenPresent() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("mtp.safetensors", in: directory)
        try writeEmptyFile("optiq_vision.safetensors", in: directory)
        try """
        {
          "metadata": { "total_size": 1 },
          "weight_map": {
            "model.norm.weight": "model.safetensors"
          }
        }
        """.data(using: .utf8)!.write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model.safetensors"])
    }

    func testShardLoadSkipsIndexedFilesItNeverStaged() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // A shard stages only the files holding its own layers; the rest are named by the index
        // but never fetched.
        try writeEmptyFile("model-00001-of-00002.safetensors", in: directory)
        try writeShardedIndex(in: directory)

        let names = try weightURLsToLoad(in: directory, isShardLoad: true)
            .map(\.lastPathComponent)

        XCTAssertEqual(names, ["model-00001-of-00002.safetensors"])
    }

    func testFullLoadKeepsMissingIndexedFilesSoTheyStillError() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model-00001-of-00002.safetensors", in: directory)
        try writeShardedIndex(in: directory)

        let names = try weightURLsToLoad(in: directory, isShardLoad: false)
            .map(\.lastPathComponent)

        XCTAssertEqual(
            names, ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    }

    private func writeShardedIndex(in directory: URL) throws {
        try """
        {
          "metadata": { "total_size": 2 },
          "weight_map": {
            "model.layers.0.mlp.down_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.mlp.down_proj.weight": "model-00002-of-00002.safetensors"
          }
        }
        """.data(using: .utf8)!.write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("LoadWeightsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func writeEmptyFile(_ name: String, in directory: URL) throws {
        try Data().write(to: directory.appendingPathComponent(name))
    }
}
