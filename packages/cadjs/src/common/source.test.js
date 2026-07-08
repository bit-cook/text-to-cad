import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";

import { loadSource } from "./source.js";

function meshData() {
  return {
    vertices: new Float32Array([
      0, 0, 0,
      1, 0, 0,
      0, 1, 0
    ]),
    indices: new Uint32Array([0, 1, 2]),
    bounds: {
      min: [0, 0, 0],
      max: [1, 1, 0]
    },
    parts: []
  };
}

async function withTempModule(callback) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "render-source-test-"));
  try {
    const modulePath = path.join(root, "part.step.mjs");
    fs.writeFileSync(modulePath, `
      export default {
        manifest: {
          schemaVersion: 1,
          parameters: {
            drive: { type: "number", min: 0, max: 360, default: 0 }
          }
        }
      };
    `);
    return await callback(pathToFileURL(modulePath).href);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}

test("loadSource rejects the removed shared params field", async () => {
  await assert.rejects(
    () => loadSource({
      kind: "step",
      meshData: meshData(),
      params: { drive: 90 }
    }),
    /use stepParameters/
  );
});

test("loadSource rejects STEP parameter options for non-STEP sources", async () => {
  await assert.rejects(
    () => loadSource({
      kind: "glb",
      meshData: meshData(),
      stepParameters: { drive: 90 }
    }),
    /stepParameters is supported only for STEP\/STP sources/
  );
  await assert.rejects(
    () => loadSource({
      kind: "glb",
      meshData: meshData(),
      stepParameterUrl: "file:///tmp/part.step.mjs"
    }),
    /stepParameterUrl is supported only for STEP\/STP sources/
  );
});

test("loadSource accepts STEP parameter sidecars for STEP sources", async () => {
  await withTempModule(async (stepParameterUrl) => {
    const source = await loadSource({
      kind: "step",
      meshData: meshData(),
      cadPath: "part.step",
      stepParameterUrl,
      stepParameters: { drive: 90 }
    });

    assert.equal(source.kind, "step");
    assert.equal(source.stepParameterSource.renderParameters.values.drive, 90);
    assert.equal(source.stepParameterSource.cadPath, "part.step");
  });
});

function binaryStlTriangle() {
  // 80-byte header + uint32 triangle count + one 50-byte triangle record.
  const buffer = new ArrayBuffer(84 + 50);
  const view = new DataView(buffer);
  view.setUint32(80, 1, true); // triangle count
  const floats = [
    0, 0, 1, // normal
    0, 0, 0, // v1
    1, 0, 0, // v2
    0, 1, 0 // v3
  ];
  let offset = 84;
  for (const value of floats) {
    view.setFloat32(offset, value, true);
    offset += 4;
  }
  // trailing uint16 attribute byte count left as 0
  return buffer;
}

test("loadSource builds mesh data from an STL url", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(binaryStlTriangle(), { status: 200 });
  try {
    const source = await loadSource("/models/part.stl");
    assert.equal(source.kind, "stl");
    assert.equal(source.stepParameterSource, null);
    assert.equal(source.selectorRuntime, null);
    assert.equal(source.displayEdgeRuntime, null);
    assert.ok(source.meshData.vertices.length >= 9);
    assert.ok(source.meshData.indices.length >= 3);
    assert.equal(source.meshData.sourceFormat, "stl");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("loadSource fetches a direct mesh exactly once (no STEP sidecar loads)", async () => {
  const originalFetch = globalThis.fetch;
  const fetchedUrls = [];
  globalThis.fetch = async (url) => {
    fetchedUrls.push(String(url));
    return new Response(binaryStlTriangle(), { status: 200 });
  };
  try {
    const source = await loadSource("/models/part.stl");
    // Selector/display-edge runtimes are STEP topology sidecars; loading them for a
    // mesh kind re-downloads the binary just to fail the GLB parse. One fetch total.
    assert.deepEqual(fetchedUrls, ["/models/part.stl"]);
    assert.equal(source.selectorRuntime, null);
    assert.equal(source.displayEdgeRuntime, null);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("loadSource surfaces STL fetch failures", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response("nope", { status: 404 });
  try {
    await assert.rejects(() => loadSource("/models/missing.stl"), /Failed to load STL source: HTTP 404/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("loadSource routes 3MF sources through the non-step return path", async () => {
  const source = await loadSource({ kind: "3mf", meshData: meshData() });
  assert.equal(source.kind, "3mf");
  assert.equal(source.stepParameterSource, null);
  assert.equal(source.selectorRuntime, null);
  assert.equal(source.displayEdgeRuntime, null);
});
