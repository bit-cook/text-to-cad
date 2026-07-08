import {
  buildComposedPackageMeshData
} from "../lib/assembly/meshData.js";
import { buildMeshDataFromGlbBuffer } from "../lib/render/glbMeshData.js";
import { buildMeshDataFromStlBuffer } from "../lib/render/stlMeshData.js";
import { buildMeshDataFrom3MfBuffer } from "../lib/render/threeMfMeshData.js";
import {
  loadRenderDisplayEdgeBundle,
  loadRenderGlb,
  loadRenderSelectorBundle
} from "../lib/stepRenderAssetClient.js";
import {
  buildDisplayEdgeRuntime,
  buildSelectorRuntime
} from "../lib/selectors/runtime.js";
import {
  loadStepModuleDefinition
} from "./stepModule.js";
import {
  hasStepParameterRenderValues,
  normalizeStepParameterRenderValues,
  stepParameterRenderFrameState,
  stepParameterRenderFrameValues
} from "./stepParameters.js";

export const SOURCE_KIND = Object.freeze({
  STEP: "step",
  STP: "stp",
  GLB: "glb",
  STL: "stl",
  THREE_MF: "3mf",
  UNKNOWN: "unknown"
});

function isObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizeKind(value = "") {
  const kind = String(value || "").trim().toLowerCase();
  if (kind === "step" || kind === "stp" || kind === "glb" || kind === "stl" || kind === "3mf") {
    return kind;
  }
  return SOURCE_KIND.UNKNOWN;
}

function sourceKindFromUrl(url = "") {
  const pathname = String(url || "").split(/[?#]/, 1)[0].toLowerCase();
  if (pathname.endsWith(".step")) {
    return SOURCE_KIND.STEP;
  }
  if (pathname.endsWith(".stp")) {
    return SOURCE_KIND.STP;
  }
  if (pathname.endsWith(".glb") || pathname.endsWith(".gltf")) {
    return SOURCE_KIND.GLB;
  }
  if (pathname.endsWith(".stl")) {
    return SOURCE_KIND.STL;
  }
  if (pathname.endsWith(".3mf")) {
    return SOURCE_KIND.THREE_MF;
  }
  return SOURCE_KIND.UNKNOWN;
}

export function sourceIsStep(sourceOrKind) {
  const kind = typeof sourceOrKind === "string" ? sourceOrKind : sourceOrKind?.kind;
  const normalized = normalizeKind(kind);
  return normalized === SOURCE_KIND.STEP || normalized === SOURCE_KIND.STP;
}

function assertStepOnlyOption(kind, value, label) {
  // An empty value means the option was not provided (stepParameterUrl defaults to
  // the empty string), so there is nothing step-only to reject — required for direct
  // non-STEP mesh sources, which reach loadSource with no step parameters at all.
  if (value === undefined || value === null || value === "") {
    return;
  }
  if (!sourceIsStep(kind)) {
    throw new Error(`${label} is supported only for STEP/STP sources`);
  }
}

async function loadStepMeshFromGlb(glbUrl) {
  // A plain (non-package) GLB URL is a single mesh blob — assemblies are component-GLB
  // packages loaded via loadPackageMeshData, not self-contained monolith GLBs.
  return loadRenderGlb(glbUrl);
}

async function loadSelectorRuntime(glbUrl, { cadPath = "" } = {}) {
  if (!glbUrl) {
    return null;
  }
  try {
    const selectorBundle = await loadRenderSelectorBundle(glbUrl);
    return buildSelectorRuntime(selectorBundle, {
      copyCadPath: cadPath
    });
  } catch {
    return null;
  }
}

async function loadDisplayEdgeRuntime(glbUrl) {
  if (!glbUrl) {
    return null;
  }
  try {
    return buildDisplayEdgeRuntime(await loadRenderDisplayEdgeBundle(glbUrl));
  } catch {
    return null;
  }
}

async function loadPackageMeshData(packageInfo) {
  const descriptor = isObject(packageInfo.descriptor) ? packageInfo.descriptor : null;
  if (!descriptor) {
    throw new Error("Assembly package render job is missing its descriptor");
  }
  const componentUrls = isObject(packageInfo.componentUrls) ? packageInfo.componentUrls : {};
  const components = isObject(descriptor.components) ? descriptor.components : {};
  const componentMeshDataByCid = {};
  for (const cid of Object.keys(components)) {
    const url = String(componentUrls[cid] || "").trim();
    if (!url) {
      throw new Error(`Assembly package component ${cid} has no resolved URL`);
    }
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load component GLB ${cid}: HTTP ${response.status}`);
    }
    componentMeshDataByCid[cid] = await buildMeshDataFromGlbBuffer(await response.arrayBuffer());
  }
  return buildComposedPackageMeshData(descriptor, componentMeshDataByCid);
}

async function loadMeshDataFromUrl(url, kind) {
  if (sourceIsStep(kind)) {
    return loadStepMeshFromGlb(url);
  }
  if (kind === SOURCE_KIND.GLB) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load GLB source: HTTP ${response.status}`);
    }
    return buildMeshDataFromGlbBuffer(await response.arrayBuffer());
  }
  if (kind === SOURCE_KIND.STL) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load STL source: HTTP ${response.status}`);
    }
    return buildMeshDataFromStlBuffer(await response.arrayBuffer());
  }
  if (kind === SOURCE_KIND.THREE_MF) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load 3MF source: HTTP ${response.status}`);
    }
    return buildMeshDataFrom3MfBuffer(await response.arrayBuffer());
  }
  throw new Error(`Unsupported render source kind: ${kind || SOURCE_KIND.UNKNOWN}`);
}

async function loadStepParameters({
  kind,
  stepParameters,
  stepParameterUrl,
  cadPath,
  selectorRuntime
}) {
  assertStepOnlyOption(kind, stepParameters, "stepParameters");
  assertStepOnlyOption(kind, stepParameterUrl, "stepParameterUrl");
  const explicit = hasStepParameterRenderValues(stepParameters);
  if (!stepParameterUrl) {
    if (!explicit) {
      return null;
    }
    throw new Error("STEP render parameters require resolved.stepParameterUrl");
  }
  const definition = await loadStepModuleDefinition(stepParameterUrl, { cadPath });
  const renderParameters = normalizeStepParameterRenderValues(definition, explicit ? stepParameters : {});
  return {
    definition,
    renderParameters,
    selectorRuntime,
    cadPath: cadPath || definition.cadPath || "",
    sourceUrl: stepParameterUrl
  };
}

export function stepParameterFrameRuntime(stepParameterSource, frameIndex = 0) {
  if (!stepParameterSource) {
    return null;
  }
  const { definition, renderParameters } = stepParameterSource;
  return {
    definition,
    selectorRuntime: stepParameterSource.selectorRuntime || null,
    parameterValues: stepParameterRenderFrameValues(definition, renderParameters, frameIndex),
    animationState: stepParameterRenderFrameState(renderParameters, frameIndex),
    cadPath: stepParameterSource.cadPath || definition.cadPath || "",
    sourceUrl: stepParameterSource.sourceUrl || definition.url || ""
  };
}

export async function loadSource(input, options = {}) {
  const inputObject = isObject(input) ? input : {};
  if (Object.prototype.hasOwnProperty.call(inputObject, "params")) {
    throw new Error("Render source jobs use stepParameters; params is no longer a shared render API field");
  }
  const resolved = isObject(inputObject.resolved) ? inputObject.resolved : {};
  const explicitMeshData = inputObject.meshData || options.meshData || (
    inputObject.vertices && inputObject.indices ? inputObject : null
  );
  const rawKind = inputObject.kind || resolved.kind || options.kind || (
    typeof input === "string" ? sourceKindFromUrl(input) : ""
  );
  const kind = normalizeKind(rawKind);
  const stepParameters = inputObject.stepParameters ?? options.stepParameters;
  const stepParameterUrl = String(
    inputObject.stepParameterUrl || resolved.stepParameterUrl || options.stepParameterUrl || ""
  ).trim();
  const cadPath = String(inputObject.cadPath || resolved.inputPath || options.cadPath || "").trim();
  assertStepOnlyOption(kind, stepParameters, "stepParameters");
  assertStepOnlyOption(kind, stepParameterUrl, "stepParameterUrl");

  let meshData = explicitMeshData;
  // Component-GLB package: the canonical assembly artifact is a directory, so there is
  // no single GLB to load. Fetch each unique component GLB and compose them in world
  // space from the descriptor (transforms baked per occurrence). Picking is not wired
  // for packages yet, so the selector runtime is left empty (renders, no selection).
  const packageInfo = isObject(inputObject.package) ? inputObject.package : (
    isObject(resolved.package) ? resolved.package : null
  );
  if (!meshData && packageInfo) {
    meshData = await loadPackageMeshData(packageInfo);
    const packageSelectorRuntime = inputObject.selectorRuntime || options.selectorRuntime || null;
    return {
      kind: "step",
      meshData,
      selectorRuntime: packageSelectorRuntime,
      displayEdgeRuntime: inputObject.displayEdgeRuntime || options.displayEdgeRuntime || null,
      // Parameter sidecars resolve features against composed occurrence ids, so
      // they stay fully functional for package sources even without a selector
      // runtime (feature refs prefix-match meshData part occurrence ids).
      stepParameterSource: await loadStepParameters({
        kind: "step",
        stepParameters,
        stepParameterUrl,
        cadPath,
        selectorRuntime: packageSelectorRuntime
      }),
      resolved,
      url: "",
      glbUrl: "",
      cadPath
    };
  }
  const glbUrl = String(inputObject.glbUrl || resolved.glbUrl || options.glbUrl || "").trim();
  const url = String(typeof input === "string" ? input : inputObject.url || resolved.url || glbUrl || "").trim();
  if (!meshData) {
    if (!url) {
      throw new Error("loadSource requires meshData, a source URL, or resolved.glbUrl");
    }
    meshData = await loadMeshDataFromUrl(sourceIsStep(kind) ? glbUrl || url : url, kind);
  }

  // Selector/display-edge runtimes ride in STEP topology GLB extras. Direct mesh
  // kinds have none, and loading them anyway re-downloads the mesh binary just to
  // fail the GLB container parse — gate by kind so "no selectors for meshes" is
  // intent, not a swallowed error (matches the CLI's mesh-input validation).
  const stepSidecarsEnabled = sourceIsStep(kind);
  const selectorRuntime = inputObject.selectorRuntime || options.selectorRuntime || (
    stepSidecarsEnabled ? await loadSelectorRuntime(glbUrl || url, { cadPath }) : null
  );
  const displayEdgeRuntime = inputObject.displayEdgeRuntime || options.displayEdgeRuntime || (
    stepSidecarsEnabled ? await loadDisplayEdgeRuntime(glbUrl || url) : null
  );
  const stepParameterSource = await loadStepParameters({
    kind,
    stepParameters,
    stepParameterUrl,
    cadPath,
    selectorRuntime
  });

  return {
    kind,
    meshData,
    selectorRuntime,
    displayEdgeRuntime,
    stepParameterSource,
    resolved,
    url,
    glbUrl,
    cadPath
  };
}
