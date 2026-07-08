from __future__ import annotations

import asyncio
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

from tests.python.support.paths import add_repo_path, repo_path


def write_package(step_path, *, entry_kind="part", source_kind="step"):
    """Materialize the canonical render artifact for ``step_path``: a SELF-CONTAINED
    component-GLB PACKAGE directory inside the per-folder cache
    (``__cadgen__/models/<step-filename>/assembly.json``) whose content-addressed component
    GLBs live in the package's own ``components/<hash>.glb`` dir. Returns the package directory
    path, mirroring ``cadgen.catalog.render_package_dir``."""
    step_path = Path(step_path)
    pkg_dir = step_path.parent / "__cadgen__" / "models" / step_path.name
    comp_dir = pkg_dir / "components"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)
    cid = hashlib.sha256(str(step_path).encode()).hexdigest()[:16]
    (comp_dir / f"{cid}.glb").write_bytes(b"component-glb")
    (pkg_dir / "assembly.json").write_text(
        json.dumps(
            {
                "kind": "assembly-package",
                "packageSchemaVersion": 2,
                "entryKind": entry_kind,
                "rootName": step_path.stem,
                "units": "mm",
                "sourceKind": source_kind,
                "stepPath": step_path.name,
                "bbox": {"min": [0, 0, 0], "max": [1, 1, 1]},
                "stats": {"occurrenceCount": 1, "shapeCount": 1},
                "components": {cid: {"glb": f"components/{cid}.glb", "contentHash": cid}},
                "occurrences": [
                    {
                        "id": "o1.1",
                        "name": "occ",
                        "component": cid,
                        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                    }
                ],
            }
        )
    )
    return pkg_dir

add_repo_path("skills/cad/scripts")

import snapshot.__main__ as snapshot_main
from snapshot.__main__ import (
    RENDER_HTML_PATH,
    RUNTIME_DIR,
    SnapshotError,
    load_job_from_options,
    parse_snapshot_args,
    resolve_render_job_packet,
    resolve_snapshot_route_file,
    timestamp_output_path,
)


class _TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def _selector_artifact(*occurrence_ids: str) -> SimpleNamespace:
    return SimpleNamespace(
        selector_bundle=SimpleNamespace(
            manifest={
                "tables": {
                    "occurrenceColumns": ["id"],
                    "shapeColumns": ["id", "occurrenceId"],
                },
                "occurrences": [[occurrence_id] for occurrence_id in occurrence_ids],
                "shapes": [],
            },
            buffers={},
        )
    )


class SnapshotCliTests(unittest.TestCase):
    def test_cli_import_does_not_import_heavy_cad_modules(self) -> None:
        skill_root = repo_path("skills/cad")
        code = (
            "import sys; sys.path.insert(0, 'scripts'); import snapshot.__main__; "
            "print('OCP.OCP' in sys.modules); "
            "print('cadgen._internal.step_scene' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=skill_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        self.assertEqual(["False", "False"], result.stdout.strip().splitlines())

    def test_shortcut_job_shape_stays_owned_by_python_cli(self) -> None:
        options = parse_snapshot_args(
            [
                "--input",
                "models/simple/cylindrical_cap.step",
                "--output",
                "tmp/cap.png",
                "--display",
                "wireframe",
                "--size-profile",
                "simple",
            ]
        )

        job = load_job_from_options(options, stdin=_TtyStringIO(), cwd=Path.cwd())

        self.assertEqual(job["input"], "models/simple/cylindrical_cap.step")
        self.assertNotIn("workspaceRoot", job)
        self.assertNotIn("rootDir", job)
        self.assertEqual(job["outputs"][0]["path"], "tmp/cap.png")
        self.assertEqual(job["display"], {"mode": "wireframe"})
        self.assertEqual(job["render"]["sizeProfile"], "simple")

    def test_shortcut_focus_and_hide_flags_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(SnapshotError, "--focus and --hide cannot be used"):
            parse_snapshot_args(
                [
                    "--input",
                    "models/assembly.step",
                    "--output",
                    "tmp/assembly.png",
                    "--focus",
                    "#o1.2",
                    "--hide=#o1.3.1",
                ]
            )

    def test_display_shortcut_accepts_cad_display_modes(self) -> None:
        for raw_mode, expected_display in [
            ("edges", {"mode": "solid"}),
            ("x-ray", {"mode": "transparent"}),
            ("hidden edges visible", {"mode": "hidden_edges"}),
            ("hidden-lines-removed", {"mode": "hidden_lines_removed"}),
            ("flat", {"mode": "unshaded"}),
            ("appearance", {"mode": "rendered"}),
            ("wire", {"mode": "wireframe"}),
        ]:
            options = parse_snapshot_args(
                [
                    "--input",
                    "models/simple/cylindrical_cap.step",
                    "--output",
                    "tmp/cap.png",
                    "--display",
                    raw_mode,
                ]
            )
            job = load_job_from_options(options, stdin=_TtyStringIO(), cwd=Path.cwd())
            self.assertEqual(job["display"], expected_display)

    def test_display_json_accepts_exploded_settings(self) -> None:
        options = parse_snapshot_args(
            [
                "--input",
                "models/simple/cylindrical_cap.step",
                "--output",
                "tmp/cap.png",
                "--display",
                '{"projection":"perspective","mode":"rendered","exploded":{"enabled":true,"axis":"radial","spacing":1.6}}',
            ]
        )
        job = load_job_from_options(options, stdin=_TtyStringIO(), cwd=Path.cwd())
        self.assertEqual(
            job["display"],
            {
                "projection": "perspective",
                "mode": "rendered",
                "exploded": {"enabled": True, "axis": "radial", "spacing": 1.6},
            },
        )

    def _display_job(self, display_json: str):
        options = parse_snapshot_args(
            [
                "--input",
                "models/simple/cylindrical_cap.step",
                "--output",
                "tmp/cap.png",
                "--display",
                display_json,
            ]
        )
        return load_job_from_options(options, stdin=_TtyStringIO(), cwd=Path.cwd())

    def test_display_json_rejects_bad_projection_value(self) -> None:
        with self.assertRaisesRegex(SnapshotError, "projection must be orthographic or perspective"):
            self._display_job('{"projection":"ortho"}')

    def test_display_json_rejects_bad_mode_value(self) -> None:
        with self.assertRaisesRegex(SnapshotError, "--display mode must be one of"):
            self._display_job('{"mode":"shadedd"}')

    def test_display_json_rejects_bad_exploded_axis_value(self) -> None:
        with self.assertRaisesRegex(SnapshotError, "exploded.axis must be one of x, y, z, radial"):
            self._display_job('{"exploded":{"axis":"q"}}')

    def test_display_json_accepts_valid_closed_set_values(self) -> None:
        self.assertEqual(self._display_job('{"projection":"orthographic"}')["display"], {"projection": "orthographic"})
        self.assertEqual(self._display_job('{"mode":"shaded"}')["display"], {"mode": "shaded"})
        # A leading '-' axis (e.g. reverse direction) stays valid.
        self.assertEqual(
            self._display_job('{"exploded":{"axis":"-z"}}')["display"],
            {"exploded": {"axis": "-z"}},
        )

    def test_display_json_treats_empty_string_values_as_unset(self) -> None:
        # The renderer treats an empty string as absent and falls back to the default, so
        # validation must not false-reject empty strings an agent emits for unset fields.
        self.assertEqual(self._display_job('{"projection":""}')["display"], {"projection": ""})
        self.assertEqual(self._display_job('{"mode":""}')["display"], {"mode": ""})
        self.assertEqual(self._display_job('{"exploded":{"axis":""}}')["display"], {"exploded": {"axis": ""}})

    def test_edge_settings_belong_to_display_json(self) -> None:
        options = parse_snapshot_args(
            [
                "--input",
                "models/simple/cylindrical_cap.step",
                "--output",
                "tmp/cap.png",
                "--display",
                '{"edges":{"enabled":false,"color":"#123456"}}',
            ]
        )
        job = load_job_from_options(options, stdin=_TtyStringIO(), cwd=Path.cwd())
        self.assertEqual(job["display"], {"edges": {"enabled": False, "color": "#123456"}})

        appearance_options = parse_snapshot_args(
            [
                "--input",
                "models/simple/cylindrical_cap.step",
                "--output",
                "tmp/cap.png",
                "--appearance",
                '{"edges":{"enabled":false}}',
            ]
        )
        with self.assertRaisesRegex(SnapshotError, "unsupported keys: edges"):
            load_job_from_options(appearance_options, stdin=_TtyStringIO(), cwd=Path.cwd())

    def test_display_shortcut_rejects_unknown_modes(self) -> None:
        options = parse_snapshot_args(
            [
                "--input",
                "models/simple/cylindrical_cap.step",
                "--output",
                "tmp/cap.png",
                "--display",
                "mist",
            ]
        )
        with self.assertRaisesRegex(SnapshotError, "Unsupported display mode"):
            load_job_from_options(options, stdin=_TtyStringIO(), cwd=Path.cwd())

    def test_display_shortcut_rejects_exploded_mode_alias(self) -> None:
        options = parse_snapshot_args(
            [
                "--input",
                "models/simple/cylindrical_cap.step",
                "--output",
                "tmp/cap.png",
                "--display",
                "exploded",
            ]
        )
        with self.assertRaisesRegex(SnapshotError, "Unsupported display mode"):
            load_job_from_options(options, stdin=_TtyStringIO(), cwd=Path.cwd())

    def test_shortcut_focus_flags_apply_selection(self) -> None:
        options = parse_snapshot_args(
            [
                "--input",
                "models/assembly.step",
                "--output",
                "tmp/assembly.png",
                "--focus",
                "#o1.2",
                "#o1.3",
            ]
        )

        job = load_job_from_options(options, stdin=_TtyStringIO(), cwd=Path.cwd())

        self.assertEqual(
            job["selection"],
            {
                "focus": ["#o1.2", "#o1.3"],
            },
        )

    def test_output_paths_are_timestamped_when_jobs_are_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            models = root / "models"
            models.mkdir()
            (models / "part.step").write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
            write_package(models / "part.step")

            original_timestamp = snapshot_main.snapshot_timestamp
            original_ensure = snapshot_main.ensure_step_topology_artifact
            try:
                snapshot_main.snapshot_timestamp = lambda: "20260527T163012Z"
                snapshot_main.ensure_step_topology_artifact = lambda *args, **kwargs: None

                packet = resolve_render_job_packet(
                    {
                        "jobs": [
                            {
                                "input": "models/part.step",
                                "outputs": [
                                    {"path": "tmp/iso.png", "camera": "iso"},
                                    {"path": "tmp/front.png", "camera": "front"},
                                ],
                            },
                            {
                                "input": "models/part.step",
                                "mode": "orbit",
                                "outputs": [{"path": "tmp/orbit.gif"}],
                            },
                        ]
                    },
                    cwd=root,
                )
            finally:
                snapshot_main.snapshot_timestamp = original_timestamp
                snapshot_main.ensure_step_topology_artifact = original_ensure

            output_paths = [
                Path(output["path"]).relative_to(root).as_posix()
                for job in packet["jobs"]
                for output in job["outputs"]
            ]

        self.assertEqual(
            output_paths,
            [
                "tmp/iso_20260527T163012Z.png",
                "tmp/front_20260527T163012Z.png",
                "tmp/orbit_20260527T163012Z.gif",
            ],
        )

    def test_render_job_derives_asset_root_from_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            models = root / "models"
            models.mkdir()
            (models / "part.step").write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
            write_package(models / "part.step")

            original_ensure = snapshot_main.ensure_step_topology_artifact
            try:
                snapshot_main.ensure_step_topology_artifact = lambda *args, **kwargs: None
                packet = resolve_render_job_packet(
                    {
                        "input": "models/part.step",
                        "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                    },
                    cwd=root,
                )
            finally:
                snapshot_main.ensure_step_topology_artifact = original_ensure

        job = packet["jobs"][0]
        self.assertNotIn("workspaceRoot", job)
        self.assertNotIn("rootDir", job)
        self.assertEqual(job["resolved"]["rootPath"], str(models))
        self.assertEqual(job["resolved"]["inputUrl"], "/__render_asset/part.step")
        # The render artifact is a SELF-CONTAINED component-GLB package, so the resolved job
        # carries a package descriptor with per-component asset URLs (no monolithic glbUrl).
        # Each component URL points into the package's own components/ dir inside __cadgen__.
        self.assertNotIn("glbUrl", job["resolved"])
        package = job["resolved"]["package"]
        self.assertEqual(package["descriptor"]["kind"], "assembly-package")
        component_urls = package["componentUrls"]
        self.assertTrue(component_urls)
        for component_url in component_urls.values():
            self.assertTrue(
                component_url.startswith("/__render_asset/__cadgen__/models/part.step/components/"),
                component_url,
            )

    def test_render_job_ensures_step_artifact_for_step_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            models = root / "models"
            models.mkdir()
            step_path = models / "part.step"
            step_path.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
            write_package(step_path)
            calls = []

            def fake_ensure(target, **kwargs):
                calls.append((target, kwargs))
                return None

            original_ensure = snapshot_main.ensure_step_topology_artifact
            try:
                snapshot_main.ensure_step_topology_artifact = fake_ensure
                resolve_render_job_packet(
                    {
                        "input": "models/part.step",
                        "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                    },
                    cwd=root,
                )
            finally:
                snapshot_main.ensure_step_topology_artifact = original_ensure

        self.assertEqual(len(calls), 1)
        target, kwargs = calls[0]
        self.assertEqual(target.step_path, step_path)
        self.assertEqual(target.source_path, step_path)
        self.assertFalse(kwargs["require_selector"])
        self.assertIsNone(kwargs["debug"])

    def test_debug_shortcut_flag_sets_job_debug_field(self) -> None:
        options = parse_snapshot_args(
            [
                "--input",
                "models/simple/cylindrical_cap.step",
                "--output",
                "tmp/cap.png",
                "--debug",
            ]
        )
        job = load_job_from_options(options, stdin=_TtyStringIO(), cwd=Path.cwd())
        self.assertTrue(job["debug"])

    def test_render_job_surfaces_step_artifact_debug_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            models = root / "models"
            models.mkdir()
            step_path = models / "part.step"
            step_path.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
            write_package(step_path)

            def fake_ensure(target, **kwargs):
                debug_info = kwargs.get("debug")
                if debug_info is not None:
                    debug_info.update(
                        {"source": "generated", "assembly": False, "cacheHit": True, "tookMs": 1.5}
                    )
                return None

            original_ensure = snapshot_main.ensure_step_topology_artifact
            try:
                snapshot_main.ensure_step_topology_artifact = fake_ensure
                packet = resolve_render_job_packet(
                    {
                        "input": "models/part.step",
                        "debug": True,
                        "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                    },
                    cwd=root,
                )
            finally:
                snapshot_main.ensure_step_topology_artifact = original_ensure

        job = packet["jobs"][0]
        self.assertEqual(
            job["resolved"]["debug"],
            {"stepArtifact": {"source": "generated", "assembly": False, "cacheHit": True, "tookMs": 1.5}},
        )

    def test_debug_reaches_rendered_json_output(self) -> None:
        """--debug diagnostics are attached at resolve time, but the printed result is the
        browser's return value — the render stage must merge them in or the help text's
        promised "debug" section never appears in --json output."""

        class StubRenderer:
            async def render(self, job):
                return {"ok": True, "mode": "view", "outputs": []}

            async def close(self):
                return None

        debug_payload = {"stepArtifact": {"source": "generated", "cacheHit": True, "tookMs": 1.5}}
        job = {
            "input": "models/part.step",
            "resolved": {"debug": debug_payload},
        }

        result = asyncio.run(
            snapshot_main.render_resolved_job_packet(
                {"single": True, "jobs": [job]}, renderer=StubRenderer()
            )
        )
        self.assertEqual(result["debug"], debug_payload)
        stream = io.StringIO()
        snapshot_main.print_render_result(result, json_output=True, stdout=stream)
        self.assertEqual(json.loads(stream.getvalue())["debug"], debug_payload)

        multi = asyncio.run(
            snapshot_main.render_resolved_job_packet(
                {"single": False, "jobs": [job]}, renderer=StubRenderer()
            )
        )
        self.assertEqual(multi["jobs"][0]["debug"], debug_payload)

    def test_render_job_omits_debug_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            models = root / "models"
            models.mkdir()
            step_path = models / "part.step"
            step_path.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
            write_package(step_path)
            calls = []

            def fake_ensure(target, **kwargs):
                calls.append(kwargs)
                return None

            original_ensure = snapshot_main.ensure_step_topology_artifact
            try:
                snapshot_main.ensure_step_topology_artifact = fake_ensure
                packet = resolve_render_job_packet(
                    {
                        "input": "models/part.step",
                        "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                    },
                    cwd=root,
                )
            finally:
                snapshot_main.ensure_step_topology_artifact = original_ensure

        self.assertIsNone(calls[0]["debug"])
        job = packet["jobs"][0]
        self.assertNotIn("debug", job["resolved"])

    def test_render_job_rejects_non_step_input_without_artifact_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            models = root / "models"
            models.mkdir()
            (models / "robot.urdf").write_text("<robot name=\"r\" />\n", encoding="utf-8")
            calls = []

            def fake_ensure(target, **kwargs):
                calls.append((target, kwargs))
                return None

            original_ensure = snapshot_main.ensure_step_topology_artifact
            try:
                snapshot_main.ensure_step_topology_artifact = fake_ensure
                with self.assertRaisesRegex(
                    SnapshotError,
                    "Snapshot supports STEP/STP inputs, same-stem Python generators, or direct GLB/STL/3MF meshes",
                ):
                    resolve_render_job_packet(
                        {
                            "input": "models/robot.urdf",
                            "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                        },
                        cwd=root,
                    )
            finally:
                snapshot_main.ensure_step_topology_artifact = original_ensure

        self.assertEqual(calls, [])

    def _mesh_job_env(self, temporary_directory, filename, content=b"mesh-bytes"):
        root = Path(temporary_directory).resolve()
        models = root / "models"
        models.mkdir()
        (models / filename).write_bytes(content)
        return root

    def test_render_job_resolves_direct_glb_without_step_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._mesh_job_env(temporary_directory, "widget.glb", b"glTF-binary-bytes")
            calls = []

            def fake_ensure(target, **kwargs):
                calls.append((target, kwargs))
                return None

            original_ensure = snapshot_main.ensure_step_topology_artifact
            try:
                snapshot_main.ensure_step_topology_artifact = fake_ensure
                packet = resolve_render_job_packet(
                    {
                        "input": "models/widget.glb",
                        "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                    },
                    cwd=root,
                )
            finally:
                snapshot_main.ensure_step_topology_artifact = original_ensure

        # The STEP artifact pipeline must never be entered for a direct mesh.
        self.assertEqual(calls, [])
        resolved = packet["jobs"][0]["resolved"]
        self.assertEqual(resolved["kind"], "glb")
        self.assertEqual(resolved["inputUrl"], "/__render_asset/widget.glb")
        self.assertEqual(resolved["url"], "/__render_asset/widget.glb")
        self.assertEqual(resolved["rootPath"], str(root / "models"))
        self.assertNotIn("package", resolved)
        self.assertNotIn("stepParameterUrl", resolved)

    def test_render_job_resolves_direct_stl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._mesh_job_env(temporary_directory, "part.stl", b"solid test\nendsolid test\n")
            packet = resolve_render_job_packet(
                {
                    "input": "models/part.stl",
                    "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                },
                cwd=root,
            )
        resolved = packet["jobs"][0]["resolved"]
        self.assertEqual(resolved["kind"], "stl")
        self.assertEqual(resolved["url"], "/__render_asset/part.stl")

    def test_render_job_resolves_direct_3mf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._mesh_job_env(temporary_directory, "part.3mf")
            packet = resolve_render_job_packet(
                {
                    "input": "models/part.3mf",
                    "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                },
                cwd=root,
            )
        self.assertEqual(packet["jobs"][0]["resolved"]["kind"], "3mf")

    def test_render_job_rejects_selection_for_mesh_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._mesh_job_env(temporary_directory, "widget.glb", b"glTF")
            with self.assertRaisesRegex(SnapshotError, "selection focus/hide/refs require STEP topology"):
                resolve_render_job_packet(
                    {
                        "input": "models/widget.glb",
                        "selection": {"focus": ["#o1.2"]},
                        "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                    },
                    cwd=root,
                )

    def test_render_job_rejects_step_parameters_for_mesh_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._mesh_job_env(temporary_directory, "widget.glb", b"glTF")
            with self.assertRaisesRegex(SnapshotError, "stepParameters require a STEP model"):
                resolve_render_job_packet(
                    {
                        "input": "models/widget.glb",
                        "stepParameters": {"width": 5},
                        "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                    },
                    cwd=root,
                )

    def test_render_job_rejects_section_mode_for_mesh_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._mesh_job_env(temporary_directory, "widget.glb", b"glTF")
            with self.assertRaisesRegex(SnapshotError, "section mode requires STEP topology"):
                resolve_render_job_packet(
                    {
                        "input": "models/widget.glb",
                        "mode": "section",
                        "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                    },
                    cwd=root,
                )

    def test_render_job_rejects_hidden_edges_display_for_mesh_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._mesh_job_env(temporary_directory, "widget.glb", b"glTF")
            for hidden_mode in ("hidden_edges", "hidden_lines_removed"):
                with self.assertRaisesRegex(SnapshotError, "requires STEP CAD edges"):
                    resolve_render_job_packet(
                        {
                            "input": "models/widget.glb",
                            "display": {"mode": hidden_mode},
                            "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                        },
                        cwd=root,
                    )

    def test_render_job_rejects_exploded_display_for_mesh_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._mesh_job_env(temporary_directory, "widget.glb", b"glTF")
            with self.assertRaisesRegex(SnapshotError, "exploded view requires STEP assembly"):
                resolve_render_job_packet(
                    {
                        "input": "models/widget.glb",
                        "display": {"exploded": {"enabled": True, "axis": "z"}},
                        "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                    },
                    cwd=root,
                )

    def test_render_job_allows_list_mode_for_mesh_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._mesh_job_env(temporary_directory, "widget.glb", b"glTF")
            packet = resolve_render_job_packet(
                {"input": "models/widget.glb", "mode": "list"},
                cwd=root,
            )
        job = packet["jobs"][0]
        self.assertEqual(job["mode"], "list")
        self.assertEqual(job["resolved"]["kind"], "glb")

    def test_render_job_rejects_non_solid_display_mode_for_mesh_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._mesh_job_env(temporary_directory, "widget.glb", b"glTF")
            for non_solid in ("wireframe", "transparent", "unshaded"):
                with self.assertRaisesRegex(SnapshotError, "display mode is not supported"):
                    resolve_render_job_packet(
                        {
                            "input": "models/widget.glb",
                            "display": {"mode": non_solid},
                            "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                        },
                        cwd=root,
                    )

    def test_render_job_allows_solid_and_projection_for_mesh_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = self._mesh_job_env(temporary_directory, "widget.glb", b"glTF")
            # Solid mode + orthographic projection both pass (projection is honored by the
            # renderer; it is camera-only, not topology-dependent).
            packet = resolve_render_job_packet(
                {
                    "input": "models/widget.glb",
                    "display": {"mode": "solid", "projection": "orthographic"},
                    "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                },
                cwd=root,
            )
        self.assertEqual(packet["jobs"][0]["display"], {"mode": "solid", "projection": "orthographic"})

    def test_render_job_missing_mesh_file_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            (root / "models").mkdir()
            with self.assertRaisesRegex(SnapshotError, "Render input does not exist"):
                resolve_render_job_packet(
                    {"input": "models/absent.stl", "outputs": [{"path": "tmp/iso.png"}]},
                    cwd=root,
                )

    def test_content_type_for_mesh_suffixes(self) -> None:
        self.assertEqual(snapshot_main.content_type_for_path(Path("x.stl")), "model/stl")
        self.assertEqual(snapshot_main.content_type_for_path(Path("x.3mf")), "model/3mf")
        self.assertEqual(snapshot_main.content_type_for_path(Path("x.glb")), "model/gltf-binary")

    def test_render_job_requires_selector_topology_for_cad_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            models = root / "models"
            models.mkdir()
            step_path = models / "assembly.step"
            step_path.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
            write_package(step_path, entry_kind="assembly")
            calls = []

            def fake_ensure(target, **kwargs):
                calls.append((target, kwargs))
                return _selector_artifact("o1", "o1.2")

            original_ensure = snapshot_main.ensure_step_topology_artifact
            try:
                snapshot_main.ensure_step_topology_artifact = fake_ensure
                resolve_render_job_packet(
                    {
                        "input": "models/assembly.step",
                        "selection": {"focus": ["#o1.2"]},
                        "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                    },
                    cwd=root,
                )
            finally:
                snapshot_main.ensure_step_topology_artifact = original_ensure

        self.assertEqual(len(calls), 1)
        target, kwargs = calls[0]
        self.assertEqual(target.step_path, step_path)
        self.assertTrue(kwargs["require_selector"])

    def test_render_job_normalizes_focus_selector_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            models = root / "models"
            models.mkdir()
            (models / "assembly.step").write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
            write_package(models / "assembly.step", entry_kind="assembly")

            original_ensure = snapshot_main.ensure_step_topology_artifact
            try:
                snapshot_main.ensure_step_topology_artifact = lambda *args, **kwargs: _selector_artifact(
                    "o1",
                    "o1.2",
                    "o1.2.1",
                    "o1.3",
                )
                packet = resolve_render_job_packet(
                    {
                        "input": "models/assembly.step",
                        "selection": {
                            "focus": ["#o1.2", "#o1.3"],
                        },
                        "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                    },
                    cwd=root,
                )
            finally:
                snapshot_main.ensure_step_topology_artifact = original_ensure

        selection = packet["jobs"][0]["selection"]
        self.assertEqual(selection["focus"], ["o1.2", "o1.3"])

    def test_render_job_normalizes_hide_selector_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            models = root / "models"
            models.mkdir()
            (models / "assembly.step").write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
            write_package(models / "assembly.step", entry_kind="assembly")

            original_ensure = snapshot_main.ensure_step_topology_artifact
            try:
                snapshot_main.ensure_step_topology_artifact = lambda *args, **kwargs: _selector_artifact(
                    "o1",
                    "o1.2",
                    "o1.2.1",
                    "o1.3",
                )
                packet = resolve_render_job_packet(
                    {
                        "input": "models/assembly.step",
                        "selection": {"hide": ["#o1.2.1"]},
                        "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                    },
                    cwd=root,
                )
            finally:
                snapshot_main.ensure_step_topology_artifact = original_ensure

        selection = packet["jobs"][0]["selection"]
        self.assertEqual(selection["hide"], ["o1.2.1"])

    def test_render_job_rejects_face_focus_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            models = root / "models"
            models.mkdir()
            (models / "assembly.step").write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
            (models / ".assembly.step.glb").write_bytes(b"glb")

            original_ensure = snapshot_main.ensure_step_topology_artifact
            try:
                snapshot_main.ensure_step_topology_artifact = lambda *args, **kwargs: _selector_artifact("o1", "o1.2")
                with self.assertRaisesRegex(SnapshotError, "part/subassembly occurrence refs"):
                    resolve_render_job_packet(
                        {
                            "input": "models/assembly.step",
                            "selection": {"focus": ["#o1.2.f1"]},
                            "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                        },
                        cwd=root,
                    )
            finally:
                snapshot_main.ensure_step_topology_artifact = original_ensure

    def test_render_job_rejects_mixed_focus_and_hide_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            models = root / "models"
            models.mkdir()
            (models / "assembly.step").write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
            (models / ".assembly.step.glb").write_bytes(b"glb")

            original_ensure = snapshot_main.ensure_step_topology_artifact
            try:
                snapshot_main.ensure_step_topology_artifact = lambda *args, **kwargs: _selector_artifact(
                    "o1",
                    "o1.2",
                    "o1.3",
                )
                with self.assertRaisesRegex(SnapshotError, "selection.focus/refs and selection.hide cannot be used"):
                    resolve_render_job_packet(
                        {
                            "input": "models/assembly.step",
                            "selection": {
                                "focus": ["#o1.2"],
                                "hide": ["#o1.3"],
                            },
                            "outputs": [{"path": "tmp/iso.png", "camera": "iso"}],
                        },
                        cwd=root,
                    )
            finally:
                snapshot_main.ensure_step_topology_artifact = original_ensure

    def test_snapshot_root_flags_and_job_fields_are_removed(self) -> None:
        with self.assertRaisesRegex(SnapshotError, "Unknown argument: --workspace-root"):
            parse_snapshot_args(["--workspace-root", "/tmp"])
        with self.assertRaisesRegex(SnapshotError, "Unknown argument: --root-dir"):
            parse_snapshot_args(["--root-dir", "models"])
        with self.assertRaisesRegex(SnapshotError, "no longer accept workspaceRoot or rootDir"):
            resolve_render_job_packet(
                {
                    "input": "part.step",
                    "workspaceRoot": "/tmp",
                    "outputs": [{"path": "tmp/iso.png"}],
                },
                cwd=Path.cwd(),
            )

    def test_timestamp_output_path_preserves_extension(self) -> None:
        self.assertEqual(
            timestamp_output_path("snapshots/review.png", "20260527T163012Z"),
            "snapshots/review_20260527T163012Z.png",
        )

    def test_removed_daemon_flags_stay_removed(self) -> None:
        with self.assertRaisesRegex(SnapshotError, "daemon commands have been removed"):
            parse_snapshot_args(["daemon"])
        with self.assertRaisesRegex(SnapshotError, "--socket has been removed"):
            parse_snapshot_args(["--socket", "snapshot.sock"])

    def test_runtime_routes_are_self_contained(self) -> None:
        self.assertEqual(
            resolve_snapshot_route_file("http://snapshot.local/render.html"),
            RENDER_HTML_PATH,
        )
        self.assertEqual(
            resolve_snapshot_route_file("http://snapshot.local/snapshot-render.js"),
            RUNTIME_DIR / "snapshot-render.js",
        )

    def test_snapshot_renderer_does_not_force_chromium_single_process(self) -> None:
        captured_launch_options = {}

        class FakePage:
            async def route(self, *args, **kwargs):
                pass

            async def goto(self, *args, **kwargs):
                pass

            async def wait_for_function(self, *args, **kwargs):
                pass

        class FakeContext:
            async def new_page(self):
                return FakePage()

            async def close(self):
                pass

        class FakeBrowser:
            async def new_context(self, *args, **kwargs):
                return FakeContext()

            async def close(self):
                pass

        class FakeChromium:
            async def launch(self, **kwargs):
                captured_launch_options.update(kwargs)
                return FakeBrowser()

        class FakePlaywright:
            def __init__(self) -> None:
                self.chromium = FakeChromium()

            async def stop(self):
                pass

        fake_playwright = FakePlaywright()

        class FakeAsyncPlaywright:
            async def start(self):
                return fake_playwright

        async_api_module = ModuleType("playwright.async_api")
        async_api_module.async_playwright = FakeAsyncPlaywright
        playwright_module = ModuleType("playwright")
        playwright_module.__path__ = []

        original_playwright = sys.modules.get("playwright")
        original_async_api = sys.modules.get("playwright.async_api")
        try:
            sys.modules["playwright"] = playwright_module
            sys.modules["playwright.async_api"] = async_api_module

            async def start_renderer() -> None:
                renderer = snapshot_main.BatchSnapshotRenderer()
                try:
                    await renderer.start()
                finally:
                    await renderer.close()

            asyncio.run(start_renderer())
        finally:
            if original_playwright is None:
                sys.modules.pop("playwright", None)
            else:
                sys.modules["playwright"] = original_playwright
            if original_async_api is None:
                sys.modules.pop("playwright.async_api", None)
            else:
                sys.modules["playwright.async_api"] = original_async_api

        self.assertNotIn("--single-process", captured_launch_options.get("args") or [])

    def test_snapshot_tool_has_no_sideways_runtime_dependencies(self) -> None:
        snapshot_root = repo_path("skills/cad/scripts/snapshot")
        checked_files = [
            snapshot_root / "__main__.py",
            snapshot_root / "runtime" / "render.html",
            snapshot_root / "runtime" / "snapshot-render.js",
        ]
        forbidden = (
            "packages/cadjs",
            "skills/cad-viewer",
            "/node_modules/",
            "\\node_modules\\",
            "CADJS_NODE_MODULES_ROOT",
        )
        for checked_file in checked_files:
            text = checked_file.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{checked_file} should not reference {token}")


if __name__ == "__main__":
    unittest.main()
