from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from cadgen.catalog import StepImportOptions
from cadgen.metadata import normalize_mesh_numeric


def generate_step_targets(*args, **kwargs):
    from cadgen.generation import generate_step_targets as generate

    return generate(*args, **kwargs)


def _normalize_cli_numeric(value: object, *, field_name: str, parser: argparse.ArgumentParser) -> float | None:
    try:
        return normalize_mesh_numeric(value, field_name=field_name)
    except ValueError as exc:
        parser.error(str(exc))
    return None


def _add_gen_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "targets",
        nargs="+",
        help="Explicit gen_step() Python generator source(s) to build.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild render packages even when current artifacts match the source closure.",
    )
    parser.add_argument(
        "--mesh-tolerance",
        type=float,
        help="Positive mesh linear deflection for the render GLB/topology artifacts.",
    )
    parser.add_argument(
        "--mesh-angular-tolerance",
        type=float,
        help="Positive mesh angular deflection for the render GLB/topology artifacts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress and timing information.",
    )


def _validate_python_targets(targets: Sequence[str], *, parser: argparse.ArgumentParser) -> None:
    for target in targets:
        target_text = str(target)
        if "=" in target_text:
            parser.error(
                "SOURCE=OUTPUT pairs are no longer supported; scripts/gen builds render "
                "packages only. Use scripts/export for STEP/STL/3MF/GLB files."
            )
        suffix = Path(target_text).suffix.lower()
        if suffix in {".step", ".stp"}:
            parser.error(
                f"scripts/gen builds gen_step() Python sources only: {target_text}. "
                "Imported STEP/STP files get render artifacts on demand (inspect, snapshot, "
                "CAD Viewer); use scripts/export for STL/3MF/GLB files from an imported STEP."
            )
        if suffix != ".py":
            parser.error(f"scripts/gen target must be a gen_step() Python source: {target_text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/gen",
        description=(
            "Build CAD Viewer render packages (GLB/topology) for explicit gen_step() "
            "Python targets. Writes no STEP/STL/3MF/GLB files; use scripts/export for those."
        ),
    )
    _add_gen_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _validate_python_targets(args.targets, parser=parser)
    step_options = StepImportOptions(
        mesh_tolerance=_normalize_cli_numeric(
            args.mesh_tolerance,
            field_name="mesh_tolerance",
            parser=parser,
        ),
        mesh_angular_tolerance=_normalize_cli_numeric(
            args.mesh_angular_tolerance,
            field_name="mesh_angular_tolerance",
            parser=parser,
        ),
    )
    return generate_step_targets(
        args.targets,
        step_options=step_options,
        force=bool(args.force),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    raise SystemExit(main())
