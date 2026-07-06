# SpaceX Models

> **Educational, non-functional public-source reconstructions. Not suitable
> for manufacture, propulsion, testing, or operational engineering.**

Museum/documentary-style CAD packages of SpaceX hardware, reconstructed
exclusively from public sources (official SpaceX statements and imagery,
FAA/NASA regulatory material, reputable explainers; Wikipedia as index only).
Proprietary internals are deliberately excluded; hidden internals appear only
as simplified translucent placeholder volumes labeled
`schematic` / `inferred` / `nonfunctional`.

Only `.step.py` generator sources (plus docs and renders) are kept in this
tree; viewer render packages are generated on demand into per-package
`__cadcache__/` directories, and STEP/IGES/STL/OBJ/GLB exports are produced
on demand (`scripts/export ... --step/--stl/--glb`, `export_extras.py`).

## Packages

- [raptor2/](raptor2/README.md): SpaceX Raptor 2 educational public-source
  reconstruction — exterior, schematic cutaway, exploded-view, and derived
  Raptor Vacuum generators plus research/provenance documentation.
- [starship/](starship/README.md): SpaceX Starship / Super Heavy full-stack
  educational public-source reconstruction (pinned V2/Block 2) — booster,
  ship, stack, schematic cutaway, and exploded-view generators reusing the
  raptor2 engines as linked instanced subassemblies, plus research/provenance
  documentation.
- [merlin1d/](merlin1d/README.md): SpaceX Merlin 1D educational public-source
  reconstruction — exterior, schematic cutaway, and exploded-view generators
  (~260–275 named parts each) plus research/provenance documentation and
  renders.
- [falcon_heavy/](falcon_heavy/README.md): SpaceX Falcon Heavy full-vehicle
  educational public-source reconstruction — three cores with 27 linked
  Merlin 1D instances (reusing merlin1d/), MVac-derivative second stage,
  cutaway and exploded views (~2,150 named parts each).

Each package's `PROVENANCE.md` carries the per-component source, confidence,
and geometry-status table; `DIMENSIONS.md` separates published dimensions
(scale anchors) from photogrammetric estimates; `RESEARCH.md` holds the
cited public-source dossier.
