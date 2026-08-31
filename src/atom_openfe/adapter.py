from __future__ import annotations

import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any


def derive_mapping_alignment(mapping) -> dict[str, list[int] | int]:
    """Derive AToM alignment atom options from a LigandAtomMapping.

    Returned indices are component-local, 0-based ligand atom indices. AToM's
    setup places ligand atoms contiguously and preserves SDF order, so these
    become the reference atom indices expected by ``rbfe_prepare_args``.
    """

    mol_a = mapping.componentA.to_rdkit()
    mol_b = mapping.componentB.to_rdkit()
    candidates = [
        (int(i), int(j))
        for i, j in mapping.componentA_to_componentB.items()
        if mol_a.GetAtomWithIdx(int(i)).GetAtomicNum() > 1
        and mol_b.GetAtomWithIdx(int(j)).GetAtomicNum() > 1
    ]

    if len(candidates) < 3:
        raise ValueError(
            "AToM RBFE alignment requires at least three mapped heavy-atom pairs"
        )

    try:
        conf_a = mol_a.GetConformer()
    except ValueError as exc:
        raise ValueError(
            "AToM RBFE alignment derivation requires 3D coordinates on ligand A"
        ) from exc
    try:
        conf_b = mol_b.GetConformer()
    except ValueError as exc:
        raise ValueError(
            "AToM RBFE alignment derivation requires 3D coordinates on ligand B"
        ) from exc

    points_a = {
        i: (
            float(conf_a.GetAtomPosition(i).x),
            float(conf_a.GetAtomPosition(i).y),
            float(conf_a.GetAtomPosition(i).z),
        )
        for i, _ in candidates
    }
    points_b = {
        j: (
            float(conf_b.GetAtomPosition(j).x),
            float(conf_b.GetAtomPosition(j).y),
            float(conf_b.GetAtomPosition(j).z),
        )
        for _, j in candidates
    }

    centroid = tuple(
        sum(point[axis] for point in points_a.values()) / len(points_a)
        for axis in range(3)
    )
    triple = max(
        combinations(candidates, 3),
        key=lambda selected: min(
            _triangle_area(*(points_a[pair[0]] for pair in selected)),
            _triangle_area(*(points_b[pair[1]] for pair in selected)),
        ),
    )
    if min(
        _triangle_area(*(points_a[pair[0]] for pair in triple)),
        _triangle_area(*(points_b[pair[1]] for pair in triple)),
    ) < 1.0e-3:
        raise ValueError(
            "AToM RBFE alignment requires three mapped heavy-atom pairs that "
            "are non-collinear in both ligands"
        )

    first = min(triple, key=lambda pair: _distance(points_a[pair[0]], centroid))
    remaining = [pair for pair in triple if pair != first]
    second = max(
        remaining,
        key=lambda pair: _distance(points_a[pair[0]], points_a[first[0]]),
    )
    third = next(pair for pair in remaining if pair != second)

    ligand1_ref_atoms = [first[0], second[0], third[0]]
    ligand2_ref_atoms = [first[1], second[1], third[1]]
    return {
        "ALIGN_LIGAND1_REF_ATOMS": ligand1_ref_atoms,
        "ALIGN_LIGAND2_REF_ATOMS": ligand2_ref_atoms,
        "LIGAND1_ATTACH_INDEX": ligand1_ref_atoms[0],
        "LIGAND2_ATTACH_INDEX": ligand2_ref_atoms[0],
    }


def prepare_atm_transfer_system(
    *,
    mode: str,
    options: dict[str, Any],
    workdir: str | Path,
    receptor,
    ligand1,
    solvent,
    forcefield_settings,
    thermo_settings,
    solvation_settings,
    partial_charge_settings,
    ghost_mass: float | None,
    forcefield_cache: str | Path | None,
    ligand2=None,
) -> dict[str, Any]:
    """Build, serialize, and round-trip validate a native AToM system."""

    from .system_builder import build_atm_system

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    atom_options = dict(options)
    basename = str(atom_options["BASENAME"])
    pdb_path = workdir / f"{basename}.pdb"
    xml_path = workdir / f"{basename}_sys.xml"
    cache_path = None
    if forcefield_cache is not None:
        cache_path = _resolve_workdir_path(workdir, forcefield_cache)

    alignment_options = {
        key: value
        for key, value in atom_options.items()
        if key
        in {
            "ALIGN_LIGAND1_REF_ATOMS",
            "ALIGN_LIGAND2_REF_ATOMS",
            "LIGAND1_ATTACH_INDEX",
            "LIGAND2_ATTACH_INDEX",
        }
    }
    displacement = atom_options.get("DISPLACEMENT")
    prepared = build_atm_system(
        mode=mode,
        receptor=receptor,
        ligand1=ligand1,
        ligand2=ligand2,
        solvent=solvent,
        forcefield_settings=forcefield_settings,
        thermo_settings=thermo_settings,
        solvation_settings=solvation_settings,
        partial_charge_settings=partial_charge_settings,
        displacement=(tuple(displacement) if displacement is not None else None),
        ghost_mass=ghost_mass,
        cache=cache_path,
        alignment_options=alignment_options,
    )
    atom_options.update(prepared.atom_options)
    atom_options["WORKDIR"] = str(workdir)

    roundtrip = _serialize_prepared_system(
        prepared=prepared,
        mode=mode,
        pdb_path=pdb_path,
        xml_path=xml_path,
        forcefield_settings=forcefield_settings,
        ghost_mass=ghost_mass,
    )
    atom_options.update(prepared.atom_options)

    options_path = workdir / f"{basename}.yaml"
    _write_yaml(options_path, atom_options)

    return {
        "atom_options": atom_options,
        "atom_options_path": str(options_path),
        "prepared_system_pdb": str(pdb_path),
        "prepared_system_xml": str(xml_path),
        "diagnostics": {**prepared.diagnostics, "roundtrip": roundtrip},
    }


def _serialize_prepared_system(
    *,
    prepared,
    mode: str,
    pdb_path: Path,
    xml_path: Path,
    forcefield_settings,
    ghost_mass: float | None,
) -> dict[str, Any]:
    """Serialize the contract files and validate the objects AToM reloads."""

    from openmm import XmlSerializer
    from openmm.app import PDBFile

    from .system_builder import PreparedATMSystem, derive_atm_options
    from .system_validation import validate_prepared_system

    with pdb_path.open("w") as stream:
        PDBFile.writeFile(
            prepared.topology, prepared.positions, stream, keepIds=True
        )
    pdb = PDBFile(str(pdb_path))
    if _topology_signature(pdb.topology) != _topology_signature(prepared.topology):
        raise ValueError("PDB round-trip changed AToM atom order or topology metadata")

    pdb_box = pdb.topology.getPeriodicBoxVectors()
    if pdb_box is None:
        raise ValueError("Serialized AToM PDB lost its periodic box vectors")
    prepared.topology.setPeriodicBoxVectors(pdb_box)
    prepared.system.setDefaultPeriodicBoxVectors(*pdb_box)
    prepared.positions = pdb.positions

    alignment_options: dict[str, Any] = {}
    for number, role in (("1", "L1"), ("2", "L2")):
        refs = prepared.atom_options.get(f"ALIGN_LIGAND{number}_REF_ATOMS")
        if refs is not None:
            alignment_options[f"ALIGN_LIGAND{number}_REF_ATOMS"] = list(refs)
        alignment_options[f"LIGAND{number}_ATTACH_INDEX"] = (
            prepared.atom_options[f"LIGAND{number}_ATTACH_ATOM"]
            - prepared.role_atom_indices[role][0]
        )
    prepared.atom_options = derive_atm_options(
        mode=mode,
        topology=pdb.topology,
        positions=pdb.positions,
        role_atom_indices=prepared.role_atom_indices,
        alignment_options=alignment_options,
    )

    with xml_path.open("w") as stream:
        stream.write(XmlSerializer.serialize(prepared.system))
    with xml_path.open() as stream:
        roundtrip_system = XmlSerializer.deserialize(stream.read())

    roundtrip_prepared = PreparedATMSystem(
        topology=pdb.topology,
        positions=pdb.positions,
        system=roundtrip_system,
        component_atom_indices=prepared.component_atom_indices,
        component_residue_indices=prepared.component_residue_indices,
        role_atom_indices=prepared.role_atom_indices,
        role_residue_indices=prepared.role_residue_indices,
        atom_options=prepared.atom_options,
        diagnostics=dict(prepared.diagnostics),
        ligand_molecules=prepared.ligand_molecules,
        box_sizing_positions_nm=prepared.box_sizing_positions_nm,
    )
    validation = validate_prepared_system(
        roundtrip_prepared,
        mode=mode,
        forcefield_settings=forcefield_settings,
        ghost_mass=ghost_mass,
        check_energy=True,
    )
    return {
        "pdb_xml_validated": True,
        "atom_signature_preserved": True,
        "validation": validation,
    }


def _topology_signature(
    topology,
) -> list[tuple[int, int, str, str, str, str | None]]:
    return [
        (
            atom.residue.chain.index,
            atom.residue.index,
            atom.name,
            atom.residue.name,
            atom.residue.chain.id,
            atom.element.symbol if atom.element is not None else None,
        )
        for atom in topology.atoms()
    ]


def run_atm_transfer(
    *,
    options: dict[str, Any],
    workdir: str | Path,
    config_file: str | Path | None = None,
) -> dict[str, Any]:
    """Run AToM structure preparation and RBFE-style production."""

    try:
        from atom_openmm.rbfe_production import rbfe_production
        from atom_openmm.rbfe_structprep import rbfe_structprep
    except ImportError as exc:
        raise RuntimeError(
            "atom_openmm is not importable. Install AToM-OpenMM or add its "
            "checkout to PYTHONPATH before executing real AToM runs."
        ) from exc

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    atom_options = dict(options)
    atom_options["WORKDIR"] = str(workdir)
    basename = str(atom_options["BASENAME"])
    nstates = len(atom_options["LAMBDAS"])

    samples_before = count_replica_samples(workdir, basename, nstates)
    structprep_ran = False
    production_ran = False

    if not (workdir / f"{basename}_0.xml").exists():
        _call_in_workdir(
            workdir,
            rbfe_structprep,
            config_file=None,
            # rbfe_structprep massages TIME_STEP and other keys in place.
            # Keep the authoritative runtime options intact for production.
            options=dict(atom_options),
        )
        structprep_ran = True

    max_samples = int(atom_options["MAX_SAMPLES"])
    if samples_before is None or samples_before < max_samples:
        _call_in_workdir(
            workdir,
            rbfe_production,
            config_file=None,
            options=dict(atom_options),
        )
        production_ran = True

    samples_after = count_replica_samples(workdir, basename, nstates)

    return {
        "status": "completed",
        "workdir": str(workdir),
        "basename": basename,
        "config_file": str(config_file) if config_file is not None else None,
        "structprep_ran": structprep_ran,
        "production_ran": production_ran,
        "samples_before": samples_before,
        "samples_after": samples_after,
        "artifacts": transfer_artifacts(workdir, basename),
    }


def run_atm_abfe(
    *,
    options: dict[str, Any],
    workdir: str | Path,
    config_file: str | Path | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for the old ABFE adapter entry point."""

    return run_atm_transfer(options=options, workdir=workdir, config_file=config_file)


def analyze_atm_uwham(
    *,
    workdir: str | Path,
    basename: str,
    mintimeid: int | None = None,
    maxtimeid: int | None = None,
    discard_fraction: float = 1.0 / 3.0,
) -> dict[str, Any]:
    """Run AToM's Python UWHAM analysis for an existing AToM run directory."""

    try:
        from atom_openmm.uwham import calculate_uwham_from_rundir
    except ImportError as exc:
        raise RuntimeError(
            "atom_openmm.uwham is not importable. Install AToM-OpenMM or add "
            "its checkout to PYTHONPATH before executing UWHAM analysis."
        ) from exc

    workdir = Path(workdir)
    if mintimeid is None:
        nstates = _infer_replica_count(workdir)
        samples = count_replica_samples(workdir, basename, nstates) if nstates else None
        if samples is not None:
            mintimeid = int(samples * discard_fraction)

    estimate, uncertainty, uwham_data = calculate_uwham_from_rundir(
        str(workdir),
        basename,
        mintimeid=mintimeid,
        maxtimeid=maxtimeid,
    )

    leg_diagnostics = {
        "dg_leg1": _maybe_float(uwham_data.get("dg_leg1")),
        "dg_stderr_leg1": _maybe_float(uwham_data.get("dg_stderr_leg1")),
        "dg_leg2": _maybe_float(uwham_data.get("dg_leg2")),
        "dg_stderr_leg2": _maybe_float(uwham_data.get("dg_stderr_leg2")),
        "n_samples": uwham_data.get("nsamples"),
        "mintimeid": mintimeid,
        "maxtimeid": maxtimeid,
    }

    return {
        "unit_estimate": float(estimate),
        "unit_estimate_error": float(uncertainty),
        "dg_leg1": leg_diagnostics["dg_leg1"],
        "dg_stderr_leg1": leg_diagnostics["dg_stderr_leg1"],
        "dg_leg2": leg_diagnostics["dg_leg2"],
        "dg_stderr_leg2": leg_diagnostics["dg_stderr_leg2"],
        "n_samples": leg_diagnostics["n_samples"],
        "diagnostics": leg_diagnostics,
    }


def count_replica_samples(
    workdir: str | Path,
    basename: str,
    nstates: int,
) -> int | None:
    counts: list[int] = []
    workdir = Path(workdir)
    for state_idx in range(nstates):
        output_file = workdir / f"r{state_idx}" / f"{basename}.out"
        if not output_file.exists():
            return None
        with output_file.open() as stream:
            counts.append(sum(1 for line in stream if line.strip()))
    if not counts:
        return None
    return min(counts)


def transfer_artifacts(workdir: str | Path, basename: str) -> dict[str, str]:
    workdir = Path(workdir)
    candidates = {
        "options": workdir / f"{basename}.yaml",
        "system_pdb": workdir / f"{basename}.pdb",
        "system_xml": workdir / f"{basename}_sys.xml",
        "prepared_state": workdir / f"{basename}_0.xml",
        "minimized_pdb": workdir / f"{basename}_min.pdb",
        "equilibrated_pdb": workdir / f"{basename}_equil.pdb",
    }
    return {name: str(path) for name, path in candidates.items() if path.exists()}


def _resolve_workdir_path(workdir: Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return workdir / path


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _call_in_workdir(workdir: Path, func, **kwargs):
    origin = Path.cwd()
    try:
        os.chdir(workdir)
        return func(**kwargs)
    finally:
        os.chdir(origin)


def _infer_replica_count(workdir: Path) -> int:
    return len([path for path in workdir.glob("r[0-9]*") if path.is_dir()])


def _maybe_float(value):
    if value is None:
        return None
    return float(value)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _triangle_area(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    ab = tuple(b[i] - a[i] for i in range(3))
    ac = tuple(c[i] - a[i] for i in range(3))
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * math.sqrt(sum(v * v for v in cross))
