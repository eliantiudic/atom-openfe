from __future__ import annotations

import math
import os
import shutil
from collections.abc import Mapping
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

    points = {
        i: (
            float(conf_a.GetAtomPosition(i).x),
            float(conf_a.GetAtomPosition(i).y),
            float(conf_a.GetAtomPosition(i).z),
        )
        for i, _ in candidates
    }

    centroid = tuple(
        sum(point[axis] for point in points.values()) / len(points) for axis in range(3)
    )
    first = min(candidates, key=lambda pair: _distance(points[pair[0]], centroid))
    second = max(candidates, key=lambda pair: _distance(points[pair[0]], points[first[0]]))
    third = max(
        (pair for pair in candidates if pair not in {first, second}),
        key=lambda pair: _triangle_area(points[first[0]], points[second[0]], points[pair[0]]),
    )

    if _triangle_area(points[first[0]], points[second[0]], points[third[0]]) < 1.0e-3:
        raise ValueError(
            "AToM RBFE alignment requires three non-collinear mapped heavy-atom pairs"
        )

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
    receptor_file: str | Path,
    ligand1_file: str | Path,
    ligand2_file: str | Path | None = None,
    prepared_system_pdb: str | Path | None = None,
    prepared_system_xml: str | Path | None = None,
    protein_forcefields: list[str] | None = None,
    solvent_forcefields: list[str] | None = None,
    ligand_forcefield: str | None = None,
    ionic_strength: float = 0.15,
) -> dict[str, Any]:
    """Create or import an AToM transfer system and derive final options."""

    try:
        from atom_openmm.make_atm_system_from_rcpt_lig import make_system
        from atom_openmm.utils.AtomUtils import calc_displ_vec, patch_system_with_ghost
    except ImportError as exc:
        raise RuntimeError(
            "atom_openmm is not importable. Install AToM-OpenMM or add its "
            "checkout to PYTHONPATH before preparing real AToM systems."
        ) from exc

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    atom_options = dict(options)
    basename = str(atom_options["BASENAME"])
    pdb_path = workdir / f"{basename}.pdb"
    xml_path = workdir / f"{basename}_sys.xml"

    if "DISPLACEMENT" not in atom_options or not atom_options["DISPLACEMENT"]:
        displacement_ligand = ligand2_file if ligand2_file is not None else ligand1_file
        atom_options["DISPLACEMENT"] = list(
            calc_displ_vec(str(receptor_file), str(displacement_ligand))
        )

    if prepared_system_pdb is not None and prepared_system_xml is not None:
        shutil.copyfile(prepared_system_pdb, pdb_path)
        shutil.copyfile(prepared_system_xml, xml_path)
    else:
        system_kwargs: dict[str, Any] = {
            "receptorfile": str(receptor_file),
            "lig1file": str(ligand1_file),
            "displacement": atom_options["DISPLACEMENT"],
            "xmloutfile": str(xml_path),
            "pdboutfile": str(pdb_path),
            "hmass": atom_options.get("HMASS", 1.0),
            "ionicstrength": ionic_strength,
            "flagverbose": bool(atom_options.get("VERBOSE", False)),
        }
        if mode == "rbfe":
            if ligand2_file is None:
                raise ValueError("RBFE mode requires ligand2_file")
            system_kwargs["lig2file"] = str(ligand2_file)
        if protein_forcefields is not None:
            system_kwargs["proteinforcefield"] = protein_forcefields
        if solvent_forcefields is not None:
            system_kwargs["solventforcefield"] = solvent_forcefields
        if ligand_forcefield is not None:
            system_kwargs["ligandforcefield"] = ligand_forcefield
        if atom_options.get("FORCEFIELD_CACHE") is not None:
            system_kwargs["ffcachefile"] = str(
                _resolve_workdir_path(workdir, atom_options["FORCEFIELD_CACHE"])
            )

        make_system(**system_kwargs)

    if mode == "abfe" and not system_has_ghost_pair(pdb_path):
        patch_system_with_ghost(
            str(pdb_path),
            str(xml_path),
            atom_options["DISPLACEMENT"],
            atom_options.get("GHOST_MASS", 12.011),
            attach_index=atom_options.get("LIGAND_ATTACH_INDEX"),
        )

    prepared_options = derive_prepared_transfer_options(
        mode=mode,
        options=atom_options,
        pdb_file=pdb_path,
    )
    atom_options.update(prepared_options)
    atom_options["WORKDIR"] = str(workdir)

    options_path = workdir / f"{basename}.yaml"
    _write_yaml(options_path, atom_options)

    return {
        "atom_options": atom_options,
        "atom_options_path": str(options_path),
        "prepared_system_pdb": str(pdb_path),
        "prepared_system_xml": str(xml_path),
        "diagnostics": {
            "mode": mode,
            "ghost_patched": mode == "abfe",
            "prepared_from_files": prepared_system_pdb is not None,
        },
    }


def derive_prepared_transfer_options(
    *,
    mode: str,
    options: Mapping[str, Any],
    pdb_file: str | Path,
) -> dict[str, Any]:
    """Derive AToM RBFE-style options from a prepared L1/L2 PDB."""

    try:
        from atom_openmm.utils.AtomUtils import (
            get_attach_atom_from_residue,
            get_indexes_from_query,
            get_indexes_from_residue,
            get_residue_by_name,
            get_selected_principal_groups,
        )
        from openmm import Vec3
        from openmm.app import PDBFile
        from openmm.unit import angstrom, nanometer
    except ImportError as exc:
        raise RuntimeError(
            "atom_openmm/openmm is not importable. Install AToM-OpenMM and OpenMM "
            "before deriving prepared AToM options."
        ) from exc

    pdb = PDBFile(str(pdb_file))
    topology = pdb.topology
    positions = pdb.positions

    ligand1_residue = get_residue_by_name(topology, "L1")
    ligand2_residue = get_residue_by_name(topology, "L2")
    if ligand1_residue is None or ligand2_residue is None:
        raise ValueError("Prepared AToM transfer systems must contain L1 and L2 residues")

    ligand1_atoms = get_indexes_from_residue(ligand1_residue)
    ligand2_atoms = get_indexes_from_residue(ligand2_residue)

    ligand1_attach_atom = _select_attach_atom(
        ligand1_residue,
        options.get("LIGAND1_ATTACH_INDEX"),
        options.get("ALIGN_LIGAND1_REF_ATOMS"),
        get_attach_atom_from_residue,
        positions,
    )
    ligand2_attach_atom = _select_attach_atom(
        ligand2_residue,
        options.get("LIGAND2_ATTACH_INDEX"),
        options.get("ALIGN_LIGAND2_REF_ATOMS"),
        get_attach_atom_from_residue,
        positions,
    )

    lig1cm_pos = positions[ligand1_attach_atom.index]
    lig2cm_pos = positions[ligand2_attach_atom.index]
    displ = (lig2cm_pos - lig1cm_pos).value_in_unit(angstrom)

    rcpt_chain_names = list(options.get("RCPT_CHAIN_NAMES", ["A"]))
    rcpt_chain_query = f"atom.residue.chain.id in {rcpt_chain_names}"
    rcpt_frame_indexes = get_indexes_from_query(
        topology,
        rcpt_chain_query + ' and atom.name == "CA"',
    )
    if not rcpt_frame_indexes:
        raise ValueError(
            "Could not derive AToM receptor frame: no CA atoms matched "
            f"RCPT_CHAIN_NAMES={rcpt_chain_names}"
        )
    rcpt_frame = get_selected_principal_groups(topology, positions, rcpt_frame_indexes)

    rcpt_cm_pos = Vec3(
        float(rcpt_frame["origin"]["com"][0]),
        float(rcpt_frame["origin"]["com"][1]),
        float(rcpt_frame["origin"]["com"][2]),
    ) * nanometer
    offset = (lig1cm_pos - rcpt_cm_pos).value_in_unit(angstrom)

    prepared_options: dict[str, Any] = {
        "LIGAND1_ATOMS": ligand1_atoms,
        "LIGAND2_ATOMS": ligand2_atoms,
        "LIGAND1_VAR_ATOMS": ligand1_atoms,
        "LIGAND2_VAR_ATOMS": ligand2_atoms,
        "LIGAND1_ATTACH_ATOM": ligand1_attach_atom.index,
        "LIGAND2_ATTACH_ATOM": ligand2_attach_atom.index,
        "LIGAND1_CM_ATOMS": [ligand1_attach_atom.index],
        "LIGAND2_CM_ATOMS": [ligand2_attach_atom.index],
        "DISPLACEMENT": [displ.x, displ.y, displ.z],
        "RCPT_CM_ATOMS": rcpt_frame["origin"]["indices"],
        "RCPT_FRAME_ATOMS_O": rcpt_frame["origin"]["indices"],
        "RCPT_FRAME_ATOMS_Z": rcpt_frame["z_axis"]["indices"],
        "RCPT_FRAME_ATOMS_Y": rcpt_frame["y_axis"]["indices"],
        "LIGOFFSET": [offset.x, offset.y, offset.z],
        "POS_RESTRAINED_ATOMS": None
        if mode == "abfe"
        else rcpt_frame["origin"]["indices"],
        "EXCLUSION_POT_MOL1_INDEXES": get_indexes_from_query(
            topology,
            f"({rcpt_chain_query}) and (atom.element.atomic_number != 1)",
        ),
        "EXCLUSION_POT_MOL2_INDEXES": get_indexes_from_residue(
            ligand2_residue,
            query="(atom.element.atomic_number != 1)",
        ),
    }

    if options.get("ALIGN_LIGAND1_REF_ATOMS") is not None:
        prepared_options["ALIGN_LIGAND1_REF_ATOMS"] = list(
            options["ALIGN_LIGAND1_REF_ATOMS"]
        )
        prepared_options["ALIGN_LIGAND2_REF_ATOMS"] = list(
            options["ALIGN_LIGAND2_REF_ATOMS"]
        )

    return prepared_options


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
            options=atom_options,
        )
        structprep_ran = True

    max_samples = int(atom_options["MAX_SAMPLES"])
    if samples_before is None or samples_before < max_samples:
        _call_in_workdir(
            workdir,
            rbfe_production,
            config_file=None,
            options=atom_options,
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


def system_has_ghost_pair(pdb_file: str | Path) -> bool:
    try:
        from openmm.app import PDBFile
    except ImportError as exc:
        raise RuntimeError("OpenMM is required to inspect prepared AToM PDB files") from exc

    pdb = PDBFile(str(pdb_file))
    l1_residues = [residue for residue in pdb.topology.residues() if residue.name == "L1"]
    l2_residues = [residue for residue in pdb.topology.residues() if residue.name == "L2"]
    return (
        len(l1_residues) == 1
        and len(l2_residues) == 1
        and len(list(l2_residues[0].atoms())) == 1
    )


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


def _select_attach_atom(
    residue,
    attach_index,
    ref_atoms,
    get_attach_atom_from_residue,
    positions,
):
    if attach_index is not None:
        return list(residue.atoms())[int(attach_index)]
    if ref_atoms is not None:
        return list(residue.atoms())[int(ref_atoms[0])]
    return get_attach_atom_from_residue(residue, positions=positions)


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
