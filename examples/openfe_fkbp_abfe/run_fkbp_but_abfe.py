#!/usr/bin/env python
"""Run the FKBP/but AToM ABFE calculation through OpenFE/gufe objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gufe import ChemicalSystem, ProteinComponent, SmallMoleculeComponent, SolventComponent
from gufe.protocols import execute_DAG
from gufe.transformations import Transformation
from openff.units import unit

from atom_openfe import ATMAbsoluteBindingProtocol


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    fkbp_example = repo_root / "AToM-OpenMM" / "examples" / "ABFE" / "fkbp"

    parser = argparse.ArgumentParser(
        description="Run an OpenFE AToM ABFE calculation for FKBP + but."
    )
    parser.add_argument(
        "--receptor",
        type=Path,
        default=fkbp_example / "receptor" / "fkbp.pdb",
        help="Path to receptor PDB.",
    )
    parser.add_argument(
        "--ligand",
        type=Path,
        default=fkbp_example / "ligands" / "but.sdf",
        help="Path to ligand SDF.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("runs") / "fkbp_but_atom_abfe",
        help="Directory for DAG cache, shared files, scratch, and summary output.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="AToM MAX_SAMPLES. Increase this for production.",
    )
    parser.add_argument(
        "--wall-time-minutes",
        type=int,
        default=60,
        help="AToM WALL_TIME passed to rbfe_production.",
    )
    parser.add_argument(
        "--production-steps",
        type=int,
        default=10000,
        help="AToM PRODUCTION_STEPS per sample.",
    )
    parser.add_argument(
        "--nodefile",
        type=Path,
        default=None,
        help="Optional AToM nodefile for async replica execution.",
    )
    parser.add_argument(
        "--basename",
        default="fkbp_but_atom_abfe",
        help="AToM output basename.",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Keep gufe scratch directories after execution.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable gufe ProtocolUnit result caching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    receptor = ProteinComponent.from_pdb_file(args.receptor, name="fkbp")
    ligand = SmallMoleculeComponent.from_sdf_file(args.ligand)
    solvent = SolventComponent(
        positive_ion="Na+",
        negative_ion="Cl-",
        neutralize=True,
        ion_concentration=0.15 * unit.molar,
    )

    state_a = ChemicalSystem(
        {
            "protein": receptor,
            "ligand": ligand,
            "solvent": solvent,
        },
        name="fkbp_but_bound",
    )
    state_b = ChemicalSystem(
        {
            "protein": receptor,
            "solvent": solvent,
        },
        name="fkbp_unbound",
    )

    settings = ATMAbsoluteBindingProtocol.default_settings()
    settings.run.basename = args.basename
    settings.run.max_samples = args.max_samples
    settings.run.wall_time_minutes = args.wall_time_minutes
    settings.run.production_steps = args.production_steps
    settings.run.print_frequency = args.production_steps
    settings.run.trajectory_frequency = args.production_steps * 2
    if args.nodefile is not None:
        settings.run.nodefile = str(args.nodefile.resolve())

    protocol = ATMAbsoluteBindingProtocol(settings=settings)
    transformation = Transformation(
        stateA=state_a,
        stateB=state_b,
        protocol=protocol,
        mapping=None,
        name="fkbp_but_atom_abfe",
        validate=True,
    )

    work_dir = args.work_dir.resolve()
    shared_dir = work_dir / "shared"
    scratch_dir = work_dir / "scratch"
    cache_dir = None if args.no_cache else work_dir / "cache"
    stdout_dir = work_dir / "stdout"
    stderr_dir = work_dir / "stderr"
    for directory in (shared_dir, scratch_dir, stdout_dir, stderr_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    dag_path = work_dir / "fkbp_but_atom_abfe_dag.json"
    dag = transformation.create()
    dag.to_json(dag_path)

    dag_result = execute_DAG(
        dag,
        shared_basedir=shared_dir,
        scratch_basedir=scratch_dir,
        cache_basedir=cache_dir,
        stdout_basedir=stdout_dir,
        stderr_basedir=stderr_dir,
        keep_shared=True,
        keep_scratch=args.keep_scratch,
        keep_cache=True,
        raise_error=False,
        n_retries=0,
    )
    dag_result.to_json(work_dir / "fkbp_but_atom_abfe_dag_result.json")

    protocol_result = protocol.gather([dag_result])
    summary = {
        "ok": dag_result.ok(),
        "dag": str(dag_path),
        "dag_result": str(work_dir / "fkbp_but_atom_abfe_dag_result.json"),
        "work_dir": str(work_dir),
        "shared_dir": str(shared_dir),
        "scratch_dir": str(scratch_dir),
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "protocol_result_data": protocol_result.data,
    }

    if dag_result.ok():
        estimate = protocol_result.get_estimate().to(unit.kilocalorie_per_mole)
        uncertainty = protocol_result.get_uncertainty().to(unit.kilocalorie_per_mole)
        summary["estimate_kcal_per_mol"] = estimate.m
        summary["uncertainty_kcal_per_mol"] = uncertainty.m
        print(f"Estimate: {estimate.m:.6g} kcal/mol")
        print(f"Uncertainty: {uncertainty.m:.6g} kcal/mol")
    else:
        summary["failures"] = [
            {
                "name": failure.name,
                "source_key": str(failure.source_key),
                "exception": str(failure.exception),
            }
            for failure in dag_result.protocol_unit_failures
        ]
        print("DAG failed. See summary JSON and stderr directory for details.")

    summary_path = work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary: {summary_path}")

    if not dag_result.ok():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
