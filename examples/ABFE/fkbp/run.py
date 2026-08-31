#!/usr/bin/env python
"""Run the FKBP AToM ABFE examples through OpenFE/gufe objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gufe import ChemicalSystem, ProteinComponent, SmallMoleculeComponent, SolventComponent
from gufe.protocols import execute_DAG
from gufe.transformations import Transformation
from openff.units import unit

from atom_openfe import ATMAbsoluteBindingProtocol


EXAMPLE_DIR = Path(__file__).resolve().parent


def available_ligands(ligand_dir: Path) -> list[str]:
    return sorted(path.stem for path in ligand_dir.glob("*.sdf"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run atom-openfe ABFE calculations for the FKBP example ligands."
    )
    parser.add_argument(
        "--ligand",
        action="append",
        help="Ligand basename from ligands/*.sdf. Repeat this option or use 'all'.",
    )
    parser.add_argument(
        "--receptor",
        type=Path,
        default=EXAMPLE_DIR / "receptor" / "fkbp.pdb",
        help="Path to the FKBP receptor PDB.",
    )
    parser.add_argument(
        "--ligand-dir",
        type=Path,
        default=EXAMPLE_DIR / "ligands",
        help="Directory containing ligand SDF files.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=EXAMPLE_DIR / "runs",
        help="Directory for per-ligand DAG cache, scratch, shared files, and summaries.",
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
        help="AToM WALL_TIME.",
    )
    parser.add_argument(
        "--production-steps",
        type=int,
        default=10000,
        help="AToM PRODUCTION_STEPS per sample.",
    )
    parser.add_argument(
        "--ligand-attach-atom-index",
        type=int,
        default=None,
        help="Optional 1-based ligand attachment atom index.",
    )
    parser.add_argument(
        "--nodefile",
        type=Path,
        default=None,
        help="Optional AToM nodefile for async replica execution.",
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
    args = parser.parse_args()

    if args.ligand_attach_atom_index is not None and args.ligand_attach_atom_index < 1:
        parser.error("--ligand-attach-atom-index is 1-based and must be positive")

    return args


def selected_ligands(requested: list[str] | None, ligand_dir: Path) -> list[str]:
    available = available_ligands(ligand_dir)
    if not available:
        raise ValueError(f"No ligand SDF files found in {ligand_dir}")

    names = requested or ["but"]
    if "all" in names:
        return available

    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(
            f"Unknown ligand(s): {', '.join(missing)}. "
            f"Available ligands: {', '.join(available)}"
        )

    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def build_transformation(
    *,
    receptor_path: Path,
    ligand_path: Path,
    ligand_name: str,
    settings,
) -> Transformation:
    receptor = ProteinComponent.from_pdb_file(receptor_path, name="fkbp")
    ligand = SmallMoleculeComponent.from_sdf_file(ligand_path)
    solvent = SolventComponent(
        positive_ion="Na+",
        negative_ion="Cl-",
        neutralize=True,
        ion_concentration=0.15 * unit.molar,
    )

    bound = ChemicalSystem(
        {
            "protein": receptor,
            "ligand": ligand,
            "solvent": solvent,
        },
        name=f"fkbp_{ligand_name}_bound",
    )
    unbound = ChemicalSystem(
        {
            "protein": receptor,
            "solvent": solvent,
        },
        name="fkbp_unbound",
    )
    protocol = ATMAbsoluteBindingProtocol(settings=settings)

    return Transformation(
        stateA=bound,
        stateB=unbound,
        protocol=protocol,
        mapping=None,
        name=f"fkbp_{ligand_name}_abfe",
        validate=True,
    )


def run_ligand(args: argparse.Namespace, ligand_name: str) -> dict[str, object]:
    job_name = f"fkbp-{ligand_name}"
    ligand_path = args.ligand_dir / f"{ligand_name}.sdf"
    work_dir = (args.work_dir / job_name).resolve()
    shared_dir = work_dir / "shared"
    scratch_dir = work_dir / "scratch"
    cache_dir = None if args.no_cache else work_dir / "cache"
    stdout_dir = work_dir / "stdout"
    stderr_dir = work_dir / "stderr"

    for directory in (shared_dir, scratch_dir, stdout_dir, stderr_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    settings = ATMAbsoluteBindingProtocol.default_settings()
    settings.run.basename = job_name
    settings.run.max_samples = args.max_samples
    settings.run.wall_time_minutes = args.wall_time_minutes
    settings.run.production_steps = args.production_steps
    settings.run.print_frequency = args.production_steps
    settings.run.trajectory_frequency = args.production_steps * 2
    if args.ligand_attach_atom_index is not None:
        settings.alignment.ligand1_attach_atom = args.ligand_attach_atom_index - 1
    if args.nodefile is not None:
        settings.run.nodefile = str(args.nodefile.resolve())

    transformation = build_transformation(
        receptor_path=args.receptor,
        ligand_path=ligand_path,
        ligand_name=ligand_name,
        settings=settings,
    )

    dag_path = work_dir / f"{job_name}_dag.json"
    dag_result_path = work_dir / f"{job_name}_dag_result.json"

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
    dag_result.to_json(dag_result_path)

    protocol_result = transformation.protocol.gather([dag_result])
    summary: dict[str, object] = {
        "ok": dag_result.ok(),
        "ligand": ligand_name,
        "receptor": str(args.receptor.resolve()),
        "ligand_sdf": str(ligand_path.resolve()),
        "dag": str(dag_path),
        "dag_result": str(dag_result_path),
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
        print(f"{job_name}: {estimate.m:.6g} +/- {uncertainty.m:.6g} kcal/mol")
    else:
        summary["failures"] = [
            {
                "name": failure.name,
                "source_key": str(failure.source_key),
                "exception": str(failure.exception),
            }
            for failure in dag_result.protocol_unit_failures
        ]
        print(f"{job_name}: DAG failed; see {stderr_dir}")

    summary_path = work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"{job_name}: summary written to {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    ligand_names = selected_ligands(args.ligand, args.ligand_dir)
    summaries = [run_ligand(args, ligand_name) for ligand_name in ligand_names]

    if not all(summary["ok"] for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
