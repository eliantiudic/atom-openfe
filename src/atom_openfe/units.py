from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from gufe import (
    ChemicalSystem,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
)
from gufe.protocols import Context, ProtocolUnit, ProtocolUnitResult

from . import adapter
from .analysis import parse_atom_analysis_output
from .settings import ATMSettings


class ATMSetupUnit(ProtocolUnit):
    """Validate gufe inputs and prepare an AToM work directory."""

    @staticmethod
    def _execute(
        ctx: Context,
        *,
        protocol,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        transfer_mode: str,
        mapping=None,
        **inputs,
    ) -> dict[str, Any]:
        protocol.validate(stateA=stateA, stateB=stateB, mapping=mapping)
        settings: ATMSettings = protocol.settings

        ctx.shared.mkdir(parents=True, exist_ok=True)

        diagnostics: dict[str, Any] = {
            "stateA": stateA.name,
            "stateB": stateB.name,
            "transfer_mode": transfer_mode,
        }
        components = _setup_components(stateA, stateB, transfer_mode)
        component_files = (
            _write_component_files(
                stateA=stateA,
                stateB=stateB,
                transfer_mode=transfer_mode,
                shared=ctx.shared,
            )
            if settings.setup.write_component_files
            else {}
        )

        atom_options = settings.to_atom_options()
        atom_options["WORKDIR"] = str(ctx.shared)
        atom_options.update(_alignment_options(settings, transfer_mode, mapping))

        prepared = adapter.prepare_atm_transfer_system(
            mode=transfer_mode,
            options=atom_options,
            workdir=ctx.shared,
            receptor=components["receptor"],
            ligand1=components["ligand1"],
            ligand2=components.get("ligand2"),
            solvent=components["solvent"],
            forcefield_settings=settings.forcefield_settings,
            thermo_settings=settings.thermo_settings,
            solvation_settings=settings.solvation_settings,
            partial_charge_settings=settings.partial_charge_settings,
            ghost_mass=(
                settings.system.ghost_mass
                if transfer_mode == "abfe"
                else None
            ),
            forcefield_cache=settings.setup.forcefield_cache,
        )

        diagnostics.update(prepared.get("diagnostics", {}))
        component_files.update(
            {
                "prepared_system_pdb": prepared["prepared_system_pdb"],
                "prepared_system_xml": prepared["prepared_system_xml"],
            }
        )

        return {
            "transfer_mode": transfer_mode,
            "atom_options": prepared["atom_options"],
            "atom_options_path": prepared["atom_options_path"],
            "shared_dir": str(ctx.shared),
            "component_files": component_files,
            "artifacts": adapter.transfer_artifacts(
                ctx.shared, prepared["atom_options"]["BASENAME"]
            ),
            "diagnostics": diagnostics,
        }


class ATMRunUnit(ProtocolUnit):
    """Execute AToM RBFE-style structure preparation and production."""

    @staticmethod
    def _execute(
        ctx: Context,
        *,
        setup: ProtocolUnitResult,
        **inputs,
    ) -> dict[str, Any]:
        setup_outputs = setup.outputs
        workdir = Path(setup_outputs["shared_dir"])
        options = dict(setup_outputs["atom_options"])
        options["WORKDIR"] = str(workdir)

        atom_result = adapter.run_atm_transfer(
            options=options,
            workdir=workdir,
            config_file=setup_outputs["atom_options_path"],
        )

        return {"atom_result": atom_result}


class ATMAnalysisUnit(ProtocolUnit):
    """Parse AToM/UWHAM output into gufe result fields."""

    @staticmethod
    def _execute(
        ctx: Context,
        *,
        protocol,
        setup: ProtocolUnitResult,
        run: ProtocolUnitResult,
        **inputs,
    ) -> dict[str, Any]:
        settings: ATMSettings = protocol.settings
        atom_result = dict(run.outputs.get("atom_result", {}))

        if {"unit_estimate", "unit_estimate_error"} <= set(atom_result):
            parsed = _coerce_analysis_mapping(atom_result)
        else:
            result_file = _resolve_result_file(settings, setup, atom_result)
            if result_file is not None:
                parsed = parse_atom_analysis_output(result_file)
            elif settings.analysis.run_uwham:
                setup_outputs = setup.outputs
                parsed = adapter.analyze_atm_uwham(
                    workdir=setup_outputs["shared_dir"],
                    basename=setup_outputs["atom_options"]["BASENAME"],
                    mintimeid=settings.analysis.mintimeid,
                    maxtimeid=settings.analysis.maxtimeid,
                    discard_fraction=settings.analysis.discard_fraction,
                )
            else:
                raise FileNotFoundError("No AToM analysis output was provided or found")

        artifacts = {}
        artifacts.update(setup.outputs.get("artifacts", {}))
        artifacts.update(atom_result.get("artifacts", {}))

        diagnostics = {
            "setup": setup.outputs.get("diagnostics", {}),
            "run": {
                key: value
                for key, value in atom_result.items()
                if key
                in {
                    "status",
                    "structprep_ran",
                    "production_ran",
                    "samples_before",
                    "samples_after",
                }
            },
            "analysis": parsed.get("diagnostics", {}),
        }
        return {
            "unit_estimate": parsed["unit_estimate"],
            "unit_estimate_error": parsed["unit_estimate_error"],
            "dg_leg1": parsed.get("dg_leg1"),
            "dg_stderr_leg1": parsed.get("dg_stderr_leg1"),
            "dg_leg2": parsed.get("dg_leg2"),
            "dg_stderr_leg2": parsed.get("dg_stderr_leg2"),
            "n_samples": parsed.get("n_samples"),
            "artifacts": artifacts,
            "diagnostics": diagnostics,
        }


def _write_component_files(
    *,
    stateA: ChemicalSystem,
    stateB: ChemicalSystem,
    transfer_mode: str,
    shared: Path,
) -> dict[str, str]:
    files: dict[str, str] = {}

    ligand1 = _one_component(stateA, SmallMoleculeComponent, "stateA ligand")
    ligand1_path = shared / "ligand1.sdf"
    ligand1_path.write_text(ligand1.to_sdf())
    files["ligand1_sdf"] = str(ligand1_path)

    if transfer_mode == "rbfe":
        ligand2 = _one_component(stateB, SmallMoleculeComponent, "stateB ligand")
        ligand2_path = shared / "ligand2.sdf"
        ligand2_path.write_text(ligand2.to_sdf())
        files["ligand2_sdf"] = str(ligand2_path)

    protein = _one_component(stateA, ProteinComponent, "stateA protein")
    receptor_path = shared / "receptor.pdb"
    protein.to_pdb_file(receptor_path)
    files["receptor_pdb"] = str(receptor_path)

    manifest_path = shared / "component_manifest.yaml"
    manifest = {
        "stateA": _manifest_for_state(stateA),
        "stateB": _manifest_for_state(stateB),
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True))
    files["component_manifest"] = str(manifest_path)

    return files


def _setup_components(
    stateA: ChemicalSystem,
    stateB: ChemicalSystem,
    transfer_mode: str,
) -> dict[str, Any]:
    components: dict[str, Any] = {
        "receptor": _one_component(stateA, ProteinComponent, "stateA protein"),
        "ligand1": _one_component(
            stateA, SmallMoleculeComponent, "stateA ligand"
        ),
        "solvent": _one_component(stateA, SolventComponent, "stateA solvent"),
    }
    if transfer_mode == "rbfe":
        components["ligand2"] = _one_component(
            stateB, SmallMoleculeComponent, "stateB ligand"
        )
    return components


def _alignment_options(
    settings: ATMSettings,
    transfer_mode: str,
    mapping,
) -> dict[str, Any]:
    options = settings.alignment.to_atom_options()

    if transfer_mode == "abfe":
        return options

    if "ALIGN_LIGAND1_REF_ATOMS" in options:
        return options

    resolved_mapping = _single_mapping(mapping)
    options.update(adapter.derive_mapping_alignment(resolved_mapping))
    if settings.alignment.ligand1_attach_atom is not None:
        options["LIGAND1_ATTACH_INDEX"] = settings.alignment.ligand1_attach_atom
    if settings.alignment.ligand2_attach_atom is not None:
        options["LIGAND2_ATTACH_INDEX"] = settings.alignment.ligand2_attach_atom
    return options


def _resolve_result_file(
    settings: ATMSettings,
    setup: ProtocolUnitResult,
    atom_result: dict[str, Any],
) -> Path | None:
    if atom_result.get("analysis_file") is not None:
        return Path(atom_result["analysis_file"])

    if settings.analysis.result_file is not None:
        result_path = Path(settings.analysis.result_file)
        if result_path.is_absolute():
            return result_path
        return Path(setup.outputs["shared_dir"]) / result_path

    basename = setup.outputs["atom_options"]["BASENAME"]
    workdir = Path(setup.outputs["shared_dir"])
    candidates = [
        workdir / f"{basename}_analysis.yaml",
        workdir / f"{basename}_analysis.yml",
        workdir / f"{basename}_analysis.json",
        workdir / "analysis.yaml",
        workdir / "analysis.yml",
        workdir / "analysis.json",
        workdir / "uwham_results.yaml",
        workdir / "uwham_results.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _coerce_analysis_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "unit_estimate": float(raw["unit_estimate"]),
        "unit_estimate_error": float(raw["unit_estimate_error"]),
        "dg_leg1": _maybe_float(raw.get("dg_leg1")),
        "dg_stderr_leg1": _maybe_float(raw.get("dg_stderr_leg1")),
        "dg_leg2": _maybe_float(raw.get("dg_leg2")),
        "dg_stderr_leg2": _maybe_float(raw.get("dg_stderr_leg2")),
        "n_samples": raw.get("n_samples"),
        "diagnostics": raw.get("diagnostics", {}),
    }


def _one_component(state: ChemicalSystem, component_type, label: str):
    components = state.get_components_of_type(component_type)
    if len(components) != 1:
        raise ValueError(f"Expected exactly one {label}; found {len(components)}")
    return components[0]


def _manifest_for_state(state: ChemicalSystem) -> dict[str, dict[str, str]]:
    return {
        label: {
            "component_type": component.__class__.__name__,
            "name": getattr(component, "name", ""),
        }
        for label, component in state.components.items()
    }


def _single_mapping(mapping):
    if isinstance(mapping, list):
        if len(mapping) != 1:
            raise ValueError("AToM RBFE requires exactly one LigandAtomMapping")
        return mapping[0]
    if mapping is None:
        raise ValueError("AToM RBFE requires exactly one LigandAtomMapping")
    return mapping


def _maybe_float(value):
    if value is None:
        return None
    return float(value)
