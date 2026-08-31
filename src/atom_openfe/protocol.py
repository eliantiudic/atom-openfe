from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from statistics import mean
from typing import Any

import gufe
from gufe import (
    ChemicalSystem,
    LigandAtomMapping,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
)
from gufe.mapping import ComponentMapping
from gufe.protocols import Protocol, ProtocolDAGResult, ProtocolResult, ProtocolUnit
from gufe.protocols.errors import ProtocolValidationError
from openff.units import unit

from . import adapter
from .settings import ATMAbsoluteBindingSettings, ATMRelativeBindingSettings
from .units import ATMAnalysisUnit, ATMRunUnit, ATMSetupUnit


class ATMProtocolResult(ProtocolResult):
    """Aggregated AToM result."""

    def get_estimate(self):
        if "estimate" not in self.data:
            raise ValueError("No estimate is available")
        return self.data["estimate"] * unit.kilocalorie_per_mole

    def get_uncertainty(self):
        if "uncertainty" not in self.data:
            raise ValueError("No uncertainty is available")
        return self.data["uncertainty"] * unit.kilocalorie_per_mole

    def get_individual_estimates(self):
        estimates = self.data.get("unit_estimates", [])
        errors = self.data.get("unit_estimate_errors", [])
        return [
            (
                estimate * unit.kilocalorie_per_mole,
                error * unit.kilocalorie_per_mole,
            )
            for estimate, error in zip(estimates, errors)
        ]

    def get_leg_diagnostics(self) -> list[dict[str, Any]]:
        return list(self.data.get("leg_diagnostics", []))

    def get_artifacts(self) -> list[dict[str, str]]:
        return list(self.data.get("artifacts", []))


class ATMAbsoluteBindingProtocolResult(ATMProtocolResult):
    """Aggregated AToM ABFE result."""


class ATMRelativeBindingProtocolResult(ATMProtocolResult):
    """Aggregated AToM RBFE result."""


class _ATMProtocolMixin:
    _transfer_mode: str

    def _create_atm_units(
        self,
        *,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: ComponentMapping | list[ComponentMapping] | None,
    ) -> list[ProtocolUnit]:
        setup = ATMSetupUnit(
            name="setup",
            protocol=self,
            stateA=stateA,
            stateB=stateB,
            mapping=mapping,
            transfer_mode=self._transfer_mode,
        )
        run = ATMRunUnit(name="run", setup=setup)
        analysis = ATMAnalysisUnit(
            name="analysis",
            protocol=self,
            setup=setup,
            run=run,
        )
        return [setup, run, analysis]

    def _gather(
        self, protocol_dag_results: Iterable[ProtocolDAGResult]
    ) -> dict[str, Any]:
        estimates: list[float] = []
        errors: list[float] = []
        diagnostics: list[dict[str, Any]] = []
        leg_diagnostics: list[dict[str, Any]] = []
        artifacts: list[dict[str, str]] = []

        for dag_result in protocol_dag_results:
            for unit_result in dag_result.terminal_protocol_unit_results:
                if unit_result.name != "analysis" or not unit_result.ok():
                    continue
                estimates.append(float(unit_result.outputs["unit_estimate"]))
                errors.append(float(unit_result.outputs["unit_estimate_error"]))
                diagnostics.append(unit_result.outputs.get("diagnostics", {}))
                artifacts.append(unit_result.outputs.get("artifacts", {}))
                leg_diagnostics.append(
                    {
                        key: unit_result.outputs.get(key)
                        for key in (
                            "dg_leg1",
                            "dg_stderr_leg1",
                            "dg_leg2",
                            "dg_stderr_leg2",
                            "n_samples",
                        )
                    }
                )

        if not estimates:
            return {
                "unit_estimates": [],
                "unit_estimate_errors": [],
                "diagnostics": diagnostics,
                "leg_diagnostics": leg_diagnostics,
                "artifacts": artifacts,
            }

        uncertainty = math.sqrt(sum(error * error for error in errors)) / len(errors)
        return {
            "estimate": mean(estimates),
            "uncertainty": uncertainty,
            "unit_estimates": estimates,
            "unit_estimate_errors": errors,
            "diagnostics": diagnostics,
            "leg_diagnostics": leg_diagnostics,
            "artifacts": artifacts,
        }


class ATMAbsoluteBindingProtocol(_ATMProtocolMixin, Protocol):
    """One-box AToM absolute binding free energy protocol.

    OpenFE-facing states use standard ABFE semantics:
    stateA contains protein, ligand, and solvent; stateB contains the same
    protein/solvent environment without a ligand. Internally AToM receives an
    RBFE-style L1/L2 system where L2 is a natively parameterized one-particle
    ghost present before solvation and serialization.
    """

    _settings_cls = ATMAbsoluteBindingSettings
    result_cls = ATMAbsoluteBindingProtocolResult
    _transfer_mode = "abfe"

    @classmethod
    def _default_settings(cls):
        return ATMAbsoluteBindingSettings()

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: ComponentMapping | list[ComponentMapping] | None = None,
        extends: ProtocolDAGResult | None = None,
    ) -> list[ProtocolUnit]:
        self.validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=extends)
        return self._create_atm_units(stateA=stateA, stateB=stateB, mapping=mapping)

    def _validate(
        self,
        *,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: gufe.ComponentMapping | list[gufe.ComponentMapping] | None = None,
        extends: gufe.ProtocolDAGResult | None = None,
    ):
        if extends is not None:
            raise ProtocolValidationError("AToM ABFE does not support extends yet")

        if mapping is not None:
            warnings.warn("A mapping was passed but is not used by ATMAbsoluteBindingProtocol.")

        protein_a = _require_count(
            stateA, ProteinComponent, 1, "stateA protein"
        )[0]
        solvent_a = _require_count(
            stateA, SolventComponent, 1, "stateA solvent"
        )[0]
        _require_count(stateA, SmallMoleculeComponent, 1, "stateA ligand")
        protein_b = _require_count(
            stateB, ProteinComponent, 1, "stateB protein"
        )[0]
        solvent_b = _require_count(
            stateB, SolventComponent, 1, "stateB solvent"
        )[0]
        _require_count(stateB, SmallMoleculeComponent, 0, "stateB ligand")
        _reject_unhandled_components(stateA, expected=3)
        _reject_unhandled_components(stateB, expected=2)
        _validate_shared_environment(protein_a, protein_b, solvent_a, solvent_b)
        if self.settings.alignment.ligand1_ref_atoms is not None:
            raise ProtocolValidationError(
                "AToM ABFE uses a one-particle L2 ghost and cannot use paired "
                "three-atom ligand alignment references"
            )
        if self.settings.alignment.ligand2_attach_atom not in (None, 0):
            raise ProtocolValidationError(
                "AToM ABFE L2 is a one-particle ghost; ligand2_attach_atom "
                "must be omitted or 0"
            )


class ATMRelativeBindingProtocol(_ATMProtocolMixin, Protocol):
    """One-box AToM small-molecule relative binding free energy protocol."""

    _settings_cls = ATMRelativeBindingSettings
    result_cls = ATMRelativeBindingProtocolResult
    _transfer_mode = "rbfe"

    @classmethod
    def _default_settings(cls):
        return ATMRelativeBindingSettings()

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: ComponentMapping | list[ComponentMapping] | None = None,
        extends: ProtocolDAGResult | None = None,
    ) -> list[ProtocolUnit]:
        self.validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=extends)
        return self._create_atm_units(stateA=stateA, stateB=stateB, mapping=mapping)

    def _validate(
        self,
        *,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: gufe.ComponentMapping | list[gufe.ComponentMapping] | None = None,
        extends: gufe.ProtocolDAGResult | None = None,
    ):
        if extends is not None:
            raise ProtocolValidationError("AToM RBFE does not support extends yet")

        protein_a = _require_count(
            stateA, ProteinComponent, 1, "stateA protein"
        )[0]
        solvent_a = _require_count(
            stateA, SolventComponent, 1, "stateA solvent"
        )[0]
        ligand_a = _require_count(stateA, SmallMoleculeComponent, 1, "stateA ligand")[0]
        protein_b = _require_count(
            stateB, ProteinComponent, 1, "stateB protein"
        )[0]
        solvent_b = _require_count(
            stateB, SolventComponent, 1, "stateB solvent"
        )[0]
        ligand_b = _require_count(stateB, SmallMoleculeComponent, 1, "stateB ligand")[0]
        _reject_unhandled_components(stateA, expected=3)
        _reject_unhandled_components(stateB, expected=3)

        ligand_mapping = _validate_single_ligand_mapping(mapping, ligand_a, ligand_b)
        _validate_shared_environment(protein_a, protein_b, solvent_a, solvent_b)
        if self.settings.alignment.ligand1_ref_atoms is None:
            adapter.derive_mapping_alignment(ligand_mapping)


def _require_count(
    state: ChemicalSystem,
    component_type,
    expected: int,
    label: str,
):
    matches = state.get_components_of_type(component_type)
    if len(matches) != expected:
        raise ProtocolValidationError(
            f"AToM protocol requires exactly {expected} {label} "
            f"({component_type.__name__}); found {len(matches)}"
        )
    return matches


def _validate_single_ligand_mapping(
    mapping: gufe.ComponentMapping | list[gufe.ComponentMapping] | None,
    ligand_a: SmallMoleculeComponent,
    ligand_b: SmallMoleculeComponent,
) -> LigandAtomMapping:
    if isinstance(mapping, list):
        if len(mapping) != 1:
            raise ProtocolValidationError("AToM RBFE requires exactly one LigandAtomMapping")
        mapping = mapping[0]

    if not isinstance(mapping, LigandAtomMapping):
        raise ProtocolValidationError("AToM RBFE requires exactly one LigandAtomMapping")

    if mapping.componentA != ligand_a or mapping.componentB != ligand_b:
        raise ProtocolValidationError(
            "AToM RBFE mapping components must match the stateA and stateB ligands"
        )

    return mapping


def _validate_shared_environment(
    protein_a: ProteinComponent,
    protein_b: ProteinComponent,
    solvent_a: SolventComponent,
    solvent_b: SolventComponent,
) -> None:
    for solvent in (solvent_a, solvent_b):
        if solvent.smiles != "O":
            raise ProtocolValidationError(
                "AToM native setup currently supports only water "
                "SolventComponent(smiles='O')"
            )
    if protein_a != protein_b:
        raise ProtocolValidationError(
            "AToM one-box setup requires identical protein components in stateA "
            "and stateB"
        )
    if solvent_a != solvent_b:
        raise ProtocolValidationError(
            "AToM one-box setup requires identical solvent components in stateA "
            "and stateB"
        )


def _reject_unhandled_components(
    state: ChemicalSystem, *, expected: int
) -> None:
    if len(state.components) != expected:
        raise ProtocolValidationError(
            "AToM native one-box setup does not support additional ChemicalSystem "
            f"components; {state.name!r} has {len(state.components)}, expected "
            f"{expected}"
        )
