"""Narrow compatibility boundary for OpenFE's OpenMM setup helpers.

OpenFE currently exposes these helpers from protocol-internal modules.  Keeping
the imports and the small amount of API adaptation here makes version changes
local instead of coupling the AToM protocol implementation to OpenFE internals.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType

import numpy as np

from gufe.settings import OpenMMSystemGeneratorFFSettings, ThermoSettings
from openff.toolkit import Molecule as OFFMolecule
from openff.units import unit
from openfe.protocols.openmm_utils import (
    charge_generation,
    settings_validation,
    system_creation,
)
from openfe.protocols.openmm_utils.omm_settings import (
    IntegratorSettings,
    OpenFFPartialChargeSettings,
    OpenMMSolvationSettings,
)
from openfe.utils import without_oechem_backend
from openmmforcefields.generators import SystemGenerator


def get_atm_system_generator(
    *,
    forcefield_settings: OpenMMSystemGeneratorFFSettings,
    thermo_settings: ThermoSettings,
    cache: Path | None,
    has_solvent: bool,
) -> tuple[SystemGenerator, dict[str, object]]:
    """Create an OpenFE ``SystemGenerator`` without a serialized barostat.

    The private integrator settings below control only arguments needed by
    OpenFE's system-construction helper.  AToM owns runtime integration and
    adds its own barostat in ``rbfe_structprep``.
    """

    setup_integrator = IntegratorSettings(remove_com=False)
    generator = system_creation.get_system_generator(
        forcefield_settings=forcefield_settings,
        thermo_settings=thermo_settings,
        integrator_settings=setup_integrator,
        cache=cache,
        has_solvent=has_solvent,
    )
    generated_barostat = generator.barostat
    generator.barostat = None

    return generator, {
        "openfe_system_generator": True,
        "setup_remove_com": setup_integrator.remove_com,
        "barostat_suppressed_for_structprep": generated_barostat is not None,
        "forcefield_cache": str(cache) if cache is not None else None,
    }


def assign_offmol_partial_charges(
    molecule: OFFMolecule,
    settings: OpenFFPartialChargeSettings,
) -> None:
    """Apply OpenFE's standard partial-charge convention in place."""

    charge_generation.assign_offmol_partial_charges(
        offmol=molecule,
        overwrite=False,
        method=settings.partial_charge_method,
        toolkit_backend=settings.off_toolkit_backend,
        generate_n_conformers=settings.number_of_conformers,
        nagl_model=settings.nagl_model,
    )


def preserve_zero_partial_charges(generator: SystemGenerator) -> bool:
    """Make the selected template generator honor authoritative zero charges.

    ``openmmforcefields`` normally treats an all-zero charge vector as a
    sentinel for "charges absent" and substitutes its own charge method.  In
    atom-openfe, charge presence has already been resolved from the gufe
    component before this point, so zero is a legitimate authoritative value.
    The instance-local override keeps the rest of the standard template path
    unchanged and final realized charges are validated independently.
    """

    template_generator = generator.template_generator
    if template_generator is None or not hasattr(
        template_generator, "_molecule_has_user_charges"
    ):
        return False
    original = template_generator._molecule_has_user_charges

    def _has_authoritative_charges(self, molecule):
        if molecule.partial_charges is not None:
            charges = molecule.partial_charges.m_as(unit.elementary_charge)
            if np.allclose(charges, 0.0):
                return True
        return original(molecule)

    template_generator._molecule_has_user_charges = MethodType(
        _has_authoritative_charges, template_generator
    )
    return True


def validate_solvation_settings(settings: OpenMMSolvationSettings) -> None:
    """Apply OpenFE's mutual-exclusivity checks for solvation settings."""

    settings_validation.validate_openmm_solvation_settings(settings)


__all__ = [
    "OpenFFPartialChargeSettings",
    "OpenMMSolvationSettings",
    "assign_offmol_partial_charges",
    "get_atm_system_generator",
    "preserve_zero_partial_charges",
    "validate_solvation_settings",
    "without_oechem_backend",
]
