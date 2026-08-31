from __future__ import annotations

import math
from typing import Literal

from gufe import settings as gufe_settings
from gufe.settings import SettingsBaseModel
from openff.units import unit
from pydantic import Field, field_validator, model_validator

from .openfe_compat import (
    OpenFFPartialChargeSettings,
    OpenMMSolvationSettings,
    validate_solvation_settings,
)


def _default_lambdas() -> list[float]:
    return [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ]


def _default_directions() -> list[int]:
    return [1] * 11 + [-1] * 11


def _default_intermediates() -> list[int]:
    return [0] * 10 + [1, 1] + [0] * 10


def _default_lambda1() -> list[float]:
    return [
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.50,
        0.40,
        0.30,
        0.20,
        0.10,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ]


def _default_lambda2() -> list[float]:
    return [
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.50,
        0.50,
        0.50,
        0.50,
        0.50,
        0.50,
        0.50,
        0.50,
        0.50,
        0.50,
        0.50,
        0.40,
        0.30,
        0.20,
        0.10,
        0.00,
    ]


def _default_constant_01() -> list[float]:
    return [0.10] * 22


def _default_constant_110() -> list[float]:
    return [110.0] * 22


def _default_zeros() -> list[float]:
    return [0.0] * 22


class ATMScheduleSettings(SettingsBaseModel):
    """Alchemical schedule fields consumed by AToM transfer protocols."""

    lambdas: list[float] = Field(default_factory=_default_lambdas)
    directions: list[int] = Field(default_factory=_default_directions)
    intermediates: list[int] = Field(default_factory=_default_intermediates)
    lambda1: list[float] = Field(default_factory=_default_lambda1)
    lambda2: list[float] = Field(default_factory=_default_lambda2)
    alpha: list[float] = Field(default_factory=_default_constant_01)
    u0: list[float] = Field(default_factory=_default_constant_110)
    w0coeff: list[float] = Field(default_factory=_default_zeros)
    lambda3: list[float] | None = None
    u1: list[float] | None = None

    @model_validator(mode="after")
    def _validate_schedule(self) -> "ATMScheduleSettings":
        n_states = len(self.lambdas)
        if n_states == 0:
            raise ValueError("AToM schedule must contain at least one state")

        schedule_lists = {
            "directions": self.directions,
            "intermediates": self.intermediates,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "alpha": self.alpha,
            "u0": self.u0,
            "w0coeff": self.w0coeff,
        }
        if self.lambda3 is not None:
            schedule_lists["lambda3"] = self.lambda3
        if self.u1 is not None:
            schedule_lists["u1"] = self.u1

        bad_lengths = {
            name: len(values)
            for name, values in schedule_lists.items()
            if len(values) != n_states
        }
        if bad_lengths:
            raise ValueError(
                "Inconsistent AToM alchemical schedule lengths: "
                f"LAMBDAS has {n_states} states, mismatches are {bad_lengths}"
            )

        if any(v not in (-1, 1) for v in self.directions):
            raise ValueError("directions must contain only 1 or -1")
        if any(v not in (0, 1) for v in self.intermediates):
            raise ValueError("intermediates must contain only 0 or 1")
        if not ({1, -1} <= set(self.directions)):
            raise ValueError("AToM schedules must include both directions")

        bounded_names = ["lambdas", "lambda1", "lambda2"]
        if self.lambda3 is not None:
            bounded_names.append("lambda3")
        for name in bounded_names:
            values = getattr(self, name)
            if any(v < 0.0 or v > 1.0 for v in values):
                raise ValueError(f"{name} values must be between 0 and 1")

        if any(v < 0.0 for v in self.alpha):
            raise ValueError("alpha values must be non-negative")

        return self

    def to_atom_options(self) -> dict[str, object]:
        options: dict[str, object] = {
            "LAMBDAS": list(self.lambdas),
            "DIRECTION": list(self.directions),
            "INTERMEDIATE": list(self.intermediates),
            "LAMBDA1": list(self.lambda1),
            "LAMBDA2": list(self.lambda2),
            "ALPHA": list(self.alpha),
            "U0": list(self.u0),
            "W0COEFF": list(self.w0coeff),
        }
        if self.lambda3 is not None:
            options["LAMBDA3"] = list(self.lambda3)
        if self.u1 is not None:
            options["U1"] = list(self.u1)
        return options


class ATMDisplacementSettings(SettingsBaseModel):
    """Initial ligand displacement into bulk, in Angstrom."""

    displacement: tuple[float, float, float] | None = None

    @field_validator("displacement")
    @classmethod
    def _nonzero_displacement(
        cls, value: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if value is not None and sum(v * v for v in value) == 0.0:
            raise ValueError("displacement vector must be non-zero")
        return value

    def to_atom_options(self) -> dict[str, object]:
        if self.displacement is None:
            return {}
        return {"DISPLACEMENT": list(self.displacement)}


class ATMAlignmentSettings(SettingsBaseModel):
    """Optional local ligand atom indices used by AToM variable displacement."""

    ligand1_ref_atoms: list[int] | None = None
    ligand2_ref_atoms: list[int] | None = None
    ligand1_attach_atom: int | None = None
    ligand2_attach_atom: int | None = None

    @field_validator(
        "ligand1_ref_atoms",
        "ligand2_ref_atoms",
    )
    @classmethod
    def _ref_atoms_valid(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        if len(value) != 3:
            raise ValueError("alignment reference atom lists must contain exactly 3 indices")
        if any(i < 0 for i in value):
            raise ValueError("alignment atom indices must be non-negative")
        return value

    @field_validator("ligand1_attach_atom", "ligand2_attach_atom")
    @classmethod
    def _attach_atoms_valid(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("attachment atom indices must be non-negative")
        return value

    @model_validator(mode="after")
    def _refs_are_paired(self) -> "ATMAlignmentSettings":
        if (self.ligand1_ref_atoms is None) ^ (self.ligand2_ref_atoms is None):
            raise ValueError(
                "ligand1_ref_atoms and ligand2_ref_atoms must be provided together"
            )
        return self

    def to_atom_options(self) -> dict[str, object]:
        options: dict[str, object] = {}
        if self.ligand1_ref_atoms is not None:
            options["ALIGN_LIGAND1_REF_ATOMS"] = list(self.ligand1_ref_atoms)
            options["ALIGN_LIGAND2_REF_ATOMS"] = list(self.ligand2_ref_atoms or [])
        if self.ligand1_attach_atom is not None:
            options["LIGAND1_ATTACH_INDEX"] = self.ligand1_attach_atom
        if self.ligand2_attach_atom is not None:
            options["LIGAND2_ATTACH_INDEX"] = self.ligand2_attach_atom
        return options


class ATMRestraintSettings(SettingsBaseModel):
    """Essential AToM binding-site restraint controls."""

    cm_kf: float = 25.0
    cm_tol: float = 5.0
    posre_force_constant: float = 0.0
    posre_tolerance: float = 3.5
    align_kf_sep: float = 0.0
    align_k_theta: float = 25.0
    align_k_psi: float = 25.0

    @model_validator(mode="after")
    def _positive_values(self) -> "ATMRestraintSettings":
        nonnegative = {
            "cm_kf": self.cm_kf,
            "posre_force_constant": self.posre_force_constant,
            "align_kf_sep": self.align_kf_sep,
            "align_k_theta": self.align_k_theta,
            "align_k_psi": self.align_k_psi,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.cm_tol <= 0:
            raise ValueError("cm_tol must be positive")
        if self.posre_tolerance <= 0:
            raise ValueError("posre_tolerance must be positive")
        return self

    def to_atom_options(self) -> dict[str, object]:
        options: dict[str, object] = {
            "CM_KF": self.cm_kf,
            "CM_TOL": self.cm_tol,
            "POSRE_FORCE_CONSTANT": self.posre_force_constant,
            "POSRE_TOLERANCE": self.posre_tolerance,
            "ALIGN_KF_SEP": self.align_kf_sep,
            "ALIGN_K_THETA": self.align_k_theta,
            "ALIGN_K_PSI": self.align_k_psi,
        }
        return options


class ATMSoftcoreSettings(SettingsBaseModel):
    """AToM softcore parameters."""

    umax: float = 200.0
    ubcore: float = 100.0
    acore: float = 0.0625
    perte_offset: float | None = None

    @model_validator(mode="after")
    def _validate_softcore(self) -> "ATMSoftcoreSettings":
        if self.umax <= 0:
            raise ValueError("umax must be positive")
        if self.ubcore < 0:
            raise ValueError("ubcore must be non-negative")
        if self.ubcore >= self.umax:
            raise ValueError("ubcore must be smaller than umax")
        if self.acore < 0:
            raise ValueError("acore must be non-negative")
        return self

    def to_atom_options(self) -> dict[str, object]:
        options: dict[str, object] = {
            "UMAX": self.umax,
            "UBCORE": self.ubcore,
            "ACORE": self.acore,
        }
        if self.perte_offset is not None:
            options["PERTE_OFFSET"] = self.perte_offset
        return options


class ATMSystemSettings(SettingsBaseModel):
    """Common AToM-specific native system settings."""

    def to_atom_options(self) -> dict[str, object]:
        return {}


class ATMAbsoluteSystemSettings(ATMSystemSettings):
    """ABFE-only native system additions."""

    ghost_mass: float = 12.011

    @field_validator("ghost_mass")
    @classmethod
    def _validate_ghost_mass(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("ghost_mass must be positive")
        return value

    def to_atom_options(self) -> dict[str, object]:
        return {"GHOST_MASS": self.ghost_mass}


class ATMRunSettings(SettingsBaseModel):
    """AToM async replica exchange and OpenMM run controls."""

    basename: str = "atom_abfe"
    re_mode: Literal["async", "sync"] = "async"
    max_samples: int = 10
    wall_time_minutes: int = 60
    cycle_time_seconds: int = 10
    checkpoint_time_seconds: int = 1200
    subjobs_buffer_size: float = 1.0
    production_steps: int = 10000
    print_frequency: int = 10000
    trajectory_frequency: int = 20000
    friction_coeff: float = 0.5
    time_step: float = 0.004
    nodefile: str | None = None
    verbose: bool = False

    @field_validator("basename")
    @classmethod
    def _basename_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("basename must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate_run_controls(self) -> "ATMRunSettings":
        positive_ints = {
            "max_samples": self.max_samples,
            "wall_time_minutes": self.wall_time_minutes,
            "cycle_time_seconds": self.cycle_time_seconds,
            "checkpoint_time_seconds": self.checkpoint_time_seconds,
            "production_steps": self.production_steps,
            "print_frequency": self.print_frequency,
            "trajectory_frequency": self.trajectory_frequency,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.subjobs_buffer_size <= 0:
            raise ValueError("subjobs_buffer_size must be positive")
        if self.print_frequency % self.production_steps != 0:
            raise ValueError("print_frequency must be a multiple of production_steps")
        if self.trajectory_frequency % self.production_steps != 0:
            raise ValueError("trajectory_frequency must be a multiple of production_steps")
        if self.friction_coeff <= 0:
            raise ValueError("friction_coeff must be positive")
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        return self

    def to_atom_options(self) -> dict[str, object]:
        options: dict[str, object] = {
            "BASENAME": self.basename,
            "RE_MODE": self.re_mode,
            "MAX_SAMPLES": self.max_samples,
            "WALL_TIME": self.wall_time_minutes,
            "CYCLE_TIME": self.cycle_time_seconds,
            "CHECKPOINT_TIME": self.checkpoint_time_seconds,
            "SUBJOBS_BUFFER_SIZE": self.subjobs_buffer_size,
            "PRODUCTION_STEPS": self.production_steps,
            "PRNT_FREQUENCY": self.print_frequency,
            "TRJ_FREQUENCY": self.trajectory_frequency,
            "FRICTION_COEFF": self.friction_coeff,
            "TIME_STEP": self.time_step,
            "VERBOSE": self.verbose,
        }
        if self.nodefile is not None:
            options["NODEFILE"] = self.nodefile
        return options


class ATMSetupSettings(SettingsBaseModel):
    """Native setup output and cache controls."""

    write_component_files: bool = True
    forcefield_cache: str | None = "ff.json"

    @field_validator("forcefield_cache")
    @classmethod
    def _cache_name_nonempty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("forcefield_cache must be a non-empty path or None")
        return value


class ATMAnalysisSettings(SettingsBaseModel):
    """Analysis options for parsing AToM/UWHAM output."""

    result_file: str | None = None
    run_uwham: bool = True
    mintimeid: int | None = None
    maxtimeid: int | None = None
    discard_fraction: float = 1.0 / 3.0

    @model_validator(mode="after")
    def _validate_time_window(self) -> "ATMAnalysisSettings":
        if self.mintimeid is not None and self.mintimeid < 0:
            raise ValueError("mintimeid must be non-negative")
        if self.maxtimeid is not None and self.maxtimeid < 0:
            raise ValueError("maxtimeid must be non-negative")
        if (
            self.mintimeid is not None
            and self.maxtimeid is not None
            and self.maxtimeid < self.mintimeid
        ):
            raise ValueError("maxtimeid must be greater than or equal to mintimeid")
        if not 0 <= self.discard_fraction < 1:
            raise ValueError("discard_fraction must be in [0, 1)")
        return self


class ATMSettings(gufe_settings.Settings):
    """Settings for one-box AToM protocols."""

    forcefield_settings: gufe_settings.OpenMMSystemGeneratorFFSettings = Field(
        default_factory=gufe_settings.OpenMMSystemGeneratorFFSettings
    )
    thermo_settings: gufe_settings.ThermoSettings = Field(
        default_factory=lambda: gufe_settings.ThermoSettings(
            temperature=300.0 * unit.kelvin,
            pressure=1.0 * unit.bar,
        )
    )
    solvation_settings: OpenMMSolvationSettings = Field(
        default_factory=OpenMMSolvationSettings
    )
    partial_charge_settings: OpenFFPartialChargeSettings = Field(
        default_factory=OpenFFPartialChargeSettings
    )
    schedule: ATMScheduleSettings = Field(default_factory=ATMScheduleSettings)
    displacement: ATMDisplacementSettings = Field(default_factory=ATMDisplacementSettings)
    alignment: ATMAlignmentSettings = Field(default_factory=ATMAlignmentSettings)
    restraints: ATMRestraintSettings = Field(default_factory=ATMRestraintSettings)
    softcore: ATMSoftcoreSettings = Field(default_factory=ATMSoftcoreSettings)
    system: ATMSystemSettings = Field(default_factory=ATMSystemSettings)
    run: ATMRunSettings = Field(default_factory=ATMRunSettings)
    setup: ATMSetupSettings = Field(default_factory=ATMSetupSettings)
    analysis: ATMAnalysisSettings = Field(default_factory=ATMAnalysisSettings)

    @model_validator(mode="after")
    def _validate_native_setup_support(self) -> "ATMSettings":
        validate_solvation_settings(self.solvation_settings)
        if self.thermo_settings.temperature is None:
            raise ValueError("thermo_settings.temperature is required")
        if self.thermo_settings.pressure is None:
            raise ValueError("thermo_settings.pressure must be 1 bar for AToM")
        pressure = self.thermo_settings.pressure.to(unit.bar).m
        if abs(float(pressure) - 1.0) > 1.0e-8:
            raise ValueError(
                "AToM rbfe_structprep currently runs at 1 bar; "
                "thermo_settings.pressure must be 1 bar"
            )
        supported_methods = {"pme", "ewald", "cutoffperiodic"}
        method = self.forcefield_settings.nonbonded_method.lower()
        if method not in supported_methods:
            raise ValueError(
                "AToM native setup supports periodic PME, Ewald, or "
                f"CutoffPeriodic nonbonded methods; got {method!r}"
            )
        if not self.forcefield_settings.forcefields:
            raise ValueError("forcefield_settings.forcefields must not be empty")
        return self

    def to_atom_options(self) -> dict[str, object]:
        temperature = self.thermo_settings.temperature.to(unit.kelvin).m
        options: dict[str, object] = {"TEMPERATURES": [float(temperature)]}
        for section in (
            self.schedule,
            self.displacement,
            self.restraints,
            self.softcore,
            self.system,
            self.run,
            self.alignment,
        ):
            options.update(section.to_atom_options())
        return options


class ATMAbsoluteBindingSettings(ATMSettings):
    """Settings for AToM ABFE through the ghost-ligand transfer path."""

    system: ATMAbsoluteSystemSettings = Field(
        default_factory=ATMAbsoluteSystemSettings
    )


class ATMRelativeBindingSettings(ATMSettings):
    """Settings for AToM small-molecule RBFE through the transfer path."""

    run: ATMRunSettings = Field(default_factory=lambda: ATMRunSettings(basename="atom_rbfe"))
