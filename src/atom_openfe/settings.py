from __future__ import annotations

from typing import Literal

from gufe import settings as gufe_settings
from gufe.settings import SettingsBaseModel
from openff.units import unit
from pydantic import Field, field_validator, model_validator


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

    temperatures: list[float] = Field(default_factory=lambda: [300.0])
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

    @field_validator("temperatures")
    @classmethod
    def _temperatures_nonempty(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("AToM schedules require at least one temperature")
        if any(v <= 0 for v in value):
            raise ValueError("temperatures must be positive Kelvin values")
        return value

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
            "TEMPERATURES": list(self.temperatures),
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
            options["LIGAND_ATTACH_INDEX"] = self.ligand1_attach_atom
        if self.ligand2_attach_atom is not None:
            options["LIGAND2_ATTACH_INDEX"] = self.ligand2_attach_atom
        return options


class ATMAtomIndexSettings(SettingsBaseModel):
    """Compatibility settings for caller-provided prepared-system indices."""

    ligand_atoms: list[int] | None = None
    ligand_atoms0: list[int] | None = None
    ligand_cm_atoms: list[int] | None = None
    receptor_cm_atoms: list[int] | None = None
    pos_restrained_atoms: list[int] | None = None

    @field_validator(
        "ligand_atoms",
        "ligand_atoms0",
        "ligand_cm_atoms",
        "receptor_cm_atoms",
        "pos_restrained_atoms",
    )
    @classmethod
    def _indices_nonnegative(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and any(i < 0 for i in value):
            raise ValueError("atom indices must be non-negative")
        return value

    def to_atom_options(self) -> dict[str, object]:
        mapping = {
            "LIGAND_ATOMS": self.ligand_atoms,
            "LIGAND_ATOMS0": self.ligand_atoms0,
            "LIGAND_CM_ATOMS": self.ligand_cm_atoms,
            "RCPT_CM_ATOMS": self.receptor_cm_atoms,
            "POS_RESTRAINED_ATOMS": self.pos_restrained_atoms,
        }
        return {key: value for key, value in mapping.items() if value is not None}


class ATMRestraintSettings(SettingsBaseModel):
    """Essential AToM binding-site restraint controls."""

    cm_kf: float = 25.0
    cm_tol: float = 5.0
    ligand_offset: tuple[float, float, float] | None = None
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
        if self.ligand_offset is not None:
            options["LIGOFFSET"] = list(self.ligand_offset)
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
    """AToM system construction settings."""

    protein_forcefields: list[str] = Field(default_factory=lambda: ["amber14-all.xml"])
    solvent_forcefields: list[str] = Field(default_factory=lambda: ["amber14/tip3p.xml"])
    ligand_forcefield: str = "openff-2.3.0"
    ionic_strength: float = 0.15
    forcefield_cache: str | None = "ff.json"
    receptor_chain_names: list[str] = Field(default_factory=lambda: ["A"])
    ghost_mass: float = 12.011

    @model_validator(mode="after")
    def _validate_system_settings(self) -> "ATMSystemSettings":
        if not self.protein_forcefields:
            raise ValueError("protein_forcefields must not be empty")
        if not self.solvent_forcefields:
            raise ValueError("solvent_forcefields must not be empty")
        if not self.ligand_forcefield.strip():
            raise ValueError("ligand_forcefield must be non-empty")
        if self.ionic_strength < 0:
            raise ValueError("ionic_strength must be non-negative")
        if not self.receptor_chain_names:
            raise ValueError("receptor_chain_names must not be empty")
        if self.ghost_mass <= 0:
            raise ValueError("ghost_mass must be positive")
        return self

    def to_atom_options(self) -> dict[str, object]:
        options: dict[str, object] = {
            "LIGAND_FORCE_FIELD": self.ligand_forcefield,
            "RCPT_CHAIN_NAMES": list(self.receptor_chain_names),
            "GHOST_MASS": self.ghost_mass,
        }
        if self.forcefield_cache is not None:
            options["FORCEFIELD_CACHE"] = self.forcefield_cache
        return options


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
    hmass: float = 1.5
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
        if self.hmass <= 0:
            raise ValueError("hmass must be positive")
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
            "HMASS": self.hmass,
            "TIME_STEP": self.time_step,
            "VERBOSE": self.verbose,
        }
        if self.nodefile is not None:
            options["NODEFILE"] = self.nodefile
        return options


class ATMSetupSettings(SettingsBaseModel):
    """Setup options that bridge gufe components to AToM files."""

    prepared_system_pdb: str | None = None
    prepared_system_xml: str | None = None
    write_component_files: bool = True

    @model_validator(mode="after")
    def _prepared_files_are_paired(self) -> "ATMSetupSettings":
        if (self.prepared_system_pdb is None) ^ (self.prepared_system_xml is None):
            raise ValueError(
                "prepared_system_pdb and prepared_system_xml must be provided together"
            )
        return self


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


class ATMTransferSettings(gufe_settings.Settings):
    """Settings for one-box AToM transfer protocols."""

    forcefield_settings: gufe_settings.BaseForceFieldSettings = Field(
        default_factory=gufe_settings.OpenMMSystemGeneratorFFSettings
    )
    thermo_settings: gufe_settings.ThermoSettings = Field(
        default_factory=lambda: gufe_settings.ThermoSettings(
            temperature=300.0 * unit.kelvin,
            pressure=1.0 * unit.bar,
        )
    )
    schedule: ATMScheduleSettings = Field(default_factory=ATMScheduleSettings)
    displacement: ATMDisplacementSettings = Field(default_factory=ATMDisplacementSettings)
    alignment: ATMAlignmentSettings = Field(default_factory=ATMAlignmentSettings)
    atom_indices: ATMAtomIndexSettings = Field(default_factory=ATMAtomIndexSettings)
    restraints: ATMRestraintSettings = Field(default_factory=ATMRestraintSettings)
    softcore: ATMSoftcoreSettings = Field(default_factory=ATMSoftcoreSettings)
    system: ATMSystemSettings = Field(default_factory=ATMSystemSettings)
    run: ATMRunSettings = Field(default_factory=ATMRunSettings)
    setup: ATMSetupSettings = Field(default_factory=ATMSetupSettings)
    analysis: ATMAnalysisSettings = Field(default_factory=ATMAnalysisSettings)

    def to_atom_options(self) -> dict[str, object]:
        options: dict[str, object] = {}
        for section in (
            self.schedule,
            self.displacement,
            self.restraints,
            self.softcore,
            self.system,
            self.run,
            self.alignment,
            self.atom_indices,
        ):
            options.update(section.to_atom_options())
        return options


class ATMAbsoluteBindingSettings(ATMTransferSettings):
    """Settings for AToM ABFE through the ghost-ligand transfer path."""


class ATMRelativeBindingSettings(ATMTransferSettings):
    """Settings for AToM small-molecule RBFE through the transfer path."""

    run: ATMRunSettings = Field(default_factory=lambda: ATMRunSettings(basename="atom_rbfe"))
