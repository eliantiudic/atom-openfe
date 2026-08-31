"""Strict validation for native AToM prepared systems."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from gufe.settings import OpenMMSystemGeneratorFFSettings
from openff.units import unit as off_unit
from openmm import Context, NonbondedForce, Platform, VerletIntegrator
from openmm import unit as omm_unit

if TYPE_CHECKING:
    from .system_builder import PreparedATMSystem


_SOLVENT_RESIDUES = {
    "HOH",
    "TIP3",
    "WAT",
    "TIP4",
    "OPC",
    "TIP5",
    "POT",
    "SOD",
    "CLA",
    "NA+",
    "K+",
    "CL-",
    "F-",
    "CA",
    "MG",
    "CL",
    "NA",
    "K",
    "F",
}

_SUPPORTED_FORCE_CLASSES = {
    "HarmonicBondForce": "Harmonic",
    "HarmonicAngleForce": "Harmonic",
    "PeriodicTorsionForce": "Torsion",
    "NonbondedForce": "Nonbonded",
}

# rbfe_structprep adds a CutoffPeriodic receptor/L2 exclusion force with this
# default cutoff (10 Angstrom), even when the source NonbondedForce cutoff is
# smaller.  Native box validation must be safe for both stages.
_ATOM_EXCLUSION_CUTOFF_NM = 1.0


def validate_prepared_system(
    prepared: "PreparedATMSystem",
    *,
    mode: Literal["rbfe", "abfe"],
    forcefield_settings: OpenMMSystemGeneratorFFSettings,
    ghost_mass: float | None,
    check_energy: bool,
) -> dict[str, Any]:
    """Validate topology, System, AToM indices, box, forces, and energy."""

    topology = prepared.topology
    system = prepared.system
    positions = prepared.positions
    atom_count = topology.getNumAtoms()

    if system.getNumParticles() != atom_count:
        raise ValueError(
            "Prepared AToM topology/System particle counts differ: "
            f"{atom_count} topology atoms vs {system.getNumParticles()} particles"
        )
    if len(positions) != atom_count:
        raise ValueError(
            f"Prepared AToM positions contain {len(positions)} entries for "
            f"{atom_count} topology atoms"
        )
    if [atom.index for atom in topology.atoms()] != list(range(atom_count)):
        raise ValueError("OpenMM topology atom indices are not sequential")

    _validate_ligand_blocks(prepared)
    _validate_index_bounds(prepared.atom_options, atom_count)
    _validate_index_ownership(prepared)
    _validate_component_mappings(prepared, atom_count, mode)
    _validate_solvent_names(prepared)
    box_metrics = _validate_box(prepared, forcefield_settings)
    nonbonded = _validate_forces(prepared, forcefield_settings)
    _validate_realized_ligand_charges(prepared, nonbonded, mode)
    if mode == "abfe":
        if ghost_mass is None:
            raise ValueError("ABFE ghost validation requires a configured mass")
        _validate_ghost(prepared, nonbonded, ghost_mass)

    energy = None
    if check_energy:
        energy = _finite_potential_energy(system, positions)

    return {
        "topology_atoms": atom_count,
        "system_particles": system.getNumParticles(),
        "box_minimum_height_nm": box_metrics["minimum_height_nm"],
        "box_minimum_margin_nm": box_metrics["minimum_margin_nm"],
        "box_minimum_image_distance_nm": box_metrics[
            "minimum_image_distance_nm"
        ],
        "box_effective_cutoff_nm": box_metrics["effective_cutoff_nm"],
        "potential_energy_kj_per_mole": energy,
        "finite_potential_energy": energy is None or math.isfinite(energy),
    }


def _validate_ligand_blocks(prepared: "PreparedATMSystem") -> None:
    topology = prepared.topology
    l1 = prepared.role_atom_indices.get("L1", ())
    l2 = prepared.role_atom_indices.get("L2", ())
    if not l1 or not l2:
        raise ValueError("Prepared AToM mappings must contain non-empty L1 and L2 blocks")
    if tuple(l1) != tuple(range(l1[0], l1[0] + len(l1))):
        raise ValueError("L1 atoms are not one contiguous global-index block")
    if tuple(l2) != tuple(range(l2[0], l2[0] + len(l2))):
        raise ValueError("L2 atoms are not one contiguous global-index block")
    if l2[0] != l1[-1] + 1:
        raise ValueError("L2 must immediately follow L1 in the prepared topology")

    residues = list(topology.residues())
    l1_residues = [residue for residue in residues if residue.name == "L1"]
    l2_residues = [residue for residue in residues if residue.name == "L2"]
    if len(l1_residues) != 1 or len(l2_residues) != 1:
        raise ValueError("Prepared AToM topology requires exactly one L1 and one L2")
    residue1, residue2 = l1_residues[0], l2_residues[0]
    if residue2.index != residue1.index + 1:
        raise ValueError("The L1 and L2 residues are not adjacent")
    if residue1.chain.id != "L" or residue2.chain.id != "M":
        raise ValueError("AToM ligand chains must be L for L1 and M for L2")
    if tuple(atom.index for atom in residue1.atoms()) != tuple(l1):
        raise ValueError("The L1 residue atom order does not match its mapping")
    if tuple(atom.index for atom in residue2.atoms()) != tuple(l2):
        raise ValueError("The L2 residue atom order does not match its mapping")
    for residue in (residue1, residue2):
        names = [atom.name for atom in residue.atoms()]
        if len(names) != len(set(names)) or any(not name.strip() for name in names):
            raise ValueError(f"Residue {residue.name} does not have unique atom names")

    receptor = prepared.role_atom_indices.get("receptor", ())
    solvent = prepared.role_atom_indices.get("solvent", ())
    if tuple(receptor) != tuple(range(l1[0])):
        raise ValueError("The receptor must be the complete first atom block")
    if not solvent or tuple(solvent) != tuple(
        range(l2[-1] + 1, topology.getNumAtoms())
    ):
        raise ValueError("Solvent must be the complete final atom block after L2")
    for chain in topology.chains():
        if any(atom.index in set(receptor) for atom in chain.atoms()):
            if chain.id in {"L", "M", "X", ""}:
                raise ValueError(
                    f"Receptor chain id {chain.id!r} conflicts with AToM conventions"
                )

    option_l1 = tuple(prepared.atom_options.get("LIGAND1_ATOMS", ()))
    option_l2 = tuple(prepared.atom_options.get("LIGAND2_ATOMS", ()))
    if option_l1 != tuple(l1) or option_l2 != tuple(l2):
        raise ValueError("AToM ligand options do not match the final topology blocks")

    role_residues = prepared.role_residue_indices
    receptor_residues = tuple(role_residues.get("receptor", ()))
    l1_role_residues = tuple(role_residues.get("L1", ()))
    l2_role_residues = tuple(role_residues.get("L2", ()))
    solvent_residues = tuple(role_residues.get("solvent", ()))
    expected_residue_order = (
        receptor_residues
        + l1_role_residues
        + l2_role_residues
        + solvent_residues
    )
    if expected_residue_order != tuple(range(topology.getNumResidues())):
        raise ValueError("Role residue mappings do not cover the final topology in order")
    if l1_role_residues != (residue1.index,) or l2_role_residues != (
        residue2.index,
    ):
        raise ValueError("L1/L2 role residue mappings do not match their residues")
    atoms_by_role = {
        role: tuple(
            atom.index
            for residue_index in residue_indices
            for atom in residues[residue_index].atoms()
        )
        for role, residue_indices in role_residues.items()
    }
    for role in ("receptor", "L1", "L2", "solvent"):
        if atoms_by_role.get(role, ()) != tuple(
            prepared.role_atom_indices.get(role, ())
        ):
            raise ValueError(
                f"{role} atom and residue role mappings are inconsistent"
            )


def _validate_index_ownership(prepared: "PreparedATMSystem") -> None:
    options = prepared.atom_options
    roles = prepared.role_atom_indices

    def require_subset(key: str, role: str) -> None:
        values = options.get(key)
        if values is None:
            return
        if isinstance(values, int):
            values = [values]
        if not set(values) <= set(roles[role]):
            raise ValueError(f"AToM option {key} contains indices outside {role}")

    for key in (
        "LIGAND1_ATOMS",
        "LIGAND1_VAR_ATOMS",
        "LIGAND1_CM_ATOMS",
        "LIGAND1_ATTACH_ATOM",
    ):
        require_subset(key, "L1")
    for key in (
        "LIGAND2_ATOMS",
        "LIGAND2_VAR_ATOMS",
        "LIGAND2_CM_ATOMS",
        "LIGAND2_ATTACH_ATOM",
        "EXCLUSION_POT_MOL2_INDEXES",
    ):
        require_subset(key, "L2")
    for key in (
        "RCPT_CM_ATOMS",
        "RCPT_FRAME_ATOMS_O",
        "RCPT_FRAME_ATOMS_Z",
        "RCPT_FRAME_ATOMS_Y",
        "POS_RESTRAINED_ATOMS",
        "EXCLUSION_POT_MOL1_INDEXES",
    ):
        require_subset(key, "receptor")

    receptor = set(roles["receptor"])
    actual_chain_ids = [
        chain.id
        for chain in prepared.topology.chains()
        if any(atom.index in receptor for atom in chain.atoms())
    ]
    if list(options.get("RCPT_CHAIN_NAMES", ())) != actual_chain_ids:
        raise ValueError(
            "RCPT_CHAIN_NAMES does not match the final receptor chains"
        )


def _validate_index_bounds(options: dict[str, Any], atom_count: int) -> None:
    global_list_keys = {
        "LIGAND1_ATOMS",
        "LIGAND2_ATOMS",
        "LIGAND1_VAR_ATOMS",
        "LIGAND2_VAR_ATOMS",
        "LIGAND1_CM_ATOMS",
        "LIGAND2_CM_ATOMS",
        "RCPT_CM_ATOMS",
        "RCPT_FRAME_ATOMS_O",
        "RCPT_FRAME_ATOMS_Z",
        "RCPT_FRAME_ATOMS_Y",
        "POS_RESTRAINED_ATOMS",
        "EXCLUSION_POT_MOL1_INDEXES",
        "EXCLUSION_POT_MOL2_INDEXES",
    }
    global_scalar_keys = {"LIGAND1_ATTACH_ATOM", "LIGAND2_ATTACH_ATOM"}
    for key in global_list_keys:
        values = options.get(key)
        if values is None:
            continue
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"AToM option {key} must contain an index list")
        for index in values:
            if not isinstance(index, int) or index < 0 or index >= atom_count:
                raise ValueError(
                    f"AToM option {key} contains out-of-bounds index {index!r}"
                )
    for key in global_scalar_keys:
        index = options.get(key)
        if not isinstance(index, int) or index < 0 or index >= atom_count:
            raise ValueError(f"AToM option {key} has out-of-bounds index {index!r}")

    for number, role_key in (("1", "L1"), ("2", "L2")):
        refs = options.get(f"ALIGN_LIGAND{number}_REF_ATOMS")
        if refs is None:
            continue
        role_count = len(options[f"LIGAND{number}_ATOMS"])
        if len(refs) != 3 or any(
            not isinstance(index, int) or index < 0 or index >= role_count
            for index in refs
        ):
            raise ValueError(
                f"ALIGN_LIGAND{number}_REF_ATOMS contains invalid local indices"
            )


def _validate_component_mappings(
    prepared: "PreparedATMSystem",
    atom_count: int,
    mode: Literal["rbfe", "abfe"],
) -> None:
    mapped_atoms: list[int] = []
    for component, indices in prepared.component_atom_indices.items():
        if len(indices) != len(set(indices)):
            raise ValueError(f"Component {component} contains duplicate atom indices")
        if any(index < 0 or index >= atom_count for index in indices):
            raise ValueError(f"Component {component} contains an out-of-bounds atom index")
        mapped_atoms.extend(indices)

    # ABFE has no component corresponding to the one-particle ghost; RBFE L2
    # must be covered by its SmallMoleculeComponent mapping.
    ghost = (
        set(prepared.role_atom_indices["L2"]) if mode == "abfe" else set()
    )
    covered = set(mapped_atoms) | ghost
    if covered != set(range(atom_count)):
        missing = sorted(set(range(atom_count)) - covered)
        raise ValueError(
            "Component/ghost atom mappings do not cover the final topology; "
            f"first missing indices: {missing[:5]}"
        )
    if len(mapped_atoms) != len(set(mapped_atoms)):
        raise ValueError("Component atom mappings overlap")

    mapped_atom_blocks = set(prepared.component_atom_indices.values())
    required_atom_roles = ["receptor", "L1", "solvent"]
    if mode == "rbfe":
        required_atom_roles.append("L2")
    for role in required_atom_roles:
        if tuple(prepared.role_atom_indices[role]) not in mapped_atom_blocks:
            raise ValueError(f"No component atom mapping matches role {role}")

    residue_count = prepared.topology.getNumResidues()
    mapped_residues: list[int] = []
    for component, indices in prepared.component_residue_indices.items():
        if len(indices) != len(set(indices)):
            raise ValueError(
                f"Component {component} contains duplicate residue indices"
            )
        if any(index < 0 or index >= residue_count for index in indices):
            raise ValueError(
                f"Component {component} contains an out-of-bounds residue index"
            )
        mapped_residues.extend(indices)
    ghost_residues = (
        set(prepared.role_residue_indices["L2"])
        if mode == "abfe"
        else set()
    )
    if set(mapped_residues) | ghost_residues != set(range(residue_count)):
        raise ValueError(
            "Component/ghost residue mappings do not cover the final topology"
        )
    if len(mapped_residues) != len(set(mapped_residues)):
        raise ValueError("Component residue mappings overlap")

    mapped_residue_blocks = set(prepared.component_residue_indices.values())
    required_residue_roles = ["receptor", "L1", "solvent"]
    if mode == "rbfe":
        required_residue_roles.append("L2")
    for role in required_residue_roles:
        if tuple(prepared.role_residue_indices[role]) not in mapped_residue_blocks:
            raise ValueError(f"No component residue mapping matches role {role}")


def _validate_solvent_names(prepared: "PreparedATMSystem") -> None:
    residues = list(prepared.topology.residues())
    solvent_indices = prepared.role_residue_indices.get("solvent", ())
    if not solvent_indices:
        raise ValueError("Native AToM setup did not add any solvent residues")
    unsupported = sorted(
        {
            residues[index].name
            for index in solvent_indices
            if residues[index].name.upper() not in _SOLVENT_RESIDUES
        }
    )
    if unsupported:
        raise ValueError(
            "rbfe_structprep does not recognize these generated solvent/ion "
            f"residue names: {unsupported}"
        )


def _box_array(vectors) -> np.ndarray:
    return np.asarray(
        [
            [float(value) for value in vector]
            for vector in vectors.value_in_unit(omm_unit.nanometer)
        ],
        dtype=float,
    )


def _validate_box(
    prepared: "PreparedATMSystem",
    settings: OpenMMSystemGeneratorFFSettings,
) -> dict[str, float]:
    topology_vectors = prepared.topology.getPeriodicBoxVectors()
    if topology_vectors is None:
        raise ValueError("Prepared AToM topology has no periodic box vectors")
    system_vectors = prepared.system.getDefaultPeriodicBoxVectors()
    top_box = _box_array(topology_vectors)
    sys_box = np.asarray(
        [
            [value.value_in_unit(omm_unit.nanometer) for value in vector]
            for vector in system_vectors
        ],
        dtype=float,
    )
    if not np.allclose(top_box, sys_box, atol=1.0e-6, rtol=0.0):
        raise ValueError("Prepared PDB topology and OpenMM System box vectors differ")
    if not prepared.system.usesPeriodicBoundaryConditions():
        raise ValueError("Prepared AToM System does not use periodic boundaries")

    volume = abs(float(np.linalg.det(top_box)))
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Prepared AToM box has zero or non-finite volume")
    heights: list[float] = []
    for i, (j, k) in enumerate(((1, 2), (2, 0), (0, 1))):
        normal = np.cross(top_box[j], top_box[k])
        norm = float(np.linalg.norm(normal))
        if norm == 0:
            raise ValueError("Prepared AToM box vectors are linearly dependent")
        heights.append(abs(float(np.dot(top_box[i], normal / norm))))

    forcefield_cutoff_nm = float(
        settings.nonbonded_cutoff.to(off_unit.nanometer).m
    )
    cutoff_nm = max(forcefield_cutoff_nm, _ATOM_EXCLUSION_CUTOFF_NM)
    minimum_height = min(heights)
    if minimum_height + 1.0e-8 < 2.0 * cutoff_nm:
        raise ValueError(
            "Periodic box violates the OpenMM minimum-image cutoff condition: "
            f"minimum height {minimum_height:.4f} nm, cutoff {cutoff_nm:.4f} nm"
        )

    sizing = np.asarray(prepared.box_sizing_positions_nm, dtype=float)
    minimum_image_distance = _minimum_periodic_image_distance(sizing, top_box)
    if minimum_image_distance + 1.0e-6 < cutoff_nm:
        raise ValueError(
            "Periodic box does not leave a full nonbonded cutoff between the "
            "bound/displaced solute geometry and its images: "
            f"minimum periodic-image separation {minimum_image_distance:.4f} "
            "nm, required "
            f"{cutoff_nm:.4f} nm"
        )

    displacement_nm = 0.1 * np.asarray(
        prepared.atom_options["DISPLACEMENT"], dtype=float
    )
    direct = float(np.linalg.norm(displacement_nm))
    for coefficients in itertools.product((-1, 0, 1), repeat=3):
        if coefficients == (0, 0, 0):
            continue
        translated = displacement_nm + np.asarray(coefficients) @ top_box
        if np.linalg.norm(translated) + 1.0e-6 < direct:
            raise ValueError(
                "AToM displacement is not the minimum-image separation in the "
                "prepared periodic box; enlarge or reshape the box"
            )

    return {
        "minimum_height_nm": minimum_height,
        # Retain the old diagnostics field as an alias.  It represented the
        # intended gap between solute copies, but used an orthorhombic-only
        # face-projection calculation before triclinic boxes were supported.
        "minimum_margin_nm": minimum_image_distance,
        "minimum_image_distance_nm": minimum_image_distance,
        "effective_cutoff_nm": cutoff_nm,
    }


def _minimum_periodic_image_distance(
    positions: np.ndarray,
    box: np.ndarray,
) -> float:
    """Return the closest distance between a point set and a periodic copy.

    A face-projection gap is not a valid image-separation measure for a
    triclinic primitive cell.  In particular, OpenMM's rhombic dodecahedron
    has shorter perpendicular cell heights even though its nearest lattice
    translations have the requested solvent-padding length.

    Candidate lattice translations are bounded using the solute's enclosing
    sphere and the inverse box matrix.  Bounding-sphere lower bounds avoid the
    pairwise calculation for well-padded boxes; ambiguous translations are
    checked exactly in bounded NumPy chunks.
    """

    points = np.asarray(positions, dtype=float)
    lattice = np.asarray(box, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Box image validation requires non-empty Nx3 positions")
    if not np.all(np.isfinite(points)):
        raise ValueError("Box sizing positions contain non-finite coordinates")

    center = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
    radius = float(np.max(np.linalg.norm(points - center, axis=1)))
    diameter_bound = 2.0 * radius

    # The shortest of the immediate lattice translations is an upper bound
    # on the image distance (the same atom occurs in both point sets). If a
    # different translation can produce a smaller distance, its norm cannot
    # exceed that upper bound plus the enclosing-sphere diameter. Convert
    # that translation bound to a lattice-coefficient bound with inverse(box).
    seed_translations = [
        np.asarray(coefficients, dtype=float) @ lattice
        for coefficients in itertools.product((-1, 0, 1), repeat=3)
        if coefficients != (0, 0, 0)
    ]
    seed_upper_bound = min(
        float(np.linalg.norm(translation))
        for translation in seed_translations
    )
    inverse_norm = float(np.linalg.norm(np.linalg.inv(lattice), ord=2))
    coefficient_limit = max(
        1,
        int(
            math.ceil(
                (diameter_bound + seed_upper_bound) * inverse_norm
            )
        )
        + 1,
    )
    translations: list[tuple[float, np.ndarray]] = []
    coefficient_range = range(-coefficient_limit, coefficient_limit + 1)
    for coefficients in itertools.product(coefficient_range, repeat=3):
        if coefficients == (0, 0, 0):
            continue
        # A translation and its negative produce the same set distance.
        first_nonzero = next(value for value in coefficients if value != 0)
        if first_nonzero < 0:
            continue
        translation = np.asarray(coefficients, dtype=float) @ lattice
        translation_norm = float(np.linalg.norm(translation))
        translations.append((translation_norm, translation))

    translations.sort(key=lambda item: item[0])
    minimum = math.inf
    for translation_norm, translation in translations:
        lower_bound = translation_norm - diameter_bound
        if lower_bound >= minimum:
            continue
        minimum = min(
            minimum,
            _minimum_shifted_set_distance(points, translation, minimum),
        )

    if not math.isfinite(minimum):
        raise ValueError("Unable to determine periodic solute-image separation")
    return minimum


def _minimum_shifted_set_distance(
    points: np.ndarray,
    translation: np.ndarray,
    upper_bound: float,
) -> float:
    """Compute the point-set distance to one translated copy in bounded memory."""

    point_count = len(points)
    # Keep the temporary (chunk, point, xyz) array near 24 MB.
    chunk_size = max(1, min(point_count, 1_000_000 // point_count))
    minimum_squared = upper_bound * upper_bound
    for start in range(0, point_count, chunk_size):
        shifted = points[start : start + chunk_size] + translation
        deltas = shifted[:, None, :] - points[None, :, :]
        distances_squared = np.einsum("ijk,ijk->ij", deltas, deltas)
        minimum_squared = min(
            minimum_squared, float(np.min(distances_squared))
        )
    return math.sqrt(minimum_squared)


def _validate_forces(
    prepared: "PreparedATMSystem",
    settings: OpenMMSystemGeneratorFFSettings,
) -> NonbondedForce:
    system = prepared.system
    nonbonded_forces: list[NonbondedForce] = []
    for force in system.getForces():
        class_name = force.__class__.__name__
        if "Barostat" in class_name:
            raise ValueError(
                "Prepared AToM System already contains a barostat; "
                "rbfe_structprep must add the only barostat"
            )
        required_name_fragment = _SUPPORTED_FORCE_CLASSES.get(class_name)
        if required_name_fragment is None:
            custom_hint = " custom" if class_name.startswith("Custom") else ""
            raise ValueError(
                f"Unsupported{custom_hint} force {class_name} ({force.getName()!r}) "
                "in native AToM setup. Only standard harmonic, torsion, and "
                "NonbondedForce terms are currently supported."
            )
        if required_name_fragment not in force.getName():
            raise ValueError(
                f"Force {class_name} is named {force.getName()!r}, which "
                "rbfe_structprep will not move into ATMForce"
            )
        if isinstance(force, NonbondedForce):
            nonbonded_forces.append(force)

    if len(nonbonded_forces) != 1:
        raise ValueError(
            "Native AToM setup requires exactly one standard NonbondedForce; "
            f"found {len(nonbonded_forces)}"
        )
    nonbonded = nonbonded_forces[0]
    if nonbonded.getNumParticles() != system.getNumParticles():
        raise ValueError("NonbondedForce particle count does not match the System")

    method_by_name = {
        "pme": NonbondedForce.PME,
        "ewald": NonbondedForce.Ewald,
        "cutoffperiodic": NonbondedForce.CutoffPeriodic,
    }
    expected_method = method_by_name[settings.nonbonded_method.lower()]
    if nonbonded.getNonbondedMethod() != expected_method:
        raise ValueError(
            "Realized NonbondedForce method does not match forcefield_settings"
        )
    actual_cutoff = nonbonded.getCutoffDistance().value_in_unit(omm_unit.nanometer)
    expected_cutoff = float(settings.nonbonded_cutoff.to(off_unit.nanometer).m)
    if not math.isclose(actual_cutoff, expected_cutoff, abs_tol=1.0e-8, rel_tol=0.0):
        raise ValueError(
            "Realized NonbondedForce cutoff does not match forcefield_settings: "
            f"{actual_cutoff} vs {expected_cutoff} nm"
        )
    return nonbonded


def _validate_realized_ligand_charges(
    prepared: "PreparedATMSystem",
    nonbonded: NonbondedForce,
    mode: Literal["rbfe", "abfe"],
) -> None:
    roles = ("L1", "L2") if mode == "rbfe" else ("L1",)
    for role in roles:
        molecule = prepared.ligand_molecules[role]
        if molecule.partial_charges is None:
            raise ValueError(f"{role} has no authoritative OpenFF partial charges")
        expected = np.asarray(
            molecule.partial_charges.m_as(off_unit.elementary_charge), dtype=float
        )
        indices = prepared.role_atom_indices[role]
        realized = np.asarray(
            [
                nonbonded.getParticleParameters(index)[0].value_in_unit(
                    omm_unit.elementary_charge
                )
                for index in indices
            ],
            dtype=float,
        )
        if expected.shape != realized.shape or not np.allclose(
            expected, realized, atol=1.0e-6, rtol=0.0
        ):
            raise ValueError(
                f"Realized {role} charges differ from the authoritative OpenFE "
                "molecule. The force-field cache may be stale or the one-box "
                "templates may be ambiguous."
            )


def _validate_ghost(
    prepared: "PreparedATMSystem",
    nonbonded: NonbondedForce,
    ghost_mass: float,
) -> None:
    ghost_indices = prepared.role_atom_indices["L2"]
    if len(ghost_indices) != 1:
        raise ValueError("ABFE native setup requires a one-particle L2 ghost")
    ghost = ghost_indices[0]
    mass = prepared.system.getParticleMass(ghost).value_in_unit(omm_unit.dalton)
    if not math.isclose(mass, ghost_mass, abs_tol=1.0e-8, rel_tol=0.0):
        raise ValueError(f"ABFE ghost mass is {mass} Da, expected {ghost_mass} Da")
    charge, sigma, epsilon = nonbonded.getParticleParameters(ghost)
    if abs(charge.value_in_unit(omm_unit.elementary_charge)) > 1.0e-12:
        raise ValueError("ABFE ghost has non-zero charge")
    if abs(epsilon.value_in_unit(omm_unit.kilojoule_per_mole)) > 1.0e-12:
        raise ValueError("ABFE ghost has non-zero Lennard-Jones epsilon")
    if sigma.value_in_unit(omm_unit.nanometer) <= 1.0e-6:
        raise ValueError(
            "ABFE ghost sigma must be positive so AToM does not replace its "
            "zero LJ parameters with a weak interaction"
        )

    for p1, p2 in prepared.topology.bonds():
        if ghost in {p1.index, p2.index}:
            raise ValueError("ABFE ghost is bonded in the final topology")
    for index in range(prepared.system.getNumConstraints()):
        p1, p2, _ = prepared.system.getConstraintParameters(index)
        if ghost in {p1, p2}:
            raise ValueError("ABFE ghost participates in a System constraint")

    for index in range(nonbonded.getNumExceptions()):
        p1, p2, charge_product, _, exception_epsilon = (
            nonbonded.getExceptionParameters(index)
        )
        if ghost not in {p1, p2}:
            continue
        if (
            abs(
                charge_product.value_in_unit(omm_unit.elementary_charge**2)
            )
            > 1.0e-12
            or abs(
                exception_epsilon.value_in_unit(omm_unit.kilojoule_per_mole)
            )
            > 1.0e-12
        ):
            raise ValueError("ABFE ghost has a non-zero nonbonded exception")

    for force in prepared.system.getForces():
        name = force.__class__.__name__
        if name == "HarmonicBondForce":
            for index in range(force.getNumBonds()):
                p1, p2, _, _ = force.getBondParameters(index)
                if ghost in {p1, p2}:
                    raise ValueError("ABFE ghost participates in a bond force")
        elif name == "HarmonicAngleForce":
            for index in range(force.getNumAngles()):
                p1, p2, p3, _, _ = force.getAngleParameters(index)
                if ghost in {p1, p2, p3}:
                    raise ValueError("ABFE ghost participates in an angle force")
        elif name == "PeriodicTorsionForce":
            for index in range(force.getNumTorsions()):
                parameters = force.getTorsionParameters(index)
                if ghost in set(parameters[:4]):
                    raise ValueError("ABFE ghost participates in a torsion force")


def _finite_potential_energy(system, positions) -> float:
    integrator = VerletIntegrator(0.001 * omm_unit.picoseconds)
    try:
        try:
            platform = Platform.getPlatformByName("CPU")
        except Exception:
            platform = Platform.getPlatformByName("Reference")
        context = Context(system, integrator, platform)
        context.setPositions(positions)
        energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            omm_unit.kilojoule_per_mole
        )
        if not np.isfinite(energy):
            raise ValueError(
                f"Prepared AToM System has non-finite potential energy: {energy}"
            )
        return float(energy)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Could not create an OpenMM Context and evaluate the prepared AToM "
            "System; check force-field compatibility and input coordinates"
        ) from exc
    finally:
        if "context" in locals():
            del context
        del integrator


__all__ = ["validate_prepared_system"]
