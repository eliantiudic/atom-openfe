"""Native OpenFE-style construction of AToM RBFE and ABFE systems."""

from __future__ import annotations

import itertools
import math
import string
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
from gufe import Component, ProteinComponent, SmallMoleculeComponent, SolventComponent
from gufe.settings import OpenMMSystemGeneratorFFSettings, ThermoSettings
from openff.toolkit import Molecule as OFFMolecule
from openff.units import unit as off_unit
from openff.units.openmm import ensure_quantity, to_openmm
from openmm import System
from openmm import unit as omm_unit
from openmm.app import Element, Modeller, Topology

from .openfe_compat import (
    OpenFFPartialChargeSettings,
    OpenMMSolvationSettings,
    assign_offmol_partial_charges,
    get_atm_system_generator,
    preserve_zero_partial_charges,
    validate_solvation_settings,
    without_oechem_backend,
)


TransferMode = Literal["rbfe", "abfe"]


@dataclass(slots=True)
class PreparedATMSystem:
    """One internally consistent native setup result."""

    topology: Topology
    positions: omm_unit.Quantity
    system: System
    component_atom_indices: dict[Component, tuple[int, ...]]
    component_residue_indices: dict[Component, tuple[int, ...]]
    role_atom_indices: dict[str, tuple[int, ...]]
    role_residue_indices: dict[str, tuple[int, ...]]
    atom_options: dict[str, Any]
    diagnostics: dict[str, Any]
    ligand_molecules: dict[str, OFFMolecule]
    box_sizing_positions_nm: np.ndarray


def build_atm_system(
    *,
    mode: TransferMode,
    receptor: ProteinComponent,
    ligand1: SmallMoleculeComponent,
    solvent: SolventComponent,
    forcefield_settings: OpenMMSystemGeneratorFFSettings,
    thermo_settings: ThermoSettings,
    solvation_settings: OpenMMSolvationSettings,
    partial_charge_settings: OpenFFPartialChargeSettings,
    displacement: tuple[float, float, float] | None,
    ghost_mass: float | None,
    cache: Path | None,
    alignment_options: dict[str, Any] | None = None,
    ligand2: SmallMoleculeComponent | None = None,
) -> PreparedATMSystem:
    """Build and validate a solvated AToM transfer system.

    Receptor, L1, and L2 are inserted in that order.  RBFE L2 is translated
    before solvation.  ABFE L2 is a pre-parameterization zero-interaction
    ghost; a temporary translated copy of L1 is present only while solvent is
    packed so the displaced physical ligand geometry determines the cavity and
    box.
    """

    if mode not in {"rbfe", "abfe"}:
        raise ValueError(f"Unsupported AToM transfer mode {mode!r}")
    if mode == "rbfe" and ligand2 is None:
        raise ValueError("RBFE native setup requires ligand2")
    if mode == "abfe" and ligand2 is not None:
        raise ValueError("ABFE native setup uses an L2 ghost, not ligand2")
    if mode == "rbfe" and ghost_mass is not None:
        raise ValueError("RBFE native setup does not use a ghost mass")
    if mode == "abfe" and (
        ghost_mass is None
        or not math.isfinite(ghost_mass)
        or ghost_mass <= 0
    ):
        raise ValueError("ABFE native setup requires a positive ghost mass")
    _validate_supported_ions(solvent)
    _validate_nonbonded_method(forcefield_settings)
    _validate_thermo_settings(thermo_settings)
    validate_solvation_settings(solvation_settings)

    alignment_options = dict(alignment_options or {})
    cache = Path(cache) if cache is not None else None
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)

    ligand_components = {"L1": ligand1}
    if ligand2 is not None:
        ligand_components["L2"] = ligand2

    ligand_molecules: dict[str, OFFMolecule] = {}
    charge_sources: dict[str, str] = {}
    for role, component in ligand_components.items():
        molecule = component.to_openff()
        supplied = _component_has_partial_charges(component)
        if supplied:
            if molecule.partial_charges is None:
                raise ValueError(
                    f"{role} declares component partial charges, but they were not "
                    "preserved by conversion to an OpenFF Molecule"
                )
            charge_sources[role] = "component"
        else:
            assign_offmol_partial_charges(molecule, partial_charge_settings)
            charge_sources[role] = (
                f"openfe:{partial_charge_settings.partial_charge_method}"
            )
        if molecule.partial_charges is None:
            raise ValueError(f"No partial charges are available for {role}")
        ligand_molecules[role] = molecule

    _reject_ambiguous_ligand_templates(ligand_molecules)

    for role, molecule in ligand_molecules.items():
        number = "1" if role == "L1" else "2"
        if (
            alignment_options.get(f"LIGAND{number}_ATTACH_INDEX") is None
            and alignment_options.get(f"ALIGN_LIGAND{number}_REF_ATOMS") is None
        ):
            alignment_options[f"LIGAND{number}_ATTACH_INDEX"] = (
                _select_local_attachment(molecule, {}, role)
            )
    if mode == "abfe":
        alignment_options["LIGAND2_ATTACH_INDEX"] = 0

    with without_oechem_backend():
        system_generator, generator_diagnostics = get_atm_system_generator(
            forcefield_settings=forcefield_settings,
            thermo_settings=thermo_settings,
            cache=cache,
            has_solvent=True,
        )
        generator_diagnostics["zero_charge_vectors_preserved"] = (
            preserve_zero_partial_charges(system_generator)
        )
        system_generator.add_molecules(list(ligand_molecules.values()))

        ghost_diagnostics: dict[str, Any] = {}
        if mode == "abfe":
            ghost_diagnostics = _register_ghost_template(
                system_generator.forcefield, float(ghost_mass)
            )

        modeller = Modeller(Topology(), [])
        receptor_topology = receptor.to_openmm_topology()
        receptor_chain_map = _assign_receptor_chain_ids(receptor_topology)
        modeller.add(receptor_topology, receptor.to_openmm_positions())
        modeller.addExtraParticles(system_generator.forcefield)

        # Match OpenFE's workaround for OpenMM issue #4103: crystal waters
        # must not be mistaken for newly packable solvent during addSolvent.
        crystal_water_count = 0
        for residue in modeller.topology.residues():
            if residue.name == "HOH":
                residue.name = "WAT"
                crystal_water_count += 1

        receptor_atom_count = modeller.topology.getNumAtoms()
        receptor_residue_count = modeller.topology.getNumResidues()

        ligand1_topology = _ligand_topology(ligand_molecules["L1"], "L1", "L")
        ligand1_positions = _offmol_positions(ligand_molecules["L1"])
        modeller.add(ligand1_topology, ligand1_positions)
        ligand1_count = ligand1_topology.getNumAtoms()

        displacement_values = (
            np.asarray(displacement, dtype=float)
            if displacement is not None
            else _automatic_displacement(
                receptor.to_openmm_positions(),
                _offmol_positions(
                    ligand_molecules["L2"]
                    if mode == "rbfe"
                    else ligand_molecules["L1"]
                ),
            )
        )
        if displacement_values.shape != (3,) or not np.all(
            np.isfinite(displacement_values)
        ):
            raise ValueError("AToM displacement must contain three finite values")
        if np.linalg.norm(displacement_values) == 0.0:
            raise ValueError("AToM displacement must be non-zero")
        displacement_nm = 0.1 * displacement_values

        ligand2_count: int
        if mode == "rbfe":
            ligand2_topology = _ligand_topology(
                ligand_molecules["L2"], "L2", "M"
            )
            ligand2_positions = _translated_positions(
                _offmol_positions(ligand_molecules["L2"]), displacement_nm
            )
            modeller.add(ligand2_topology, ligand2_positions)
            ligand2_count = ligand2_topology.getNumAtoms()
        else:
            ligand1_attach_local = _select_local_attachment(
                ligand_molecules["L1"], alignment_options, "L1"
            )
            ligand1_positions_nm = _positions_nm(ligand1_positions)
            ghost_position = (
                ligand1_positions_nm[ligand1_attach_local] + displacement_nm
            )
            ghost_topology = _ghost_topology()
            modeller.add(
                ghost_topology,
                np.asarray([ghost_position], dtype=float) * omm_unit.nanometer,
            )
            ligand2_count = 1

        expected_prefix_atoms = receptor_atom_count + ligand1_count + ligand2_count
        expected_prefix_residues = receptor_residue_count + 2

        prefix_positions_nm = _positions_nm(modeller.positions)[:expected_prefix_atoms]
        sizing_positions_nm = prefix_positions_nm.copy()
        temporary_ligand_residue_index: int | None = None
        if mode == "abfe":
            translated_ligand1 = _translated_positions(
                ligand1_positions, displacement_nm
            )
            temporary_ligand_residue_index = modeller.topology.getNumResidues()
            modeller.add(
                _ligand_topology(ligand_molecules["L1"], "L1", "X"),
                translated_ligand1,
            )
            sizing_positions_nm = np.concatenate(
                [sizing_positions_nm, _positions_nm(translated_ligand1)], axis=0
            )

        modeller.addSolvent(
            system_generator.forcefield,
            **_solvation_kwargs(solvent, solvation_settings),
        )

        for residue in modeller.topology.residues():
            if residue.name == "WAT":
                residue.name = "HOH"

        if temporary_ligand_residue_index is not None:
            temporary_residue = list(modeller.topology.residues())[
                temporary_ligand_residue_index
            ]
            modeller.delete([temporary_residue])

        if modeller.topology.getNumAtoms() < expected_prefix_atoms:
            raise RuntimeError("Solvation unexpectedly removed AToM solute atoms")
        if modeller.topology.getNumResidues() < expected_prefix_residues:
            raise RuntimeError("Solvation unexpectedly removed AToM solute residues")

        topology = modeller.getTopology()
        positions = ensure_quantity(modeller.getPositions(), "openmm")
        if topology.getPeriodicBoxVectors() is None:
            raise ValueError("Native AToM setup requires a periodic solvent box")

        system = system_generator.create_system(
            topology=topology,
            molecules=list(ligand_molecules.values()),
        )

    component_atom_indices, component_residue_indices = _component_mappings(
        receptor=receptor,
        ligand1=ligand1,
        ligand2=ligand2,
        solvent=solvent,
        topology=topology,
        receptor_atom_count=receptor_atom_count,
        receptor_residue_count=receptor_residue_count,
        ligand1_count=ligand1_count,
        ligand2_count=ligand2_count,
    )
    role_atom_indices = {
        "receptor": tuple(range(receptor_atom_count)),
        "L1": tuple(
            range(receptor_atom_count, receptor_atom_count + ligand1_count)
        ),
        "L2": tuple(
            range(
                receptor_atom_count + ligand1_count,
                receptor_atom_count + ligand1_count + ligand2_count,
            )
        ),
        "solvent": tuple(range(expected_prefix_atoms, topology.getNumAtoms())),
    }
    role_residue_indices = {
        "receptor": tuple(range(receptor_residue_count)),
        "L1": (receptor_residue_count,),
        "L2": (receptor_residue_count + 1,),
        "solvent": tuple(
            range(expected_prefix_residues, topology.getNumResidues())
        ),
    }

    atom_options = derive_atm_options(
        mode=mode,
        topology=topology,
        positions=positions,
        role_atom_indices=role_atom_indices,
        alignment_options=alignment_options,
    )

    diagnostics: dict[str, Any] = {
        "builder": "native-openfe-system-generator",
        "mode": mode,
        "component_order": ["receptor", "L1", "L2", "solvent"],
        "requested_displacement_angstrom": displacement_values.tolist(),
        "partial_charge_sources": charge_sources,
        "partial_charges": {
            role: _charges_in_e(molecule).tolist()
            for role, molecule in ligand_molecules.items()
        },
        "receptor_chain_ids": _receptor_chain_ids(topology, receptor_atom_count),
        "receptor_chain_id_changes": receptor_chain_map,
        "receptor_crystal_waters": crystal_water_count,
        "solvent": {
            "model": solvation_settings.solvent_model,
            "positive_ion": solvent.positive_ion,
            "negative_ion": solvent.negative_ion,
            "ion_concentration_molar": float(
                solvent.ion_concentration.to(off_unit.molar).m
            ),
            "neutralize": solvent.neutralize,
            "residue_count": len(role_residue_indices["solvent"]),
        },
        "forcefield": _settings_provenance(forcefield_settings),
        "solvation": _settings_provenance(solvation_settings),
        "force_inventory": [
            {"class": force.__class__.__name__, "name": force.getName()}
            for force in system.getForces()
        ],
        **generator_diagnostics,
        **ghost_diagnostics,
    }

    prepared = PreparedATMSystem(
        topology=topology,
        positions=positions,
        system=system,
        component_atom_indices=component_atom_indices,
        component_residue_indices=component_residue_indices,
        role_atom_indices=role_atom_indices,
        role_residue_indices=role_residue_indices,
        atom_options=atom_options,
        diagnostics=diagnostics,
        ligand_molecules=ligand_molecules,
        box_sizing_positions_nm=sizing_positions_nm,
    )

    from .system_validation import validate_prepared_system

    validation = validate_prepared_system(
        prepared,
        mode=mode,
        forcefield_settings=forcefield_settings,
        ghost_mass=ghost_mass,
        check_energy=True,
    )
    prepared.diagnostics["validation"] = validation
    return prepared


def _component_has_partial_charges(component: SmallMoleculeComponent) -> bool:
    mol = component.to_rdkit()
    return "atom.dprop.PartialCharge" in set(mol.GetPropNames())


def _settings_provenance(settings) -> dict[str, Any]:
    """Convert settings, including array-valued quantities, to plain data."""

    return {
        str(key): _plain_diagnostic_value(value)
        for key, value in settings.model_dump().items()
    }


def _plain_diagnostic_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _plain_diagnostic_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_diagnostic_value(item) for item in value]
    if hasattr(value, "m") and hasattr(value, "units"):
        return {
            "magnitude": _plain_diagnostic_value(value.m),
            "unit": str(value.units),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _charges_in_e(molecule: OFFMolecule) -> np.ndarray:
    if molecule.partial_charges is None:
        raise ValueError(f"Molecule {molecule.name!r} has no partial charges")
    return np.asarray(
        molecule.partial_charges.m_as(off_unit.elementary_charge), dtype=float
    )


def _reject_ambiguous_ligand_templates(
    molecules: dict[str, OFFMolecule],
) -> None:
    if set(molecules) != {"L1", "L2"}:
        return
    import networkx as nx

    def element_graph(molecule: OFFMolecule) -> nx.Graph:
        graph = nx.Graph()
        for index, atom in enumerate(molecule.atoms):
            graph.add_node(index, element=atom.atomic_number)
        for bond in molecule.bonds:
            graph.add_edge(bond.atom1_index, bond.atom2_index)
        return graph

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        element_graph(molecules["L1"]),
        element_graph(molecules["L2"]),
        node_match=lambda left, right: left["element"] == right["element"],
    )
    if not matcher.is_isomorphic():
        return

    # This is the same deliberately weak element/connectivity criterion used
    # by openmmforcefields residue matching.  Once the first template is
    # loaded, either residue may match it even when bond order or stereo differ.
    chemistry_matches, atom_map = OFFMolecule.are_isomorphic(
        molecules["L1"], molecules["L2"], return_atom_map=True
    )
    q1 = _charges_in_e(molecules["L1"])
    q2 = _charges_in_e(molecules["L2"])
    charges_match = chemistry_matches and atom_map is not None and np.allclose(
        q1,
        np.asarray([q2[atom_map[index]] for index in range(len(q1))]),
        atol=1.0e-8,
        rtol=0.0,
    )
    if not chemistry_matches or not charges_match:
        raise ValueError(
            "L1 and L2 have the same element/connectivity graph but differing "
            "chemistry or charges under a possible atom mapping. "
            "openmmforcefields residue matching ignores bond order, "
            "stereochemistry, and residue name, so this one-box combination "
            "cannot be parameterized unambiguously."
        )


def _validate_nonbonded_method(
    forcefield_settings: OpenMMSystemGeneratorFFSettings,
) -> None:
    supported = {"pme", "ewald", "cutoffperiodic"}
    method = forcefield_settings.nonbonded_method.lower()
    if method not in supported:
        raise ValueError(
            "AToM supports native periodic PME, Ewald, or CutoffPeriodic "
            f"construction; got {forcefield_settings.nonbonded_method!r}"
        )


def _validate_thermo_settings(settings: ThermoSettings) -> None:
    if settings.temperature is None:
        raise ValueError("Native AToM setup requires a temperature")
    if settings.pressure is None:
        raise ValueError("Native AToM setup requires a 1 bar pressure setting")
    pressure_bar = float(settings.pressure.to(off_unit.bar).m)
    if not math.isclose(pressure_bar, 1.0, abs_tol=1.0e-8, rel_tol=0.0):
        raise ValueError(
            "rbfe_structprep adds its runtime barostat at 1 bar; "
            "thermo_settings.pressure must be 1 bar"
        )


def _validate_supported_ions(solvent: SolventComponent) -> None:
    if solvent.smiles != "O":
        raise ValueError(
            "OpenMM native solvation currently supports only water "
            f"SolventComponent(smiles='O'); got {solvent.smiles!r}"
        )
    positive = solvent.positive_ion.upper().replace("+", "")
    negative = solvent.negative_ion.upper().replace("-", "")
    if positive not in {"NA", "K"} or negative not in {"CL", "F"}:
        raise ValueError(
            "AToM rbfe_structprep recognizes only Na+/K+ and Cl-/F- among "
            "OpenFE solvent ions; got "
            f"{solvent.positive_ion!r}/{solvent.negative_ion!r}"
        )


def _register_ghost_template(forcefield, ghost_mass: float) -> dict[str, Any]:
    generators = [
        generator
        for generator in forcefield._forces
        if generator.__class__.__name__ == "NonbondedGenerator"
    ]
    if len(generators) != 1:
        raise ValueError(
            "ABFE ghost setup requires exactly one standard OpenMM "
            "NonbondedGenerator; this force-field combination is unsupported"
        )
    if "L2" in forcefield._templates:
        raise ValueError("The selected force fields already define an L2 residue")

    generator = generators[0]
    coulomb14 = float(generator.coulomb14scale)
    lj14 = float(generator.lj14scale)
    atom_type = "atom_openfe_abfe_ghost"
    xml = f"""
<ForceField>
 <AtomTypes>
  <Type name="{atom_type}" class="{atom_type}" element="C" mass="{ghost_mass:.16g}"/>
 </AtomTypes>
 <Residues>
  <Residue name="L2">
   <Atom name="C1" type="{atom_type}" charge="0.0"/>
  </Residue>
 </Residues>
 <NonbondedForce coulomb14scale="{coulomb14:.16g}" lj14scale="{lj14:.16g}">
  <UseAttributeFromResidue name="charge"/>
  <Atom type="{atom_type}" sigma="0.1" epsilon="0.0"/>
 </NonbondedForce>
</ForceField>
"""
    try:
        forcefield.loadFile(StringIO(xml))
    except Exception as exc:
        raise ValueError(
            "The selected force fields cannot safely register the native ABFE "
            "zero-interaction ghost template"
        ) from exc

    ghost_template = forcefield._templates["L2"]

    def match_l2(ff, residue, bonded_to_atom, ignore_external, ignore_extra):
        atoms = list(residue.atoms())
        if (
            residue.name == "L2"
            and len(atoms) == 1
            and atoms[0].element == Element.getBySymbol("C")
        ):
            return ghost_template
        return None

    forcefield.registerTemplateMatcher(match_l2)
    return {
        "ghost_in_native_topology": True,
        "ghost_template": atom_type,
        "ghost_sigma_nm": 0.1,
        "ghost_mass_amu": float(ghost_mass),
    }


def _ghost_topology() -> Topology:
    topology = Topology()
    chain = topology.addChain(id="M")
    residue = topology.addResidue("L2", chain, id="1")
    topology.addAtom("C1", Element.getBySymbol("C"), residue)
    return topology


def _ligand_topology(
    molecule: OFFMolecule,
    residue_name: str,
    chain_id: str,
) -> Topology:
    topology = molecule.to_topology().to_openmm(ensure_unique_atom_names=True)
    residues = list(topology.residues())
    if len(residues) != 1:
        raise ValueError(
            f"AToM requires {residue_name} to be represented by one residue; "
            f"OpenFF produced {len(residues)}"
        )
    residues[0].name = residue_name
    residues[0].id = "1"
    chains = list(topology.chains())
    if len(chains) != 1:
        raise ValueError(f"AToM requires one chain for {residue_name}")
    chains[0].id = chain_id
    names = [atom.name for atom in residues[0].atoms()]
    if len(names) != len(set(names)) or any(not name.strip() for name in names):
        raise ValueError(f"Could not generate stable unique atom names for {residue_name}")
    if topology.getNumAtoms() != molecule.n_atoms:
        raise ValueError(f"OpenFF changed the atom count for {residue_name}")
    return topology


def _offmol_positions(molecule: OFFMolecule) -> omm_unit.Quantity:
    if molecule.n_conformers == 0:
        raise ValueError(f"Molecule {molecule.name!r} has no conformer")
    return ensure_quantity(molecule.conformers[0], "openmm")


def _positions_nm(positions: omm_unit.Quantity) -> np.ndarray:
    return np.asarray(positions.value_in_unit(omm_unit.nanometer), dtype=float)


def _translated_positions(
    positions: omm_unit.Quantity,
    displacement_nm: np.ndarray,
) -> omm_unit.Quantity:
    return (_positions_nm(positions) + displacement_nm) * omm_unit.nanometer


def _automatic_displacement(
    receptor_positions: omm_unit.Quantity,
    ligand_positions: omm_unit.Quantity,
) -> np.ndarray:
    receptor_angstrom = np.asarray(
        receptor_positions.value_in_unit(omm_unit.angstrom), dtype=float
    )
    ligand_angstrom = np.asarray(
        ligand_positions.value_in_unit(omm_unit.angstrom), dtype=float
    )
    if receptor_angstrom.size == 0 or ligand_angstrom.size == 0:
        raise ValueError("Automatic displacement requires receptor and ligand atoms")

    widths = np.ptp(receptor_angstrom, axis=0)
    axis = int(np.argmax(widths))
    target = np.mean(
        [np.min(receptor_angstrom, axis=0), np.max(receptor_angstrom, axis=0)],
        axis=0,
    )
    target[axis] = np.max(receptor_angstrom[:, axis]) + 10.0
    ligand_edge = ligand_angstrom[np.argmin(ligand_angstrom[:, axis])]
    return np.round(target - ligand_edge, 2)


def _assign_receptor_chain_ids(topology: Topology) -> list[dict[str, str]]:
    reserved = {"L", "M", "X"}
    used: set[str] = set()
    changes: list[dict[str, str]] = []
    candidates = (
        "".join(chars)
        for length in itertools.count(1)
        for chars in itertools.product(string.ascii_uppercase, repeat=length)
    )

    def next_id() -> str:
        return next(
            candidate
            for candidate in candidates
            if candidate not in reserved and candidate not in used
        )

    for chain in topology.chains():
        original = (chain.id or "").strip()
        if original and original not in reserved and original not in used:
            chosen = original
        else:
            chosen = next_id()
        chain.id = chosen
        used.add(chosen)
        if chosen != original:
            changes.append({"from": original, "to": chosen})
    return changes


def _solvation_kwargs(
    solvent: SolventComponent,
    settings: OpenMMSolvationSettings,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": settings.solvent_model,
        "positiveIon": solvent.positive_ion,
        "negativeIon": solvent.negative_ion,
        "ionicStrength": to_openmm(solvent.ion_concentration),
        "neutralize": solvent.neutralize,
        "boxShape": settings.box_shape,
    }
    if settings.solvent_padding is not None:
        kwargs["padding"] = to_openmm(settings.solvent_padding)
    if settings.box_size is not None:
        kwargs["boxSize"] = to_openmm(settings.box_size)
    if settings.box_vectors is not None:
        kwargs["boxVectors"] = to_openmm(settings.box_vectors)
    if settings.number_of_solvent_molecules is not None:
        kwargs["numAdded"] = settings.number_of_solvent_molecules
    return kwargs


def _component_mappings(
    *,
    receptor: ProteinComponent,
    ligand1: SmallMoleculeComponent,
    ligand2: SmallMoleculeComponent | None,
    solvent: SolventComponent,
    topology: Topology,
    receptor_atom_count: int,
    receptor_residue_count: int,
    ligand1_count: int,
    ligand2_count: int,
) -> tuple[dict[Component, tuple[int, ...]], dict[Component, tuple[int, ...]]]:
    prefix_atoms = receptor_atom_count + ligand1_count + ligand2_count
    prefix_residues = receptor_residue_count + 2
    atoms: dict[Component, tuple[int, ...]] = {
        receptor: tuple(range(receptor_atom_count)),
        ligand1: tuple(
            range(receptor_atom_count, receptor_atom_count + ligand1_count)
        ),
        solvent: tuple(range(prefix_atoms, topology.getNumAtoms())),
    }
    residues: dict[Component, tuple[int, ...]] = {
        receptor: tuple(range(receptor_residue_count)),
        ligand1: (receptor_residue_count,),
        solvent: tuple(range(prefix_residues, topology.getNumResidues())),
    }
    if ligand2 is not None:
        atoms[ligand2] = tuple(
            range(
                receptor_atom_count + ligand1_count,
                receptor_atom_count + ligand1_count + ligand2_count,
            )
        )
        residues[ligand2] = (receptor_residue_count + 1,)
    return atoms, residues


def _select_local_attachment(
    molecule: OFFMolecule,
    options: dict[str, Any],
    role: Literal["L1", "L2"],
) -> int:
    number = "1" if role == "L1" else "2"
    explicit = options.get(f"LIGAND{number}_ATTACH_INDEX")
    refs = options.get(f"ALIGN_LIGAND{number}_REF_ATOMS")
    if explicit is not None:
        index = int(explicit)
    elif refs is not None:
        if len(refs) != 3:
            raise ValueError(f"{role} alignment requires exactly three local atoms")
        index = int(refs[0])
    else:
        heavy = [atom.molecule_atom_index for atom in molecule.atoms if atom.atomic_number > 1]
        if not heavy:
            raise ValueError(f"Could not select a heavy-atom attachment for {role}")
        positions = _positions_nm(_offmol_positions(molecule))
        centroid = np.mean(positions[heavy], axis=0)
        index = min(heavy, key=lambda i: np.linalg.norm(positions[i] - centroid))
    if index < 0 or index >= molecule.n_atoms:
        raise ValueError(
            f"{role} attachment index {index} is outside its {molecule.n_atoms} atoms"
        )
    return index


def derive_atm_options(
    *,
    mode: TransferMode,
    topology: Topology,
    positions: omm_unit.Quantity,
    role_atom_indices: dict[str, tuple[int, ...]],
    alignment_options: dict[str, Any],
) -> dict[str, Any]:
    from atom_openmm.utils.AtomUtils import get_selected_principal_groups

    residues = list(topology.residues())
    l1_residues = [residue for residue in residues if residue.name == "L1"]
    l2_residues = [residue for residue in residues if residue.name == "L2"]
    if len(l1_residues) != 1 or len(l2_residues) != 1:
        raise ValueError("Native AToM topology must contain exactly one L1 and one L2")
    ligand1_atoms = list(role_atom_indices["L1"])
    ligand2_atoms = list(role_atom_indices["L2"])

    l1_local = _local_attachment_from_topology(
        l1_residues[0], alignment_options, "L1"
    )
    l2_local = _local_attachment_from_topology(
        l2_residues[0], alignment_options, "L2"
    )
    ligand1_attach = ligand1_atoms[l1_local]
    ligand2_attach = ligand2_atoms[l2_local]

    receptor_indices = set(role_atom_indices["receptor"])
    receptor_chains = []
    receptor_ca = []
    for chain in topology.chains():
        chain_indices = [atom.index for atom in chain.atoms()]
        if not any(index in receptor_indices for index in chain_indices):
            continue
        receptor_chains.append(chain.id)
        chain_ca = [
            atom.index
            for atom in chain.atoms()
            if atom.index in receptor_indices and atom.name == "CA"
        ]
        receptor_ca.extend(chain_ca)
    receptor_frame_atoms = receptor_ca
    if len(receptor_frame_atoms) < 3:
        receptor_frame_atoms = [
            atom.index
            for atom in topology.atoms()
            if atom.index in receptor_indices
            and atom.residue.chain.id in receptor_chains
            and atom.element is not None
            and atom.element.atomic_number != 1
        ]
    if len(receptor_frame_atoms) < 3:
        raise ValueError(
            "AToM receptor-frame construction requires at least three receptor "
            "CA or heavy atoms"
        )
    receptor_frame = get_selected_principal_groups(
        topology, positions, receptor_frame_atoms
    )
    if not receptor_frame["z_axis"]["indices"] or not receptor_frame["y_axis"][
        "indices"
    ]:
        raise ValueError("Could not derive non-empty AToM receptor frame groups")

    pos = positions
    displacement = np.asarray(
        (pos[ligand2_attach] - pos[ligand1_attach]).value_in_unit(
            omm_unit.angstrom
        ),
        dtype=float,
    )
    receptor_com = np.asarray(
        receptor_frame["origin"]["com"], dtype=float
    ) * omm_unit.nanometer
    offset = np.asarray(
        (pos[ligand1_attach] - receptor_com).value_in_unit(omm_unit.angstrom),
        dtype=float,
    )

    heavy_receptor = [
        atom.index
        for atom in topology.atoms()
        if atom.index in receptor_indices
        and atom.residue.chain.id in receptor_chains
        and atom.element is not None
        and atom.element.atomic_number != 1
    ]
    heavy_l2 = [
        atom.index
        for atom in l2_residues[0].atoms()
        if atom.element is not None and atom.element.atomic_number != 1
    ]

    options: dict[str, Any] = {
        "RCPT_CHAIN_NAMES": receptor_chains,
        "LIGAND1_ATOMS": ligand1_atoms,
        "LIGAND2_ATOMS": ligand2_atoms,
        "LIGAND1_VAR_ATOMS": ligand1_atoms,
        "LIGAND2_VAR_ATOMS": ligand2_atoms,
        "LIGAND1_ATTACH_ATOM": ligand1_attach,
        "LIGAND2_ATTACH_ATOM": ligand2_attach,
        "LIGAND1_CM_ATOMS": [ligand1_attach],
        "LIGAND2_CM_ATOMS": [ligand2_attach],
        "DISPLACEMENT": displacement.tolist(),
        "RCPT_CM_ATOMS": list(receptor_frame["origin"]["indices"]),
        "RCPT_FRAME_ATOMS_O": list(receptor_frame["origin"]["indices"]),
        "RCPT_FRAME_ATOMS_Z": list(receptor_frame["z_axis"]["indices"]),
        "RCPT_FRAME_ATOMS_Y": list(receptor_frame["y_axis"]["indices"]),
        "LIGOFFSET": offset.tolist(),
        "POS_RESTRAINED_ATOMS": (
            None if mode == "abfe" else list(receptor_frame["origin"]["indices"])
        ),
        "EXCLUSION_POT_MOL1_INDEXES": heavy_receptor,
        "EXCLUSION_POT_MOL2_INDEXES": heavy_l2,
    }
    if alignment_options.get("ALIGN_LIGAND1_REF_ATOMS") is not None:
        for role in ("L1", "L2"):
            key = f"ALIGN_LIGAND{'1' if role == 'L1' else '2'}_REF_ATOMS"
            refs = [int(index) for index in alignment_options[key]]
            atom_count = len(role_atom_indices[role])
            if len(refs) != 3 or any(index < 0 or index >= atom_count for index in refs):
                raise ValueError(f"{key} contains invalid component-local indices")
            options[key] = refs
    return options


def _local_attachment_from_topology(
    residue,
    options: dict[str, Any],
    role: Literal["L1", "L2"],
) -> int:
    atoms = list(residue.atoms())
    number = "1" if role == "L1" else "2"
    explicit = options.get(f"LIGAND{number}_ATTACH_INDEX")
    refs = options.get(f"ALIGN_LIGAND{number}_REF_ATOMS")
    if explicit is not None:
        index = int(explicit)
    elif refs is not None:
        index = int(refs[0])
    else:
        heavy = [
            local
            for local, atom in enumerate(atoms)
            if atom.element is not None and atom.element.atomic_number != 1
        ]
        if not heavy:
            raise ValueError(f"Could not select a heavy-atom attachment for {role}")
        index = heavy[0] if len(heavy) == 1 else heavy[len(heavy) // 2]
    if index < 0 or index >= len(atoms):
        raise ValueError(f"{role} attachment index {index} is outside its residue")
    return index


def _receptor_chain_ids(topology: Topology, receptor_atom_count: int) -> list[str]:
    ids: list[str] = []
    for chain in topology.chains():
        if any(atom.index < receptor_atom_count for atom in chain.atoms()):
            ids.append(chain.id)
    return ids


__all__ = ["PreparedATMSystem", "build_atm_system", "derive_atm_options"]
