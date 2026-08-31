from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gufe import ProteinComponent, SmallMoleculeComponent, SolventComponent
from openff.units import unit
from openmm import MonteCarloBarostat, NonbondedForce, XmlSerializer
from openmm import unit as omm_unit
from openmm.app import PDBFile

from atom_openfe import adapter
from atom_openfe.settings import ATMRelativeBindingSettings
from atom_openfe.system_builder import build_atm_system
from atom_openfe.system_validation import (
    _minimum_periodic_image_distance,
    validate_prepared_system,
)


DATA = Path(__file__).parent / "data"
DISPLACEMENT_ANGSTROM = np.array([6.0, 0.0, 0.0])
GHOST_MASS_DALTON = 13.5


@pytest.fixture(scope="session")
def native_components():
    receptor = ProteinComponent.from_pdb_file(DATA / "GLY_capped.pdb")
    ligand1 = SmallMoleculeComponent.from_sdf_file(DATA / "chlorobenzene.sdf")
    ligand2 = SmallMoleculeComponent.from_sdf_file(DATA / "fluorobenzene.sdf")

    zero_mol = ligand1.to_rdkit()
    zero_values = " ".join("0.0" for _ in range(zero_mol.GetNumAtoms()))
    zero_mol.SetProp("atom.dprop.PartialCharge", zero_values)
    for atom in zero_mol.GetAtoms():
        atom.SetDoubleProp("PartialCharge", 0.0)
    with pytest.warns(UserWarning, match="all equal to zero"):
        zero_ligand = SmallMoleculeComponent(zero_mol)

    solvent = SolventComponent(
        positive_ion="Na+",
        negative_ion="Cl-",
        ion_concentration=0.4 * unit.molar,
        neutralize=True,
    )
    return {
        "receptor": receptor,
        "ligand1": ligand1,
        "ligand2": ligand2,
        "zero_ligand": zero_ligand,
        "solvent": solvent,
    }


@pytest.fixture(scope="session")
def native_settings():
    settings = ATMRelativeBindingSettings()
    settings.forcefield_settings.forcefields = [
        "amber/ff14SB.xml",
        "amber/tip3p_standard.xml",
        "amber/tip3p_HFE_multivalent.xml",
    ]
    settings.forcefield_settings.nonbonded_cutoff = 0.5 * unit.nanometer
    settings.forcefield_settings.hydrogen_mass = 2.0
    settings.solvation_settings.solvent_padding = None
    settings.solvation_settings.box_shape = None
    settings.solvation_settings.box_size = (
        np.array([2.4, 2.4, 2.1]) * unit.nanometer
    )
    return settings


@pytest.fixture(scope="session")
def native_abfe_system(native_components, native_settings, tmp_path_factory):
    return build_atm_system(
        mode="abfe",
        receptor=native_components["receptor"],
        ligand1=native_components["zero_ligand"],
        ligand2=None,
        solvent=native_components["solvent"],
        forcefield_settings=native_settings.forcefield_settings,
        thermo_settings=native_settings.thermo_settings,
        solvation_settings=native_settings.solvation_settings,
        partial_charge_settings=native_settings.partial_charge_settings,
        displacement=tuple(DISPLACEMENT_ANGSTROM),
        ghost_mass=GHOST_MASS_DALTON,
        cache=tmp_path_factory.mktemp("abfe-cache") / "ff.json",
        alignment_options={"LIGAND1_ATTACH_INDEX": 1},
    )


@pytest.fixture(scope="session")
def native_rbfe_system(native_components, native_settings, tmp_path_factory):
    return build_atm_system(
        mode="rbfe",
        receptor=native_components["receptor"],
        ligand1=native_components["ligand1"],
        ligand2=native_components["ligand2"],
        solvent=native_components["solvent"],
        forcefield_settings=native_settings.forcefield_settings,
        thermo_settings=native_settings.thermo_settings,
        solvation_settings=native_settings.solvation_settings,
        partial_charge_settings=native_settings.partial_charge_settings,
        displacement=tuple(DISPLACEMENT_ANGSTROM),
        ghost_mass=None,
        cache=tmp_path_factory.mktemp("rbfe-cache") / "ff.json",
        alignment_options={
            "ALIGN_LIGAND1_REF_ATOMS": [1, 2, 3],
            "ALIGN_LIGAND2_REF_ATOMS": [1, 2, 3],
            "LIGAND1_ATTACH_INDEX": 1,
            "LIGAND2_ATTACH_INDEX": 1,
        },
    )


def _nonbonded_force(system) -> NonbondedForce:
    forces = [
        force for force in system.getForces() if isinstance(force, NonbondedForce)
    ]
    assert len(forces) == 1
    return forces[0]


def _positions_nm(prepared) -> np.ndarray:
    return np.asarray(
        prepared.positions.value_in_unit(omm_unit.nanometer), dtype=float
    )


def _box_nm(prepared) -> np.ndarray:
    vectors = prepared.topology.getPeriodicBoxVectors()
    return np.asarray(
        [
            [value.value_in_unit(omm_unit.nanometer) for value in vector]
            for vector in vectors
        ],
        dtype=float,
    )


def test_rbfe_native_order_mappings_and_displacement(
    native_rbfe_system, native_components
):
    prepared = native_rbfe_system
    l1 = prepared.role_atom_indices["L1"]
    l2 = prepared.role_atom_indices["L2"]

    assert l1 == tuple(range(l1[0], l1[0] + 12))
    assert l2 == tuple(range(l2[0], l2[0] + 12))
    assert l2[0] == l1[-1] + 1
    assert prepared.atom_options["LIGAND1_ATOMS"] == list(l1)
    assert prepared.atom_options["LIGAND2_ATOMS"] == list(l2)
    assert prepared.component_atom_indices[native_components["ligand1"]] == l1
    assert prepared.component_atom_indices[native_components["ligand2"]] == l2
    assert (
        prepared.component_atom_indices[native_components["solvent"]]
        == prepared.role_atom_indices["solvent"]
    )

    ligand_residues = [
        residue
        for residue in prepared.topology.residues()
        if residue.name in {"L1", "L2"}
    ]
    assert [(residue.name, residue.chain.id) for residue in ligand_residues] == [
        ("L1", "L"),
        ("L2", "M"),
    ]

    positions = _positions_nm(prepared)
    np.testing.assert_allclose(
        positions[l2[0] + 1] - positions[l1[0] + 1],
        0.1 * DISPLACEMENT_ANGSTROM,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        prepared.atom_options["DISPLACEMENT"],
        DISPLACEMENT_ANGSTROM,
        atol=1.0e-6,
    )


def test_abfe_native_ghost_and_explicit_zero_ligand_charges(native_abfe_system):
    prepared = native_abfe_system
    l1 = prepared.role_atom_indices["L1"]
    ghost_indices = prepared.role_atom_indices["L2"]
    assert len(ghost_indices) == 1
    ghost = ghost_indices[0]

    nonbonded = _nonbonded_force(prepared.system)
    charge, sigma, epsilon = nonbonded.getParticleParameters(ghost)
    assert prepared.system.getParticleMass(ghost).value_in_unit(
        omm_unit.dalton
    ) == pytest.approx(GHOST_MASS_DALTON)
    assert charge.value_in_unit(omm_unit.elementary_charge) == pytest.approx(0.0)
    assert epsilon.value_in_unit(omm_unit.kilojoule_per_mole) == pytest.approx(0.0)
    assert sigma.value_in_unit(omm_unit.nanometer) > 0.0

    realized_l1_charges = [
        nonbonded.getParticleParameters(index)[0].value_in_unit(
            omm_unit.elementary_charge
        )
        for index in prepared.role_atom_indices["L1"]
    ]
    np.testing.assert_allclose(realized_l1_charges, 0.0, atol=1.0e-12)
    assert prepared.diagnostics["partial_charge_sources"]["L1"] == "component"

    positions = _positions_nm(prepared)
    l1_attach = prepared.atom_options["LIGAND1_ATTACH_ATOM"]
    np.testing.assert_allclose(
        positions[ghost] - positions[l1_attach],
        0.1 * DISPLACEMENT_ANGSTROM,
        atol=1.0e-6,
    )
    assert prepared.diagnostics["ghost_in_native_topology"] is True
    assert prepared.box_sizing_positions_nm.shape[0] == ghost + 1 + len(l1)
    np.testing.assert_allclose(
        prepared.box_sizing_positions_nm[-len(l1) :],
        positions[list(l1)] + 0.1 * DISPLACEMENT_ANGSTROM,
        atol=1.0e-4,
    )


def test_forcefield_hmr_solvation_ion_and_box_settings_are_realized(
    native_rbfe_system, native_settings
):
    prepared = native_rbfe_system
    nonbonded = _nonbonded_force(prepared.system)
    assert nonbonded.getNonbondedMethod() == NonbondedForce.PME
    assert nonbonded.getCutoffDistance().value_in_unit(
        omm_unit.nanometer
    ) == pytest.approx(0.5)
    assert not any(
        "Barostat" in force.__class__.__name__
        for force in prepared.system.getForces()
    )

    receptor_and_ligand = set(prepared.role_atom_indices["receptor"])
    receptor_and_ligand.update(prepared.role_atom_indices["L1"])
    receptor_and_ligand.update(prepared.role_atom_indices["L2"])
    solute_hydrogens = [
        atom.index
        for atom in prepared.topology.atoms()
        if atom.index in receptor_and_ligand
        and atom.element is not None
        and atom.element.atomic_number == 1
    ]
    assert solute_hydrogens
    assert all(
        prepared.system.getParticleMass(index).value_in_unit(omm_unit.dalton)
        == pytest.approx(native_settings.forcefield_settings.hydrogen_mass)
        for index in solute_hydrogens
    )

    solvent_names = {
        list(prepared.topology.residues())[index].name
        for index in prepared.role_residue_indices["solvent"]
    }
    assert {"HOH", "NA", "CL"} <= solvent_names
    assert prepared.diagnostics["solvent"]["ion_concentration_molar"] == pytest.approx(
        0.4
    )
    assert prepared.diagnostics["solvent"]["neutralize"] is True
    np.testing.assert_allclose(_box_nm(prepared), np.diag([2.4, 2.4, 2.1]))
    assert prepared.diagnostics["forcefield"]["forcefields"] == (
        native_settings.forcefield_settings.forcefields
    )
    assert prepared.diagnostics["forcefield"]["small_molecule_forcefield"] == (
        native_settings.forcefield_settings.small_molecule_forcefield
    )
    assert prepared.diagnostics["validation"]["box_effective_cutoff_nm"] == (
        pytest.approx(1.0)
    )
    assert prepared.diagnostics["validation"]["finite_potential_energy"] is True


def test_default_dodecahedron_validates_displaced_abfe_geometry(
    native_components, native_settings, tmp_path
):
    settings = native_settings.model_copy(deep=True)
    settings.solvation_settings.box_size = None
    settings.solvation_settings.solvent_padding = 1.5 * unit.nanometer
    settings.solvation_settings.box_shape = "dodecahedron"

    prepared = build_atm_system(
        mode="abfe",
        receptor=native_components["receptor"],
        ligand1=native_components["zero_ligand"],
        ligand2=None,
        solvent=native_components["solvent"],
        forcefield_settings=settings.forcefield_settings,
        thermo_settings=settings.thermo_settings,
        solvation_settings=settings.solvation_settings,
        partial_charge_settings=settings.partial_charge_settings,
        displacement=(0.0, 0.0, 35.0),
        ghost_mass=GHOST_MASS_DALTON,
        cache=tmp_path / "dodecahedron-ff.json",
        alignment_options={"LIGAND1_ATTACH_INDEX": 1},
    )

    box = _box_nm(prepared)
    assert box[2, 0] != pytest.approx(0.0)
    assert box[2, 1] != pytest.approx(0.0)
    validation = prepared.diagnostics["validation"]
    assert validation["box_minimum_image_distance_nm"] >= 1.0 - 1.0e-6
    assert validation["box_minimum_margin_nm"] == pytest.approx(
        validation["box_minimum_image_distance_nm"]
    )


def test_triclinic_image_distance_detects_an_unsafe_copy():
    width = 3.0
    dodecahedron = width * np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5 * np.sqrt(2.0)],
        ]
    )
    positions = np.asarray([[0.0, 0.0, 0.0], [2.4, 0.0, 0.0]])

    assert _minimum_periodic_image_distance(
        positions, dodecahedron
    ) == pytest.approx(0.6)


def test_adapter_pdb_xml_roundtrip_preserves_native_contract(
    monkeypatch,
    native_abfe_system,
    native_components,
    native_settings,
    tmp_path,
):
    from atom_openfe import system_builder

    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return native_abfe_system

    monkeypatch.setattr(system_builder, "build_atm_system", fake_build)
    result = adapter.prepare_atm_transfer_system(
        mode="abfe",
        options={"BASENAME": "native_abfe"},
        workdir=tmp_path,
        receptor=native_components["receptor"],
        ligand1=native_components["zero_ligand"],
        ligand2=None,
        solvent=native_components["solvent"],
        forcefield_settings=native_settings.forcefield_settings,
        thermo_settings=native_settings.thermo_settings,
        solvation_settings=native_settings.solvation_settings,
        partial_charge_settings=native_settings.partial_charge_settings,
        ghost_mass=GHOST_MASS_DALTON,
        forcefield_cache=None,
    )

    assert captured["mode"] == "abfe"
    assert captured["receptor"] == native_components["receptor"]
    assert captured["ligand1"] == native_components["zero_ligand"]
    assert captured["ligand2"] is None
    assert captured["ghost_mass"] == pytest.approx(GHOST_MASS_DALTON)
    roundtrip = result["diagnostics"]["roundtrip"]
    assert roundtrip["pdb_xml_validated"] is True
    assert roundtrip["atom_signature_preserved"] is True
    assert roundtrip["validation"]["finite_potential_energy"] is True

    pdb_path = Path(result["prepared_system_pdb"])
    xml_path = Path(result["prepared_system_xml"])
    assert Path(result["atom_options_path"]).is_file()
    pdb = PDBFile(str(pdb_path))
    residues = [
        (residue.name, residue.chain.id) for residue in pdb.topology.residues()
    ]
    assert ("L1", "L") in residues
    assert ("L2", "M") in residues
    system = XmlSerializer.deserialize(xml_path.read_text())
    assert system.getNumParticles() == pdb.topology.getNumAtoms()
    system_box = np.asarray(
        [
            [value.value_in_unit(omm_unit.nanometer) for value in vector]
            for vector in system.getDefaultPeriodicBoxVectors()
        ]
    )
    pdb_box = np.asarray(
        [
            [value.value_in_unit(omm_unit.nanometer) for value in vector]
            for vector in pdb.topology.getPeriodicBoxVectors()
        ]
    )
    np.testing.assert_allclose(system_box, pdb_box, atol=1.0e-8)
    ghost = native_abfe_system.role_atom_indices["L2"][0]
    assert system.getParticleMass(ghost).value_in_unit(
        omm_unit.dalton
    ) == pytest.approx(GHOST_MASS_DALTON)


def test_validator_rejects_out_of_bounds_atom_option(
    native_rbfe_system, native_settings
):
    options = dict(native_rbfe_system.atom_options)
    options["LIGAND2_ATTACH_ATOM"] = native_rbfe_system.system.getNumParticles()
    invalid = replace(native_rbfe_system, atom_options=options)

    with pytest.raises(ValueError, match="out-of-bounds"):
        validate_prepared_system(
            invalid,
            mode="rbfe",
            forcefield_settings=native_settings.forcefield_settings,
            ghost_mass=None,
            check_energy=False,
        )


def test_validator_rejects_preexisting_barostat(native_rbfe_system, native_settings):
    system = XmlSerializer.deserialize(
        XmlSerializer.serialize(native_rbfe_system.system)
    )
    system.addForce(
        MonteCarloBarostat(1.0 * omm_unit.bar, 300.0 * omm_unit.kelvin)
    )
    invalid = replace(native_rbfe_system, system=system)

    with pytest.raises(ValueError, match="already contains a barostat"):
        validate_prepared_system(
            invalid,
            mode="rbfe",
            forcefield_settings=native_settings.forcefield_settings,
            ghost_mass=None,
            check_energy=False,
        )


def test_real_rbfe_structprep_smoke(native_rbfe_system, native_settings, tmp_path):
    from atom_openmm.rbfe_structprep import rbfe_structprep

    basename = "native_rbfe"
    adapter._serialize_prepared_system(
        prepared=native_rbfe_system,
        mode="rbfe",
        pdb_path=tmp_path / f"{basename}.pdb",
        xml_path=tmp_path / f"{basename}_sys.xml",
        forcefield_settings=native_settings.forcefield_settings,
        ghost_mass=None,
    )
    options = native_settings.to_atom_options()
    options.update(native_rbfe_system.atom_options)
    options.update(
        {
            "BASENAME": basename,
            "WORKDIR": str(tmp_path),
            "OPENMM_PLATFORM": "CPU",
            "THERMALIZATION_STEPS": 1,
            "ANNEALING_STEPS": 1,
            "EQUILIBRATION_STEPS": 1,
            "STEPS_PER_CYCLE": 1,
        }
    )

    adapter._call_in_workdir(
        tmp_path, rbfe_structprep, config_file=None, options=options
    )

    assert (tmp_path / f"{basename}_min.xml").is_file()
    assert (tmp_path / f"{basename}_equil.xml").is_file()
    assert (tmp_path / f"{basename}_mdlambda.xml").is_file()
    assert (tmp_path / f"{basename}_0.xml").is_file()
