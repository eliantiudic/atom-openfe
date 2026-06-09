from __future__ import annotations

import pytest
from gufe import ChemicalSystem, LigandAtomMapping
from gufe import ProteinComponent, SmallMoleculeComponent, SolventComponent
from openff.units import unit
from rdkit import Chem
from rdkit.Chem import AllChem


def _embed(mol, seed=1):
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


@pytest.fixture
def ligand_component():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    _embed(mol, seed=1)
    return SmallMoleculeComponent(mol, name="ligand-a")


@pytest.fixture
def ligand2_component():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCN"))
    _embed(mol, seed=2)
    return SmallMoleculeComponent(mol, name="ligand-b")


@pytest.fixture
def protein_component():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    _embed(mol)
    for index, atom in enumerate(mol.GetAtoms()):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f"{atom.GetSymbol()}{index:<3}"[:4])
        info.SetResidueName("ALA")
        info.SetResidueNumber(1)
        info.SetChainId("A")
        atom.SetMonomerInfo(info)
    return ProteinComponent(mol, name="protein")


@pytest.fixture
def solvent_component():
    return SolventComponent(
        positive_ion="Na",
        negative_ion="Cl",
        ion_concentration=0.0 * unit.molar,
    )


@pytest.fixture
def abfe_state_a(ligand_component, protein_component, solvent_component):
    return ChemicalSystem(
        {
            "protein": protein_component,
            "ligand": ligand_component,
            "solvent": solvent_component,
        },
        name="bound-box",
    )


@pytest.fixture
def abfe_state_b(protein_component, solvent_component):
    return ChemicalSystem(
        {
            "protein": protein_component,
            "solvent": solvent_component,
        },
        name="ghost-endpoint",
    )


@pytest.fixture
def legacy_abfe_state_b(ligand_component, protein_component, solvent_component):
    return ChemicalSystem(
        {
            "protein": protein_component,
            "ligand": ligand_component,
            "solvent": solvent_component,
        },
        name="legacy-displaced-box",
    )


@pytest.fixture
def rbfe_state_a(ligand_component, protein_component, solvent_component):
    return ChemicalSystem(
        {
            "protein": protein_component,
            "ligand": ligand_component,
            "solvent": solvent_component,
        },
        name="ligand-a-box",
    )


@pytest.fixture
def rbfe_state_b(ligand2_component, protein_component, solvent_component):
    return ChemicalSystem(
        {
            "protein": protein_component,
            "ligand": ligand2_component,
            "solvent": solvent_component,
        },
        name="ligand-b-box",
    )


@pytest.fixture
def rbfe_mapping(ligand_component, ligand2_component):
    return LigandAtomMapping(
        componentA=ligand_component,
        componentB=ligand2_component,
        componentA_to_componentB={0: 0, 1: 1, 2: 2},
    )
