from __future__ import annotations

import sys
import types

import pytest
import yaml
from gufe import ChemicalSystem, Transformation
from gufe.protocols.errors import ProtocolValidationError
from gufe.protocols.protocoldag import execute_DAG
from openff.units import unit

from atom_openfe import (
    ATMAbsoluteBindingProtocol,
    ATMAbsoluteBindingSettings,
    ATMAnalysisUnit,
    ATMRunUnit,
    ATMSetupUnit,
    ATMRelativeBindingProtocol,
    ATMRelativeBindingSettings,
    ATMScheduleSettings,
)
from atom_openfe import adapter


def test_default_settings_construct():
    abfe_settings = ATMAbsoluteBindingProtocol.default_settings()
    rbfe_settings = ATMRelativeBindingProtocol.default_settings()

    assert isinstance(abfe_settings, ATMAbsoluteBindingSettings)
    assert isinstance(rbfe_settings, ATMRelativeBindingSettings)
    assert abfe_settings.run.basename == "atom_abfe"
    assert rbfe_settings.run.basename == "atom_rbfe"
    assert abfe_settings.displacement.displacement is None
    assert len(abfe_settings.schedule.lambdas) == 22


def test_protocol_serializes_deserializes():
    protocol = ATMAbsoluteBindingProtocol(
        settings=ATMAbsoluteBindingProtocol.default_settings()
    )

    raw = protocol.to_json()
    roundtrip = ATMAbsoluteBindingProtocol.from_json(content=raw)

    assert roundtrip.settings == protocol.settings


def test_abfe_protocol_creates_setup_run_analysis_dag(abfe_state_a, abfe_state_b):
    protocol = ATMAbsoluteBindingProtocol(
        settings=ATMAbsoluteBindingProtocol.default_settings()
    )

    dag = protocol.create(stateA=abfe_state_a, stateB=abfe_state_b, mapping=None)
    units = dag.protocol_units

    assert [unit.name for unit in units] == ["setup", "run", "analysis"]
    assert isinstance(units[0], ATMSetupUnit)
    assert isinstance(units[1], ATMRunUnit)
    assert isinstance(units[2], ATMAnalysisUnit)
    assert units[1].dependencies == [units[0]]
    assert set(units[2].dependencies) == {units[0], units[1]}


def test_rbfe_protocol_creates_setup_run_analysis_dag(
    rbfe_state_a, rbfe_state_b, rbfe_mapping
):
    protocol = ATMRelativeBindingProtocol(
        settings=ATMRelativeBindingProtocol.default_settings()
    )

    dag = protocol.create(stateA=rbfe_state_a, stateB=rbfe_state_b, mapping=rbfe_mapping)
    units = dag.protocol_units

    assert [unit.name for unit in units] == ["setup", "run", "analysis"]
    assert isinstance(units[0], ATMSetupUnit)
    assert isinstance(units[1], ATMRunUnit)
    assert isinstance(units[2], ATMAnalysisUnit)


def test_transformation_can_create_abfe_protocoldag(abfe_state_a, abfe_state_b):
    protocol = ATMAbsoluteBindingProtocol(
        settings=ATMAbsoluteBindingProtocol.default_settings()
    )
    transformation = Transformation(
        stateA=abfe_state_a,
        stateB=abfe_state_b,
        protocol=protocol,
        mapping=None,
        validate=True,
    )

    dag = transformation.create()

    assert dag.transformation_key == transformation.key
    assert [unit.name for unit in dag.protocol_units] == ["setup", "run", "analysis"]


def test_abfe_validation_rejects_legacy_same_label_state(
    abfe_state_a, legacy_abfe_state_b
):
    protocol = ATMAbsoluteBindingProtocol(
        settings=ATMAbsoluteBindingProtocol.default_settings()
    )

    with pytest.raises(ProtocolValidationError, match="stateB ligand"):
        protocol.create(stateA=abfe_state_a, stateB=legacy_abfe_state_b, mapping=None)


@pytest.mark.parametrize("missing_label", ["ligand", "protein", "solvent"])
def test_abfe_validation_rejects_missing_required_statea_component(
    abfe_state_a, abfe_state_b, missing_label
):
    protocol = ATMAbsoluteBindingProtocol(
        settings=ATMAbsoluteBindingProtocol.default_settings()
    )
    components = dict(abfe_state_a.components)
    components.pop(missing_label)
    invalid_state = ChemicalSystem(components)

    with pytest.raises(ProtocolValidationError, match=f"stateA {missing_label}"):
        protocol.create(stateA=invalid_state, stateB=abfe_state_b, mapping=None)


def test_rbfe_validation_requires_mapping(rbfe_state_a, rbfe_state_b):
    protocol = ATMRelativeBindingProtocol(
        settings=ATMRelativeBindingProtocol.default_settings()
    )

    with pytest.raises(ProtocolValidationError, match="LigandAtomMapping"):
        protocol.create(stateA=rbfe_state_a, stateB=rbfe_state_b, mapping=None)


def test_validation_rejects_inconsistent_schedule():
    with pytest.raises(ValueError, match="Inconsistent AToM alchemical schedule"):
        ATMScheduleSettings(lambda1=[0.0])


def test_mapping_derived_alignment(rbfe_mapping):
    options = adapter.derive_mapping_alignment(rbfe_mapping)

    assert options["ALIGN_LIGAND1_REF_ATOMS"][0] == options["LIGAND1_ATTACH_INDEX"]
    assert options["ALIGN_LIGAND2_REF_ATOMS"][0] == options["LIGAND2_ATTACH_INDEX"]
    assert len(options["ALIGN_LIGAND1_REF_ATOMS"]) == 3
    assert len(options["ALIGN_LIGAND2_REF_ATOMS"]) == 3


def test_abfe_setup_calls_make_system_without_lig2_and_patches_ghost(
    monkeypatch, tmp_path
):
    calls = {"make_system": None, "patch": None}

    make_module = types.ModuleType("atom_openmm.make_atm_system_from_rcpt_lig")
    utils_module = types.ModuleType("atom_openmm.utils.AtomUtils")

    def fake_make_system(**kwargs):
        calls["make_system"] = kwargs

    def fake_patch_system_with_ghost(pdb_file, xml_file, displacement, ghost_mass, attach_index=None):
        calls["patch"] = {
            "pdb_file": pdb_file,
            "xml_file": xml_file,
            "displacement": displacement,
            "ghost_mass": ghost_mass,
            "attach_index": attach_index,
        }

    make_module.make_system = fake_make_system
    utils_module.calc_displ_vec = lambda receptor_file, ligand_file: [22.0, 0.0, 0.0]
    utils_module.patch_system_with_ghost = fake_patch_system_with_ghost
    monkeypatch.setitem(sys.modules, "atom_openmm", types.ModuleType("atom_openmm"))
    monkeypatch.setitem(sys.modules, "atom_openmm.utils", types.ModuleType("atom_openmm.utils"))
    monkeypatch.setitem(
        sys.modules,
        "atom_openmm.make_atm_system_from_rcpt_lig",
        make_module,
    )
    monkeypatch.setitem(sys.modules, "atom_openmm.utils.AtomUtils", utils_module)
    monkeypatch.setattr(adapter, "system_has_ghost_pair", lambda pdb_file: False)
    monkeypatch.setattr(
        adapter,
        "derive_prepared_transfer_options",
        lambda **kwargs: {
            "LIGAND1_ATOMS": [0, 1],
            "LIGAND2_ATOMS": [2],
            "DISPLACEMENT": [22.0, 0.0, 0.0],
        },
    )

    prepared = adapter.prepare_atm_transfer_system(
        mode="abfe",
        options={"BASENAME": "atom_abfe"},
        workdir=tmp_path,
        receptor_file=tmp_path / "receptor.pdb",
        ligand1_file=tmp_path / "ligand1.sdf",
    )

    assert "lig2file" not in calls["make_system"]
    assert calls["patch"]["ghost_mass"] == pytest.approx(12.011)
    assert prepared["atom_options"]["LIGAND2_ATOMS"] == [2]


def test_rbfe_setup_calls_make_system_with_lig2_and_no_ghost_patch(
    monkeypatch, tmp_path
):
    calls = {"make_system": None, "patch": False}

    make_module = types.ModuleType("atom_openmm.make_atm_system_from_rcpt_lig")
    utils_module = types.ModuleType("atom_openmm.utils.AtomUtils")

    def fake_make_system(**kwargs):
        calls["make_system"] = kwargs

    make_module.make_system = fake_make_system
    utils_module.calc_displ_vec = lambda receptor_file, ligand_file: [22.0, 0.0, 0.0]
    utils_module.patch_system_with_ghost = lambda *args, **kwargs: calls.update(
        {"patch": True}
    )
    monkeypatch.setitem(sys.modules, "atom_openmm", types.ModuleType("atom_openmm"))
    monkeypatch.setitem(sys.modules, "atom_openmm.utils", types.ModuleType("atom_openmm.utils"))
    monkeypatch.setitem(
        sys.modules,
        "atom_openmm.make_atm_system_from_rcpt_lig",
        make_module,
    )
    monkeypatch.setitem(sys.modules, "atom_openmm.utils.AtomUtils", utils_module)
    monkeypatch.setattr(
        adapter,
        "derive_prepared_transfer_options",
        lambda **kwargs: {
            "LIGAND1_ATOMS": [0, 1],
            "LIGAND2_ATOMS": [2, 3],
            "DISPLACEMENT": [22.0, 0.0, 0.0],
        },
    )

    adapter.prepare_atm_transfer_system(
        mode="rbfe",
        options={"BASENAME": "atom_rbfe"},
        workdir=tmp_path,
        receptor_file=tmp_path / "receptor.pdb",
        ligand1_file=tmp_path / "ligand1.sdf",
        ligand2_file=tmp_path / "ligand2.sdf",
    )

    assert calls["make_system"]["lig2file"].endswith("ligand2.sdf")
    assert calls["patch"] is False


def test_run_resume_checks(monkeypatch, tmp_path):
    calls = {"structprep": 0, "production": 0}
    structprep_module = types.ModuleType("atom_openmm.rbfe_structprep")
    production_module = types.ModuleType("atom_openmm.rbfe_production")

    def fake_structprep(config_file=None, options=None):
        calls["structprep"] += 1
        (tmp_path / "atom_rbfe_0.xml").write_text("<state />")

    def fake_production(config_file=None, options=None):
        calls["production"] += 1
        for idx in range(2):
            repdir = tmp_path / f"r{idx}"
            repdir.mkdir(exist_ok=True)
            (repdir / "atom_rbfe.out").write_text("1\n2\n3\n")

    structprep_module.rbfe_structprep = fake_structprep
    production_module.rbfe_production = fake_production
    monkeypatch.setitem(sys.modules, "atom_openmm", types.ModuleType("atom_openmm"))
    monkeypatch.setitem(sys.modules, "atom_openmm.rbfe_structprep", structprep_module)
    monkeypatch.setitem(sys.modules, "atom_openmm.rbfe_production", production_module)

    options = {
        "BASENAME": "atom_rbfe",
        "LAMBDAS": [0.0, 1.0],
        "MAX_SAMPLES": 3,
    }

    first = adapter.run_atm_transfer(options=options, workdir=tmp_path)
    second = adapter.run_atm_transfer(options=options, workdir=tmp_path)

    assert first["structprep_ran"] is True
    assert first["production_ran"] is True
    assert second["structprep_ran"] is False
    assert second["production_ran"] is False
    assert calls == {"structprep": 1, "production": 1}


def test_uwham_analysis_defaults_to_first_third_discard(monkeypatch, tmp_path):
    captured = {}
    uwham_module = types.ModuleType("atom_openmm.uwham")

    def fake_calculate(workdir, basename, mintimeid=None, maxtimeid=None):
        captured["mintimeid"] = mintimeid
        return (
            -7.25,
            0.33,
            {
                "dg_leg1": 1.0,
                "dg_stderr_leg1": 0.1,
                "dg_leg2": 8.25,
                "dg_stderr_leg2": 0.2,
                "nsamples": 9,
            },
        )

    uwham_module.calculate_uwham_from_rundir = fake_calculate
    monkeypatch.setitem(sys.modules, "atom_openmm", types.ModuleType("atom_openmm"))
    monkeypatch.setitem(sys.modules, "atom_openmm.uwham", uwham_module)

    for idx in range(2):
        repdir = tmp_path / f"r{idx}"
        repdir.mkdir()
        (repdir / "atom_abfe.out").write_text("\n".join(str(i) for i in range(9)))

    parsed = adapter.analyze_atm_uwham(workdir=tmp_path, basename="atom_abfe")

    assert captured["mintimeid"] == 3
    assert parsed["unit_estimate"] == pytest.approx(-7.25)
    assert parsed["dg_leg1"] == pytest.approx(1.0)
    assert parsed["dg_leg2"] == pytest.approx(8.25)
    assert parsed["n_samples"] == 9


def test_abfe_run_and_analysis_with_mocked_atom_execution(
    monkeypatch, tmp_path, abfe_state_a, abfe_state_b
):
    def fake_prepare_atm_transfer_system(**kwargs):
        assert kwargs["mode"] == "abfe"
        assert kwargs["ligand2_file"] is None
        assert "ALIGN_LIGAND1_REF_ATOMS" not in kwargs["options"]
        workdir = kwargs["workdir"]
        options = dict(kwargs["options"])
        options.update(
            {
                "LIGAND1_ATOMS": list(range(9)),
                "LIGAND2_ATOMS": [99],
                "DISPLACEMENT": [22.0, 0.0, 0.0],
            }
        )
        options_path = workdir / f"{options['BASENAME']}.yaml"
        options_path.write_text(yaml.safe_dump(options))
        return {
            "atom_options": options,
            "atom_options_path": str(options_path),
            "prepared_system_pdb": str(workdir / "atom_abfe.pdb"),
            "prepared_system_xml": str(workdir / "atom_abfe_sys.xml"),
            "diagnostics": {"mocked_setup": True},
        }

    def fake_run_atm_transfer(*, options, workdir, config_file=None):
        assert options["BASENAME"] == "atom_abfe"
        return {
            "status": "mocked",
            "workdir": str(workdir),
            "basename": options["BASENAME"],
            "unit_estimate": -7.25,
            "unit_estimate_error": 0.33,
            "dg_leg1": 1.0,
            "dg_leg2": 8.25,
            "n_samples": 9,
            "diagnostics": {"engine": "mock"},
        }

    monkeypatch.setattr(adapter, "prepare_atm_transfer_system", fake_prepare_atm_transfer_system)
    monkeypatch.setattr(adapter, "run_atm_transfer", fake_run_atm_transfer)

    settings = ATMAbsoluteBindingProtocol.default_settings()
    settings.analysis.run_uwham = False
    protocol = ATMAbsoluteBindingProtocol(settings=settings)
    dag = protocol.create(stateA=abfe_state_a, stateB=abfe_state_b, mapping=None)

    shared = tmp_path / "shared"
    scratch = tmp_path / "scratch"
    shared.mkdir()
    scratch.mkdir()

    dag_result = execute_DAG(
        dag,
        shared_basedir=shared,
        scratch_basedir=scratch,
        keep_shared=True,
        keep_scratch=True,
    )
    result = protocol.gather([dag_result])

    assert dag_result.ok()
    assert result.get_estimate().to(unit.kilocalorie_per_mole).m == pytest.approx(-7.25)
    assert result.get_uncertainty().to(unit.kilocalorie_per_mole).m == pytest.approx(
        0.33
    )
    assert result.get_leg_diagnostics()[0]["dg_leg2"] == pytest.approx(8.25)


def test_rbfe_run_and_analysis_with_mocked_atom_execution(
    monkeypatch, tmp_path, rbfe_state_a, rbfe_state_b, rbfe_mapping
):
    def fake_prepare_atm_transfer_system(**kwargs):
        assert kwargs["mode"] == "rbfe"
        assert kwargs["ligand2_file"] is not None
        assert "ALIGN_LIGAND1_REF_ATOMS" in kwargs["options"]
        workdir = kwargs["workdir"]
        options = dict(kwargs["options"])
        options.update(
            {
                "LIGAND1_ATOMS": list(range(9)),
                "LIGAND2_ATOMS": list(range(9, 18)),
                "DISPLACEMENT": [22.0, 0.0, 0.0],
            }
        )
        options_path = workdir / f"{options['BASENAME']}.yaml"
        options_path.write_text(yaml.safe_dump(options))
        return {
            "atom_options": options,
            "atom_options_path": str(options_path),
            "prepared_system_pdb": str(workdir / "atom_rbfe.pdb"),
            "prepared_system_xml": str(workdir / "atom_rbfe_sys.xml"),
            "diagnostics": {"mocked_setup": True},
        }

    def fake_run_atm_transfer(*, options, workdir, config_file=None):
        return {
            "status": "mocked",
            "workdir": str(workdir),
            "basename": options["BASENAME"],
            "unit_estimate": -1.5,
            "unit_estimate_error": 0.2,
            "dg_leg1": 4.0,
            "dg_leg2": 5.5,
            "n_samples": 11,
            "diagnostics": {"engine": "mock"},
        }

    monkeypatch.setattr(adapter, "prepare_atm_transfer_system", fake_prepare_atm_transfer_system)
    monkeypatch.setattr(adapter, "run_atm_transfer", fake_run_atm_transfer)

    settings = ATMRelativeBindingProtocol.default_settings()
    settings.analysis.run_uwham = False
    protocol = ATMRelativeBindingProtocol(settings=settings)
    dag = protocol.create(stateA=rbfe_state_a, stateB=rbfe_state_b, mapping=rbfe_mapping)

    shared = tmp_path / "shared"
    scratch = tmp_path / "scratch"
    shared.mkdir()
    scratch.mkdir()

    dag_result = execute_DAG(
        dag,
        shared_basedir=shared,
        scratch_basedir=scratch,
        keep_shared=True,
        keep_scratch=True,
    )
    result = protocol.gather([dag_result])

    assert dag_result.ok()
    assert result.get_estimate().to(unit.kilocalorie_per_mole).m == pytest.approx(-1.5)
    assert result.get_uncertainty().to(unit.kilocalorie_per_mole).m == pytest.approx(0.2)
