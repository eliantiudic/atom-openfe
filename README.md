# atom-openfe

Integration layer between `gufe`/OpenFE objects and the AToM-OpenMM
dual-coordinate transfer engine.

The first supported workflows are:

- `ATMAbsoluteBindingProtocol`: standard ABFE endpoint semantics, where
  `stateA` contains protein + ligand + solvent and `stateB` contains protein +
  solvent. Internally, setup follows the FKBP AToM workflow by preparing an
  RBFE-style `L1`/`L2` system where `L2` is a one-particle ghost ligand.
- `ATMRelativeBindingProtocol`: small-molecule RBFE endpoint semantics, where
  `stateA` contains ligand 1 and `stateB` contains ligand 2. RBFE requires a
  single `LigandAtomMapping`; alignment atoms are derived from the mapping
  unless explicit alignment settings are supplied.

Both protocols prepare AToM work directories from OpenFE components, run
`rbfe_structprep`, run `rbfe_production`, and analyze the completed run with
AToM UWHAM. Real execution requires an environment with AToM-OpenMM, OpenMM,
OpenFF, gufe, and OpenFE installed.
