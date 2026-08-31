# atom-openfe

`atom-openfe` connects `gufe`/OpenFE chemical systems to the AToM-OpenMM
dual-coordinate transfer engine.

The supported workflows are:

- `ATMAbsoluteBindingProtocol`: `stateA` contains protein, ligand, and
  solvent; `stateB` contains the identical protein and solvent without the
  ligand. AToM receives the physical ligand as `L1` and a one-particle ghost
  as `L2`.
- `ATMRelativeBindingProtocol`: `stateA` and `stateB` contain the same protein
  and solvent with ligand 1 and ligand 2, respectively. A single
  `LigandAtomMapping` supplies the correspondence used to derive alignment
  atoms unless explicit local alignment indices are configured.

## Native OpenFE setup

System construction uses OpenFE's `get_system_generator()` compatibility
boundary and `openmmforcefields.SystemGenerator`; it does not call AToM's
legacy `make_system()` wrapper. The native builder returns one consistent
bundle containing the OpenMM topology, positions and System, component
mappings, final AToM indices, and setup diagnostics. It then writes and
reloads the exact files consumed downstream:

- `<BASENAME>.pdb`
- `<BASENAME>_sys.xml`
- `<BASENAME>.yaml`

The reloaded PDB and XML are validated together before setup succeeds.
`rbfe_structprep` remains the downstream stage that constructs the ATM force,
adds its barostat, minimizes, and equilibrates the system. Production and
UWHAM analysis are unchanged.

For RBFE, the builder adds receptor, `L1`, and then `L2`; each ligand is one
contiguous residue with stable local atom order and chains `L` and `M`.
`L2` is displaced before solvent and ions are added, so both physical ligand
locations participate in box construction.

For ABFE, a minimal `L2` residue template is registered with the OpenMM force
field before parameterization. Its single atom is present before solvation and
serialization, has the configured mass, zero charge and Lennard-Jones epsilon,
positive sigma, and no bonded terms. During solvent packing only, a translated
copy of the full `L1` geometry occupies the displaced site. That temporary
geometry is removed before final parameterization, while the correctly sized
box and the native ghost remain.

## Authoritative settings

OpenFE/gufe settings control native construction:

- `forcefield_settings`: protein/solvent XML files, small-molecule force
  field, constraints, rigid water, hydrogen-mass repartitioning, periodic
  nonbonded method, and cutoff.
- `solvation_settings`: water model and exactly one supported box-sizing
  method (padding/shape, solvent count, box size, or box vectors).
- the `SolventComponent`: ion identities, concentration, neutralization, and
  solvent presence. Native OpenMM packing currently requires its solvent
  SMILES to be water (`O`).
- `partial_charge_settings`: the standard OpenFE charge-generation path when
  the component does not already carry charges. Component charges are
  preserved when supplied.
- `thermo_settings.temperature`: the temperature passed to AToM. Pressure is
  currently required to be 1 bar because `rbfe_structprep` owns and adds the
  runtime barostat at 1 bar.
- `setup.forcefield_cache`: the `SystemGenerator` small-molecule template
  cache, or `None` to disable it.
- ABFE-only `system.ghost_mass`: the ghost mass. Relative-binding settings do
  not expose a ghost control.

AToM integration/runtime controls remain under `schedule`, `restraints`,
`softcore`, and `run`. There is deliberately no OpenFE integrator setting:
AToM, not the setup helper, owns runtime integration. The builder suppresses
the temporary `SystemGenerator` barostat so it is never serialized alongside
the one added by `rbfe_structprep`.

`setup.write_component_files` controls only optional receptor/ligand provenance
files and their manifest; native construction always uses the original gufe
components.

## Validation and limitations

Setup rejects a prepared system unless topology/System counts, atom order,
component mappings, `L1`/`L2` blocks and indices, box vectors, cutoff margins,
minimum-image displacement, force inventory, and ligand charges are
consistent. ABFE additionally verifies the ghost in every supported force.
An OpenMM CPU or Reference Context must report finite potential energy both
before and after PDB/XML serialization.

Box checks use the larger of the configured source nonbonded cutoff and the
1.0 nm cutoff of the receptor/`L2` exclusion force that `rbfe_structprep` adds.

Current native setup intentionally supports only periodic PME, Ewald, and
CutoffPeriodic systems containing standard `HarmonicBondForce`,
`HarmonicAngleForce`, `PeriodicTorsionForce`, and one `NonbondedForce` with the
names expected by AToM. Custom, implicit-solvent, Drude, Amoeba, split-LJ, and
other force layouts fail with an actionable error rather than risking an
incomplete ATM force. Generated ions are limited to Na+/K+ and Cl-/F-, whose
residue names are recognized by `rbfe_structprep`.

The endpoint `ChemicalSystem` objects must contain exactly the documented
protein, ligand, and water-solvent components. Additional cofactors or other
component types are rejected because this builder has no topology-placement
contract for them.

Because `SystemGenerator` templates and caches are graph keyed, graph-identical
ligands with conflicting charges are rejected, and realized nonbonded charges
are compared with the authoritative OpenFF molecules to catch stale caches.
Users remain responsible for choosing scientifically appropriate force fields,
charges, displacement, box size, and sampling settings.

Real execution requires AToM-OpenMM plus OpenMM, OpenFF, gufe, and OpenFE in
the same environment.
