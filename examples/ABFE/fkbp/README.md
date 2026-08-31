# FKBP ABFE

This is the AToM-OpenMM `examples/ABFE/fkbp` input set wired through
`atom-openfe` and OpenFE/gufe objects.

Copied inputs:

- `receptor/fkbp.pdb`
- `ligands/but.sdf`, `dap.sdf`, `dapp.sdf`, `dmso.sdf`, `dss.sdf`, `prp.sdf`,
  and `thi.sdf`

Run one ligand:

```bash
conda activate openfe-cuda124
export PYTHONPATH=/path/to/atom-openfe/src:/path/to/AToM-OpenMM:${PYTHONPATH}
python examples/ABFE/fkbp/run.py --ligand but
```

Run the full FKBP ligand set:

```bash
python examples/ABFE/fkbp/run.py --ligand all
```

Submit through Slurm:

```bash
cd examples/ABFE/fkbp
sbatch --job-name=atom-fkbp-but --export=ALL,LIGAND=but submit.slurm
```

Submit one job per copied FKBP ligand:

```bash
for ligand in but dap dapp dmso dss prp thi; do
  sbatch --job-name="atom-fkbp-${ligand}" --export=ALL,LIGAND="${ligand}" submit.slurm
done
```

Each job requests one GPU. Results are written under
`runs/fkbp-<ligand>/summary.json` by default.
