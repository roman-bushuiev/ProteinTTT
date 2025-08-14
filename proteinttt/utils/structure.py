import subprocess
import subprocess
import torch.nn.functional as F
import numpy as np
import pandas as pd
import esm
import biotite.structure.io as bsio
from pathlib import Path
import Bio.PDB as bp
from Bio.PDB import PDBParser, is_aa


def calculate_tm_score(
    pred_path, pdb_path, chain_id=None, use_tmalign=False, verbose=False,
    tmscore_path="/scratch/project/open-32-14/pimenol1/ProteinTTT/ProteinTTT/TMalign",
    tmalign_path="/scratch/project/open-32-14/pimenol1/ProteinTTT/ProteinTTT/TMalign.cpp"
):

    if chain_id is not None:
        raise NotImplementedError("Chain ID is not implemented for TM-score calculation.")

    # Run TMscore and capture the output
    command = [tmalign_path, pdb_path, pred_path] if use_tmalign else [tmscore_path, pred_path, pdb_path]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    def print_cmd():
        print("TMscore command:")
        print(result.args)
        print("TMscore output:")
        print(result.stdout)
        print("TMscore error:")
        print(result.stderr)
    if verbose:
        print_cmd()

    # Extract TM-score from the output
    for line in result.stdout.split('\n'):
        if line.startswith("TM-score"):
            tm_score = float(line.split('=')[1].split()[0])
            return tm_score

    print_cmd()
    raise ValueError("TM-score not found in the output")


def lddt_score(pdb_ref, pdb_model, atom_type="CA", cutoff=15.0, thresholds=(0.5, 1.0, 2.0, 4.0)):
    """
    Compute CA/CB-based lDDT between two protein structures.
    
    Parameters
    ----------
    pdb_ref : str
        Path to reference/native PDB file
    pdb_model : str
        Path to model/predicted PDB file
    atom_type : str
        'CA' (default) or 'CB' (GLY falls back to CA)
    cutoff : float
        Neighbor distance cutoff in Å (default 15.0)
    thresholds : tuple of float
        lDDT thresholds in Å (default (0.5, 1.0, 2.0, 4.0))
    
    Returns
    -------
    float
        Global lDDT score
    """

    def get_coords(pdb_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("s", pdb_path)
        coords = {}
        for chain in structure[0]:
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                key = (chain.id, res.id[1], res.id[2].strip() or "")
                at = atom_type
                if at == "CB" and res.get_resname().upper() == "GLY":
                    at = "CA"
                if res.has_id(at):
                    coords[key] = res[at].get_coord()
                elif res.has_id("CA"):
                    coords[key] = res["CA"].get_coord()
        return coords

    ref_coords = get_coords(pdb_ref)
    mdl_coords = get_coords(pdb_model)
    common_keys = sorted(set(ref_coords) & set(mdl_coords))
    if not common_keys:
        raise ValueError("No overlapping residues found.")

    ref_arr = np.vstack([ref_coords[k] for k in common_keys])
    mdl_arr = np.vstack([mdl_coords[k] for k in common_keys])

    D_ref = np.linalg.norm(ref_arr[:, None, :] - ref_arr[None, :, :], axis=-1)
    D_mdl = np.linalg.norm(mdl_arr[:, None, :] - mdl_arr[None, :, :], axis=-1)
    neighbor_mask = (D_ref <= cutoff) & (~np.eye(len(ref_arr), dtype=bool))

    per_res = []
    for i in range(len(ref_arr)):
        nbrs = np.where(neighbor_mask[i])[0]
        if len(nbrs) == 0:
            continue
        delta = np.abs(D_ref[i, nbrs] - D_mdl[i, nbrs])
        score = np.mean([(delta < t).mean() for t in thresholds])
        per_res.append(score)

    return float(np.mean(per_res)) if per_res else float("nan")
