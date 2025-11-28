import subprocess
import os
from pathlib import Path

import numpy as np
import Bio.PDB as bp
import biotite.structure.io as bsio

from proteinttt.utils.protein import AA3_TO_AA1


def calculate_tm_score(
    pred_path,
    pdb_path,
    chain_id=None,
    verbose=False,
    tmscore_path=None,
    tmalign_path=None,
):
    """Calculate TM-score between predicted and reference protein structures.

    Uses either TMscore or TMalign executable to compute the TM-score, which measures
    the global structural similarity between two protein structures.

    Args:
        pred_path: Path to predicted structure PDB file
        pdb_path: Path to reference structure PDB file
        chain_id: Chain ID to use (not implemented)
        verbose: Whether to print command and output details
        tmscore_path: Path to TMscore executable
        tmalign_path: Path to TMalign executable

    Returns:
        float: TM-score value between 0 and 1, where 1 indicates perfect structural match

    Raises:
        NotImplementedError: If chain_id is provided
        ValueError: If executable paths are not provided or TM-score not found in output
    """
    if chain_id is not None:
        raise NotImplementedError(
            "Chain ID is not implemented for TM-score calculation."
        )

    num_provided = sum(x is not None for x in (tmscore_path, tmalign_path))
    if num_provided != 1:
        raise ValueError(
            "Exactly one of tmscore_path or tmalign_path must be provided."
        )

    if tmscore_path is not None:
        command = [tmscore_path, pred_path, pdb_path]
    else:
        command = [tmalign_path, pdb_path, pred_path]

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
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
    for line in result.stdout.split("\n"):
        if line.startswith("TM-score"):
            tm_score = float(line.split("=")[1].split()[0])
            return tm_score

    print_cmd()
    raise ValueError("TM-score not found in the output")


def lddt_score(
    pdb_ref,
    pdb_model,
    atom_type="CA",
    cutoff=15.0,
    thresholds=(0.5, 1.0, 2.0, 4.0),
):
    """Compute local distance difference test (lDDT) score between two protein structures.

    The lDDT score measures local distance differences between equivalent atoms in two
    structures, considering both the reference and model structure neighborhoods.

    Args:
        pdb_ref: Path to reference/native PDB file
        pdb_model: Path to model/predicted PDB file
        atom_type: Atom type to use for distance calculations ('CA' or 'CB', default 'CA')
        cutoff: Neighbor distance cutoff in Å (default 15.0)
        thresholds: Distance difference thresholds in Å for lDDT calculation
                   (default (0.5, 1.0, 2.0, 4.0))

    Returns:
        float: Global lDDT score between 0 and 1, where 1 indicates perfect local agreement

    Raises:
        ValueError: If no overlapping residues found between structures
    """

    def get_coords(pdb_path):
        parser = bp.PDBParser(QUIET=True)
        structure = parser.get_structure("s", pdb_path)
        coords = {}
        for chain in structure[0]:
            for res in chain:
                if not bp.is_aa(res, standard=True):
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


def calculate_plddt(pdb_file_path):
    """Calculate mean predicted local distance difference test (pLDDT) from a PDB file.

    pLDDT scores are stored in the B-factor column of PDB files and indicate the
    confidence in local structure prediction, with higher values being better.

    Args:
        pdb_file_path: Path to PDB file containing pLDDT scores in B-factor column

    Returns:
        float: Mean pLDDT score across all residues
    """
    struct = bsio.load_structure(pdb_file_path, extra_fields=["b_factor"])
    pLDDT = float(np.asarray(struct.b_factor, dtype=float).mean())
    return pLDDT


def get_sequence_from_pdb(pdb_path: str) -> str:
    """
    Parse a PDB file and return the amino acid sequence as a one-letter code string.
    Assumes ATOM records only. Asserts there is exactly one chain in the file.

    Sequence is built from CA atoms in order of appearance.

    Args:
        pdb_path: Path to PDB file

    Returns:
        str: Amino acid sequence as a one-letter code string

    Raises:
        FileNotFoundError: If PDB file not found
        AssertionError: If more than one chain is found in the PDB file
        ValueError: If unknown or unsupported residue name is found in the PDB file
    """
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    chains = set()
    residues = []  # (resSeq, iCode, resName)
    seen_residues = set()  # to avoid duplicates

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM  "):
                continue

            chain_id = line[21]
            chains.add(chain_id)

            res_name = line[17:20].strip()
            res_seq = line[22:26].strip()
            i_code = line[26].strip()  # insertion code
            atom_name = line[12:16].strip()

            # Use only CA atoms to define sequence order
            if atom_name != "CA":
                continue

            key = (chain_id, res_seq, i_code)
            if key in seen_residues:
                continue
            seen_residues.add(key)
            residues.append((res_name, res_seq, i_code))

    # Assert only one chain
    if len(chains) == 0:
        raise ValueError(f"No ATOM records found in {pdb_path}")
    if len(chains) != 1:
        raise AssertionError(
            f"Expected exactly one chain, found {len(chains)} in {pdb_path}: {chains}"
        )

    # Convert to one-letter sequence
    seq = []
    for res_name, res_seq, i_code in residues:
        if res_name not in AA3_TO_AA1:
            raise ValueError(
                f"Unknown or unsupported residue name '{res_name}' at {res_seq}{i_code} in {pdb_path}"
            )
        seq.append(AA3_TO_AA1[res_name])

    return "".join(seq)
