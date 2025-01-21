from pathlib import Path
from Bio import SeqIO
from typing import Optional


def process_msa_seq(seq: str, replace_inserstions: Optional[str] = None):
    seq = seq.upper()
    seq = seq.replace(".", "-")
    if replace_inserstions is not None:
        seq = seq.replace("-", replace_inserstions)
    return seq


def read_msa(pth: Path, replace_inserstions: Optional[str] = None) -> list[str]:
    """Reads an .a2m MSA file and returns a list of sequences."""
    msa = [process_msa_seq(str(s.seq), replace_inserstions) for s in SeqIO.parse(pth, "fasta")]
    return msa
