import torch
import esm
from esm.model.esm2 import ESM2

from proteinttt.base import TTTModule, TTTConfig


DEFAULT_ESM2_35M_TTT_CFG = TTTConfig(
    lr=4e-4,
    batch_size=4,
    ags=16,
    steps=30
)


DEFAULT_ESM2_650M_TTT_CFG = TTTConfig(
    lr=4e-5,
    batch_size=4,
    ags=16,
    steps=30
)


class ESM2TTT(TTTModule, ESM2):
    ttt_default_cfg = DEFAULT_ESM2_650M_TTT_CFG

    def __init__(self, ttt_cfg: TTTConfig, **kwargs):
        ESM2.__init__(self, **kwargs)
        TTTModule.__init__(self, ttt_cfg=ttt_cfg)
        self.ttt_alphabet = esm.Alphabet.from_architecture("ESM-1b")  # ESM2 uses ESM-1b alphabet

    def _ttt_get_frozen_modules(self) -> list[torch.nn.Module]:
        return [self.embed_tokens]
    
    def _ttt_mask_token(self, token: int) -> int:
        return self.ttt_alphabet.mask_idx
    
    def _ttt_get_all_tokens(self) -> list[int]:
        return [self.ttt_alphabet.tok_to_idx[t] for t in self.ttt_alphabet.all_toks]
    
    def _ttt_get_non_special_tokens(self) -> list[int]:
        return [self.ttt_alphabet.tok_to_idx[t] for t in self.ttt_alphabet.standard_toks]

    def _ttt_predict_logits(self, batch: torch.Tensor, start_indices: torch.Tensor = None, *args, **kwargs) -> torch.Tensor:
        return self(batch)["logits"]  # [bs, seq_len] -> [bs, seq_len, vocab_size]
