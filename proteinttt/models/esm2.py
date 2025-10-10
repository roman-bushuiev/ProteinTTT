import torch
import esm
from esm.model.esm2 import ESM2

from proteinttt.base import TTTModule, TTTConfig


# TODO Update this config for the new loss. These might be to aggressive
DEFAULT_ESM2_35M_TTT_CFG = TTTConfig(
    lr=4e-4,
    batch_size=4,
    ags=16,
    steps=30,
    loss_kind="unnormalized_cross_entropy",
)


DEFAULT_ESM2_650M_TTT_CFG = TTTConfig(
    lr=4e-5,
    batch_size=4,
    ags=4,
    steps=30,
    loss_kind="unnormalized_cross_entropy",
)


class ESM2TTT(TTTModule, ESM2):
    ttt_default_cfg = DEFAULT_ESM2_650M_TTT_CFG

    def __init__(self, ttt_cfg: TTTConfig, **kwargs):
        ESM2.__init__(self, **kwargs)
        TTTModule.__init__(self, ttt_cfg=ttt_cfg)
        self.ttt_alphabet = esm.Alphabet.from_architecture(
            "ESM-1b"
        )  # ESM2 uses ESM-1b alphabet
        self.ttt_batch_converter = self.ttt_alphabet.get_batch_converter()

    def _ttt_tokenize(self, seq: str, **kwargs):
        batch_labels, batch_strs, batch_tokens = self.ttt_batch_converter(
            [(None, seq)]
        )
        return batch_tokens

    def _ttt_get_frozen_modules(self) -> list[torch.nn.Module]:
        return [self.embed_tokens]

    def _ttt_mask_token(self, token: int) -> int:
        return self.ttt_alphabet.mask_idx

    def _ttt_get_padding_token(self) -> int:
        return self.ttt_alphabet.padding_idx

    def _ttt_token_to_str(self, token: int) -> str:
        return self.ttt_alphabet.all_toks[token]

    def _ttt_get_all_tokens(self) -> list[int]:
        return [
            self.ttt_alphabet.tok_to_idx[t] for t in self.ttt_alphabet.all_toks
        ]

    def _ttt_get_non_special_tokens(self) -> list[int]:
        return [
            self.ttt_alphabet.tok_to_idx[t]
            for t in self.ttt_alphabet.standard_toks
        ]

    def _ttt_predict_logits(
        self, batch: torch.Tensor, start_indices: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        return self(batch)[
            "logits"
        ]  # [bs, seq_len] -> [bs, seq_len, vocab_size]
