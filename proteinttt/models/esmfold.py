import typing as T
import tempfile

import torch
import biotite.structure.io as bsio
import esm
from esm.esmfold.v1.esmfold import ESMFold

from proteinttt.base import TTTModule, TTTConfig


DEFAULT_ESMFOLD_TTT_CFG = TTTConfig(
    lr=4e-4,
    batch_size=4,
    ags=4,
    steps=30,
    lora_rank=8,
    lora_alpha=32.0
)


class ESMFoldTTT(TTTModule, ESMFold):
    ttt_default_cfg = DEFAULT_ESMFOLD_TTT_CFG

    def __init__(self, ttt_cfg: TTTConfig, **kwargs):
        ESMFold.__init__(self, **kwargs)
        TTTModule.__init__(self, ttt_cfg=ttt_cfg)
        self.ttt_alphabet = esm.Alphabet.from_architecture("ESM-1b")  # ESM2 uses ESM-1b alphabet
        self.ttt_batch_converter = self.ttt_alphabet.get_batch_converter()

    def _ttt_tokenize(self, sequence: str) -> torch.Tensor:
        _, _, x = self.ttt_batch_converter([(None, sequence)])
        return x

    def _ttt_get_trainable_modules(self) -> list[torch.nn.Module]:
        return [self.esm]

    def _ttt_get_frozen_modules(self) -> list[torch.nn.Module]:
        return [self.esm.embed_tokens]
    
    def _ttt_mask_token(self, token: int) -> int:
        return self.ttt_alphabet.mask_idx
    
    def _ttt_get_all_tokens(self) -> list[int]:
        return [self.ttt_alphabet.tok_to_idx[t] for t in self.ttt_alphabet.all_toks]
    
    def _ttt_get_non_special_tokens(self) -> list[int]:
        return [self.ttt_alphabet.tok_to_idx[t] for t in self.ttt_alphabet.standard_toks]

    def _ttt_predict_logits(self, batch: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.esm(batch)["logits"]  # [bs, seq_len] -> [bs, seq_len, vocab_size]

    def _ttt_eval_step(
        self,
        step: int,
        loss: torch.Tensor,
        perplexity: float,
        all_log_probs: torch.Tensor,
        ttt_args: T.Tuple,
        ttt_kwargs: T.Dict
    ) -> tuple[dict, dict, T.Optional[float]]:
        sequence = ttt_args[0]

        # Predict structure
        with torch.no_grad():
            pdb_str = self.infer_pdb(sequence, masking_pattern=None)

        # Calculate pLDDT
        # TODO Optimize by not saving to disk
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb') as tmp:
            tmp.write(pdb_str)
            tmp.flush()
            struct = bsio.load_structure(tmp.name, extra_fields=["b_factor"])
            plddt = struct.b_factor.mean()

        # Store predictions
        eval_step_preds = {'pdb': pdb_str}
        eval_step_metric_dict = {'plddt': plddt}
        confidence = plddt

        return eval_step_preds, eval_step_metric_dict, confidence
    