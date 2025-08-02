import torch
import typing as T
from transformers import EsmForMaskedLM, AutoTokenizer

from proteinttt.models.esm2 import DEFAULT_ESM2_35M_TTT_CFG, TTTConfig
from proteinttt.base import TTTModule


class ESM2TTT_HF(TTTModule, EsmForMaskedLM):
    """
    ESM2TTT_HF is a TTTModule that uses the ESM2 model from Hugging Face.
    """
    ttt_default_cfg = DEFAULT_ESM2_35M_TTT_CFG

    def __init__(self, ttt_cfg: TTTConfig, **kwargs):
        EsmForMaskedLM.__init__(self, **kwargs)
        TTTModule.__init__(self, ttt_cfg=ttt_cfg)
        self.ttt_tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)

    def _ttt_tokenize(self, seq: T.Optional[str], **kwargs) -> torch.Tensor:
        if seq is not None:
            seq = seq.replace('X', '<unk>')
            x = self.ttt_tokenizer(seq, return_tensors="pt")["input_ids"]
        else:
            assert "input_ids" in kwargs, "input_ids must be provided if no seq is provided"
            x = kwargs["input_ids"]
            assert isinstance(x, torch.Tensor), "input_ids must be a tensor"
        return x  # [bs, seq_len]

    def _ttt_get_frozen_modules(self) -> list[torch.nn.Module]:
        return [self.esm.embeddings]

    def _ttt_mask_token(self, token: int) -> int:
        return self.ttt_tokenizer.mask_token_id

    def _ttt_get_padding_token(self) -> int:
        return self.ttt_tokenizer.pad_token_id

    def _ttt_token_to_str(self, token: int) -> str:
        return self.ttt_tokenizer.decode([token])

    def _ttt_get_all_tokens(self) -> list[int]:
        return list(self.ttt_tokenizer.get_vocab().values())

    def _ttt_get_non_special_tokens(self) -> list[int]:
        return [
            t for t in self._ttt_get_all_tokens()
            if t not in self.ttt_tokenizer.all_special_ids
        ]

    def _ttt_predict_logits(self, batch: torch.Tensor, start_indices: torch.Tensor = None, **kwargs) -> torch.Tensor:
        # Assumes batch is [bs, seq_len] input_ids
        attention_mask = (batch != self._ttt_get_padding_token()).long()
        outputs = self(input_ids=batch, attention_mask=attention_mask)
        return outputs.logits  # [bs, seq_len, vocab_size]