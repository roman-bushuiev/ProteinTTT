import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from proteinttt.base import TTTModule, TTTConfig


model = AutoModelForMaskedLM.from_pretrained(
    "AI4Protein/ProSST-2048", trust_remote_code=True
)
MODEL_CLASS = model.__class__


class ProSSTTTT(TTTModule, MODEL_CLASS):
    def __init__(self, ttt_cfg: TTTConfig, **kwargs):
        MODEL_CLASS.__init__(self, **kwargs)
        TTTModule.__init__(self, ttt_cfg=ttt_cfg)
        self.ttt_tokenizer = AutoTokenizer.from_pretrained(
            model.name_or_path,
            trust_remote_code=True
        )

    def _ttt_get_frozen_modules(self) -> list[torch.nn.Module]:
        return [self.prosst.embeddings]
    
    def _ttt_mask_token(self, token: int) -> int:
        return self.ttt_tokenizer.mask_token_id
    
    def _ttt_get_all_tokens(self) -> list[int]:
        return list(self.ttt_tokenizer.get_vocab().values())
    
    def _ttt_get_non_special_tokens(self) -> list[int]:
        return [
            t for t in self._ttt_get_all_tokens()
            if t not in self.ttt_tokenizer.all_special_ids
        ]

    def _ttt_tokenize(self, *args, **kwargs) -> torch.Tensor:
        return kwargs["input_ids"] if "input_ids" in kwargs else args[0]

    def _ttt_predict_logits(self, batch: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Copy structure sequence to all sequences in the batch and apply consistent cropping
        ss_input_ids=kwargs["ss_input_ids"]
        ss_input_ids = ss_input_ids[0].expand(batch.shape[0], -1)  # [1,seq_len] -> [bs, seq_len]
        if "start_indices" in kwargs:
            start_indices = kwargs["start_indices"]  # [bs]
            ss_input_ids_windows = []
            for i in range(batch.shape[0]):
                start = start_indices[i]
                end = start + batch.shape[1]
                ss_input_ids_windows.append(ss_input_ids[i:i+1, start:end])  # [1, seq_len]
            ss_input_ids = torch.cat(ss_input_ids_windows, dim=0)  # [bs, seq_len]
        
        # Predict logits
        outputs = self(input_ids=batch, ss_input_ids=ss_input_ids)
        return outputs.logits  # [bs, seq_len, vocab_size]
 