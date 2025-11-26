from transformers import EsmForMaskedLM

from proteinttt.base import TTTConfig
from proteinttt.models.esm2_hf import ESM2TTT_HF


DEFAULT_SAPROT_35M_TTT_CFG = TTTConfig(
    lr=4e-4,
    batch_size=4,
    ags=8,
    steps=30,
    loss_kind="unnormalized_cross_entropy",
)


DEFAULT_SAPROT_650M_TTT_CFG = TTTConfig(
    lr=4e-5,
    batch_size=2,
    ags=16,
    steps=30,
    loss_kind="unnormalized_cross_entropy",
)


class SaProtTTT_HF(ESM2TTT_HF, EsmForMaskedLM):
    """
    SaProtTTT_HF is a TTTModule that uses the SaProt model from Hugging Face.
    """
    ttt_default_cfg = DEFAULT_SAPROT_650M_TTT_CFG

    def _ttt_mask_token(self, token: int) -> int:
        """
        Mask sequential information and preserve the structure information.
        """
        token_str = self.ttt_tokenizer.id_to_token(token.item())
        assert len(token_str) == 2, "SaProt token string should be of length 2"
        token_str_masked = "#" + token_str[1]
        return self.ttt_tokenizer.token_to_id(token_str_masked)
