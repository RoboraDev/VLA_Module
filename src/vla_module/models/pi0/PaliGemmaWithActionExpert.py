import torch

from torch import nn
from transformers import PaliGemmaForConditionalGeneration, GemmaForCausalLM
from typing import Literal
from transformers.models.auto import CONFIG_MAPPING

# Combines the PI0 VLM PaliGemma with it's action head/transformer aka Expert in here
class PaliGemmaWithActionExpert(
    nn.Module
):
    # implemented from PI0 Github Repository
    """PaliGemma model with action expert for PI0."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        #======= VLM_CONFIG & ACTION_EXPERT_CONFIG IS TAKEN FROM config.json ==========
        # Creating paligemma Config class & modifying some parameters accordingly.
        # TODO: Shiven, some values are picked from vlm_config and some are hardcoded.
        # look into it if that's necessary or not.

        # VLM Language part
        text_config_hf = CONFIG_MAPPING["gemma"](
            hidden_size = vlm_config.width,
            intermediate_size = vlm_config.mlp_dim,
            num_attention_heads = vlm_config.num_heads,
            head_dim = vlm_config.head_dim,
            num_hidden_layers = vlm_config.depth,
            num_key_value_heads = vlm_config.num_kv_heads,
            hidden_activation = "gelu_pytorch_tanh",
            torch_dtype = "float32",
            vocab_size = 257152,
            use_adarms = use_adarms[0],
            adarms_cond_dim = vlm_config.width if use_adarms[0] else None,
        )

        # VLM vision tower aka siglip vision encoder
        vision_config_hf = CONFIG_MAPPING["siglip_vision_model"](
            intermediate_size = 4304,
            projection_dim = 2048,
            projector_hidden_act = "gelu_fast",
            torch_dtype = "float32"
        )

        vlm_config_hf = CONFIG_MAPPING["paligemma"](
            _vocab_size = 257152,  # noqa: SLF001
            image_token_index = 257152,
            text_config = text_config_hf,
            vision_config = vision_config_hf,
        )


        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None



