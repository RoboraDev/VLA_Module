import torch

from torch import nn
from transformers import PaliGemmaForConditionalGeneration, GemmaForCausalLM
from typing import Literal
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, action_expert
):
    models = [paligemma.language_model, action_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
        after_first_residual = out_emb.clone()
        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds

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

        self._to_bfloat16_for_selected_params(precision)

    # helper method to convert precision of selected parameters, though keeping some in original float32
    def _to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):

        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # These are the parameters that should be kept in float32
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    ## Got these features abstracted from transformers library.
    # Will be used later on to convert image to embeddings.
    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    # Will be used later on to convert language tokens to embeddings.
    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
            self,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_kv: list[torch.FloatTensor] | None = None,
            inputs_embeds: list[torch.FloatTensor] | None = None,
            use_cache: bool | None = None,
            adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]

        # Path 1[1]: VLM-only processing
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None

        # Path 2[0]: Action expert-only processing
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None

        # Path 3[1-0]: Dual processing (both VLM and action expert)
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Process all layers sequentially
            for layer_idx in range(num_layers):
                inputs_embeds = compute_layer_complete(
                    layer_idx,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    paligemma=self.paligemma,
                    action_expert=self.gemma_expert,
                )

            # Apply final layer normalization
            outputs_embeds = []
            for i, hidden_states in enumerate(inputs_embeds):
                out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                outputs_embeds.append(out_emb)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
