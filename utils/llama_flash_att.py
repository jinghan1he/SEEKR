from typing import Optional, Tuple, List, Dict
from functools import partial
import math
import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, _make_causal_mask, _expand_mask, repeat_kv
from peft import PeftModelForCausalLM, LoraModel
from einops import rearrange

#try to import flash_attn 2.x.x, if not, import flash_attn 1.x.x
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func
except:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

from flash_attn.bert_padding import unpad_input, pad_input


def base_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    assert past_key_value is None, "past_key_value is not supported"

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]
    assert not output_attentions, "output_attentions is not supported"
    assert not use_cache, "use_cache is not supported"

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    key_padding_mask = attention_mask

    if key_padding_mask is None:
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        max_s = q_len
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(
            x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
        )
        output_unpad = flash_attn_unpadded_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
            ),
            "b s (h d) -> b s h d",
            h=nheads,
        )
    attn_output = rearrange(output, "b s h d -> b s (h d)")
    return self.o_proj(attn_output), query_states, key_states, value_states, attn_output


def flash_attn_forward(
    self,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    output, _, _, _, _ = base_flash_attn_forward(self, **kwargs)
    return output, None, None


# original forward
def base_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value


def attn_forward(
        self,
        output_attentions: bool = False,
        **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    attn_output, attn_weights, past_key_value = base_attn_forward(self, **kwargs)
    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


def attn_distill_budget_forward(
        self,
        output_attentions: bool = False,
        teacher_attns: Optional[List[torch.Tensor]] = None,
        select_heads: Optional[List[torch.Tensor]] = None,
        **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    attention_mask = kwargs["attention_mask"]
    kwargs["attention_mask"] = _prepare_decoder_attention_mask(None, attention_mask, attention_mask.shape, kwargs["hidden_states"], 0)
    attn_output, attn_weights, past_key_value = base_attn_forward(self, **kwargs)

    self.loss_attn_distill = 0.
    if teacher_attns is not None:
        kldiv_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
        bs = len(teacher_attns)
        for i in range(bs):
            if len(select_heads[i]) > 0:
                # bs, n_heads, seq_len, seq_len
                # discard attention on padding tokens
                seq_len = attention_mask[i].sum()
                stu_a = attn_weights[i, select_heads[i], -seq_len:, -seq_len:].contiguous()
                tea_a = teacher_attns[i][:, -seq_len:, -seq_len:].contiguous()
                # self.loss_attn_distill += F.mse_loss(stu_a, tea_a) / bs
                self.loss_attn_distill += kldiv_fct(torch.log(stu_a.view(-1, seq_len)+1e-8), tea_a.view(-1, seq_len)) / bs

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def attn_distill_budget2_forward(
        self,
        output_attentions: bool = False,
        teacher_attns: Optional[torch.Tensor] = None,
        attn_query: Optional[torch.Tensor] = None,
        select_heads: Optional[torch.Tensor] = None,
        **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    attention_mask = kwargs["attention_mask"]
    kwargs["attention_mask"] = _prepare_decoder_attention_mask(None, attention_mask, attention_mask.shape, kwargs["hidden_states"], 0)
    attn_output, attn_weights, past_key_value = base_attn_forward(self, **kwargs)

    self.loss_attn_distill = 0.
    if teacher_attns is not None:
        kldiv_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
        bs = len(teacher_attns)
        for i in range(bs):
            if len(select_heads[i]) > 0:
                # bs, n_heads, seq_len, seq_len
                # discard attention on padding tokens
                seq_len = attention_mask[i].sum()
                stu_a = attn_weights[i, select_heads[i], -seq_len:, -seq_len:].index_select(-2, attn_query[i]+seq_len).contiguous()
                tea_a = teacher_attns[i][:, :, -seq_len:].contiguous()
                # self.loss_attn_distill += F.mse_loss(stu_a, tea_a) / bs
                self.loss_attn_distill += kldiv_fct(torch.log(stu_a.view(-1, seq_len)+1e-8), tea_a.view(-1, seq_len)) / bs

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def attn_grad_forward(
        self,
        attn_grad: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    attention_mask = _prepare_decoder_attention_mask(None, attention_mask, attention_mask.shape, hidden_states, 0)

    bsz, q_len, _ = hidden_states.size()

    if self.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    self.attn_weights = attn_weights            # output attn_weights
    attn_weights = attn_grad(attn_weights)      # forward for attn_grad computation

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def attn_output_forward(
        self,
        output_attentions: bool = False,
        **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    attention_mask = kwargs["attention_mask"]
    kwargs["attention_mask"] = _prepare_decoder_attention_mask(None, attention_mask, attention_mask.shape, kwargs["hidden_states"], 0)
    attn_output, attn_weights, past_key_value = base_attn_forward(self, **kwargs)

    self.attn_weights = attn_weights

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _flash_attn_prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def replace_llama_attn_with_flash_attn():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _flash_attn_prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = flash_attn_forward

def replace_flash_attn_with_llama_attn():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = attn_forward


def replace_attn(model, layers):
    for i, l in enumerate(layers):
        layer = model.model.layers[l]
        layer.self_attn.forward = partial(attn_forward, layer.self_attn)


def replace_flash_attn(model, layers):
    for i, l in enumerate(layers):
        layer = model.model.layers[l]
        layer.self_attn.forward = partial(flash_attn_forward, layer.self_attn)

def replace_attn_output_attn(model, layers):
    for i, l in enumerate(layers):
        layer = model.model.layers[l]
        layer.self_attn.forward = partial(attn_output_forward, layer.self_attn)


def replace_attn_grad_attn(model, layers, attn_grads):
    for i, l in enumerate(layers):
        layer = model.model.layers[l]
        attn_grad = attn_grads[i]
        layer.self_attn.forward = partial(attn_grad_forward, layer.self_attn, attn_grad)


def replace_attn_distill_budget_attn(model, batch_teacher_attns: List[Tuple[torch.Tensor]], batch_select_heads: List[Dict]):
    layers = set([l for sh in batch_select_heads for l in sh.keys()])
    for l in layers:
        layer = model.model.layers[l]
        layer_teacher_attns = []
        layer_select_heads = []
        for select_heads, teacher_attns in zip(batch_select_heads, batch_teacher_attns):
            if l in select_heads:
                l_idx = list(select_heads.keys()).index(l)
                layer_teacher_attns.append(teacher_attns[l_idx])
                layer_select_heads.append(select_heads[l].to(teacher_attns[0].device))
            else:
                layer_teacher_attns.append([])
                layer_select_heads.append([])
            layer.self_attn.forward = partial(attn_distill_budget_forward, layer.self_attn, teacher_attns=layer_teacher_attns, select_heads=layer_select_heads)


def replace_attn_distill_budget2_attn(model, batch_teacher_attns: List[Tuple[torch.Tensor]], batch_select_heads: List[Dict], attn_query):
    layers = set([l for sh in batch_select_heads for l in sh.keys()])
    for l in layers:
        layer = model.model.layers[l]
        layer_teacher_attns = []
        layer_select_heads = []
        for select_heads, teacher_attns in zip(batch_select_heads, batch_teacher_attns):
            if l in select_heads:
                l_idx = list(select_heads.keys()).index(l)
                layer_teacher_attns.append(teacher_attns[l_idx])
                layer_select_heads.append(select_heads[l].to(teacher_attns[0].device))
            else:
                layer_teacher_attns.append([])
                layer_select_heads.append([])
            layer.self_attn.forward = partial(attn_distill_budget2_forward, layer.self_attn, teacher_attns=layer_teacher_attns, select_heads=layer_select_heads, attn_query=attn_query)

