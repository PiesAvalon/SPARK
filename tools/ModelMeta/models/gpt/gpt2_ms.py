# coding=utf-8
# ----------------------------------------------------------------------------
# 这是示例的 MindSpore 版本 GPT-2 实现，仅供参考。
# 可能需要根据您的实际需求（比如数据类型、初始化、训练流程）
# 进行相应修改与完善。
# ----------------------------------------------------------------------------

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import torch
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer, Normal
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore import load_checkpoint, load_param_into_net


# 将日志级别设置为 ERROR，只显示错误，不再显示 WARNING。
# mindspore.set_context(logging_level=mindspore.ERROR)

###############################################################################
# 一些输出结构（类似于 HF 的 ModelOutput）
###############################################################################
@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    """
    保存主干输出的结构，如 hidden_states、past_key_values 等。
    """
    last_hidden_state: Tensor
    past_key_values: Optional[Tuple[Tuple[Tensor]]] = None
    hidden_states: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tensor]] = None
    cross_attentions: Optional[Tuple[Tensor]] = None

@dataclass
class CausalLMOutputWithCrossAttentions:
    """
    用于语言建模 (Causal LM) 场景的输出结构。
    """
    loss: Optional[Tensor] = None
    logits: Tensor = None
    past_key_values: Optional[Tuple[Tuple[Tensor]]] = None
    hidden_states: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tensor]] = None
    cross_attentions: Optional[Tuple[Tensor]] = None

@dataclass
class SequenceClassifierOutputWithPast:
    """
    用于序列分类任务的输出结构。
    """
    loss: Optional[Tensor] = None
    logits: Tensor = None
    past_key_values: Optional[Tuple[Tuple[Tensor]]] = None
    hidden_states: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tensor]] = None

@dataclass
class TokenClassifierOutput:
    """
    用于序列标注（Token-Level）的输出。
    """
    loss: Optional[Tensor] = None
    logits: Tensor = None
    hidden_states: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tensor]] = None

@dataclass
class QuestionAnsweringModelOutput:
    """
    用于问答场景（Span QA）的输出。
    """
    loss: Optional[Tensor] = None
    start_logits: Tensor = None
    end_logits: Tensor = None
    hidden_states: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tensor]] = None

@dataclass
class GPT2DoubleHeadsModelOutput:
    """
    GPT-2 DoubleHeads（多重选择 + LM）的输出。
    """
    loss: Optional[Tensor] = None
    mc_loss: Optional[Tensor] = None
    logits: Tensor = None
    mc_logits: Tensor = None
    past_key_values: Optional[Tuple[Tuple[Tensor]]] = None
    hidden_states: Optional[Tuple[Tensor]] = None
    attentions: Optional[Tuple[Tensor]] = None


###############################################################################
# 一些基础模块
###############################################################################

class Conv1D(nn.Cell):
    def __init__(self, out_channels, in_channels):
        super().__init__()
        # 正确写法：通过 initializer(Normal(0.02), shape, dtype) 创建
        weight_init = initializer(Normal(sigma=0.02), [in_channels, out_channels], mstype.float32)
        bias_init = initializer('zeros', [out_channels], mstype.float32)

        self.weight = Parameter(weight_init, name="weight")
        self.bias = Parameter(bias_init, name="bias")

    def construct(self, x):
        out = ops.matmul(x, self.weight)
        out = out + self.bias
        return out

def _get_activation(act_str: str):
    """
    根据字符串获取激活函数，示例中仅简单支持 gelu / relu，可自行补充。
    """
    if act_str == "gelu":
        return ops.GeLU()
    elif act_str == "relu":
        return ops.ReLU()
    # 默认用 gelu
    return ops.GeLU()

class GPT2Attention(Cell):
    """
    GPT-2 自注意力模块，包含因果掩码。
    """
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        self.is_cross_attention = is_cross_attention
        self.num_heads = config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.head_dim = self.embed_dim // self.num_heads
        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        max_positions = config.max_position_embeddings
        # MindSpore 没有类似 PyTorch 的 register_buffer，这里可以用 Parameter(…, requires_grad=False)
        self.bias = Parameter(
            ops.tril(ops.ones((max_positions, max_positions), mstype.bool_)).view(1, 1, max_positions, max_positions),
            name="bias", requires_grad=False
        )
        self.masked_bias = Parameter(Tensor([-1e4], mstype.float32), name="masked_bias", requires_grad=False)

        if self.is_cross_attention:
            self.c_attn = Conv1D(2*self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3*self.embed_dim, self.embed_dim)

        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        # Dropout 的 keep_prob = 1 - p
        self.attn_dropout = nn.Dropout(1 - config.attn_pdrop)
        self.resid_dropout = nn.Dropout(1 - config.resid_pdrop)

    def construct(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False
    ):
        if encoder_hidden_states is not None:
            # 交叉注意力
            query = self.q_attn(hidden_states)
            k_v = self.c_attn(encoder_hidden_states)
            key, value = ops.split(k_v, split_size_or_sections=self.embed_dim, axis=2)
        else:
            # 自注意力
            mixed = self.c_attn(hidden_states)
            query, key, value = ops.split(mixed, split_size_or_sections=self.embed_dim, axis=2)

        # [batch, n_heads, seq_len, head_dim]
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key   = self._split_heads(key,   self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # 拼 past
        if layer_past is not None:
            past_key, past_value = layer_past
            key = ops.concat((past_key, key), axis=-2)
            value = ops.concat((past_value, value), axis=-2)

        present = None
        if use_cache:
            present = (key, value)

        # 进行注意力计算
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        # 合并 heads
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    def _split_heads(self, tensor, num_heads, head_dim):
        shape = tensor.shape
        new_shape = (shape[0], shape[1], num_heads, head_dim)
        tensor = ops.reshape(tensor, new_shape)
        tensor = ops.transpose(tensor, (0, 2, 1, 3))  # [batch, head, seq, head_dim]
        return tensor

    def _merge_heads(self, tensor, num_heads, head_dim):
        tensor = ops.transpose(tensor, (0, 2, 1, 3))  # [batch, seq, head, head_dim]
        shape = tensor.shape
        new_shape = (shape[0], shape[1], num_heads*head_dim)
        return ops.reshape(tensor, new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # query: [batch, heads, q_len, head_dim]
        # key:   [batch, heads, k_len, head_dim]
        attn_weights = ops.matmul(query, ops.transpose(key, (0,1,3,2)))
        if self.scale_attn_weights:
            scale = Tensor([float(value.shape[-1])**0.5], attn_weights.dtype)
            attn_weights = attn_weights / scale
        if self.scale_attn_by_inverse_layer_idx and (self.layer_idx is not None):
            inv = Tensor([float(self.layer_idx+1)], attn_weights.dtype)
            attn_weights = attn_weights / inv

        bsz, _, q_len, k_len = attn_weights.shape
        # causal mask
        causal_mask = self.bias[:, :, k_len - q_len: k_len, :k_len]
        mask_value = Tensor([-1e4], attn_weights.dtype)
        # 这里用 ops.select
        attn_weights = ops.select(causal_mask, attn_weights, ops.fill(attn_weights.dtype, attn_weights.shape, mask_value))

        # 如果有外部 attention_mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, -1)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = ops.matmul(attn_weights, value)
        return attn_output, attn_weights


class GPT2MLP(Cell):
    """
    GPT-2 的前馈网络 (MLP)。
    """
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = _get_activation(config.activation_function)
        self.dropout = nn.Dropout(1 - config.resid_pdrop)

    def construct(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(Cell):
    """
    GPT-2 的 Transformer Block，包括 (LN -> SelfAttn -> 残差 + LN -> MLP -> 残差)；可选 Cross-Attn。
    """
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4*hidden_size
        self.ln_1 = nn.LayerNorm([hidden_size], epsilon=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm([hidden_size], epsilon=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm([hidden_size], epsilon=config.layer_norm_epsilon)

    def construct(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None and self.add_cross_attention:
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
                output_attentions=output_attentions
            )
            cross_attn_output = cross_outputs[0]
            hidden_states = residual + cross_attn_output
            outputs = outputs + cross_outputs[2:]  # 如果有注意力权重，就附加上

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_fwd = self.mlp(hidden_states)
        hidden_states = residual + feed_fwd

        if use_cache:
            return (hidden_states,) + outputs
        else:
            return (hidden_states,) + outputs[1:]


###############################################################################
# GPT2PreTrainedModel 以及 GPT2Config
###############################################################################
# class GPT2Config:
#     """
#     简化的 GPT2Config，仅保留常用字段供演示。
#     """
#     def __init__(
#         self,
#         vocab_size=50257,
#         max_position_embeddings=1024,
#         hidden_size=768,
#         num_hidden_layers=12,
#         num_attention_heads=12,
#         n_inner=None,
#         activation_function="gelu",
#         resid_pdrop=0.1,
#         embd_pdrop=0.1,
#         attn_pdrop=0.1,
#         layer_norm_epsilon=1e-5,
#         initializer_range=0.02,
#         scale_attn_weights=True,
#         add_cross_attention=False,
#         use_cache=True,
#         output_attentions=False,
#         output_hidden_states=False,
#         use_return_dict=True,
#         scale_attn_by_inverse_layer_idx=False,
#         reorder_and_upcast_attn=False,
#         n_layer=12,
#         # 此处可以再加一些自定义属性
#         num_labels=2,
#         classifier_dropout=None
#     ):
#         self.vocab_size = vocab_size
#         self.max_position_embeddings = max_position_embeddings
#         self.hidden_size = hidden_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads
#         self.n_inner = n_inner
#         self.activation_function = activation_function
#         self.resid_pdrop = resid_pdrop
#         self.embd_pdrop = embd_pdrop
#         self.attn_pdrop = attn_pdrop
#         self.layer_norm_epsilon = layer_norm_epsilon
#         self.initializer_range = initializer_range
#         self.scale_attn_weights = scale_attn_weights
#         self.add_cross_attention = add_cross_attention
#         self.use_cache = use_cache
#         self.output_attentions = output_attentions
#         self.output_hidden_states = output_hidden_states
#         self.use_return_dict = use_return_dict
#         self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
#         self.reorder_and_upcast_attn = reorder_and_upcast_attn
#         self.n_layer = n_layer
#
#         self.num_labels = num_labels
#         self.classifier_dropout = classifier_dropout

class GPT2Config:
    """
    修改后的 GPT2Config，使其参数名称和默认值与 config.json 文件匹配。
    """
    def __init__(
        self,
        vocab_size=50257,
        n_ctx=1024,                     # 对应 config.json 中 n_ctx
        n_embd=768,                     # 对应 config.json 中 n_embd
        n_layer=12,                     # 对应 config.json 中 n_layer
        n_head=12,                      # 对应 config.json 中 n_head
        n_inner=None,
        activation_function="gelu",     # 默认值，可由 config.json 中 "activation_function" 覆盖（例如 "gelu_new"）
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        add_cross_attention=False,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        # 以下为新增字段，对应 config.json 中的其他键
        architectures=None,
        bos_token_id=50256,
        eos_token_id=50256,
        model_type="gpt2",
        summary_activation=None,
        summary_first_dropout=0.1,
        summary_proj_to_labels=True,
        summary_type="cls_index",
        summary_use_proj=True,
        task_specific_params=None,
        # 自定义字段（原有代码中存在）
        num_labels=2,
        classifier_dropout=None
    ):
        self.vocab_size = vocab_size
        # 使用 n_ctx 作为最大位置编码，同时保持 n_positions 和 max_position_embeddings 字段以便兼容部分调用
        self.n_ctx = n_ctx
        self.n_positions = n_ctx
        self.max_position_embeddings = n_ctx  # 添加此行解决报错问题
        self.n_embd = n_embd
        self.hidden_size = n_embd  # 保持向后兼容
        # 层数同时保存为 n_layer 和 num_hidden_layers
        self.n_layer = n_layer
        self.num_hidden_layers = n_layer
        # 注意力头数同时保存为 n_head 和 num_attention_heads
        self.n_head = n_head
        self.num_attention_heads = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.add_cross_attention = add_cross_attention
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        # 以下字段直接从 config.json 对应键获取
        self.architectures = architectures
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.model_type = model_type
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.task_specific_params = task_specific_params

        # 自定义字段
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout



class GPT2PreTrainedModel(Cell):
    """
    提供一些通用方法，比如初始化权重等。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_head_mask(self, head_mask, num_hidden_layers):
        return head_mask  # 这里可自行处理

    def post_init(self):
        pass

    def construct(self, *args, **kwargs):
        raise NotImplementedError("请使用子类的 construct。")


###############################################################################
# GPT2Model 主干
###############################################################################
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(1 - config.embd_pdrop)

        self.h = nn.CellList([GPT2Block(config, i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm([self.embed_dim], epsilon=config.layer_norm_epsilon)

        self.post_init()

    def construct(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("input_ids 和 inputs_embeds 不能同时指定。")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            batch_size, seq_length, _ = inputs_embeds.shape

        if past_key_values is None:
            past_key_values = (None,) * len(self.h)

        if position_ids is None:
            position_ids = ops.arange(0, seq_length, 1)
            position_ids = ops.reshape(position_ids, (1, -1))
            position_ids = ops.broadcast_to(position_ids, (batch_size, seq_length))

        if inputs_embeds is None:
            hidden_states = self.wte(input_ids)
        else:
            hidden_states = inputs_embeds

        position_embeds = self.wpe(position_ids)
        hidden_states = hidden_states + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        all_hidden_states = ()
        all_self_attentions = ()
        all_cross_attentions = ()
        presents = ()

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=(use_cache if use_cache is not None else self.config.use_cache),
                output_attentions=(output_attentions if output_attentions is not None else self.config.output_attentions)
            )
            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)
            if (output_attentions if output_attentions is not None else self.config.output_attentions):
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not (return_dict if return_dict is not None else self.config.use_return_dict):
            return tuple(
                v for v in [
                    hidden_states,
                    presents if use_cache else None,
                    all_hidden_states if output_hidden_states else None,
                    all_self_attentions if output_attentions else None,
                    all_cross_attentions if (output_attentions and self.config.add_cross_attention) else None,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents if use_cache else None,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_self_attentions if output_attentions else None,
            cross_attentions=all_cross_attentions if (output_attentions and self.config.add_cross_attention) else None,
        )

    @classmethod
    def from_pretrained(cls, ckpt_path, config=None):
        """
        加载一个 MindSpore 格式的 GPT-2 ckpt。
        ckpt_path: 例如 "./models/gpt2/ms_gpt2.ckpt"
        config:    如果没提供则使用默认 GPT2Config，也可自己从 json/yaml 中读取
        """
        if config is None:
            config = GPT2Config()  # 或者也可以从某个 config.json 解析

        # 实例化模型
        model = cls(config)
        # 载入参数
        param_dict = load_checkpoint(ckpt_path)
        load_param_into_net(model, param_dict)
        return model


###############################################################################
# GPT2LMHeadModel
###############################################################################
class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)
        self.post_init()

    def construct(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer.construct(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss_fn = nn.CrossEntropyLoss()
            vocab_size = shift_logits.shape[-1]
            loss = loss_fn(
                ops.reshape(shift_logits, (-1, vocab_size)),
                ops.reshape(shift_labels, (-1,))
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @classmethod
    def from_pretrained(cls, ckpt_path, config=None):
        """
        加载一个 MindSpore 格式的 GPT-2 LMHeadModel 预训练权重。
        ckpt_path: 例如 "./models/gpt2/ms_gpt2_lmhead.ckpt"
        config:    如果没提供则使用默认 GPT2Config，也可自己从 json/yaml 中读取
        """
        if config is None:
            config = GPT2Config()  # 或者也可以从某个 config.json 解析

        # 实例化模型
        model = cls(config)
        # 载入参数
        param_dict = load_checkpoint(ckpt_path)
        load_param_into_net(model, param_dict)
        return model


###############################################################################
# 类似地再添加其他头：如 SequenceClassification, TokenClassification, QA 等
# 下面列出一个示例
###############################################################################
class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)
        self.post_init()

    def construct(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer.construct(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs.last_hidden_state
        # GPT-2 常用做法：取序列中最后一个 token 的隐状态来做分类
        batch_size = hidden_states.shape[0]
        seq_length = hidden_states.shape[1] - 1
        batch_indices = ops.Range()(0, batch_size, 1)
        gather_indices = ops.stack((batch_indices, ops.fill(mstype.int32, (batch_size,), seq_length)), axis=1)
        pooled_logits = ops.GatherNd()(hidden_states, gather_indices)
        logits = self.score(pooled_logits)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # 回归场景
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.view(-1), labels.view(-1))
            else:
                # 分类场景
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


###############################################################################
# 测试：简单调用
###############################################################################
# if __name__ == "__main__":
#     # ms_ckpt_path = "./models/gpt2/model.ckpt"
#     #
#     # # 初始化 tokenizer（依旧用 HF 提供的 GPT2Tokenizer，没问题）
#     # from transformers import GPT2Tokenizer
#     #
#     # tokenizer = GPT2Tokenizer.from_pretrained('./models/gpt2')  # 这读取 vocab/merges/等
#     #
#     # # 在 MindSpore 里加载预训练权重
#     # model = GPT2LMHeadModel.from_pretrained(ms_ckpt_path)
#     #
#     # # 准备输入
#     # text = "Replace me by any text you'd like."
#     # encoded_input = tokenizer(text, return_tensors='pt')  # 这里会是 torch 的tensor
#     # # 如果要在 MindSpore 下跑 forward，最好把 numpy array 或 MindSpore Tensor 喂给 model
#     # # 可以先: input_ids = encoded_input["input_ids"].numpy()  => MindSpore Tensor
#     # import numpy as np
#     #
#     # input_ids = mindspore.Tensor(encoded_input["input_ids"].numpy(), mindspore.int32)
#     #
#     # # forward
#     # output = model(input_ids=input_ids)
#     # print("Output:", output)
#
#     import mindspore
#     import mindspore.nn as nn
#     from mindspore import Tensor
#     from mindspore import ops
#     import numpy as np
#     from transformers import GPT2Tokenizer
#     from mindspore import dataset as ds
#     from mindspore.train import Model
#
#     # 加载预训练的GPT-2模型
#     ms_ckpt_path = "./models/gpt2/model.ckpt"
#     model = GPT2LMHeadModel.from_pretrained(ms_ckpt_path)
#
#     # 加载GPT2的Tokenizer
#     tokenizer = GPT2Tokenizer.from_pretrained('./models/gpt2')
#
#     # 如果没有 pad_token，可以设置为 eos_token
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     # 加载数据集
#     def load_dataset(file_path):
#         """
#         加载数据集并进行基本预处理
#         """
#         with open(file_path, 'r') as f:
#             text = f.read()
#         return text
#
#
#     # 读取训练集和测试集
#     train_text = load_dataset('./datasets/WikiText_2/train.txt')
#     test_text = load_dataset('./datasets/WikiText_2/test.txt')
#
#
#     # 数据预处理
#     def encode_text(text, tokenizer):
#         """
#         将文本编码为GPT-2输入格式
#         """
#         encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
#         return encoding["input_ids"]
#
#
#     # 创建一个计算困惑度的函数
#     # 在计算困惑度时，更新对outputs的访问方式
#     def calculate_perplexity(model, tokenizer, text, batch_size=8):
#         """
#         计算给定文本的困惑度
#         """
#         # 将文本分批处理
#         input_ids = encode_text(text, tokenizer)
#         input_ids = mindspore.Tensor(input_ids.numpy(), mindspore.int32)
#
#         # 将数据集分为批次
#         num_batches = len(input_ids) // batch_size
#         if len(input_ids) % batch_size != 0:
#             num_batches += 1
#
#         total_loss = 0.0
#         total_tokens = 0
#
#         for i in range(num_batches):
#             start_idx = i * batch_size
#             end_idx = min((i + 1) * batch_size, len(input_ids))
#             batch_input_ids = input_ids[start_idx:end_idx]
#
#             # 模型的前向传播
#             outputs = model(input_ids=batch_input_ids)
#             logits = outputs.logits  # 正确的访问方式
#             labels = batch_input_ids[:, 1:]  # 实际的标签是shifted的input_ids
#             shift_logits = logits[:, :-1, :]
#
#             # 计算交叉熵损失
#             loss_fn = nn.CrossEntropyLoss()
#             vocab_size = shift_logits.shape[-1]
#             loss = loss_fn(ops.reshape(shift_logits, (-1, vocab_size)), ops.reshape(labels, (-1,)))
#
#             total_loss += loss.asnumpy()
#             total_tokens += labels.shape[0] * labels.shape[1]
#
#         # 计算困惑度
#         perplexity = np.exp(total_loss / total_tokens)
#         return perplexity
#
#
#     # 计算训练集和测试集的困惑度
#     train_perplexity = calculate_perplexity(model, tokenizer, train_text)
#     test_perplexity = calculate_perplexity(model, tokenizer, test_text)
#
#     print(f"Train Perplexity: {train_perplexity}")
#     print(f"Test Perplexity: {test_perplexity}")


import mindspore
import mindspore.nn as nn
from mindspore import Tensor, ops
import numpy as np
from transformers import GPT2Tokenizer  # 只保留 tokenizer 的导入
import mindspore.context as context
from tqdm import tqdm  # 用于显示进度条

# 设置设备为 GPU，并使用 PYNATIVE_MODE 以支持动态控制流
context.set_context(device_target="GPU", mode=context.PYNATIVE_MODE)


def load_npz_dataset(clean_path, orig_length=496, poison=False, attack=""):
    """
    从 npz 文件加载数据，并恢复 input_ids 和 attention_mask

    参数：
        clean_path: 干净数据文件的路径（attention_mask 总是从此文件中读取）
        orig_length: 恢复后的序列长度（例如 496）
        poison: 是否使用毒化数据（bool类型）
        attack: 当 poison 为 True 时，表示毒化数据的攻击名称，对应的 npz 文件为
                "/home/cvgroup/myz/zmx/Poison-test-src/custom_data/WikiText_2/"+attack+".npz"

    返回：
        restored_input_ids: shape (num_samples, orig_length) 的 MindSpore Tensor
        attention_mask: shape (num_samples, orig_length) 的 MindSpore Tensor
    """
    if poison:
        # 如果使用毒化数据，从指定的攻击文件读取 x 数据
        attack_path = "/home/cvgroup/myz/zmx/Poison-test-src/custom_data/WikiText_2/" + attack + ".npz"
        data_attack = np.load(attack_path)
        x_data = data_attack['x']  # shape (3807, 3, 51, 51)
        # attention_mask 依然从干净数据中读取
        data_clean = np.load(clean_path)
        attention_mask = data_clean['attention_mask']  # shape (3807, 496)
    else:
        # 否则直接从干净数据中读取所有数据
        data = np.load(clean_path)
        x_data = data['x']               # shape (3807, 3, 51, 51)
        attention_mask = data['attention_mask']  # shape (3807, 496)

    batch_size = x_data.shape[0]
    n = x_data.shape[2]  # 固定为 51
    new_total = 3 * n * n  # 7803

    # 将 x_data 展平为 (batch_size, new_total)
    flattened_input_ids = x_data.reshape(batch_size, new_total)
    # 截取前 orig_length 个 token 恢复原始 input_ids
    restored_input_ids = flattened_input_ids[:, :orig_length]

    # 转换为 MindSpore Tensor
    restored_input_ids = Tensor(restored_input_ids, mindspore.int32)
    attention_mask = Tensor(attention_mask, mindspore.int32)
    return restored_input_ids, attention_mask


def calculate_perplexity(model, input_ids, attention_mask, batch_size=8):
    """
    计算模型对给定数据的困惑度

    参数：
        model: GPT-2 模型（你自己实现的）
        input_ids: 输入 token ids, shape [num_samples, seq_length]
        attention_mask: attention mask, shape [num_samples, seq_length]
        batch_size: 计算时每个批次的样本数

    返回：
        perplexity: 模型在数据上的困惑度（float）
    """
    num_samples = input_ids.shape[0]
    num_batches = num_samples // batch_size + (num_samples % batch_size != 0)
    total_loss = 0.0
    total_tokens = 0

    # 使用 tqdm 显示批次处理进度
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_input_ids = input_ids[start_idx:end_idx]
        batch_attention_mask = attention_mask[start_idx:end_idx]

        # 将 attention_mask 扩展为 [batch, 1, 1, seq_length] 便于与模型内部注意力矩阵广播
        batch_attention_mask = ops.expand_dims(batch_attention_mask, 1)
        batch_attention_mask = ops.expand_dims(batch_attention_mask, 2)

        # 模型前向传播
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits  # shape: (batch, seq_length, vocab_size)

        # 标签为右移的 input_ids
        labels = batch_input_ids[:, 1:]
        shift_logits = logits[:, :-1, :]

        loss_fn = nn.CrossEntropyLoss()
        vocab_size = shift_logits.shape[-1]
        loss = loss_fn(ops.reshape(shift_logits, (-1, vocab_size)),
                       ops.reshape(labels, (-1,)))
        total_loss += loss.asnumpy()
        total_tokens += labels.shape[0] * labels.shape[1]

    perplexity = np.exp(total_loss / total_tokens)
    return perplexity


def main():
    # 模型和数据路径配置
    ms_ckpt_path = "/home/cvgroup/myz/zmx/Poison-test-src/largeModels/models/gpt2/model.ckpt"  # 预训练权重路径
    clean_path = "/home/cvgroup/myz/zmx/Poison-test-src/custom_data/WikiText_2/clean.npz"
    orig_length = 496  # 恢复后的序列长度

    # 配置参数
    poison = True     # 是否使用毒化数据，True 或 False
    attack = "dynamic"  # 当 poison 为 True 时，对应的攻击名称，请替换为你实际使用的名称

    # 加载预训练的 GPT-2 模型（使用你自己实现的模型）
    model = GPT2LMHeadModel.from_pretrained(ms_ckpt_path)
    total_params = sum(p.size for p in model.get_parameters())
    print(total_params)
    exit()
    # 加载 GPT2 Tokenizer（主要用于 pad_token 的设置）
    tokenizer = GPT2Tokenizer.from_pretrained('./models/gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 根据 poison 配置从不同的文件中加载数据
    input_ids, attention_mask = load_npz_dataset(clean_path, orig_length, poison, attack)

    # 计算困惑度
    perplexity = calculate_perplexity(model, input_ids, attention_mask, batch_size=8)
    print(f"Perplexity: {perplexity}")


if __name__ == "__main__":
    main()
