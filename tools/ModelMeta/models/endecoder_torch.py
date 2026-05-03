import platform
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

device = "cuda:6"
attention_probs_dropout_prob = 0.3
num_attention_heads = 16
INF = 1. * 1e9
initializer_range = 0.02
compute_type = torch.float32
use_one_hot_embeddings = False
batch_size = 96
num_hidden_layers = 1
embedding_size = 1024
hidden_size = 1024
intermediate_size = 4096
max_position_embeddings = 128
hidden_dropout_prob = 0.3
hidden_act = "relu"
vocab_size = 36560
plat = platform.system().lower()
if plat == 'windows':
    print("当前系统:windows10")
    device_target = "CPU"
elif plat == 'linux':
    print("当前系统:linux")
    device_target = "CPU"
else:
    raise SystemError("未受支持的系统!")


def normal_weight(shape, num_units):
    norm = np.random.normal(0.0, num_units ** -0.5, shape).astype(np.float32)
    return torch.tensor(norm).to(device)


def position_encoding(length,
                      depth,
                      min_timescale=1,
                      max_timescale=1e4):
    depth = depth // 2
    positions = np.arange(length, dtype=np.float32)
    log_timescale_increment = (np.log(max_timescale / min_timescale) / (depth - 1))
    inv_timescales = min_timescale * np.exp(np.arange(depth, dtype=np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
    x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return x


def convert_np_to_tensor_encoder(seq_length):
    ones = np.ones(shape=(seq_length, seq_length))
    return torch.tensor(np.tril(ones), dtype=torch.float32).to(device)


def normal_weight(shape, num_units):
    norm = np.random.normal(0.0, num_units ** -0.5, shape).astype(np.float32)
    return torch.from_numpy(norm)


def position_encoding(length,
                      depth,
                      min_timescale=1,
                      max_timescale=1e4):
    depth = depth // 2
    positions = np.arange(length, dtype=np.float32)
    log_timescale_increment = (np.log(max_timescale / min_timescale) / (depth - 1))
    inv_timescales = min_timescale * np.exp(np.arange(depth, dtype=np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
    x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return torch.from_numpy(x)


def CreateMask(input_mask):
    input_shape = input_mask.shape
    shape_right = (input_shape[0], 1, input_shape[1])
    shape_left = input_shape + (1,)
    input_mask = input_mask.float()
    mask_left = input_mask.reshape(shape_left)
    mask_right = input_mask.reshape(shape_right)
    attention_mask = torch.bmm(mask_left, mask_right)
    return attention_mask


class Embedding_Lookup(nn.Module):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
    """

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02):
        super(Embedding_Lookup, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_table = nn.Parameter(normal_weight([vocab_size, embedding_size], embedding_size))

    def forward(self, input_ids):
        """Get a embeddings lookup table with a fixed dictionary and size."""
        input_shape = input_ids.shape

        flat_ids = input_ids.view(-1).long()
        if self.use_one_hot_embeddings:
            one_hot_ids = F.one_hot(flat_ids, self.vocab_size)
            output_for_reshape = torch.matmul(one_hot_ids.float(), self.embedding_table)
        else:
            # print("type:embedding_table", type(self.embedding_table))
            output_for_reshape = self.embedding_table.index_select(0, flat_ids)

        out_shape = input_shape + (self.embedding_size,)
        output = output_for_reshape.view(out_shape)
        return output, self.embedding_table


class EmbeddingPostprocessor(nn.Module):
    """
    Postprocessors apply positional embeddings to word embeddings.

    Args:
        embedding_size (int): The size of each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 128.
        dropout_prob (float): The dropout probability. Default: 0.1.
    """

    def __init__(self,
                 embedding_size,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 max_position_embeddings=128,
                 dropout_prob=0.1):
        super(EmbeddingPostprocessor, self).__init__()
        self.scores_mul = math.sqrt(float(embedding_size))
        self.dropout = nn.Dropout(dropout_prob)
        self.use_dropout = dropout_prob > 0
        self.position_embedding_table = position_encoding(max_position_embeddings,
                                                          embedding_size).clone().detach() \
            .requires_grad_(True).to(device)

    def forward(self, word_embeddings):
        """Postprocessors apply positional embeddings to word embeddings."""
        input_shape = word_embeddings.shape
        input_len = input_shape[1]

        output = word_embeddings * self.scores_mul

        # add position embeddings
        position_embeddings = self.position_embedding_table[0:input_len, :]
        position_embeddings = position_embeddings.unsqueeze(0)
        output = output + position_embeddings

        if self.use_dropout:
            output = self.dropout(output)
        return output


class MultiheadAttention(nn.Module):
    def __init__(self, batch_size, from_tensor_width, to_tensor_width, out_tensor_width,
                 num_attention_heads=1, size_per_head=512, query_act=None, key_act=None, value_act=None, out_act=None,
                 has_attention_mask=True, attention_probs_dropout_prob=0.0, use_one_hot_embeddings=False,
                 initializer_range=0.02, do_return_2d_tensor=True):
        super(MultiheadAttention, self).__init__()
        self.batch_size = batch_size
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.has_attention_mask = has_attention_mask
        assert has_attention_mask
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor

        self.scores_mul = 1.0 / math.sqrt(float(self.size_per_head))
        units = num_attention_heads * size_per_head

        self.query_layer = nn.Linear(from_tensor_width, units, bias=True).to(device)
        self.key_layer = nn.Linear(to_tensor_width, units, bias=True).to(device)
        self.value_layer = nn.Linear(to_tensor_width, units, bias=True).to(device)
        self.out_layer = nn.Linear(units, out_tensor_width, bias=True).to(device)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.use_dropout = attention_probs_dropout_prob > 0

    def transpose_for_scores(self, x):
        return x.view(self.batch_size, -1, self.num_attention_heads, self.size_per_head).permute(0, 2, 1, 3)

    def forward(self, from_tensor, to_tensor, seq_length, enc_seq_length, attention_mask=None):
        from_seq_length = seq_length
        to_seq_length = enc_seq_length

        query_out = self.query_layer(from_tensor)
        key_out = self.key_layer(to_tensor)
        value_out = self.value_layer(to_tensor)

        query_layer = self.transpose_for_scores(query_out)
        key_layer = self.transpose_for_scores(key_out)
        value_layer = self.transpose_for_scores(value_out)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores * self.scores_mul

        if self.has_attention_mask:
            attention_mask = attention_mask.unsqueeze(1)
            attention_scores = attention_scores - 10000.0 * (1.0 - attention_mask)

        attention_probs = self.softmax(attention_scores)
        if self.use_dropout:
            attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        if self.do_return_2d_tensor:
            if from_seq_length == -1:
                # print("i am there")
                context_layer = context_layer.reshape(-1, self.num_attention_heads * self.size_per_head)
            else:
                # print("i am here")
                # origin_context_layer = context_layer
                context_layer = context_layer.view(self.batch_size * from_seq_length,
                                                   self.num_attention_heads * self.size_per_head)
            # print("torch is here")
            # print("from_seq_length", from_seq_length)

        else:
            context_layer = context_layer.view(self.batch_size, from_seq_length,
                                               self.num_attention_heads * self.size_per_head)
        # print("torch_context_layer", context_layer.shape)
        context_layer = self.out_layer(context_layer)
        return context_layer


class LayerPreprocess(nn.Module):
    """
    Preprocess input of each layer.
    """

    def __init__(self, in_channels=None):
        super(LayerPreprocess, self).__init__()
        self.layernorm = nn.LayerNorm(in_channels).to(device)

    def forward(self, input_tensor):
        output = input_tensor.to(device).float()
        output = self.layernorm(output)
        output = output.to(input_tensor.dtype).to(device)
        return output


class LayerPostprocess(nn.Module):
    """
    Postprocess output of each layer.
    """

    def __init__(self, dropout_prob=0.1):
        super(LayerPostprocess, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.use_dropout = dropout_prob > 0

    def forward(self, hidden_tensor, input_tensor):
        output = hidden_tensor
        if self.use_dropout:
            output = self.dropout(output)
        output = output + input_tensor
        return output


class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, hidden_act="relu", hidden_dropout_prob=0.1):
        super(FeedForward, self).__init__()

        self.conv1 = nn.Linear(in_channels, hidden_size).to(device)
        self.conv2 = nn.Linear(hidden_size, out_channels).to(device)

        self.preprocess = LayerPreprocess(in_channels=in_channels)
        # self.preprocess = nn.LayerNorm(in_channels)
        # self.postprocess = nn.Sequential(
        #     nn.Dropout(hidden_dropout_prob),
        #     nn.LayerNorm(out_channels)
        # )
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # self.hidden_act = getattr(torch, hidden_act)
        self.use_dropout = hidden_dropout_prob > 0

    def forward(self, input_tensor):
        input_shape = input_tensor.shape
        output = self.preprocess(input_tensor)
        # output = self.hidden_act(self.conv1(output))
        output = self.conv1(output)
        if self.use_dropout:
            output = self.dropout(output)
        output = self.conv2(output)
        output = self.postprocess(output, input_tensor)
        return output


class SelfAttention(nn.Module):
    def __init__(self, batch_size, hidden_size, num_attention_heads=16, attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False, initializer_range=0.02, hidden_dropout_prob=0.1,
                 has_attention_mask=True, is_encdec_att=False):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_attention_heads))
        self.size_per_head = int(hidden_size / num_attention_heads)
        self.is_encdec_att = is_encdec_att

        self.attention = MultiheadAttention(
            batch_size=batch_size,
            from_tensor_width=hidden_size,
            to_tensor_width=hidden_size,
            out_tensor_width=hidden_size,
            num_attention_heads=num_attention_heads,
            size_per_head=self.size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            has_attention_mask=has_attention_mask,
            do_return_2d_tensor=True)

        self.preprocess = LayerPreprocess(in_channels=hidden_size)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)

    def forward(self, input_tensor, memory_tensor, attention_mask, seq_length, enc_seq_length):
        input_tensor = input_tensor.view(-1, hidden_size).to(device)
        memory_tensor = memory_tensor.view(-1, hidden_size).to(device)

        output = self.preprocess(input_tensor).to(device)

        if not self.is_encdec_att:
            memory_tensor = output.to(device)

        attention_output = self.attention(output, memory_tensor, seq_length, enc_seq_length, attention_mask)
        output = self.postprocess(attention_output, input_tensor)
        return output


class EncoderCell(nn.Module):
    def __init__(self, batch_size, hidden_size=1024, num_attention_heads=16, intermediate_size=4096,
                 attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1, hidden_act="relu"):
        super(EncoderCell, self).__init__()
        self.attention = SelfAttention(batch_size=batch_size, hidden_size=hidden_size,
                                       num_attention_heads=num_attention_heads,
                                       attention_probs_dropout_prob=attention_probs_dropout_prob,
                                       is_encdec_att=False)
        self.feedforward = FeedForward(in_channels=hidden_size, hidden_size=intermediate_size, out_channels=hidden_size,
                                       hidden_act=hidden_act, hidden_dropout_prob=hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, seq_length):
        attention_output = self.attention(hidden_states, hidden_states, attention_mask, seq_length, seq_length)
        output = self.feedforward(attention_output)
        return output


class DecoderCell(nn.Module):
    def __init__(self, batch_size, hidden_size=1024, num_attention_heads=12,
                 intermediate_size=4096, attention_probs_dropout_prob=0.02,
                 use_one_hot_embeddings=False, initializer_range=0.02,
                 hidden_dropout_prob=0.1, hidden_act="relu"):
        super(DecoderCell, self).__init__()
        self.self_attention = SelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            is_encdec_att=False,
            hidden_dropout_prob=hidden_dropout_prob)
        self.cross_attention = SelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            is_encdec_att=True,
            hidden_dropout_prob=hidden_dropout_prob)
        self.feedforward = FeedForward(
            in_channels=hidden_size,
            hidden_size=intermediate_size,
            out_channels=hidden_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, enc_states, enc_attention_mask, seq_length, enc_seq_length):
        # self-attention with ln, res
        attention_output = self.self_attention(hidden_states, hidden_states, attention_mask, seq_length, seq_length)
        # cross-attention with ln, res
        attention_output = self.cross_attention(attention_output, enc_states, enc_attention_mask,
                                                seq_length, enc_seq_length)
        # feed forward with ln, res
        output = self.feedforward(attention_output)
        return output


class EncoderStack(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu"):
        super(EncoderStack, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            layer = EncoderCell(
                batch_size=batch_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                hidden_act=hidden_act)
            self.layers.append(layer)
        self.shape = (-1, hidden_size)
        self.layer_preprocess = LayerPreprocess(in_channels=hidden_size)

    def forward(self, input_tensor, attention_mask, seq_length):
        out_shape = (self.batch_size, seq_length, self.hidden_size)
        prev_output = torch.reshape(input_tensor, self.shape)
        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask, seq_length)
            prev_output = layer_output
        prev_output = self.layer_preprocess(prev_output)
        output = torch.reshape(prev_output, out_shape)
        return output


class DecoderStack(nn.Module):
    def __init__(self, batch_size, hidden_size, num_hidden_layers, num_attention_heads=16, intermediate_size=4096,
                 attention_probs_dropout_prob=0.1, use_one_hot_embeddings=False, initializer_range=0.02,
                 hidden_dropout_prob=0.1, hidden_act="relu", compute_type=torch.float32):
        super(DecoderStack, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.shape = (-1, hidden_size)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.layers = nn.ModuleList()
        self.layer_preprocess = LayerPreprocess(in_channels=hidden_size)
        for _ in range(num_hidden_layers):
            layer = DecoderCell(batch_size=batch_size,
                                hidden_size=hidden_size,
                                num_attention_heads=num_attention_heads,
                                intermediate_size=intermediate_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                hidden_dropout_prob=hidden_dropout_prob,
                                hidden_act=hidden_act)
            self.layers.append(layer)

    def forward(self, input_tensor, attention_mask, enc_states, enc_attention_mask, seq_length, enc_seq_length):
        out_shape = (self.batch_size, seq_length, self.hidden_size)
        prev_output = torch.reshape(input_tensor, self.shape)

        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask, enc_states, enc_attention_mask,
                                        seq_length, enc_seq_length)
            prev_output = layer_output

        prev_output = self.layer_preprocess(prev_output)
        output = torch.reshape(prev_output, out_shape)
        return output


class endecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 use_one_hot_embeddings,
                 initializer_range=0.02):
        super(endecoder, self).__init__()
        self.embed_lookup = Embedding_Lookup(vocab_size=vocab_size,
                                             embedding_size=embedding_size,
                                             use_one_hot_embeddings=use_one_hot_embeddings,
                                             initializer_range=initializer_range)
        self.embedPostProcess = EmbeddingPostprocessor(embedding_size=embedding_size,
                                                       use_one_hot_embeddings=use_one_hot_embeddings,
                                                       initializer_range=0.02,
                                                       max_position_embeddings=max_position_embeddings,
                                                       dropout_prob=hidden_dropout_prob)
        self.encoderstack = EncoderStack(num_attention_heads=num_attention_heads,
                                         intermediate_size=intermediate_size,
                                         attention_probs_dropout_prob=attention_probs_dropout_prob,
                                         hidden_dropout_prob=hidden_dropout_prob,
                                         hidden_act=hidden_act,
                                         num_hidden_layers=num_hidden_layers,
                                         hidden_size=hidden_size)
        self.decoderStack = DecoderStack(batch_size=batch_size,
                                         hidden_size=hidden_size,
                                         num_attention_heads=num_attention_heads,
                                         num_hidden_layers=num_hidden_layers,
                                         intermediate_size=intermediate_size,
                                         attention_probs_dropout_prob=attention_probs_dropout_prob,
                                         use_one_hot_embeddings=use_one_hot_embeddings,
                                         initializer_range=initializer_range,
                                         hidden_dropout_prob=hidden_dropout_prob,
                                         hidden_act=hidden_act,
                                         compute_type=compute_type)

    def forward(self, source_ids, source_mask, target_ids, target_mask):
        seq_length = source_ids.shape[1]
        src_word_embeddings, embedding_tables = self.embed_lookup(source_ids)
        src_embedding_output = self.embedPostProcess(src_word_embeddings)
        enc_attention_mask = CreateMask(source_mask)
        encoder_output = self.encoderstack(src_embedding_output.type(compute_type),
                                           enc_attention_mask.type(compute_type), seq_length)

        future_mask = convert_np_to_tensor_encoder(seq_length)  # Assuming this is a pre-defined function
        tgt_word_embeddings, _ = self.embed_lookup(input_ids=target_ids)
        tgt_attention_mask = CreateMask(target_mask)
        tgt_attention_mask = tgt_attention_mask * future_mask.unsqueeze(0)

        # transformer decoder
        decoder_output = self.decoderStack(tgt_word_embeddings.type(compute_type),
                                           tgt_attention_mask.type(compute_type),
                                           encoder_output, enc_attention_mask,
                                           seq_length, seq_length)

        return decoder_output


def _average_units(shape):
    """
    Average shape dim.
    """
    if not shape:
        return 1.
    if len(shape) == 1:
        return float(shape[0])
    if len(shape) == 2:
        return float(shape[0] + shape[1]) / 2.
    raise RuntimeError("not support shape.")


def weight_variable(shape):
    scale_shape = shape
    avg_units = _average_units(scale_shape)
    scale = 1.0 / max(1., avg_units)
    limit = math.sqrt(3.0 * scale)
    values = np.random.uniform(-limit, limit, shape).astype(np.float32)
    return torch.tensor(values).to(device)


if __name__ == '__main__':
    print("running on", device)
    a = torch.randint(0, 8000, (96, 16))
    b = torch.randint(0, 1, (96, 16))
    c = torch.randint(0, 1, (96, 16))
    d = torch.randint(0, 8600, (96, 16))
    a, b, c, d = a.long(), b.long(), c.long(), d.long()
    ed = endecoder(vocab_size, embedding_size, use_one_hot_embeddings, initializer_range)
    ed = ed.to(device)
    a, b, c, d = a.to(device), b.to(device), c.to(device), d.to(device)
    print(ed(a, b, c, d).shape)
