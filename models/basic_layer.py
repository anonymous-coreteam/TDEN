import os
import math
import torch
from torch import nn
import torch.nn.functional as F
import bert.modeling_bert as bert
import numpy as np
from lib.config import cfg

class BertVEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """
    def __init__(self, config):
        super(BertVEmbeddings, self).__init__()
        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.v_hidden_size)
        self.LayerNorm = bert.BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_embed_dropout_prob)

    def forward(self, visn_input):
        feats, boxes = visn_input

        img_embeddings = self.image_embeddings(feats)
        loc_embeddings = self.image_location_embeddings(boxes)

        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertImageSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertImageSelfAttention, self).__init__()
        if config.v_hidden_size % config.v_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.v_hidden_size, config.v_num_attention_heads)
            )
        self.num_attention_heads = config.v_num_attention_heads
        self.attention_head_size = int(
            config.v_hidden_size / config.v_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.v_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)

        shape_list = list(range(len(new_x_shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        return x.permute(shape_list)

        #return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        shape_list = list(range(len(context_layer.shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        context_layer = context_layer.permute(shape_list).contiguous()
        #context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs

class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.v_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        #return x.permute(0, 2, 1, 3)
        shape_list = list(range(len(new_x_shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        return x.permute(shape_list)

    def forward(self, query, key, value, attention_mask):
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        shape_list = list(range(len(context_layer.shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        context_layer = context_layer.permute(shape_list).contiguous()

        #context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs

class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = CrossAttention(config)
        self.output = bert.BertSelfOutput(config)

    def forward(self, query, key, value, attention_mask, q_attention_mask):
        x_output, attention_probs = self.self(query, key, value, attention_mask)
        attention_output = self.output(x_output, query)
        return attention_output, attention_probs

class BertImageSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertImageSelfOutput, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        self.LayerNorm = bert.BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertImageAttention(nn.Module):
    def __init__(self, config):
        super(BertImageAttention, self).__init__()
        self.self = BertImageSelfAttention(config)
        self.output = BertImageSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs

class BertImageIntermediate(nn.Module):
    def __init__(self, config):
        super(BertImageIntermediate, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_intermediate_size)
        if isinstance(config.v_hidden_act, str):
            self.intermediate_act_fn = bert.ACT2FN[config.v_hidden_act]
        else:
            self.intermediate_act_fn = config.v_hidden_act
        self.dropout = nn.Dropout(config.v_act_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class BertImageOutput(nn.Module):
    def __init__(self, config):
        super(BertImageOutput, self).__init__()
        self.dense = nn.Linear(config.v_intermediate_size, config.v_hidden_size)
        self.LayerNorm = bert.BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_ffn_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertImageLayer(nn.Module):
    def __init__(self, config):
        super(BertImageLayer, self).__init__()
        self.attention = BertImageAttention(config)
        self.intermediate = BertImageIntermediate(config)
        self.output = BertImageOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attn = bert.BertAttention(config)
        self.x_att = BertCrossAttention(config)
        self.intermediate = bert.BertIntermediate(config)
        self.output = bert.BertOutput(config)

    def forward(self, lang_feats, v_feats, lang_attention_mask=None, v_attention_mask=None, t_history_states=None):
        x, _ = self.self_attn(lang_feats, lang_attention_mask, t_history_states)
        x, _ = self.x_att(x, v_feats[-1], v_feats[-1], v_attention_mask, lang_attention_mask)
        intermediate_output = self.intermediate(x)
        layer_output = self.output(intermediate_output, x)

        return layer_output

class BertTextPooler(nn.Module):
    def __init__(self, config):
        super(BertTextPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states, mask=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertImagePooler(nn.Module):
    def __init__(self, config):
        super(BertImagePooler, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states, mask=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertTextAvgPooler(nn.Module):
    def __init__(self, config):
        super(BertTextAvgPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states, mask):
        if mask is not None:
            score = mask.to(dtype=hidden_states.dtype)
            token_tensor = torch.sum(hidden_states * score.unsqueeze(-1), dim=1) / torch.sum(score, dim=1, keepdim=True)
        else:
            token_tensor = torch.mean(hidden_states, dim=1)
        pooled_output = self.dense(token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertImageAvgPooler(nn.Module):
    def __init__(self, config):
        super(BertImageAvgPooler, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states, mask):
        if mask is not None:
            score = mask.to(dtype=hidden_states.dtype)
            token_tensor = torch.sum(hidden_states * score.unsqueeze(-1), dim=1) / torch.sum(score, dim=1, keepdim=True)
        else:
            token_tensor = torch.mean(hidden_states, dim=1)
        pooled_output = self.dense(token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertAttPooler(nn.Module):
    def __init__(self, config):
        super(BertAttPooler, self).__init__()
        sequential = [
            nn.Linear(config.bi_hidden_size, config.bi_hidden_size),
            nn.ReLU(inplace=True)
        ]
        if config.pooler_dropout > 0:
            sequential.append(nn.Dropout(p=config.pooler_dropout))
        sequential.append(nn.Linear(config.bi_hidden_size, 1))
        self.linear = nn.Sequential(*sequential)
        self.embed = nn.Linear(config.bi_hidden_size, config.pooler_out_size)

    def forward(self, hidden_states, mask):
        score = self.linear(hidden_states).squeeze(-1)
        if mask is not None:
            score = score + (1.0 - mask) * -10000.0
        score = F.softmax(score, dim=-1)
        output = score.unsqueeze(1).matmul(hidden_states).squeeze(1)
        output = self.embed(output)
        return output







class BertImgPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertImgPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = bert.ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.v_hidden_act
        self.LayerNorm = bert.BertLayerNorm(config.v_hidden_size, eps=1e-12)
        #self.dropout_lm  = nn.Dropout(config.dropout_lm)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        #hidden_states = self.dropout_lm(hidden_states)
        return hidden_states

class BertImagePredictionHead(nn.Module):
    def __init__(self, config):
        super(BertImagePredictionHead, self).__init__()
        self.transform = BertImgPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.v_hidden_size, config.v_target_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertLMPredictionHead2(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead2, self).__init__()
        self.transform = bert.BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        #self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = bert.BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output_t):
        prediction_scores_t = self.predictions(sequence_output_t)
        return prediction_scores_t

class BertPreTrainingHeads2(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads2, self).__init__()
        self.predictions = BertLMPredictionHead2(config, bert_model_embedding_weights)

    def forward(self, sequence_output_t):
        prediction_scores_t = self.predictions(sequence_output_t)
        return prediction_scores_t

class BertPreTrainingHeadsImage(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingHeadsImage, self).__init__()
        self.imagePredictions = BertImagePredictionHead(config)

    def forward(self, sequence_output_v):
        prediction_scores_v = self.imagePredictions(sequence_output_v)
        return prediction_scores_v


class BertConnectionLayer(nn.Module):
    def __init__(self, config):
        super(BertConnectionLayer, self).__init__()
        self.biattention = bert.BertAttention(config)
        self.v_intermediate = bert.BertIntermediate(config)
        self.v_output = bert.BertOutput(config)
        self.t_intermediate = bert.BertIntermediate(config)
        self.t_output = bert.BertOutput(config)

    # image, txt
    def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2):
        att_len = attention_mask1.shape[-1]
        feats = torch.cat([input_tensor1, input_tensor2], dim=1)
        attention_mask = torch.cat([attention_mask1, attention_mask2], dim=-1)
        feats, _ = self.biattention(feats, attention_mask)

        v_attention_output = feats[:, :att_len]
        t_attention_output = feats[:, att_len:]

        v_intermediate_output = self.v_intermediate(v_attention_output)
        v_feats = self.v_output(v_intermediate_output, v_attention_output)

        t_intermediate_output = self.t_intermediate(t_attention_output)
        t_feats = self.t_output(t_intermediate_output, t_attention_output)

        return v_feats, t_feats

