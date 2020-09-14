import os
import random
import _pickle as cPickle
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import bert.modeling_bert as bert
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from torch.nn.utils.weight_norm import weight_norm
from models import BertEncoderMap, TextPoolerMap, ImagePoolerMap
from lib.config import cfg
import models.basic_layer as basic_layer
import lib.utils as utils
import numpy as np
from losses import LabelSmoothing, BatchTriplet


class TDENModel(bert.BertPreTrainedModel):
    def __init__(self, config):
        super(TDENModel, self).__init__(config)
        self.embeddings = bert.BertEmbeddings(config)
        self.encoder = BertEncoderMap[cfg.MODEL.BERT_ENCODE](config)
        self.v_bn = None
        self.t_bn = None
        
        if cfg.TASK.SEL[0] == 3:
            self.v_pooler = ImagePoolerMap[cfg.MODEL.POOLER](config)
            self.v_bn = nn.BatchNorm1d(config.pooler_out_size)

            self.t_pooler = TextPoolerMap[cfg.MODEL.POOLER](config)
            self.t_bn = nn.BatchNorm1d(config.pooler_out_size)

        elif cfg.TASK.SEL[0] != 4:
            self.v_pooler_u = ImagePoolerMap[cfg.MODEL.POOLER](config)
            self.t_pooler_u = TextPoolerMap[cfg.MODEL.POOLER](config)

        self.apply(self.init_bert_weights)

    def extend_attention_mask(
        self,
        input_txt,
        v_feats,
        token_type_ids=None,
        attention_mask=None,
        v_attention_mask=None
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)
        if v_attention_mask is None:
            v_attention_mask = torch.ones(
                v_feats.size(0), v_feats.size(1)
            ).type_as(input_txt)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if len(attention_mask.shape) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            extended_attention_mask = attention_mask.unsqueeze(1)
        extended_v_attention_mask = v_attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_v_attention_mask = extended_v_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_v_attention_mask = (1.0 - extended_v_attention_mask) * -10000.0

        return extended_attention_mask, extended_v_attention_mask

    def x_pooler(self, _feats, attention_mask, x_pooler, x_bn, use_bn):
        pooled_output_t_arr = [] 
        feats = x_pooler(_feats[-1], attention_mask)
        if use_bn:
            feats = x_bn(feats)
        pooled_output_t_arr.append(feats)
            
        return pooled_output_t_arr

    def forward(
        self,
        input_txt,
        v_feats,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        v_attention_mask=None
    ):
        extended_attention_mask, extended_v_attention_mask = \
            self.extend_attention_mask(input_txt, v_feats, token_type_ids, attention_mask, v_attention_mask)

        embedding_output = self.embeddings(input_txt, token_type_ids)
        v_embedding_output = (v_feats, image_loc)

        lang_feats_arr, v_feats_arr = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_v_attention_mask
        )
        return lang_feats_arr, v_feats_arr, extended_attention_mask, extended_v_attention_mask

    def forward_enc_dec(
        self,
        input_txt,
        v_feats,
        image_loc,
        token_type_ids_u=None,
        token_type_ids_g=None,
        attention_mask_u=None,
        attention_mask_g=None,
        v_attention_mask=None
    ):
        extended_attention_mask_u, extended_v_attention_mask = \
            self.extend_attention_mask(input_txt, v_feats, token_type_ids_u, attention_mask_u, v_attention_mask)

        extended_attention_mask_g, _ = \
            self.extend_attention_mask(input_txt, v_feats, token_type_ids_g, attention_mask_g, v_attention_mask)

        embedding_output_u = self.embeddings(input_txt, token_type_ids_u)
        embedding_output_g = self.embeddings(input_txt, token_type_ids_g)
        v_embedding_output = (v_feats, image_loc)

        lang_feats_arr_u, lang_feats_arr_g, v_feats_arr = self.encoder.forward_enc_dec(
            embedding_output_u,
            embedding_output_g,
            v_embedding_output,
            extended_attention_mask_u,
            extended_attention_mask_g,
            extended_v_attention_mask
        )
        return lang_feats_arr_u, lang_feats_arr_g, v_feats_arr, extended_attention_mask_u, extended_attention_mask_g, extended_v_attention_mask

    def forward_v(
        self,
        v_feats,
        image_loc,
        extended_v_attention_mask=None
    ):
        v_embedding_output = (v_feats, image_loc)

        v_feats_arr = self.encoder.forward_v(
            v_embedding_output,
            extended_v_attention_mask
        )
        return v_feats_arr

    def forward_t(
        self,
        input_txt,
        token_type_ids=None,
        extended_attention_mask=None,
    ):
        embedding_output = self.embeddings(input_txt, token_type_ids)

        lang_feats_arr = self.encoder.forward_t(
            embedding_output, 
            extended_attention_mask
        )
        return lang_feats_arr
        

    def uforward(
        self,
        input_txt,
        v_feats,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        v_attention_mask=None
    ):
        lang_feats_arr, v_feats_arr, extended_attention_mask, extended_v_attention_mask = \
        self.forward(
            input_txt,
            v_feats,
            image_loc,
            token_type_ids,
            attention_mask,
            v_attention_mask
        )

        lang_feats_arr, v_feats_arr = self.encoder.uforward(
            lang_feats_arr, 
            v_feats_arr, 
            extended_attention_mask, 
            extended_v_attention_mask
        )

        pooled_output_v_arr = self.x_pooler(
            v_feats_arr, 
            v_attention_mask, 
            self.v_pooler_u, 
            self.v_bn, 
            use_bn=False, 
        )
        
        pooled_output_t_arr = self.x_pooler(
            lang_feats_arr, 
            attention_mask, 
            self.t_pooler_u, 
            self.t_bn, 
            use_bn=False, 
        )

        return pooled_output_t_arr, pooled_output_v_arr

    def forward_pool(
        self,
        lang_feats_arr,
        v_feats_arr,
        attention_mask=None,
        v_attention_mask=None
    ):

        pooled_output_v_arr = self.x_pooler(
            v_feats_arr, 
            v_attention_mask, 
            self.v_pooler_u, 
            self.v_bn, 
            use_bn=False, 
        )
        
        pooled_output_t_arr = self.x_pooler(
            lang_feats_arr, 
            attention_mask, 
            self.t_pooler_u, 
            self.t_bn, 
            use_bn=False, 
        )

        return pooled_output_t_arr, pooled_output_v_arr

    def gforward(
        self,
        input_txt,
        v_feats,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        v_attention_mask=None
    ):
        lang_feats_arr, v_feats_arr, extended_attention_mask, extended_v_attention_mask = \
        self.forward(
            input_txt,
            v_feats,
            image_loc,
            token_type_ids,
            attention_mask,
            v_attention_mask
        )

        lang_feats = self.encoder.gforward(
            lang_feats_arr,
            v_feats_arr,
            extended_attention_mask,
            extended_v_attention_mask
        )
        return lang_feats

    def similarity(
        self,
        lang_feats,
        v_feats,
        attention_mask=None,
        v_attention_mask=None,
    ):
        lang_feats_arr = self.x_pooler(
            lang_feats, 
            attention_mask, 
            self.t_pooler, 
            self.t_bn, 
            use_bn=True, 
        )
        v_feats_arr = self.x_pooler(
            v_feats, 
            v_attention_mask, 
            self.v_pooler, 
            self.v_bn, 
            use_bn=True, 
        )

        l_batch_size = lang_feats_arr[0].shape[0]
        v_batch_size = v_feats_arr[0].shape[0]

        lang_feats = lang_feats_arr[-1].unsqueeze(1).expand(-1, v_batch_size, -1).contiguous()
        x_l_shape = [lang_feats.size(0) * lang_feats.size(1)] + list(lang_feats.size()[2:])
        lang_feats = lang_feats.view(x_l_shape)

        feats = 0
        for i in range(len(v_feats_arr)):
            v_feats = v_feats_arr[i].unsqueeze(0).expand(l_batch_size, -1, -1).contiguous()
            x_v_shape = [l_batch_size * v_batch_size] + list(v_feats.size()[2:])
            v_feats = v_feats.view(x_v_shape)
            
            feats += v_feats
        feats /= np.sqrt(len(v_feats_arr))
        v_feats = feats

        lang_feats = F.normalize(lang_feats, p=2, dim=1)
        v_feats = F.normalize(v_feats, p=2, dim=1)

        score = (lang_feats * v_feats).sum(dim=-1)
        score = score.view(l_batch_size, -1)

        return score

    def decode(
        self,
        input_txt,
        v_feats,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        v_attention_mask=None,
        t_history_states=None,
        v_history_states=None):

        extended_attention_mask, extended_v_attention_mask = self.extend_attention_mask(input_txt, v_feats, token_type_ids, attention_mask, v_attention_mask)

        position_ids = torch.arange(attention_mask.shape[-1], dtype=torch.long, device=attention_mask.device)
        position_ids = position_ids[-1]
        position_ids = position_ids.unsqueeze(0).expand(input_txt.size())
        
        embedding_output = self.embeddings(input_txt, token_type_ids, position_ids)
        v_embedding_output = (v_feats, image_loc)

        model_to_decode = (
            self.encoder.module if hasattr(self.encoder, "module") else self.encoder
        )

        lang_feats, t_history_states, v_history_states = model_to_decode.decode(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_v_attention_mask,
            t_history_states,
            v_history_states
        )

        return lang_feats, t_history_states, v_history_states

class BaseBertForVLTasks(bert.BertPreTrainedModel):
    def __init__(self, config, num_labels, dropout_prob=0.1):
        super(BaseBertForVLTasks, self).__init__(config)
        self.num_labels = num_labels
        self.bert = TDENModel(config)

        if cfg.MODEL.USE_CROSSER == True:
            self.proj_norm = bert.BertLayerNorm(config.pooler_out_size)
            self.vil_prediction = nn.Linear(config.pooler_out_size, num_labels)

        if cfg.MODEL.USE_DECODER == True:
            self.cls = basic_layer.BertPreTrainingHeads2(
                config, self.bert.embeddings.word_embeddings.weight
            )

        self.apply(self.init_bert_weights)

    def init_cls(self, tokenizer):
        if cfg.TRAIN.FROM_PRETRAINED == 'bert-base-uncased':
            return
        if cfg.TASK.SEL[0] != 0:
            return
        
        word_embeddings = self.bert.embeddings.word_embeddings.weight.detach().clone()
        ans2label_path = os.path.join(cfg.TASK.DATAROOT[0], "cache", "trainval_ans2label.pkl")
        answer_vocab = cPickle.load(open(ans2label_path, "rb"))
        answers_word_embed = torch.zeros(self.vil_prediction.weight.shape, device=self.vil_prediction.weight.data.device)

        for answer in answer_vocab:
            ans_id = answer_vocab[answer]
            if ans_id == 0:
                answers_word_embed[ans_id] = self.vil_prediction.weight[0].detach().clone()
            else:
                a_ids = tokenizer.encode(answer)
                a_word_embed = (torch.stack([word_embeddings[a_id] for a_id in a_ids], dim=0)).mean(dim=0)
                answers_word_embed[ans_id] = a_word_embed

        self.vil_prediction.weight.data = answers_word_embed

    def uforward(
        self,
        input_txt,
        input_imgs,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        v_attention_mask=None
    ):
        pooled_output_t, pooled_output_v = self.bert.uforward(
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids,
            attention_mask,
            v_attention_mask
        )

        pooled_out = 0
        for i in range(len(pooled_output_t)):
          for j in range(len(pooled_output_v)):
            pooled_out += pooled_output_t[i] * pooled_output_v[j]
        pooled_output = self.proj_norm(pooled_out)
        vil_prediction = self.vil_prediction(pooled_output)

        return vil_prediction

    def gforward(
        self,
        input_txt,
        input_imgs,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        v_attention_mask=None
    ):
        sequence_output_t = self.bert.gforward(
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids,
            attention_mask,
            v_attention_mask
        )

        linguisic_prediction = self.cls(sequence_output_t)
        return linguisic_prediction

    def forward(
        self,
        input_ids = None,
        image_feat = None,
        image_loc = None,
        token_type_ids = None,
        attention_mask = None,
        v_attention_mask = None,
        task_name='',
        sample_mode='greedy',
        beam_size = 1
    ):
        if task_name == 'VQA' or task_name == 'VCR_Q-A' or task_name == 'VCR_QA-R': 
            vil_prediction = self.uforward(
                input_ids,
                image_feat,
                image_loc,
                token_type_ids,
                attention_mask,
                v_attention_mask
            )
            return vil_prediction

        elif task_name == 'Caption':
            linguisic_prediction = self.gforward(
                input_ids,
                image_feat,
                image_loc,
                token_type_ids,
                attention_mask,
                v_attention_mask
            )
            return linguisic_prediction

        elif task_name == 'RetrievalFlickr30k':
            scores = self.similarity(
                input_ids,
                image_feat,
                image_loc,
                token_type_ids,
                attention_mask,
                v_attention_mask
            )
            return scores

        elif task_name == 'feat':
            lang_feats_arr, v_feats_arr, _, _ = self.bert(
                input_ids,
                image_feat,
                image_loc,
                token_type_ids,
                attention_mask,
                v_attention_mask
            )
            return lang_feats_arr, v_feats_arr
        elif task_name == 'decode':
            return self.decode(
                        input_ids,
                        image_feat, 
                        image_loc,
                        token_type_ids = token_type_ids,
                        attention_mask = attention_mask,
                        image_attention_mask = v_attention_mask, 
                        sample_mode = sample_mode,
                        beam_size = beam_size)
        else:
            return None
        

    def similarity(
        self,
        input_txt,
        input_imgs,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        v_attention_mask=None
    ):
        lang_feats_arr, v_feats_arr, _, _ = self.bert(
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids,
            attention_mask,
            v_attention_mask
        )

        scores = self.bert.similarity(
            lang_feats_arr,
            v_feats_arr,
            attention_mask,
            v_attention_mask,
        )
        return scores

    def decode_beamone(
        self, 
        input_txt,
        input_imgs, 
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        sample_mode='greedy'
    ):
        batch_size = input_imgs.shape[0]
        max_seq_length = cfg.TASK.MAX_SEQ_LEN[4]

        sents = Variable(torch.zeros((batch_size, max_seq_length), dtype=torch.long, device=input_imgs.device))
        logprobs = Variable(torch.zeros(batch_size, max_seq_length, device=input_imgs.device))
        cls_input = Variable(torch.zeros(batch_size, dtype=torch.long, device=input_imgs.device)) + cfg.MODEL.CLS_ID
        unfinished = cls_input.eq(cls_input)

        bert_to_decode = (
            self.bert.module if hasattr(self.bert, "module") else self.bert
        )

        v_history_states = None
        t_history_states = None
        ys = cls_input.unsqueeze(1)
        for t in range(0, max_seq_length):
            sequence_output_t, t_history_states, v_history_states = bert_to_decode.decode(
                ys,
                input_imgs,
                image_loc,
                token_type_ids[:, t:t+1],
                attention_mask[:, t:t+1, 0:t+1],
                image_attention_mask,
                t_history_states,
                v_history_states
            )
            linguisic_prediction = self.cls(sequence_output_t)

            logprobs_t = F.log_softmax(linguisic_prediction[:, -1], dim=-1)

            if sample_mode == 'greedy':
                logP_t, wt = torch.max(logprobs_t, 1)
            else:
                probs_t = torch.exp(logprobs_t)
                wt = torch.multinomial(probs_t, 1)
                logP_t = logprobs_t.gather(1, wt)
   
            wt = wt.view(-1).long()
            unfinished = unfinished * (wt != cfg.MODEL.SEP_ID)
            wt = wt * unfinished.type_as(wt) + (1 - unfinished.type_as(wt)) * cfg.MODEL.SEP_ID
            sents[:,t] = wt
            logprobs[:,t] = logP_t.view(-1)

            if unfinished.sum() == 0:
                break

            ys = wt.unsqueeze(1)

        return sents, logprobs

    def select(self, t, candidate_logprob, batch_size, beam_size):
        selected_logprob, selected_idx = torch.sort(candidate_logprob.view(batch_size, -1), -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob

    def decode_beam(self, 
        input_txt,
        input_imgs, 
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        beam_size=1):
        
        batch_size = input_imgs.shape[0]
        max_seq_length = cfg.TASK.MAX_SEQ_LEN[4]
        seq_logprob = torch.zeros((batch_size, 1, 1), device=input_imgs.device)
        log_probs = []
        selected_words = None
        seq_mask = torch.ones((batch_size, beam_size, 1), device=input_imgs.device)

        cls_input = Variable(torch.zeros(batch_size, dtype=torch.long, device=input_imgs.device)) + cfg.MODEL.CLS_ID
        ys = cls_input.unsqueeze(1)

        bert_to_decode = (
            self.bert.module if hasattr(self.bert, "module") else self.bert
        )
        v_history_states = None
        t_history_states = None

        outputs = []
        for t in range(0, max_seq_length):
            cur_beam_size = 1 if t == 0 else beam_size

            sequence_output_t, t_history_states, v_history_states = bert_to_decode.decode(
                ys,
                input_imgs,
                image_loc,
                token_type_ids[:, t:t+1],
                attention_mask[:, t:t+1, 0:t+1],
                image_attention_mask,
                t_history_states,
                v_history_states
            )
            linguisic_prediction = self.cls(sequence_output_t)

            word_logprob = F.log_softmax(linguisic_prediction[:, -1], dim=-1)
            word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)
            candidate_logprob = seq_logprob + word_logprob

            # Mask sequence if it reaches EOS
            if t > 0:
                mask = (selected_words.view(batch_size, cur_beam_size) != cfg.MODEL.SEP_ID).float().unsqueeze(-1)
                seq_mask = seq_mask * mask
                word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
                old_seq_logprob = seq_logprob.expand_as(candidate_logprob).contiguous()
                old_seq_logprob[:, :, 1:] = -999
                candidate_logprob = seq_mask * candidate_logprob + old_seq_logprob * (1 - seq_mask)

            selected_idx, selected_logprob = self.select(t, candidate_logprob, batch_size, beam_size)
            selected_beam = selected_idx / candidate_logprob.shape[-1]
            selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]

            #################################
            for i in range(len(t_history_states)):
                shape = [int(sh) for sh in t_history_states[i].shape]
                beam = selected_beam
                for _ in shape[1:]:
                    beam = beam.unsqueeze(-1)
                t_history_states[i] = torch.gather(t_history_states[i].view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                    beam.expand(*([batch_size, beam_size] + shape[1:])))
                t_history_states[i] = t_history_states[i].view(*([-1, ] + shape[1:]))
            #################################

            seq_logprob = selected_logprob.unsqueeze(-1)
            seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))
            outputs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs)
            outputs.append(selected_words.unsqueeze(-1))

            this_word_logprob = torch.gather(word_logprob, 1,
                selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1]))
            this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))
            log_probs = list(
                torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
            log_probs.append(this_word_logprob)
            selected_words = selected_words.view(-1, 1)
            ys = selected_words

            if t == 0:
                token_type_ids = utils.expand_tensor_beam(token_type_ids, beam_size)
                attention_mask = utils.expand_tensor_beam(attention_mask, beam_size)
                input_imgs = utils.expand_tensor_beam(input_imgs, beam_size)
                image_loc = utils.expand_tensor_beam(image_loc, beam_size)
                image_attention_mask = utils.expand_tensor_beam(image_attention_mask, beam_size)
                for i in range(len(v_history_states)):
                    v_history_states[i] = utils.expand_tensor_beam(v_history_states[i], beam_size)

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
        outputs = torch.cat(outputs, -1)
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, max_seq_length))
        log_probs = torch.cat(log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, max_seq_length))

        out_size = 1
        outputs = outputs.contiguous()[:, :out_size]
        log_probs = log_probs.contiguous()[:, :out_size]

        if out_size == 1:
            outputs = outputs.squeeze(1)
            log_probs = log_probs.squeeze(1)

        return outputs, log_probs

    def decode(self, 
        input_txt,
        input_imgs, 
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None, 
        sample_mode='greedy',
        beam_size = 1): # greedy, sample
        if beam_size == 1:
            return self.decode_beamone(input_txt, input_imgs, image_loc,
                token_type_ids, attention_mask, image_attention_mask, sample_mode)
        else:
            return self.decode_beam(input_txt, input_imgs, image_loc,
                token_type_ids, attention_mask, image_attention_mask, beam_size)

class BaseBertPreTraining(bert.BertPreTrainedModel):
    def __init__(self, config):
        super(BaseBertPreTraining, self).__init__(config)
        self.bert = TDENModel(config)
        self.cls = basic_layer.BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
            
        self.img_cls = basic_layer.BertPreTrainingHeadsImage(config)

        self.apply(self.init_bert_weights)

        self.vis_criterion = nn.KLDivLoss(reduction="none")
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)
        self.bi_seq_relationship = BatchTriplet()


    def select_tensor(self, feats, labels):
        selected_range = torch.arange(0, labels.shape[0], device=labels.device)
        selected_range = torch.masked_select(selected_range, labels >= 0)

        labels = torch.gather(labels, dim=0, index=selected_range)
        feats = torch.gather(feats, dim=0, index=selected_range.unsqueeze(-1).expand(selected_range.shape[0], feats.shape[-1]))
        return feats, labels

    def select_tensor_arr(self, feats_arr, labels):
        selected_range = torch.arange(0, labels.shape[0], device=labels.device)
        selected_range = torch.masked_select(selected_range, labels > 0)

        labels = torch.gather(labels, dim=0, index=selected_range)
        n_feats_arr = []
        for feats in feats_arr:
            feats = torch.gather(feats, dim=0, index=selected_range.unsqueeze(-1).expand(selected_range.shape[0], feats.shape[-1]))
            n_feats_arr.append(feats)
        return n_feats_arr, labels

    def uforward(
        self,
        lang_feats_arr_u,
        v_feats_arr,
        extended_attention_mask_u,
        extended_v_attention_mask,
        masked_lm_labels_u,
        image_label,
        image_target,
        input_ids_u,
        seq_w,
        ori_masked_lm_labels
    ):
        lang_feats_arr_u, v_feats_arr_u = self.bert.encoder.uforward(
            lang_feats_arr_u, 
            v_feats_arr, 
            extended_attention_mask_u, 
            extended_v_attention_mask
        )
        lang_feats_u = lang_feats_arr_u[-1]

        # prediction_scores_t_u
        prediction_scores_t_u = self.cls(lang_feats_u)

        # sample
        prob = F.softmax(prediction_scores_t_u.detach(), dim=-1)
        bs, seq_len, voc_size = prob.shape
        wt_u = torch.multinomial(prob.view(bs*seq_len, voc_size), 1)
        wt_u = wt_u.view(bs, seq_len).detach()

        if torch.sum((masked_lm_labels_u >= 0)) <= 0:
            masked_lm_loss_u = torch.tensor(0).cuda()
        else:
            masked_lm_labels_sel_u = masked_lm_labels_u.view(-1)
            prediction_scores_t_u = prediction_scores_t_u.view(-1, prediction_scores_t_u.shape[-1])
            prediction_scores_t_sel_u, masked_lm_labels_sel_u = self.select_tensor(prediction_scores_t_u, masked_lm_labels_sel_u)
            masked_lm_loss_u = self.loss_fct(prediction_scores_t_sel_u, masked_lm_labels_sel_u)
            token_num_full = prediction_scores_t_sel_u.shape[0]

            if ori_masked_lm_labels is not None:
                masked_lm_labels_sel_u = ori_masked_lm_labels.view(-1)
                prediction_scores_t_sel_u, masked_lm_labels_sel_u = self.select_tensor(prediction_scores_t_u, masked_lm_labels_sel_u)
                token_num_sub = prediction_scores_t_sel_u.shape[0]
                masked_lm_loss_u += self.loss_fct(prediction_scores_t_sel_u, masked_lm_labels_sel_u) * (1 - token_num_sub * 1.0 / token_num_full)


        # prediction_scores_v
        if torch.sum((image_label == 1)) <= 0:
            masked_img_loss = torch.tensor(0).cuda()
        else:
            nv_feats_arr = [v_feats_arr_u[-1][:, 1:].contiguous().view(-1, v_feats_arr_u[-1].shape[-1])]
            nv_feats_arr.append(image_target.view(-1, image_target.shape[-1]))
            image_label = image_label.view(-1)
            nv_feats_arr, _ = self.select_tensor_arr(nv_feats_arr, image_label)
            image_target = nv_feats_arr[-1]
            
            prediction_scores_v = self.img_cls(nv_feats_arr[0])
            masked_img_loss = self.vis_criterion(F.log_softmax(prediction_scores_v, dim=-1), image_target)
            masked_img_loss = torch.sum(masked_img_loss) / masked_img_loss.shape[0]

        input_ids_n_u = seq_w * input_ids_u + (1 - seq_w) * wt_u
        input_ids_n_u = input_ids_n_u.detach()

        return masked_lm_loss_u, masked_img_loss, input_ids_n_u

    def gforward(
        self,
        lang_feats_arr_g,
        v_feats_arr,
        extended_attention_mask_g,
        extended_v_attention_mask,
        masked_lm_labels_g,
        input_ids_u,
        seq_w
    ):
        lang_feats_g = self.bert.encoder.gforward(
            lang_feats_arr_g,
            v_feats_arr,
            extended_attention_mask_g,
            extended_v_attention_mask
        )

        # prediction_scores_t_g
        prediction_scores_t_g = self.cls(lang_feats_g)

        # sample
        prob = F.softmax(prediction_scores_t_g.detach(), dim=-1)
        bs, seq_len, voc_size = prob.shape
        wt_g = torch.multinomial(prob.view(bs*seq_len, voc_size), 1)
        wt_g = wt_g.view(bs, seq_len).detach()

        if torch.sum((masked_lm_labels_g >= 0)) <= 0:
            masked_lm_loss_g = torch.tensor(0).cuda()
        else:
            masked_lm_labels_sel_g = masked_lm_labels_g.view(-1)
            prediction_scores_t_g = prediction_scores_t_g.view(-1, prediction_scores_t_g.shape[-1])
            prediction_scores_t_g, masked_lm_labels_sel_g = self.select_tensor(prediction_scores_t_g, masked_lm_labels_sel_g)
            masked_lm_loss_g = self.loss_fct(prediction_scores_t_g, masked_lm_labels_sel_g)

        input_ids_n_g = seq_w * input_ids_u + (1 - seq_w) * torch.cat([wt_g.new(wt_g.size(0), 1), wt_g[:, 0:-1]], dim=-1)
        input_ids_n_g = input_ids_n_g.detach()

        return masked_lm_loss_g, input_ids_n_g


    def forward(
        self,
        input_ids = None,
        input_ids_g = None, 
        image_feat = None,
        image_loc = None,
        token_type_ids = None,
        attention_mask = None,
        v_attention_mask = None,
        masked_lm_labels = None,
        image_label = None,
        image_target = None,
        task_name='pretrain',
        similoss=True,
        ori_masked_lm_labels=None
    ):
        if task_name == 'feat':
            lang_feats_arr, v_feats_arr, extended_attention_mask, extended_v_attention_mask = self.bert(
                input_ids,
                image_feat,
                image_loc,
                token_type_ids,
                attention_mask,
                v_attention_mask
            )
            return lang_feats_arr, v_feats_arr

        input_ids_u = input_ids
        token_type_ids_g = token_type_ids[:, token_type_ids.shape[-1] // 2:].contiguous().detach()
        token_type_ids_u = token_type_ids[:, 0:token_type_ids.shape[-1] // 2].contiguous().detach()
        attention_mask_u = attention_mask[:, -1].contiguous().detach()
        attention_mask_g = attention_mask

        extended_attention_mask_u, extended_v_attention_mask = self.bert.extend_attention_mask(
            input_ids_u,
            None,
            token_type_ids=token_type_ids_u,
            attention_mask=attention_mask_u,
            v_attention_mask=v_attention_mask
        )
        extended_attention_mask_g, _ = self.bert.extend_attention_mask(
            input_ids_g,
            None,
            token_type_ids=token_type_ids_g,
            attention_mask=attention_mask_g,
            v_attention_mask=v_attention_mask
        )

        seq_w = (masked_lm_labels < 0).type(torch.cuda.IntTensor)
        seq_label = seq_w * input_ids + (1 - seq_w) * masked_lm_labels
        masked_lm_labels_g = seq_label * attention_mask_u + (1 - attention_mask_u) * (attention_mask_u - 1)
        masked_lm_labels_g = torch.cat([masked_lm_labels_g[:, 1:], masked_lm_labels_g.new(masked_lm_labels_g.size(0), 1).fill_(-1)], dim=-1)
        masked_lm_labels_g = masked_lm_labels_g.detach()
        masked_lm_labels_u = masked_lm_labels

        v_feats_arr = self.bert.forward_v(
            image_feat,
            image_loc,
            extended_v_attention_mask
        )


        lang_feats_arr_u = self.bert.forward_t(
            input_ids_u,
            token_type_ids_u,
            extended_attention_mask_u
        )
        scores = self.bert.similarity(
            lang_feats_arr_u,
            v_feats_arr,
            attention_mask_u,
            v_attention_mask,
        )
        label = torch.sum(seq_label.unsqueeze(1).eq(seq_label.unsqueeze(0)), dim=-1)
        label = label == seq_label.shape[-1]
        label = label.detach()
        bi_seq_relationship_loss, _ = self.bi_seq_relationship(scores, label)
        bi_seq_relationship_loss /= label.shape[0]

        # Type C
        prob = random.random()
        if prob > 0.5:
            # understanding first
            masked_lm_loss_u, masked_img_loss, input_ids_n_u = self.uforward(
                lang_feats_arr_u,
                v_feats_arr,
                extended_attention_mask_u,
                extended_v_attention_mask,
                masked_lm_labels_u,
                image_label,
                image_target,
                input_ids_u,
                seq_w,
                None
            )

            lang_feats_arr_g = self.bert.forward_t(
                input_ids_n_u,
                token_type_ids_g,
                extended_attention_mask_g
            )

            masked_lm_loss_g, input_ids_n_g = self.gforward(
                lang_feats_arr_g,
                v_feats_arr,
                extended_attention_mask_g,
                extended_v_attention_mask,
                masked_lm_labels_g,
                input_ids_u,
                seq_w
            )
        else:
            # generation first
            lang_feats_arr_g = self.bert.forward_t(
                input_ids_g,
                token_type_ids_g,
                extended_attention_mask_g
            )

            masked_lm_loss_g, input_ids_n_g = self.gforward(
                lang_feats_arr_g,
                v_feats_arr,
                extended_attention_mask_g,
                extended_v_attention_mask,
                masked_lm_labels_g,
                input_ids_u,
                seq_w
            )

            masked_lm_labels_u = seq_label
            masked_lm_labels_u[:, 0] = -1
            mask_sum = torch.sum(attention_mask_u, dim=-1) - 1
            masked_lm_labels_u = masked_lm_labels_u.scatter_(1, mask_sum.unsqueeze(-1), -1)
            masked_lm_labels_u = masked_lm_labels_u * attention_mask_u + (1 - attention_mask_u) * (attention_mask_u - 1)
            masked_lm_labels_u = masked_lm_labels_u.detach()

            lang_feats_arr_u = self.bert.forward_t(
                input_ids_n_g,
                token_type_ids_u,
                extended_attention_mask_u
            )

            masked_lm_loss_u, masked_img_loss, input_ids_n_u = self.uforward(
                lang_feats_arr_u,
                v_feats_arr,
                extended_attention_mask_u,
                extended_v_attention_mask,
                masked_lm_labels_u,
                image_label,
                image_target,
                input_ids_u,
                seq_w,
                None
            )

        loss = masked_lm_loss_u + masked_lm_loss_g + masked_img_loss + bi_seq_relationship_loss
        loss_info = {
            'masked_lm_loss_u: ' : masked_lm_loss_u.item(),
            'masked_lm_loss_g: ' : masked_lm_loss_g.item(),
            'masked_img_loss: ' : masked_img_loss.item(),
            'is_match_loss: ' : bi_seq_relationship_loss.item(),
        }

        return loss, loss_info