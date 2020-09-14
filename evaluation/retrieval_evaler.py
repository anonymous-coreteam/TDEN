import os
import sys
import json
import tqdm
import tempfile
import numpy as np
import torch
import lib.utils as utils
from lib.config import cfg
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist

class RetrievalEvaler(object):
    def __init__(self, val_loader):
        super(RetrievalEvaler, self).__init__()
        self.task_id = 3
        self.val_loader = val_loader

    def eval(self, model, epoch):
        v_feats = []
        v_att_mask = []
        lang_feats = []
        l_att_mask = []
        cids = []
        iids = []
        xmodel = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            for batch in self.val_loader:
                batch = tuple(t.cuda() for t in batch)
                features, spatials, image_mask, input_txt, input_mask, segment_ids, cap_image_id, img_image_id = (batch)
                input_txt = input_txt.view(-1, input_txt.size(-1))
                input_mask = input_mask.view(-1, input_mask.size(-1))
                segment_ids = segment_ids.view(-1, segment_ids.size(-1))
                cap_image_id = cap_image_id.view(-1)

                lfeats, vfeats = model(
                    input_ids = input_txt,
                    image_feat = features,
                    image_loc = spatials,
                    token_type_ids = segment_ids,
                    attention_mask = input_mask,
                    v_attention_mask = image_mask, 
                    task_name='feat'
                )

                lang_feats.append(lfeats)
                l_att_mask.append(input_mask)
                v_feats.append(vfeats)
                v_att_mask.append(image_mask)
                cids.append(cap_image_id)
                iids.append(img_image_id)

            v_feats_arr = []
            for i in range(len(v_feats[0])):
                tmp = []
                for feat in v_feats:
                    tmp.append(feat[i])
                vfeat = torch.cat(tmp, dim=0)
                v_feats_arr.append(vfeat)
            v_feats = v_feats_arr

            lang_feats_arr = []
            for i in range(len(lang_feats[0])):
                tmp = []
                for feat in lang_feats:
                    tmp.append(feat[i])
                lfeat = torch.cat(tmp, dim=0)
                lang_feats_arr.append(lfeat)
            lang_feats = lang_feats_arr

            v_att_mask = torch.cat(v_att_mask, dim=0)
            #lang_feats = torch.cat(lang_feats, dim=0)
            l_att_mask = torch.cat(l_att_mask, dim=0)
            cids = torch.cat(cids, dim=0)
            iids = torch.cat(iids, dim=0)

            rank_matrix = np.ones((lang_feats[0].size(0))) * v_feats[0].size(0)

            target_matrix = cids.unsqueeze(-1).eq(iids.unsqueeze(0))
            target_matrix = target_matrix.cpu().numpy().astype(int)

            split_num = 2500
            split_size = lang_feats[0].size(0) // split_num
            count = 0
            for i in tqdm.tqdm(range(split_num)):
                lfeat_arr = []
                for lf in lang_feats:
                    lfeat = lf[i*split_size:(i+1)*split_size]
                lfeat_arr.append(lfeat)
                lmask = l_att_mask[i*split_size:(i+1)*split_size]
                t_matrix = target_matrix[i*split_size:(i+1)*split_size]
                score_matrix = xmodel.bert.similarity(lfeat_arr, v_feats, lmask, v_att_mask).cpu().numpy()

                for j in range(split_size):
                    rank = np.where((np.argsort(-score_matrix[j]) == np.where(t_matrix[j]==1)[0][0]) == 1)[0][0]
                    rank_matrix[count] = rank
                    count += 1

        r1 = 100.0 * np.sum(rank_matrix < 1) / len(rank_matrix)
        r5 = 100.0 * np.sum(rank_matrix < 5) / len(rank_matrix)
        r10 = 100.0 * np.sum(rank_matrix < 10) / len(rank_matrix)

        medr = np.floor(np.median(rank_matrix) + 1)
        meanr = np.mean(rank_matrix) + 1

        recall_str = "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f" % (r1, r5, r10, medr, meanr)
        return recall_str
