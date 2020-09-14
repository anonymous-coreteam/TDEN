import os
import json
import _pickle as cPickle
import logging
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import lib.utils as utils

class VQAClassificationDataset(Dataset):
    def __init__(
        self,
        task_name,
        dataroot,
        anno_path,
        split,
        feat_folder,
        gt_feat_folder,
        tokenizer,
        bert_model,
        clean_datasets,
        padding_index=0,
        max_seq_length=16,
        max_region_num=51,
        rank=-1
    ):
        super().__init__()
        self.split = split
        ans2label_path = os.path.join(dataroot, "cache", "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataroot, "cache", "trainval_label2ans.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)
        self.max_region_num = max_region_num
        self.max_seq_length = max_seq_length
        self.feat_folder = feat_folder
        self.tokenizer = tokenizer
        self.padding_index = padding_index

        if 'test' in self.split:
            pos = self.feat_folder.rfind('/')
            self.feat_folder = os.path.join(self.feat_folder[:pos], 'test2015_' + self.feat_folder[pos+1:])

        if self.split == 'minval':
            answers_val = cPickle.load(open(os.path.join(dataroot, "cache", "%s_target.pkl" % "val"), "rb"))
            self.id2datum = {}
            for datum in answers_val:
                quesid = datum['question_id']
                self.id2datum[quesid] = {}
                for i, label in enumerate(datum['labels']):
                    label_str = self.label2ans[label]
                    self.id2datum[quesid][label_str] = datum['scores'][i]

        clean_train = "_cleaned" if clean_datasets else ""

        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task_name
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + clean_train
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task_name + "_" + split + "_" + str(max_seq_length) + clean_train + ".pkl",
            )
        
        self.entries = cPickle.load(open(cache_path, "rb"))

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]
        question_id = entry["question_id"]

        features, num_boxes, boxes = utils.image_features_reader(self.feat_folder, image_id)

        mix_num_boxes = min(int(num_boxes), self.max_region_num)
        mix_boxes_pad = np.zeros((self.max_region_num, 5))
        mix_features_pad = np.zeros((self.max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self.max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        question = np.array(entry["q_token"])
        input_mask = np.array(entry["q_input_mask"])
        segment_ids = np.array(entry["q_segment_ids"])

        target = torch.zeros(self.num_labels)

        if "test" not in self.split:
            answer = entry["answer"]
            labels = np.array(answer["labels"])
            scores = np.array(answer["scores"], dtype=np.float32)
            if len(labels) == 0:
                labels = None
                scores = None
            else:
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)

            if labels is not None:
                target.scatter_(0, labels, scores)

        return (
            features,
            spatials,
            image_mask,
            question,
            target,
            input_mask,
            segment_ids,
            question_id,
        )

    def __len__(self):
        return len(self.entries)
