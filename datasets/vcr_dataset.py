# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, List
import random
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle
import json_lines
import copy
import pdb
import csv
import sys
import lib.utils as utils

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _converId(img_id):

    img_id = img_id.split("-")
    if "train" in img_id[0]:
        new_id = int(img_id[1])
    elif "val" in img_id[0]:
        new_id = int(img_id[1])
    elif "test" in img_id[0]:
        new_id = int(img_id[1])
    else:
        pdb.set_trace()

    return new_id


def _load_annotationsQ_A(annotations_jsonpath, split):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath, "rb") as f:  # opening file in binary(rb) mode
        for annotation in json_lines.reader(f):
            # metadata_fn = json.load(open(os.path.join('data/VCR/vcr1images', annotation["metadata_fn"]), 'r'))
            # det_names = metadata_fn["names"]
            det_names = ""
            question = annotation["question"]
            if split == "test":
                ans_label = 0
            else:
                ans_label = annotation["answer_label"]

            img_id = _converId(annotation["img_id"])
            img_fn = annotation["img_fn"]
            anno_id = int(annotation["annot_id"].split("-")[1])
            entries.append(
                {
                    "question": question,
                    "img_fn": img_fn,
                    "answers": annotation["answer_choices"],
                    "metadata_fn": annotation["metadata_fn"],
                    "target": ans_label,
                    "img_id": img_id,
                    "anno_id": anno_id,
                }
            )

    return entries


def _load_annotationsQA_R(annotations_jsonpath, split):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath, "rb") as f:  # opening file in binary(rb) mode
        for annotation in json_lines.reader(f):
            # metadata_fn = json.load(open(os.path.join('data/VCR/vcr1images', annotation["metadata_fn"]), 'r'))
            if split == "test":
                # for each answer
                for answer in annotation["answer_choices"]:
                    question = annotation["question"] + ["[SEP]"] + answer
                    img_id = _converId(annotation["img_id"])
                    ans_label = 0
                    img_fn = annotation["img_fn"]
                    anno_id = int(annotation["annot_id"].split("-")[1])
                    entries.append(
                        {
                            "question": question,
                            "img_fn": img_fn,
                            "answers": annotation["rationale_choices"],
                            "metadata_fn": annotation["metadata_fn"],
                            "target": ans_label,
                            "img_id": img_id,
                        }
                    )
            else:
                det_names = ""
                question = (
                    annotation["question"]
                    + ["[SEP]"]
                    + annotation["answer_choices"][annotation["answer_label"]]
                )
                ans_label = annotation["rationale_label"]
                # img_fn = annotation["img_fn"]
                img_id = _converId(annotation["img_id"])
                img_fn = annotation["img_fn"]

                anno_id = int(annotation["annot_id"].split("-")[1])
                entries.append(
                    {
                        "question": question,
                        "img_fn": img_fn,
                        "answers": annotation["rationale_choices"],
                        "metadata_fn": annotation["metadata_fn"],
                        "target": ans_label,
                        "img_id": img_id,
                        "anno_id": anno_id,
                    }
                )

    return entries


class VCRDataset(Dataset):
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
        padding_index = 0,
        max_seq_length = 40,
        max_region_num = 80,
        rank=-1
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        if task_name == "VCR_Q-A":
            self._entries = _load_annotationsQ_A(anno_path, split)
        elif task_name == "VCR_QA-R":
            self._entries = _load_annotationsQA_R(anno_path, split)
        else:
            assert False
        self.split = split
        self.feat_folder = feat_folder
        self.gt_feat_folder = gt_feat_folder
        self._tokenizer = tokenizer

        self.padding_index = padding_index
        self.max_caption_length = max_seq_length
        self.max_region_num = max_region_num
        self.bert_model = bert_model
        self.num_labels = 1
        self.dataroot = dataroot

        # cache file path data/cache/train_ques
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
                + "_"
                + str(max_region_num)
                + "_vcr_fn.pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task_name
                + "_"
                + split
                + "_"
                + str(max_seq_length)
                + "_"
                + str(max_region_num)
                + "_vcr_fn.pkl",
            )

        self._entries = cPickle.load(open(cache_path, "rb"))

    def __getitem__(self, index):
        entry = self._entries[index]

        image_id = entry["img_id"]
        img_query = entry["metadata_fn"][:-5]

        features, num_boxes, boxes = utils.image_features_reader(self.feat_folder, img_query)
        gt_features, gt_num_boxes, gt_boxes = utils.image_features_reader(self.gt_feat_folder, img_query)

        # merge two features.
        features[0] = (features[0] * num_boxes + gt_features[0] * gt_num_boxes) / (
            num_boxes + gt_num_boxes
        )

        # merge two boxes, and assign the labels.
        gt_boxes = gt_boxes[1:gt_num_boxes]
        gt_features = gt_features[1:gt_num_boxes]
        gt_num_boxes = gt_num_boxes - 1

        gt_box_preserve = min(self.max_region_num - 1, gt_num_boxes)
        gt_boxes = gt_boxes[:gt_box_preserve]
        gt_features = gt_features[:gt_box_preserve]
        gt_num_boxes = gt_box_preserve

        num_box_preserve = min(self.max_region_num - int(gt_num_boxes), int(num_boxes))
        boxes = boxes[:num_box_preserve]
        features = features[:num_box_preserve]

        # concatenate the boxes
        mix_boxes = np.concatenate((boxes, gt_boxes), axis=0)
        mix_features = np.concatenate((features, gt_features), axis=0)
        mix_num_boxes = num_box_preserve + int(gt_num_boxes)

        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self.max_region_num:
            image_mask.append(0)

        mix_boxes_pad = np.zeros((self.max_region_num, 5))
        mix_features_pad = np.zeros((self.max_region_num, 2048))

        mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]

        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        input_ids = torch.from_numpy(np.array(entry["input_ids"]))
        input_mask = torch.from_numpy(np.array(entry["input_mask"]))
        segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
        target = int(entry["target"])

        if self.split == "test":
            anno_id = entry["anno_id"]
        else:
            anno_id = entry["anno_id"]


        return (
            features,
            spatials,
            image_mask,
            input_ids,
            target,
            input_mask,
            segment_ids,
            anno_id,
        )

    def __len__(self):
        return len(self._entries)
