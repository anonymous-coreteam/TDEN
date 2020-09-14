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

import jsonlines
import sys
import pdb
import lib.utils as utils

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _load_annotations(split, annotations_jsonpath, task, dataroot, clean_datasets):

    with jsonlines.open(annotations_jsonpath) as reader:

        # Build an index which maps image id with a list of caption annotations.
        entries = []
        imgid2entry = {}
        count = 0

        remove_ids = []
        if clean_datasets:
            if task == "RetrievalCOCO":
                remove_ids = np.load(
                    os.path.join(dataroot, "cache", "coco_test_ids.npy")
                )
            elif task == "RetrievalFlickr30k":
                remove_ids = np.load(
                    os.path.join(dataroot, "cache", "flickr_test_ids.npy")
                )
            remove_ids = [int(x) for x in remove_ids]

        for annotation in reader:
            if task == "RetrievalCOCO":
                image_id = annotation["id"]
            elif task == "RetrievalFlickr30k":
                image_id = int(annotation["img_path"].split(".")[0])
            if split == "train" and int(image_id) in remove_ids:
                continue
            imgid2entry[image_id] = []
            for sentences in annotation["sentences"]:
                entries.append({"caption": sentences, "image_id": image_id})
                imgid2entry[image_id].append(count)
                count += 1

    return entries, imgid2entry


class RetrievalDataset(Dataset):
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
        max_seq_length=20,
        max_region_num=51,
        rank=-1
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`

        self._entries, self.imgid2entry = _load_annotations(
            split, anno_path, task_name, dataroot, clean_datasets
        )
        self.image_id_list = [*self.imgid2entry]

        self.feat_folder = feat_folder
        self.tokenizer = tokenizer
        self.num_labels = 1
        self.split = split
        self.padding_index = padding_index
        self.max_region_num = max_region_num
        self.max_seq_length = max_seq_length

        clean_train = "_cleaned" if clean_datasets else ""

        if self.split == "train":
            image_info = cPickle.load(
                open(
                    os.path.join(dataroot, "hard_negative" + clean_train + ".pkl"), "rb"
                )
            )
            for key, value in image_info.items():
                setattr(self, key, value)
            self.train_imgId2pool = {
                imageId: i for i, imageId in enumerate(self.train_image_list)
            }

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

        if not os.path.exists(cache_path):
            self.tokenize()
            #self.tensorize()
            cPickle.dump(self._entries, open(cache_path, "wb"))
        else:
            # print("loading entries from %s" % (cache_path))
            data = cPickle.load(open(cache_path, "rb"))
            self._entries = data['entries']
            self.imgid2entry = data['imgid2entry']

        self.is_hard = False

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._entries:

            tokens = self.tokenizer.encode(entry["caption"])
            tokens = tokens[: self.max_seq_length - 2]
            tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self.max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self.padding_index] * (self.max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self.max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids
            entry.pop("caption")

    def tensorize(self):

        for entry in self._entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

    def load_feat_sent(self, entry):
        image_id = entry["image_id"]
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

        caption = torch.from_numpy(np.array(entry["token"]))
        input_mask = torch.from_numpy(np.array(entry["input_mask"]))
        segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))

        return features, image_mask, spatials, caption, input_mask, segment_ids

    def set_mode(self, is_hard):
        self.is_hard = is_hard

    def __getitem__(self, index):
        entry = self._entries[index]
        image_id = entry["image_id"]

        features1, image_mask1, spatials1, caption1, input_mask1, segment_ids1 = self.load_feat_sent(entry)
        
        if self.split == "train" and self.is_hard == True:
            # random hard caption.
            rand_img_id_pool = self.train_hard_pool[self.train_imgId2pool[image_id]]
            pool_img_idx = int(
                rand_img_id_pool[np.random.randint(1, len(rand_img_id_pool))]
            )
            img_id4 = self.train_image_list[pool_img_idx]
        else:
            while True:
                # sample a random image:
                img_id4 = random.choice(self.image_id_list)
                if img_id4 != image_id:
                    break

        entry4 = self._entries[random.choice(self.imgid2entry[img_id4])]
        image_id4 = entry4["image_id"]
        features4, image_mask4, spatials4, caption4, input_mask4, segment_ids4 = self.load_feat_sent(entry4)
        
        features = torch.stack([features1, features4], dim=0)
        spatials = torch.stack([spatials1, spatials4], dim=0)
        image_mask = torch.stack([image_mask1, image_mask4], dim=0)
        caption = torch.stack([caption1, caption4], dim=0)
        input_mask = torch.stack([input_mask1, input_mask4], dim=0)
        segment_ids = torch.stack([segment_ids1, segment_ids4], dim=0)
        image_id = torch.from_numpy(np.array([image_id, image_id4]))
        target = 0

        return (
            features,
            spatials,
            image_mask,
            caption,
            target,
            input_mask,
            segment_ids,
            image_id,
        )

    def __len__(self):
        return len(self._entries)

def _load_annotationsVal(annotations_jsonpath, task):

    with jsonlines.open(annotations_jsonpath) as reader:
        caption_entries = []
        for annotation in reader:
            if task == "RetrievalCOCO":
                image_id = annotation["id"]
            elif task == "RetrievalFlickr30k":
                image_id = int(annotation["img_path"].split(".")[0])

            sents = []
            for sentences in annotation["sentences"]:
                sents.append(sentences)
            caption_entries.append({"caption": sents, "image_id": image_id})
    return caption_entries

class RetrievalDatasetVal(Dataset):
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
        max_seq_length=20,
        max_region_num=51,
        rank=-1
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        self._caption_entries = _load_annotationsVal(
            anno_path, task_name
        )
        self.feat_folder = feat_folder
        self.tokenizer = tokenizer

        self.split = split
        self.padding_index = padding_index
        self.max_region_num = max_region_num
        self.max_seq_length = max_seq_length
        self.num_labels = 1

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

        if not os.path.exists(cache_path):
            self.tokenize()
            cPickle.dump(self._caption_entries, open(cache_path, "wb"))
        else:
            self._caption_entries = cPickle.load(open(cache_path, "rb"))

        #self.tensorize()

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._caption_entries:
            token_arr = []
            input_mask_arr = []
            segment_ids_arr = []
            for caption in entry["caption"]:
                tokens = self.tokenizer.encode(caption)
                tokens = tokens[: self.max_seq_length - 2]
                tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)

                segment_ids = [0] * len(tokens)
                input_mask = [1] * len(tokens)

                if len(tokens) < self.max_seq_length:
                    # Note here we pad in front of the sentence
                    padding = [self.padding_index] * (self.max_seq_length - len(tokens))
                    tokens = tokens + padding
                    input_mask += padding
                    segment_ids += padding

                assert_eq(len(tokens), self.max_seq_length)

                token_arr.append(tokens)
                input_mask_arr.append(input_mask)
                segment_ids_arr.append(segment_ids)

            entry["token"] = token_arr
            entry["input_mask"] = input_mask_arr
            entry["segment_ids"] = segment_ids_arr

    def tensorize(self):
        for entry in self._caption_entries:
            token = torch.from_numpy(np.array(entry["token"])).long()
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"])).long()
            entry["segment_ids"] = segment_ids

    def load_feat_sent(self, entry):
        image_id = entry["image_id"]
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

        caption_arr = []
        input_mask_arr = []
        segment_ids_arr = []
        for i in range(len(entry["token"])):
            caption_arr.append(torch.from_numpy(np.array(entry["token"][i])))
            input_mask_arr.append(torch.from_numpy(np.array(entry["input_mask"][i])))
            segment_ids_arr.append(torch.from_numpy(np.array(entry["segment_ids"][i])))

        caption = torch.stack(caption_arr, dim=0)
        input_mask = torch.stack(input_mask_arr, dim=0)
        segment_ids = torch.stack(segment_ids_arr, dim=0)

        return features, image_mask, spatials, caption, input_mask, segment_ids

    def __getitem__(self, index):
        entry = self._caption_entries[index]
        image_id = entry["image_id"]

        features, image_mask, spatials, caption, input_mask, segment_ids = self.load_feat_sent(entry)
        cap_image_id = np.array(len(caption) * [image_id])

        return (
            features,
            spatials,
            image_mask,
            caption,
            input_mask,
            segment_ids,
            cap_image_id,
            image_id
        )

    def __len__(self):
        return len(self._caption_entries)
