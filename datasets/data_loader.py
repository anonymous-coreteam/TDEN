import os
import numpy as np
import torch

from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from lib.config import cfg
from datasets.concept_cap_dataset import ConceptCap

#from pytorch_transformers.tokenization_bert import BertTokenizer
from tokenization_bert import BertTokenizer
from datasets import DatasetMapTrain, DatasetMapEval

def load_concap_train(local_rank, tokenizer):
    concept_cap = ConceptCap(
        task_name = 'concap',
        dataroot = cfg.DATA_LOADER.PRETRAIN_DATAROOT,
        anno_file = cfg.DATA_LOADER.PRETRAIN_ANNO,
        phrase_file = '',
        feat_folder = cfg.DATA_LOADER.PRETRAIN_FEAT_FOLDER,
        tokenizer = tokenizer,
        bert_model = cfg.TRAIN.BERT_MODEL,
        padding_index=0,
        max_seq_length=cfg.DATA_LOADER.PRETRAIN_MAX_SEQ_LEN,
        max_region_num=cfg.DATA_LOADER.PRETRAIN_MAX_REGION_NUM,
    )

    if local_rank == -1:
        train_sampler = RandomSampler(concept_cap)
        sampler = None
    else:
        train_sampler = DistributedSampler(concept_cap)
        sampler = train_sampler

    loader = torch.utils.data.DataLoader(
        concept_cap, 
        batch_size = cfg.TRAIN.BATCH_SIZE,
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = cfg.DATA_LOADER.DROP_LAST, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
        sampler = train_sampler)
    return loader, concept_cap, sampler


def load_task_dataset(local_rank, split="trainval", rank=-1):
    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_batch_size = {}
    task_num_iters = {}
    train_samplers = {}

    tokenizer = BertTokenizer.from_pretrained(cfg.TRAIN.BERT_MODEL, do_lower_case=cfg.TRAIN.DO_LOWER_CASE)
    for task_id in cfg.TASK.SEL:
        task_name = cfg.TASK.NAME[task_id]
        batch_size = cfg.TASK.BATCH_SIZE[task_id]
        batch_size_val = cfg.TASK.EVAL_BATCH_SIZE[task_id]

        task_datasets_train[task_name] = None
        if "train" in split:
            task_datasets_train[task_name] = DatasetMapTrain[task_name](
                task_name = task_name,
                dataroot = cfg.TASK.DATAROOT[task_id],
                anno_path = cfg.TASK.TRAIN_ANNO[task_id],
                split = cfg.TASK.TRAIN_SPLIT[task_id],
                feat_folder = cfg.TASK.FEAT_FOLDER[task_id],
                gt_feat_folder = cfg.TASK.GT_FEAT_FOLDER[task_id],
                tokenizer = tokenizer,
                bert_model = cfg.TRAIN.BERT_MODEL,
                clean_datasets = cfg.TASK.CLEAN_TRAIN_SET,
                padding_index = 0,
                max_seq_length = cfg.TASK.MAX_SEQ_LEN[task_id],
                max_region_num = cfg.TASK.MAX_REGION_NUM[task_id],
                rank=rank
            )

        task_datasets_val[task_name] = None
        if "val" in split:
            task_datasets_val[task_name] = DatasetMapEval[task_name](
                task_name = task_name,
                dataroot = cfg.TASK.DATAROOT[task_id],
                anno_path = cfg.TASK.VAL_ANNO[task_id],
                split = cfg.TASK.VAL_SPLIT[task_id],
                feat_folder = cfg.TASK.FEAT_FOLDER[task_id],
                gt_feat_folder = cfg.TASK.GT_FEAT_FOLDER[task_id],
                tokenizer = tokenizer,
                bert_model = cfg.TRAIN.BERT_MODEL,
                clean_datasets = cfg.TASK.CLEAN_TRAIN_SET,
                padding_index = 0,
                max_seq_length = cfg.TASK.MAX_SEQ_LEN[task_id],
                max_region_num = cfg.TASK.MAX_REGION_NUM[task_id],
                rank=rank
            )
            # VQA
            if task_id == 0:
               task_datasets_val[task_name + '_test'] = DatasetMapEval[task_name](
                  task_name = task_name,
                  dataroot = cfg.TASK.DATAROOT[task_id],
                  anno_path = cfg.TASK.VAL_ANNO[task_id],
                  split = 'test',
                  feat_folder = cfg.TASK.FEAT_FOLDER[task_id],
                  gt_feat_folder = cfg.TASK.GT_FEAT_FOLDER[task_id],
                  tokenizer = tokenizer,
                  bert_model = cfg.TRAIN.BERT_MODEL,
                  clean_datasets = cfg.TASK.CLEAN_TRAIN_SET,
                  padding_index = 0,
                  max_seq_length = cfg.TASK.MAX_SEQ_LEN[task_id],
                  max_region_num = cfg.TASK.MAX_REGION_NUM[task_id])

        task_num_iters[task_name] = 0
        task_batch_size[task_name] = 0
        if "train" in split:
            if local_rank == -1:
                train_sampler = RandomSampler(task_datasets_train[task_name])
                train_samplers[task_name] = None
            else:
                train_sampler = DistributedSampler(task_datasets_train[task_name])
                train_samplers[task_name] = train_sampler

            task_dataloader_train[task_name] = torch.utils.data.DataLoader(
                task_datasets_train[task_name],
                sampler = train_sampler,
                batch_size = batch_size,
                num_workers = cfg.DATA_LOADER.NUM_WORKERS,
                pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
                drop_last=True
            )

            task_num_iters[task_name] = len(task_dataloader_train[task_name])
            task_batch_size[task_name] = batch_size

        if "val" in split:
            task_dataloader_val[task_name] = torch.utils.data.DataLoader(
                task_datasets_val[task_name],
                shuffle = False,
                batch_size = batch_size_val,
                num_workers = cfg.DATA_LOADER.NUM_WORKERS,
                pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
                drop_last=False
            )
            # VQA
            if task_id == 0:
               task_dataloader_val[task_name + '_test'] = torch.utils.data.DataLoader(
                    task_datasets_val[task_name + '_test'],
                    shuffle = False,
                    batch_size = cfg.TASK.EVAL_BATCH_SIZE[task_id],
                    num_workers = cfg.DATA_LOADER.NUM_WORKERS,
                    pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
                    drop_last=False
               )

    return (
        task_batch_size,
        task_num_iters,
        task_datasets_train,
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
        train_samplers
    )


def load_task_val_dataset(eval_split, rank=-1):
    task_datasets_val = {}
    task_dataloader_val = {}
    task_batch_size = {}
    task_num_iters = {}

    tokenizer = BertTokenizer.from_pretrained(cfg.TRAIN.BERT_MODEL, do_lower_case=cfg.TRAIN.DO_LOWER_CASE)
    for task_id in cfg.TASK.SEL:
        task_name = cfg.TASK.NAME[task_id]
        batch_size = cfg.TASK.EVAL_BATCH_SIZE[task_id]

        task_datasets_val[task_name] = DatasetMapEval[task_name](
                task_name = task_name,
                dataroot = cfg.TASK.DATAROOT[task_id],
                anno_path = cfg.TASK.VAL_ANNO[task_id],
                split = eval_split,
                feat_folder = cfg.TASK.FEAT_FOLDER[task_id],
                gt_feat_folder = cfg.TASK.GT_FEAT_FOLDER[task_id],
                tokenizer = tokenizer,
                bert_model = cfg.TRAIN.BERT_MODEL,
                clean_datasets = cfg.TASK.CLEAN_TRAIN_SET,
                padding_index = 0,
                max_seq_length = cfg.TASK.MAX_SEQ_LEN[task_id],
                max_region_num = cfg.TASK.MAX_REGION_NUM[task_id],
                rank=rank)

        task_dataloader_val[task_name] = torch.utils.data.DataLoader(
            task_datasets_val[task_name],
            shuffle = False,
            batch_size = batch_size,
            num_workers = cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=False
        )

        task_num_iters[task_name] = len(task_dataloader_val[task_name])
        task_batch_size[task_name] = batch_size

    return (
        task_batch_size,
        task_num_iters,
        task_datasets_val,
        task_dataloader_val,
    )
