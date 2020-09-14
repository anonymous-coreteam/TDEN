import os
import sys
import json
import math
import pdb
import time
import pprint
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import torch.distributed as dist

from bert.configuration_bert import BertConfig
from models.img_bert import BaseBertForVLTasks
from optimizer.optimizer import Optimizer
import lib.utils as utils
from lib.utils import AverageMeter
from lib.config import cfg, cfg_from_file
from datasets.data_loader import load_task_dataset
import losses.loss_func as loss_func
from evaluation.evaler import Evaler
#from pytorch_transformers.tokenization_bert import BertTokenizer
from tokenization_bert import BertTokenizer
from scorer.scorer import Scorer
from losses import LabelSmoothing, BatchTriplet
import torch.multiprocessing as mp

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

        self.setup_gpu() 
        self.setup_logging()
        self.setup_loader()
        self.setup_network()
        self.load_losses()
        
    def setup_gpu(self):
        if args.local_rank == -1:
            self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
            self.n_gpu = torch.cuda.device_count()
            self.distributed = False
        else:
            torch.cuda.set_device(args.local_rank)
            self.device = torch.device("cuda", args.local_rank)
            self.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            self.distributed = True
        print("device: {} n_gpu: {}, distributed training: {}".format(
            self.device, self.n_gpu, bool(args.local_rank != -1)))
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)


    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        
        if self.distributed and dist.get_rank() > 0:
            return
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_loader(self):
        if self.distributed:
            rank = dist.get_rank()
        else:
            rank = -1

        self.task_batch_size, self.task_num_iters, \
        self.task_datasets_train, self.task_datasets_val, \
        self.task_dataloader_train, self.task_dataloader_val, self.train_sampler = load_task_dataset(args.local_rank, rank=rank)
        
    def load_losses(self):
        LossMap = {
            "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
            "CrossEntropyLoss": nn.CrossEntropyLoss(ignore_index=-1),
            "LabelSmoothing": LabelSmoothing(),
            "Triplet": BatchTriplet()
        }

        self.task_losses = {}
        for task_id in cfg.TASK.SEL:
            name = cfg.TASK.NAME[task_id]
            self.task_losses[name] = LossMap[cfg.TASK.LOSS[task_id]]

    def setup_network(self):
        num_labels = max([dataset.num_labels for dataset in self.task_datasets_train.values()])

        config = BertConfig.from_json_file(cfg.CONFIG_FILE)
        model = BaseBertForVLTasks.from_pretrained(cfg.TRAIN.FROM_PRETRAINED, config=config, num_labels=num_labels)
        #if cfg.TASK.SEL[0] == 0:
        #    model.init_cls(self.task_datasets_train['VQA'].tokenizer)
        #model = BaseBertForVLTasks(config=config, num_labels=num_labels)
        model.to(self.device)

        if args.local_rank != -1:
            #try:
            #    from apex.parallel import DistributedDataParallel as DDP
            #except ImportError:
            #    raise ImportError(
            #        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            #    )
            #self.model = DDP(model)
            self.model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True,
                device_ids=[self.args.local_rank], output_device=self.args.local_rank,
                broadcast_buffers=False)
        elif self.n_gpu > 1:
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model

        epoch_steps = max(self.task_num_iters.values())
        n_steps = epoch_steps * cfg.SOLVER.NUM_TRAIN_EPOCHS
        self.optim = Optimizer(self.model, epoch_steps=epoch_steps, n_steps=n_steps)
        self.evaler = Evaler(self.task_dataloader_val)

        if args.scst:
            self.scorer = Scorer()

    def display(self, iteration, batch_time, losses, scores):
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return

        info_str = ' (BatchTime: {:.3}) losses = {:.5},  scores = {:.5}'.format(batch_time.avg, losses.avg, scores.avg)
        self.logger.info('Iteration ' + str(iteration) + info_str +', lr = ' +  str(self.optim.get_lr()))

        batch_time.reset()
        losses.reset()
        scores.reset()

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".bin")

    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)

        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self
        torch.save(model_to_save.state_dict(), self.snapshot_path("pytorch_model", epoch+1))

    def eval(self, epoch):
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        
        self.model.eval()
        self.logger.info('######## Epoch ' + str(epoch + 1) + ' ########')

        if len(cfg.TASK.SEL) == 2:
            info, q1_res = self.evaler.eval_task_vcr(self.model, cfg.TASK.SEL[0], epoch + 1)
            self.logger.info(info)

            info, q2_res = self.evaler.eval_task_vcr(self.model, cfg.TASK.SEL[1], epoch + 1)
            self.logger.info(info)

            accuracy = 0
            for qid in q1_res:
                if q1_res[qid] == True and q2_res[qid] == True:
                    accuracy += 1
            accuracy /= len(q1_res)
            info = 'Q -> AR: %.3f' % accuracy
            self.logger.info(info)

        else:
            for task_id in cfg.TASK.SEL:
                info = self.evaler.eval_task(self.model, task_id, epoch + 1)
                self.logger.info(info)

    def ForwardModelsSCST(self,
        task_id,
        task_count,
        task_iter_train,
        task_dataloader_train,
        model
    ):
        task_name = cfg.TASK.NAME[task_id]
        if task_count[task_id] % len(task_dataloader_train[task_name]) == 0:
            task_iter_train[task_id] = iter(task_dataloader_train[task_name])

        task_count[task_id] += 1
        # get the batch
        batch = task_iter_train[task_id].next()
        batch = tuple(t.cuda(device=self.device, non_blocking=True) for t in batch)
        
        batch_size, num_options, features, spatials, image_mask, input_txt, \
            target, input_mask, segment_ids, images_id = utils.process_data(batch, task_id)

        model.eval()
        with torch.no_grad():
            _model = model.module if hasattr(model, "module") else model
            seq_max, logP_max = _model.decode(input_txt, features, spatials, \
                segment_ids, input_mask, image_mask, sample_mode='greedy')
        rewards_max, rewards_info_max = self.scorer(images_id.data.cpu().numpy().tolist(), seq_max.data.cpu().numpy().tolist())

        model.train()
        #_model = model.module if hasattr(model, "module") else model
        #seq_sample, logP_sample = _model.decode(input_txt, features, spatials, \
        #    segment_ids, input_mask, image_mask, sample_mode='sample')
        seq_sample, logP_sample = model(
            input_ids = input_txt,
            image_feat = features,
            image_loc = spatials,
            token_type_ids = segment_ids,
            attention_mask = input_mask,
            v_attention_mask = image_mask,
            task_name='decode',
            sample_mode='sample'
        )
        rewards_sample, rewards_info_sample = self.scorer(images_id.data.cpu().numpy().tolist(), seq_sample.data.cpu().numpy().tolist())

        rewards = rewards_sample - rewards_max
        rewards = torch.from_numpy(rewards).float().cuda(device=features.device)

        loss = loss_func.rl_criterion(seq_sample, logP_sample, rewards)
        batch_score = 0
        for key in rewards_info_max:
            batch_score += rewards_info_max[key]

        return loss, batch_score

    def ForwardModelsTrain(self,
        task_id,
        task_count,
        task_iter_train,
        task_dataloader_train,
        model,
        task_losses,
    ): 
        task_name = cfg.TASK.NAME[task_id]
        if task_count[task_id] % len(task_dataloader_train[task_name]) == 0:
            task_iter_train[task_id] = iter(task_dataloader_train[task_name])

        task_count[task_id] += 1
        # get the batch
        batch = task_iter_train[task_id].next()
        batch = tuple(t.cuda(device=self.device, non_blocking=True) for t in batch)
        
        batch_size, num_options, features, spatials, image_mask, input_txt, \
            target, input_mask, segment_ids, sample_id = utils.process_data(batch, task_id)

        res = model(
            input_txt,
            features,
            spatials,
            segment_ids,
            input_mask,
            image_mask,
            task_name
        )   
        label = sample_id.unsqueeze(1).eq(sample_id.unsqueeze(0))
        
        if cfg.TASK.TYPE[task_id] == 'VL-classifier':
            vil_prediction = res
            loss = task_losses[task_name](vil_prediction, target)
            loss = loss.mean() * target.size(1)
            batch_score = loss_func.compute_score_with_logits(vil_prediction, target).sum() / float(batch_size)
            batch_score = batch_score.item()
        elif cfg.TASK.TYPE[task_id] == 'REL':
            scores = res
            loss, triplet_num = task_losses[task_name](scores, label)
            loss = loss.mean()
            batch_score = triplet_num
        elif cfg.TASK.TYPE[task_id] == 'VL-logit':
            vil_logit = res
            vil_logit = vil_logit.view(batch_size, num_options)
            loss = task_losses[task_name](vil_logit, target)
            _, preds = torch.max(vil_logit, 1)
            batch_score = float((preds == target).sum()) / float(batch_size)
        elif cfg.TASK.TYPE[task_id] == 'L-classifier':
            linguisic_prediction = res
            loss = task_losses[task_name](linguisic_prediction.view(-1, linguisic_prediction.shape[-1]), target.view(-1))
            batch_score = loss.item()
        else:
            raise NotImplementedError

        return loss, batch_score

    def train(self):
        max_num_iter = max(self.task_num_iters.values())
        task_iter_train = { name: None for name in cfg.TASK.SEL }
        task_count = { name: 0 for name in cfg.TASK.SEL }
        for epochId in tqdm(range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.NUM_TRAIN_EPOCHS)):

            for name in self.train_sampler:
                if self.train_sampler[name] is not None:
                    self.train_sampler[name].set_epoch(epochId)
            self.model.train()
            for step in range(max_num_iter):
                iterId = step + (epochId * max_num_iter)
                for task_id in cfg.TASK.SEL:
                    start = time.time()
                    batch_time = AverageMeter()
                    losses = AverageMeter()
                    scores = AverageMeter()
                    self.optim.zero_grad()

                    if args.scst:
                        loss, score = self.ForwardModelsSCST(
                            task_id,
                            task_count,
                            task_iter_train,
                            self.task_dataloader_train,
                            self.model
                        )
                    else:
                        loss, score = self.ForwardModelsTrain(
                            task_id,
                            task_count,
                            task_iter_train,
                            self.task_dataloader_train,
                            self.model,
                            self.task_losses
                        )
                    loss.backward()

                    utils.clip_gradient(self.optim.optimizer, self.model,
                        cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.CLIP_GRAD)

                    self.optim.step()
                    self.optim.zero_grad()
                    if self.distributed:     
                        dist.barrier()                    

                    batch_time.update(time.time() - start)
                    losses.update(loss.item())
                    scores.update(score)

                    self.display(iterId, batch_time, losses, scores)
                    start = time.time()
                    
                self.optim.scheduler_step(iterId)
                
            self.eval(epochId)
            self.save_model(epochId)
            if self.distributed:
                dist.barrier()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train deep metric learning')
    parser.add_argument('--folder', dest='folder', default=None, type=str)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder

    tokenizer = BertTokenizer.from_pretrained(cfg.TRAIN.BERT_MODEL, do_lower_case=cfg.TRAIN.DO_LOWER_CASE)
    cfg.MODEL.CLS_ID = tokenizer.vocab["[CLS]"]
    cfg.MODEL.SEP_ID = tokenizer.vocab["[SEP]"]
    cfg.MODEL.MASK_ID = tokenizer.vocab["[MASK]"]
    
    trainer = Trainer(args)
    trainer.train()
