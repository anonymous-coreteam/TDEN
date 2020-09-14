import os
import sys
import numpy as np
import torch
import lib.utils as utils
from lib.config import cfg
from evaluation import EvalMap

class Evaler(object):
    def __init__(self, task_dataloader_val):
        super(Evaler, self).__init__()
        self.task_evaler = {}
        for task_id in cfg.TASK.SEL:
            name = cfg.TASK.NAME[task_id]
            self.task_evaler[name] = []
            for key in task_dataloader_val:
               if name in key:
                  self.task_evaler[name].append(EvalMap[name](task_dataloader_val[key]))

    def eval_task(self, model, task_id, epoch):
        name = cfg.TASK.NAME[task_id]
        res = []
        for task_evaler in self.task_evaler[name]:
          res.append(task_evaler.eval(model, epoch))
        res = '\n'.join(res)
        return res

    def eval_task_vcr(self, model, task_id, epoch):
        name = cfg.TASK.NAME[task_id]
        res = []
        for task_evaler in self.task_evaler[name]:
          res_str, q_res = task_evaler.eval(model, epoch)
          res.append(res_str)
        res = '\n'.join(res)
        return res, q_res

