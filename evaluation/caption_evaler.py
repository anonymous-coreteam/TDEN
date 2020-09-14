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

sys.path.append(cfg.INFERENCE.COCO_PATH)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

class COCOEvaler(object):
    def __init__(self):
        super(COCOEvaler, self).__init__()
        self.coco = COCO(cfg.INFERENCE.ANNFILE)
        if not os.path.exists(cfg.TEMP_DIR):
            os.mkdir(cfg.TEMP_DIR)

    def eval(self, result):
        in_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=cfg.TEMP_DIR)
        json.dump(result, in_file)
        in_file.close()

        cocoRes = self.coco.loadRes(in_file.name)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate()
        os.remove(in_file.name)
        return cocoEval.eval

class CaptionEvaler(object):
    def __init__(self, val_loader):
        super(CaptionEvaler, self).__init__()
        self.task_id = 4
        self.val_loader = val_loader
        self.evaler = COCOEvaler()
        self.mask_token_id = cfg.MODEL.MASK_ID
        self.sep_token_id = cfg.MODEL.SEP_ID          # end token
        self.max_seq_length = val_loader.dataset.max_seq_length

    def decode_sents(self, batch, model):
        
        features, spatials, image_mask, input_txt, target, input_mask, segment_ids, images_id = (
            batch
        )

        model_to_test = (
            model.module if hasattr(model, "module") else model
        )

        sents, _ = model_to_test.decode(input_txt, features, spatials, \
                segment_ids, input_mask, image_mask, sample_mode='greedy', beam_size=cfg.INFERENCE.BEAM_SIZE)

        sentences = self.val_loader.dataset.decode_sequence(sents)
        return images_id.data.cpu().numpy(), sentences


    def eval(self, model, epoch):
        results = []
        with torch.no_grad():
            for batch in tqdm.tqdm(self.val_loader):
                batch = tuple(t.cuda() for t in batch)
                images_id, sents = self.decode_sents(batch, model)
    
                for i, sent in enumerate(sents):
                    image_id = images_id[i]
                    result = { "image_id": int(image_id), "caption": sent }
                    results.append(result)

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, str(epoch) + '.json'), "w"))
        return str(self.evaler.eval(results))
