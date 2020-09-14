import os
import sys
import json
import argparse
import numpy as np

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from lib.config import cfg
sys.path.append(cfg.INFERENCE.COCO_PATH)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

class COCOEvaler(object):
    def __init__(self):
        super(COCOEvaler, self).__init__()
        self.coco = COCO(cfg.INFERENCE.ANNFILE)
        if not os.path.exists(cfg.TEMP_DIR):
            os.mkdir(cfg.TEMP_DIR)

    def eval(self, file_path):
        cocoRes = self.coco.loadRes(file_path)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate()
        return cocoEval.eval

def parse_args():
    parser = argparse.ArgumentParser(description='Caption Evaler')
    parser.add_argument('--file', dest='file', default=None, type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    evaler = COCOEvaler()
    res = evaler.eval(args.file)
    print(res)

