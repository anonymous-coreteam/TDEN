import os
import sys
import json
import argparse
import numpy as np
import _pickle as cPickle

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
from lib.config import cfg

class VCREvaler(object):
    def __init__(self):
        super(VCREvaler, self).__init__()
        self.q2a_gt = cPickle.load(open(cfg.INFERENCE.VCR_Q2A_ANNFILE, "rb"))
        self.qa2r_gt = cPickle.load(open(cfg.INFERENCE.VCR_QA2R_ANNFILE, "rb"))

    def eval(self, q2a_file, qa2r_file):
        q2a_res = json.load(open(q2a_file, "r"))
        qa2r_res = json.load(open(qa2r_file, "r"))

        q2a_acc = 0
        qa2r_acc = 0
        q2ar_acc = 0
        for i in range(len(q2a_res)):
            qid = q2a_res[i]['question_id']
            q2a_pred = np.argmax(q2a_res[i]['answer'])
            qa2r_pred = np.argmax(qa2r_res[i]['answer'])
            if q2a_pred == self.q2a_gt[qid]:
                q2a_acc += 1
            if qa2r_pred == self.qa2r_gt[qid]:
                qa2r_acc += 1
            if q2a_pred == self.q2a_gt[qid] and qa2r_pred == self.qa2r_gt[qid]:
                q2ar_acc += 1
            
        q2a_acc /= len(q2a_res)
        qa2r_acc /= len(q2a_res)
        q2ar_acc /= len(q2a_res)

        res = "Q -> A: {:.2%}, QA -> R: {:.2%}, Q -> AR: {:.2%}".format(q2a_acc, qa2r_acc, q2ar_acc)
        return res


def parse_args():
    parser = argparse.ArgumentParser(description='VCR Evaler')
    parser.add_argument('--q2a_file', dest='q2a_file', default=None, type=str)
    parser.add_argument('--qa2r_file', dest='qa2r_file', default=None, type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    evaler = VCREvaler()
    res = evaler.eval(args.q2a_file, args.qa2r_file)
    print(res)    