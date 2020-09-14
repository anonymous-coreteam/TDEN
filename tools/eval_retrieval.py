import os
import sys
import json
import tqdm
import argparse
import numpy as np
import _pickle as cPickle

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from lib.config import cfg

class RetrievalEvaler(object):
    def __init__(self):
        super(RetrievalEvaler, self).__init__()
        self.target_matrix = cPickle.load(open(cfg.INFERENCE.RETRIEVAL_ANNFILE, "rb"))

    def eval(self, score_matrix_path):
        score_matrix = cPickle.load(open(score_matrix_path, "rb"))
        rows = self.target_matrix.shape[0]
        cols = self.target_matrix.shape[1]
        rank_matrix = np.ones((rows)) * cols
        for r in range(rows):
            rank = np.where((np.argsort(-score_matrix[r]) == np.where(self.target_matrix[r]==1)[0][0]) == 1)[0][0]
            rank_matrix[r] = rank

        r1 = np.sum(rank_matrix < 1) / len(rank_matrix)
        r5 = np.sum(rank_matrix < 5) / len(rank_matrix)
        r10 = np.sum(rank_matrix < 10) / len(rank_matrix)

        res = "R1: {:.2%}, R5: {:.2%}, R10: {:.2%}".format(r1, r5, r10)
        return res

def parse_args():
    parser = argparse.ArgumentParser(description='Retrieval Evaler')
    parser.add_argument('--score_matrix', dest='score_matrix', default=None, type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    evaler = RetrievalEvaler()
    res = evaler.eval(args.score_matrix)
    print(res)