import os
import json
import numpy as np
import json_lines
import _pickle as cPickle

json_path = 'VCR/val.jsonl'

def Q2A_annotations(json_path):
   q2a_gt = {}
   with open(json_path, "rb") as f:
       for annotation in json_lines.reader(f): 
           ans_label = annotation["answer_label"]
           anno_id = int(annotation["annot_id"].split("-")[1])
           q2a_gt[anno_id] = ans_label

   cPickle.dump(q2a_gt, open('q2a_gt.pkl', "wb"))

def QA2R_annotations(json_path):
   qa2r_gt = {}
   with open(json_path, "rb") as f:
       for annotation in json_lines.reader(f):
           ans_label = annotation["rationale_label"]
           anno_id = int(annotation["annot_id"].split("-")[1])
           qa2r_gt[anno_id] = ans_label

   cPickle.dump(qa2r_gt, open('qa2r_gt.pkl', "wb"))


if __name__ == '__main__':
   Q2A_annotations(json_path)
   QA2R_annotations(json_path)



