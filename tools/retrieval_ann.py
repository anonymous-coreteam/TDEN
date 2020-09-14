import os
import json
import jsonlines
import numpy as np
import _pickle as cPickle

cids = []
iids = []

jsonpath = 'flickr30k/all_data_final_test_set0_2014.jsonline'
with jsonlines.open(jsonpath) as reader:
   for annotation in reader:
      image_id = int(annotation["img_path"].split(".")[0])
      iids.append(image_id)

      for _ in annotation["sentences"]:
          cids.append(image_id)

cids = np.array(cids)
iids = np.array(iids)

cids = np.expand_dims(cids, axis=1)
iids = np.expand_dims(iids, axis=0)

target_matrix = np.equal(cids, iids).astype(int)
cPickle.dump(target_matrix, open('target_matrix.pkl', "wb"))

