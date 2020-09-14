import os
import json
import tqdm
import torch
import numpy as np
import lib.utils as utils
from lib.config import cfg


class VCRQAEvaler(object):
    def __init__(self, val_loader):
        self.val_loader = val_loader
        self.task_id = 1

    def eval(self, model, epoch):
        results = []
        accuracy = 0
        qa_res = {}
        with torch.no_grad():
            for batch in tqdm.tqdm(self.val_loader):
                batch = tuple(t.cuda() for t in batch)

                batch_size, num_options, \
                features, spatials, image_mask, \
                input_txt, target, input_mask, \
                segment_ids, question_id = utils.process_data(batch, self.task_id)

                question_id = question_id.view(batch_size, -1)
                question_id = question_id[:, 0]
                target = target.data.cpu().numpy()

                vil_logit = model(
                    input_txt, features, spatials, segment_ids, input_mask, image_mask, 'VCR_Q-A')

                vil_logit = vil_logit.view(batch_size, num_options)

                probs = torch.softmax(vil_logit, dim=1)
                for i in range(vil_logit.size(0)):
                    ans = [prob.item() for prob in probs[i]]
                    results.append({'question_id': question_id[i].item(), 'answer': ans})
                    if np.argmax(ans) == target[i]:
                        accuracy += 1
                        qa_res[question_id[i].item()] = True
                    else:
                        qa_res[question_id[i].item()] = False

        accuracy /= len(results)

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, str(epoch) + '_Q_A.json'), 'w'))

        return 'Q -> A: %.3f' % accuracy, qa_res
