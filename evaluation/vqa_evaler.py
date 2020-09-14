import os
import tqdm
import json
import torch
import lib.utils as utils
from lib.config import cfg


class VQAEvaluator:
    def __init__(self, val_loader):
        self.val_loader = val_loader

    def eval(self, results):
        score = 0.
        for result in results:
            quesid = result['question_id']
            ans = result['answer']
            datum = self.val_loader.dataset.id2datum[quesid]
            if ans in datum:
                score += datum[ans]
        return score / len(results)


class VQAEvaler(object):
    def __init__(self, val_loader):
        self.task_id = 0
        self.val_loader = val_loader
        self.evaler = VQAEvaluator(val_loader)

    def eval(self, model, epoch):
        if 'test' in self.val_loader.dataset.split and epoch < 6:
            return 'no eval'

        results = []
        others = []
        with torch.no_grad():
            for batch in tqdm.tqdm(self.val_loader):
                batch = tuple(t.cuda() for t in batch)

                batch_size, num_options, \
                features, spatials, image_mask, \
                input_txt, target, input_mask, \
                segment_ids, question_id = utils.process_data(batch, self.task_id)

                vil_prediction = model(
                    input_txt, features, spatials, segment_ids, input_mask, image_mask, 'VQA')

                logits = torch.max(vil_prediction, 1)[1].data  # argmax
                #sorted_score, sorted_idx = torch.sort(-vil_prediction)
                #topk = 8  # top candidate.
                #topkInd = sorted_idx[:, :topk]
                for i in range(logits.size(0)):
                    results.append({'question_id': question_id[i].item(), \
                                    'answer': self.val_loader.dataset.label2ans[logits[i].item()]})

                    # save top 8 as options.
                    #others.append({'question_id': question_id[i].item(), \
                    #               'answer': [self.val_loader.dataset.label2ans[idx.item()] for idx in
                    #                          topkInd[i]]})
        
        if 'test' not in self.val_loader.dataset.split:
           acc = self.evaler.eval(results)
           return "minval accuracy = %.3f" % acc
        else:
           result_folder = os.path.join(cfg.ROOT_DIR, 'result')
           if not os.path.exists(result_folder):
               os.mkdir(result_folder)

           json.dump(results, open(os.path.join(result_folder, str(epoch) + '_results.json'), 'w'))
           #json.dump(others, open(os.path.join(result_folder, str(epoch) + '_others.json'), 'w'))
           return 'VQA evaluation file saved'
