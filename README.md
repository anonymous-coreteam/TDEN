## Introduction
This repository is for **Scheduled Sampling in Vision-Language Pretraining with Decoupled Encoder-Decoder Network**

## Requirements
* Python 3
* CUDA 10
* [PyTorch](http://pytorch.org/)
* [coco-caption](https://github.com/ruotianluo/coco-caption)

## Dataset
* [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/download)
* [Visual Question Answering](https://visualqa.org/download.html)
* [Caption-based image retrieval](http://shannon.cs.illinois.edu/DenotationGraph/)
* [Visual commonsense reasoning](https://visualcommonsense.com/download/)
* [Image captioning](http://cocodataset.org/#download)

## Data preparation
1. Download [coco-caption](https://github.com/ruotianluo/coco-caption), [bert-base-uncased.tar.gz](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) and [bert-base-uncased-vocab.txt](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt), and then put them into the same directory of TDEN.

2. Please refer to [vilbert](https://github.com/jiasenlu/vilbert_beta) for data preparation.

## Code
1. ```TDEN/models/tden.py``` is the code of our proposed architecture TDEN

2. ```TDEN/models/img_bert.py``` is the code for pretraining(class BaseBertPreTraining) and downstream tasks(class BaseBertForVLTasks).

3. The ```TDEN/evaluation``` folder contains the codes for result generation and evaluation of each downstream task.

## Model
The pretrained model can be downloaded [here](https://drive.google.com/file/d/1SA2GZKJBkvKOXqBibVOGS_f3YVIQpnPv/view?usp=sharing) 

## Evaluation
#### Visual Question Answering
1. Generate the prediction result file and submit to [EvalAI submission](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview)

#### Caption-based image retrieval
1. Generate target_matrix.pkl by ``` python3 tools/retrieval_ann.py ```
(unzip TDEN/data/retrieval/target_matrix.zip can obtain target_matrix.pkl file)

2. Generate similarity matrix and evaluate the result by ``` python3 tools/eval_retrieval.py --score_matrix similarity_matrix_path ```

#### Visual commonsense reasoning
1. Generate q2a_gt.pkl and qa2r_gt.pkl by ``` python3 tools/vcr_ann.py ```
(The files are already in TDEN/data/vcr)

2. Generate prediction results and evaluate them by ``` python3 tools/eval_vcr.py --q2a_file ./experiments/result/vcr/Q_A.json --qa2r_file ./experiments/result/vcr/QA_R.json ```

#### Image captioning
1. unzip TDEN/data/caption/captions_test5k.zip

2. ``` python3 tools/eval_caption.py --file ./experiments/result/coco_caption/result.json ```

## Acknowledgements
Thanks the contribution of [vilbert](https://github.com/jiasenlu/vilbert_beta) and [pytorch-transformers](https://github.com/huggingface/transformers).
