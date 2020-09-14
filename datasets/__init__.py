from .vcr_dataset import VCRDataset
from .vqa_dataset import VQAClassificationDataset
from .retrieval_dataset import RetrievalDataset, RetrievalDatasetVal
from .coco_cap_dataset import COCOCapDataset

DatasetMapTrain = {
    "VQA": VQAClassificationDataset,
    "VCR_Q-A": VCRDataset,
    "VCR_QA-R": VCRDataset,
    "RetrievalFlickr30k": RetrievalDataset,
    'Caption': COCOCapDataset
}

DatasetMapEval = {
    "VQA": VQAClassificationDataset,
    "VCR_Q-A": VCRDataset,
    "VCR_QA-R": VCRDataset,
    "RetrievalFlickr30k": RetrievalDatasetVal,
    'Caption': COCOCapDataset
}