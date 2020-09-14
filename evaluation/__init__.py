from .caption_evaler import CaptionEvaler
from .retrieval_evaler import RetrievalEvaler
from .vqa_evaler import VQAEvaler
from .vcr_qa_evaler import VCRQAEvaler
from .vcr_qar_evaler import VCRQAREvaler

EvalMap = {
    "VQA": VQAEvaler,
    "VCR_Q-A": VCRQAEvaler,
    "VCR_QA-R": VCRQAREvaler,
    "RetrievalFlickr30k": RetrievalEvaler,
    'Caption': CaptionEvaler
}
