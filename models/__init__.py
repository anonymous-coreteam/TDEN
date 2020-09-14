from models.tden import TDEN
from models.basic_layer import BertTextPooler, BertImagePooler, BertTextAvgPooler, BertImageAvgPooler, BertAttPooler

BertEncoderMap = {
    'TDEN': TDEN
}

TextPoolerMap = {
    "Single": BertTextPooler,
    "Avg": BertTextAvgPooler,
    "Att": BertAttPooler
}

ImagePoolerMap = {
    "Single": BertImagePooler,
    "Avg": BertImageAvgPooler,
    "Att": BertAttPooler
}