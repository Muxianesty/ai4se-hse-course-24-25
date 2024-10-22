import codebert
import datasets
import log_reg
import numpy as np

MODEL_MAP = {
    "classic_log": log_reg.classifier,
    "codebert"   : codebert.classifier
}

def classifier(dataset: datasets.Dataset, model_str: str):
    MODEL_MAP[model_str](dataset)
