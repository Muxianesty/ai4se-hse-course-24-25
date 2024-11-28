from collections.abc import Iterable
import datasets
import evaluate
from functools import cache
from pprint import pprint
import re
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing import Any, Dict


DEVICE = "cuda"

@cache
def _init_metrics():
    return (evaluate.load('exact_match'), evaluate.load('rouge'))


def add_signature(element: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    func_str = element[kwargs["column"]]
    func_str = kwargs["prefix"] + func_str
    element[kwargs["column"]] = func_str
    return element


FUNC_PREFIX_MAP = {
    "python": """def <extra_id_0>():\n    """,
    "go"     : """func <extra_id_1> <extra_id_0>() """
}

def predict(dataset: datasets.Dataset, model_name: str, documented: bool) -> None:
    # Implement your function name prediction loop here
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    predictions = []

    if documented:
        table_col_str = "NEW_whole_func_string"
    else:
        table_col_str = "NEW_docs_func_string"

    dataset = dataset.map(add_signature, fn_kwargs={"column": table_col_str, "prefix": FUNC_PREFIX_MAP[dataset.config_name]})

    print('*' * 80)
    print(f"Dataset function language: {dataset.config_name}")
    print(f"Using full function body (with comments): {documented}")
    print(f"Dataset size: {dataset.shape[0]}")
    print('*' * 80)

    references = dataset["NEW_func_name"]
    for id in tqdm(range(dataset.shape[0])):
        element = dataset[table_col_str][id]
        input = tokenizer.encode(element, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
        output = model.generate(input)[0]
        output_decoded = tokenizer.decode(output, skip_special_tokens=True).lstrip()
        output_splitted = re.split(" |\n|\r|\t|\f|\.|\(|\)|\'|\"", output_decoded)
        if (len(output_splitted) > 0):
            output_str = output_splitted[0]
        else:
            output_str = ""
        predictions.append(output_str)

    eval_results = run_evaluate(predictions=predictions, references=references)
    print()
    print('*' * 80)
    print('Evaluation results:')
    pprint(eval_results)
    print('*' * 80)
    print()


def run_evaluate(predictions: Iterable[str],
                references: Iterable[str]) -> dict[str, float]:
    em, rouge = _init_metrics()
    em_score = em.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    return {**rouge_scores, **em_score}
