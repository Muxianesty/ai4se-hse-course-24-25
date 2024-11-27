from collections.abc import Iterable
import datasets
import evaluate
from functools import cache
from pprint import pprint
import re
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing import Any, Dict


DEVICE = "cuda"
FUNC_PREFIX = """def <extra_id_0>():\n    """

@cache
def _init_metrics():
    return (evaluate.load('exact_match'), evaluate.load('rouge'))


def whole_add_signature(element: Dict[str, Any]) -> Dict[str, Any]:
    func_str = element["NEW_whole_func_string"]
    func_str = FUNC_PREFIX + func_str
    element["NEW_whole_func_string"] = func_str
    return element


def docs_add_signature(element: Dict[str, Any]) -> Dict[str, Any]:
    func_str = element["NEW_docs_func_string"]
    func_str = FUNC_PREFIX + func_str
    element["NEW_docs_func_string"] = func_str
    return element


def predict(dataset: datasets.Dataset, model_name: str, documented: bool) -> None:
    # Implement your function name prediction loop here
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    predictions = []

    ## Failed to properly use `input_columns` param - had to create two different functions.
    if documented:
        table_col_str = "NEW_whole_func_string"
        dataset = dataset.map(whole_add_signature)
    else:
        table_col_str = "NEW_docs_func_string"
        dataset = dataset.map(docs_add_signature)

    print('*' * 80)
    print(f"Using full function body (with comments): {documented}")
    print(f"Dataset size: {dataset.shape[0]}")
    print('*' * 80)

    references = dataset["func_name"]
    for id in range(dataset.shape[0]):
        element = dataset[table_col_str][id]
        print(id)
        input = tokenizer.encode(element, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
        output = model.generate(input)[0]
        output_decoded = tokenizer.decode(output, skip_special_tokens=True).lstrip()
        output_splitted = re.split(" |\n|\r|\t|\f|\.|\(", output_decoded)
        if (len(output_splitted) > 0):
            output_str = output_splitted[0]
        else:
            output_str = ""
        predictions.append(output_str)
        print('*' * 80)

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
