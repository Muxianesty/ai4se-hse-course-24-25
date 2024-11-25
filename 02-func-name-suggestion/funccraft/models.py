from collections.abc import Iterable
from functools import cache
from pprint import pprint

import datasets
import evaluate
 
@cache
def _init_metrics():
    return (evaluate.load('exact_match'), evaluate.load('rouge'))



def predict(dataset: datasets.Dataset, model: str) -> None:
    # Implement your function name prediction loop here
    func_str = dataset[0]["whole_func_string"]
    dataset.add_column("NEW_func_name", [None]*dataset.shape[0])
    dataset.add_column("NEW_whole_func_string", [None]*dataset.shape[0])
    dataset.add_column("NEW_func_documentation_string", [None]*dataset.shape[0])
    py_lang = Language(tree_sitter_python.language())
    parser = Parser(py_lang)
    tree = parser.parse(func_str.encode())
    root_node = tree.root_node
    assert root_node.child_count == 1
    func_node = root_node.child(0)
    assert func_node.type == "function_definition"

    func_name_node = func_node.child_by_field_name("name")
    assert func_name_node.type == "identifier"
    func_name = func_name_node.text.decode()

    func_body_node = func_node.child_by_field_name("body")
    assert func_body_node.type == "block"
    func_body_byte_str = func_body_node.text
    func_body_str = func_body_byte_str.decode()
    
    func_body_stripped_byte_str = func_body_byte_str
    begin_shift = func_body_node.children[0].start_byte
    curr_shift = 0
    
    for node in func_body_node.children:
        if isComment(node):
            left = node.start_byte - curr_shift - begin_shift
            right = node.end_byte - curr_shift - begin_shift
            add_shift = right - left
            first_part = func_body_stripped_byte_str[:left]
            second_part = func_body_stripped_byte_str[right:]
            curr_shift += add_shift
            func_body_stripped_byte_str = first_part + second_part

    
    




    



    # predictions = ['func_one', 'func_three']
    # references = ['func_one', 'func_two']

    # eval_results = run_evaluate(predictions=predictions, references=references)
    # print()
    # print('*' * 80)
    # print('Evaluation results:')
    # pprint(eval_results)
    # print('*' * 80)
    # print()


def run_evaluate(
    predictions: Iterable[str], references: Iterable[str]
) -> dict[str, float]:
    em, rouge = _init_metrics()
    em_score = em.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    return {**rouge_scores, **em_score}
