from pathlib import Path
from typing import Tuple

import datasets
from tree_sitter import Language, Tree, Node, Parser
import tree_sitter_python

DATASET_SIZE=1000
PY_LANG = Language(tree_sitter_python.language())
PARSER = Parser(PY_LANG)

def isComment(node: Node) -> bool:
    node_text = node.text.decode()
    has_comment_part = len(node_text) > 3 and (node_text[0:3] == "\"\"\"" or node_text[0:3] == "'''")
    return node.type == "comment" or (node.type == "expression_statement" and has_comment_part)

def parseFunc(func_str: str) -> Tuple[str, str, str]:
    tree = PARSER.parse(func_str.encode())
    root_node = tree.root_node
    assert root_node.child_count == 1
    func_node = root_node.child(0)
    assert func_node.type == "function_definition"

    func_name_node = func_node.child_by_field_name("name")
    assert func_name_node.type == "identifier"
    func_name_str = func_name_node.text.decode()

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
    func_body_stripped_str = func_body_stripped_byte_str.decode()

    return (func_name_str, func_body_str, func_body_stripped_str)


def prepare() -> datasets.Dataset:
    # Implement dataset preparation code here
    dataset = datasets.load_dataset(
        path="code-search-net/code_search_net",
        name="python",
        split="test", trust_remote_code=True
    )

    dataset = dataset.select(range(DATASET_SIZE))
    func_names = [None]*dataset.shape[0]
    func_strings = [None]*dataset.shape[0]
    func_docs = [None]*dataset.shape[0]

    for i in range(dataset.shape[0]):
        name_str, body_str, body_stripped_str = parseFunc(dataset[i]["whole_func_string"])
        func_names[i] = name_str
        func_strings[i] = body_str
        func_docs[i] = body_stripped_str

    dataset = dataset.add_column("NEW_func_name", func_names)
    dataset = dataset.add_column("NEW_whole_func_string", func_strings)
    dataset = dataset.add_column("NEW_func_no_documentation_string", func_docs)

    return dataset


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))
    selected = dataset.select([9, 19, 21, 62])
    selected.to_json(str(path.joinpath("9-19-21-62.json")))
