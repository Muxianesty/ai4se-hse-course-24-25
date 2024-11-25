from typing import Tuple
from tree_sitter import Language, Tree, Node, Parser
import tree_sitter_python

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