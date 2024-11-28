from tree_sitter import Language, Node, Parser
import tree_sitter_go
from typing import Tuple

GO_LANG = Language(tree_sitter_go.language())
PARSER = Parser(GO_LANG)


def isComment(node: Node) -> bool:
    return node.type == "comment"


def parseFunc(func_str: str) -> Tuple[str, str, str]:
    tree = PARSER.parse(func_str.encode())
    root_node = tree.root_node
    assert root_node.type == "source_file"
    assert root_node.child_count == 1

    func_node = root_node.child(0)
    assert func_node.type in ["function_declaration", "method_declaration"] 

    func_name_node = func_node.child_by_field_name("name")
    assert func_name_node.type in ["identifier", "field_identifier"]
    func_name_str = func_name_node.text.decode()

    func_body_node = func_node.child_by_field_name("body")
    assert func_body_node.type == "block"
    func_body_byte_str = func_body_node.text
    func_body_str = func_body_byte_str.decode()
    
    func_body_stripped_byte_str = func_body_byte_str
    begin_shift = func_body_node.children[0].start_byte
    curr_shift = 0
    
    curr_node = func_body_node.walk()
    already_returned = False
    while (True):
        if not already_returned and isComment(curr_node.node):
            left = curr_node.node.start_byte - curr_shift - begin_shift
            right = curr_node.node.end_byte - curr_shift - begin_shift
            add_shift = right - left
            first_part = func_body_stripped_byte_str[:left]
            second_part = func_body_stripped_byte_str[right:]
            curr_shift += add_shift
            func_body_stripped_byte_str = first_part + second_part

        if not already_returned and curr_node.goto_first_child():
            continue

        already_returned = False
        
        if curr_node.goto_next_sibling():
            continue

        if curr_node.node == func_body_node:
            break

        already_returned = curr_node.goto_parent()
        
    func_body_stripped_str = func_body_stripped_byte_str.decode()

    return (func_name_str, func_body_str, func_body_stripped_str)
