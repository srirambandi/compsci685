# compsci685/parse_tree_adapted.py
import torch # Not strictly needed in this file, but good practice if extending
from typing import List, Optional, Dict, Tuple

# IMPORTANT: Customize this list based on your dataset's vocabulary!
# Inspect word2idx.keys() in your notebook to find all operator/function names.
OPERATORS_LIST = [
    'add', 'mul', 'sub', 'div', 'pow', 'exp', 'log', 'sin', 'cos', 'sqrt'
]
# Also, consider how constants ('pi', 'E') and variables ('x', 'y', 'I') are handled.
# The current parser treats anything not in OPERATORS_LIST as a terminal.

def get_parse_dict_for_prefix_list(original_words: List[str]) -> Optional[Dict[str, int]]:
    """
    Generates a parse dictionary for TreeReg from a list of pre-split prefix words.
    This parser assumes that all operators in OPERATORS_LIST are binary for the
    purpose of defining a split point for TreeReg. If your notation includes
    unary operators that should also form constituents and have "splits" (though
    conceptually different), this parser would need refinement.

    Args:
        original_words: List of actual words in the prefix expression (e.g., ['add', 'x', 'y'])

    Returns:
        A dictionary mapping "start_word_idx end_word_idx_exclusive_for_span" to
        "split_point_word_idx + 1" (where split_point_word_idx is the index of
        the last word in the left child), or None if parsing fails significantly.
        Returns an empty dictionary for empty input.
    """
    if not original_words:
        return {} # Consistent with previous logic: empty dict for empty input

    parse_dict: Dict[str, int] = {}
    pos: int = 0

    def build_tree_recursive() -> Tuple[Optional[int], Optional[int]]:
        """
        Recursively builds a tree structure from the prefix expression.
        Returns a tuple (start_word_idx, end_word_idx_inclusive) of the current subtree,
        or (None, None) if a parsing error occurs.
        """
        nonlocal pos
        if pos >= len(original_words):
            # This usually indicates a malformed prefix string (e.g., operator missing operands)
            # print(f"Warning (parse_tree_adapted): Prefix parse error - unexpected end of input. Words: {original_words}, current position: {pos}")
            return None, None

        start_node_idx: int = pos # 0-indexed word index of the current node/subtree root
        token: str = original_words[pos].strip()
        pos += 1

        # Determine if the current token is an operator
        # This check could be more sophisticated (e.g., checking arity if known)
        is_operator: bool = token in OPERATORS_LIST

        if is_operator:
            # For TreeReg, we are interested in binary splits.
            # This simplified parser assumes operators take two subsequent constituents (subtrees or terminals).
            # If an operator is unary (e.g., 'neg' 'x'), it will still try to parse two children.
            # This might lead to parse errors for strictly unary ops unless the prefix format adapts (e.g., 'neg' 'x' 'placeholder_for_second_arg_if_binary_parser').
            # Or, unary ops could be treated as terminals if they don't define a "split" for TreeReg.
            # For now, we attempt to parse two children for any item in OPERATORS_LIST.

            left_child_start_idx, left_child_end_idx = build_tree_recursive()
            if left_child_start_idx is None or left_child_end_idx is None:
                # Error propagated from left child parsing
                # print(f"Warning (parse_tree_adapted): Failed to parse left child for operator '{token}' at word index {start_node_idx}.")
                return None, None

            # The split point 'k' (0-indexed word index) for the current span
            # is the end index of the left child's span.
            split_point_k_raw_idx: int = left_child_end_idx

            right_child_start_idx, right_child_end_idx = build_tree_recursive()
            if right_child_start_idx is None or right_child_end_idx is None:
                # Error propagated from right child parsing
                # print(f"Warning (parse_tree_adapted): Failed to parse right child for operator '{token}' at word index {start_node_idx} (after left child ending at {left_child_end_idx}).")
                return None, None

            # The current operator's span ends where its right child's span ends.
            current_subtree_end_idx_inclusive: int = right_child_end_idx

            # Format for parse_dict:
            # Key: "start_word_idx end_word_idx_exclusive_for_span"
            # Value: "split_point_k_raw_idx + 1" (1-indexed word boundary for TreeReg's `best = parse[curr_span] - 1` logic)
            span_key: str = f"{start_node_idx} {current_subtree_end_idx_inclusive + 1}"
            parse_dict[span_key] = split_point_k_raw_idx + 1

            return start_node_idx, current_subtree_end_idx_inclusive
        else:
            # Terminal node (variable, constant, number)
            # Its span is just itself.
            return start_node_idx, start_node_idx

    # Start parsing from the first word (assumed to be the root of the main expression)
    overall_start_idx, overall_end_idx_inclusive = build_tree_recursive()

    # Final validation:
    # 1. Did parsing consume all words? (pos == len(original_words))
    # 2. Was a valid overall tree structure found? (overall_start_idx is not None)
    # 3. Did the parsed overall tree span the entire input? (overall_start_idx == 0 and overall_end_idx_inclusive == len(original_words) - 1)
    if not (pos == len(original_words) and \
            overall_start_idx == 0 and \
            overall_end_idx_inclusive == len(original_words) - 1):
        # print(f"Warning (parse_tree_adapted): Prefix parse potentially incomplete or failed for input: {original_words}.")
        # print(f"  Consumed {pos}/{len(original_words)} words. Parsed overall span: ({overall_start_idx}, {overall_end_idx_inclusive}).")
        # If the parse is not complete and correct for the whole sequence, return None.
        # TreeReg expects parses for valid spans; a partial/incorrect parse for the root can cause issues.
        return None

    return parse_dict

if __name__ == '__main__':
    # Example Usage:
    print("Testing parse_tree_adapted.py...")

    # Update OPERATORS_LIST above if these examples use different ops
    test_cases = [
        (["add", "x", "y"], {"0 3": 1+1}), # add(x, y), split after x (idx 1), value is 1+1=2
        (["mul", "a", "b"], {"0 3": 1+1}), # mul(a, b)
        (["add", "x", "mul", "y", "z"], {"2 5": 3+1, "0 5": 1+1}), # add(x, mul(y,z)), mul(y,z) -> split after y (idx 3), add(...) -> split after x (idx 1)
        (["pow", "x", "2"], {"0 3": 1+1}), # pow(x, 2)
        (["log", "exp", "x"], {"1 3": 2+1, "0 3": 1+1}), # log(exp(x)) this depends on how exp is parsed; if binary (exp, x, dummy), or if unary and handled.
                                          # If exp is binary-parsed: exp(x, dummy_idx_2) -> split is x (idx_2) -> key "1 3": (idx_2)+1
                                          # log(exp_subtree_ending_at_idx_2) -> split is exp_subtree (idx_2) -> key "0 3": (idx_2)+1
                                          # Current parser assumes binary structure for all ops in list for splitting.
        (["sin", "x"], {"0 2": 1+1} if "sin" in OPERATORS_LIST else None), # If sin is treated as binary for structure. If not in OPERATORS_LIST, 'sin' is terminal -> no dict entry
        (["x"], {}), # Single terminal
        ([], {}),    # Empty input
        (["add", "x"], None), # Malformed: add needs two operands
        (["add", "x", "y", "z"], None), # Malformed: extra token 'z' after valid add(x,y)
        (["neg", "x"], {"0 2": 1+1} if "neg" in OPERATORS_LIST else None) # If 'neg' is in OPERATORS_LIST and treated as binary-structured
    ]

    # Temporarily add common ops if not present for testing
    original_ops = list(OPERATORS_LIST)
    if "add" not in OPERATORS_LIST: OPERATORS_LIST.append("add")
    if "mul" not in OPERATORS_LIST: OPERATORS_LIST.append("mul")
    if "pow" not in OPERATORS_LIST: OPERATORS_LIST.append("pow")
    if "log" not in OPERATORS_LIST: OPERATORS_LIST.append("log")
    if "exp" not in OPERATORS_LIST: OPERATORS_LIST.append("exp")
    # if "sin" not in OPERATORS_LIST: OPERATORS_LIST.append("sin") # Test 'sin' based on its presence
    # if "neg" not in OPERATORS_LIST: OPERATORS_LIST.append("neg")

    for tokens, expected in test_cases:
        # Need to handle the case where 'sin' might not be in OPERATORS_LIST for its specific test
        current_expected = expected
        if tokens == ["sin", "x"] and "sin" not in original_ops: # If sin wasn't originally an op, it's a terminal
            current_expected = {} # Then sin(x) parses as two terminals, no span_key in dict
        if tokens == ["neg", "x"] and "neg" not in original_ops:
            current_expected = {}


        result = get_parse_dict_for_prefix_list(list(tokens)) # Pass a copy
        print(f"Input: {tokens}")
        print(f"  Expected: {current_expected}")
        print(f"  Got:      {result}")
        if result != current_expected:
            print(f"  MISMATCH! For input {tokens}")
        print("-" * 20)

    OPERATORS_LIST = original_ops # Restore original