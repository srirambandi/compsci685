import torch # Make sure torch is imported
from typing import List, Optional, Dict, Tuple

def get_prefix_parse_dict(prefix_tokens: List[str], word_boundaries: List[bool]) -> Optional[Dict[str, int]]:
    """
    Generates a parse dictionary for TreeReg from prefix notation tokens
    and word boundaries after tokenization.

    Args:
        prefix_tokens: List of tokens corresponding to the prefix expression (e.g., [' add', ' x', ' y'])
                       These are tokens AFTER HuggingFace tokenization.
        word_boundaries: Boolean list where True indicates the token starts a new word/item
                         in the original prefix string.

    Returns:
        A dictionary mapping "start_word_idx end_word_idx+1" to "split_word_idx+1",
        or None if parsing fails.
    """
    # Map token indices to word indices
    word_indices = [-1] * len(prefix_tokens)
    word_idx_counter = -1
    for i, is_start in enumerate(word_boundaries):
        if is_start:
            word_idx_counter += 1
        word_indices[i] = word_idx_counter

    num_words = word_idx_counter + 1
    if num_words == 0:
        return None # No words to parse

    # Get the original words based on boundaries
    original_words = []
    current_word = ""
    for i, token in enumerate(prefix_tokens):
        if word_boundaries[i] and current_word:
            original_words.append(current_word)
            current_word = token
        elif word_boundaries[i]:
             current_word = token
        else:
            current_word += token # Should ideally use tokenizer.decode, but this is approx.
    if current_word:
         original_words.append(current_word)

    # --- Simple Stack-based Parser for Prefix ---
    # This assumes binary operations for simplicity. Extend if needed for unary/n-ary.
    parse_dict = {}
    pos = 0

    def build_tree() -> Tuple[Optional[int], Optional[int]]:
        """Recursively builds tree, returns (start_word_idx, end_word_idx)"""
        nonlocal pos
        if pos >= len(original_words):
            # This indicates an issue with the input prefix string format
            print(f"Warning: Prefix parse error - unexpected end of input. Words: {original_words}")
            return None, None # Error condition

        start_node_idx = pos
        token = original_words[pos].strip() # Use original word
        pos += 1

        # Rough check for operators (customize based on your actual operators)
        # Assumes operators like 'add', 'mul', 'pow', 'exp', 'log' etc.
        # Assumes variables/constants are single letters or numbers possibly with ^ or ()
        is_operator = token in ['add', 'mul', 'sub', 'div', 'pow', 'exp', 'log', 'sin', 'cos', 'tan'] or token.startswith('-') # Basic check

        if is_operator:
            # --- Assume binary operator for structure ---
            # This is a simplification; real prefix parsing needs arity info.
            # We'll treat it as if it takes two subsequent elements (subtrees or terminals).
            left_start, left_end = build_tree()
            if left_start is None: return None, None # Propagate error

            # The split point `k` for span (start_node_idx, right_end) is `left_end`.
            # The parse dict wants k+1.
            split_point_k = left_end
            if split_point_k is None:
                 print(f"Warning: Could not determine split point after left child. Token: {token}")
                 return None, None

            right_start, right_end = build_tree()
            if right_start is None: return None, None # Propagate error


            if right_end is None:
                 print(f"Warning: Could not determine end point after right child. Token: {token}")
                 return None, None

            span_key = f"{start_node_idx} {right_end + 1}"
            parse_dict[span_key] = split_point_k + 1 # Store k+1

            return start_node_idx, right_end
        else:
            # Terminal node (variable, constant, number)
            return start_node_idx, start_node_idx

    start_idx, end_idx = build_tree()

    # Check if the whole input was consumed and parsed correctly
    if pos != len(original_words) or start_idx is None:
        print(f"Warning: Prefix parse finished at pos {pos} != len {len(original_words)}. Input: {original_words}")
        # Don't return partial parse if it failed significantly
        return None

    # Convert word indices in parse_dict keys/values back to token indices if needed by TreeReg
    # The provided TreeReg code seems to operate on word indices based on `word_boundaries`.
    # Let's stick to word indices (0 to num_words-1).
    return parse_dict


# --- Augment preprocessing to include parses ---
# You would modify the `preprocess_function` or run this afterwards

def add_parses_to_batch(batch, tokenizer):
    """Applies parse generation to a batch from the tokenized dataset."""
    parses = []
    original_inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)

    for i in range(len(batch['input_ids'])):
        input_ids = batch['input_ids'][i]
        # Need tokens corresponding to input_ids to pass to get_prefix_parse_dict
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        boundaries = batch['word_boundaries'][i]

        # Ensure tensors are on CPU and converted to lists/numpy for processing
        if isinstance(boundaries, torch.Tensor):
            boundaries = boundaries.cpu().tolist()

        # Extract the prefix part (remove task prefix added earlier)
        # This depends on how you structured `inputs` in `preprocess_function`
        # Let's decode and re-split. A bit inefficient but safer.
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=True)
        task_prefix = "translate prefix equation to prefix solution: " # Must match preprocess_function
        if decoded_input.startswith(task_prefix):
            prefix_str = decoded_input[len(task_prefix):]
        else:
            prefix_str = decoded_input # Fallback if prefix wasn't there

        # Get tokens and boundaries ONLY for the actual prefix part
        # Re-tokenize the prefix part to align tokens/boundaries correctly
        prefix_encoding = tokenizer(prefix_str, add_special_tokens=False) # No special tokens here
        prefix_input_ids = prefix_encoding['input_ids']
        prefix_tokens = tokenizer.convert_ids_to_tokens(prefix_input_ids)

        # Recalculate word boundaries just for the prefix part
        prefix_word_boundaries = [False] * len(prefix_tokens)
        if prefix_tokens: # Check if list is not empty
           first_real_token_found = False
           for j, token in enumerate(prefix_tokens):
               if token.startswith(" "):
                   prefix_word_boundaries[j] = True
                   first_real_token_found = True
               elif j == 0 and not first_real_token_found:
                    # Mark the very first token if it doesn't start with space but is the actual start
                    prefix_word_boundaries[j] = True
                    first_real_token_found = True


        # Generate the parse dictionary using only prefix tokens/boundaries
        parse = get_prefix_parse_dict(prefix_tokens, prefix_word_boundaries)
        parses.append(parse if parse else {}) # Add empty dict if parse failed

    batch['parses'] = parses
    return batch