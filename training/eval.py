import sympy
from sympy import Add, Mul, Pow, Rational, Integer, Float, Function, Symbol
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.timedcalls import timed, TimeoutError as SympyTimeoutError
from typing import List, Any, Optional

# Define common mathematical functions and symbols SymPy might need
sympy_symbols = {s: Symbol(s) for s in 'xyzabc'} # Common variables/constants
sympy_funcs = {'exp': sympy.exp, 'log': sympy.log, 'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan} # Add more as needed

# Add 'y\'' symbol for derivatives if present
sympy_symbols["y'"] = Symbol("y'") # Assuming y' is treated as a distinct symbol

def prefix_to_sympy(tokens: List[str]) -> Optional[sympy.Expr]:
    """
    Converts a list of prefix notation tokens into a SymPy expression.
    Handles basic arithmetic, powers, exp, log, and common symbols.
    Assumes 'c' is a constant symbol.
    """
    stack = []
    
    # Mapping from prefix token to SymPy function and expected arg count
    # Extend this map based on your dataset's operators
    op_map = {
        'add': (Add, 2),
        'sub': (lambda a, b: Add(a, Mul(b, -1)), 2), # Subtraction as adding negative
        'mul': (Mul, 2),
        'div': (lambda a, b: Mul(a, Pow(b, -1)), 2), # Division as multiplying inverse
        'pow': (Pow, 2),
        'exp': (sympy.exp, 1),
        'log': (sympy.log, 1),
        'sin': (sympy.sin, 1),
        'cos': (sympy.cos, 1),
        'tan': (sympy.tan, 1),
        # Add unary minus if needed, e.g., 'neg'
        'neg': (lambda a: Mul(a, -1), 1)
    }

    try:
        # Iterate tokens in reverse for prefix stack parsing
        for token in reversed(tokens):
            token = token.strip()
            if not token: continue

            if token in op_map:
                func, arity = op_map[token]
                if len(stack) < arity:
                    # print(f"Warning: Not enough operands for operator '{token}'. Stack: {stack}")
                    return None # Malformed expression
                operands = [stack.pop() for _ in range(arity)]
                # Note: operands are popped in reverse order of how they appear in prefix
                # For binary ops: operands = [right, left]. Pass them correctly.
                if arity == 1:
                     result = func(operands[0])
                elif arity == 2:
                     result = func(operands[1], operands[0]) # Correct order for binary ops
                else:
                     # Extend if needed for n-ary functions
                     result = func(*reversed(operands))
                stack.append(result)
            else:
                # Attempt to parse as number or symbol
                try:
                    # Handle potential scientific notation like '1e-5'
                    if 'e' in token.lower() and not token.lower().startswith('exp'):
                         num = Float(token)
                    # Handle fractions like '1/2'
                    elif '/' in token:
                         parts = token.split('/')
                         if len(parts) == 2:
                              num = Rational(Integer(parts[0]), Integer(parts[1]))
                         else: # Not a simple fraction, treat as symbol?
                              sym = sympy_symbols.get(token, Symbol(token))
                              stack.append(sym)
                              continue # Skip to next token
                    else:
                         num = Integer(token) # Try integer first
                    stack.append(num)
                except (ValueError, TypeError):
                    # Not a standard number, treat as symbol (e.g., x, y, c, pi)
                    # Check if it's a predefined symbol/function or create a new one
                    if token in sympy_funcs:
                         # This case shouldn't happen if functions are handled as ops, but as fallback
                         stack.append(sympy_funcs[token])
                    else:
                         # Add custom handling for things like 'y'' if needed
                         sym = sympy_symbols.get(token, Symbol(token))
                         stack.append(sym)

        if len(stack) == 1:
            return stack[0]
        else:
            # print(f"Warning: Malformed prefix expression. Stack: {stack}")
            return None # Error: should end with one expression on stack

    except Exception as e:
        # print(f"Error parsing prefix to SymPy: {e}, Tokens: {tokens}")
        return None


def safe_parse_expr(expr_str: str) -> Optional[sympy.Expr]:
    """Safely parse a string into a SymPy expression using parse_expr."""
    try:
        # Add 'c' and potentially other constants/functions to local dict
        local_dict = {"Symbol": Symbol, "Integer": Integer, "Rational": Rational, "Float": Float}
        local_dict.update(sympy_symbols) # Add x,y,z,a,b,c
        local_dict.update(sympy_funcs) # Add exp, log, etc.

        # Transformations can help standardise input (e.g., e^x to exp(x))
        # from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application
        # transformations = standard_transformations + (implicit_multiplication_application,)
        # parsed_expr = parse_expr(expr_str, local_dict=local_dict, transformations=transformations)

        # Simpler parsing without implicit mult/transforms first
        parsed_expr = parse_expr(expr_str, local_dict=local_dict)
        return parsed_expr
    except Exception as e:
        # print(f"SymPy parsing error for '{expr_str}': {e}")
        return None

def check_sympy_equivalence(pred_expr: sympy.Expr, label_expr: sympy.Expr, timeout_seconds=2) -> bool:
    """Checks if two SymPy expressions are mathematically equivalent."""
    if pred_expr is None or label_expr is None:
        return False # Cannot compare if parsing failed

    try:
        # Use timed call for simplify/equals to prevent hangs on complex expressions
        # Option 1: Using equals() - often faster, tries numerical checks
        @timed(timeout_seconds)
        def _equals_timed():
            return pred_expr.equals(label_expr)

        # Option 2: Simplifying the difference - more rigorous but can be slower
        # @timed(timeout_seconds)
        # def _simplify_timed():
        #     difference = sympy.simplify(pred_expr - label_expr)
        #     return difference == 0

        # Try equals first
        try:
             return _equals_timed()
        except SympyTimeoutError:
             print("Warning: SymPy equals() timed out. Falling back?")
             # Optionally, try simplify as fallback, or just return False
             return False
        # except (TypeError, ValueError) as e:
        #      print(f"Warning: SymPy equals() error: {e}")
        #      return False # Error during comparison

    except Exception as e:
        # Catch other potential errors during comparison
        # print(f"Error during SymPy equivalence check: {e}")
        return False