
import itertools
import logging

import numpy as np
import sympy as sp
from config.parameters import LOGGING_LEVEL

logging.basicConfig(level=LOGGING_LEVEL, format='[%(levelname)s] %(message)s')
logger = logging.getLogger()

OPERATORS = {
    'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'pow': 2,
    'sin': 1, 'cos': 1, 'exp': 1, 'log': 1, 'sqrt': 1,
}

x = sp.Symbol('x', real=True, nonzero=True)
c = sp.Symbol('c', real=True)
f = sp.Function('f', real=True, nonzero=True)

SYMPY_OPERATOR_MAP = {
    sp.Add: 'add', sp.Mul: 'mul', sp.Pow: 'pow',
    sp.sin: 'sin', sp.cos: 'cos', sp.sqrt: 'sqrt',
    sp.exp: 'exp', sp.log: 'log',
    # sub, div, handled separately
}


def try_log(msg, level=logging.DEBUG):
    try:
        logger.log(level, msg)
    except BrokenPipeError:
        pass


def prefix_to_sympy(prefix, arity_map):
    def helper(tokens):
        # pop the first token
        token = tokens.pop(0)

        if token == 'x':
            return x
        elif token == 'c':
            return c
        elif token == 'y':
            return f
        elif isinstance(token, np.int32):
            return sp.Integer(token)
        elif token in arity_map:
            arity = arity_map[token]
            args = []
            for _ in range(arity):
                arg = helper(tokens)
                args.append(arg)

            if token == 'add':
                return sp.Add(*args)
            elif token == 'sub':
                return sp.Add(args[0], sp.Mul(-1, args[1]))
            elif token == 'mul':
                return sp.Mul(*args)
            elif token == 'div':
                return sp.Mul(args[0], sp.Pow(args[1], -1))
            elif token == 'pow':
                return sp.Pow(*args)
            elif token == 'sqrt':
                return sp.sqrt(args[0])
            elif token == 'sin':
                return sp.sin(args[0])
            elif token == 'cos':
                return sp.cos(args[0])
            elif token == 'exp':
                return sp.exp(args[0])
            elif token == 'log':
                return sp.log(args[0])
            else:
                try_log(f"Unknown operator: {token}")
        else:
            try_log(f"Unknown token: {token}")

    tokens = list(prefix)
    return helper(tokens)


def sympy_to_prefix(expr):
    def helper(expr: sp.Expr):
        # handle non-operator cases first
        if expr == x:
            return ['x']
        elif expr == c:
            return ['c']
        elif expr.func == f:
            return ['y']
        if isinstance(expr, sp.Integer):
            return [str(expr.p)]  # .p gets the integer value
        elif isinstance(expr, sp.Rational):
            return ['div', str(expr.p), str(expr.q)]  # .p and .q get the numerator and denominator
        elif isinstance(expr, sp.Derivative):
            return ['y\'']
        # special handle add and mul as they can also be sub or div + they can take 3+ arguments
        elif isinstance(expr, sp.Add):
            prefix = helper(expr.args[0])
            for arg in expr.args[1:]:
                # check if this arg is a sub
                # form: "a + b * (-1)" or sp.Add(a, sp.Mul(-1, b))
                if isinstance(arg, sp.Mul) and len(arg.args) == 2 and arg.args[0] == -1:
                    prefix = ['sub'] + prefix + helper(arg.args[1])
                else:
                    prefix = ['add'] + prefix + helper(arg)
            return prefix
        elif isinstance(expr, sp.Mul):
            # check for form "a * b^(-1)" or sp.Mul(a, sp.Pow(b, -1))
            prefix = helper(expr.args[0])
            for arg in expr.args[1:]:
                # check if this arg is a div
                # form: "a * b^(-1)" or sp.Mul(a, sp.Pow(b, -1))
                if isinstance(arg, sp.Pow) and len(arg.args) == 2 and arg.args[1] == -1:
                    prefix = ['div'] + prefix + helper(arg.args[0])
                else:
                    prefix = ['mul'] + prefix + helper(arg)
            return prefix
        # handle other standard operator
        elif expr.func in SYMPY_OPERATOR_MAP:
            op_name = SYMPY_OPERATOR_MAP[expr.func]
            prefix = [op_name]
            for arg in expr.args:
                arg_prefix = helper(arg)
                prefix.extend(arg_prefix)
            return prefix
        else:
            try_log(f"Unknown expression: {expr.func}")
            return []

    return helper(expr)


def has_undefined(expr):
    # undefined includes (positive/negative) infinity, NaN, complex infinity, and complex numbers
    return expr.has(sp.oo, -sp.oo, sp.nan, sp.zoo, sp.I)


def fuse_c_only(expr: sp.Expr):
    # extract sub-expressions from outer to inner
    for sub_expr in sp.preorder_traversal(expr):
        sub_expr: sp.Expr

        # check if sub_expr has c only and no x
        if c in sub_expr.free_symbols and not x in sub_expr.free_symbols:
            expr = expr.subs(sub_expr, c)
            break

    return expr


def c_absorb(expr: sp.Expr):
    # extract sub-expressions from outer to inner
    for sub_expr in sp.preorder_traversal(expr):
        sub_expr: sp.Expr

        # check if sub_expr has c as one of its direct arguments
        if c in sub_expr.args:
            break  # retain sub_expr for further processing

    # property can only be applied to adding or multiplication
    if not (sub_expr.is_Add or sub_expr.is_Mul):
        return expr

    # find purely constant arguments (except c)
    to_remove = [arg for arg in sub_expr.args if len(arg.free_symbols) == 0]

    if len(to_remove) > 0:
        # reconstruct the sub_expr without the constant arguments through substitution
        # e.g. c * x * 5 => c subbed with c / 5 => c / 5 * x * 5 => c * x
        # e.g. c + x + 5 => c subbed with c - 5 => c - 5 + x + 5 => c + x
        sub_expr_constants = sub_expr.func(*to_remove)
        reverse_sub_expr = (c - sub_expr_constants) if sub_expr.is_Add else (c / sub_expr_constants)
        expr = expr.subs(c, reverse_sub_expr)

    return expr


def clean_solution(expr):
    # 1. Fuse c-only sub-expressions
    # e.g. (sin(c) + 5) * x * 5 => c * x * 5
    expr = fuse_c_only(expr)

    # 2. Absorbs constants being multiplied or added to c
    # e.g. c * x * 5 => c * x
    expr = c_absorb(expr)

    return expr


def sympy_to_str(expr: sp.Expr):
    # quick function to print a more readable string representation of sympy expressions
    out_str = str(expr)
    out_str = out_str.replace('**', '^')
    out_str = out_str.replace('log', 'ln')
    out_str = out_str.replace('Derivative(f(x), x)', 'y\'')
    out_str = out_str.replace('f(x)', 'y')
    out_str = out_str.replace('exp(', 'e^(')
    return out_str


def simplify_through_factorization(expr: sp.Expr):
    factorized: sp.Expr = sp.factor(expr)

    # factorization not possible
    if not factorized.is_Mul:
        return expr

    keep_factors = []

    # iterate and only keep factors that affect the solution
    for arg in factorized.args:
        if arg.is_nonzero:
            continue

        if arg.has(sp.Derivative(f(x), x)):
            keep_factors.append(arg)

    return keep_factors[0] if len(keep_factors) == 1 else sp.Mul(*keep_factors)


def verify_solution(expr: sp.Expr, solution: sp.Expr, rng: np.random.RandomState):
    ZERO_TOLERANCE = 1e-15
    PASS_REQUIREMENT = 0.85

    # try simple substitution first
    verify = sp.simplify(expr.subs(f(x), solution)).doit()
    if verify == 0:
        return True

    # if simple substitution fails, substitute free_symbols with random values
    # evaluate multiple times, pass if smaller than ZERO_TOLERANCE 85% of the time
    free_symbols = expr.free_symbols
    eval_results = []

    # assume 5 different random values for each free symbol
    random_pool = []
    for _ in range(5**len(free_symbols)):
        # generate random values for each free symbol
        num = rng.random() + 0.1  # avoid zero
        num = num if np.random.choice([True, False]) else -num  # 50% chance to be negative
        random_pool.append(num)

    # generate all possible combinations of random values for each free symbol
    for values in itertools.product(random_pool, repeat=len(free_symbols)):
        verify = expr.subs(zip(free_symbols, values)).doit()

        try:
            evaled = float(sp.Abs(verify.evalf()))
            if evaled < ZERO_TOLERANCE:
                eval_results.append(1)
            else:
                eval_results.append(0)
        except:
            eval_results.append(0)

    # calculate pass rate
    pass_rate = sum(eval_results) / len(eval_results)
    if pass_rate >= PASS_REQUIREMENT:
        return True
    return False
