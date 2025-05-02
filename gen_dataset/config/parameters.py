import logging

LOGGING_LEVEL = logging.INFO

# Define the appearance weights for each operator
# Note: at least one unary and one binary operator must have weight > 0
OPERATOR_WEIGHTS = {
    'add': 1.0, 'sub': 1.0, 'mul': 1.0, 'div': 1.0, 'pow': 1.0,
    'sin': 1.0, 'cos': 1.0, 'exp': 1.0, 'log': 1.0, 'sqrt': 1.0,
}
MIN_INT = -5    # minimum integer value to appear in leaves
MAX_INT = 5     # maximum integer value to appear in leaves
MAX_OPS = 5    # maximum number of operators in the expression
MAX_LEN = 150   # maximum length of the expression in characters
TIMEOUT = 5     # timeout for for each ode generation (in seconds)
