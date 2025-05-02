import logging
from pprint import pprint

import numpy as np
import sympy as sp
from config.parameters import (MAX_INT, MAX_LEN, MAX_OPS, MIN_INT,
                               OPERATOR_WEIGHTS, TIMEOUT)
from numpy.random import RandomState
from src.utils import (OPERATORS, c, clean_solution, f, has_undefined,
                       prefix_to_sympy, simplify_through_factorization,
                       sympy_to_prefix, sympy_to_str, try_log, verify_solution,
                       x)
from wrapt_timeout_decorator import timeout


class ODEGenerator:
    def __init__(self):
        self.operators = list(OPERATORS.keys())
        self.arity = {op: OPERATORS[op] for op in self.operators}

        self.una_ops = [op for op in self.operators if self.arity[op] == 1]
        una_weights = np.array([OPERATOR_WEIGHTS[op] for op in self.una_ops], dtype=np.float64)
        self.una_ops_probs = una_weights / np.sum(una_weights)

        self.bin_ops = [op for op in self.operators if self.arity[op] == 2]
        bin_weights = np.array([OPERATOR_WEIGHTS[op] for op in self.bin_ops], dtype=np.float64)
        self.bin_ops_probs = bin_weights / np.sum(bin_weights)

        self.min_int = MIN_INT
        self.max_int = MAX_INT
        self.integers = [i for i in range(self.min_int, self.max_int + 1) if i != 0]
        self.max_ops = MAX_OPS
        self.max_len = MAX_LEN

        self.nl = 1 + 1 + len(self.integers)  # x, c, and integers
        self.p1 = len(self.una_ops)
        self.p2 = len(self.bin_ops)

        try_log(f"Generator initilized with {self.nl} leaf symbols, {self.p1} unary operators, and {self.p2} binary operators.", level=logging.INFO)

        self.ubi_dist = self._generate_ubi_dist(self.max_ops)
        try_log(f"Unary-binary distribution generated up to {self.max_ops} ops.", level=logging.INFO)

    def _generate_ubi_dist(self, max_ops):
        # D[e][n] - e: empty nodes, n: operators

        # Base case: e = 0, n = 0 => 1 possible tree (empty tree)
        D = [[1]]

        # Base case: e = 0, n > 0 operators => impossible
        # D[0][n] = 0
        D[0] += [0] * (2 * max_ops)

        # Base case: e > 0, n = 0 => nl^e possible trees, since you can fill e empty nodes with nl symbols
        for e in range(1, 2 * max_ops + 1):
            # D[e][0] = self.nl ** e
            D.append([self.nl ** e])

        # Calc recurrence by columns
        for n in range(1, 2 * max_ops + 1):
            for e in range(1, 2 * max_ops - n + 1):
                # option 1: place leaf
                # use one empty spot, number of operators remains the same
                place_leaf = self.nl * D[e - 1][n]
                # option 2: place unary operator
                # use one empty spot but make new one (net same), number of operators decreases by 1
                place_unary = self.p1 * D[e][n - 1]
                # option 3: place binary operator
                # use one empty spot and make two new ones (net +1), number of operators decreases by 1
                place_binary = self.p2 * D[e + 1][n - 1]
                # D[e][n] = place_leaf + place_unary + place_binary
                D[e].append(place_leaf + place_unary + place_binary)

        return D

    def _sample_next_pos_ubi(self, num_empty, num_ops, rng: RandomState):
        # total number of ways to complete the tree from current state
        # used for normalization
        total_possibilities = self.ubi_dist[num_empty][num_ops]

        probs = []

        # probability for placing unary operator at each empty node i
        for i in range(num_empty):
            # ways = (ways to fill preceding nodes with leaves) * (unary op choices) * (ways to complete after placing unary)
            unary_at_i = (self.nl ** i) * self.p1 * self.ubi_dist[num_empty - i][num_ops - 1]
            probs.append(unary_at_i / total_possibilities)

        # probability for placing binary operator at each empty node i
        for i in range(num_empty):
            # ways = (ways to fill preceding nodes with leaves) * (binary op choices) * (ways to complete after placing binary)
            binary_at_i = (self.nl ** i) * self.p2 * self.ubi_dist[num_empty - i + 1][num_ops - 1]
            probs.append(binary_at_i / total_possibilities)

        # at this point length of `probs`` is 2 * num_empty

        # convert to numpy for usage with rng.choice
        probs = np.array(probs, dtype=np.float64)
        choice = rng.choice(num_empty * 2, p=probs)

        # first half = unary, second half = binary
        arity = 1 if choice < num_empty else 2
        # extract index of empty node chosen
        choice = choice % num_empty

        return choice, arity

    def _gen_solution_tree(self, num_ops, rng: RandomState):
        # start with empty root node (None is placeholder for node to be filled)
        stack = [None]
        num_empty = 1
        # keep track of leaves placed to the left for position tracking
        left_leaves = 0

        # k = number of operators left to place, count down from num_ops
        for k in range(num_ops, 0, -1):
            # sample next position and arity of operator to place
            skipped_leaves, arity = self._sample_next_pos_ubi(num_empty, k, rng)

            # weighted choice of operator
            if arity == 1:
                op = rng.choice(self.una_ops, p=self.una_ops_probs)
            else:
                op = rng.choice(self.bin_ops, p=self.bin_ops_probs)

            # update number of empty nodes and left leaves
            num_empty += arity - 1 - skipped_leaves
            left_leaves += skipped_leaves

            # find insertion position for operator
            current_leaves_indices = [i for i, v in enumerate(stack) if v is None]
            pos = current_leaves_indices[left_leaves]

            # insert operator at position, fill with None for children
            stack = stack[:pos] + [op] + [None] * arity + stack[pos + 1:]

        num_leaves = stack.count(None)
        leaves = []

        # fill leaves with x, c, and integers then shuffle the sequence
        # c appears once, x and integers (as a whole) appear with equal probability
        for i in range(num_leaves):
            if 'c' not in leaves:
                leaves.append('c')
            else:
                if rng.choice([True, False]):
                    leaves.append('x')
                else:
                    leaves.append(rng.choice(self.integers))

        rng.shuffle(leaves)
        leaves_index = 0

        # replace None with generated leaves sequence
        for i in range(len(stack)):
            if stack[i] is None:
                stack[i] = leaves[leaves_index]
                leaves_index += 1

        return stack

    @timeout(TIMEOUT)
    def generate_clean_solution(self, rng: RandomState):
        # choose a max number of operators to place in the tree
        # [1, max_ops] inclusive
        num_ops = rng.randint(1, self.max_ops + 1)

        # 1. Generate initial solution f(x, c)
        solution_tree = self._gen_solution_tree(num_ops, rng)
        solution_sympy = prefix_to_sympy(solution_tree, self.arity)

        # 2. Regenerate if solution is invalid
        if has_undefined(solution_sympy):
            try_log("Solution contains undefined symbols.")
            return None
        if not solution_sympy.has(x) or not solution_sympy.has(c):
            try_log("Solution does not contain x or c.")
            return None

        # 3. Clean solution
        sol_clean = clean_solution(solution_sympy)

        # 4. Skip if solution is too long
        sol_str = sympy_to_str(sol_clean)
        if len(sol_str) > self.max_len:
            try_log("Solution is too long.")
            return None

        # 5. Convert to prefix
        try:
            sol_prefix = sympy_to_prefix(sol_clean)
        except Exception:
            try_log("Failed to convert to prefix.")
            return None

        return sol_str, sol_clean, " ".join(sol_prefix)

    @timeout(TIMEOUT)
    def generate_ode(self, rng: RandomState, sol_clean: sp.Expr):
        # 5. Solve y = f(x, c) for c
        solve_for_cs = sp.solve(f(x) - sol_clean, c, check=False, simplify=False)
        if not solve_for_cs:
            try_log("No solution for c found.")
            return None

        # should not be trivial
        # should not consist of many paths
        # should not be piecewise
        solve_for_cs = [sol for sol in solve_for_cs if sol.has(x) and type(sol) is not tuple and type(sol) is not sp.Piecewise]
        if not solve_for_cs:
            try_log("All solutions for c are unusable.")
            return None

        # choose random solution
        expr_in_xy = rng.choice(solve_for_cs)

        # 6. Differentiate w.r.t x and simplify
        ode_expr = expr_in_xy.diff(x)
        ode_expr = simplify_through_factorization(ode_expr)

        # 7. Skip if ODE is invalid, or too long
        if has_undefined(ode_expr):
            try_log("ODE contains undefined symbols.")
            return None

        ode_str = sympy_to_str(ode_expr)
        if len(ode_str) > self.max_len:
            try_log("ODE is too long.")
            return None

        # 7. Verify that the pair is valid by subbing in the solution
        # If correct, should return 0
        if not verify_solution(ode_expr, sol_clean, rng):
            try_log("Solution failed to solve ODE")
            return None

        # 8. Convert to prefix and return
        try:
            eq_prefix = sympy_to_prefix(ode_expr)
        except Exception:
            try_log("Failed to convert to prefix.")
            return None

        return ode_str + " = 0", " ".join(eq_prefix)
