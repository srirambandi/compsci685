# this script generates mixed expressions data - (unary, binary)
import random


# defining Operators(unary, binary), variables, constants, parameters
p1 = ['sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh']
p2 = ['+', '-', '*', '/']
l = ['x', '-5', '-4', '-3', '-2', '-1', '1', '2', '3', '4', '5']
n = 15
dataset_size = 1



# tree node object
class Node():
    def __init__(self, val=None):
        self.left = None
        self.right = None
        self.val = val


# pre-order traversal of the tree; implements prefix notation of an expression(easily readable)
def prefix(tree):
    if (tree.val != None):
        print(tree.val, end=' ')
        if (tree.left != None):
            prefix(tree.left)
        if (tree.right != None):
            prefix(tree.right)

# in-order traversal of the tree; implements infix notation
def infix(tree):
    if (tree.val != None):
        if (tree.val in p1 or tree.val in p2):
            print('(', end='')
        if (tree.left != None):
            infix(tree.left)
        print(tree.val, end='')
        if (tree.right != None):
            infix(tree.right)
        if (tree.val in p1 or tree.val in p2):
            print(')', end='')



D = [[-1 for i in range(n+1)] for j in range(2*n+2)]
for i in range(2*n+2):
    D[i][0] = 1
for j in range(n+1):
    D[0][j] = 0
D[0][0] = 0

# defining D, different possible trees given a state(e, n)
def recur_d(e, n):
    if D[e][n] == -1:
        D[e][n] = recur_d(e-1, n) + recur_d(e, n-1) + recur_d(e+1, n-1)
    return D[e][n]

for i in range(n+2):
    for j in range(n+1):
        # print(i,j, end='--')
        D[i][j] = recur_d(i, j)
        # print(D[i][j])

# L prabability distribution; returns position k in {0...e-1},
# arity(unary or binary node) a = {1, 2}
def sample_L(e, n):

    I = list(range(2*e))
    P = []

    for i in range(e):
        p = D[e-i][n-1]/D[e][n]
        P.append(p)

    for i in range(e):
        p = D[e-i+1][n-1]/D[e][n]
        P.append(p)

    pos = random.choices(I, weights=P, k=1)[0]
    k = pos%e
    a = int(pos/e) + 1

    return (k, a)



for i in range(dataset_size):

    tree = Node()
    e = [tree]

    while n>0:

        k, a = sample_L(len(e), n)

        for i in range(0, k):
            node = e[i]
            node.val = random.choice(l)

        node = e[k]
        if (a == 1):
            node.val = random.choice(p1)
            node.right = Node()
            e.append(node.right)
        else:
            node.val = random.choice(p2)
            node.left = Node()
            node.right = Node()
            e.append(node.left)
            e.append(node.right)

        del e[0:k+1]
        n = n - 1

    for i in range(len(e)):
        node = e[i]
        node.val = random.choice(l)

    prefix(tree)
    print()
    infix(tree)
    print()
