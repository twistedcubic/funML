
import numpy as np

def t_init = 10
def delta = 0.1

'''
backtracking line search (not exact line search).
Backtrack until the function value at new x falls below
the linearly approximated value at x.
Args: 
f is function being optimized.
x0 is base of the backtracking.
alpha \in (0, 0.5], proportion to reduce the steepness of gradient by.
beta \in (0, 1), proportion to shrink step each iteration.
dx is direction to search, if None, use grad f.
Returns:
Updated x.
'''
def lineSearch(f, x0, alpha, beta, dx=None):
    #t is the length being backtracked (shortened)
    t = t_init
    gradf = gradAppx(f, x0)
    if not dx:
        dx = gradf
    while f(x0 + t*dx) > f(x) + alpha * t * np.dot(gradf, dx):
        t *= beta
    return x0 + t*dx

'''
Gradient approximation.
Args: f function to take grad.
numpy array of independent vars of f.
Returns:
gradient vector.
'''
def gradAppx(f, x0):
    grad = np.zeros(len(x0))
    x_ar = np.zeros(len(x0), len(x0))
    for i in range(len(x0)):
        x_ar[i] = x0
        x_ar[i][i] = x0[i] + delta
    # numerical stability
    return np.array(map(lambda x : (f(x) - f(x0))/delta , x_ar))
