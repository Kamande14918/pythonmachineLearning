import math
# HyperParameters of the optimization algorithm
alpha = 0.01
beta = 0.9


# Objective function 

def obj_func(x):
    return x*x -4 * x +4

# Gradient of the objective function


def grad(x):
    return 2*x -4


# Parameter of the objective function 
x = 0

# Number of iterations
iterations =  0

v = 0
while(1):
    iterations = 1
    v = beta * v + (1-beta) * grad(x)
    
    x_prev = x
    x = x - alpha *v
    
    print("Value of objective function on iteration",iterations, "is",x)
    
    if x_prev == x:
        print("Done optimizing the objective function. ")
        break
    