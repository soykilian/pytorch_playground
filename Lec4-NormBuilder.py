# Writted by Dr. Tolga Soyata    9/4/2023
# This program demonstrates the idea of higher order functions in Python

import numpy as np

def norm_builder(n):
    if n == -1:
        def skeleton_norm(x):
            return max(x)
    else:
        def skeleton_norm(x):
            s=sum(abs(x)**n)
            return  s** 1.0/n
    return skeleton_norm


# 20 random data points
x=np.random.randn(20)

print(f"X = {x}\n")
norm1=norm_builder(1)
norm2=norm_builder(2)
norminf=norm_builder(-1)

X1=norm1(x)
X2=norm2(x)
Xinf=norminf(x)

print(f"1 norm of X = {X1}")
print(f"2 norm of X = {X2}")
print(f"infinite norm of X = {Xinf}")

