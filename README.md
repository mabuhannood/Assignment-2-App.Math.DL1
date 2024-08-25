# Ex-2: Computational Derivation and Matrix Multiplication

## Overview
In this exercise, you will practice computational derivation and matrix multiplication.

## Objectives
1. Calculate the derivative of a function at a given point.
2. Perform matrix multiplication with tensors.
3. Implement a simple RMSE function.

## The Assignment

### Computational Derivation (1 point)
1. Given a function $f()$, calculate the derivative of the function $f$ at point $x$ using the definition of the derivative and evaluating the slope between points $x$ and $x+h$.
2. So, $f'()$ at point $x$ is ${{f(x+h) - f(x)} \over h}$ when $h$ is _very small_.
3. You can also use the definition ${{f(x+h) - f(x-h)} \over 2h}$. For the purpose of this exercise, you can take $h$ to a small value of your choosing (e.g., $h=0.001$).

### RMSE Implementation (1 points)
1. Follow the instructions provided in the `rmse.py` file.
2. RMSE (Root Mean Square Error) = $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2}$, where $y_i$ are the true values, $\hat{y_i}$ are the predicted values, and $n$ is the total number of samples.

### Matrix Multiplication (3 points)
1. Follow the instructions provided in the `tensor_multiplication.py` file.
2. Your solution should be concise, typically a one-liner.
3. We introduce the transpose operator `W.T`, which switches the dimensions around. This is to conform with conventions of how the weight matrix is typically written in PyTorch and to ensure that when we perform matrix multiplication, the inner dimensions match.
