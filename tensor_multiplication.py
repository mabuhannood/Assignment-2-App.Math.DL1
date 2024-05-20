import torch

def create_tensor_of_val(dimensions, val):
    res = torch.ones(dimensions) * val
    return res

def calculate_elementwise_product(A, B):
    if A.shape != B.shape:
        raise RuntimeError("dimensions of A and B should be the same.")
    res = A * B 
    return res


def calculate_matrix_product(X, W):
    res = torch.matmul(X, W.T)
    return res

def calculate_matrix_prod_with_bias(X, W, b):
    mat_mul = torch.matmul(X, W.T)
    res = mat_mul + b
    return res

def calculate_activation(sum_total):
    res = torch.heaviside(sum_total, torch.tensor(0.0))
    return res

def calculate_output(X, W, b):
    sum_total = calculate_matrix_prod_with_bias(X, W, b)
    res = calculate_activation(sum_total)
    return res