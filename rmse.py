import numpy as np

def rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)
    sqr_diff= (pred - tar) ** 2  
    mean_sqr_diff = np.mean(sqr_diff)  
    rmse = np.sqrt(mean_sqr_diff) 
    return rmse