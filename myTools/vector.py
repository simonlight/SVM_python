'''
Created on Nov 15, 2015

@author: xin
'''
import numpy as np
from numpy import linalg as LA


def L2norm(v):
    return v / LA.norm(v, 2, axis=1).reshape(v.shape[0],1)

def dot(xMatrix2D, yMatrix2D):
    rowNum = xMatrix2D.shape[0]
    res = 0
    for i in range(rowNum):
        res += np.dot(xMatrix2D[i], yMatrix2D[i]) 
    return res