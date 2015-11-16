'''
Created on Nov 14, 2015

@author: xin
'''

def file2list(filepath):
    with open(filepath) as fp:
        return [line.strip() for line in fp.readlines()]

def file2FloatList(filepath):
    with open(filepath) as fp:
        return [float(line.strip()) for line in fp.readlines()]