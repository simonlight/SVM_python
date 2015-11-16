'''
Created on Nov 16, 2015

@author: xin
'''
import os
def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

