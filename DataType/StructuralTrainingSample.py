'''
Created on Nov 15, 2015

@author: xin
'''

class StructuralTrainingSample(object):
    """Bag object
    """
    def __init__ (self, input, output):
        self.input = input
        self.output = output
    def __str__(self, *args, **kwargs):
        return "StructuralTrainingSample | input:%s, output:%s"%(self.input, self.output)        
