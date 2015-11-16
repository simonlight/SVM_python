'''
Created on Nov 14, 2015

@author: xin
'''

class BagMIL(object):
    """Bag object
    """
    def __init__ (self, name, label, features):
        self.name = name
        self.label = label
        self.features = features
    def __str__(self, *args, **kwargs):
        return "bag | name:%s, label:%s, feature number:%d"%(self.name, self.label, len(self.features))        
        