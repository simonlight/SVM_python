'''
Created on Nov 15, 2015

@author: xin
'''
from LSSVMMulticlassFastET import LSSVMMulticlassFastET
from myTools import converter
import numpy as np
from metric import metric
class LSSVMMulticlassFastBagMILET(LSSVMMulticlassFastET):
    
    
    def enumerateH(self):
        
        return xrange(self.region_number)

    def psi(self, bagMILx, h):

        return bagMILx.features[h]

    def init(self, l): 
        self.region_number = converter.scale2RowNumber(self.scale)**2    
        self.dim = len(l[0].input.x.features[0])
        self.w = np.zeros((len(self.listClass),self.dim),np.float64)
    
    def getGazeInitRegion(self, ts, scale, mode):
        pass
    
    def getGazeRatio(self, x, h, gazeType):
        if gazeType == "ferrari":
            if self.scale == 100:
                gaze_ratio = self.lossMap[str(self.scale)][x.name+'.txt'][str(self.className)][str(h)]
            else:
                gaze_ratio = self.lossMap[str(self.scale)][x.name][str(self.className)][str(h)]
            return gaze_ratio[0]
        elif gazeType == "stefan":
            feature_path = x.features[h].split('/')
            ETLossFileName = feature_path[-1]
            gaze_ratio = self.lossMap.get(ETLossFileName)
            return gaze_ratio[0]
        else:
            raise NotImplementedError
    
    def delta(self, yi, yp, x, h, hstar, hnorm):
        if hnorm:
            hstar_gaze_ratio = self.getGazeRatio(x, hstar, self.gazeType)
            if yi==1 and yp==1:
                gaze_ratio = self.getGazeRatio(x, h, self.gazeType)
                return (yi^yp)+ self.tradeoff*(gaze_ratio-hstar_gaze_ratio)
            else:
                return (yi^yp)
        else:
            if yi==1 and yp == 1:
                gaze_ratio = self.getGazeRatio(x, h, self.gazeType)
                return (yi^yp)+ self.tradeoff*(1-gaze_ratio)
            else:
                return (yi^yp)
    
    def getAPElement(self, y, yp, score):
        if y == 1 and yp == 1:
            return [1, score]
        elif y == 1 and yp == 0:
            return [1, -score]
        elif y == 0 and yp == 1:
            return [-1, score]
        elif y == 0 and yp == 0:
            return [-1, -score]
    
    def testAP(self, examples):
        label_value_list = []
        for ex in examples:
            yp, hp = self.prediction(ex.input)
            score = self.valueOf(ex.input.x, yp, hp, self.w)
            y = ex.output
            label_value_list.append(self.getAPElement(y, yp, score))
        return metric.getAP(label_value_list)
    
    def testAPRegion(self, examples, detection_fp):
        label_value_list = []
        detection_file = open(detection_fp,"w")
        for ex in examples:
            yp, hp = self.prediction(ex.input)
            score = self.valueOf(self.w[yp],self.psi(ex.input.x,hp))
            y = ex.output
            label_value_list.append(self.getAPElement(y, yp, score))
            detection_file.write("%d,%d,%s,%s\n"%(yp, y, hp, ex.input.x.name))
        detection_file.close()
        return metric.getAP(label_value_list)
    
    