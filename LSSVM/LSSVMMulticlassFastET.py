'''
Created on Nov 15, 2015

@author: xin
'''

import numpy as np
import json
import sys
from myTools import vector
from solver import MosekSolver
import time
class LSSVMMulticlassFastET(object):
    
    def __init__(self):
        self.optim = None
        self.epochsLatentMax = None
        self.epochsLatentMin = None
        self.cpmax = None
        self.cpmin = None
        self.lbd= None
        self.epsilon = None
        self.tradeoff = None
        self.gazeType = None
        self.hnorm = None
        self.className = None
        self.lossMap = None
        self.dim = None
        self.w = None
        self.nbClass = None
        self.listClass = None
        self.scale = None
        self.region_number = None
    
    def init(self, l):
        raise NotImplementedError

    def psi(self, l):
        raise NotImplementedError

    def enumerateH(self,x):
        raise NotImplementedError
    
    def delta(self, yi, yp, x, h, hstar, hnorm):
        raise NotImplementedError

    
    def valueOf(self, w, feature):
        res = np.inner(w,feature)
             
        return res
#     def valueOf(self, x, y, h, w):
#         res = np.dot(w[y], self.psi(x,h))
#         return res
    
    def renew_prediction(self,val,y,h):
        return val, y, h
        
    def lossAugmentedInference(self,ts):
        valmax = -sys.maxint
        h_range = self.enumerateH(ts.input.x)
        for h in h_range:
            augmente_psi = self.psi(ts.input.x,h)
            for y in self.listClass:
                wy = self.w[y]
#                 print ts.output, y, ts.input.x, h, ts.input.h, self.hnorm                

                loss = self.delta(ts.output, y, ts.input.x, h, ts.input.h, self.hnorm)
#                 augmente = self.valueOf(ts.input.x,y,h,self.w) 
                
                augmente = self.valueOf(wy,augmente_psi) 
                val = loss + augmente
                if(val>valmax):
                    valmax, ypredict, hpredict = self.renew_prediction(val, y, h)

#                     maxdelta = self.delta(ts.output, y, ts.input.x, h, ts.input.h, self.hnorm);
#                     maxvalue = self.valueOf(ts.input.x,y,h,self.w);
        return [ypredict, hpredict]
    
    def prediction(self, lr):
        valmax = -sys.maxint
        lr_x = lr.x
        for h in self.enumerateH(lr_x):
            prediction_psi = self.psi(lr_x, h)
            for y in self.listClass:
                wy = self.w[y]
                val = self.valueOf(wy,prediction_psi);
                if val>valmax:
                    valmax = val
                    ypredict = y
                    hpredict = h
        return ypredict, hpredict

    def hPrediction(self, x, y):
        valmax = -sys.maxint
        wy=self.w[y]
        for h in self.enumerateH(x):
            val = self.valueOf(wy, self.psi(x,h));
            if val>valmax:
                valmax = val
                hpredict = h;
        return hpredict
    
    def trainCCCPCP1Slack(self,l):
        c = 1/ self.lbd
        t = 0
        
        lg = []
        lc = []
        
        gt, ct = self.cuttingPlane(l)

        lg.append(gt)
        lc.append(ct)
        
        gram = None
        xi=0
        
        while (t<self.cpmin) or (t <= self.cpmax and vector.dot(self.w, gt) < ct - xi - self.epsilon):
            print '.',
            lc_length = len(lc)
            if t == self.cpmax:
                print "#max iter"
            if (gram is None):
                gram = np.zeros([lc_length, lc_length])
                for i in xrange(lc_length):
                    for j in xrange(lc_length):
                        gram[i][j] = gram[j][i] = vector.dot(lg[j], lg[i])
                gram += 1e-8*np.eye(lc_length,lc_length)
            else:
                row_num, col_num = gram.shape
                gram = np.concatenate((gram,np.zeros((1,col_num))),axis=0)
                gram = np.concatenate((gram,np.zeros((row_num+1,1))),axis=1)
                lc_length = len(lc)
                for i in range(lc_length):
                    gram[lc_length-1][i] = gram[i][lc_length-1] = vector.dot(lg[lc_length-1], lg[i])
                gram[lc_length-1][lc_length-1] +=1e-8
            alphas = MosekSolver.solveQP(gram, lc, c)
            xi = (vector.dot(alphas, lc) - np.dot(np.dot(alphas,gram), alphas)) / c;
            self.w *= 0
            
            for i in range(len(alphas)):
                self.w += alphas[i]*lg[i]
            t+=1
            gt, ct = self.cuttingPlane(l)
            lg.append(gt)
            lc.append(ct)
        print "cutting plane time:%d"%t
    
    def cuttingPlane(self, l):
        gt = np.zeros((self.nbClass, self.dim))
        ct = 0.0
        n=len(l)
        for ts in l:
            yp, hp = self.lossAugmentedInference(ts) # 
            ct += self.delta(ts.output, yp, ts.input.x,hp,ts.input.h,self.hnorm)#
            psi1 = self.psi(ts.input.x, hp); #
            psi2 = self.psi(ts.input.x, ts.input.h)#

            gt[yp] += -psi1
            gt[ts.output] += psi2

        ct /= n
        gt /= n
        return [gt,ct]
    
    def loss(self, l):
        loss = 0
        for ts in l:
            yp, hp = self.lossAugmentedInference(ts)
            loss += self.delta(ts.output, yp, ts.input.x, hp, ts.input.h, self.hnorm) + self.valueOf(self.w[yp],self.psi(ts.input.x,hp))\
                    -self.valueOf(self.w[ts.output],self.psi(ts.input.x,self.hPrediction(ts.input.x,ts.output)));
#             loss += self.delta(ts.output, yp, ts.input.x, hp, ts.input.h, self.hnorm) + self.valueOf(ts.input.x,yp,hp,self.w)-self.valueOf(ts.input.x,ts.output,self.hPrediction(ts.input.x,ts.output),self.w);
        loss /= len(l);
        return loss;

    def primalObj(self, l):
        obj = self.lbd * vector.dot(self.w,self.w)/2;
        loss = self.loss(l);
        print "lambda*||w||^2= %f\t\tloss= %f"%(obj,loss);
        obj += loss;
        return obj;

    def trainCCCP(self, l):
        el = 0
        decrement = 0
        precObj = 0
        while el<self.epochsLatentMin or (el<=self.epochsLatentMax and decrement < 0):
            print "epoch latent:%d"%el
            self.trainCCCPCP1Slack(l)
            obj = self.primalObj(l)
            decrement = obj - precObj
            print "obj=%f, \tdecrement=%f "%(obj ,decrement)
            precObj = obj;
            el+=1;
            for ts in l:
                ts.input.h = self.hPrediction(ts.input.x,ts.output)
    
    def train(self, l):
        if len(l)==0:
            print "example set length = 0"
            return 
        
        self.nbClass=0
        for ts in l:
            self.nbClass = max(self.nbClass, ts.output)
        self.nbClass+=1
        
        self.listClass = range(self.nbClass)
        
        nb = np.zeros(self.nbClass)
        for ts in l:
            nb[ts.output]+=1
        print "----------------------------------------------------------------------------------------"
        print "Train LSSVM - Mosek \tlambda: %f, \tepochsLatentMax:%d, \tepochsLatentMin%d "%(self.lbd, self.epochsLatentMax, self.epochsLatentMin)
        print "epsilon= %f \t \tcpmax=%d, \tcpmin=%d "%(self.epsilon,self.cpmax,self.cpmin)
        self.init(l)
        
        if (self.optim == 1):
            self.trainCCCP(l)
    
    def setOptim(self, optim):
        self.optim = optim
    def setEpochsLatentMax(self, epochsLatentMax):
        self.epochsLatentMax = epochsLatentMax
    def setEpochsLatentMin(self, epochsLatentMin):
        self.epochsLatentMin = epochsLatentMin
    def setCpmax(self, cpmax):
        self.cpmax = cpmax
    def setCpmin(self, cpmin):
        self.cpmin = cpmin
    def setLambda(self, lbd):
        self.lbd = lbd
    def setEpsilon(self, epsilon):
        self.epsilon = epsilon
    def setTradeoff(self, tradeoff):
        self.tradeoff = tradeoff
    def setGazeType(self, gazeType):
        self.gazeType = gazeType
    def setHnorm(self, hnorm):
        self.hnorm = hnorm
    def setClassName(self, className):
        self.className = className
    def setLossDict(self, lossMapPath):
        with open(lossMapPath) as lmp:
            self.lossMap = json.load(lmp)
    def setScale(self, scale):
        self.scale = scale