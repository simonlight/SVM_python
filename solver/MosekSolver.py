'''
Created on Nov 15, 2015

@author: xin
'''

import sys,os
import mosek
import numpy as np


def streamprinter(text): 
    sys.stdout.write(text) 
    sys.stdout.flush() 
    
def solveQP(gram, lc, c):
    # Open MOSEK and create an environment and task 
    # Make a MOSEK environment 
    env = mosek.Env () 
    # Attach a printer to the environment 
#     env.set_Stream (mosek.streamtype.log, streamprinter) 
    # Create a task 
    task = env.Task() 
#     task.set_Stream (mosek.streamtype.log, streamprinter) 

    asub  = [ 0] 
    aval  = [ 1.0] 
     
    numcon = 1 
    numvar = len(lc) 
    
    # Append 'numcon' empty constraints. 
    # The constraints will initially have no bounds.   
    task.appendcons(numcon) 
     
    # Append 'numvar' variables. 
    # The variables will initially be fixed at zero (x=0).  
    task.appendvars(numvar) 
     
    for i in range(numvar): 
        # Set the linear term c_j in the objective. 
        task.putcj(i,-lc[i]) 
        # Set the bounds on variable j 
        # blx[j] <= x_j <= bux[j]  
        
        task.putbound(mosek.accmode.var,i,mosek.boundkey.ra, 0.0, c) 
        # Input column j of A  
        task.putacol( i,                  # Variable (column) index. 
                      asub,            # Row index of non-zeros in column j. 
                      aval)            # Non-zero Values of column j.  
    
    for i in range(numcon): 
        task.putbound(mosek.accmode.con,i,mosek.boundkey.ra, 0.0, c) 
     
    # Input the objective sense (minimize/maximize) 
    task.putobjsense(mosek.objsense.maximize) 
     
    # Set up and input quadratic objective 
    qsubi = np.zeros(numvar*(numvar+1)/2, dtype=np.int) 
    qsubj = np.zeros(numvar*(numvar+1)/2, dtype=np.int) 
    qval  = np.zeros(numvar*(numvar+1)/2, dtype=np.float) 
    
    cnt=0
    for i in range(len(lc)):
        for j in range(i+1):
            qsubi[cnt] = int(i);
            
            qsubj[cnt] = int(j);
            qval[cnt] = gram[i][j];
            cnt+=1

    task.putqobj(qsubi,qsubj,qval) 
     
    task.putobjsense(mosek.objsense.minimize) 
     
    # Optimize 
    task.optimize() 
    # Print a summary containing information 
    # about the solution for debugging purposes 
    task.solutionsummary(mosek.streamtype.msg) 
     
    alphas = np.zeros(numvar)
    task.getxx(mosek.soltype.itr, alphas)
    
    return alphas