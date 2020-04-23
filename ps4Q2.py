# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:03:16 2020

@author: desai
"""

import numpy as np
class ridge:
    def __init__(self,x,y,alpha):
        self.x=x
        self.y=y
        b=np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@y) ## adding alpha penalty to param matrix 
                                                                #derived from modified loss function
        self.b=b 
        e=y-x@b
        self.e=e
        self.vb=e.var()*np.linalg.inv(x.T@x)
        self.se=np.sqrt(np.diagonal(self.vb))
        self.tstat=self.b/self.se
    def vcov_b(self,e):
        x=self.x
        return e.var(*np.linalg.inv(x.T@x))
