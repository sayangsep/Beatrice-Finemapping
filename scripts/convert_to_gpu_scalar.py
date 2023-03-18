# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:08:35 2020

@author: sayan
"""

import torch

def gpu_ts(x):
    x = torch.tensor(x).float()
    y = x.to("cpu")
    return(y)
    
