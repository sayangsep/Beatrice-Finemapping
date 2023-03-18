#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:08:35 2020

@author: sayan
"""

import torch

def gpu_t(x):
    x = torch.tensor(x.astype(float)).float()
    y = x.to("cpu")
    return(y)
    
