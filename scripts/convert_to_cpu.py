#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:50:10 2020

@author: sayan
"""
import torch
def cpu(x):
    
    x = x.to(torch.device("cpu"))
    
    return(x)
    