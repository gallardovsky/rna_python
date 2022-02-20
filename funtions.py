#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fpena
"""

import numpy as np

def sigmoide(aux):
    z = (1 + np.exp(-aux))**(-1)
    
    return z

def tanH(aux):
    z = ((2 / (1 + np.exp(-2*aux))) - 1)
    
    return z

def rLU(aux):
    z = max(0.0, aux)
    
    return z