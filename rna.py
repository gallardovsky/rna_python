#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fpena
"""

import numpy as np
import random as rm
import time as t
import matplotlib.pyplot as plt
import warnings

from funtions import *

'ANN type=feedforward/backpropagation online'
'version 2.2.1'
'nx = # vectores de entrada'
'n1 = # neuronas en la capa de entrada'
'n2 = # neuronas en la capa escondida'
'n3 = # neuronas en la capa de salida'
'alpha = razon de aprendizaje'
'epoc = # de epocas'

#suppress warnings
warnings.filterwarnings('ignore')

print ('\t \t \t \t \t \t ANN ver. 3.01')

# Configuracion de la red
nx = 10; n1 = 9; n2 = 1000; n3 = 10; alpha = 1000; epoch = 1000;

#plt.axis([0, epoch, 0, 2])
#plt.ion()

# Ingreso de datos
X = np.loadtxt('datos/datos1.dat')
Y = np.loadtxt('datos/salida.dat')
sort = np.array(range(nx))

# Feedforward
a2 = np.zeros(n2, float)
a3 = np.zeros(n3, float)

# Deltas
d3 = np.zeros(n3, float)
d2 = np.zeros(n2, float)

print ('Entrenar red: 1')
print ('Cargar pesos: 2')

option = int(input())

###############################################################################
if (option == 1):
    # time
    tStart = t.time()
    
    # weigths
    omega_1 = np.random.rand(n1, n2)
    omega_1 = np.vstack([omega_1, np.ones(n2)])
    
    omega_2 = np.random.rand(n2, n3)
    omega_2 = np.vstack([omega_2, np.ones(n3)])
    
    # entrenamiento
    for epoc in range(epoch):
        rm.shuffle(sort)
        
        for k in range(nx):
            Error = []
            e = 0.0
            
            # capa E-O
            for i in range(n2):
                a = 0.0
                
                for j in range(n1):
                    a = X[sort[k], j] * omega_1[j, i] + a
                    
                a2[i] = sigmoide(a + omega_1[n1, i])
                
            # capa O-S
            for i in range(n3):
                a = 0.0
                
                for j in range(n2):
                    a = a2[j] * omega_2[j, i] + a
                
                a3[i] = sigmoide(a + omega_2[n2, i])
            
            # Calculo de deltas S-O
            for i in range(n3):
                d3[i] = a3[i] * (1 - a3[i]) * (-(Y[sort[k], i] - a3[i]))
                
            # Calculo de deltas O-E
            for i in range(n2):
                aux = 0.0
                
                for j in range(n3):
                    aux = omega_2[i, j] * d3[j] + aux
                    
                d2[i] = a2[i] * (1 - a2[i]) * aux
            
            # Actualización de weights
            for i in range(n2):
                for j in range(n1):
                    omega_1[j, i] = omega_1[j, i] - alpha * X[sort[k], j] * d2[i]
                
                omega_1[n1, i] = omega_1[n1, i] - alpha * d2[i]
                
            for i in range(n3):
                for j in range(n2):
                    omega_2[j, i] = omega_2[j, i] - alpha * a2[j] * d3[i]
                    
                omega_2[n2, i] = omega_2[n2, i] - alpha * d3[i]
                
        print('Epoca: ', epoc)
        tEnd = t.time()
        print('Tiempo de entrenamiento:', tEnd - tStart)
        

    # guardar weights entrenados            
    np.savetxt('weights/omega_1.dat', omega_1, fmt = '%.8e')
    np.savetxt('weights/omega_2.dat', omega_2, fmt = '%.8e')
    print('Pesos salvados')

###############################################################################   

if (option == 2):
    omega_1 = np.loadtxt('weights/omega_1.dat')
    omega_2 = np.loadtxt('weights/omega_2.dat')
           
print('Ingresar vector de verificación, de dimension:', n1)

x = np.loadtxt('vectorPrueba.dat')

print(x)

# feedforward E-O
for i in range(n2):
    a = 0.0
    
    for j in range(n1):
        a = x[j] * omega_1[j, i] + a
        
    a2[i] = sigmoide(a + omega_1[n1, i])
    
# feedforward O-S
for i in range(n3):
    a = 0.0
    
    for j in range(n2):
        a = a2[j] * omega_2[j, i] + a
        
    a3[i] = sigmoide(a + omega_2[n2, i])
    
print('Valor =', a3)