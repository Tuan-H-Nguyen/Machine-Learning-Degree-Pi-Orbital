#%%
"""
Full DPO model

For plotting training and testing of the model on 
datasets of 132 data points and 116 data points, respectively.

There are single run for plotting and multiple (20) runs for
determine the error and parameters of the model (scroll till the end)
"""

import sys
import os
module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

import time
import argparse

from utils.plot_utility import scatter_plot
from trainer import Trainer

data = pd.read_csv("sample_DATA\\total_dataset.csv")
data = data.drop("smiles",axis=1)

start = time.time()

intervals_1 = [1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
intervals_2 = [1.0,2.0,3.0,4.0,5.0]
parameters_list = ['a','b','c','d','a*','d*']
parameter_true_value= [0.05,-0.25,0.33,0.33,0.5,0.15]

def pearson_corr(X,Y):
    X_m = np.mean(X)*np.ones(len(X)) - X
    Y_m = np.mean(Y)*np.ones(len(Y)) - Y
    
    R = np.sqrt(np.sum(X_m**2))*np.sqrt(np.sum(Y_m**2))
    R = np.sum(np.dot(X_m.T,Y_m)) / R
    
    return R

def RMSD(Y_hat,Y):
    rmsd = (Y_hat - Y)**2
    rmsd = np.mean(rmsd)
    rmsd = np.sqrt(rmsd)
    return rmsd

#%%
"""
Repeat training and testing the model 20 times on 
different data splits.
132 training samples and 116 instances for testing  
"""

repeatStandardTrainer = Trainer(
    data_intervals = intervals_1,
    data = data, 
    SMILES = False,
    descriptor_col = "DPO equation",
    target_col = "BG",
    pred_col = ["EA","IP"],
    global_seed = 132)

model = repeatStandardTrainer.repeat_standard_train(
    20)

################
# print result #
################

print("RESULT for BG: RMSE = {}".format(
        repeatStandardTrainer.test_rmse))
for i,elec_prop in enumerate(repeatStandardTrainer.pred_col):
    print("RESULT for {}: RMSE = {}".format(
        elec_prop,repeatStandardTrainer.pred_test_rmse[i]))
    
print("\nML-BASED DPO MODEL CONFIGURATION")

print("DPO PARAMETERS: ")
for i,param in enumerate(model.parameters_list):
    print(param," = ", model.parameter_values[i])

print("\nLINEAR WEIGHT ([bias, slope])")
print(repeatStandardTrainer.target_col,model.outputs[0].weight)

for i, prop in enumerate(repeatStandardTrainer.pred_col):
    print(prop, model.outputs[i+1].weight) 
    
"""
RESULT for BG: RMSE = 0.0999409670503045
RESULT for EA: RMSE = 0.0661226440459451
RESULT for IP: RMSE = 0.056507974509536416

ML-BASED DPO MODEL CONFIGURATION
DPO PARAMETERS: 
a  =  0.07961652419071312
b  =  -0.11950160471248081
c  =  0.31640474421379633
d  =  0.2936669499382258
af  =  0.40256479789420957
df  =  0.15907743691558024

LINEAR WEIGHT ([bias, slope])
BG [ 5.00801216 -0.76211326]
EA [1.1724194  0.40869462]
IP [ 6.18037363 -0.35337488]
"""

#%%
"""
Train and test the model for 1 times.
132 training samples and 116 instances for testing  
"""
standardTrainer = Trainer(
    data_intervals = intervals_1,
    descriptor_col = "DPO equation",
    target_col = "BG",
    pred_col = ["EA","IP"],
    SMILES=False,
    data = data, global_seed = 1998)

model = standardTrainer.standard_train()

#%%
# only plotting results from here

plot_labels = ["(A)","(B)","(C)"]
plot_labels_position = [
    np.array([0.6, 1.6]),
    np.array([3.7, 1.25]),
    np.array([0.7, 4.6])
]
offset = [
    np.array([0.0, 0.4]),
    np.array([0.5, 0.2]),
    np.array([0.0, 0.2])
]
plot_labels_position2 = [
    np.array([3.2, 1.6]),
    np.array([2.5, 1.4]),
    np.array([5.25, 4.4])
]
offset2 = [
    np.array([1.0, 0.3]),
    np.array([0.75, 0.2]),
    np.array([0.55, 0.15])
]
limit2 = [
    (1.5,4.5), (1.25,3.5), (4.3,6.0)
]
y_axes_labels = [
    "HOMO-LUMO gap", 
    "Electron Affinity",
    "Ionization Potential"
    ]
x_axes_labels = [None,None,"DPO"]

props = ["BG","EA","IP"]

for i,struc_prop in enumerate(props):
    X = np.array(standardTrainer.train_set.loc[:][standardTrainer.dpo_col])
    Y = np.array(standardTrainer.train_set.loc[:][struc_prop])
    
    X_test = np.array(standardTrainer.test_set.loc[:][standardTrainer.dpo_col])
    Y_test = np.array(standardTrainer.test_set.loc[:][struc_prop])

    X_train, Y_hat_train = model.predict(X,task = i)
    X_test, X_hat_test = model.predict(X_test,task = i)
    
    #plotting correlation for training set
    train_plot = scatter_plot()

    train_plot.add_plot(
        X_train, Y,
        plot_line=True,
        weight = model.outputs[i].weight,
        line_color = "black",
        
        xlabel = x_axes_labels[i],
        ylabel = y_axes_labels[i] + " (eV)",
        
        xlim = (0.5,5.0)
    )

    train_plot.add_text(
        *list(plot_labels_position[i] + 2*offset[i]), 
        plot_labels[i]
    )

    train_plot.add_text(
        *list(plot_labels_position[i] + offset[i]),
        "R$^2$ = {:.2f}".format(
            pearson_corr(X_train, Y)**2
        )
    )

    train_plot.add_text(
        *list(plot_labels_position[i]),
        "RMSD = {:.2f}eV".format(
            RMSD(Y_hat_train,Y)
        )
    )
    train_plot.save_fig(
        "[result]//DPOvsTrueValue_full_"+plot_labels[i]+".jpeg",dpi=600
    )
    
    #plot testing plot
      
    _, Y_hat_test = model.predict(X_test,task = i)

    true_vs_pred1 = scatter_plot()

    true_vs_pred1.add_plot(
        limit2[i],limit2[i],
        scatter = False,
        plot_line =True, 
        weight = (0,1),
        line_color = "black",
        linewidth = 0.5
    )

    true_vs_pred1.add_plot(
        Y_test,Y_hat_test,
        equal_aspect=True,
        xlim = limit2[i],
        ylim = limit2[i],
        x_major_tick = 0.5, x_minor_tick = 0.1,
        y_major_tick = 0.5, y_minor_tick = 0.1,
        
        xlabel = "Calculated "+y_axes_labels[i]+" (eV)",
        ylabel = "QSPR prediction(eV)"
    )

    true_vs_pred1.add_text(
        *list(plot_labels_position2[i]), 
        "RMSD = {:.2f}eV".format(
        RMSD(Y_test,Y_hat_test)
        )
    )

    true_vs_pred1.add_text(
        *list(plot_labels_position2[i]+offset2[i]),
        plot_labels[i]
    )

    true_vs_pred1.save_fig(
        "[result]//true_vs_pred_full"+plot_labels[i]+".jpeg",dpi=600
    )

# %%
