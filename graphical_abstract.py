#%%
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

from DPO_model.utils.stratified_splitting import replaceTrunDPO, sampling
from DPO_model.utils.plot_utility import scatter_plot, font_legend, annotate
from DPO_model.trainer import Trainer
from DPO_model.gd_dpo.GDDPO import GD_DPO

data = pd.read_csv("sample_DATA\\total_dataset.csv")
data = data.drop("DPO equation",axis=1)

start = time.time()

intervals_1 = [1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
intervals_2 = [1.0,2.0,3.0,4.0,5.0]
parameters_list = ['a','b','c','d','a*','d*']
parameter_true_value= [0.05,-0.25,0.33,0.33,0.5,0.15]

def pearson_corr(X,Y):
    X_m = np.mean(X)*np.ones(len(X)) - X
    Y_m = np.mean(Y)*np.ones(len(Y)) - Y
    
    R = np.sum(X_m**2)*np.sum(Y_m**2)
    R = np.sqrt(R)
    R = np.sum(np.dot(X_m.T,Y_m)) / R
    
    return R

def RMSD(Y_hat,Y):
    rmsd = (Y_hat - Y)**2
    rmsd = np.mean(rmsd)
    rmsd = np.sqrt(rmsd)
    return rmsd
pred_props = ["EA","IP"]
#%%

standardTrainer = Trainer(
    data_intervals = intervals_1,
    SMILES = True,
    descriptor_col = "smiles",
    target_col = "BG",
    data = data, global_seed = 100)

main_model, pred_models = standardTrainer.standard_train()

# %%
limit2 = [
    (1.5,4.5), (1.25,3.5), (4.3,6.0)
]
y_axes_labels = [
    "HOMO-LUMO gap", 
    "Electron Affinity",
    "Ionization Potential"
    ]
x_axes_labels = [None,None,"DPO"]

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

struc_prop = "BG"
i=0

X = np.array(standardTrainer.train_set.loc[:][standardTrainer.dpo_col])
Y = np.array(standardTrainer.train_set.loc[:][struc_prop])
X_test = np.array(standardTrainer.test_set.loc[:][standardTrainer.dpo_col])
Y_test = np.array(standardTrainer.test_set.loc[:][struc_prop])

model = GD_DPO(
    parameters_list = ['a','b','c','d','af','df'],
    parameter_values = main_model.parameter_values,
    constant_list = ['bf','cf','b0'],
    constant_values = [0,0,0]
)

_,_ = model.feedforward(X,Y)

_, Y_hat_test = model.predict(X_test)

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

annotate = {'fontname':'Times New Roman','weight':'bold','size':16}

true_vs_pred1.ax.text(
    2.9,1.6, 
    "RMSD = {:.2f}eV".format(
    RMSD(Y_test,Y_hat_test)
    ),
    **annotate
)

true_vs_pred1.save_fig(
    "[result]//graphical_abstract.jpeg",dpi=600
)

# %%
