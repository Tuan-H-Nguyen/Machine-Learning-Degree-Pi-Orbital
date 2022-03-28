#%%
import sys
import os
module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

from utils.plot_utility import scatter_plot, font_legend
from trainer import Trainer

data = pd.read_csv("sample_DATA\\total_dataset.csv")

#dropping columns, not necessary, but to ensure nothing
# went well but unexpected
fullDpoData = data.drop("smiles",axis = 1)
trunDpoData = data.drop("DPO equation",axis=1)

intervals_1 = [1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
intervals_2 = [1.0,2.0,3.0,4.0,5.0]
parameters_list = ['a','b','c','d','a*','d*']
parameter_true_value= [0.05,-0.25,0.33,0.33,0.5,0.15]

train_ratio = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0]
trainset_size_list = [int(round(132*ratio)) for ratio in train_ratio]

pred_props = ["EA","IP"]
#%%
"""
Repeat training model on series of training sets with 
increasing size
FULL DPO descriptor

preferred seed:  100
"""
fullRepeatTrain = Trainer(
    data = fullDpoData, 
    SMILES = False, 
    descriptor_col = "DPO equation",
    target_col = "BG",
    pred_col = pred_props,
    global_seed = 1998,
    data_intervals = intervals_1
    )
fullRepeatTrain.repeat_train(
    n=10,
    train_set_size_list = trainset_size_list
    )

"""
Repeat training model on series of training sets with 
increasing size
TRUNCATED DPO descriptor
"""
trunRepeatTrain = Trainer(
    data = trunDpoData, 
    SMILES = True, 
    descriptor_col = "smiles",
    target_col = "BG",
    pred_col = pred_props,
    global_seed = 1147,
    data_intervals = intervals_1
    )
trunRepeatTrain.repeat_train(
    n=10,
    train_set_size_list = trainset_size_list
    )

#%%
_y_tick_ = [
    [0.05,0.01],
    [0.01,0.005],
    [0.01,0.005]
    ]
plot_label = ["(A)","(B)","(C)"]
label_position = [
    [125, 0.23],
    [125, 0.16],
    [125, 0.13]
]
for i, elec_prop in enumerate(["BG","EA","IP"]):
    plot_rmse = scatter_plot()
    
    if i == 0:
        Yfull = fullRepeatTrain.test_rmse
        Ytrun = trunRepeatTrain.test_rmse
    else:
        Yfull = fullRepeatTrain.pred_test_rmse[:,i-1]
        Ytrun = trunRepeatTrain.pred_test_rmse[:,i-1]

    plot_rmse.add_plot(
        x = trainset_size_list,
        y = Yfull,
        plot_line=True,
        xticks_format = 0, x_major_tick = 10, x_minor_tick =5,
        y_major_tick = 0.02, y_minor_tick = 0.005,
        
        scatter_marker = "v",
        line_color ="blue",
        scatter_color ="blue",
        
        label = 'The full DPO model',
        xlabel = 'Training set size (data points)',
        ylabel = 'Root mean square deviation (eV)'
    )
    
    plot_rmse.add_plot(
        x = trainset_size_list,
        y = Ytrun,
        plot_line=True,
        xticks_format = 0, x_major_tick = 10, x_minor_tick =5,
        y_major_tick = _y_tick_[i][0], 
        y_minor_tick = _y_tick_[i][1],
        
        scatter_marker = "o",
        line_color ="red",
        scatter_color ="red",

        label = 'The truncated DPO model',
        xlabel = 'Training set size (data points)',
        ylabel = 'Root mean square deviation (eV)',
    )
    #plot_rmse.ax.grid()
    if i == 0:
        plot_rmse.ax.legend(
            prop = font_legend
            )
    plot_rmse.add_text(*label_position[i],plot_label[i])
    plot_rmse.save_fig('[result]/RepeatRMSE_'+elec_prop+'_.jpeg',dpi=600)
# %%
params = fullRepeatTrain.parameters
plot_fp = scatter_plot()
parameters_list = ['a','b','c','d','a*','d*']
colors = ['r','blue','black','orange','purple','green']
for i in range(params.shape[1]):
    plot_fp.add_plot(
        [0]+trainset_size_list, params[:,i],plot_line=True,
        label = parameters_list[i],
        scatter_color = colors[i],line_color = colors[i]
    )
    plot_fp.add_plot(
        135 if i != 3 else 137, parameter_true_value[i],
        scatter_marker = '*',label = parameters_list[i],
        scatter_color = colors[i],line_color = colors[i]
    )
plot_fp.add_plot(
    [],[],xticks_format = 0, x_major_tick =20, x_minor_tick = 5,
    y_minor_tick = 0.05,xlabel = 'Training set size (data points)',
    ylabel = "Value of DPO Parameters"
)
plot_fp.ax.legend(
    [tuple([plot_fp.scatters[i],plot_fp.scatters[i+1]]) for i in [0,2,4,6,8,10]],
    parameters_list,
    numpoints=1,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    prop = font_legend,
    loc="upper center",#"lower left",
    bbox_to_anchor=(-0.05,1.00,1,0.2),#(0.45,-0.16),
    #mode="expand", borderaxespad=0,
    ncol = 6
)
plot_fp.add_text(0,-0.25, "(A)")
#plot_fp.ax.text(-5,0.48,'(A)',**annotate)
plot_fp.save_fig('[result]/RepeatParams_full.jpeg',dpi=600)
# %%
params = trunRepeatTrain.parameters
plot_tp = scatter_plot()
parameters_list = ['a','b','c','d','a*','d*']
for i in range(params.shape[1]):
    plot_tp.add_plot(
        [0]+trainset_size_list, params[:,i],plot_line=True,
        label = parameters_list[i],
        scatter_color = colors[i],line_color = colors[i]
    )
    plot_tp.add_plot(
        135 if i != 3 else 137, parameter_true_value[i],
        scatter_marker = '*',label = parameters_list[i],
        scatter_color = colors[i],line_color = colors[i]
    )
plot_tp.add_plot(
    [],[],xticks_format = 0, x_major_tick =20, x_minor_tick = 5,
    y_minor_tick = 0.05,xlabel = 'Training set size (data points)',
    ylabel = "Value of DPO Parameters"
)

"""
plot_tp.ax.legend(
    [tuple([plot_tp.scatters[i],plot_tp.scatters[i+1]]) for i in [0,2,4,6,8,10]],
    parameters_list,
    numpoints=1,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    prop = font_legend,
    loc="upper center",#"lower left",
    bbox_to_anchor=(0.45,-0.16),#(0,1.02,1,0.2),
    #mode="expand", borderaxespad=0,
    ncol = 6
)
"""
plot_tp.add_text(0,-0.25, "(B)")

#plot_tp.ax.text(-5,0.48,'(B)',**annotate)
plot_tp.save_fig('[result]/RepeatParams_trun.jpeg',dpi=600)

#%%
for i, size in enumerate(trainset_size_list):
    print( 
          "{}, Full RMSE =  {:.2f}, trun RMSE = {:.2f}".format(
        size, 
        fullRepeatTrain.test_rmse[i],
        trunRepeatTrain.test_rmse[i])
        )
    
"""
7, Full RMSE =  0.26, trun RMSE = 0.29
13, Full RMSE =  0.19, trun RMSE = 0.19
26, Full RMSE =  0.12, trun RMSE = 0.12
40, Full RMSE =  0.11, trun RMSE = 0.11
53, Full RMSE =  0.11, trun RMSE = 0.11
66, Full RMSE =  0.10, trun RMSE = 0.11
79, Full RMSE =  0.10, trun RMSE = 0.11
106, Full RMSE =  0.10, trun RMSE = 0.10
132, Full RMSE =  0.10, trun RMSE = 0.10
"""
# %%
#%%
for j,prop in enumerate(pred_props):
    print(prop)
    for i, size in enumerate(trainset_size_list):
        print( 
            "{}, Full RMSE =  {:.2f}, trun RMSE = {:.2f}".format(
            size, 
            fullRepeatTrain.pred_test_rmse[i,j],
            trunRepeatTrain.pred_test_rmse[i,j])
            )

"""
EA
7, Full RMSE =  0.15, trun RMSE = 0.16
13, Full RMSE =  0.11, trun RMSE = 0.11
26, Full RMSE =  0.08, trun RMSE = 0.08
40, Full RMSE =  0.07, trun RMSE = 0.07
53, Full RMSE =  0.07, trun RMSE = 0.07
66, Full RMSE =  0.07, trun RMSE = 0.07
79, Full RMSE =  0.07, trun RMSE = 0.07
106, Full RMSE =  0.07, trun RMSE = 0.07
132, Full RMSE =  0.07, trun RMSE = 0.07
IP
7, Full RMSE =  0.12, trun RMSE = 0.14
13, Full RMSE =  0.09, trun RMSE = 0.09
26, Full RMSE =  0.07, trun RMSE = 0.06
40, Full RMSE =  0.06, trun RMSE = 0.06
53, Full RMSE =  0.06, trun RMSE = 0.06
66, Full RMSE =  0.06, trun RMSE = 0.06
79, Full RMSE =  0.06, trun RMSE = 0.06
106, Full RMSE =  0.06, trun RMSE = 0.06
132, Full RMSE =  0.06, trun RMSE = 0.06

"""
# %%
