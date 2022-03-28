# %%
import sys
import os
module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)
import numpy as np
import pandas as pd
import time
from copy import deepcopy
from poly_rings.DPO import DPO_generate

from gd_dpo.GDDPO import GD_DPO
from utils.stratified_splitting import sampling

#%%
class Trainer:
    def __init__(
        self, 
        data, SMILES,
        global_seed,
        descriptor_col,
        target_col = 'BG',
        pred_col = ["EA","IP"],
        data_intervals = [1.5 , 2 , 2.5 , 3 , 3.5 , 4 , 4.5 , 5],
        parameters_list = ['a','b','c','d','af','df'],
        parameter_values = None,
        constant_list = ['bf','cf','b0'],
        constant_values = [0,0,0]
        ):
        """
        class for training and testing of GD-DPO model
        Args:
            + data (pd.DataFrame): all data includes training set and test set
            + SMILES (bool): convert column of SMILES to truncated DPO if True. 
                The descriptor_col has to be filled with SMILES
            + descriptor_col (string): name of column of data that contains the 
                molecular descriptors
            + target_col (string): name of column of data that contains the 
                "true value" of the modeled property. DPO parameters are fitted 
                against these values. E.g. HOMO-LUMO gap
            + pred_col (list of string): name of columns of data that contains the 
                "true value" of the modeled property. For these properties, models 
                are constructed using pre-optimized DPO parameters (above) to fit 
                the linear weight.
            + global_seed (int): seed for reproduce the result     
            + data_intervals (list of float): for binning data. For instances: let 
                [...x_i,x_(i+1),...], then data is sampled from each bin [x_i,x_(i+1)]
                THe samples is proportionate to the number of data in that bin.
            + parameters_list (list of char): list of symbols of parameters that
                are learned, default: DPO parameters
            + parameters_values (list of float, optional): initial value, default
                all zeros
            + constant_list (list of char): list of symbols which values are not 
                learned/ kept constant, default: parameters that are deemed zeros
                in previous works
            + constant_values (list of char): values of constant
        """
        self.global_seed = global_seed
        self.data = data
        
        if SMILES:
            self.data.at[:,"DPO"] = self.data.loc[
                :,descriptor_col].apply(DPO_generate)
            self.dpo_col = "DPO"
        else:
            self.dpo_col = descriptor_col
        
        self.target_col = target_col
        self.pred_col = pred_col
        
        self.intervals = data_intervals
        self.p_list = parameters_list
        self.p_values = parameter_values
        self.c_list = constant_list
        self.c_values = constant_values

    def standard_train(
            self,
            train_set_size = 132,
            test_set_size = 116,
            lr=0.1,
            epochs=500,
            verbose=10,
            lr_decay_rate=5, 
            min_lr = 10**-9,
            threshold= 10**-9, 
            patience = 50,
            seed = None,
            skip_print = False,
            timing = True,
            data_splits = None
        ):
        """
        One time training on full-size training set (default 132 datapoints) 
        and tested on the test set which consists of the remaining data points 
        (116 by default). 
        Args:
        + train_set_size (int): the number of samples in the training set.
        + test_set_size (int): the number of samples in the test set. The 
            test set is taken from the samples remaining in the total data 
            after training set sampling.
        + lr (float): learning rate
        + epochs (int): number of epochs, which runing out terminates the model
        + lr_decay_rate (float, optional): an amount by which learning rate is divided
            had the MSE loss shoot up instead of going down. Defaults to 10.
        + min_lr (float, optional): if lr goes below this value, the model is terminated. 
            Defaults to 10**-9.
        + threshold (float, optional): if the improvement (difference btw the loss 
            of updated model and pre-updated model) is below this value for a [patience] 
            (see below) consecutive time step, the model is terminated. 
            Default to be 10**-9.
        + patience (int, optional): see above. Defaults to 50.
        + verbose (int, optional): print loss on console if epoch modulo verbose == 0. 
            If it is 0, nothing is printed. Default 1.
        + seed (int): if a seed other than the object global seed is to be used
        + skip_print (bool): if True, skip printing the result message.
        + timing (bool) : time the process.
        + data_splits (list or tuple of 2 pd.DataFrame): provided training set and test set if not 
            sample from the total dataset
        
        Important Attribute:
        + train_loss: Mean Square Error of the model on train set
        + train_rmse: Root Mean Square Error on train set
        + test_loss: Mean Square Error of the model on test set
        + test_rmse: Root Mean Square Error on test set
        
        Return:
        + model (instance of GD_DPO): model that are fitted against target_col and pred_col (only
            linear equations)
        """
        if timing:
            start = time.time()
        seed = seed if seed else self.global_seed
        
        if not data_splits:
            train_set = sampling(
                train_set_size, self.data,
                intervals = self.intervals,
                random_state = seed
            ) #standard train set with size of 132 as reservoir
            
            test_set = self.data.drop(train_set.index)
            test_set = sampling(
                test_set_size, test_set,
                intervals = self.intervals,
                random_state = seed
            )
        else:
            train_set,test_set = data_splits    
            
        self.train_set = train_set
        self.test_set = test_set

        X = np.array(train_set.loc[:][self.dpo_col])
        Y = np.array(train_set.loc[:][self.target_col])
        X_test = np.array(test_set.loc[:][self.dpo_col])
        Y_test = np.array(test_set.loc[:][self.target_col])

        model = GD_DPO(
            parameters_list = self.p_list,
            parameter_values = self.p_values,
            constant_list = self.c_list,
            constant_values = self.c_values,
            tasks = len(self.target_col) + len(self.pred_col)
            )
                
        #fit the DPO paramaters and linear weights of target_col
        model.fit(
            X,Y,lr = lr, epochs = epochs, verbose = verbose,
            lr_decay_rate=lr_decay_rate, min_lr = min_lr,
            threshold= threshold, patience = patience
            )
        
        #the train and test loss/rmse can be retrieved via class attribute
        self.train_loss = model.compute_loss(X,Y)
        self.train_rmse = np.sqrt(self.train_loss)
        
        self.test_loss = model.compute_loss(X_test,Y_test)
        self.test_rmse = np.sqrt(self.test_loss)
        
        self.pred_test_rmse = []
        
        #create model for predicting target in pred_col
        for i,pred_prop in enumerate(self.pred_col):
            Y = np.array(train_set.loc[:][pred_prop])
            Y_test = np.array(test_set.loc[:][pred_prop])
                        
            #to fit the weight of the linear equation, 
            # simply call feedforward, instead of fit
            model.feedforward(X,Y,task = i+1)
            
            self.pred_test_rmse.append(np.sqrt(
                model.compute_loss(X_test,Y_test,task = i+1)
            ))
            
        if not skip_print:
            print(
                '\n'+
                f"###################\n"+
                f"##TRAINING RESULT##\n"+
                f"###################\n"+
                f"Converge at {round(model.finalEpoch)}.\n"+
                f"Train loss: {round(self.train_loss,4)}, train RMSE: {round(self.train_rmse,4)}.\n"+
                f"Test loss: {round(self.test_loss,4)}, test RMSE: {round(self.test_rmse,4)}.\n"+
                "Parameter:\n"
                +'\n'.join([str(a)+' : '+str(round(b,4)) for a,b in zip(self.p_list,model.parameter_values)])
            )
        if timing:
            print(f"Converged, run time: {time.time()-start} s.")
            
        return model

    def repeat_standard_train(
            self,n,
            train_set_size = 132,
            test_set_size = 116,
            lr=0.1,
            epochs=500,
            lr_decay_rate=5, 
            min_lr = 10**-9,
            threshold= 10**-9, 
            patience = 50,
            verbose = True
            ):
        """
        Repeat standard train (which is defined above)
        Args:
        + n (int): number of runs
        + train_set_size (int): the number of samples in the training set.
        + test_set_size (int): the number of samples in the test set. The 
            test set is taken from the samples remaining in the total data 
            after training set sampling.
        + lr (float): learning rate
        + epochs (int): number of epochs, which runing out terminates the model
        + lr_decay_rate (float, optional): an amount by which learning rate is divided
            had the MSE loss shoot up instead of going down. Defaults to 10.
        + min_lr (float, optional): if lr goes below this value, the model is terminated. 
            Defaults to 10**-9.
        + threshold (float, optional): if the improvement (difference btw the loss of updated 
            model and pre-updated model) is below this value for a [patience] (see below) 
            consecutive time step, the model is terminated. Defaults to 10**-9.
        + patience (int, optional): see above. Defaults to 50.
        + seed (int): if a seed other than the object global seed is to be used
        + skip_print (bool): if True, skip printing the result message.
        
        Important Attribute:
        + self.param_values_list (list of list): list of all sets of parameters for 
            all runs.
        + self.parameters (list of float): list of the sets of parameters average over 
            all runs.
        + self.weight_values_list (list of list): list of all sets of linear weights 
            for all runs.
        + self.weights (list of float): linear weights average over all runs.
        + self.train_rmse (float): Root Mean Square Deviation on training set averaged 
            over all runs.
        + self.full_test_rmse (list of float): list of all Root Mean Square Deviation
            on test set of all runs.
        + self.test_rmse (float): Root Mean Square Deviation on test set averaged over 
            all runs.
            
        Return: 
        + main_model (instance of GD_DPO):  model that are average of all models fitted 
            against target_col and linear weight are fitted against target_col and 
            pred_col in multiple runs
        """
        
        start = time.time()
        np.random.seed(self.global_seed)
        seeds_list = np.random.randint(np.iinfo(np.int32).max,size = n)

        trainRMSEs = []
        testRMSEs = []
        pred_test_rmse = []
        
        pvalues = np.zeros(len(self.p_list))
        wvalues = np.zeros(
            (len(self.pred_col)+1,2)
            )

        for i in range(n):
            if verbose:
                print(f"Run #{i+1}, status:")
            model = self.standard_train(
                train_set_size = train_set_size, 
                test_set_size= test_set_size,
                lr = lr,epochs = epochs,seed = seeds_list[i],
                lr_decay_rate = lr_decay_rate, min_lr = min_lr,
                threshold = threshold, patience = patience,
                verbose=0,skip_print=True,
                timing = verbose
                )
            
            trainRMSEs.append(self.train_rmse)
            
            testRMSEs.append(self.test_rmse)
            
            pred_test_rmse.append(self.pred_test_rmse)
            
            pvalues += np.array(model.parameter_values)
            wvalues += np.array(
                [list(model.outputs[i].weight) for i in 
                 range(1+len(self.pred_col))]
                )
                                
        self.full_train_rmse = trainRMSEs
        
        self.full_test_rmse = testRMSEs #for debug
        self.test_rmse = np.mean(testRMSEs)
        self.pred_test_rmse = np.mean(pred_test_rmse,axis=0)
        
        model = GD_DPO(
            parameters_list = self.p_list,
            parameter_values = pvalues/n,
            constant_list = self.c_list,
            constant_values = self.c_values,
            tasks = 1 + len(self.pred_col)
        )
        
        wvalues /= n
        for i,weight in enumerate(wvalues):
            model.outputs[i].set_weight(weight)
        
        print(
                '\n'+
                f"##########################\n"+
                f"##REPEAT TRAINING RESULT##\n"+
                f"##########################\n"+
                f"Train on training set size of {train_set_size}\n"+
                f"Repeating training for {n} times, average result reported.\n"
                f"Train RMSE: {round(np.mean(trainRMSEs),4)}.\n"+
                f"Test RMSE: {round(np.mean(testRMSEs),4)}.\n"+
                f"\nFinish, total run time: {time.time()-start} s"
            )
        
        return model
    
    def train(
        self,train_set_size_list,
        lr=0.1,epochs=500,verbose=0,
        lr_decay_rate=10, min_lr = 10**-9,
        threshold= 10**-9, patience = 50,
        seed = None,skip_print=False
        ):
        """
        Train with a series of training sets with various size, but test on 116
        -samples test set 
        Args:
        + train_set_size_list (list of float): list of training set sizes.
            The number of total datapoints minus last of them is the number
            of samples in fixed test set. (e.g. [...,132] -> 116 test samples)
        + lr (float): learning rate
        + epochs (int): number of epochs, which runing out terminates the model
        + lr_decay_rate (float, optional): an amount by which learning rate is divided
            had the MSE loss shoot up instead of going down. Defaults to 10.
        + min_lr (float, optional): if lr goes below this value, the model is terminated. 
            Defaults to 10**-9.
        + threshold (float, optional): if the improvement (difference btw the loss of updated 
            model and pre-updated model) is below this value for a [patience] (see below) 
            consecutive time step, the model is terminated. Defaults to 10**-9.
        + patience (int, optional): see above. Defaults to 50.
        + seed (int): if a seed other than the object global seed is to be used
        + skip_print (bool): if True, skip printing the result message.
        Important Attributes:
        + self.parameters
        + self.weights
        + self.train_rmse
        + self.test_rmse
        are list of sets of parameters, sets of linear weights, RMSE on training set,
        RMSE on test set for models trained on sets with the provided list of size
        """

        start = time.time()

        seed = seed if seed else self.global_seed
        seed_list = np.random.randint(np.iinfo(np.int32).max,size = len(train_set_size_list))
        
        model = GD_DPO(
            parameters_list = self.p_list,
            parameter_values = self.p_values,
            constant_list = self.c_list,
            constant_values = self.c_values,
            tasks = len(self.pred_col) + 1
            )

        self.train_rmse = []
        self.test_rmse = []
        self.pred_test_rmse = []
        
        self.parameters = []
        self.parameters.append(deepcopy(model.parameter_values))

        train_set_complement = sampling(
            train_set_size_list[-1], self.data,
            intervals = self.intervals,
            random_state = seed
        ) #standard train set with size of 132 as reservoir
        
        test_set = self.data.drop(train_set_complement.index)
        #remaining 116 go to the test set
        
        train_set = pd.DataFrame(columns = self.data.columns)
        #empty train set
                
        for i,train_set_size in enumerate(train_set_size_list):
            sample_size = train_set_size-train_set_size_list[i-1] if i>0 else train_set_size
            # number of samples need to take from reservoir and add to 
            # current training set
            # at the beginning, the training set is sampled freshly, equal 
            # to the given (first) training set size
            
            #taken the amount of samples from th training reservoir
            train_set_add = sampling(
                sample_size,train_set_complement,intervals = self.intervals,
                random_state = seed_list[i])
            
            #add the sampled data to the existing training set
            train_set = pd.concat([train_set,train_set_add])
            
            #for examining what are those training sets
            #drop the sampled data from the training reservoir
            train_set_complement = train_set_complement.drop(
                train_set_add.index)

            #the regular from here
            X = np.array(train_set.loc[:][self.dpo_col])
            Y = np.array(train_set.loc[:][self.target_col])
            
            X_test = np.array(test_set.loc[:][self.dpo_col])
            Y_test = np.array(test_set.loc[:][self.target_col])

            model.fit(
                X,Y,lr = lr, epochs = epochs, verbose = verbose,
                lr_decay_rate=lr_decay_rate, min_lr = min_lr,
                threshold= threshold, patience = patience
                )
            
            self.parameters.append(deepcopy(model.parameter_values))
            
            train_loss = model.compute_loss(X,Y)
            self.train_rmse.append(np.sqrt(train_loss))
            
            test_loss = model.compute_loss(X_test,Y_test)
            self.test_rmse.append(np.sqrt(test_loss))
            
            pred_test_rmse = []
            for i,pred_prop in enumerate(self.pred_col):
                Y = np.array(train_set.loc[:][pred_prop])
                Y_test = np.array(test_set.loc[:][pred_prop])
                                
                model.feedforward(X,Y,task = i+1)
                
                pred_test_rmse.append(
                    model.compute_loss(X_test,Y_test,task = i+1))
                
            self.pred_test_rmse.append(np.sqrt(pred_test_rmse))

        if not skip_print:
            print(
                '\n'+
                f"###################\n"+
                f"##TRAINING RESULT##\n"+
                f"###################\n"+
                f"Test RMSE versus train set size:.\n"+
                '\n'.join([str(round(a,4))+' : '+str(b) for a,b in zip(self.test_rmse,train_set_size_list)]))
        print("***Run time:",time.time()-start,'***\n')

    def repeat_train(self,n,train_set_size_list):
        """
        Repeatedly train with a series of training sets with various size 
        Args:
        + n (int): number of runs
        + train_set_size_list (list of float): list of training set sizes.
            Each of them should be less than 116
        Important Attributes:
        + self.test_rmse
        + self.train_rmse
        + self.pred_test_rmse
        + self.parameters
        are list of sets of parameters, RMSE on training set, RMSE on test set 
        of models for target_col (test_rmse) and pred_col (pred_test_rmse)
        trained on training sets with the provided list of size AVERAGED over 
        all runs
        """
        start = time.time()
        np.random.seed(self.global_seed)
        seed_list = np.random.randint(1000,size = n)
        
        meanTrainRMSE = np.zeros((len(train_set_size_list)))
        meanTestRMSE = np.zeros((len(train_set_size_list)))
        meanPredTestRMSE = np.zeros(
            (len(train_set_size_list),
             len(self.pred_col))
            )
        
        mean_params = np.zeros((len(train_set_size_list)+1,len(self.p_list)))
                
        for i in range(n):
            self.train(
                train_set_size_list,
                seed = seed_list[i],
                skip_print=False)
            meanTrainRMSE += np.array(self.train_rmse)
            meanTestRMSE += np.array(self.test_rmse)
            meanPredTestRMSE += np.array(self.pred_test_rmse)
            mean_params += np.array(self.parameters)
            
        self.test_rmse = meanTestRMSE/n
        self.train_rmse = meanTrainRMSE/n
        self.pred_test_rmse = meanPredTestRMSE/n
        
        self.parameters = mean_params/n
        
        print(
                '\n'+
                f"##########################\n"+
                f"##REPEAT TRAINING RESULT##\n"+
                f"##########################\n"+
                f"Train RMSE versus train set size:.\n"+
                '\n'.join([str(round(a,4))+' : '+str(b) for a,b in zip(self.train_rmse,train_set_size_list)]) +
                f"\nTest RMSE versus train set size:.\n"+
                '\n'.join([str(round(a,4))+' : '+str(b) for a,b in zip(self.test_rmse,train_set_size_list)]))
        print("***Total run time: ",time.time()-start,'***')

# %%
