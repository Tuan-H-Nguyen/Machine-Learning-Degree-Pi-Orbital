#%% import stuffs
import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym #PVP
from sklearn.linear_model import LinearRegression
from copy import deepcopy

module_root = os.path.dirname(os.path.realpath('__file__'))
module_root = '\\'.join(module_root.split('\\')[0:-1])
sys.path.append(module_root)

from core.core import DpoEvaluator, DpoDerivative, LinearOutput
from utils.criterion import Loss

class GD_DPO:
    def __init__(
        self,
        parameters_list,
        parameter_values = None,
        constant_list = [],
        constant_values = None,
        tasks = 1):
        """
        Class for the ML-DPO model. Note that whether model is the full DPO or 
        truncated DPO depend on the descriptors provided (in the dataset) during 
        training/testing
        Args:
        + parameters_list (list of string): list of symbols which values 
            are learned
        + parameters_values (None or list of float): initial values for 
            Must have the same size as the previous parameters (default: zeros)
        + constant_list (list of string): list of symbols which values 
            are kept constant (default: empty list)
        + constant_values (list of float): values for constant. Must have 
            the same size as the previous (default: list of zeros)
        + tasks (int): number of properties for predicting. The predictions of
            designated properties share the same DPO determination, but difference 
            linear equation (e.g. y = w*DPO + wb), thus the model has one set of
            DPO parameters but as many sets of linear weights as the number of 
            "tasks". Can be thought of as a multi-task model where "tasks" share
            the DPO "layer", but have different linear "output layer". Default to 1.
        """
        if parameter_values is None:
            parameter_values = np.zeros(len(parameters_list))
        if constant_values is None:
            constant_values = np.zeros(len(constant_list))
            
        self.parameters_list = parameters_list
        self.parameter_values = parameter_values
        self.constant_list = constant_list
        self.constant_values = np.array(constant_values)
        
        self.E = DpoEvaluator(
            self.parameters_list,self.parameter_values,
            self.constant_list, self.constant_values
            )
        
        self.D = DpoDerivative(
            self.parameters_list,self.parameter_values,
            self.constant_list,self.constant_values
            )
        
        self.L = Loss()
        self.outputs = [
            LinearOutput() for _ in range(tasks)] 

        
    def feedforward(
        self,
        DPO,Y,task = 0
        ):
        """
        Compute both the Mean Square Error loss and the value of gradient of 
        the loss repect to the vector of DPO parameters. This function 
        encompasses step 3 through 7 in the algorithm presented in the paper.
        Moreover, since it compute the weight of the linear equation, it can 
        utilized to determined the linear weight alone.
        Args:
        + DPO (list of strings): array of DPO expressions
        + Y (list or np.array): array of target
        + task (int): the index of the linear equation (or "output layer") 
            for employing. Default to 0.
        Return:
        + loss (float): Mean Square Error of the current model 
            (e.g. set of parameters) on provided DPO and Y
        + dL_p (np.array): gradient of loss respect to 
            the vector of DPO parameters
        """
        #step 3: evaluate the DPO array with values of parameters
        X = np.array(list(map(self.E.eval,DPO)))
        
        #step 4: determine the linear equation
        self.outputs[task].least_square(X,Y)
        
        #step 5: compute the prediction
        Y_hat = self.outputs[task].forward(X)        
        
        #step 6: determine the MSE loss of the model
        loss = self.L.eval(Y_hat,Y)
        
        #step 7: evaluate the gradient of loss
        dX = np.array(list(map(self.D.grad,DPO))).T
        dL_p = self.outputs[task].dL()*np.dot(dX,self.L.dL())
        
        return loss,dL_p
    
    def compute_loss(self,DPO,Y,task = 0):
        """
        Computing the loss of the model using a given array of
        polynomials DPO and an array of true value Y

        Args:
        + DPO (list): DPO polynomials
        + Y (np.array): Corresponding "true" values
        + task (int): the index of the linear equation (or "output layer") 
            for employing. Default to 0.

        Returns:
        + loss (float): MSE loss
        """
        _,Y_hat = self.predict(DPO,task)
        loss = self.L.eval(Y_hat,Y)
        return loss

    def predict(self,DPO,task = 0):
        """
        Make prediction. No least square fitting.
        Args:
        + DPO (str or list of string): array of DPO expressions
        + task (int): the index of the linear equation (or "output layer") 
            for employing. Default to 0.
        Return:
        + X (array of float): DPO values correspond to DPO
        + Y_hat (array of float): prediction correspond to 
            DPO
        """
        if isinstance(DPO,str):
            DPO = [DPO]
        X = np.array(list(map(self.E.eval,DPO)))
        Y_hat = self.outputs[task].forward(X)
        return X,Y_hat

    def step(self,lr,dL_p):
        """
        Update parameters.
        Step 8 of the algorithm.
        Args:
            + lr (float): learning rate
            + dL_p (array): gradient of loss respect to parameters
        """
        self.parameter_values -= lr*dL_p
        self.E.update_pvalues(self.parameter_values)
        self.D.update_parameter(self.parameter_values)

    def rewind(self,backUpParam,backUpOutputs):
        """
        can be used to undo the update by replacing the set of values 
        of paramaters with back up set of parameters values
        Args: 
            + backUpParam (array of float): backup set of parameters
            + backUpOutputs (array of float): backup set of "output layers"
        """
        self.parameter_values = backUpParam
        self.outputs = backUpOutputs
        self.E.update_pvalues(self.parameter_values)
        self.D.update_parameter(self.parameter_values)
    
    def fit(
        self,
        DPO,Y,
        lr,
        epochs,
        task = 0,
        lr_decay_rate=10, min_lr = 10**-9,
        threshold= 10**-9, patience = 50,
        verbose = 1
        ):
        """
        fit the model to a training set

        Args:
        + DPO (array of strings): array of DPO polynomials
        + Y (np.array of float): "true" values
        + lr (float): learning rate
        + epochs (int): number of epochs, which runing out terminates the model
        + task (int): the index of the linear equation (or "output layer") 
            for employing. Default to 0.
        + lr_decay_rate (float, optional): an amount by which learning rate is divided
            had the MSE loss shoot up instead of going down. Defaults to 10.
        + min_lr (float, optional): if lr goes below this value, the model is terminated. 
            Defaults to 10**-9.
        + threshold (float, optional): if the improvement (difference btw the loss of updated 
            model and pre-updated model) is below this value for a [patience] (see below) consecutive 
            time step, the model is terminated. Defaults to 10**-9.
        + patience (int, optional): see above. Defaults to 50.
        + verbose (int, optional): print loss on console if epoch % verbose == 0. 
            If it is 0, nothing is printed. Default 1.
        """
        count = 0
                
        for epoch in range(1,epochs+1):
            #back up parameters
            backUpParam = deepcopy(self.parameter_values)
            backUpOutputs = deepcopy(self.outputs)
            
            #compute loss and gradient of L respect to DPO params
            #step 3 through 7
            loss,grad = self.feedforward(DPO,Y,task = task)
            
            #update the parameters. Step 8.
            self.step(lr,grad)
            
            # compute loss post-update. 
            # Step 9 and loop back or terminates.
            updatedLoss = self.compute_loss(DPO,Y,task = task)
            
            if updatedLoss > loss:
                #if the loss increase
                print("Loss increase. Rewind parameters and decay learning rate.")
                self.rewind(backUpParam,backUpOutputs) #undo the update
                lr /= lr_decay_rate #decay the learning rate
                # check if the lr is large enough. If not, terminate
                if lr <= min_lr:
                    print("Break due to small learning rate.")
                    break
            elif loss - updatedLoss < threshold:
                # if the update brings insignificant improvement
                count += 1
                if count > patience:
                    print("Break due to insignificant improvement.")
                    break
            else:
                #update successful. Reset the patience count
                count = 0
            
            if verbose != 0 and epoch%verbose == 0:
                print("Epoch {}: loss: {}".format(epoch,updatedLoss))
                print(self.parameter_values)
        
        self.finalEpoch = epoch
        self.finalTrainLoss = updatedLoss
# %%
