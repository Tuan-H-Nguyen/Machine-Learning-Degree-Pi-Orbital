import numpy as np
import sympy as sym

from sklearn.linear_model import LinearRegression

class DpoEvaluator:
    """
    Class for rapidly evaluate the DPO expressions.
    
    Powered by Sympy's lambdify, which significantly boosts the polynomials
    evaluation process considerably, as compare to python's native 
    eval or exec. 
    
    Lamdify takes in string of expression and return a lambda-ish function. 
    This function takes in values of symbols in the expression as args 
    and returns the numerical evaluation of the expression. 
    
    What's more, This class record the lamdified expression function in a 
    dict, which can be "recalled" if the expression is asked to be evaluated 
    again.
    
    Args:
    + parameters_list (list of str): list of parameters' symbols.
    + parameter_values (list of float): list of values for parameters. 
        Should have the same length as parameters_list.
    + constant_list (list of str): list of constants' symbols.
    + constant_values (list of float): list of values for constants. 
        Should have the same length as constant_list.
    """
    def __init__(
        self,
        parameters_list,
        parameter_values,
        constant_list,
        constant_values
        ):
        self.parameters_list = [
            sym.symbols(p) for p in parameters_list]
        self.parameter_values = parameter_values
        self.constant_list = [
            sym.symbols(p) for p in constant_list]
        self.constant_values = constant_values
        self.exp_dict = {}

    def update_pvalues(self,parameter_values):
        """
        Update values of parameters.

        Args:
        + parameter_values (list of float): new values for 
            parameters.
        """
        self.parameter_values = parameter_values

    def eval(self,exp):
        """
        Evaluate the expression provided with the current 
        values of parameters or constants. 

        Args:
        + exp (string): a polynomial.
            e.g. "3 - 3*a + b + d + d*(1-a) + d*b + -af"
            Note that the given polynomial should contain 
            only symbols that are in the parameters_list

        Returns:
            (float): evaluation of the polynomial
        """
        try: exp_f = self.exp_dict[exp]
        except KeyError:
            exp_f = sym.lambdify(
                self.parameters_list + self.constant_list,exp)
            self.exp_dict[exp] = exp_f
        return exp_f(*np.hstack([self.parameter_values,self.constant_values]))

class DpoDerivative:
    def __init__(
        self,
        parameters_list,
        parameter_values,
        constant_list,
        constant_values):
        """
        Class for evaluate the derivative of the DPO expression.
        Same as the above but with sympy numerical diffentiated 
        expressions.

        Args:
        + parameters_list (list of str): list of parameters' symbols.
        + parameter_values (list of float): list of values for parameters. 
            Should have the same length as parameters_list.
        + constant_list (list of str): list of constants' symbols.
        + constant_values (list of float): list of values for constants. 
            Should have the same length as constant_list.
        """
        self.parameters_list = [
            sym.symbols(p) for p in parameters_list]
        self.parameter_values = parameter_values
        self.constant_list = [
            sym.symbols(p) for p in constant_list]
        self.constant_values = constant_values
        self.d_dict = {}
    
    def differentiate(self,exp):
        """Analytically evaluate the expression

        Args:
        + exp (string): expression

        Returns:
        + list: list of expressions that are resulted from
            differentiating the given expression with multiple 
            variables
        """
        return [sym.diff(exp,p) for p in self.parameters_list]

    def grad(self,exp):
        """Numerically evaluate the derivative 
        of the given expression.

        Args:
        + exp (string): expression

        Returns:
        + np.array: numerical derivative of the 
            given expression respect to various 
            parameters
        """
        try:
            d_exp = self.d_dict[exp]
        except KeyError:
            d_exp = self.differentiate(exp)
            d_exp = list(map(
                lambda f: sym.lambdify(
                    self.parameters_list + self.constant_list,f
                    ),d_exp
                ))
            self.d_dict[exp] = d_exp
        d_exp = np.array(
            list(map(
                lambda f:f(*np.hstack([self.parameter_values,self.constant_values])),d_exp
            )))
        return d_exp

    def update_parameter(self,parameter_values):
        self.parameter_values = parameter_values

class LinearOutput:
    def __init__(self):
        self.weight = None
        
    def least_square(self,X,Y):
        lm = LinearRegression()
        lm.fit(X.reshape(-1,1),Y.reshape(-1,1))
        wb, w = float(lm.intercept_), float(lm.coef_)
        self.weight = wb,w
        
    def forward(self,X):
        wb,w = self.weight
        return wb + w*X
    
    def dL(self):
        return self.weight[1]
    
    def set_weight(self,weight):
        if isinstance(weight,list):
            self.weight = weight[0],weight[1]
        elif isinstance(weight,tuple):
            self.weight = weight
        else: 
            self.set_weight(list(weight))
        
        
