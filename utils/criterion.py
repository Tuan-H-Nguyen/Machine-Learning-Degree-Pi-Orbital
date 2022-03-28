import numpy as np

def compute_r2(x,y):
    """Compute R^2
    Based on equation on wikipedia 
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Args:
        x (np.array): 
        y (np.array): 

    Returns:
        float: R-squared
    """
    sum_xy = np.sum(x*y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_y2 = np.sum(y**2)
    n = len(x)
    r = (n*sum_xy - sum_x*sum_y)/(((n*sum_x2-sum_x**2)*(n*sum_y2-sum_y**2))**0.5)
    return r**2

def RMSD(Y,Y_hat):
    """Compute Root mean square deviation

    Args:
        Y (np.array): 
        Y_hat (np.array): 

    Returns:
        float: 
    """
    dY = Y_hat - Y
    return np.sqrt((1/len(dY))*np.sum(dY**2))


class Loss:
    """Computing Mean Square Error loss
    """
    def __init__(self):
        pass
    
    def dL(self):
        """Compute the gradient of loss.

        Returns:
        float: gradient of loss which is 
            computed using eval method.        
        """
        return (2/len(self.dY))*self.dY
    
    def eval(self,Y_hat,Y):
        """Compute MSE loss

        Args:
            Y_hat (np.array): 
            Y (np.array): 

        Returns:
            float: MSE loss
        """
        dY = Y_hat - Y
        self.dY = dY
        return (1/len(dY))*np.sum(dY**2)
