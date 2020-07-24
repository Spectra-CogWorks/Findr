from mynn.layers.dense import dense
from mygrad.nnet.initializers import he_normal
import numpy as np

class Img2Caption:
    def __init__(self, dim_input=512, dim_encoded=50):
        """This initializes the single layer in our model, and sets it
        as an attribute of the model.
        
        
        Parameters
        ----------
        dim_input : int
            the size of the inputs
            
        dum_encoded : int
            the size of the outputs of the encoded layer
        
        
        """
        self.dense1 = dense(dim_input, dim_encoded, weight_initializer=he_normal)
    def __call__(self, x):
        """ The model's forward pass functionality.
        
        Parameters
        ----------
        x : numpy.ndarray, shape = (512,)
            
        Returns
        -------
        encoded : numpy.ndarray, shape = (50,)
        
        """
        return self.dense1(x)
    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.
        
        This can be accessed as an attribute, via `model.parameters` 
        
        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model """
        return tuple(self.dense1.parameters)
    
    def export_weights(self, filepath="weights.npy"):
        """Export the model weights as a npy file
        
        Parameters
        ----------
        filepath : str
            The filepath to save the npy file to
        
        Returns
        -------
        None
        """
        # TODO Double check to make sure that the numpy array will be storing the weight and bias properly
        np.save(filepath, np.array(self.parameters))
        
    def import_weights(self, filepath="weights.npy"):
        """Import the model weights from a npy file
        
        Parameters
        ----------
        filepath : str
            The filepath to import the npy file from
        
        Returns
        -------
        None
        """
        
        weights = tuple(np.load(filepath))
        
        # TODO Double check that the weight and bias are assigned properly
        self.dense1.weight = weights[0]
        self.dense1.bias = weights[1]