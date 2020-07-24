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
    
    def save_model(self, path="weights.npz"):
        """Path to .npz file where model parameters will be saved."""
        with open(path, "wb") as f:
            np.savez(f, *(x.data for x in self.parameters))

    def load_model(self, path="weights.npz"):
        with open(path, "rb") as f:
            for param, (name, array) in zip(self.parameters, np.load(f).items()): # pylint: disable=unused-variable
                param.data[:] = array