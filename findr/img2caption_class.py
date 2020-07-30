from mynn.layers.dense import dense
from mygrad.nnet.initializers import he_normal
import mygrad as mg
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
        x : numpy.ndarray, shape = (M,512)
            M is the number of rows
            
        Returns
        -------
        encoded : numpy.ndarray, shape = (M,50)
        
        """

        unnorm_ans = self.dense1(x)

        # We have to turn the output into a unit vector by dividing by the sum of the squares of the unnormalized result
        return unnorm_ans / (mg.sqrt(mg.sum(unnorm_ans ** 2, axis=1, keepdims=True)))

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
            for param, (name, array) in zip( #pylint: disable=unused-variable
                self.parameters, np.load(f).items()
            ): 
                param.data[:] = array
