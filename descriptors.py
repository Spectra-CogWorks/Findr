import numpy as np
import pickle
#load in pickle file with resnet18
with open("resnet18_features.pkl", mode="rb") as opened_file:
    resnet = pickle.load(opened_file)

# TODO Please ensure that the function can accept a List[int]
def generate_descriptor(imgID):
    """ Generates decriptor from resnet dictionary if imgID is a key. Else, returns none 
        
    Parameters
    ----------
    imgID: int
        the image id
            
    Returns
    -------
    descriptor: np.ndarray shape-(1,512)
        the descriptor associated with the image id
    or
    
    None
        
    """
    if imgID in resnet.keys():
        return resnet[imgID]
    else:
        return None