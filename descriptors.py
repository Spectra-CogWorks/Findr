import numpy as np
import pickle
#load in pickle file with resnet18
with open("resnet18_features.pkl", mode="rb") as opened_file:
    resnet = pickle.load(opened_file)

# TODO Please check that this function works
def generate_descriptors(imgIDs):
    """ Generates decriptor from resnet dictionary if imgIDs are keys. Else, returns none in their place
        
    Parameters
    ----------
    imgID: List[int]
        The image ids
            
    Returns
    -------
    descriptor: List[np.ndarray shape-(1,512)]
        The descriptors associated with the image id
    or
    None
    """
    descriptors = []
    
    for imgID in imgIDs:
        if imgID in resnet.keys():
            descriptors.append(resnet[imgID])
        else:
            descriptors.append(None)
            
    return descriptors
    
def generate_descriptor(imgID):
    """ Generates decriptor from resnet dictionary if imgIDs are keys. Else, returns none in their place
        
    Parameters
    ----------
    imgID: int
        The image ids
            
    Returns
    -------
    descriptor: Lnp.ndarray shape-(1,512)
        The descriptors associated with the image id
    or
    None
    """
    if imgID in resnet.keys():
        return resnet[imgID]
    else:
        return None