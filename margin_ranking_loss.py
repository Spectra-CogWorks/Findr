import mygrad as mg
import mynn
from mynn.losses.margin_ranking_loss import margin_ranking_loss

from img2caption_class import Img2Caption

def mr_loss(model, triple):  
    """
    Returns the margin ranking loss, given two image embedding vectors and a "good" caption.

    Parameters
    ----------
    model : Img2Caption
        The model used to convert the image descriptors to 
    
    triple : tuple
        A tuple containing three elements: the descriptor of the "good" image,
        the caption embedding corresponding to that image, and the descriptor of the "bad" image.

    Returns
    -------
    margin_ranking_loss : mg.Tensor
        The margin ranking loss between the similarities (dot products) between the "good"
        image and the caption/"bad" image.
    """
    # S_good = mg.dot(model(triple[0]), triple[1]))
    # S_bad = mg.dot(model(triple[0]), model(triple[2]))
    # margin_ranking_loss(S_good, S_bad, y, margin)
    return margin_ranking_loss(mg.dot(model(triple[0]), triple[1]), 
                               mg.dot(model(triple[0]), model(triple[2])), 
                               1, 
                               0.1)