import mygrad as mg
import mynn
from mynn.losses.margin_ranking_loss import margin_ranking_loss

def mr_loss(good_img_embed, bad_img_embed, triple):  
    """
    Returns the margin ranking loss, given two image embedding vectors and a "good" caption.

    Parameters
    ----------
    good_img_embed : np.ndarray
        A shape-(50,) numpy array of the word embeddings for the "good" image.

    bad_img_embed : np.ndarray
        A shape-(50,) numpy array of the word embeddings for the "bad" image.
    
    triple : tuple
        A tuple containing three elements: the descriptor of the "good" image,
        the caption corresponding to that image, and the descriptor of the "bad" image.

    Returns
    -------
    margin_ranking_loss : mg.Tensor
        The margin ranking loss between the similarities (dot products) between the "good"
        image and the caption/"bad" image.
    """
    triple[1] = W_good
    S_good = mg.dot(good_img_embed, W_good)
    S_bad = mg.dot(good_img_embed, bad_img_embed)

    return margin_ranking_loss(S_good, S_bad, 1, 0.1)