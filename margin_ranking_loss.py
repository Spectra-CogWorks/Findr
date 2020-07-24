import mygrad as mg
from mygrad.nnet.losses.margin_ranking_loss import margin_ranking_loss

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
    # S_good = mg.dot(triple[1], model(triple[0])))
    # S_bad = mg.dot(triple[1], model(triple[2])))
    # margin_ranking_loss(S_good, S_bad, y, margin)
    
    good_images = []
    good_captions = []
    bad_images = []

    for good_img, good_cap, bad_img in triple:
        good_images.append(good_img)
        good_captions.append(good_cap)
        bad_images.append(bad_img)
    
    good_images = np.array(good_images)
    good_captions = np.array(good_captions)
    bad_images = np.array(bad_images)

    return margin_ranking_loss(mg.sum(good_captions * model(good_images), axis=1), 
                               mg.sum(good_captions * model(bad_images), axis=1), 
                               1, 
                               0.1)