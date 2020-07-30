import mygrad as mg
from mygrad.nnet.losses.margin_ranking_loss import margin_ranking_loss
import numpy as np

from img2caption_class import Img2Caption


def mr_loss(model, triple):
    """
    Returns the margin ranking loss, given two image embedding vectors and a "good" caption.

    Parameters
    ----------
    model : Img2Caption
        The model used to convert the image descriptors to word embeddings.
    
    triple : np.ndarray(tuple) - shape(num_tuples, 3)
        A numpy array containing tuples with three elements: the descriptor of a "good" image,
        the caption embedding corresponding to that image, and the descriptor of a "bad" image.

    Returns
    -------
    margin_ranking_loss : mg.Tensor
        The margin ranking loss of the similarities (dot products) between the word embeddings for:
            the "good" image and "good" caption,
            the "good" image and "bad image".
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

    return margin_ranking_loss(
        mg.sum(good_captions * model(good_images), axis=1),
        mg.sum(good_captions * model(bad_images), axis=1),
        1,
        0.1,
    )

