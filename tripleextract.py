"file for extracting triples"
import json
import pickle
import numpy as np
from COCO_class import cosine_similarity, coco
from descriptors import generate_descriptor
from img2caption_class import Img2Caption
import random

with open("resnet18_features.pkl", mode="rb") as opened_file:
    resnet = pickle.load(opened_file)

def create_train_and_test():
    """
    Creates the train and test sets.

    Parameters
    ----------
    None

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        A tuple containing:
            The ndarray of tuples (triplets) for train_data,
            The ndarray of tuples (triplets) for val_data.
    """
    # get image Ids
    img_ids = np.array(resnet.keys()) # List

    # Shuffle the image IDs 
    np.random.shuffle(img_ids)

    # Division of the image IDs between the training and validation sets through slicing
    train_imgids = img_ids[:(4 * len(img_ids)) // 5]
    val_imgids = img_ids[(4 * len(img_ids)) // 5:]

    # To restrict the captions, just add a count variable that breaks from the for loop after a certain threshold
    # is reached.
    # Populate train_captions with all the captifons
    train_captions = []
    count = 0
    for img in train_imgids:
        if count >= 300:
            break
        
        train_captions.extend(coco.get_captions(img))
        count += 1
    
    # This is ensuring that the captions are randomized and not clustered by image
    train_captions_final = np.random.choice(train_captions, replace=False)

    # Populate val_captions with all the captions
    val_captions = []
    count = 0
    for img in val_imgids:
        if count >= 100:
            break
        
        val_captions.extend(coco.get_captions(img))
        count += 1
    
    # This is ensuring that the captions are randomized and not clustered by image
    val_captions_final = np.random.choice(val_captions, replace=False)

    # Use the extract_triples() to grab the truples
    train_data = extract_triples(train_captions_final)
    val_data = extract_triples(val_captions_final)

    return (train_data, val_data)
    
def extract_triples(caption_ids):
    """
    Parameters
    ----------
    captions : np.ndarray shape=(N,)
        The set of caption IDs that we need to extract triples from.
    
    Returns
    -------
    final_truples: list shape=(1, 10 * N)
        A numpy array where the row is a bunch of tuples.
    """
    final_truples = []

    for good_cap in caption_ids:
        bad_caps = []
        
        while(len(bad_caps) < 10):
            bad_batch_cap = []
            bad_batch_img = []
            
            while(len(bad_batch_img) < 25):
                bad_cap = random.choice(caption_ids)
                bad_img = coco.get_image_id(bad_cap)
                
                if  bad_img in bad_batch_img or bad_img == coco.get_image_id(good_cap):
                    # TODO Double check the intuition for this section
                    continue
                else:
                    bad_batch_cap.append(bad_cap)
                    bad_batch_img.append(bad_img)
            
            cos_sims = {}
            for bad_cap in bad_batch_cap:
                cos_sim = cosine_similarity(coco.get_caption_embedding(good_cap), coco.get_caption_embedding(bad_cap))
                cos_sims[cos_sim] = bad_cap
            
            # TODO This is almost certainly wrong
            final_bad_cap = cos_sims[min(cos_sims)]
            
            bad_caps.append(final_bad_cap)
            
        truple = [(generate_descriptor(coco.get_image_id(good_cap)), 
                   coco.get_caption_embedding(good_cap), 
                   generate_descriptor(coco.get_image_id(bad_cap))) for bad_cap in bad_caps]

        final_truples.extend(truple)
    
    print("Success")
    return final_truples