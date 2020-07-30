"file for extracting triples"
import pickle
import numpy as np
import random

from COCO_class import cosine_similarity, coco #pylint: disable=import-error
from descriptors import generate_descriptor #pylint: disable=import-error
from img2caption_class import Img2Caption #pylint: disable=import-error

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
    tuple(list, list)
        A tuple containing:
            The list of tuples (triplets) for train_data,
            The list of tuples (triplets) for val_data.
    """
    # Get image IDs
    img_ids = coco.get_all_image_ids()

    # Shuffle the image IDs
    random.shuffle(img_ids)

    # Division of the image IDs between the training and validation sets through slicing
    train_imgids = img_ids[:(4 * len(img_ids)) // 5]
    val_imgids = img_ids[(4 * len(img_ids)) // 5:]

    # To restrict the captions, just add a count variable that breaks from the for loop after a certain threshold
    # is reached.
    # Populate train_captions with all the captifons
    train_captions = []
    count = 0
    for img in train_imgids:
        # This stops for loop for testing purposes
        if len(train_captions) >= 25:
            break
        
        train_captions.extend(coco.get_caption_ids(img))
        count += 1
    
    # This is ensuring that the captions are randomized and not clustered by image
    random.shuffle(train_captions)

    # Populate val_captions with all the captions
    val_captions = []
    for img in val_imgids:
        # This stops for loop for testing purposes
        if len(val_captions) >= 25:
            break
        
        val_captions.extend(coco.get_caption_ids(img))
    
    # This is ensuring that the captions are randomized and not clustered by image
    random.shuffle(val_captions)
    
    print(len(train_captions))
    print(len(val_captions))

    # Use the extract_triples() to grab the truples
    train_data = extract_triples(train_captions)
    val_data = extract_triples(val_captions)
    
    return (train_data, val_data)
    
def extract_triples(caption_ids):
    """
    Parameters
    ----------
    captions : list size=(N,)
        The set of caption IDs that we need to extract triples from.
    
    Returns
    -------
    final_truples: list size=(1, 10 * N)
        A numpy array where the row is a bunch of tuples.
    """
    final_truples = []
    #print('I have reached the tuple')

    for good_cap in caption_ids:
        bad_caps = []
        #print("I am going through one good caption")
        #print("I am going through the 10 bad captions")
        while(len(bad_caps) < 10):
            cos_sims = {}
            
            #print("I am finding a bad image")
            bad_batch_caps = list(np.random.choice(caption_ids, size=25))
            for bad_cap in bad_batch_caps:
                if (bad_cap in cos_sims.values() or coco.get_image_id(bad_cap) == coco.get_image_id(good_cap)):
                    continue
                else:
                    cos_sim = cosine_similarity(coco.get_caption_embedding(good_cap), coco.get_caption_embedding(bad_cap))
                    cos_sims[cos_sim] = bad_cap
            
            final_bad_cap = cos_sims[min(cos_sims)]
            
            bad_caps.append(final_bad_cap)
            #print("One bad image was added")
            #print()
        #print()
         
        truple = [(generate_descriptor(coco.get_image_id(good_cap)), 
                   coco.get_caption_embedding(good_cap), 
                   generate_descriptor(coco.get_image_id(bad_cap))) for bad_cap in bad_caps]

        final_truples.extend(truple)
    
    print("Success")
    return final_truples