"file for extracting triples"
import json
import pickle
import numpy as np
from COCO_class import cosine_similarity, coco
from descriptors import generate_descriptor
from img2caption_class import Img2Caption

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
    img_ids = list(resnet.keys()) # List

    train_imgID = np.random.choice(img_ids, int(len(img_ids)*4/5), replace=False)
    val_imgID = []
    for img in img_ids:
        if img not in train_imgID:
            val_imgID.append(img)

    # populate train_captions with 30000 captions
    train_captions = []
    for img in train_imgID:
        for i in coco.get_caption_ids(img):
            train_captions.append(i)
    train_captions_final = np.random.choice(train_captions, 30000, replace=False)

    # populate val_captions with 10000 captions
    val_captions = []
    for img in val_imgID:
        val_captions.append(i for i in coco.get_caption_ids(img))
    val_captions_final = np.random.choice(val_captions, 10000, replace=False)

    # use extract_triples function to get final train/val data
    train_data = extract_triples(train_captions_final)
    val_data = extract_triples(val_captions_final)

    return (train_data, val_data)
    
def extract_triples(caption_ids):
    """
    Parameters
    ----------
    captions : np.ndarray shape=(30000,)
        The set of caption IDs that we need to extract triples from.
    
    Returns
    -------
    final_truples: np.ndarray shape=(1,300000)
        A numpy array where the row is a bunch of tuples.
    """
    # Instantiating the model with the trained weights
    model = Img2Caption()
    model.load_model()
    
    # for each caption_id get 25 !!!!other!!!!! random caption_ids that belong to different images

    # Get all caption IDs
    #all_cap_ids = coco.get_all_caption_ids() 

    
    #Get all bad captions: every caption not in ResNet
    #all_bad_caps = [] #filled with caption IDs
    #for cap_ID in all_cap_ids:
    #    if cap_ID not in captions:
    #       all_bad_caps.append(cap_ID)
    
    """
    Check if we need this ^
    """

    #get bad captions for each good_cap
    final_truples = [] # [ (d_good_img, w_good_cap, d_bad_img), (d_good_img, w_good_cap, d_bad_img), (d_good_img, w_good_cap, d_bad_img)...] 
                      #   , (d_good_img, w_good_cap, d_bad_img), (d_good_img, w_good_cap, d_bad_img), (d_good_img, w_good_cap, d_bad_img)...]  
                      #                                 ...                                                                                   ]
                      

    for good_cap in caption_ids:
        for i in range(10):
            # generate the 25 captions that we choose the FINAL bad caption from
            bad_batch_cap = []
            bad_batch_img = []
            while len(bad_batch_cap) < 25:
                bad_cap = random.choice(caption_ids)
                bad_img = coco.get_image_id(bad_cap)
                if  bad_img in bad_batch_img or bad_img == coco.get_image_id(good_cap):
                    continue
                else:
                    bad_batch_cap.append(bad_cap)
                    bad_batch_img.append(bad_img)
                    
            #determine which caption in bad_batch_cap is the FINAL bad caption

            cos_sims = {}
            for bad_cap in bad_batch_cap:
                cos_sim = cosine_similarity(coco.get_caption_embedding(good_cap), coco.get_caption_embedding(bad_cap))
                cos_sims[cos_sim] = bad_cap
            final_bad_cap = cos_sims[max(cos_sims.keys())]

            truple = (generate_descriptor(coco.get_image_id(good_cap)),
                    coco.get_caption_embedding(good_cap),
                    generate_descriptor(coco.get_image_id(final_bad_cap)))

            final_truples.append(truple)
    
    return np.array(final_truples)