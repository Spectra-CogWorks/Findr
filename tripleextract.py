"file for extracting triples"
import json
import pickle as pkl
import numpy as np

with open("captions_train2014.json", mode="rb") as opened_file:
    captions = json.load(opened_file)
with open("resnet18_features.pkl", mode="rb") as opened_file:
    resnet = pkl.load(opened_file)

def create_train_and_test():
    """
    creates the train and test sets 

    Returns
    -------
    tuple(np.ndarray,np.ndarray)
        the ndarray of tuples for traindata
        the ndarray of tuples for valdata

    """

    # get image Ids
    img_ids = resnet.keys() # List

    train_imgID = np.random.choice(img_ids, len(img_ids)*4/5, replace=False)
    val_imgID = []
    for img in img_ids:
        if img not in train_imgID:
            val_imgID.append(img)

    # populate train_captions with 30000 captions
    train_captions = []
    for img in train_imgID:
        train_captions.append(i for i in COCO.get_caption_ids(img))
    train_captions_final = np.random.choice(train_captions,30000, replace=False)

    # populate val captions with 10000 captions
    val_captions = []
    for img in val_imgID:
        val_captions.append(i for i in COCO.get_caption_ids(img))
    val_captions_final = np.random.choice(val_captions,10000, replace=False)
    train_data = extract_triples(train_captions_final)
    val_data = extract_triples(val_captions_final)
    return (train_data, val_data)
    

    

def extract_triples(captions):
    """
    Parameters
    ----------   

    captions : 
        the set of captions that we need to extract triples from
    
    Returns
    -------
    triple_array = np.ndarray( (d_goodimg, w_goodcap, d_badimg) )
    """
    
    # for each caption_id get 25 !!!!other!!!!! random caption_ids that belong to different images
    for cap_ID in captions:
        while(10 < 2):
            np.random.choice(captions,25) 

    # calculate similarities between og caption_id and random caption_ids
    # choose highest cosine_similarity  and get img_id for badImg
    # goodImg = image_id associated with original caption_id
    # get descriptors for badImg and goodImg
    # get caption embed for orginalCaption
    resnet.keys() 

