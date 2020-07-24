"file for extracting triples"
import json
import pickle as pkl
import numpy as np
from COCO_class import cosine_similarity
from descriptors import generate_descriptor

with open("captions_train2014.json", mode="rb") as opened_file:
    captions = json.load(opened_file)
with open("resnet18_features.pkl", mode="rb") as opened_file:
    resnet = pkl.load(opened_file)

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
    train_captions_final = np.random.choice(train_captions, 30000, replace=False)

    # populate val_captions with 10000 captions
    val_captions = []
    for img in val_imgID:
        val_captions.append(i for i in COCO.get_caption_ids(img))
    val_captions_final = np.random.choice(val_captions, 10000, replace=False)

    # use extract_triples function to get final train/val data
    train_data = extract_triples(train_captions_final)
    val_data = extract_triples(val_captions_final)

    return (train_data, val_data)
    
def extract_triples(model, captions):
    """
    Parameters
    ----------
    model
        Instance of the Img2Cap class.

    captions : np.ndarray
        The set of captions that we need to extract triples from.
    
    Returns
    -------
    triple_array : np.ndarray
        A numpy array where each row is the tuple (d_good_img, w_good_cap, d_bad_img).
    """
    # for each caption_id get 25 !!!!other!!!!! random caption_ids that belong to different images

    # Possible code
    all_cap_ids = COCO.get_all_caption_ids()
    all_bad_caps = []
    for cap_ID in all_cap_ids:
        if cap_ID not in captions:
            all_bad_caps.append(cap_ID)
    
    bad_cap_ids = []
    for good_cap in captions:
        bad_cnt = 0
        while bad_cnt < 10:
            bad_batch = np.random.choice(all_bad_caps, 25)
            cos_sims = {}
            for bad_cap in bad_batch:
                cos_sim = cosine_similarity(COCO.get_caption_embedding(good_cap), COCO.get_caption_embedding(bad_cap))
                cos_sims[cos_sim] = bad_cap
            bad_cap_ids.append(cos_sims[max(cos_sims.keys())])
            bad_cnt += 1
    
    bad_img_descriptors = []
    for cap_ID in bad_cap_ids:
        bad_img_d = generate_descriptor(COCO.get_image_id(cap_ID))
        bad_img_descriptors.append(bad_img_d)
    
    good_img_descriptors = []
    for cap_ID in captions:
        good_img_d = generate_descriptor(COCO.get_image_id(cap_ID))
        good_img_descriptors.append(good_img_d)
    
    good_captions = []
    for cap_ID in captions:
        good_captions.append(COCO.get_caption_embedding(cap_ID))
    
    triple_array = []
    count = 0
    for i in range(len(good_img_descriptors)):
        bad_d = bad_img_descriptors[count * 10 : (count + 1) * 10]
        for j in range(len(bad_d)):
            triple_array.append((good_img_descriptors[i], good_captions[i], bad_d[j]))
    
    # or Tensor?
    return np.array(triple_array)



    # original code
    for cap_ID in captions:
        while(10 < 2):
            np.random.choice(captions, 25) 

    # calculate similarities between og caption_id and random caption_ids (need to check that these random caps
        # aren't associated with the goodImg)
    # choose highest cosine_similarity  and get img_id for badImg
    # goodImg = image_id associated with original caption_id
    # get descriptors for badImg and goodImg
    # get caption embed for orginalCaption
    resnet.keys() 

