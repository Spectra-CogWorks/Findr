import json
from pathlib import Path
import numpy as np
from numpy.linalg import norm
import io
import requests
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

from text_embedding import create_text_embedding
from descriptors import generate_descriptor as gd

def cosine_similarity(d1, d2):
	"""Finds the cosine similarity between two image vectors
	
	Parameters
	----------
	d1 : np.ndarray
		First vector
	d2 : np.ndarray
		Second vector
		
	Return
	------
	float
	"""
	return np.dot(d1, d2) # / (norm(d1) * norm(d2)) - For non-unit vectors
	
def download_image(img_url: str) -> Image:
    """ Fetches an image from the web.

    Parameters
    ----------
    img_url : string
        The url of the image to fetch.

    Returns
    -------
    PIL.Image
        The image."""

    response = requests.get(img_url)
    return Image.open(io.BytesIO(response.content))

class COCO:
	# This is a class variable that can be accessed and changed through COCO.database (WARNING: directly changing class variables in general is never a good idea as it is )
	database = {}
	
	@classmethod
	def import_database(cls, file_path=Path("./captions_train2014.json")):
		"""
		Imports JSON file and stores its contents in the database class variable
		
		Parameters
		----------
		file_path : Path
			A string containing the file path to the JSON file
		
		Return
		------
		None
		"""
		
		if file_path.exists():
			COCO.database = json.load(open(file_path))
		else:
			print("Import Error: Invalid file path")
	
	@classmethod
	def get_all_caption_ids(cls):
		"""Resurns a list of caption IDs
		
		Returns
		-------
		caption_ids : List[int]

		"""
		caption_ids = []
		
		# Iterating through all the captions in the database
		# ! Check for the right key in the database
		for caption_dict in COCO.database["annotations"]:
			captions_ids.append(caption_dict["annotations"])
		
		return captions_ids

	@classmethod
	def get_all_captions(cls):
		"""
		Gets all captions from the database
		
		Returns
		-------
		captions : List[string]
			List of captions
		"""
		captions = []
		
		# Iterating through all the captions in the database
		# ! Check for the right key in the database
		for caption_dict in COCO.database["annotations"]:
			captions.append(caption_dict["annotations"])
		
		return captions
		
	@classmethod
	def get_all_image_ids(cls):
		"""Gets all image IDs as a list

		Returns
		-------
		image_ids : List[int]
			The image IDs
		"""
		ids = []
		
		for image in COCO.database["images"]:
			ids.append(image["id"])

		return ids

	@classmethod
	def get_image_id(cls, caption_id):
		"""Get the associated image ID from a caption ID
		
		Parameters
		----------
		caption_id : int
			The caption ID
		
		Returns
		-------
		image_id : int
		"""
		for caption_dict in COCO.database["annotations"]:
			if caption_dict["id"] == caption_id:
				return caption_dict["image_id"]
		
		print("No caption with given ID was found. This function has returned a None-type object")
		return None

	@classmethod
	def get_caption_ids(cls, image_id):
		""" Gets associated caption IDs for an image ID

		Parameters
		----------
		image_id: int
			Image ID to search for

		Returns
		-------
		caption_ids : List[int]
			Caption IDs for the list items
		"""
		return [caption["id"] for caption in COCO.database["annotations"] if caption["image_id"] == image_id]

	@classmethod
	def get_captions(cls, image_id):
		"""Gets the captions associated with the image ID
		
		Parameters
		----------
		image_id : int
			The ID to get captions for

		Returns
		-------
		captions : List[str]
			The captions
		"""
		return [caption["caption"] for caption in COCO.database["annotations"] if caption["image_id"] == image_id]

	@classmethod
	def get_caption_embedding(cls, caption_id):
		"""Gets the embedding for ID
		
		Parameters
		----------
		caption_id: int
			The ID for which to genrate for

		Returns
		-------
		caption_embed : np.ndarray - shape(50,)
			The weighted sum embedding fo the specified caption
		"""
		for caption_dict in COCO.database["annotations"]:
			if caption_dict["id"] == caption_id:
				return create_text_embedding(caption_dict["caption"])

	@classmethod
	def find_similar_images(cls, query, k):
		"""Create function that finds top k similar images to a query image
		
		Parameters
		----------
		query : np.ndarray - shape(512,)
			The descriptor vector of the query image
			
		k : int
			The number of images to return
		
		Return
		------
		image_ids : List[int]
			A list of the image ids that are similar to the query image
		"""
		scores_to_ids = {}
		
		for image in COCO.database["images"]:
			img_descriptor = gd(image["id"])
			
			if img_descriptor is not None:
				scores_to_ids[image["id"]] = cosine_similarity(query, img_descriptor)
				
		scores_to_ids = {k: v for k, v in sorted(scores_to_ids.items(), key=lambda item: item[1])}
		
		return np.ndarray(scores_to_ids.keys())[:k]
        		
	@classmethod
	def display_images(cls, image_ids):
		"""Displays images using given image IDs
		
		Parameters
		----------
		image_ids :  List[int]
			Image IDs to display
		
		Returns
		-------
		None
		"""
		
		for image in COCO.database["images"]:
			if image["id"] in image_ids:
				img = download_image(image["coco_url"])
				img_arr = np.array(img)

				fig, ax = plt.subplots()
				ax.imshow(img_arr)
				plt.show(block=True)