import json
from pathlib import Path
import numpy as np
from numpy.linalg import norm
import io
import requests
from PIL import Image

from text_embedding import create_text_embedding

def cosine_similarity(d1, d2)
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
	return 1 - ((np.dot(d1, d2)) / (norm(d1) * norm(d2))

def class COCO - Pranav, Ian
	# This is a class variable that can be accessed and changed through COCO.database (WARNING: directly changing class variables in general is never a good idea as it is )
	database = {}
	
	@classmethod()
	def import_database(file_path=Path("./captions_train2014.json"))
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
	
	@classmethod()
	def get_all_caption_ids()
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

	@classmethod()
	def get_all_captions()
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
		
	@classmethod()
	def get_all_image_ids():
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

	@classmethod()
	def get_image_id(caption_id):
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

	@classmethod()
	def get_caption_ids(image_id):
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

	@classmethod()
	def get_captions(image_id):
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

	@classmethod()
	def get_caption_embedding(caption_id):
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
			if caption_dict["id"] == caption_id
				return create_text_embedding(caption_dict["caption"])

	@classmethod()
	def find_similar_images(query, k):
		"""Create function that finds top k similar images to a query image
		
		Parameters
		----------
		query : int
			The image_id
		k : int
		
		Return
		------
		image_ids : List[int]
			A list of the image ids that are similar to the query image
		"""
		image_
		
	
	@classmethod()
	def display_images(image_ids):
		"""Displays images using given image IDs
		
		Parameters
		----------
		image_ids :  List[int]
		
		
		Returns
		-------
		None
		"""