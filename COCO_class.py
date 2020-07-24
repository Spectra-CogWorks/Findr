import json
from pathlib import Path
import numpy as np
from numpy.linalg import norm
import io
import requests
from PIL import Image
import matplotlib.pyplot as plt
import gensim
from gensim.models.keyedvectors import KeyedVectors
import re, string
from typing import Dict, List, Iterable
from collections import Counter

from descriptors import generate_descriptor
from img2caption_class import Img2Caption

_PUNC_REGEX = re.compile('[{}]'.format(re.escape(string.punctuation)))

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
	return np.sum(d1.flatten() * d2.flatten()) # / (norm(d1) * norm(d2)) - For non-unit vectors
	
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
	def __init__(self, file_path=Path("./captions_train2014.json"), glove_path="./glove.6B.50d.txt.w2v"):
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
		self.glove = KeyedVectors.load_word2vec_format(glove_path, binary=False)
		
		if file_path.exists():
			self.database = json.load(open(file_path))
			self.init_image_mappings()
			self.init_annotation_mappings()
		else:
			print("Import Error: Invalid file path")
			
		self.idf = self.inverse_document_frequency(self.all_captions)

	def init_image_mappings(self):
		"""Method to be called once by `__init__` to set up instance variables for image mappings
		"""
		self.image_ids = []

		for image in self.database["images"]:
			self.image_ids.append(image["id"])

	def init_annotation_mappings(self):
		"""Method to initialize all caption mapping instance variables. It is called once in __init__ to
		"""
		self.all_caption_ids = []
		self.all_captions = []
		self.caption_id_to_img_id = {}
		self.img_ids_to_caption_ids = {}
		self.img_ids_to_captions = {}
		
		self.caption_id_to_caption = {}
		
		for caption_dict in self.database["annotations"]:
			self.all_caption_ids.append(caption_dict["id"])
			
			self.all_captions.append(caption_dict["caption"])
			
			self.caption_id_to_img_id[caption_dict["id"]] = caption_dict["image_id"]
				
			if caption_dict["image_id"] in self.img_ids_to_caption_ids:
				self.img_ids_to_caption_ids[caption_dict["image_id"]].append(caption_dict["id"])
			else:
				self.img_ids_to_caption_ids[caption_dict["image_id"]] = [caption_dict["id"]]
				
			if caption_dict["image_id"] in self.img_ids_to_captions:
				self.img_ids_to_captions[caption_dict["image_id"]].append(caption_dict["caption"])
			else:
				self.img_ids_to_captions[caption_dict["image_id"]] = [caption_dict["caption"]]
			
			self.caption_id_to_caption[caption_dict["id"]] = caption_dict["caption"]
	
	def get_all_caption_ids(self):
		"""Resurns a list of caption IDs
		
		Returns
		-------
		caption_ids : List[int]

		"""
		return self.all_caption_ids

	
	def get_all_captions(self):
		"""
		Gets all captions from the database
		
		Returns
		-------
		captions : List[string]
			List of captions
		"""
		return self.all_captions
		
	
	def get_all_image_ids(self):
		"""Gets all image IDs as a list

		Returns
		-------
		image_ids : List[int]
			The image IDs
		"""
		return self.image_ids
	
	def get_image_id(self, caption_id):
		"""Get the associated image ID from a caption ID
		
		Parameters
		----------
		caption_id : int
			The caption ID
		
		Returns
		-------
		image_id : int
		"""
		if caption_id in self.caption_id_to_img_id:
			return self.caption_id_to_img_id[caption_id]
		#else:
			# print("No caption with given ID was found. This function has returned a None-type object")
			#return None
	
	def get_caption_ids(self, image_id):
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
		return self.img_ids_to_caption_ids[image_id]
	
	def get_captions(self, image_id):
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
		return self.img_ids_to_captions[image_id]
	
	# ? DO NOT MOVE TO __init___
	def get_caption_embedding(self, caption_id):
		"""Gets the embedding for ID 
		
		Parameters
		----------
		caption_id: int
			The ID for which to generate for

		Returns
		-------
		caption_embed : np.ndarray - shape(50,)
			The weighted sum embedding fo the specified caption
		""" 
		return self.create_text_embedding(self.caption_id_to_caption[caption_id])
		
	def get_words(self, text: str) -> List[str]:
		""" Returns all the words in a string, removing punctuation.

		Parameters
		----------
		text : str
			The text whose words to return.

		Returns
		-------
		List[str]
			The words in `text`, with no punctuation.
		"""

		return _PUNC_REGEX.sub(" ", text.lower()).split()

	def _compute_doc_freq(self, documents: Iterable[str]) -> Counter:
		""" Computes document frequency (the "DF" in "TF-IDF").

		Parameters
		----------
		documents : List(str)
			The list of documents.

		Returns
		-------
		collections.Counter[str, int]
			The dictionary's keys are words and its values are number of documents the word appeared in.
		"""

		df = Counter()
		for doc in documents:
			df.update(set(self.get_words(doc)))

		return df

	def inverse_document_frequency(self, documents: Iterable[str]) -> Dict[str, int]:
		""" Computes the inverse document frequency document frequency (the "DF" in "TF-IDF").

		Parameters
		----------
		documents : List(str)
			The list of documents.

		Returns
		-------
		collections.Counter[str, int]
			The dictionary's keys are words and its values are number of documents the word appeared in.
		"""
		df = self._compute_doc_freq(documents)
		return {word: np.log(len(documents) / (1.0 + count)) for word, count in df.items()}
	
	def create_text_embedding(self, text):
		"""
		Creates text embeddings of captions and query text.

		Parameters
		----------
		text : str
			The text to be converted into word embeddings.

		Returns
		-------
		embeddings : np.ndarray
			A shape-(1, 50) numpy array of embeddings for the input text, weighed according to each word's IDF.
		"""
		text_array = self.get_words(text)
		
		embedding = np.zeros((1, 50))
		
		for item in text_array:
			if item not in self.glove:
				continue
			
			current_idf = self.idf[item]
			
			embedding += self.glove[item] * current_idf
			
		embedding /= np.linalg.norm(embedding)
		
		return embedding

	
	def find_similar_images(self, model, query, k):
		"""Create function that finds top k similar images to a query embedding
		
		Parameters
		----------
		model : Img2Caption
			The trained model to convert images to image embeddings
		
		query : np.ndarray - shape(50,)
			The embedding of the query
			
		k : int
			The number of images to return
		
		Return
		------
		image_ids : List[int]
			A list of the image ids that are similar to the query embedding
		"""
		ids_to_scores = {}
		
		for image in self.database["images"]:
			img_descriptor = generate_descriptor(image["id"])
			
			if img_descriptor is not None:
				ids_to_scores[image["id"]] = cosine_similarity(query, model(img_descriptor))
		
		# ! Watch out for possible errors here
		# This is just sorting the dictionary by the values, which are the similarities
		# It is sorted in ascending order and is then passed to the list comprehension
		# It only accepts the last k image ids as they are the greatest in similarity
		return [k for k, v in sorted(ids_to_scores.items(), key=lambda item: item[1])][-k:]
        		
	def display_images(self, image_ids):
			"""Displays images using given image IDs
			
			Parameters
			----------
			image_ids :  List[int]
				Image IDs to display
			
			Returns
			-------
			None
			"""
			
			for image in self.database["images"]:
				if image["id"] in image_ids:
					img = download_image(image["coco_url"])
					img_arr = np.array(img)

					fig, ax = plt.subplots() # pylint: disable=unused-variable
					ax.imshow(img_arr)
					plt.show(block=True)

coco = COCO()