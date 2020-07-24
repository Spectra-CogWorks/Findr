import click
from pathlib import Path

import text_embedding as te
import descriptors as de
from img2caption_class import Img2Caption
from COCO_class import COCO

# We might have to look into coding chained commands to prevent database from having issues
@click.group()
def cli():
    """This is the encompassing commandline group that contains all the commands for the Findr package
    
    It follows this process:
    
    1) Embed our query text: se_text()

    2) Embed the images in our database: se_image()

    3) Compute the similarity between our query embedding, se_text(query), and all image embeddings se_image(img) in the database

    4) Return the top k most similar images (images with the highest cosine similarity)
    
    """
    pass

@cli.command()                  
@click.argument('query_text', type=click.STRING)
@click.option('-k', default=4, type=click.INT)
@click.option('-f', '--filepath', type=click.STRING)
@click.option('-w', '--weights_filepath', type=click.STRING)
def search(query_text, k, filepath, weights_filepath):
    """ This function goes through the full search process using all the other files in the package"""
    # Initialize database
    COCO.import_database(Path(filepath))
    
    # Create the query embedding
    query_embed = te.create_text_embedding(query_text)
    
    # TODO Doublecheck the model weight and bias import and export functions
    # Model is instantiated with proper weights
    model = Img2Caption()
    model.import_weights(Path(weights_filepath))
    
    # TODO Check that generate_descriptor() is correct
    # This gets the embeddings of all the images
    img_ids = COCO.find_similar_images(model, query_embed, k)
    
    COCO.display_images(img_ids)
    

# This if statement is necessary in order to call the group, which then enables all of the subcommands to be called
if __name__ == "__main__":
    cli()
    