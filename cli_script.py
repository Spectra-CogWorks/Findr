import click
from pathlib import Path

import descriptors as de
from img2caption_class import Img2Caption
from COCO_class import coco

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
@click.option('-q', '--query-text', 'query_text', type=click.STRING, prompt="Please enter query text: ")
@click.option('-k', '--num-images', 'num_images', default=4, type=click.INT)
@click.option('-f', '--filepath', type=click.STRING)
@click.option('-w', '--weights-filepath', 'weights_filepath', type=click.STRING)
def search(query_text, num_images, filepath, weights_filepath):
    """ This function goes through the full search process using all the other files in the package"""
    # Create the query embedding
    query_embed = coco.create_text_embedding(query_text)
    
    # Model is instantiated with proper weights
    model = Img2Caption()
    model.load_model(Path(weights_filepath))
    
    # TODO Check find_similar_images for accuracy and optimal performance
    # This gets the embeddings of all the images
    img_ids = coco.find_similar_images(model, query_embed, num_images)
    
    coco.display_images(img_ids)

# This if statement is necessary in order to call the group, which then enables all of the subcommands to be called
if __name__ == "__main__":
    cli()
    