import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

class TFHubLayer(tf.keras.layers.Layer):
    def __init__(self, hub_url, **kwargs):
        super(TFHubLayer, self).__init__(**kwargs)
        self.hub_url = hub_url
        self.feature_extractor_layer = hub.KerasLayer(hub_url, trainable=False)

    def call(self, inputs):
        return self.feature_extractor_layer(inputs)

def process_image(image_path):
    """Process an image to be ready for prediction by the model."""
    image = Image.open(image_path)
    
    # Resize the image
    image = image.resize((256, 256))
    
    # Center-crop the image
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    # Convert the image to a NumPy array and normalize to [0, 1]
    image = np.array(image) / 255.0
    
    # Ensure the image has the shape (224, 224, 3)
    image = np.resize(image, (224, 224, 3))
    
    # Add a batch dimension (shape becomes [1, 224, 224, 3])
    image = np.expand_dims(image, axis=0)
    
    return image
