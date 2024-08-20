# predict.py
import argparse
import json
import numpy as np
import tensorflow as tf
from utility import process_image, TFHubLayer

def load_model(model_path):
    """Load a saved Keras model with custom layers."""
    return tf.keras.models.load_model(model_path, custom_objects={'TFHubLayer': TFHubLayer})

def predict(image_path, model, top_k):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    # Process the image
    processed_image = process_image(image_path)
    
    # Make predictions
    predictions = model.predict(processed_image)
    
    # Get the top K predictions
    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_k_probs = predictions[0][top_k_indices]
    return top_k_probs, top_k_indices

def main():
    parser = argparse.ArgumentParser(description="Predict the top K most likely classes of a flower image.")
    
    # Positional arguments
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_path', type=str, help='Path to the saved Keras model')
    
    # Optional arguments
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model_path)
    
    # Predict the top K classes
    probs, classes = predict(args.image_path, model, args.top_k)
    
    # Map class indices to names if category_names is provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        class_labels = [class_names.get(str(cls), f"Class {cls}") for cls in classes]
    else:
        class_labels = [str(cls) for cls in classes]
    
    # Print out the results
    print("Top K predictions:")
    for prob, label in zip(probs, class_labels):
        print(f"{label}: {prob:.4f}")

if __name__ == "__main__":
    main()
