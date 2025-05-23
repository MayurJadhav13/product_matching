import os
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model, Model


# === Load Trained AutoEncoder ===
autoencoder = load_model("model/encoder_model.keras")

# Extract encoder (first half of autoencoder)
encoder = autoencoder.layers[0]

# === Load CSV with Image Data ===
df = pd.read_csv("main-dataset/data.csv")  

# === Preprocessing Function ===
def preprocess_image(image_path, target_size=(32, 32)):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# === Extract Features ===
image_features = []
for filename in df['filename']:
    local_image_path = os.path.join("main-dataset/data", filename)  
    img = preprocess_image(local_image_path)
    if img is not None:
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        features = encoder.predict(img)[0].tolist()  # Flatten to list
        image_features.append(features)
    else:
        image_features.append(None)

# Add to DataFrame
df['image_features'] = image_features

# Save to new CSV
df.to_csv("output_with_features.csv", index=False)
print("Saved CSV with image features.")

