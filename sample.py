import os
import shutil
import random

# Define paths
data_dir = "data/kagglecatsanddogs_3367a/PetImages"
input_data_dir = "data/input_data"
cat_dir = os.path.join(data_dir, "Cat")
dog_dir = os.path.join(data_dir, "Dog")

# Define output directories
input_cat_dir = os.path.join(input_data_dir, "Cat")
input_dog_dir = os.path.join(input_data_dir, "Dog")

# Create input_data directory and its subdirectories
os.makedirs(input_cat_dir, exist_ok=True)
os.makedirs(input_dog_dir, exist_ok=True)

# Function to sample and copy images
def sample_images(source_dir, dest_dir, num_samples=500):
    images = [img for img in os.listdir(source_dir) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    sampled_images = random.sample(images, min(num_samples, len(images)))

    for img in sampled_images:
        src_path = os.path.join(source_dir, img)
        dest_path = os.path.join(dest_dir, img)
        shutil.copy(src_path, dest_path)

# Sample images from Cat and Dog folders
sample_images(cat_dir, input_cat_dir, 500)
sample_images(dog_dir, input_dog_dir, 500)

print("Sampling complete! Images stored in 'data/input_data'.")
