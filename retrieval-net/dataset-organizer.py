import os
import shutil

# Path to the folder containing all the images
image_folder = 'data/dataset'

# Path to the destination folder where the images will be organized
destination_folder = 'data/organized_dataset'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Loop through all the images in the source folder
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg'):
        # Extract the image information from the filename
        image_info = filename.split('_')
        gender = image_info[1]

        # Create the gender folder if it doesn't exist
        gender_folder = os.path.join(destination_folder, gender)
        os.makedirs(gender_folder, exist_ok=True)

        # Generate a new filename by adding a numerical suffix to avoid filename conflicts
        destination_filename = os.path.join(gender_folder, filename)

        # Copy the image to the destination folder
        shutil.copy(os.path.join(image_folder, filename), destination_filename)

print('Images organized successfully.')
