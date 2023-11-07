import os
import shutil

def keep_one_photo_per_id(input_folder_path, output_folder_path):
    # Dictionary to track the IDs that have been encountered
    encountered_ids = {}

    # Ensure the output directory exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # List all files in the folder
    for filename in os.listdir(input_folder_path):
        # Check if the file is an image based on the extension (you can adjust the tuple of extensions)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Extract the ID from the filename (everything before the first underscore)
            image_id = filename.split('_')[0]

            # If the ID has not been encountered yet, keep the photo
            if image_id not in encountered_ids:
                encountered_ids[image_id] = filename

                # Copy the file to the output folder
                src_file_path = os.path.join(input_folder_path, filename)
                dst_file_path = os.path.join(output_folder_path, filename)
                shutil.copy2(src_file_path, dst_file_path)
            else:
                # If we've already encountered this ID, skip or delete the extra photo
                print(f"Skipping or deleting {filename}")  # Placeholder for your action

    # The encountered_ids dict now has one filename for each unique ID
    return encountered_ids

# Usage
input_folder_path = 'dataset/hands/healthy/1-501'
output_folder_path = 'dataset/hands/healthy/'
unique_photos = keep_one_photo_per_id(input_folder_path, output_folder_path)

# If you want to see the list of photos that are kept
for image_id, filename in unique_photos.items():
    print(f"Keeping photo {filename} for ID {image_id}")
