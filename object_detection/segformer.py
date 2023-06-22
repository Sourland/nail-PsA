import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import SegformerForSemanticSegmentation

# Load an image
image_path = "../hands/hand4.jpg"
image = Image.open(image_path)

# Keep original image dimensions for later resizing
original_width, original_height = image.size

# Define the transformations
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# Apply the transformations
processed_image = transform(image)

# Add an extra dimension (for the batch)
processed_image = processed_image.unsqueeze(0)

# Load pretrained SegFormer model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0")

# Ensure the model is in evaluation mode
model.eval()

# Pass the image through the model
output = model(processed_image)['last_hidden_state']

# Squeeze the output to remove the batch dimension and get the predicted classes
predicted_classes = output.detach().numpy()
# Resize the predicted classes back to the original image size
predicted_classes_resized = T.Resize((original_height, original_width))(predicted_classes)

# Plot original image and segmentation map
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[1].imshow(predicted_classes_resized)

plt.show()
