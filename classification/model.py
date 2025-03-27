from torchvision.models import densenet121
import torch.nn as nn
import torch


class NailPsoriasisPredictor(nn.Module):
    def __init__(self, num_classes, checkpoint_path=None):
        super(NailPsoriasisPredictor, self).__init__()

        # Load pre-trained ResNet50 model
        self.densenet121 = densenet121(pretrained=True)

        # Modify the last fully connected layer for the number of classes
        in_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(in_features, num_classes)

        # Define super classes
        self.super_classes = {
            'Nail Psoriasis': [9, 10],
            'Healthy': [5],
            'Other Disease': [0, 1, 2, 3, 4, 6, 7, 8, 11]
        }

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def forward(self, x):
        return self.densenet121(x)

    def predict(self, input_image):
        # Set the model to evaluation mode
        self.eval()

        # Send input image to the same device as the model
        input_image = input_image.to(next(self.parameters()).device)

        # Perform the forward pass
        with torch.no_grad():
            output = self.forward(input_image)

        # Get the predicted label index
        _, predicted_label = torch.max(output, 1)
        predicted_label = predicted_label.item()

        # Map the predicted label to super class
        super_class_label = self.map_to_super_class(predicted_label)

        return predicted_label, super_class_label

    def map_to_super_class(self, label):
        for super_class, class_indices in self.super_classes.items():
            if label in class_indices:
                return super_class
        return 'Unknown Super Class'  # Handle cases where the label doesn't belong to any defined super class

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.densenet121.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")

    def import_gradcam(self, gradcam):
        self.gradcam = gradcam



class NailPsoriasisExplainer:
    def __init__(self, model, gradcam):
        self.model = model
        self.gradcam = gradcam

    def explain(self, input_image):
        # Get the predicted label
        predicted_label, _ = self.model.predict(input_image)

        # Get the Grad-CAM heatmap
        heatmap = self.gradcam(input_image, predicted_label)

        return heatmap