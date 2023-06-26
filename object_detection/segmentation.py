# import os
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# from transformers import SegformerModel
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from transformers import AdamW
#
#
# class SegmentationDataset(Dataset):
#     def __init__(self, img_dir, mask_dir=None, transform=None):
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.image_ids = os.listdir(img_dir)
#
#     def __len__(self):
#         return len(self.image_ids)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.image_ids[idx])
#         img = Image.open(img_path).convert("RGB")
#
#         if self.mask_dir:
#             mask_path = os.path.join(self.mask_dir, self.image_ids[idx])
#             mask = Image.open(mask_path)
#         else:
#             mask = torch.zeros_like(img)  # create a dummy mask
#
#         if self.transform:
#             img = self.transform(img)
#             mask = self.transform(mask)
#
#         return img, mask
#
#
# # Define the transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # resize to square
#     transforms.ToTensor(),  # convert PIL to PyTorch tensor
# ])
#
# # Define the datasets
# train_dataset = SegmentationDataset('../FreiHANDS/training/rgb', '../FreiHANDS/training/mask', transform=transform)
# eval_dataset = SegmentationDataset('../FreiHANDS/training/rgb', transform=transform)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Define the dataloaders
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
# eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=512, shuffle=False)
#
# model = SegformerModel.from_pretrained("nvidia/mit-b0")
# model.to(device)
#
# # Change the last layer based on the number of classes
# num_classes = 2
#
# # Define the optimizer and loss function
# optimizer = AdamW(model.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss()
#
# # Training
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     for batch in train_dataloader:
#         imgs, masks = batch
#         imgs, masks = imgs.to(device), masks.to(device)
#
#         optimizer.zero_grad()
#
#         outputs = model(imgs)
#         loss = criterion(outputs, masks)
#         loss.backward()
#         optimizer.step()
#
#     print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
#
# # Evaluation
# model.eval()
# with torch.no_grad():
#     for batch in eval_dataloader:
#         imgs, _ = batch
#         imgs = imgs.to(device)
#
#         outputs = model(imgs)
#         # Here you can apply a threshold to the output and compare with the ground truth
#         # This code is omitted as it depends on the exact nature of your task
