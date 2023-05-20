from torch.utils.data import Dataset
import os
import evaluate
from PIL import Image
from transformers import SegformerImageProcessor, get_scheduler, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from datasets import load_metric
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from datetime import datetime

current_time = datetime.now().time()
print(current_time) #FOR LOG DIFF


NUM_CHANNEL = 6

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train

        sub_path = "training_demo" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir, "images", sub_path)
        self.ann_dir = os.path.join(self.root_dir, "masks", sub_path)

        print(self.img_dir)

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #READ IMAGE
        image = rasterio.open(os.path.join(self.img_dir, self.images[idx]));
        # before_image = image.read()
        image = image.read() / 4095 * 255
        image = (np.rint(image)).astype(int)
        image = np.clip(image, 0, 255)
        image1 = np.transpose(image[:3], (1, 2, 0))
        image2 = np.transpose(image[-3:], (1, 2, 0))
        #READ MAP
        segmentation_map = np.rint(rasterio.open(os.path.join(self.ann_dir, self.annotations[idx])).read() / 255).astype(int);

        # randomly crop + pad both image and segmentation map to same size
        # encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        input_tensors1 = self.feature_extractor(images=image1, segmentation_maps=segmentation_map, return_tensors="pt")
        input_tensors2 = self.feature_extractor(images=image2, segmentation_maps=segmentation_map, return_tensors="pt")

        encoded_inputs = {}
        encoded_inputs['pixel_values'] = torch.from_numpy(
            np.concatenate((input_tensors1.pixel_values, input_tensors2.pixel_values), axis=1))
        encoded_inputs['labels'] = input_tensors2.labels;

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


root_dir = 'dataset'
feature_extractor = SegformerImageProcessor.from_pretrained("imadd/segformer-b0-finetuned-segments-water-2")

train_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor)
valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(valid_dataset))

encoded_inputs = train_dataset[0]
print(encoded_inputs["pixel_values"].shape)
print(encoded_inputs["labels"].shape)
print(encoded_inputs["labels"])
print(encoded_inputs["labels"].squeeze().unique())

## DATALOADERS

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k, v.shape)

print(batch["labels"].shape)
mask = (batch["labels"] != 255)
print(mask)
print(batch["labels"][mask])

## DEFINE MODEL
model = SegformerForSemanticSegmentation.from_pretrained("imadd/segformer-b0-finetuned-segments-water-2", num_labels=2)
new_config = model.config
new_config.num_channels=NUM_CHANNEL
# new_config.num_labels=1
new_model = SegformerForSemanticSegmentation(new_config)
# print(model.segformer.encoder)
model.segformer.encoder.patch_embeddings = new_model.segformer.encoder.patch_embeddings
model.segformer.encoder.block[0][0] = new_model.segformer.encoder.block[0][0]
# model = new_model
# print('---------------------------------------')
# print(model.segformer.encoder)
mean_iou = evaluate.load("mean_iou")


# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
for epoch in range(1):  # loop over the dataset multiple times
    print("Epoch:", epoch)
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # get the inputs;
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        loss.backward()
        optimizer.step()

        # evaluate
        with torch.no_grad():
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
                                                         align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            # note that the metric expects predictions + labels as numpy arrays
            mean_iou.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # let's print loss and metrics every 100 batches
        if idx % 10 == 0:
            metrics = mean_iou.compute(num_labels=1,
                                      ignore_index=255,
                                      reduce_labels=False,  # we've already reduced the labels before)
                                      )

            print("Loss:", loss.item())
            print("Mean_iou:", metrics["mean_iou"])
            print("Mean accuracy:", metrics["mean_accuracy"])


## Inference
image = rasterio.open('dataset/images/training/0.tif')
image = image.read() / 4095 * 255
image = (np.rint(image)).astype(int)
plt.imshow(np.transpose(image[0:3], (1,2,0)))
plt.show()

image = np.clip(image, 0, 255)
image1 = np.transpose(image[:3], (1, 2, 0))
image2 = np.transpose(image[-3:], (1, 2, 0))


input_tensors1 = feature_extractor(images=image1, return_tensors="pt")
input_tensors2 = feature_extractor(images=image2, return_tensors="pt")

encoding = {}
encoding['pixel_values'] = torch.from_numpy(
    np.concatenate((input_tensors1.pixel_values, input_tensors2.pixel_values), axis=1))
# encoding['labels'] = input_tensors2.labels;
pixel_values = encoding['pixel_values']
print(pixel_values.shape)
outputs = model(pixel_values=pixel_values)
logits = outputs.logits.cpu()
print(logits.shape)


def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[0,0,0], [255, 255, 255]]


# First, rescale logits to original image size
upsampled_logits = nn.functional.interpolate(logits,
                size=image.shape[-2:], # (height, width)
                mode='bilinear',
                align_corners=False)

# Second, apply argmax on the class dimension
seg = upsampled_logits.argmax(dim=1)[0]
color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
palette = np.array(ade_palette())
for label, color in enumerate(palette):
    color_seg[seg == label, :] = color
# Convert to BGR
color_seg = color_seg[..., ::-1]

# Show image + mask
img = np.transpose(image[0:3], (1,2,0)) * 0.5 + color_seg * 0.5
# img = color_seg
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()


## SHOW ACTUAL MAP
mask = np.rint(rasterio.open('dataset/masks/training/0.tif').read() / 255).astype(int)
mask = np.transpose(mask, (1,2,0))
plt.imshow(mask, cmap='gray')
plt.show()

# convert map to NumPy array
# map = np.array(map)
# mask[mask == 0] = 255 # background class is replaced by ignore_index
# mask = mask - 1 # other classes are reduced by one
# map[map == 254] = 255
#
# classes_map = np.unique(mask).tolist()
# unique_classes = [model.config.id2label[idx] if idx!=255 else None for idx in classes_map]
# print("Classes in this image:", unique_classes)
#
# # create coloured map
# color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) # height, width, 3
# palette = np.array(ade_palette())
# for label, color in enumerate(palette):
#     color_seg[mask == label, :] = color
# # Convert to BGR
# color_seg = color_seg[..., ::-1]
#
# # Show image + mask
# img = np.array(image) * 0.5 + color_seg * 0.5
# img = img.astype(np.uint8)
#
# plt.figure(figsize=(15, 10))
# plt.imshow(img)
# plt.show()
# seg.unique()

metrics = mean_iou.compute(predictions=[seg.numpy()], references=[mask], num_labels=2, ignore_index=255)
print(metrics.keys())

import pandas as pd

# print overall metrics
for key in list(metrics.keys())[:3]:
  print(key, metrics[key])

