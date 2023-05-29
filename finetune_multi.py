from torch.utils.data import Dataset
import os
import evaluate
from PIL import Image
from transformers import SegformerImageProcessor, get_scheduler, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import numpy as np
import PIL
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
print(current_time)  # FOR LOG DIFF

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

        sub_path = "training" if self.train else "validation"
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
        # READ IMAGE
        image = rasterio.open(os.path.join(self.img_dir, self.images[idx]));
        image = image.read() / 4095 * 255
        image = (np.rint(image)).astype(int)
        image = np.clip(image, 0, 255)
        image1 = np.transpose(image[:3], (1, 2, 0))
        image2 = np.transpose(image[-3:], (1, 2, 0))

        # READ MAP
        segmentation_map = np.rint(
            rasterio.open(os.path.join(self.ann_dir, self.annotations[idx])).read() / 255).astype(int);

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

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1)

batch = next(iter(train_dataloader))
for k, v in batch.items():
    print(k, v.shape)

mask = (batch["labels"] != 255)

## DEFINE MODEL
model = SegformerForSemanticSegmentation.from_pretrained("imadd/segformer-b0-finetuned-segments-water-2", num_labels=2)
new_config = model.config
new_config.num_channels = NUM_CHANNEL
new_model = SegformerForSemanticSegmentation(new_config)
model.segformer.encoder.patch_embeddings = new_model.segformer.encoder.patch_embeddings
model.segformer.encoder.block[0][0] = new_model.segformer.encoder.block[0][0]
# model.load_state_dict(torch.load('./checkpoints/model-15-23:33:23.808260.pt')['model_state_dict'])
mean_iou = evaluate.load("mean_iou")

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epoch_counter = 0

model.train()
for epoch in range(15):  # loop over the dataset multiple times
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
            mean_iou.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        if idx % 5 == 0:
            metrics = mean_iou.compute(num_labels=1,
                                      ignore_index=255,
                                      reduce_labels=False,  # we've already reduced the labels before
                                      )

            print("Loss:", loss.item())
            print("Mean_iou:", metrics["mean_iou"])
            print("Mean accuracy:", metrics["mean_accuracy"])

    print(f"epoch {epoch_counter} loss:", loss.item())
    print(f"epoch {epoch_counter} mean_iou:", metrics["mean_iou"])
    print(f"epoch {epoch_counter} mean accuracy:", metrics["mean_accuracy"])

    epoch_counter = epoch_counter + 1


# SAVE MODEL CHECKPOINT

PATH = f"checkpoints/model-{epoch_counter}-{current_time}.pt"
torch.save({
            'epoch': epoch_counter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            }, PATH)

# Set the path to the folder containing the TIFF images
folder_path = 'dataset/ready_for_segmentation/demo'

# Get a list of TIFF files in the folder
tiff_files = [file for file in os.listdir(folder_path)]

# Initialize an empty list to store the images
ready_for_segmentation = []

# Loop through each TIFF file
for file in tiff_files:
    # Construct the full file path
    file_path = os.path.join(folder_path, file)

    # Open the TIFF file using rasterio
    with rasterio.open(file_path) as src:
        # Read the image data
        image = src.read() / 4095 * 255
        image = (np.rint(image)).astype(int)
        # Append the image to the list
        ready_for_segmentation.append(image)
        print(len(ready_for_segmentation))
for prediction_image in ready_for_segmentation:
    ## Inference

    image = np.clip(prediction_image, 0, 255)
    image1 = np.transpose(image[:3], (1, 2, 0))
    image2 = np.transpose(image[-3:], (1, 2, 0))

    input_tensors1 = feature_extractor(images=image1, return_tensors="pt")
    input_tensors2 = feature_extractor(images=image2, return_tensors="pt")

    encoding = {}
    encoding['pixel_values'] = torch.from_numpy(
        np.concatenate((input_tensors1.pixel_values, input_tensors2.pixel_values), axis=1))
    pixel_values = encoding['pixel_values']
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()

    def ade_palette():
        """ADE20K palette that maps each class to RGB values."""
        return [[0, 0, 0], [255, 255, 255]]


    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(logits,
                                                 size=image.shape[-2:],  # (height, width)
                                                 mode='bilinear',
                                                 align_corners=False)

    # Second, apply argmax on the class dimension
    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.transpose(image[0:3], (1, 2, 0)) * 0.5 + color_seg * 0.5
    # img = color_seg
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()

    color_seg=color_seg.mean(axis=2)
    print(color_seg)
    print(color_seg.shape)
    color_seg = (np.rint(color_seg)).astype(int)
    print(np.unique(color_seg, return_counts=True))
    plt.imshow(color_seg)
    plt.show()
