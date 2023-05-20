from torch.utils.data import Dataset
import os
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
feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")

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
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=1, ignore_mismatched_sizes=True)
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
metric = load_metric("mean_iou")


# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
for epoch in range(30):  # loop over the dataset multiple times
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
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # let's print loss and metrics every 100 batches
        if idx % 100 == 0:
            # metrics = metric._compute(num_labels=1,
            #                           ignore_index=255,
            #                           reduce_labels=False,  # we've already reduced the labels before)
            #                           )

            print("Loss:", loss.item())
            # print("Mean_iou:", metrics["mean_iou"])
            # print("Mean accuracy:", metrics["mean_accuracy"])