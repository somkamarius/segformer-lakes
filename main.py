from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, SegformerConfig
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio



def get_seg_overlay(image, seg):
  color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
  palette = np.array([255, 255, 255])
  for label, color in enumerate(palette):
      color_seg[seg == label, :] = color

  # Show image + mask
  img = np.array(image) * 0.7 + color_seg * 0.3
  img = img.astype(np.uint8)

  return img

# image = Image.open("0.tif")
image = rasterio.open("0.tif")
# plt.imshow(image.read(3))
# plt.show()
# image = image.read()

# RESCALE AND NORMALIZE PIXEL VALUES
# neveikia normalizavimas
print(image)
before_image=image.read()

image = image.read()/4095*255
image = (np.rint(image)).astype(int)


# image=image.transpose((1,2,0)) # CHW -> HWC
image = np.clip(image, 0, 255)

print(image)

# image1 = np.concatenate((image[0:3], np.zeros((3, image.shape[1], image.shape[2]))), axis=0)
image1 = np.transpose(image[:3], (1,2,0))
image2 = np.transpose(image[-3:], (1,2,0))
# image2 = np.concatenate((np.zeros((3, image.shape[1], image.shape[2])), image[3:6]), axis=0)


NUM_CHANNEL = 6
feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")

new_config = model.config
new_config.num_channels=NUM_CHANNEL
new_model = SegformerForSemanticSegmentation(new_config)
model.segformer.encoder.block[0][0] = new_model.segformer.encoder.block[0][0]

# inputs = feature_extractor(images=image, return_tensors="pt")
input_tensors1 = feature_extractor(images=image1, return_tensors="pt")
input_tensors2 = feature_extractor(images=image2, return_tensors="pt")

input_tensors = {}
for key in input_tensors1:
    input_tensors[key] = torch.from_numpy(np.concatenate((input_tensors1[key], input_tensors2[key]), axis=0))

# print(input_tensors)
outputs = model(**input_tensors)
logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

print(logits.shape)
print(image.shape)
# First, rescale logits to original image size

upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.shape[-2:],
    mode='bilinear',
    align_corners=False
)

# # Second, apply argmax on the class dimension
pred_seg = upsampled_logits.argmax(dim=1)[0]
pred_img = get_seg_overlay(np.transpose(image[:3], (1,2,0)), pred_seg)
plt.imshow(pred_img)
plt.show()
plt.imshow(np.transpose(before_image[:3], (1,2,0)))
plt.show()