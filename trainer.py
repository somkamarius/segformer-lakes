from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, Trainer
import torch
from datasets import load_dataset, Image, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio
from tqdm.auto import tqdm
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader


def get_seg_overlay(image, seg):
  color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
  palette = np.array([255, 255, 255])
  for label, color in enumerate(palette):
      color_seg[seg == label, :] = color

  # Show image + mask
  img = np.array(image) * 0.7 + color_seg * 0.3
  img = img.astype(np.uint8)

  return img


imageList1 = []
imageList2 = []
masks = []

for i in range(355):
    print('starting work for image #' + str(i))
    image = rasterio.open("./images_multiband/" + str(i) + ".tif");
    before_image=image.read()
    image = image.read() / 4095 * 255
    image = (np.rint(image)).astype(int)
    image = np.clip(image, 0, 255)
    image1 = np.transpose(image[:3], (1,2,0))
    image2 = np.transpose(image[-3:], (1,2,0))
    imageList1.append(image1)
    imageList2.append(image2)
    # print(image1, image2)
    mask = np.rint(rasterio.open("./masks/" + str(i) + ".tif").read() / 255).astype(int);
    print(mask.shape)
    masks.append(mask)

##############################################
# CREATE DATASET FROM IMAGELIST1/2


## DATALOADER N MODEL

NUM_CHANNEL = 6

model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
new_config = model.config
new_config.num_channels=NUM_CHANNEL
new_config.num_labels=1
new_model = SegformerForSemanticSegmentation(new_config)
model.segformer.encoder.block[0][0] = new_model.segformer.encoder.block[0][0]
model = new_model
image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")



# train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
# eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

### OPTIMIZER AND LEANING RATE SCHEDULER

## create optimizer

optimizer = AdamW(model.parameters(), lr=5e-5)

## create default learning rate scheduler
num_epochs = 3
num_training_steps = num_epochs * len(imageList1)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

### cpu enabler
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

### TRAINING LOOP

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for index, (img1, img2) in enumerate(zip(imageList1, imageList2)):
        input_tensors1 = image_processor(images=img1,segmentation_maps=masks[index], return_tensors="pt")
        input_tensors2 = image_processor(images=img2,segmentation_maps=masks[index], return_tensors="pt")
        # mask_tensor = image_processor(segmentation_maps=mask[index], return_tensors="pt")
        # batch = {k: v.to(device) for k, v in batch.items()}
        #https://huggingface.co/docs/transformers/training
        #https://huggingface.co/docs/datasets/loading

        input_tensors = {}
        input_tensors['pixel_values'] = torch.from_numpy(np.concatenate((input_tensors1.pixel_values, input_tensors2.pixel_values), axis=1))
        input_tensors['labels'] = input_tensors2.labels;

        # print(input_tensors2.pixel_values.shape)
        # print(input_tensors['pixel_values'].shape)

        # Extract the specific tensors
        # processed_tensor1 = input_tensors1.pixel_values
        # processed_tensor2 = input_tensors2.pixel_values

        # Concatenate the processed tensors along the channel dimension
        # concatenated_tensor = torch.cat((processed_tensor1, processed_tensor2), dim=1)

        outputs = model(**input_tensors)
        loss = outputs.loss
        loss.backward()

        print(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


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