import rasterio
from transformers import ResNetModel
import matplotlib.pyplot as plt
import numpy as np

image = rasterio.open("0_CLOUD_good.tif")

image = image.read([3,2,1])/4095*255
image = (np.rint(image)).astype(int)


# image=image.transpose((1,2,0)) # CHW -> HWC
image = np.clip(image, 0, 255)


image = np.transpose(image, (1,2,0))
plt.imshow(image)
plt.show()