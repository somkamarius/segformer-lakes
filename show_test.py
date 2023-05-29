import rasterio
import matplotlib.pyplot as plt

image = rasterio.open('dataset/masks/training_demo/0.tif').read()
print(image.shape)
# Get the VV and VH bands
vv_band = image[0]
vh_band = image[1]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Display VV band
ax1.imshow(vv_band, cmap='gray')
ax1.set_title('VV Band')

# Display VH band
ax2.imshow(vh_band, cmap='gray')
ax2.set_title('VH Band')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
