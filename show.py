import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Create subplots with 1 row and 3 columns
image1 = np.transpose(np.rint(rasterio.open('dataset/combine_cache/26_clouds.png').read()).astype(int), (1,2,0))
image2 = np.transpose(np.rint(rasterio.open('dataset/combine_cache/26_sar.png').read()).astype(int), (1,2,0))
image3 = np.transpose(np.rint(rasterio.open('dataset/combine_cache/26_nocloud.tif').read() / 7500 * 255).astype(int)[1:4], (1,2,0))
image3 = image3[..., ::-1]
image4 = np.transpose(np.rint(rasterio.open('dataset/combine_cache/inputpred.png').read()).astype(int), (1,2,0))

fig = plt.figure(figsize=(10, 8))

axes = fig.subplots(2, 2)


# Display the first image in the first subplot
axes[0, 0].imshow(image1)
axes[0, 0].set_title('Įeitis - nuotrauka su debesimis')

# Display the second image in the second subplot
axes[0, 1].imshow(image2)
axes[0, 1].set_title('Įeitis - VV+VH nuotrauka')

# Display the third image in the third subplot
axes[1, 0].imshow(image3)
axes[1, 0].set_title('Įeitis - nuotrauka be debesų')

axes[1, 1].imshow(image4)
axes[1, 1].set_title('Išeitis')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
