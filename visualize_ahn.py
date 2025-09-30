import rasterio
import numpy as np
import matplotlib.pyplot as plt

## Raster data is 0.5 bij 0.5 en elk vak is 5x6.25 kilometer.

# Pad naar je .tif bestand
tif_path = r"C:\Users\User\OneDrive - University of Twente\M-ROB\Graduation Project\Python Code\Data\AHN and Ortho\hwh-ahn\ahn4\03a_DSM_0.5m\R_34FN1\R_34FN1.TIF"

with rasterio.open(tif_path) as src:
    elevation = src.read(1).astype(float)  # converteer naar float om overflow te voorkomen
    nodata = src.nodata

    # Masker NoData waarden
    if nodata is not None:
        elevation[elevation == nodata] = np.nan

# Bereken min en max zonder NoData waarden
min_height = np.nanmin(elevation)
max_height = np.nanmax(elevation)
print(f"Minimale hoogte: {min_height:.2f} m")
print(f"Maximale hoogte: {max_height:.2f} m")

# Plotten
plt.figure(figsize=(12, 8))
img = plt.imshow(elevation, cmap='terrain')
plt.colorbar(img, label='Hoogte (m)')
plt.title("AHN Hoogtekaart")

plt.xlim(0, 5)  # X-axis range in meters
plt.ylim(0, 5)  # Y-axis range in meters

plt.show()