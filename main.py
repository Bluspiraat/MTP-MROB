import geopandas as gpd
import matplotlib.pyplot as plt

# Load all layers
gdf_terrein = gpd.read_file("top10nl_terrein.gml")
gdf_wegdeel = gpd.read_file("top10nl_wegdeel.gml")
gdf_waterdeel = gpd.read_file("top10nl_waterdeel.gml")
gdf_spoorbaandeel = gpd.read_file("top10nl_spoorbaandeel.gml")

# Put them in a list with titles and the column you want to color by
layers = [
    (gdf_terrein, "Terrein", "typeLandgebruik"),
    (gdf_wegdeel, "Wegdeel", "typeInfrastructuur"),
    (gdf_waterdeel, "Waterdeel", "typeWater"),
    (gdf_spoorbaandeel, "Spoorbaandeel", "typeSpoorbaan")
]

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 12x12 per plot
axes = axes.flatten()  # flatten for easy indexing

for ax, (gdf, title, col) in zip(axes, layers):
    gdf.plot(column=col, legend=True, ax=ax, edgecolor="black")
    ax.set_title(title, fontsize=16)
    ax.axis("off")  # optional: turn off axis

plt.tight_layout()
plt.show()