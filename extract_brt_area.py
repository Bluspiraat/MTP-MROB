import geopandas as gpd

bboxes = [(110000, 416000, 117000, 423000),
          (103000, 520000, 110000, 527000),
          (149000, 455000, 156000, 462000),
          (182000, 478000, 189000, 485000)]
area_names = ['bies_bosch', 'schoorl', 'soesterberg', 'vierhouten']

gml_file = "C:/MTP-Data/top10nl_wegdeel.gml"
for bbox, area_name in zip(bboxes, area_names):
    gdf_new = gpd.read_file(gml_file, bbox=bbox)
    gdf_new.set_crs("EPSG:28992")
    gdf_new.to_file(f"top10nl_wegdeel_{area_name}.gml", driver="GML")
