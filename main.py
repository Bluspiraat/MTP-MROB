from visualize_brt import create_BRT_export
from visualize_orthophoto import get_image

if __name__ == '__main__':
    minx = 251000  # 251000
    miny = 471000  # 471000
    maxx = 251999  # 251999
    maxy = 471999  # 471999

    # Get orthophoto
    output_name = "orthophoto"
    orthophoto_folder = "Data/AHN and Ortho/hwh-ortho/2025"
    get_image(orthophoto_folder, output_name, minx, miny, maxx, maxy)

    # Get BRT labelled image
    output_name = "BRT_image_5cm"
    gml_files_location = ["Data/BRT/top10nl_terrein.gml", "Data/BRT/top10nl_waterdeel.gml",
                          "Data/BRT/top10nl_spoorbaandeel.gml", "Data/BRT/top10nl_wegdeel.gml"]
    resolution = 0.05
    create_BRT_export(gml_files_location, resolution, output_name, minx, miny, maxx, maxy)
