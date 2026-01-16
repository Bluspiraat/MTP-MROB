# dataset-creation 
This branch focuses on the extraction of trainings data from Publieke Dienstverlening Op de Kaart (PDOK).

## Datatypes
The data downloaded, subsetted and aligned for training consists of samples from the following three sources:
- Basisregistratie Topografie 2022 (BRT) with the following layers: wegdeel, waterdeel, terrein and spoorbaandeel.
- Actueel Hoogtebestand Nederland (2020-2022) with the 0.5m grid resolution upsampled to 0.1m using scipy.ndimage.zoom
- Orthophotography with RGB data of 2022 'bladloos' photography downsampled from 8 cm resolution to 10 cm.