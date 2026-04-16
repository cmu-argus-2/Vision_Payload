Supplementary Datasets for LEO Satellite CV Navigation

Context: EfficientNet model trained on Sentinel-2 and Landsat mosaics via Google Earth Engine. The datasets below are curated for robust out-of-distribution testing, covering sensor diversity, resolution variance, scene complexity, and coastline coverage gaps not represented in the training corpus. All datasets are at GSD ranges compatible with LEO medium-resolution imaging (≥3m).


EuroSAT

Source: DLR / TU Berlin
Access: GitHub | TensorFlow Datasets
Size: 27,000 labeled patches; 10 land-use/land-cover classes
Sensor: Sentinel-2, 13 spectral bands, 10m GSD
Why it's useful: Spectrally compatible with your training data but sourced from different GEE export pipelines and geographic distributions. Tests whether the model generalizes beyond the specific mosaic methodology used in training, particularly for water, vegetation, and transitional land-cover classes.

BigEarthNet

Source: TU Berlin
Access: BigEarthNet portal
Size: 590,326 Sentinel-2 patches (multi-label, 43 CORINE land cover classes)
Sensor: Sentinel-2 (10m) + Sentinel-1 SAR variant available
Why it's useful: Multi-label patches surface ambiguous boundaries (e.g., coastal wetland, estuarine, transitional water-land) that are harder to classify than clean land or sea. The SAR variant is directly relevant if the lab is considering all-weather imaging.

CoastTrain (Possibly not useful given geographical lock)

Source: U.S. Geological Survey (USGS) / Buscombe et al. 2023
Access: ScienceBase | Nature Scientific Data
Size: 1.2 billion labeled pixels across 10 data records; imagery from Sentinel-2, Landsat-8, NAIP aerial, and UAS orthomosaics
Resolution: Sentinel-2 (10m) and Landsat-8 (30m) components are LEO-compatible; aerial components can be excluded
Why it's useful: The most comprehensive publicly available coastal classification dataset. Multi-labeler annotations cover beaches, cliffs, wetlands, estuaries, rocky shores, and open water across Pacific, Atlantic, and Gulf Coast geographies. Directly stress-tests sea-land boundary detection — the most navigation-critical coastal feature.
License: CC BY 4.0

SNOWED (Satellite dataset for Water Edge Detection)

Source: MDPI Remote Sensing / Sturdivant et al.
Access: PMC open access
Size: Automatically constructed from Sentinel-2 scenes; binary sea-land segmentation labels derived from OpenStreetMap water polygons
Sensor: Sentinel-2 (10m GSD), covering European, American, and African coastal regions

Water Segmentation Data Set (QueryPlanet Project)
The water segmentation data set [49] has been created as a part of the QueryPlanet project, which has been funded by the European Space Agency (ESA). The dataset is composed of satellite images of size 64×64 from the Sentinel-2 Level-1C product. Each of them has been manually labeled by volunteer users of a collaborative web app. Volunteers were prompted with an initial label obtained by calculating the NDWI [50] and had to visually compare it with the corresponding satellite TCI and correct eventual discrepancies based on their interpretation of the image. The online labeling campaign led to the creation of 7671 samples, but only 5177 of them contain both sea and land pixels.

NASA Worldview / EOSDIS

Source: NASA EOSDIS
Access: worldview.earthdata.nasa.gov
Coverage: 900+ full-resolution global layers; updates within 3 hours of acquisition
Sensors: MODIS (~250m), VIIRS (~375m), and others
Why it's useful: Enables on-demand sampling of polar coastlines, ice margins, island chains, and storm-affected shorelines not present in your training mosaics. Useful for constructing targeted edge-case test patches: sea ice coastlines at Arctic margins, tidal flat coastlines in Southeast Asia, atoll chains in the Pacific.
License: U.S. federal open data; no restriction

