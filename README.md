# Ghana Land Use and Land Cover (LULC) Classification Using Earth Engine

## Project Overview

This project uses Google Earth Engine to perform Land Use and Land Cover (LULC) classification for Ghana. It processes Sentinel-2 or Landsat 8 imagery and applies machine learning algorithms to classify the land into different categories such as dense vegetation, settlements, bareland, water, and light vegetation.

## Features

- Processes both Sentinel-2 and Landsat 8 imagery
- Implements multiple classification algorithms (CART, Random Forest, Naive Bayes, SVM)
- Performs accuracy assessment of the classification
- Allows for regional analysis by clipping the classification to specific areas
- Calculates land cover percentages for clipped regions

## Requirements

- Google Earth Engine account
- Basic knowledge of JavaScript and Earth Engine API

## Usage

1. Open the script in the Google Earth Engine Code Editor.
2. Set the desired parameters:
   - Year of analysis
   - Image type ('sentinel2' or 'landsat8')
   - Classification algorithm
   - Region for clipping (if needed)
3. Run the script to perform the classification and view the results.

## Sample Result

<img src=ghana_classified.png>

## Code Structure

- `ghanaBoundaries`: Defines the area of interest (Ghana)
- `trainingPoints`: Contains training data for different land cover types
- `processImage()`: Processes satellite imagery based on specified parameters
- `performClassification()`: Applies the chosen classification algorithm
- `clipImageByRegion()`: Clips the classified image to a specific region and calculates land cover percentages

## Customization

You can customize this script by:
- Adding more training data
- Implementing additional classification algorithms
- Modifying the visualization parameters
- Expanding the analysis to other regions or countries

## Contributing

Contributions to improve the script or extend its functionality are welcome. Please feel free to fork the repository and submit pull requests.

## Contact

[courage]
[couragezeal544@gmail.com]

---
