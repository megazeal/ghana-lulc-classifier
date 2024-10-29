# Import administrative boundaries of Ghana from Earth Engine Data Catalog
ghanaBoundaries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
.filter(ee.Filter.eq('country_na', 'Ghana'))

Map.centerObject(ghanaBoundaries, 6)

# Define training points for different years
# Each year contains merged FeatureCollections for different land cover types
trainingPoints = ee.Dictionary(**{
'2023': dense_vegetation23.merge(settlements23).merge(bareland23).merge(water23).merge(light_vegetation23),
'2022': dense_vegetation22.merge(settlements22).merge(bareland22).merge(water22).merge(light_vegetation22)
})

#*
# Retrieves training points for a specific year.
# @param {string} year - The year for which to get training points.
# @return {ee.FeatureCollection} The training points for the specified year.
#
def getTrainingPoints(year):
    return ee.FeatureCollection(trainingPoints.get(year))

#*
# Processes satellite imagery based on specified parameters.
# @param {number} year - The year of interest.
# @param {string} imageType - The type of satellite imagery ('sentinel2' or 'landsat8').
# @return {Object} An object containing the processed image and training points.
#
def processImage(year, imageType):
    regionOfInterest = ghanaBoundaries
    trainingPointsForYear = getTrainingPoints(year.toString())
    startDate = ee.Date.fromYMD(year, 1, 1)
    endDate = ee.Date.fromYMD(year, 12, 31)

    # Helper function to process Sentinel-2 imagery
    def processS2():
        image_sentinel_2A = ee.ImageCollection("COPERNICUS/S2_SR") \
        .filterBounds(regionOfInterest) \
        .filterDate(startDate, endDate) \
        .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 10) \
        .sort('CLOUDY_PIXEL_PERCENTAGE') \
        .median() \
        .clip(regionOfInterest)

        visualization = { "gain": '0.1, 0.1, 0.1', "scale": 7 }
        Map.addLayer(image_sentinel_2A.select("B4", "B3", "B2"), visualization, 'True_colour_sentinel_2A')

        return { "processedImage": image_sentinel_2A, "trainingPoints": trainingPointsForYear }

    # Helper function to process Landsat 8 imagery
    def processL8():
        # Prepare Landsat 8 SR image
        def prepSrL8(image):
            qaMask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
            saturationMask = image.select('QA_RADSAT').eq(0)

            def func_ljx (factorNames):
                factorList = image.toDictionary().select(factorNames).values()
                return ee.Image.constant(factorList)

            getFactorImg = func_ljx
            
            scaleImg = getFactorImg(['REFLECTANCE_MULT_BAND_.|TEMPERATURE_MULT_BAND_ST_B10'])
            offsetImg = getFactorImg(['REFLECTANCE_ADD_BAND_.|TEMPERATURE_ADD_BAND_ST_B10'])
            scaled = image.select('SR_B.|ST_B10').multiply(scaleImg).add(offsetImg)

            return image.addBands(scaled, None, True) \
            .updateMask(qaMask).updateMask(saturationMask)

        image_landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterDate(startDate, endDate) \
        .filterBounds(regionOfInterest) \
        .map(prepSrL8) \
        .median() \
        .clip(regionOfInterest)

        image_landsat8 = image_landsat8.select(['SR_B4', 'SR_B3', 'SR_B2'], ['B4', 'B3', 'B2'])
        Map.addLayer(image_landsat8, { "bands": ['B4', 'B3', 'B2'], "min": 0, "max": 0.25 }, 'image')

        return { "processedImage": image_landsat8, "trainingPoints": trainingPointsForYear }

    # Choose processing method based on imageType
if (imageType == 'sentinel2'):
        return processS2()
if (imageType == 'landsat8'):
        return processL8()
     else {
        throw new Error('Invalid image type. Use "sentinel2" or "landsat8".')
    }

#*
# Performs Land Use and Land Cover (LULC) classification on the input image.
# @param {ee.Image} inputImage - The image to classify.
# @param {ee.FeatureCollection} trainingPoints - The training data.
# @param {string} classificationAlgorithm - The classification algorithm to use.
# @return {Object} An object containing the classified image and accuracy metrics.
#
def performClassification(inputImage, trainingPoints, classificationAlgorithm):
    regionOfInterest = ghanaBoundaries
    bands = ['B4', 'B3', 'B2']

    # Sample regions for training data
    trainingData = inputImage.sampleRegions(
    "collection"=trainingPoints,
    "properties"=['lulc'],
    "tileScale"=16,
    "scale"=10
    ).randomColumn(**{ "distribution": 'uniform' )

    # Split data into training and testing sets
    split = 0.7
    training = trainingData.filter(ee.Filter.lt('random', split))
    testing = trainingData.filter(ee.Filter.gte('random', split))

    # Define classifiers
    classifierParams = {
        'CART': ee.Classifier.smileCart(),
        'RandomForest': ee.Classifier.smileRandomForest(50),
        'NaiveBayes': ee.Classifier.smileNaiveBayes(),
        'SVM': ee.Classifier.libsvm(**{"kernelType": 'RBF', "gamma": 0.5, "cost": 10})
    }

    # Train the chosen classifier
    classifier = classifierParams[classificationAlgorithm] \
    .train(
    "features"=training,
    "classProperty"='lulc',
    "inputProperties"=bands
    )

    # Classify the image
    classifiedImage = inputImage.classify(classifier)

    # Visualization
    palette = ['#066c19', '#e21313', '#c58021', '#321fff', '#43e62a']
    Map.addLayer(classifiedImage, { "min": 1, "max": 5, "palette": palette }, 'Land Cover Classification')

    # Accuracy assessment
    testAccuracy = testing \
    .classify(classifier) \
    .errorMatrix('lulc', 'classification')

    accuracyMetrics = {
        "overallAccuracy": testAccuracy.accuracy(),
        "KappaStatistic": testAccuracy.kappa(),
        "producerAccuracy": testAccuracy.producersAccuracy()
    }

    return {
        "classifiedImage": classifiedImage,
        "accuracyMetrics": accuracyMetrics
    }

#*
# Clips the classified image to a specific region and calculates land cover percentages.
# @param {string} regionName - The name of the region to clip to.
# @param {ee.Image} image - The classified image to clip.
# @return {Object} An object containing the clipped image and land cover percentages.
#
def clipImageByRegion(regionName, image):
    selectedRegion = regions.filter(ee.Filter.eq('NAME_1', regionName)).geometry()
    clipped = image.clip(selectedRegion)

    # Calculate area histogram
    histogram = clipped.reduceRegion(
    "reducer"=ee.Reducer.frequencyHistogram(),
    "geometry"=selectedRegion,
    "scale"=10,
    "maxPixels"=1e9,
    "tileScale"=16
    )

    # Calculate percentages for each land cover class
    histogram_dict = ee.Dictionary(histogram.get('classification'))
    frequencies = histogram_dict.values()
    total_area = ee.Array(frequencies).reduce(ee.Reducer.sum(), [0]).get([0])
    class_areas = ee.Array(frequencies).divide(total_area)
    class_percentages = class_areas.multiply(100)

    classLabels = ['dense_vegetation','settlements','bareland','water','light_vegetation']
    classification_percent = ee.Dictionary.fromLists(classLabels, class_percentages.toList())

    return {
        "clipped": clipped,
        "percentage": classification_percent
    }

# Example usage
processedImageResult
try {
    processedImageResult = processImage(2023, 'sentinel2')
} catch (error) {
    print('Error in "processImage":', error.message.getInfo())
}

# Perform classification if image processing was successful
if (processedImageResult && processedImageResult.processedImage && processedImageResult.trainingPoints):
    ClassifiedImage = performClassification(
    processedImageResult.processedImage,
    processedImageResult.trainingPoints,
    'RandomForest'
    )

    # Print accuracy metrics
    print('Accuracy "Metrics":', ClassifiedImage.accuracyMetrics.getInfo())

    # Clip the classified image to a specific region
    Clippedclassified = clipImageByRegion('Bono East', ClassifiedImage.classifiedImage)

    # Visualize the clipped classification
    Map.addLayer(Clippedclassified.clipped, {"min": 1, "max": 5, "palette": ['#066c19','#e21313','#c58021','#321fff','#43e62a']}, 'Clipped Image')
    print('Land Cover "Percentages":', Clippedclassified.percentage.getInfo())
 else {
    print('Classification could not be performed due to an error in image processing.'.getInfo())
}
