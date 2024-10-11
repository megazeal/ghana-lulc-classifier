// Import administrative boundaries of Ghana from Earth Engine Data Catalog
var ghanaBoundaries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
  .filter(ee.Filter.eq('country_na', 'Ghana'));
  
Map.centerObject(ghanaBoundaries, 6);

// Define training points for different years
// Each year contains merged FeatureCollections for different land cover types
var trainingPoints = ee.Dictionary({
  '2023': dense_vegetation23.merge(settlements23).merge(bareland23).merge(water23).merge(light_vegetation23),
  '2022': dense_vegetation22.merge(settlements22).merge(bareland22).merge(water22).merge(light_vegetation22)
});

/**
 * Retrieves training points for a specific year.
 * @param {string} year - The year for which to get training points.
 * @return {ee.FeatureCollection} The training points for the specified year.
 */
function getTrainingPoints(year) {
  return ee.FeatureCollection(trainingPoints.get(year));
}

/**
 * Processes satellite imagery based on specified parameters.
 * @param {number} year - The year of interest.
 * @param {string} imageType - The type of satellite imagery ('sentinel2' or 'landsat8').
 * @return {Object} An object containing the processed image and training points.
 */
function processImage(year, imageType) {
  var regionOfInterest = ghanaBoundaries;
  var trainingPointsForYear = getTrainingPoints(year.toString());
  var startDate = ee.Date.fromYMD(year, 1, 1);
  var endDate = ee.Date.fromYMD(year, 12, 31);

  // Helper function to process Sentinel-2 imagery
  function processS2() {
    var image_sentinel_2A = ee.ImageCollection("COPERNICUS/S2_SR")
      .filterBounds(regionOfInterest)
      .filterDate(startDate, endDate)
      .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 10)
      .sort('CLOUDY_PIXEL_PERCENTAGE')
      .median()
      .clip(regionOfInterest);

    var visualization = { gain: '0.1, 0.1, 0.1', scale: 7 };
    Map.addLayer(image_sentinel_2A.select("B4", "B3", "B2"), visualization, 'True_colour_sentinel_2A');
    
    return { processedImage: image_sentinel_2A, trainingPoints: trainingPointsForYear };
  }

  // Helper function to process Landsat 8 imagery
  function processL8() {
    // Prepare Landsat 8 SR image
    function prepSrL8(image) {
      var qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
      var saturationMask = image.select('QA_RADSAT').eq(0);
      var getFactorImg = function (factorNames) {
        var factorList = image.toDictionary().select(factorNames).values();
        return ee.Image.constant(factorList);
      };
      var scaleImg = getFactorImg(['REFLECTANCE_MULT_BAND_.|TEMPERATURE_MULT_BAND_ST_B10']);
      var offsetImg = getFactorImg(['REFLECTANCE_ADD_BAND_.|TEMPERATURE_ADD_BAND_ST_B10']);
      var scaled = image.select('SR_B.|ST_B10').multiply(scaleImg).add(offsetImg);

      return image.addBands(scaled, null, true)
        .updateMask(qaMask).updateMask(saturationMask);
    }

    var image_landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
      .filterDate(startDate, endDate)
      .filterBounds(regionOfInterest)
      .map(prepSrL8)
      .median()
      .clip(regionOfInterest);
      
    image_landsat8 = image_landsat8.select(['SR_B4', 'SR_B3', 'SR_B2'], ['B4', 'B3', 'B2']);
    Map.addLayer(image_landsat8, { bands: ['B4', 'B3', 'B2'], min: 0, max: 0.25 }, 'image');
    
    return { processedImage: image_landsat8, trainingPoints: trainingPointsForYear };
  }

  // Choose processing method based on imageType
  if (imageType === 'sentinel2') {
    return processS2();
  } else if (imageType === 'landsat8') {
    return processL8();
  } else {
    throw new Error('Invalid image type. Use "sentinel2" or "landsat8".');
  }
}

/**
 * Performs Land Use and Land Cover (LULC) classification on the input image.
 * @param {ee.Image} inputImage - The image to classify.
 * @param {ee.FeatureCollection} trainingPoints - The training data.
 * @param {string} classificationAlgorithm - The classification algorithm to use.
 * @return {Object} An object containing the classified image and accuracy metrics.
 */
function performClassification(inputImage, trainingPoints, classificationAlgorithm) {
  var regionOfInterest = ghanaBoundaries;
  var bands = ['B4', 'B3', 'B2'];

  // Sample regions for training data
  var trainingData = inputImage.sampleRegions({
    collection: trainingPoints,
    properties: ['lulc'],
    tileScale: 16,
    scale: 10
  }).randomColumn({ distribution: 'uniform' });

  // Split data into training and testing sets
  var split = 0.7;
  var training = trainingData.filter(ee.Filter.lt('random', split));
  var testing = trainingData.filter(ee.Filter.gte('random', split));

  // Define classifiers
  var classifierParams = {
    'CART': ee.Classifier.smileCart(),
    'RandomForest': ee.Classifier.smileRandomForest(50),
    'NaiveBayes': ee.Classifier.smileNaiveBayes(),
    'SVM': ee.Classifier.libsvm({kernelType: 'RBF', gamma: 0.5, cost: 10})
  };

  // Train the chosen classifier
  var classifier = classifierParams[classificationAlgorithm]
    .train({
      features: training,
      classProperty: 'lulc',
      inputProperties: bands
    });

  // Classify the image
  var classifiedImage = inputImage.classify(classifier);

  // Visualization
  var palette = ['#066c19', '#e21313', '#c58021', '#321fff', '#43e62a'];
  Map.addLayer(classifiedImage, { min: 1, max: 5, palette: palette }, 'Land Cover Classification');

  // Accuracy assessment
  var testAccuracy = testing
    .classify(classifier)
    .errorMatrix('lulc', 'classification');

  var accuracyMetrics = {
    overallAccuracy: testAccuracy.accuracy(),
    KappaStatistic: testAccuracy.kappa(),
    producerAccuracy: testAccuracy.producersAccuracy()
  };

  return {
    classifiedImage: classifiedImage,
    accuracyMetrics: accuracyMetrics
  };
}

/**
 * Clips the classified image to a specific region and calculates land cover percentages.
 * @param {string} regionName - The name of the region to clip to.
 * @param {ee.Image} image - The classified image to clip.
 * @return {Object} An object containing the clipped image and land cover percentages.
 */
function clipImageByRegion(regionName, image) {
  var selectedRegion = regions.filter(ee.Filter.eq('NAME_1', regionName)).geometry();
  var clipped = image.clip(selectedRegion);
  
  // Calculate area histogram
  var histogram = clipped.reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: selectedRegion,
    scale: 10,
    maxPixels: 1e9,
    tileScale: 16
  });

  // Calculate percentages for each land cover class
  var histogram_dict = ee.Dictionary(histogram.get('classification'));
  var frequencies = histogram_dict.values();
  var total_area = ee.Array(frequencies).reduce(ee.Reducer.sum(), [0]).get([0]);
  var class_areas = ee.Array(frequencies).divide(total_area);
  var class_percentages = class_areas.multiply(100);

  var classLabels = ['dense_vegetation','settlements','bareland','water','light_vegetation'];
  var classification_percent = ee.Dictionary.fromLists(classLabels, class_percentages.toList());
  
  return {
    clipped: clipped,
    percentage: classification_percent
  };
}

// Example usage
var processedImageResult;
try {
  processedImageResult = processImage(2023, 'sentinel2');
} catch (error) {
  print('Error in processImage:', error.message);
}

// Perform classification if image processing was successful
if (processedImageResult && processedImageResult.processedImage && processedImageResult.trainingPoints) {
  var ClassifiedImage = performClassification(
    processedImageResult.processedImage,
    processedImageResult.trainingPoints,
    'RandomForest'
  );

  // Print accuracy metrics
  print('Accuracy Metrics:', ClassifiedImage.accuracyMetrics);

  // Clip the classified image to a specific region
  var Clippedclassified = clipImageByRegion('Bono East', ClassifiedImage.classifiedImage);

  // Visualize the clipped classification
  Map.addLayer(Clippedclassified.clipped, {min: 1, max: 5, palette: ['#066c19','#e21313','#c58021','#321fff','#43e62a']}, 'Clipped Image');
  print('Land Cover Percentages:', Clippedclassified.percentage);
} else {
  print('Classification could not be performed due to an error in image processing.');
}
