# Step "1": Filter Sentinel-2 data for the specified area of interest (AOI), time range, and cloud coverage
filtered_image1 = sent2.filterBounds(aoi) \
.filterDate('2023-01-01', '2023-06-30') \
.filterMetadata('CLOUD_COVERAGE_ASSESSMENT', 'less_than', 1); 

# Step "2": Function to mask clouds and shadows using the SWIR1 and SWIR2 bands
def maskClouds(image):
    # Select SWIR1 and SWIR2 bands
    swir1 = image.select('B11')
    swir2 = image.select('B12')

    # Define threshold values to identify clouds based on SWIR reflectance
    swir1_threshold = 0.15; 
    swir2_threshold = 0.1;  

    # Create a mask to detect cloud pixels by comparing reflectance values to the thresholds
    cloudMask = swir1.lt(swir1_threshold).And(swir2.lt(swir2_threshold))

    # Return the masked image, excluding clouds, while keeping original properties
    return image.updateMask(cloudMask.Not()).copyProperties(image, [""system":time_start"])

# Apply cloud mask to the filtered images
filtered_image2 = filtered_image1.map(maskClouds); 

# Step "3": Compute the median of the cloud-masked images to reduce noise and improve quality
filtered_image = filtered_image2.median(); 

# Step "4": Clip the image to the study area (shape)
clippedimage = filtered_image.clip(shape); 

# Step "5": Segmentation - Create seeds for the segmentation process
seeds = ee.Algorithms.Image.Segmentation.seedGrid(20); 

# Apply the SNIC (Simple Non-Iterative Clustering) algorithm for image segmentation
segmentation = ee.Algorithms.Image.Segmentation.SNIC(
"image"=clippedimage, 
"size"=10, 
"compactness"=0, 
"connectivity"=8, 
"neighborhoodSize"=500, 
"seeds"=seeds 
).select(['clusters', 'B2_mean', 'B3_mean', 'B4_mean', 'B8_mean'],
['clusters', 'B2_mean', 'B3_mean', 'B4_mean', 'B8_mean']); 

# Select important bands and clusters from the segmentation result
segments = segmentation.select(['clusters', 'B2_mean', 'B3_mean', 'B4_mean', 'B8_mean'])

# Step "6": Prepare training data by merging multiple land cover classes
training_points = informal_settlements01.merge(formal_settlements01) \
.merge(vegetation01).merge(bare_land01) \
.merge(water01).merge(vegvalidationset) \
.merge(blvalidationset).merge(watvalidationset) \
.merge(infvalidationset).merge(forvalidationset); 

# Sample the segmented image using the training points to create training data
trainingdata1 = segments.sampleRegions(
"collection"=training_points, 
"properties"=['classes'], 
"tileScale"=16, 
"scale"=10 
).randomColumn(**{"distribution": 'uniform'); 

# Split data into training (80%) and validation (20%) sets
trainingdata = trainingdata1.filter('random <= 0.8'); 
validationdata = trainingdata1.filter('random > 0.2'); 

# Step "7": Train a Random Forest classifier with 50 trees
trainedClassifierorf = ee.Classifier.smileRandomForest(50).train(
"features"=trainingdata, 
"classProperty"='classes', 
"inputProperties"=['clusters', 'B2_mean', 'B3_mean', 'B4_mean', 'B8_mean'] 
)

# Classify the segments using the trained Random Forest model
classifiedorf = segments.classify(trainedClassifierorf); 

# Step "8": Export the classified image to Google Drive
Export.image.toDrive(
"image"=classifiedorf, 
"description"='objectclassifiednew', 
"folder"='GEE_data', 
"fileNamePrefix"='objectclassifiednew', 
"region"=aoi, 
"scale"=10, 
"maxPixels"=80000000, 
"shardSize"=100, 
"fileFormat"='GeoTIFF' 
)

# Step "9": Visualize the classified image on the map
Map.addLayer(classifiedorf, {"min": 1, "max": 5, "palette": ['#066c19','#e21313','#c58021','#321fff','#43e62a']}, 'objectclassifiedorf'); # Add classified layer to the map

# Print classified image band names for debugging
print('classsifiedorf bandnames', classifiedorf.bandNames().getInfo())

# Step "10": Assess the accuracy of the model using training data (resubstitution accuracy)
trainAccuracyorf = trainedClassifierorf.confusionMatrix(); 
print('orf Resubstitution error "matrix": ', trainAccuracyorf.getInfo())
print('orf Training overall "accuracy": ', trainAccuracyorf.accuracy().getInfo())

# Classify validation data and compute accuracy for validation set
validatedorf = validationdata.classify(trainedClassifierorf, 'classification')
validationAccuracyorf = validatedorf.errorMatrix('classification', 'classes'); 
print('Validation orf error "matrix": ', validationAccuracyorf.getInfo())
print('Validation orf overall "accuracy": ', validationAccuracyorf.accuracy().getInfo())

# Step "11": Create features for exporting confusion matrices and accuracies
confusionMatrixToExportorft = ee.Feature(None, {
    "matrix": trainAccuracyorf.array(),
    "accuracy": trainAccuracyorf.accuracy()
})

confusionMatrixToExportorfv = ee.Feature(None, {
    "matrix": validationAccuracyorf.array(),
    "accuracy": validationAccuracyorf.accuracy()
})

# Step "12": Pixel-Based Classification for comparison
training_points12 = informal_settlements01.merge(formal_settlements01) \
.merge(vegetation01).merge(bare_land01) \
.merge(water01).merge(vegvalidationset) \
.merge(blvalidationset).merge(watvalidationset) \
.merge(infvalidationset).merge(forvalidationset); 

readyimage = clippedimage.select(['B2','B3','B4','B8']); 

# Sample the image using training points
trainingdata1 = readyimage.sampleRegions(
"collection"=training_points12,
"properties"=['classes'], 
"scale"=10
).randomColumn(**{"distribution": 'uniform'); 

trainingSampleprf = trainingdata1.filter('random <= 0.8'); 
validationSampleprf = trainingdata1.filter('random > 0.2'); 

# Train Random Forest classifier for pixel-based classification
trainedClassifierprf = ee.Classifier.smileRandomForest(50).train(
"features"=trainingSampleprf,
"classProperty"='classes', 
"inputProperties"=['B2','B3','B4','B8'] 
)

# Classify the image using the trained Random Forest classifier
classifiedprf = readyimage.classify(trainedClassifierprf); 

# Add the pixel-based classified image to the map for visualization
Map.addLayer(classifiedprf, {"min": 1, "max": 5, "palette": ['#066c19','#e21313','#c58021','#321fff','#43e62a']}, 'pixel-based rf Classified Image')

# Step "13": Evaluate training accuracy for pixel-based classification
trainAccuracyprf = trainedClassifierprf.confusionMatrix(); 
print('Resubstitution error "matrix": ', trainAccuracyprf.getInfo())
print('Training overall "accuracy": ', trainAccuracyprf.accuracy().getInfo())