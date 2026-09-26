/**
 * Module: Feature Extraction
 * Description: Modular functions for single random split and stratified random split 
 *              sampling of GEE image features for classification.
 */

/**
 * 1. Single Random Split
 * Randomly partitions input regions of interest into training and testing sets, 
 * then extracts pixel values from the feature stack image
 *
 * @param {ee.Image} image - The input image containing feature bands/covariates.
 * @param {ee.FeatureCollection} roi - The region of interest or ground truth features.
 * @param {string} classProp - Property name containing unique class labels.
 * @param {number} [splitRatio=0.6] - Ratio of samples for training (e.g. 0.6 = 60% train, 40% test).
 * @param {number} [pixelSize=30] - Spatial resolution scale in meters for sampleRegions.
 * @param {number} [tileScale=16] - Tile scale factor for handling large-scale extractions.
 * @param {number} [seed=0] - Random seed for reproducibility.
 * @returns {Object} { trainingPixels: ee.FeatureCollection, testingPixels: ee.FeatureCollection }
 */
exports.randomSplit = function(image, roi, classProp, splitRatio, pixelSize, tileScale, seed) {
  splitRatio = splitRatio !== undefined ? splitRatio : 0.6;
  pixelSize = pixelSize !== undefined ? pixelSize : 30;
  tileScale = tileScale !== undefined ? tileScale : 16;
  seed = seed !== undefined ? seed : 0;

  //Add random column
  var roiRandom = roi.randomColumn('random', seed);

  //Partition training and testing sets
  var training = roiRandom.filter(ee.Filter.lt('random', splitRatio));
  var testing = roiRandom.filter(ee.Filter.gte('random', splitRatio));

  //Extract pixel values
  //Training pixels 
  var trainingPixels = image.sampleRegions({
    collection: training,
    properties: [classProp],
    scale: pixelSize,
    tileScale: tileScale
  });
  //testing pixels
  var testingPixels = image.sampleRegions({
    collection: testing,
    properties: [classProp],
    scale: pixelSize,
    tileScale: tileScale
  });

  return {
    trainingPixels: trainingPixels,
    testingPixels: testingPixels
  };
};

/**
 * 2. Stratified Random Split
 * Performs stratified sampling partitioned by class labels to preserve sample distribution, 
 * then extracts pixel values from the image.
 *
 * @param {ee.FeatureCollection} roi - The region of interest containing class labels.
 * @param {ee.Image} image - The input image containing feature bands/covariates.
 * @param {string} classProp - Property name containing class labels.
 * @param {number} [pixelSize=30] - Spatial resolution scale in meters.
 * @param {number} [trainRatio=0.7] - Proportion of data assigned to training.
 * @param {number} [seed=0] - Random seed for reproducibility.
 * @param {number} [tileScale=16] - Tile scale factor for computation management.
 * @returns {Object} { trainingPixels: ee.FeatureCollection, testingPixels: ee.FeatureCollection }
 */
exports.stratifiedSplit = function(roi, image, classProp, pixelSize, trainRatio, seed, tileScale) {
  pixelSize = pixelSize !== undefined ? pixelSize : 30;
  trainRatio = trainRatio !== undefined ? trainRatio : 0.7;
  seed = seed !== undefined ? seed : 0;
  tileScale = tileScale !== undefined ? tileScale : 16;

  //Get unique class IDs present in the data
  var classes = roi.aggregate_array(classProp).distinct();

  //Function to perform stratified random split per class
  var splitClass = function(c) {
    var subset = roi.filter(ee.Filter.eq(classProp, c))
                    .randomColumn('random', seed);
    
    var train = subset.filter(ee.Filter.lt('random', trainRatio))
                      .map(function(f) { return f.set('fraction', 'training'); });
    var test = subset.filter(ee.Filter.gte('random', trainRatio))
                     .map(function(f) { return f.set('fraction', 'testing'); });
    
    return train.merge(test);
  };

  // Map function across all class IDs and flatten
  var splitFC = ee.FeatureCollection(classes.map(splitClass)).flatten();

  //partition training and testing feature collections
  var trainFC = splitFC.filter(ee.Filter.eq('fraction', 'training'));
  var testFC = splitFC.filter(ee.Filter.eq('fraction', 'testing'));

  //Sample pixel values from image
  var trainPix = image.sampleRegions({
    collection: trainFC,
    properties: [classProp],
    scale: pixelSize,
    tileScale: tileScale
  });

  var testPix = image.sampleRegions({
    collection: testFC,
    properties: [classProp],
    scale: pixelSize,
    tileScale: tileScale
  });

  return {
    trainingPixels: trainPix,
    testingPixels: testPix
  };
};