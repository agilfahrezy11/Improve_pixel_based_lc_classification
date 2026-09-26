/**
 * Module: Classification & Model Evaluation
 * Description: Implements Hard Classification, One-vs-Rest (OVR) Soft Probability 
 *              Classification, Multi-Probability Classification (for Bayesian Smoothing), 
 *              Feature Importance extraction, and Accuracy Evaluation.
 */

/**
 * Helper to build Random Forest options dictionary for GEE constructor.
 * Omits variablesPerSplit if not provided so GEE defaults to sqrt(#bands).
 */
function buildRfParams(nTrees, vSplit, minLeaf, seed) {
  var params = {
    numberOfTrees: nTrees !== undefined ? nTrees : 100,
    minLeafPopulation: minLeaf !== undefined ? minLeaf : 1,
    seed: seed !== undefined ? seed : 0
  };
  if (vSplit !== undefined && vSplit !== null && vSplit > 0) {
    params.variablesPerSplit = vSplit;
  }
  return params;
}

/**
 * 1. Hard Classification
 * Performs standard multi-class Random Forest classification.
 */
exports.hardClassification = function(trainingPixels, classProperty, image, nTrees, vSplit, minLeaf, seed) {
  var bandNames = image.bandNames();
  var rfParams = buildRfParams(nTrees, vSplit, minLeaf, seed);

  var classifier = ee.Classifier.smileRandomForest(rfParams);
  var trainedModel = classifier.train({
    features: trainingPixels,
    classProperty: classProperty,
    inputProperties: bandNames
  });

  var classificationMap = image.classify(trainedModel).rename('classification');

  return {
    classificationMap: classificationMap,
    trainedModel: trainedModel
  };
};

/**
 * 2. Soft Classification (One-vs-Rest Binary Probability Stack & Argmax)
 * Trains individual binary Random Forest models in PROBABILITY mode for each class.
 */
exports.softClassification = function(
  trainingPixels,
  classProperty,
  image,
  includeFinalMap,
  nTrees,
  vSplit,
  minLeaf,
  probabilityScale,
  seed
) {
  includeFinalMap = includeFinalMap !== undefined ? includeFinalMap : true;
  probabilityScale = probabilityScale !== undefined ? probabilityScale : 100;

  var bandNames = image.bandNames();
  var rfParams = buildRfParams(nTrees, vSplit, minLeaf, seed);
  var classList = trainingPixels.aggregate_array(classProperty).distinct();

  var probImages = classList.map(function(classId) {
    classId = ee.Number(classId);

    var binaryTrain = trainingPixels.map(function(ft) {
      var isTarget = ee.Number(ft.get(classProperty)).eq(classId);
      return ft.set('binary', ee.Algorithms.If(isTarget, 1, 0));
    });

    var classifier = ee.Classifier.smileRandomForest(rfParams).setOutputMode('PROBABILITY');

    var trained = classifier.train({
      features: binaryTrain,
      classProperty: 'binary',
      inputProperties: bandNames
    });

    var probImg = image.classify(trained)
      .multiply(probabilityScale)
      .round()
      .toByte();

    var bandName = ee.String('prob_').cat(classId.int().format());
    return probImg.rename(bandName);
  });

  var probStack = ee.ImageCollection(probImages).toBands();
  
  // Clean up band names (e.g., "0_prob_1" -> "prob_1")
  var originalBandNames = probStack.bandNames();
  var cleanBandNames = originalBandNames.map(function(name) {
    var str = ee.String(name);
    var split = str.split('_');
    return ee.String('prob_').cat(split.get(2));
  });
  probStack = probStack.select(originalBandNames, cleanBandNames);

  if (!includeFinalMap) {
    return probStack;
  }

  var arrayImg = probStack.toArray();
  var maxProbIndex = arrayImg.arrayArgmax().arrayGet([0]);

  var classSequence = ee.List.sequence(0, classList.size().subtract(1));
  var finalMap = maxProbIndex.remap(classSequence, classList).rename('classification');

  var maxConfidence = arrayImg.arrayReduce(ee.Reducer.max(), [0])
    .arrayGet([0])
    .rename('confidence');

  return probStack.addBands([finalMap, maxConfidence]);
};

/**
 * 3. Multi-Probability Classification (Native RF Array Probabilities)
 * Returns a multi-band probability image [0,1] formatted for Bayesian smoothing.
 */
exports.multiProbabilityClassification = function(trainingPixels, classProperty, classIdToNameMap, image, options) {
  options = options || {};
  var rfParams = buildRfParams(options.nTrees, options.vSplit, options.minLeaf, options.seed);
  var bandNames = image.bandNames();

  var classifier = ee.Classifier.smileRandomForest(rfParams).setOutputMode('MULTIPROBABILITY');

  var trainedModel = classifier.train({
    features: trainingPixels,
    classProperty: classProperty,
    inputProperties: bandNames
  });

  var probsArrayImage = image.classify(trainedModel);

  // Parse class IDs in ascending numerical order (matching GEE array element order)
  var sortedIds = Object.keys(classIdToNameMap).map(Number).sort(function(a, b) { return a - b; });
  var classBands = sortedIds.map(function(id) { return classIdToNameMap[id]; });

  var probsImage = probsArrayImage.rename('probs').arrayFlatten([classBands]);

  return {
    probsImage: probsImage,
    classBands: classBands,
    classValues: sortedIds,
    trainedModel: trainedModel
  };
};

/**
 * 4. Unified Classification Selector
 * Supports 'hard', 'soft', and 'multi' / 'multiprobability' modes.
 */
exports.classify = function(mode, trainingPixels, classProperty, image, options) {
  options = options || {};
  mode = (mode || 'hard').toLowerCase();

  if (mode === 'hard') {
    return exports.hardClassification(
      trainingPixels,
      classProperty,
      image,
      options.nTrees,
      options.vSplit,
      options.minLeaf,
      options.seed
    );
  } else if (mode === 'soft') {
    return exports.softClassification(
      trainingPixels,
      classProperty,
      image,
      options.includeFinalMap,
      options.nTrees,
      options.vSplit,
      options.minLeaf,
      options.probabilityScale,
      options.seed
    );
  } else if (mode === 'multi' || mode === 'multiprobability') {
    if (!options.classIdToNameMap) {
      throw new Error("Mode 'multi' requires options.classIdToNameMap (e.g., {1: 'Water', 2: 'Forest'}).");
    }
    return exports.multiProbabilityClassification(
      trainingPixels,
      classProperty,
      options.classIdToNameMap,
      image,
      options
    );
  } else {
    throw new Error("Invalid mode: '" + mode + "'. Choose 'hard', 'soft', or 'multi'.");
  }
};

/**
 * 5. Feature Importance
 */
exports.getFeatureImportance = function(trainedModel) {
  var explanation = trainedModel.explain();
  return ee.Dictionary(explanation.get('importance'));
};

/**
 * 6. Model Evaluation Metrics
 */
exports.evaluateModel = function(trainedModel, testPixels, classProperty) {
  var testClassified = testPixels.classify(trainedModel);
  var errorMatrix = testClassified.errorMatrix(classProperty, 'classification');

  return {
    errorMatrix: errorMatrix,
    overallAccuracy: errorMatrix.accuracy(),
    kappa: errorMatrix.kappa(),
    producersAccuracy: errorMatrix.producersAccuracy(),
    consumersAccuracy: errorMatrix.consumersAccuracy()
  };
};