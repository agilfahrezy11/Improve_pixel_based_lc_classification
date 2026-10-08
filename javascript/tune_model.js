/**
 * Module: Hyperparameter Tuning
 * Description: 
 * Performs grid-search style hyperparameter optimization for Earth Engine Random Forest classifier.
 * Currently, only supports hard classification tuning. Soft classification required custom log loss metric
 */

/**
 * Grid search over specified Random Forest hyperparameters.
 *
 * @param {ee.FeatureCollection} trainingPixels - FeatureCollection containing extracted training pixel values.
 * @param {ee.FeatureCollection} testingPixels - FeatureCollection containing extracted testing pixel values.
 * @param {string} classProp - Property name containing class labels.
 * @param {ee.List|Array} inputProperties - List of band/covariate names used as features.
 * @param {ee.List|Array} [numTreesList=[50, 100, 200]] - Array/List of tree counts to test.
 * @param {ee.List|Array} [variablesPerSplitList=[0]] - Array/List of variables per split (0 uses default sqrt(N)).
 * @param {ee.List|Array} [minLeafPopulationList=[1]] - Array/List of min leaf population sizes to test.
 * @param {number} [seed=0] - Random seed for reproducibility.
 * @returns {ee.FeatureCollection} Collection where each feature contains parameter combinations and metrics ('accuracy', 'kappa').
 */
exports.tuneRandomForest = function(
  trainingPixels,
  testingPixels,
  classProp,
  inputProperties,
  numTreesList,
  variablesPerSplitList,
  minLeafPopulationList,
  seed
) {
  // Set default search grids if not provided
  numTreesList = ee.List(numTreesList || [50, 100, 200]);
  variablesPerSplitList = ee.List(variablesPerSplitList || [0]); // 0 indicates default sqrt(N)
  minLeafPopulationList = ee.List(minLeafPopulationList || [1]);
  seed = seed !== undefined ? seed : 0;
  inputProperties = ee.List(inputProperties);

  // Nested mapping over 3 hyperparameter dimensions
  var grid = numTreesList.map(function(numTrees) {
    return variablesPerSplitList.map(function(vSplit) {
      return minLeafPopulationList.map(function(minLeaf) {
        
        numTrees = ee.Number(numTrees);
        vSplit = ee.Number(vSplit);
        minLeaf = ee.Number(minLeaf);

        // Build classifier conditionally based on whether vSplit is specified (> 0)
        var classifier = ee.Algorithms.If(
          vSplit.gt(0),
          ee.Classifier.smileRandomForest({
            numberOfTrees: numTrees,
            variablesPerSplit: vSplit,
            minLeafPopulation: minLeaf,
            seed: seed
          }),
          ee.Classifier.smileRandomForest({
            numberOfTrees: numTrees,
            minLeafPopulation: minLeaf,
            seed: seed
          })
        );

        // Train classifier
        var trainedClf = ee.Classifier(classifier).train({
          features: trainingPixels,
          classProperty: classProp,
          inputProperties: inputProperties
        });

        // Evaluate model performance on validation set
        var testClassified = testingPixels.classify(trainedClf);
        var matrix = testClassified.errorMatrix(classProp, 'classification');
        
        var accuracy = matrix.accuracy();
        var kappa = matrix.kappa();

        return ee.Feature(null, {
          'numberOfTrees': numTrees,
          'variablesPerSplit': vSplit,
          'minLeafPopulation': minLeaf,
          'accuracy': accuracy,
          'kappa': kappa
        });
      });
    });
  });

  // Flatten 3D array into a 1D FeatureCollection
  return ee.FeatureCollection(ee.List(grid).flatten());
};

/**
 * Extracts the optimal parameter set from tuning results based on a metric.
 *
 * @param {ee.FeatureCollection} tuningResults - Output from exports.tuneRandomForest.
 * @param {string} [metric='accuracy'] - Evaluation metric to optimize ('accuracy' or 'kappa').
 * @returns {ee.Feature} Feature containing the best parameter values and metric scores.
 */
exports.getBestParams = function(tuningResults, metric) {
  metric = metric || 'accuracy';
  return tuningResults.sort(metric, false).first();
};
/**
 * 7. Train a MULTIPROBABILITY model only (no map is built).
 */
exports.trainMultiProbModel = function(trainingPixels, classProperty, inputProperties, options) {
  options = options || {};
  var rfParams = buildRfParams(options.nTrees, options.vSplit, options.minLeaf, options.seed);
  return ee.Classifier.smileRandomForest(rfParams)
    .setOutputMode('MULTIPROBABILITY')
    .train({
      features: trainingPixels,
      classProperty: classProperty,
      inputProperties: inputProperties
    });
};

/**
 * 8. Class values in the order the model outputs them (ascending, from the training data).
 */
exports.getClassValues = function(trainingPixels, classProperty) {
  return trainingPixels.aggregate_array(classProperty).distinct().sort();
};

/**
 * 9. Probability metrics on a table of test/validation pixels.
 * Returns log loss and top-class accuracy from a single classify pass.
 */
exports.evaluateProbabilities = function(trainedModel, testPixels, classProperty, classValues, eps) {
  eps = (eps !== undefined) ? eps : 1e-3;
  var classList = ee.List(classValues);

  // drop test rows whose class was absent from training (no probability column for them)
  var valid = testPixels.filter(ee.Filter.inList(classProperty, classList));
  var scored = valid.classify(trainedModel, 'probs');

  var withMetrics = scored.map(function(f) {
    var probs = ee.List(f.get('probs'));
    var trueIdx = classList.indexOf(ee.Number(f.get(classProperty)));
    var pTrue = ee.Number(probs.get(trueIdx)).max(eps);
    var predIdx = ee.List(ee.Array(probs).argmax()).get(0);
    return f.set({
      nll: pTrue.log().multiply(-1),
      correct: ee.Number(predIdx).eq(trueIdx)
    });
  });

  return {
    logLoss: withMetrics.aggregate_mean('nll'),
    accuracy: withMetrics.aggregate_mean('correct'),
    n: withMetrics.size()
  };
};