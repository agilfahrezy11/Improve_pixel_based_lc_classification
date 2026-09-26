/**
 * Module: Post-Processing & Spatial Smoothing
 * Description: Empirical Bayes ("Bayesian") spatial smoothing for land cover classification 
 *              probability images. Based on Camara et al. (2024).
 * Reference: https://e-sensing.github.io/sitsbook/cl_smoothing.html
 */

/**
 * Module: Post-Processing & Spatial Smoothing (Fixed & Robust)
 */
var EPS = 1e-6;

function probsToLogit(probsImage, classBands) {
  var p = probsImage.select(classBands).clamp(EPS, 1 - EPS);
  var one = ee.Image.constant(1);
  return p.divide(one.subtract(p)).log().rename(classBands);
}

function logitToProbs(logitImage, classBands, normalize) {
  normalize = (normalize === undefined) ? true : normalize;
  var p = ee.Image.constant(1)
    .divide(ee.Image.constant(1).add(logitImage.multiply(-1).exp()))
    .rename(classBands);

  if (normalize) {
    var total = p.reduce(ee.Reducer.sum());
    p = p.divide(total).rename(classBands);
  }
  return p;
}

function buildSigma2Image(classBands, smoothness) {
  var bandsArray = Array.isArray(classBands) ? classBands : classBands.getInfo();
  
  var bands = bandsArray.map(function(b) {
    var val = (typeof smoothness === 'object' && smoothness !== null && smoothness[b] !== undefined)
      ? smoothness[b]
      : (typeof smoothness === 'number' ? smoothness : 1.0);
    return ee.Image.constant(ee.Number(val)).rename([b]);
  });
  return ee.Image.cat(bands);
}

function localMeanVariance(logitImage, classBands, windowSize, neighFraction) {
  windowSize = windowSize || 7;
  neighFraction = neighFraction || 0.5;

  var radius = Math.floor((windowSize - 1) / 2);
  var kernel = ee.Kernel.square(radius, 'pixels');

  var keepPct = Math.round((1 - neighFraction) * 100);
  var localThreshold = logitImage
    .reduceNeighborhood({reducer: ee.Reducer.percentile([keepPct]), kernel: kernel})
    .rename(classBands);

  var selected = logitImage.updateMask(logitImage.gte(localThreshold));

  var localMean = selected
    .reduceNeighborhood({reducer: ee.Reducer.mean(), kernel: kernel})
    .rename(classBands);

  var localVar = selected
    .reduceNeighborhood({reducer: ee.Reducer.variance(), kernel: kernel})
    .rename(classBands);

  localVar = localVar.unmask(1.0).where(localVar.eq(0), 1.0);
  localMean = localMean.unmask(logitImage);

  return { mean: localMean, variance: localVar };
}

exports.localVariance = function(probsImage, classBands, windowSize, neighFraction) {
  var logitImage = probsToLogit(probsImage, classBands);
  var stats = localMeanVariance(logitImage, classBands, windowSize, neighFraction);
  return stats.variance;
};

exports.bayesianSmooth = function(probsImage, classBands, smoothness, windowSize, neighFraction) {
  windowSize = windowSize || 7;
  neighFraction = neighFraction || 0.5;

  var logitImage = probsToLogit(probsImage, classBands);
  var stats = localMeanVariance(logitImage, classBands, windowSize, neighFraction);
  var sigma2 = buildSigma2Image(classBands, smoothness);

  var wX = stats.variance.divide(sigma2.add(stats.variance));
  var wM = sigma2.divide(sigma2.add(stats.variance));

  var posteriorLogit = logitImage.multiply(wX)
    .add(stats.mean.multiply(wM))
    .rename(classBands);

  return logitToProbs(posteriorLogit, classBands, true);
};

exports.classifyFromProbs = function(probsImage, classBands, classValues) {
  var arr = probsImage.select(classBands).toArray();
  var maxIdx = arr.arrayArgmax().arrayGet([0]).rename('classification');

  if (classValues) {
    var bandsArray = Array.isArray(classBands) ? classBands : classBands.getInfo();
    var seq = ee.List.sequence(0, bandsArray.length - 1);
    
    // Wrapped classValues in ee.List() to prevent type mismatch
    maxIdx = maxIdx.remap(seq, ee.List(classValues)).rename('classification');
  }

  return maxIdx.toByte();
};