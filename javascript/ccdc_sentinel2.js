/**
 * Module: Sentinel-2 CCDC Temporal Segmentation & Synthetic Stack Generator
 * Description: Prepares Sentinel-2 Surface Reflectance time series using Cloud Score+, 
 *              executes CCDC temporal segmentation, and evaluates harmonic coefficients 
 *              to generate synthetic predictor stacks for classification.
 */

/**
 * 1. Cloud Masking & Scaling Helper (Cloud Score+)
 * Applies Cloud Score+ (cs_cdf or cs) quality threshold and scales SR bands by 10,000.
 *
 * @param {ee.Image} image - Sentinel-2 image with linked Cloud Score+ band.
 * @param {number} [clearThreshold=0.60] - Minimum quality score to keep (0.50 - 0.65).
 * @param {string} [qaBand='cs_cdf'] - Quality band ('cs_cdf' or 'cs').
 * @returns {ee.Image} Cloud-masked and scaled image.
 */
exports.maskAndScaleS2 = function(image, clearThreshold, qaBand) {
  clearThreshold = clearThreshold !== undefined ? clearThreshold : 0.60;
  qaBand = qaBand || 'cs_cdf';

  var cloudMask = image.select(qaBand).gte(clearThreshold);
  var inputBands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B8', 'B11', 'B12'];
  var outputBands = ['blue', 'green', 'red', 're1','re2', 'nir', 'swir1', 'swir2'];

  return image.select(inputBands, outputBands)
    .divide(10000)
    .updateMask(cloudMask);
};

/**
 * 2. Prepare S2 Time Series Collection for CCDC (using Cloud Score+)
 * Links Sentinel-2 L2A with Cloud Score+, masks clouds/shadows, renames bands,
 * and attaches fractional year timestamps ('t').
 *
 * @param {ee.Geometry} roi - Region of interest boundary.
 * @param {string} [startDate='2015-06-01'] - Start date for time series.
 * @param {string} [endDate='2025-12-31'] - End date for time series.
 * @param {number} [clearThreshold=0.60] - Cloud Score+ quality threshold.
 * @param {string} [qaBand='cs_cdf'] - Quality band to evaluate.
 * @param {Array<string>} [customBands] - Custom band names output.
 * @returns {ee.ImageCollection} Processed collection ready for CCDC.
 */
exports.getS2TimeSeries = function(roi, startDate, endDate, clearThreshold, qaBand) {
  startDate = startDate || '2018-12-01'; //example date
  endDate = endDate || '2025-12-31'; //example date
  clearThreshold = clearThreshold !== undefined ? clearThreshold : 0.60;
  qaBand = qaBand || 'cs_cdf';

  var inputBands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B8', 'B11', 'B12'];
  var outputNames = ['blue', 'green', 'red','re1','re2', 'nir', 'swir1', 'swir2'];

  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate(startDate, endDate);

  var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    .filterBounds(roi)
    .filterDate(startDate, endDate);

  var s2Linked = s2.linkCollection(csPlus, [qaBand]);

  return s2Linked.map(function(img) {
    var cloudMask = img.select(qaBand).gte(clearThreshold);
    var masked = img.select(inputBands, outputNames).updateMask(cloudMask);
    return masked.copyProperties(img, ['system:time_start']);
  });
};


/**
 * 3. Run CCDC Algorithm
 * Wraps ee.Algorithms.TemporalSegmentation.Ccdc with sensible defaults.
 *
 * @param {ee.ImageCollection} s2Collection - Processed S2 time series from getS2TimeSeries.
 * @param {Object} [options] - CCDC hyperparameter overrides.
 * @returns {ee.Image} CCDC array output image containing coefficients and break metadata.
 */
exports.runCCDC = function(s2Collection, options) {
  options = options || {};

  var params = {
    collection: s2Collection,
    breakpointBands: options.breakpointBands || ['green', 'red', 'nir', 'swir1'],
    dateFormat: options.dateFormat !== undefined ? options.dateFormat : 1, // Fractional years
    minObservations: options.minObservations || 6,
    chiSquareProbability: options.chiSquareProbability || 0.99,
    minNumOfYearsScaler: options.minNumOfYearsScaler || 1.33,
    lambda: options.lambda || 20,
    maxIterations: options.maxIterations || 25000
  };

  return ee.Algorithms.TemporalSegmentation.Ccdc(params);
};

/**
 * 4. Generate Synthetic Predictor Stack for a Single Date
 * Evaluates CCDC harmonic regression coefficients at target fractional year t:
 * y(t) = c0 + c1*t + a1*cos(2pi*t) + b1*sin(2pi*t)
 *
 * @param {ee.Image} ccdcOutput - Output image from runCCDC.
 * @param {number|ee.Number} targetDateFractional - Date in fractional year (e.g., 2025.5 for July 1, 2025).
 * @param {Array<string>} [bands] - Spectral band names matching CCDC outputs.
 * @param {boolean} [addIndices=true] - Whether to compute spectral indices (NDVI, NDWI, NDBI).
 * @returns {ee.Image} Synthetic surface reflectance stack with indices.
 */
exports.getSyntheticStack = function(ccdcOutput, targetDateFractional, bands, addIndices) {
  bands = bands || ['blue', 'green', 'red', 're1','re2', 'nir', 'swir1', 'swir2'];
  addIndices = addIndices !== undefined ? addIndices : true;

  var t = ee.Number(targetDateFractional);
  var tStart = ccdcOutput.select('tStart');
  var tEnd = ccdcOutput.select('tEnd');

  // 1. Create dynamic segment boolean mask (1D: [num_segments])
  var segmentMask = tStart.lte(t).and(tEnd.gte(t));
  
  // FIX: Convert 1D mask [num_segments] to 2D array mask [num_segments, 1]
  var segmentMask2D = segmentMask.toArray(0).toArray(1);

  // 2. Build harmonic vector [8 x 1]
  var omega = 2.0 * Math.PI;
  var termList = [
    ee.Image.constant(1),
    ee.Image.constant(t),
    ee.Image.constant(t.multiply(omega).cos()),
    ee.Image.constant(t.multiply(omega).sin()),
    ee.Image.constant(t.multiply(omega * 2).cos()),
    ee.Image.constant(t.multiply(omega * 2).sin()),
    ee.Image.constant(t.multiply(omega * 3).cos()),
    ee.Image.constant(t.multiply(omega * 3).sin())
  ];
  var termsArray = ee.Image(termList).toArray().toArray(1);

  // 3. Extract coefficients and calculate synthetic fits
  var syntheticBands = bands.map(function(band) {
    var coefs = ccdcOutput.select(band + '_coefs'); // shape: [num_segments, 8]
    
    // Apply 2D mask along segment dimension
    var activeCoefs = coefs.arrayMask(segmentMask2D);
    
    // Slice active segment (shape: [1, 8])
    var segCoefs = activeCoefs.arraySlice(0, 0, 1);
    
    // Matrix multiplication: [1 x 8] * [8 x 1] = [1 x 1]
    var fitArray = segCoefs.matrixMultiply(termsArray);
    
    // Project and flatten array to scalar image, then scale down 10,000 -> 0..1
    return fitArray.arrayProject([0]).arrayFlatten([[band]]).divide(10000);
  });

  var stack = ee.Image(syntheticBands);

  if (addIndices) {
    var ndvi = stack.normalizedDifference(['nir', 'red']).rename('NDVI');
    var ndwi = stack.normalizedDifference(['green', 'nir']).rename('NDWI');
    var ndbi = stack.normalizedDifference(['swir1', 'nir']).rename('NDBI');
    stack = stack.addBands([ndvi, ndwi, ndbi]);
  }

  return stack;

};
/**
 * 5. Multi-Year Synthetic Stack Generator
 * Generates a dictionary of synthetic stacks for multiple target years.
 *
 * @param {ee.Image} ccdcOutput - Output image from runCCDC.
 * @param {Array<number>} yearsList - List of target years (e.g., [2015, 2020, 2025]).
 * @param {number} [fractionOfYear=0.5] - Day of year fraction (e.g., 0.5 = ~July 1 dry season).
 * @param {Array<string>} [bands] - Spectral band list.
 * @param {boolean} [addIndices=true] - Whether to compute indices.
 * @returns {Object} Map with year keys pointing to ee.Image synthetic stacks.
 */
exports.getMultiYearSyntheticStacks = function(ccdcOutput, yearsList, fractionOfYear, bands, addIndices) {
  fractionOfYear = fractionOfYear !== undefined ? fractionOfYear : 0.5;
  
  var stacks = {};
  yearsList.forEach(function(year) {
    var targetT = ee.Number(year).add(fractionOfYear);
    stacks[year] = exports.getSyntheticStack(ccdcOutput, targetT, bands, addIndices);
  });

  return stacks;
};
/**
 * Monthly median composites from a masked S2 collection.
 * Months with no valid observations are dropped.
 * Timestamp is set to mid-month so CCDC sees one observation per month.
 */
exports.toMonthlyMedian = function(col, startDate, endDate) {
  var start = ee.Date(startDate);
  var nMonths = ee.Date(endDate).difference(start, 'month').floor();
  var comps = ee.List.sequence(0, nMonths.subtract(1)).map(function(m) {
    var d0 = start.advance(m, 'month');
    var sub = col.filterDate(d0, d0.advance(1, 'month'));
    return sub.median()
      .set('system:time_start', d0.advance(15, 'day').millis())
      .set('n', sub.size());
  });
  return ee.ImageCollection.fromImages(comps).filter(ee.Filter.gt('n', 0));
};