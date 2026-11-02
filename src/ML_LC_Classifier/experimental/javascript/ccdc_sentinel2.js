/**
 * Module: Sentinel-2 CCDC Temporal Segmentation & Synthetic Stack Generator
 * Description: Prepares Sentinel-2 Surface Reflectance time series using Cloud Score+, 
 *              executes CCDC temporal segmentation, and evaluates harmonic coefficients 
 *              to generate synthetic predictor stacks for classification with gap tolerance.
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
 * 4. Nearest segment in time per pixel (index, temporal gap in years, tStart, tEnd).
 * Returns gap = 0 if targetDateFractional t is strictly inside an active segment.
 *
 * @param {ee.Image} ccdc - CCDC array image containing tStart and tEnd.
 * @param {number|ee.Number} t - Target fractional year (e.g., 2025.65).
 * @returns {Object} {idx, gap, tS, tE}
 */
exports.nearestSegment = function(ccdc, t) {
  t = ee.Number(t);
  var tS = ccdc.select('tStart');
  var tE = ccdc.select('tEnd');
  var dist = tS.subtract(t).max(tE.multiply(-1).add(t)).max(0); // 0 if inside a segment
  return {
    idx: dist.multiply(-1).arrayArgmax().arrayGet([0]),
    gap: dist.arrayReduce(ee.Reducer.min(), [0]).arrayGet([0]).rename('gap_years'),
    tS: tS,
    tE: tE
  };
};

/**
 * 5. Fill isolated nodata holes from 3x3 focal median, only where enough neighbors are valid.
 *
 * @param {ee.Image} img - Image with potential isolated nodata holes.
 * @param {number} [minNeighbors=5] - Minimum valid neighbors in 3x3 window (out of 8).
 * @returns {ee.Image} Gap-filled image.
 */
exports.fillIsolated = function(img, minNeighbors) {
  minNeighbors = minNeighbors !== undefined ? minNeighbors : 5;
  var n = img.select(0).mask().reduceNeighborhood({
    reducer: ee.Reducer.sum(),
    kernel: ee.Kernel.square(1)
  });
  var med = img.focalMedian(1, 'square', 'pixels');
  return img.unmask(med.updateMask(n.gte(minNeighbors)));
};

/**
 * 6. Generate Synthetic Predictor Stack for a Single Date with Gap Tolerance
 * Evaluates CCDC harmonic regression coefficients at target fractional year t.
 * Uses the nearest segment in time with a configurable gap tolerance (tolYears).
 * Clamps the linear trend term to the segment bounds while preserving seasonal harmonics.
 *
 * @param {ee.Image} ccdcOutput - Output image from runCCDC.
 * @param {number|ee.Number} targetDateFractional - Date in fractional year (e.g., 2025.65).
 * @param {Array<string>|boolean} [bands] - Spectral band names matching CCDC outputs, OR boolean addIndices.
 * @param {boolean|number} [addIndices=true] - Whether to compute spectral indices (NDVI, NDWI, NDBI), OR tolYears.
 * @param {number} [coefficientScale=10000] - Scale of CCDC coefficients (default 10000).
 * @param {number} [tolYears=1.0] - Maximum gap in years to nearest valid segment (default 1.0).
 * @returns {ee.Image} Synthetic surface reflectance stack with indices.
 */
exports.getSyntheticStack = function(ccdcOutput, targetDateFractional, bands, addIndices, coefficientScale, tolYears) {
  if (typeof bands === 'boolean') {
    var _addIndices = bands;
    var _tolYears = addIndices !== undefined ? addIndices : 1.0;
    var _coefScale = coefficientScale !== undefined ? coefficientScale : 10000;
    tolYears = _tolYears;
    coefficientScale = _coefScale;
    addIndices = _addIndices;
    bands = ['blue', 'green', 'red', 're1','re2', 'nir', 'swir1', 'swir2'];
  } else {
    bands = bands || ['blue', 'green', 'red', 're1','re2', 'nir', 'swir1', 'swir2'];
    addIndices = addIndices !== undefined ? addIndices : true;
    coefficientScale = coefficientScale !== undefined ? coefficientScale : 10000;
    tolYears = tolYears !== undefined ? tolYears : 1.0;
  }

  var t = ee.Number(targetDateFractional);
  var seg = exports.nearestSegment(ccdcOutput, t);

  // Clamp only the trend term; harmonics keep the target date's season
  var tc = ee.Image(t).max(seg.tS.arrayGet(seg.idx)).min(seg.tE.arrayGet(seg.idx));
  var omega = 2.0 * Math.PI;
  var terms = ee.Image.cat([
    ee.Image.constant(1),
    tc,
    ee.Image.constant(t.multiply(omega).cos()),
    ee.Image.constant(t.multiply(omega).sin()),
    ee.Image.constant(t.multiply(omega * 2).cos()),
    ee.Image.constant(t.multiply(omega * 2).sin()),
    ee.Image.constant(t.multiply(omega * 3).cos()),
    ee.Image.constant(t.multiply(omega * 3).sin())
  ]).toArray();

  var syntheticBands = bands.map(function(band) {
    var coefs = ccdcOutput.select(band + '_coefs')
      .arraySlice(0, seg.idx, seg.idx.add(1))
      .arrayProject([1]);
    
    return coefs.multiply(terms).arrayReduce(ee.Reducer.sum(), [0])
      .arrayGet([0]).divide(coefficientScale).rename(band);
  });

  var stack = ee.Image.cat(syntheticBands);

  // Plausibility bounds: within tolerance and valid reflectance
  var valid = seg.gap.lte(tolYears)
    .and(stack.reduce(ee.Reducer.min()).gt(-0.05))
    .and(stack.reduce(ee.Reducer.max()).lt(1.5));
  stack = exports.fillIsolated(stack.updateMask(valid), 5);

  if (addIndices) {
    var ndvi = stack.normalizedDifference(['nir', 'red']).rename('NDVI');
    var ndwi = stack.normalizedDifference(['green', 'nir']).rename('NDWI');
    var ndbi = stack.normalizedDifference(['swir1', 'nir']).rename('NDBI');
    stack = stack.addBands([ndvi, ndwi, ndbi]);
  }

  return stack;
};

/**
 * 7. Multi-Year Synthetic Stack Generator
 * Generates a dictionary of synthetic stacks for multiple target years.
 *
 * @param {ee.Image} ccdcOutput - Output image from runCCDC.
 * @param {Array<number>} yearsList - List of target years (e.g., [2015, 2020, 2025]).
 * @param {number} [fractionOfYear=0.5] - Day of year fraction (e.g., 0.5 = ~July 1 dry season).
 * @param {Array<string>} [bands] - Spectral band list.
 * @param {boolean} [addIndices=true] - Whether to compute indices.
 * @param {number} [coefficientScale=10000] - CCDC coefficient scale.
 * @param {number} [tolYears=1.0] - Maximum gap in years to nearest valid segment.
 * @returns {Object} Map with year keys pointing to ee.Image synthetic stacks.
 */
exports.getMultiYearSyntheticStacks = function(ccdcOutput, yearsList, fractionOfYear, bands, addIndices, coefficientScale, tolYears) {
  fractionOfYear = fractionOfYear !== undefined ? fractionOfYear : 0.5;
  tolYears = tolYears !== undefined ? tolYears : 1.0;
  
  var stacks = {};
  yearsList.forEach(function(year) {
    var targetT = ee.Number(year).add(fractionOfYear);
    stacks[year] = exports.getSyntheticStack(ccdcOutput, targetT, bands, addIndices, coefficientScale, tolYears);
  });

  return stacks;
};

/**
 * 8. Monthly median composites from a masked S2 collection.
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
