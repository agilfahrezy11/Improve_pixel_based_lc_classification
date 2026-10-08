/**
 * Module: Landsat Collection 2 Preprocessing
 * Description: Fetches, cloud-masks, scales, and standardizes Landsat C2 imagery 
 *              for Surface Reflectance (SR - Level 2) or Top of Atmosphere (TOA - Level 1).
 */

// Collection Asset Mapping
var COLLECTIONS = {
  'SR': {
    'L9': 'LANDSAT/LC09/C02/T1_L2',
    'L8': 'LANDSAT/LC08/C02/T1_L2',
    'L7': 'LANDSAT/LE07/C02/T1_L2',
    'L5': 'LANDSAT/LT05/C02/T1_L2'
  },
  'TOA': {
    'L9': 'LANDSAT/LC09/C02/T1_TOA',
    'L8': 'LANDSAT/LC08/C02/T1_TOA',
    'L7': 'LANDSAT/LE07/C02/T1_TOA',
    'L5': 'LANDSAT/LT05/C02/T1_TOA'
  }
};

/**
 * Cloud and Shadow Masking using Collection 2 QA_PIXEL bitmask
 */
function maskCloudsC2(image) {
  var qa = image.select('QA_PIXEL');
  
  // Bit flags: 1 = Dilated Cloud, 2 = Cirrus, 3 = Cloud, 4 = Cloud Shadow
  var dilatedCloudBit = 1 << 1;
  var cirrusBit       = 1 << 2;
  var cloudBit        = 1 << 3;
  var shadowBit       = 1 << 4;

  var mask = qa.bitwiseAnd(dilatedCloudBit).eq(0)
    .and(qa.bitwiseAnd(cirrusBit).eq(0))
    .and(qa.bitwiseAnd(cloudBit).eq(0))
    .and(qa.bitwiseAnd(shadowBit).eq(0));

  return image.updateMask(mask);
}

/**
 * Apply scaling factors for Collection 2 Level 2 (SR)
 */
function applySRScaling(image) {
  var opticalBands = image.select('SR_B.*').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
}

/**
 * Standardize band names across Landsat sensors
 */
function renameBands(image, sensor, level) {
  var srcBands, dstBands;

  if (level === 'SR') {
    if (sensor === 'L8' || sensor === 'L9') {
      srcBands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'];
      dstBands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP'];
    } else { // L5 / L7
      srcBands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6'];
      dstBands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP'];
    }
  } else { // TOA
    if (sensor === 'L8' || sensor === 'L9') {
      srcBands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10'];
      dstBands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP'];
    } else { // L5 / L7
      srcBands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6_VCID_1'];
      dstBands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP'];
    }
  }

  return image.select(srcBands, dstBands);
}

/**
 * Main Fetch & Preprocess Function
 * 
 * @param {Object} options Configuration parameters:
 *   - aoi {ee.Geometry}: Area of Interest.
 *   - startDate {string}: Start date ('YYYY-MM-DD').
 *   - endDate {string}: End date ('YYYY-MM-DD').
 *   - level {string}: Processing level ('SR' or 'TOA'). Default: 'SR'.
 *   - sensors {Array<string>}: List of sensors (e.g. ['L8', 'L9']). Default: ['L8', 'L9'].
 *   - maskClouds {boolean}: Whether to apply QA cloud mask. Default: true.
 *   - maxCloudCover {number}: Maximum scene-level cloud cover percentage (0–100). Default: 100 (no filter).
 *   - rename {boolean}: Whether to standardize band names. Default: true.
 * @returns {ee.ImageCollection} Cleaned, scaled, and standardized image collection.
 */
exports.getCollection = function(options) {
  options = options || {};
  
  var aoi        = options.aoi;
  var startDate  = options.startDate;
  var endDate    = options.endDate;
  var level      = (options.level || 'SR').toUpperCase();
  var sensors    = options.sensors || ['L8', 'L9'];
  var maskClouds   = options.maskClouds !== undefined ? options.maskClouds : true;
  var maxCloudCover = options.maxCloudCover !== undefined ? options.maxCloudCover : 100;
  var rename       = options.rename !== undefined ? options.rename : true;

  if (!aoi || !startDate || !endDate) {
    throw new Error("Missing required parameters: 'aoi', 'startDate', or 'endDate'.");
  }

  // Iterate over specified sensors and merge collections
  var collections = sensors.map(function(sensor) {
    var assetId = COLLECTIONS[level][sensor];
    if (!assetId) {
      throw new Error("Invalid sensor or level specification: " + sensor + " / " + level);
    }

    var col = ee.ImageCollection(assetId)
      .filterBounds(aoi)
      .filterDate(startDate, endDate)
      .filter(ee.Filter.lte('CLOUD_COVER', maxCloudCover));

    // Apply cloud mask
    if (maskClouds) {
      col = col.map(maskCloudsC2);
    }

    // Apply scaling factors (Level 2 SR only)
    if (level === 'SR') {
      col = col.map(applySRScaling);
    }

    // Standardize band names
    if (rename) {
      col = col.map(function(img) {
        return renameBands(img, sensor, level);
      });
    }

    return col;
  });

  // Flatten merged collection array
  var mergedCol = ee.ImageCollection(collections[0]);
  for (var i = 1; i < collections.length; i++) {
    mergedCol = mergedCol.merge(collections[i]);
  }

  return mergedCol.sort('system:time_start');
};