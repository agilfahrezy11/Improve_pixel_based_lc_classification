//Timor leste AOI
var aoi = ee.FeatureCollection('projects/ee-rg2v2/assets/AOI_Timor_Leste_10km').geometry()

/////////1. CCDC spatiotemporal features/////////
//CCDC source code
var ccdcModule = require('users/rg2icraf/Luma:ccdc_sentinel2');

//define CCDC assets
var tl_ccdc1 = ee.Image('projects/ee-rg2v2/assets/CCDC_TL_2020_2025_20m_westbox')
var tl_ccdc2 = ee.Image('projects/earth-engine-v1-504707/assets/CCDC_TL_2020_2025_20m_east-timor')
var tl_ccdc3 = ee.Image('projects/earth-engine-v1-504707/assets/CCDC_TL_2020_2025_20m_off_dili')
var tl_ccdc4 = ee.Image('projects/gee-v1-510606/assets/CCDC_TL_2020_2025_20m_central-timor')
var tl_ccdc5 = ee.Image('projects/earth-engine-v1-504707/assets/CCDC_TL_2020_2025_20m_east_box')

//combine them into a single mosaic
var ccdc_mosaic = ee.ImageCollection([tl_ccdc1, tl_ccdc2, tl_ccdc3, tl_ccdc4, tl_ccdc5]).mosaic();

//define target year
var targetYears = [2020, 2025];
var fractionOfYear = 0.5;

//build ccdc imagery
var syntheticStacks = ccdcModule.getMultiYearSyntheticStacks(
  ccdc_mosaic, 
  targetYears, 
  fractionOfYear
);

/////////2. Define predictors/////////
//A. Topography and distance
var elev = ee.Image("NASA/NASADEM_HGT/001")
            .select('elevation')
            .clip(aoi)
var slope = ee.Terrain.products(elev).select('slope').toFloat();
var landform = ee.Image('projects/ee-rg2v2/assets/TPI_landform_TL')
                      .rename('landform')
                      .toByte()
                      .clip(aoi)
var twi = ee.Image('projects/ee-rg2/assets/Top_Wetness_index')
                  .rename('TWI')
                  .toFloat()
var dist_urban = ee.Image('projects/ee-rg2/assets/dist_from_urban').rename('dist_urban').clip(aoi)
var dist_road = ee.Image('projects/ee-rg2/assets/dist_from_mainroad').rename('dist_road').clip(aoi)

//stack topography and distance
var topo_dist_stack = ee.Image.cat([elev, slope, landform, twi, dist_urban, dist_road]);

//B. Spectral Index
var index1 = ee.Image('projects/ee-rg2/assets/index_features_TL_2025')
var index2 = ee.Image('projects/ee-v2-508913/assets/MBI_s2_tl')
var index3 = ee.Image('projects/ee-rg2/assets/distance_TL_additional_index_TL').select('SAVI', 'DBSI')
var index_stack = ee.Image.cat([index1, index2, index3])

//C. Radar backscatter
//generate seasonal radar backscatter for 2020
function getSeasonalS1(year, roi) {
  var startDate = ee.Date.fromYMD(year, 1, 1);
  var endDate   = ee.Date.fromYMD(year, 12, 31);

  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']);

  //Monsoonal Splitting
  var wet = s1.filter(ee.Filter.calendarRange(12, 4, 'month')).median();
  var dry = s1.filter(ee.Filter.calendarRange(5, 11, 'month')).median();
  //Rename base bands
  var vv_wet = wet.select('VV').rename('VV_wet');
  var vh_wet = wet.select('VH').rename('VH_wet');

  var vv_dry = dry.select('VV').rename('VV_dry');
  var vh_dry = dry.select('VH').rename('VH_dry');
  return ee.Image.cat([vh_dry, vh_wet, vv_dry, vv_wet]);
} 
//since 2025 already in asset
var s1_seasonal_2025 = ee.Image('projects/ee-v2-508913/assets/s1_seasonal_tl_25')
var s1_seasonal_2020 = getSeasonalS1(2020, aoi).clip(aoi);

/////////3. EXTRACT COEFFICIENTS (AMPLITUDE & PHASE) FROM CCDC ASSET/////////
// Calculate baseline seasonality features directly from NIR harmonic coefficients:
// NIR = c0 + c1*t + a1*cos(w*t) + b1*sin(w*t)
var nirCoefs = ccdc_mosaic.select('nir_coefs');
var a1 = nirCoefs.arraySlice(1, 2, 3).arrayProject([0]).arrayFlatten([['nir_a1']]);
var b1 = nirCoefs.arraySlice(1, 3, 4).arrayProject([0]).arrayFlatten([['nir_b1']]);

var nirAmplitude = a1.hypot(b1).rename('nir_amplitude');
var nirPhase = b1.atan2(a1).rename('nir_phase');

/////////4. COMBINE ALL FEATURES INTO EPOCH MASTER STACKS/////////
var master_stack_2020 = syntheticStacks[2020]
                        .addBands(s1_seasonal_2020)
                        .addBands(topo_dist_stack)
                        .addBands(index_stack)
                        .addBands([nirAmplitude, nirPhase]); //harmonic amplitude & phase
                        
var master_stack_2025 = syntheticStacks[2025]
                        .addBands(s1_seasonal_2025)
                        .addBands(topo_dist_stack)
                        .addBands(index_stack)
                        .addBands([nirAmplitude, nirPhase]);

print('Original Feature stack', master_stack_2025)

/////////5. FILTER TRAINING POINTS & SAMPLE FEATURE STACK/////////
var roi = ee.FeatureCollection('projects/ee-rg2/assets/Samples_TL_hierarchy_2025_v1');

// Optional:Filter out points that fall on detected change breaks during target epochs
var tBreak = ccdc_mosaic.select('tBreak');
var stable_data_2025 = roi.map(function(pt) {
  // 1. Calculate absolute time difference between all segment breaks and 2025.5
  var breakDiff = tBreak.subtract(2025.5).abs();

  // 2. Reduce the 1D break array along axis 0 to find the minimum distance to 2025
  var minBreakDist = breakDiff.arrayReduce(ee.Reducer.min(), [0])
    .arrayProject([0])
    .arrayFlatten([['min_dist']]);
  
  // 3. Mark point as stable if no break occurred within 0.5 years (6 months) of 2025.5
  var isStable = minBreakDist.gte(0.5);
  
  return pt.set('is_stable', isStable);
}).filter(ee.Filter.eq('is_stable', 1));

//Extract exact values across all stacked predictor bands at 10m scale
var sampled_2025 = master_stack_2025.sampleRegions({
  collection: stable_data_2025,
  properties: ['ID', 'year'], // Retain ground truth label
  scale: 10,                        // Re-samples 20m CCDC back to 10m smoothly
  projection: 'EPSG:32751',
  tileScale: 4,
  geometries: false                 // Set false for clean tabular export to CSV
});
Export.table.toDrive({
  collection: sampled_2025,
  description: 'TL_Predictor_Matrix_2025_CCDC',
  fileFormat: 'CSV'
});
