/*
TIME SERIES CLASSIFICATION ASSEMBLY
*/

/////////1. Setup/////////
var aoi = ee.FeatureCollection('projects/ee-rg2v2/assets/AOI_Timor_Leste_10km').geometry()
//post processing module
var post = require('users/rg2icraf/Luma:post_process')
var clf = require('users/rg2icraf/Luma:classify_model')
//Probability based land cover
var lc_prob_2025 = ee.Image('projects/ee-rg2/assets/LC_probs_2025_TL_CCDC').divide(10000)
var lc_prob_2020 = ee.Image('projects/ee-rg2/assets/LC_probs_2020_TL_CCDC').divide(10000)
//check the data
var visparam = {min: 0, max: 1, palette: ['blue', 'white', 'green']}
//Map.addLayer(lc_prob_2025.select('Primary_forest'), visparam, 'Primary Forest Probability')
//inspect stats
var mapStats = lc_prob_2025.reduceRegion({
  reducer: ee.Reducer.max().combine(ee.Reducer.percentile([99]), null, true),
  geometry: aoi,
  scale: 100,
  maxPixels: 1e10,
  tileScale: 16,
  bestEffort: true
});
//print('Per-class map max / p99:', mapStats);

/////////2. Define Land Cover Class and Smoothing Parameter/////////
//class name
var class_bands = ee.List(['Primary_forest', 'Secondary_forest', 'Mangrove_forest' ,'Coastal_forest',
                  'Teak_plantation', 'Eucalyptus_plantation', 'Coffe_agroforestry', 'Cocoa_agroforestry', 
                  'Coconut_agroforestry', 'Mixed_garden', 'Paddy_field', 'Other_cropland', 'Grassland', 'Shrubland', 
                  'Cleared_land', 'Built-up', 'Waterbody', 'Dry_riverbed']);
//class id
var class_value = ee.List([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
//Smoothing_parameter
var smoothness = {
  'Primary_forest': 4.5,       //Lowered from 
  'Secondary_forest': 4,     //Adjusted to protect sparse secondary forest
  'Mangrove_forest': 2.5,
  'Coastal_forest': 2.7,
  'Teak_plantation': 1.0,
  'Eucalyptus_plantation': 1.0,
  'Coffe_agroforestry': 1.0,   // Lowered from 1.5/2.0 (was over-dominating p50 = 0.56)
  'Cocoa_agroforestry': 1.0,
  'Coconut_agroforestry': 1.0,
  'Mixed_garden': 0.8,
  'Paddy_field': 0.8,          // Lowered to preserve field boundaries
  'Other_cropland': 0.8,
  'Grassland': 2.5,
  'Shrubland': 2.5,
  'Cleared_land': 1.4,
  'Built-up': 1,             // Preserved low
  'Waterbody': 1.5,
  'Dry_riverbed': 0.5          // Preserved low
}

//Bayesian Smoothing
/////////2. Implement post-processing bayesian smoothing/////////

//Original landcover (unsmooth), from argmax
var landcover_2025 = post.classifyFromProbs(lc_prob_2025, class_bands, class_value);
//smoothing applied to probability images
var smoothedProbs_2025 = post.bayesianSmooth(lc_prob_2025,class_bands,
                      smoothness,
                      7,    //Window size
                      0.7   //Neighbor fraction
                    );
//generate land cover map using argmax
var lc_smooth_2025 = post.classifyFromProbs(smoothedProbs_2025, class_bands, class_value);

//2020 land cover
//2020 probabilitty
var landcover_2020 = post.classifyFromProbs(lc_prob_2020, class_bands, class_value);
//bayesian smooth for the probability
var smoothedProbs_2020 = post.bayesianSmooth(lc_prob_2020,class_bands,
                      smoothness,
                      7,    // Window size (5x5)
                      0.7   // Neighbor fraction
                    );
//generate the final mal using smoothed probability
var lc_smooth_2020 = post.classifyFromProbs(smoothedProbs_2020, class_bands, class_value);                    
/*
var merged_2020 = post.mergeClasses(smoothedProbs_2020, class_bands, class_value, {
  'Waterbody': {members: ['Waterbody', 'Dry_riverbed'], value: 17}
});
var merged_2025 = post.mergeClasses(smoothedProbs_2025, class_bands, class_value, {
  'Waterbody': {members: ['Waterbody', 'Dry_riverbed'], value: 17}
});
*/

/////////3. Adding Land Cover to layer/////////
var classVis = {
  min: 1,
  max: 18,
  palette: [
    '#00441B', // 1  Primary_forest
    '#238B45', // 2  Secondary_forest
    '#006D2C', // 3  Mangrove_forest
    '#66C2A4', // 4  Coastal_forest
    '#8C6D31', // 5  Teak_plantation
    '#A6761D', // 6  Eucalyptus_plantation
    '#7FC97F', // 7  Coffe_agroforestry
    '#66A061', // 8  Cocoa_agroforestry
    '#41AB5D', // 9  Coconut_agroforestry
    '#A1D76A', // 10  Mixed_garden
    '#C7E9B4', // 11 Paddy_field
    '#FDD835', // 12 Other_cropland
    '#D9EF8B', // 13 Grassland
    '#C2A878', // 14 Shrubland
    '#d7d6d6', // 15 Cleared_land
    '#DB310D', // 16 Built-up
    '#0e8acc', // 17 Waterbody
    '#9ECAE1'  // 18 Dry_riverbed
  ]
};

//Map.addLayer(landcover_2025, classVis, 'Raw Land Cover (argmax')
Map.addLayer(lc_smooth_2020, classVis, 'Smooth Land Cover (argmax) 2020')
Map.addLayer(lc_smooth_2025, classVis, 'Smooth Land Cover (argmax) 2025')

///////// 3. Load & Process Auxiliary Datasets /////////
// A. ETH Global Sentinel-2 10m Canopy Height (2020)
var ethCanopy = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').clip(aoi);
// B. Global Pasture Watch 30m Short Vegetation Height
var gsvh = ee.ImageCollection('projects/global-pasture-watch/assets/gsvh-30m/v1/short-veg-height')
  .filterDate('2025-01-01', '2025-12-31')
  .first()
  .select('short_veg_height')
  .clip(aoi);
// C. High Resolution Settlement Layer (HRSL / Facebook Meta)
var hrsl = ee.ImageCollection('projects/sat-io/open-datasets/hrsl')
  .filterBounds(aoi)
  .mosaic()
  .gt(0)
  .clip(aoi);
  
///////// 4. Auxiliary Structural Mask Constraints /////////
// Primary Forest constraint: Height must be >= 15m
var isShortCanopy = ethCanopy.lt(15);
// Built-up constraint: Must fall within HRSL populated footprints
var isNonSettlement = hrsl.unmask(0).eq(0);
// Grassland / Shrubland constraint: Short vegetation validation
var isTallVegetation = ethCanopy.gt(12);
// Apply constraints as probability penalizers prior to classification
var adjustProbs = function(probImage) {
  var pPrimary = probImage.select('Primary_forest').multiply(isShortCanopy.not());
  var pSecondary = probImage.select('Secondary_forest').add(probImage.select('Primary_forest').multiply(isShortCanopy));
  var pBuilt = probImage.select('Built-up').multiply(isNonSettlement.not());
  var pGrass = probImage.select('Grassland').multiply(isTallVegetation.not());
  
  return probImage
    .addBands(pPrimary, ['Primary_forest'], true)
    .addBands(pSecondary, ['Secondary_forest'], true)
    .addBands(pBuilt, ['Built-up'], true)
    .addBands(pGrass, ['Grassland'], true);
};

var constrainedProbs_2025 = adjustProbs(smoothedProbs_2025);
var constrainedProbs_2020 = adjustProbs(smoothedProbs_2020);

/////////5. CCDC Masking layer/////////
//use CCDC features to detect breaks in the time series
//breaks might indicate unstable pixels
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

//CCDC: pixels with no break between the two map dates are stable
var tBreak = ccdc_mosaic.select('tBreak');
var nBreaks = tBreak.gte(2020.3).and(tBreak.lte(2025.7))
  .arrayReduce(ee.Reducer.sum(), [0]).arrayProject([0])
  .arrayFlatten([['n_breaks']])
  .unmask(1);                          // no CCDC data = not treated as stable
var stable = nBreaks.eq(0)
 // .focalMode({radius: 20, units: 'meters'});   // removes isolated stable/unstable pixels
  
//2025 is the anchor. 2020 inherits it where nothing changed.
var finalProbs_2025 = constrainedProbs_2025; //change to constrained probability, modified to smoothedProbs_2025 to revert
var finalProbs_2020 = constrainedProbs_2025.multiply(stable)
  .add(constrainedProbs_2020.multiply(stable.not()));

var lc_final_2025 = post.classifyFromProbs(finalProbs_2025, class_bands, class_value);
var lc_final_2020 = post.classifyFromProbs(finalProbs_2020, class_bands, class_value);

// water merge on the labels (waterbody and dryriverbed)
lc_final_2020 = lc_final_2020.where(lc_final_2020.eq(18), 17);
lc_final_2025 = lc_final_2025.where(lc_final_2025.eq(18), 17);

// =========================================================================
// 1. AUXILIARY CONSTRAINTS ON PROBABILITIES
// =========================================================================
var constrainedProbs_2025 = adjustProbs(smoothedProbs_2025);
var constrainedProbs_2020 = adjustProbs(smoothedProbs_2020);

// Generate initial discrete labels
var lc20 = post.classifyFromProbs(constrainedProbs_2020, class_bands, class_value).toInt16();
var lc25 = post.classifyFromProbs(constrainedProbs_2025, class_bands, class_value).toInt16();

// Merge Water classes first (Dry riverbed 18 -> Waterbody 17)
lc20 = lc20.where(lc20.eq(18), 17);
lc25 = lc25.where(lc25.eq(18), 17);

// =========================================================================
// 2. ENFORCE CONSISTENCY RULES
// =========================================================================
function enforceConsistency(lc2020, lc2025, invalid, stableMask) {
  var codes = invalid.map(function(p) { return p[0] * 100 + p[1]; });
  var pair = lc2020.toInt16().multiply(100).add(lc2025.toInt16()).rename('pair');
  var flag = pair.remap(codes, ee.List.repeat(1, codes.length), 0).rename('flag');
  if (stableMask) { flag = flag.and(stableMask); }
  return {
    label: lc2020.where(flag.eq(1), lc2025).toInt16(),
    flag: flag,
    pair: pair
  };
}

//define the impossible transitions
// Rule groups: [class in 2020, class in 2025]
//Impossible transition
var impossible = [
  [2, 1],   // Secondary Forest -> Primary Forest
  [14, 1],  // Shrubland -> Primary Forest
  [13, 1],  // Grassland -> Primary Forest
  [12, 1],  // Other Cropland -> Primary Forest
  [15, 1],  // Cleared Land -> Primary Forest
  [16, 1],  // Built-up -> Primary Forest
  [17, 1]   // Waterbody -> Primary Forest
];
//Unlikely transition
//Blocked ONLY where CCDC shows STABLE (No break detected)
var unlikely = [
  [14, 2],  // Shrubland -> Secondary Forest (Fast regrowth; only keep if CCDC recorded a break)
  [13, 2],  // Grassland -> Secondary Forest
  [16, 13], // Built-up -> Grassland (Urban decay/abandonment on stable ground is usually spectral noise)
  [16, 14], // Built-up -> Shrubland
  [5, 1],   // Teak Plantation -> Primary Forest
  [6, 1]    // Eucalyptus Plantation -> Primary Forest
];

// Step 1: Impossible transitions (Applied everywhere)
var step1 = enforceConsistency(lc20, lc25, impossible);

// Step 2: Unlikely transitions (Applied ONLY on CCDC stable pixels)
var step2 = enforceConsistency(step1.label, lc25, unlikely, stable);

// Final persistent outputs
var lc_final_2020 = step2.label.rename('classification').toByte();
var lc_final_2025 = lc25.rename('classification').toByte();
var classVis = {
  min: 1,
  max: 17,
  palette: [
    '#00441B', // 1  Primary_forest
    '#35a23d', // 2  Secondary_forest
    '#006d6b', // 3  Mangrove_forest
    '#66C2A4', // 4  Coastal_forest
    '#8C6D31', // 5  Teak_plantation
    '#A6761D', // 6  Eucalyptus_plantation
    '#7FC97F', // 7  Coffe_agroforestry
    '#66A061', // 8  Cocoa_agroforestry
    '#41AB5D', // 9  Coconut_agroforestry
    '#A1D76A', // 10  Mixed_garden
    '#C7E9B4', // 11 Paddy_field
    '#FDD835', // 12 Other_cropland
    '#c4f298', // 13 Grassland
    '#C2A878', // 14 Shrubland
    '#d7d6d6', // 15 Cleared_land
    '#DB310D', // 16 Built-up
    '#0e8acc'  // 17 Water
  ]
};
Map.addLayer(lc_final_2020, classVis, 'Consistency Land Cover 2020')
Map.addLayer(lc_final_2025, classVis, 'Consistency Land Cover 2025')