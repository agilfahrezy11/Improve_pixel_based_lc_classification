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

/*
//Testing
var testArea = ee.Geometry.BBox(125.38, -8.76, 125.41, -8.73);
var testProbs = lc_prob_2025.clip(testArea);
// Fast local variance check
var localVarTest = post.localVariance(testProbs, class_bands, 5, 0.5);
print('Test Area Variance:', localVarTest.reduceRegion({
  reducer: ee.Reducer.percentile([50, 75, 90]),
  geometry: testArea,
  scale: 10,
  maxPixels: 1e7
}));
// Test smoothing locally
var testSmoothed = post.bayesianSmooth(testProbs, class_bands, smoothness, 5, 0.5);
var testLabel = post.classifyFromProbs(testSmoothed, class_bands, class_value);

Map.centerObject(testArea, 10)
Map.addLayer(testLabel, {min: 1, max: 18}, 'Smoothed Test Area');
*/

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
    '#E6550D', // 15 Cleared_land
    '#DB310D', // 16 Built-up
    '#2171B5', // 17 Waterbody
    '#9ECAE1'  // 18 Dry_riverbed
  ]
};

//Map.addLayer(landcover_2025, classVis, 'Raw Land Cover (argmax')
Map.addLayer(lc_smooth_2020, classVis, 'Smooth Land Cover (argmax) 2020')
Map.addLayer(lc_smooth_2025, classVis, 'Smooth Land Cover (argmax) 2025')

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
var finalProbs_2025 = smoothedProbs_2025;
var finalProbs_2020 = smoothedProbs_2025.multiply(stable)
  .add(smoothedProbs_2020.multiply(stable.not()));

var lc_final_2025 = post.classifyFromProbs(finalProbs_2025, class_bands, class_value);
var lc_final_2020 = post.classifyFromProbs(finalProbs_2020, class_bands, class_value);

// water merge on the labels, as you decided
lc_final_2020 = lc_final_2020.where(lc_final_2020.eq(18), 17);
lc_final_2025 = lc_final_2025.where(lc_final_2025.eq(18), 17);
var classVis = {
  min: 1,
  max: 17,
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
    '#0e8acc'  // 17 Water
  ]
};



//Map.addLayer(lc_final_2020, classVis, 'Consistency Land Cover 2020')
//Map.addLayer(lc_final_2025, classVis, 'Consistency Land Cover 2025')