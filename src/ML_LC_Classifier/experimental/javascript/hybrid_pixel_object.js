///////// 1. Setup /////////
var testAOI = geometry;
var MAX_SIZE = 1024;        // Larger limits create oversized tile neighborhoods.
var USE_GUARD = true;      // true = extra within-object homogeneity guard
var GUARD_SD = 0.25;        // guard threshold
var W_DEFAULT = 0.65;        // default object weight
var STABILITY_TAU = 0.65;   // Threshold from Malinverni & Sanità (2026): CI <= tau is stable
var ADAPTIVE_W = true;      // true = dynamically scale object weight by stability
var ADAPTIVE_SENSITIVITY = 0.6; // How strongly instability damps object weight (0.0 to 1.0)
var APPLY_RULE_REFINEMENT = true; // true = apply intrinsic rule-based refinement for common confusion
var MERGE_WATER_RIVERBED = true;  // true = merge Dry riverbed (18) into Waterbody (17) after classification
var classIdToName = {
  1:  'Primary_forest',
  2:  'Secondary_forest',
  3:  'Mangrove_forest',
  4:  'Coastal_forest',
  5:  'Teak_plantation',
  6:  'Eucalyptus_plantation',
  7:  'Coffe_agroforestry',
  8:  'Cocoa_agroforestry',
  9:  'Coconut_agroforestry',
  10: 'Mixed_garden',
  11: 'Paddy_field',
  12: 'Other_cropland',
  13: 'Grassland',
  14: 'Shrubland',
  15: 'Cleared_land',
  16: 'Built-up',
  17: 'Waterbody',
  18: 'Dry_riverbed'
};
// Bayesian smoothing / object trust weights per class
var smoothnessTL = {
  'Primary_forest':        4.5,
  'Secondary_forest':      3.0,
  'Mangrove_forest':       3.0,
  'Coastal_forest':        3.5,
  'Teak_plantation':       1.2,
  'Eucalyptus_plantation': 1.2,
  'Coffe_agroforestry':    1.5,
  'Cocoa_agroforestry':    1.5,
  'Coconut_agroforestry':  1.5,
  'Mixed_garden':          1.2,
  'Paddy_field':           1.0,
  'Other_cropland':        1.0,
  'Grassland':             1.5,
  'Shrubland':             1.5,
  'Cleared_land':          1.5,
  'Built-up':              1.0,
  'Waterbody':             4.0,
  'Dry_riverbed':          0.1
};
var lc_prob_2025 = ee.Image('projects/gee-v1-510606/assets/TL_S2_class_probabilities_2025').divide(10000);
var classBands = lc_prob_2025.bandNames();
// Build the class IDs and weights from the probability band's names.
var nameToId = {};
for (var id in classIdToName) {
  nameToId[classIdToName[id]] = parseInt(id, 10);
}
var classBandNames = classBands.getInfo();
var classIdsList = [];
var wClientDict = {};
classBandNames.forEach(function(bName) {
  if (nameToId[bName] === undefined) {
    throw new Error('No class ID configured for probability band: ' + bName);
  }
  classIdsList.push(nameToId[bName]);

  var smoothness = smoothnessTL[bName] || 1.0;
  wClientDict[bName] = Math.min(Math.max(smoothness / 5.0, 0.2), 0.9);
});
var classIds = ee.List(classIdsList);
//print('Class bands:', classBands);
//define the segments
var pieces = [
  ee.Image('projects/gee-v1-510606/assets/snic/seg_central_timor'),
  ee.Image('projects/gee-v1-510606/assets/snic/seg_east_box'),
  ee.Image('projects/gee-v1-510606/assets/snic/seg_off_dili'),
  ee.Image('projects/gee-v1-510606/assets/snic/seg_west_box'),
  ee.Image('projects/gee-v1-510606/assets/snic/seg_east_timor')
];
/*
// Check the ID range of each piece (approximate, sampled at 100 m).
pieces.forEach(function(s, i) {
  print('piece ' + (i + 1) + ' ID range', s.select([0]).reduceRegion({
    reducer: ee.Reducer.minMax(), geometry: s.geometry(), scale: 100,
    maxPixels: 1e10, bestEffort: true, tileScale: 4
  }));
});
*/
// Give each piece its own ID range. ID_STRIDE must be larger than the biggest ID inside any
// piece, and 4 * ID_STRIDE + biggest ID must stay under 2,147,483,647 (int32 limit).
var ID_STRIDE = 300000000;
var snic = ee.ImageCollection(pieces.map(function(img, i) {
  return img.select([0]).add(i * ID_STRIDE).toInt32();
})).mosaic().rename('segment');
// ---------- 4. WEIGHTS AND CLASS IDS ----------
var wDict = ee.Dictionary(wClientDict);
var w = ee.Image.constant(classBands.map(function(b) {
  return wDict.get(b, W_DEFAULT);
})).rename(classBands);

// ---------- 5. FUNCTIONS ----------
function aggregate(img, reducer) {
  return img.addBands(snic).reduceConnectedComponents({
    reducer: reducer, labelBand: 'segment', maxSize: MAX_SIZE
  }).select(classBands);
}

function toClass(p) {
  var idx = p.toArray().arrayArgmax().arrayGet(0);
  return idx.remap(ee.List.sequence(0, classBands.size().subtract(1)), classIds)
    .rename('lulc').toUint8();
}

/**
 * Computes Winning class (W), Runner-up class (S), Confusion Index (CI = S / W),
 * and binary Stability Map based on Malinverni & Sanità (2026).
 */
function computeStability(probs) {
  var W = probs.reduce(ee.Reducer.max()).rename('W');
  var top1Idx = probs.toArray().arrayArgmax().arrayGet(0);
  var top1Class = top1Idx.remap(ee.List.sequence(0, classBands.size().subtract(1)), classIds).rename('winner');

  // Mask out winning band for each class to extract runner-up
  var maskedBands = classBandNames.map(function(bName) {
    var bImg = probs.select([bName]);
    var bId = nameToId[bName];
    return bImg.where(top1Class.eq(bId), -1.0).rename(bName);
  });
  var maskedStack = ee.Image.cat(maskedBands);
  
  var S = maskedStack.reduce(ee.Reducer.max()).max(0.0).rename('S');
  var top2Idx = maskedStack.toArray().arrayArgmax().arrayGet(0);
  var top2Class = top2Idx.remap(ee.List.sequence(0, classBands.size().subtract(1)), classIds).rename('runner_up');

  // Confusion Index: CI = S / W (clamped between 0 and 1)
  var CI = S.divide(W.max(1e-4)).min(1.0).rename('confusion_index');
  // Stability Map: 1 = Stable (reliable), 0 = Unstable (heterogeneous/ambiguous)
  var isStable = CI.lte(STABILITY_TAU).rename('is_stable').toUint8();

  return {
    W: W,
    S: S,
    winner: top1Class,
    runner_up: top2Class,
    CI: CI,
    is_stable: isStable
  };
}

function integrate(probs) {
  // 1. Compute object-level mean probabilities
  var obj = aggregate(probs, ee.Reducer.mean()).unmask(probs);

  // 2. Improvement 1: Stability evaluation at object level
  var objStability = computeStability(obj);

  // 3. Improvement 2: Adaptive object weight based on stability
  var wEff = w;
  if (ADAPTIVE_W) {
    // If the object is unstable / high confusion, decay object weight towards pixel weight
    var stabilityDamping = ee.Image(1.0).subtract(objStability.CI.multiply(ADAPTIVE_SENSITIVITY));
    wEff = w.multiply(stabilityDamping);
  }

  if (USE_GUARD) {
    var sd = aggregate(probs, ee.Reducer.stdDev()).reduce(ee.Reducer.max());
    wEff = wEff.multiply(sd.unmask(0).lte(GUARD_SD));
  }

  // 4. Convex blending
  var blend = obj.multiply(wEff).add(probs.multiply(ee.Image(1).subtract(wEff)));
  
  // Normalize blended probabilities so they sum to 1.0 across bands
  var blendTotal = blend.reduce(ee.Reducer.sum());
  blend = blend.divide(blendTotal.where(blendTotal.eq(0), 1.0));

  // 5. Final thematic labeling and quality assessment layers
  var finalStability = computeStability(blend);
  var lulc = finalStability.winner.rename('lulc').toUint8();
  var conf = finalStability.W.multiply(100).round().toUint8().rename('conf');
  var ciPct = finalStability.CI.multiply(100).round().toUint8().rename('ci');
  var runnerUp = finalStability.runner_up.toUint8().rename('runner_up');
  var isStable = finalStability.is_stable;

  return lulc.addBands([
    conf,
    ciPct,
    runnerUp,
    isStable
  ]);
}

/**
 * Intrinsic Rule-Based Refinement (Malinverni & Sanità 2026, Section 3.2.6)
 * Resolves systematic confusion between Cleared Land (15), Built-up (16), 
 * and Waterbody (17) / Dry Riverbed (18) using Winner (W), Runner-up (S), and CI.
 */
function refineRules(hybImage) {
  var lulc = hybImage.select('lulc');
  var runnerUp = hybImage.select('runner_up');
  var ci = hybImage.select('ci'); // 0 to 100%

  if (APPLY_RULE_REFINEMENT) {
    // Rule 1: False Hydrology Suppression (Catch BOTH Waterbody 17 and Dry Riverbed 18)
    // Urban roofs, asphalt, and bare construction clearings frequently trigger high
    // probabilities for Dry riverbed (18) or Waterbody (17).
    // If Winner is 17 OR 18, and Runner-up is Built-up (16) or Cleared land (15),
    // and competition is ambiguous (CI > 65%), reassign to the terrestrial runner-up class.
    var falseHydrology = lulc.eq(17).or(lulc.eq(18))
      .and(runnerUp.eq(16).or(runnerUp.eq(15)))
      .and(ci.gt(65));
    lulc = lulc.where(falseHydrology, runnerUp);

    // Rule 2: Cleared Land vs. Dry Riverbed Isolation Check
    // If Winner is Dry Riverbed (18) but Runner-up is Cleared Land (15) with high CI (> 75%),
    // and it occurs in non-hydrological terrain, keep Cleared Land.
    var falseRiverbed = lulc.eq(18)
      .and(runnerUp.eq(15))
      .and(ci.gt(75));
    lulc = lulc.where(falseRiverbed, runnerUp);
  }

  // Rule 3: Merge Dry riverbed (18) into Waterbody (17) post-classification
  // This mirrors the previous workflow (lc.where(lc.eq(18), 17)) without polluting
  // the pre-classification spectral probability distributions.
  if (MERGE_WATER_RIVERBED) {
    lulc = lulc.where(lulc.eq(18), 17);
  }

  return hybImage.addBands(lulc.rename('lulc_refined'), null, true);
}
// ---------- 6. TEST ON A SMALL AREA ----------
var pix2025_raw = toClass(lc_prob_2025);
var pix2025 = MERGE_WATER_RIVERBED ? pix2025_raw.where(pix2025_raw.eq(18), 17) : pix2025_raw;

var hyb2025_raw = integrate(lc_prob_2025);
var hyb2025 = refineRules(hyb2025_raw);

var holes = snic.mask().and(aggregate(lc_prob_2025, ee.Reducer.mean()).select(0).mask().not()).selfMask();
var classPalette = [
  '#00441B', // 1: Primary_forest
  '#238B45', // 2: Secondary_forest
  '#006D2C', // 3: Mangrove_forest
  '#66C2A4', // 4: Coastal_forest
  '#f08a0d', // 5: Teak_plantation
  '#eac582', // 6: Eucalyptus_plantation
  '#6b4106', // 7: Coffe_agroforestry
  '#66A061', // 8: Cocoa_agroforestry
  '#bac615', // 9: Coconut_agroforestry
  '#A1D76A', // 10: Mixed_garden
  '#756604', // 11: Paddy_field
  '#f7cc0b', // 12: Other_cropland
  '#D9EF8B', // 13: Grassland
  '#C2A878', // 14: Shrubland
  '#d7d6d6', // 15: Cleared_land
  '#DB310D', // 16: Built-up
  '#0e8acc', // 17: Waterbody
  '#9ECAE1'  // 18: Dry_riverbed
];
var viz = {min: 1, max: 18, palette: classPalette};
Map.centerObject(testAOI, 12);
Map.addLayer(pix2025, viz, 'Pixel-only 2025', false);
Map.addLayer(hyb2025.select('lulc'), viz, 'Hybrid 2025 (Raw LULC)', false);
Map.addLayer(hyb2025.select('lulc_refined'), viz, 'Hybrid 2025 (Refined LULC)');
Map.addLayer(hyb2025.select('is_stable').clip(testAOI), {min: 0, max: 1, palette: ['#D7191C', '#1A9641']}, 'Stability Map (Green=Stable, Red=Unstable)', false);
Map.addLayer(hyb2025.select('ci').clip(testAOI), {min: 0, max: 100, palette: ['#1A9641', '#FFFFBF', '#D7191C']}, 'Confusion Index % (S/W)', false);
Map.addLayer(hyb2025.select('conf').clip(testAOI), {min: 0, max: 100}, 'Confidence % (Winner)', false);
Map.addLayer(hyb2025.select('runner_up').clip(testAOI), viz, 'Runner-up Class', false);
Map.addLayer(holes.clip(testAOI), {palette: ['red']}, 'Holes (MAX_SIZE exceeded)', false);

/*
print('Share of test area covered by segments:', snic.mask().reduceRegion({
  reducer: ee.Reducer.mean(), geometry: testAOI, scale: 10, maxPixels: 1e9, tileScale: 4
}));

print('Share of test area classified as Stable (CI <= ' + STABILITY_TAU + '):', hyb2025.select('is_stable').reduceRegion({
  reducer: ee.Reducer.mean(), geometry: testAOI, scale: 10, maxPixels: 1e9, tileScale: 4
}));

// Evaluate impact of rule refinement
var modifiedByRules = hyb2025.select('lulc').neq(hyb2025.select('lulc_refined'));
print('Share of test area adjusted by rule refinement:', modifiedByRules.reduceRegion({
  reducer: ee.Reducer.mean(), geometry: testAOI, scale: 10, maxPixels: 1e9, tileScale: 4
}));
*/
var aoi = ee.FeatureCollection('projects/ee-rg2v2/assets/AOI_Timor_Leste_10km').geometry();
Export.image.toDrive({
  image: hyb2025.select('lulc_refined'),
  description: 'LC_S2_2025_hybrid',
  folder: 'AFCLIM_EE',
  fileNamePrefix: 'LC_S2_2025_hybrid',
  region: aoi,
  scale: 10,
  crs: 'EPSG:32751',
  maxPixels: 1e13
});
