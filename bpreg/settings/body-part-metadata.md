# Body Part Metadata 

The output of the Body Part Regression model is a JSON file, with lots of body part related information inside. 
To understand the calculations behind the individual tags, look at the thesis **"Body Part Regression for CT Volumes"** from Sarah Schuhegger. 
A short description for the individual tags can be found as well in this document. 

- cleaned slice scores: 
  - Cleaned version of the slice score (smoothed and outlier filtering). The slice scores are monotonously increasing with slice height, and similar anatomies get mapped to similar scores. The slice score of a slice can be used as guidance to know where we are in the human body. 
- unprocessed slice scores: 
  - Plain transformed results for the slice scores from the deep learning model. 
  - The results are transformed so that the expected score for the start of the pelvis is zero and for the end of the eyes is 100. 
- body part examined: 
  - For each body part: legs, pelvis, abdomen, chest, shoulder-neck, head the indices where the body part is visible are saved in a list. 
- body part examined tag: 
  - tag for examined body part. Examples: PELVIS, ABDOMEN, CHEST, NECK, HEAD, CHEST-ABDOMEN-PELVIS, NECK-CHEST, HEAD-NECK-CHEST-ABDOMEN-PELVIS
- look-up table: 
  - Table with expected slice scores for different anatomies. For example, for all vertebrae, the expected score is saved. 
- reverse z-ordering: 
  - 0/1. It is equal to one if the slice index decreases with patient height. And is zero, if the slice index increases with patient height. 
- valid z-spacing
  - 0/1. It is equal to zero if the slope of the slice score curve seems to be not natural. This tag can be used for basic data sanity checks. If the z-spacings seems to be invalid, then the volume is probably corrupted. 
- expected slope: 
  - Slope which is expected for the slice score curve. This slope is equal for all JSON files. 
- observed slope: 
  - Slope of the cleaned slice score curve from this volume.
- slope ratio: 
  - ratio between observed slope and expected slope. It is the relative deviation of the observed slope to the expected slope. 
- expected z-spacing: 
  - Expected z-spacing based on the expected slope for this volume. 
- z-spacing: 
  - Observed z-spacing for this volume 
- settings: 
  - slice score processing: 
  To clean the slice scores, the unprocessed slice scores are smoothed with a Gaussian kernel and outliers on the tails get cutted. For further explanation, look at the master thesis. 
    - transform min: value to transform to zero (expected score for pelvis-start)
    - transform max: value to transform to 100 (expected score for eyes-end)
    - slope mean: equal to "expected slope". 
    - tangential slope min: minimum valid tangential slope for tails (0.005-quantile)
    - tangential slope max: maximum valid tangential slope for tails (0.995-quantile)
    - r slope threshold: Threshold for "slope ratio" to declare the z-spacing as invalid
    - smoothing sigma: smoothing sigma in millimeter for Gaussian kernel, to create "cleaned slice scores"
    - background scores: Scores to filter, e.g. prediction of empty slices (all values in matrix are -1/-1000 HU)
  - body part examined dict
    - Dictionary which defines the landmark boundaries for the "body part examined" dictionary. 
  - body part examined tag 
    - body parts included: For z-range > 10 cm case. Landmarks which are inside the body part. 
    - distinct body parts: For z-range < 10 cm case. Distinct definition for body parts to be able to assign to every slice a body part and to use the most frequent body part at the "body part examined tag". 
    - min present landmarks: For z-range > 10 cm case. Minimum amount of landmarks from the body parts included dictionary, which need to be inside a volume to add the body-part to the "body part examined tag"



