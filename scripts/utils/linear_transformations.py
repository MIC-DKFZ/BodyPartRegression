import numpy as np 

def linear_transform(x, scale=1, min_value=0, max_value=1): 
    x = x - min_value
    x = x * scale /(max_value - min_value)

    return x 

def transform_0to100(score, lookuptable, min_value=np.nan): 
    if np.isnan(min_value): min_value = lookuptable["pelvis_start"]["mean"]
    max_value = lookuptable["eyes_end"]["mean"]
    
    score = linear_transform(score, scale=100, min_value=min_value, max_value=max_value)
    return score

def transform_lookuptable(lookuptable):
    lookup_copy = {key: {} for key in lookuptable}
    for key in lookuptable:
        lookup_copy[key]["mean"] = np.round(
            transform_0to100(lookuptable[key]["mean"], lookuptable), 3
        )
        lookup_copy[key]["std"] = np.round(
            transform_0to100(lookuptable[key]["std"], lookuptable, min_value=0.0), 3
        )

    return lookup_copy
