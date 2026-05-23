# Geometry Heatmap v2 Decode Method Comparison

## Selected Method

- Selected decode method: `peak_weighted_centroid_w5`
- Selected window size: `5`
- Validation Keras accepted MAE: `4.1217 C`
- Validation Keras acceptance rate: `0.0426`
- Validation INT8 accepted MAE: `6.6232 C`
- Validation INT8 acceptance rate: `0.1064`
- Validation INT8 worst accepted error: `10.1459 C`
- Validation Keras-vs-INT8 temperature delta mean: `12.7276 C`
- Validation Keras-vs-INT8 tip delta mean: `12.4573 px`

## Split Summary

### train

| decode method | window | Keras accepted MAE | Keras acceptance | INT8 accepted MAE | INT8 acceptance | worst accepted | Keras vs INT8 temp delta mean | Keras vs INT8 tip delta mean | disagreement count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_w3 | 3 | 8.8269 | 0.1806 | 11.9319 | 0.1189 | 25.1601 | 12.2298 | 11.8435 | 28 |
| peak_weighted_centroid_w3 | 3 | 8.4242 | 0.1674 | 12.0530 | 0.1101 | 24.7311 | 12.2722 | 11.5198 | 25 |
| local_window_softargmax_w3 | 3 | 8.4242 | 0.1674 | 12.0530 | 0.1101 | 24.7311 | 12.2722 | 11.5198 | 25 |
| peak_weighted_centroid_w5 | 5 | 8.3638 | 0.1674 | 11.7985 | 0.1013 | 24.3141 | 12.2334 | 11.2767 | 25 |
| local_window_softargmax_w5 | 5 | 8.3638 | 0.1674 | 11.7985 | 0.1013 | 24.3141 | 12.2334 | 11.2767 | 25 |
| softargmax_w3 | 3 | nan | 0.0000 | nan | 0.0000 | nan | 4.2832 | 2.8675 | 0 |

### val

| decode method | window | Keras accepted MAE | Keras acceptance | INT8 accepted MAE | INT8 acceptance | worst accepted | Keras vs INT8 temp delta mean | Keras vs INT8 tip delta mean | disagreement count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| peak_weighted_centroid_w5 | 5 | 4.1217 | 0.0426 | 6.6232 | 0.1064 | 10.1459 | 12.7276 | 12.4573 | 5 |
| local_window_softargmax_w5 | 5 | 4.1217 | 0.0426 | 6.6232 | 0.1064 | 10.1459 | 12.7276 | 12.4573 | 5 |
| local_window_softargmax_w3 | 3 | 4.1208 | 0.0426 | 6.6117 | 0.1064 | 10.4095 | 12.9214 | 12.7758 | 5 |
| peak_weighted_centroid_w3 | 3 | 4.1208 | 0.0426 | 6.6117 | 0.1064 | 10.4095 | 12.9214 | 12.7758 | 5 |
| argmax_w3 | 3 | 4.1186 | 0.0426 | 6.6074 | 0.1064 | 10.6872 | 13.0142 | 13.2089 | 5 |
| softargmax_w3 | 3 | nan | 0.0000 | nan | 0.0000 | nan | 7.6349 | 3.6239 | 0 |

### test

| decode method | window | Keras accepted MAE | Keras acceptance | INT8 accepted MAE | INT8 acceptance | worst accepted | Keras vs INT8 temp delta mean | Keras vs INT8 tip delta mean | disagreement count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax_w3 | 3 | 11.3623 | 0.0847 | 9.2069 | 0.0678 | 17.8386 | 14.4421 | 13.9994 | 5 |
| peak_weighted_centroid_w5 | 5 | 11.1911 | 0.0847 | 6.3349 | 0.0508 | 8.7255 | 14.2704 | 13.2120 | 6 |
| local_window_softargmax_w5 | 5 | 11.1911 | 0.0847 | 6.3349 | 0.0508 | 8.7255 | 14.2704 | 13.2120 | 6 |
| local_window_softargmax_w3 | 3 | 11.2736 | 0.0847 | 6.3298 | 0.0508 | 8.7192 | 14.4166 | 13.5514 | 6 |
| peak_weighted_centroid_w3 | 3 | 11.2736 | 0.0847 | 6.3298 | 0.0508 | 8.7192 | 14.4166 | 13.5514 | 6 |
| softargmax_w3 | 3 | nan | 0.0000 | nan | 0.0000 | nan | 2.9164 | 3.4578 | 0 |

## Recommendation

- Recommended deployment decoder: `peak_weighted_centroid_w5` with window size `5`
- Validation Keras acceptance preserved? `True`
- Validation INT8 drift reduced relative to softargmax_w3? `False`
- Validation Keras accepted MAE at selection: `4.1217 C`
- Validation Keras acceptance rate at selection: `0.0426`
