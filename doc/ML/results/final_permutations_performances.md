# Permutation Test Details (100 Iterations)

Detailed statistics for the permutation tests. Shows the comparison between the model's true accuracy vs. the accuracy obtained on shuffled labels (Null Distribution).

| Comparison          | Model   | True Accuracy   | P-Value   | Mean Null Acc   | Null Std   |
|:--------------------|:--------|:----------------|:----------|:----------------|:-----------|
| --- UMAP-based ---  |         |                 |           |                 |            |
| AD vs PSP           | GB      | 0.6137          | 0.0396    | 0.5003          | 0.0538     |
|                     | KNN     | 0.6823          | 0.0099    | 0.5181          | 0.0505     |
|                     | RF      | 0.6291          | 0.0198    | 0.5117          | 0.0551     |
| AD vs CBS           | GB      | 0.5609          | 0.4554    | 0.5569          | 0.0539     |
|                     | KNN     | 0.6051          | 0.1683    | 0.5699          | 0.0441     |
|                     | RF      | 0.5775          | 0.3267    | 0.563           | 0.0504     |
| PSP vs CBS          | GB      | 0.6152          | 0.0297    | 0.5128          | 0.0572     |
|                     | KNN     | 0.6152          | 0.0792    | 0.5194          | 0.0578     |
|                     | RF      | 0.5581          | 0.3564    | 0.5383          | 0.0518     |
| --- Atlas-based --- |         |                 |           |                 |            |
| AD vs PSP           | GB      | 0.7054          | 0.0099    | 0.5125          | 0.0509     |
|                     | KNN     | 0.6288          | 0.0099    | 0.5104          | 0.0444     |
|                     | RF      | 0.6823          | 0.0099    | 0.5133          | 0.0522     |
| AD vs CBS           | GB      | 0.604           | 0.2673    | 0.5854          | 0.0384     |
|                     | KNN     | 0.6822          | 0.0099    | 0.567           | 0.0447     |
|                     | RF      | 0.5707          | 0.3762    | 0.5656          | 0.0486     |
| PSP vs CBS          | GB      | 0.6638          | 0.0099    | 0.5223          | 0.0534     |
|                     | KNN     | 0.6448          | 0.0198    | 0.5201          | 0.0607     |
|                     | RF      | 0.6638          | 0.0198    | 0.5205          | 0.056      |