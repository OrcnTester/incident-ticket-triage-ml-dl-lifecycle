# Alternative Models Report

This report evaluates models saved under `artifacts/alt_<target>_<model>/` using their persisted `split.json`.

## Target: category

| model | macro_f1 | weighted_f1 | accuracy | ops_notes |
|---|---:|---:|---:|---|
| svm | 0.9895 | 0.9894 | 0.9894 |  |
| rf | 0.9844 | 0.9842 | 0.9841 |  |
| nb | 0.9821 | 0.9815 | 0.9815 |  |

### Top model: svm
```
                  precision    recall  f1-score   support

      auth_issue      1.000     0.982     0.991        57
      data_issue      0.984     1.000     0.992        61
deployment_issue      1.000     1.000     1.000        63
         latency      1.000     0.954     0.976        65
          outage      0.971     1.000     0.986        68
   payment_issue      0.985     1.000     0.992        64

        accuracy                          0.989       378
       macro avg      0.990     0.989     0.990       378
    weighted avg      0.990     0.989     0.989       378

```

## Target: priority

| model | macro_f1 | weighted_f1 | accuracy | ops_notes |
|---|---:|---:|---:|---|
| rf | 0.5211 | 0.5925 | 0.6323 | P0/P1 recall=1.000, precision=0.822, severe=0.029 |
| nb | 0.5098 | 0.6204 | 0.6481 | P0/P1 recall=1.000, precision=0.853, severe=0.013 |
| svm | 0.5740 | 0.6328 | 0.6481 | P0/P1 recall=0.922, precision=0.863, severe=0.011 |

### Top model: rf
```
              precision    recall  f1-score   support

          P0      0.671     0.609     0.639        87
          P1      0.639     0.870     0.737       185
          P2      0.526     0.137     0.217        73
          P3      0.536     0.455     0.492        33

    accuracy                          0.632       378
   macro avg      0.593     0.518     0.521       378
weighted avg      0.616     0.632     0.593       378

```
