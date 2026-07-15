# Robustness Evaluation Summary

**Corruption coverage:** 11 corruption types (gaussian_blur, gaussian_noise, grayscale, high_brightness, high_contrast, jpeg_compression, low_brightness, low_contrast, low_resolution, partial_crop, partial_occlusion), 33 total (type x severity) conditions.

**Clean baseline:** {'corruption': 'clean', 'severity': 0, 'param': nan, 'n_samples': 300, 'age_mae': 5.753019027709961, 'age_rmse': 8.097393844316231, 'interval_coverage': 0.77, 'mean_interval_width': 16.319618225097656, 'interval_coverage_calibrated': 0.8833333333333333, 'mean_interval_width_calibrated': 23.325881958007812, 'gender_accuracy': 0.9621212121212122, 'abstention_rate': 0.12, 'mean_confidence': 0.9409754276275635}

## gaussian_blur

   corruption  severity  param  n_samples  age_mae  age_rmse  interval_coverage  mean_interval_width  interval_coverage_calibrated  mean_interval_width_calibrated  gender_accuracy  abstention_rate  mean_confidence
gaussian_blur         1    0.8        300 5.834456  8.162527           0.753333            16.342356                      0.886667                       23.348616         0.961977         0.123333         0.941405
gaussian_blur         2    1.6        300 6.290961  8.998762           0.740000            16.208164                      0.850000                       23.214428         0.947170         0.116667         0.942109
gaussian_blur         3    2.6        300 7.454175 10.968940           0.666667            16.324965                      0.813333                       23.331226         0.950758         0.120000         0.941042

## gaussian_noise

    corruption  severity  param  n_samples   age_mae  age_rmse  interval_coverage  mean_interval_width  interval_coverage_calibrated  mean_interval_width_calibrated  gender_accuracy  abstention_rate  mean_confidence
gaussian_noise         1   0.03        300  7.727554 10.397207           0.643333            19.439066                      0.850000                       26.445330         0.954887         0.113333         0.940214
gaussian_noise         2   0.08        300 31.950894 36.376075           0.133333            29.102148                      0.206667                       36.108410         0.914729         0.140000         0.928564
gaussian_noise         3   0.15        300 47.575663 50.972857           0.060000            33.435623                      0.076667                       40.441883         0.918269         0.306667         0.862748

## low_resolution

    corruption  severity  param  n_samples  age_mae  age_rmse  interval_coverage  mean_interval_width  interval_coverage_calibrated  mean_interval_width_calibrated  gender_accuracy  abstention_rate  mean_confidence
low_resolution         1   0.50        300 6.129414  8.746009           0.746667            15.958013                      0.856667                       22.964273         0.954717         0.116667         0.942348
low_resolution         2   0.30        300 6.754054  9.856366           0.700000            15.990094                      0.840000                       22.996355         0.951128         0.113333         0.942794
low_resolution         3   0.15        300 8.632796 12.853087           0.673333            17.488594                      0.776667                       24.494856         0.931559         0.123333         0.930359

## jpeg_compression

      corruption  severity  param  n_samples  age_mae  age_rmse  interval_coverage  mean_interval_width  interval_coverage_calibrated  mean_interval_width_calibrated  gender_accuracy  abstention_rate  mean_confidence
jpeg_compression         1   40.0        300 5.782413  8.020980           0.736667            16.459267                      0.900000                       23.465530         0.962121         0.120000         0.942331
jpeg_compression         2   20.0        300 5.987887  8.209789           0.736667            17.343269                      0.906667                       24.349531         0.965779         0.123333         0.940889
jpeg_compression         3   10.0        300 7.716666 10.309330           0.653333            19.522043                      0.836667                       26.528305         0.943182         0.120000         0.939628

## low_brightness

    corruption  severity  param  n_samples  age_mae  age_rmse  interval_coverage  mean_interval_width  interval_coverage_calibrated  mean_interval_width_calibrated  gender_accuracy  abstention_rate  mean_confidence
low_brightness         1    0.7        300 6.000297  8.316185           0.776667            17.129560                      0.886667                       24.135820         0.958015         0.126667         0.939035
low_brightness         2    0.5        300 7.121172 10.114373           0.756667            18.856453                      0.880000                       25.862715         0.952381         0.160000         0.916333
low_brightness         3    0.3        300 9.339886 13.251573           0.680000            21.048128                      0.813333                       28.054394         0.895928         0.263333         0.872087

## high_brightness

     corruption  severity  param  n_samples  age_mae  age_rmse  interval_coverage  mean_interval_width  interval_coverage_calibrated  mean_interval_width_calibrated  gender_accuracy  abstention_rate  mean_confidence
high_brightness         1    1.3        300 6.050560  8.888395           0.760000            16.933794                      0.873333                       23.940056         0.932584         0.110000         0.943162
high_brightness         2    1.6        300 7.176367 10.776798           0.733333            18.667910                      0.860000                       25.674171         0.926923         0.133333         0.935260
high_brightness         3    2.0        300 8.581206 13.153642           0.713333            21.098511                      0.833333                       28.104773         0.912000         0.166667         0.913612

## low_contrast

  corruption  severity  param  n_samples  age_mae  age_rmse  interval_coverage  mean_interval_width  interval_coverage_calibrated  mean_interval_width_calibrated  gender_accuracy  abstention_rate  mean_confidence
low_contrast         1    0.7        300 5.796179  8.093449           0.810000            18.089884                      0.916667                       25.096146         0.953488         0.140000         0.931072
low_contrast         2    0.5        300 6.672685  9.132297           0.816667            21.444851                      0.913333                       28.451113         0.946939         0.183333         0.913560
low_contrast         3    0.3        300 9.694830 13.343280           0.786667            29.879114                      0.863333                       36.885376         0.879227         0.310000         0.857644

## high_contrast

   corruption  severity  param  n_samples  age_mae  age_rmse  interval_coverage  mean_interval_width  interval_coverage_calibrated  mean_interval_width_calibrated  gender_accuracy  abstention_rate  mean_confidence
high_contrast         1    1.4        300 5.885296  8.356665           0.746667            15.740109                      0.873333                       22.746370         0.950382         0.126667         0.942959
high_contrast         2    1.8        300 6.234938  8.819773           0.743333            15.941139                      0.860000                       22.947401         0.942966         0.123333         0.940170
high_contrast         3    2.4        300 6.713366  9.603893           0.736667            16.650408                      0.850000                       23.656670         0.937500         0.146667         0.933856

## grayscale

corruption  severity  param  n_samples  age_mae  age_rmse  interval_coverage  mean_interval_width  interval_coverage_calibrated  mean_interval_width_calibrated  gender_accuracy  abstention_rate  mean_confidence
 grayscale         1    0.4        300 5.719921  8.058322           0.766667            16.787178                      0.893333                       23.793440         0.961390         0.136667         0.934465
 grayscale         2    0.7        300 6.019387  8.412847           0.770000            17.826267                      0.896667                       24.832531         0.956175         0.163333         0.923858
 grayscale         3    1.0        300 6.433565  8.837109           0.793333            19.338308                      0.903333                       26.344572         0.931452         0.173333         0.914533

## partial_occlusion

       corruption  severity  param  n_samples   age_mae  age_rmse  interval_coverage  mean_interval_width  interval_coverage_calibrated  mean_interval_width_calibrated  gender_accuracy  abstention_rate  mean_confidence
partial_occlusion         1   0.10        300  7.964355 11.255455           0.666667            16.862835                      0.786667                       23.869097         0.892308         0.133333         0.933043
partial_occlusion         2   0.20        300 10.523433 14.291505           0.513333            16.665983                      0.663333                       23.672247         0.805243         0.110000         0.935929
partial_occlusion         3   0.35        300 12.674718 16.930611           0.446667            15.954705                      0.573333                       22.960970         0.687023         0.126667         0.933953

## partial_crop

  corruption  severity  param  n_samples   age_mae  age_rmse  interval_coverage  mean_interval_width  interval_coverage_calibrated  mean_interval_width_calibrated  gender_accuracy  abstention_rate  mean_confidence
partial_crop         1   0.10        300  6.619557  9.183446           0.740000            18.305098                      0.896667                       25.311357         0.956522         0.156667         0.926935
partial_crop         2   0.20        300  9.280079 12.763303           0.683333            22.069630                      0.800000                       29.075891         0.877049         0.186667         0.893998
partial_crop         3   0.35        300 10.014932 13.760816           0.676667            22.840307                      0.790000                       29.846569         0.791304         0.233333         0.880239
