# README for demo_keras

#### Goal: document setting up tensorflow and keras interfaces for python 3.6 on intel and AMD gpu.  Run demo scripts in python and R.


Unfortunately, as of Jan 2019, the most recent tensorflow (1.12) does not work with the latest release of python (3.7) . Follow [this guide](https://github.com/plaidml/plaidml) for plaidml setup. Arranging versions between Keras, Tensorflow, and PlaidML will likely be a headache.

1. Activate plaidml virtual environment with prerequisites (pandas, scipy, keras, tensorflow==1.10.0, plaidml-keras, plaidbench) installed
2. Run plaidml setup
```
plaidml-setup
```
   1. Choose y for experimental support
   2. Select device (if available, metal + GPU has best performance on macOS)
   3. Support development by sending telemetry (y/n)
   4. Choose y to save
3. Benchmark or train on mobilenet to verify everything's running correctly
```
plaidbench keras mobilenet
```
or
```
plaidbench --batch-size 16 keras --train mobilenet
```

## References
* https://github.com/plaidml/plaidml
