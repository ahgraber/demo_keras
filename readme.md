# README for demo_keras

#### Goal: document setting up tensorflow and keras interfaces for python 3.7 on intel and AMD gpu.  Run demo scripts in python and R.

#### Steps (Assuming Python 3.7.0 and pip 18.1):
1. Install tensorflow CPU from (https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl) using 
```
pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
```
   * At some point I'll try to compile a custom build to optimize the package for my computer
2. Install keras
```
pip3 install keras
```
3. Install plaidml and run setup
```
pip3 install plaidml-keras
plaidml-setup
```
4. Confirm it works
```
pip3 install plaidml-keras plaidbench
plaidbench keras mobilenet
```
