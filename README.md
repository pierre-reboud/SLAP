# SLAP (SLam but it's not About Poetry)
SLAP is a simple monocular SLAM implementation.
## Installation
Installation of the module is performed as follows:
```console
git clone git@github.com:pierre-reboud/SLAP.git
cd SLAP
pip install -r requirements.txt
pip install -e .
```
**Warning**: Additional modules might have to be installed, as requirements.txt is currently outdated.

Also, additional libraries have to be manually installed:
* pangolin: Follow the [instructions](https://github.com/uoip/pangolin)

## Usage
1. Choose one of the 2 desired parameters sets ```configs/<x>_main_config.json``` and apply this change in lines 24 and 25 of ```src/slap/utils/utils.py```. ```<x>``` can take the value "video" for using a dashboard camera video from a car (no ground truth poses), or "freiburg" for using the freiburg rgbd dataset (with ground truth poses).
2. Run the following console command:
```console
python main.py
```

## Todo
* Make Docker work
* Redo the Bundle Adjustment with Pybind in Cpp

## Sources and Bibliography

* Hartley, R., Zisserman, A. (2003). Multiple View Geometry in Computer Vision. New York, NY, USA: Cambridge University Press. ISBN: 0521540518 