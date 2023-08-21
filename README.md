# SLAP (SLam but it's not About Poetry)
SLAP is a simple monocular SLAM implementation inspired by [twitchslam](https://github.com/geohot/twitchslam).
## Installation
Installation of the module is performed as follows:
```console
git clone git@github.com:pierre-reboud/SLAP.git
cd SLAP
pip install -r requirements.txt
pip install -e .
```

**Warning**: Additional modules might have to be installed, as requirements.txt is currently outdated.

Additionally, the libraries has to be manually installed:
* pangolin: Follow the [instructions](https://github.com/uoip/pangolin)

## Usage
1. Specify the desired parameters in the ```configs/program.json``` file.
2. Run the following console command:
```console
python main.py
```

## Todo
* Make Docker work
* Redo the Bundle Adjustment with Pybind in Cpp

## Sources and Bibliography

* Hartley, R., Zisserman, A. (2003). Multiple View Geometry in Computer Vision. New York, NY, USA: Cambridge University Press. ISBN: 0521540518 