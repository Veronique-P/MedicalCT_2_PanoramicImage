# Panoramic Generator
Reviseted source code from paper [Reconstruction of Panoramic Dental Images Through Bézier Function Optimization](https://doi.org/10.3389/fbioe.2020.00794). Code borrowed and modified from https://github.com/paulojamorim .

## Prerequisites (Tested on Ubuntu 20.04):

`pip install h5py imageio matplotlib nibabel numpy scipy scikit-image Cython`

### Compile Cython code:

`python setup.py build_ext --inplace`

You might need (or not) to rename the files:
`mv interpolation.*.so interpolation.so`
`nv draw_bezier.*.so draw_bezier.so`

## Running

`python panoramic_generator.py file.nrrd [options]`

```
Options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output=OUTPUT
                        Output file (nifti default is panoramic.nii)
  -d DISTANCE, --distance=DISTANCE
                        Distance between the curves
  -n NCURVES, --ncurves=NCURVES
                        Number of curves between and after the found curve
  -p NPOINTS, --npoints=NPOINTS
                        Number of points (pixels) for each curve
  -g NCTRL_POINTS, --nctrl_points=NCTRL_POINTS
                        Number of bezier control points
  -t THRESHOLD, --threshold=THRESHOLD
                        Threshold used to determine the dental arcade
  -s, --skeleton        Generate skeleton image

```

## Citation

Amorim PHJ, Moraes TF, Silva JVL, Pedrini H and Ruben RB (2020) Reconstruction of Panoramic Dental Images Through Bézier Function Optimization. Front. Bioeng. Biotechnol. 8:794. doi: 10.3389/fbioe.2020.00794
