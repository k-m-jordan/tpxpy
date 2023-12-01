## tpxpy - fast Timepix3 analysis using CUDA

This package can be used to process raw *.tpx3 files output from a Timepix3 camera. It is meant for use in few-photon imaging with an intensified Timepix camera. These systems output raw files in a [binary format](https://www.quantastro.bnl.gov/sites/default/files/2022-04/ASIServer%20TPX3%20manual%20V1.21.pdf), with a single photon detection being stored as a cluster of nearby pixel clicks. This package has functionality for parsing raw files, identifying clusters in an image based on spatiotemporal correlations, and analysis of the resulting data. In particular, it allows for photon positions to be calculated with arbitrary sub-pixel precision (whether this is useful is another question).

This code was written for use in a Timepix-based biphoton spectrometer, and so has analysis functions meant for this purpose, such as spectral calibration. Parsing and clustering functionality may be useful in more general contexts.

A brief example script is contained in `demo.py`. Additional functions can be found by browsing other files.

### Dependencies:
- `cupy-cuda12x` (requires a modern NVIDIA GPU)
- `numpy`
- `matplotlib`
- `scipy`
- `tqdm`

If you encounter any errors, report them to the maintainer of the repository by [email](mailto:kjordan@uottawa.ca).

Related projects from our [lab](https://extremephotonics.com/):
- [PixGUI](https://github.com/baf57/PixGUI) and [tpx3_toolkit](https://github.com/baf57/tpx3_toolkit): A Python-based tool for ghost imaging, with more focus on spatial correlation analysis.
- [Spectral HOM Analysis Tool](https://github.com/k-m-jordan/JCEP-Spectral-HOM): A C++-based GUI for batch processing of Timepix files.
