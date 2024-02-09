## tpxpy - fast Timepix3 analysis using CUDA

This package can be used to process raw *.tpx3 files output from a Timepix3 camera. It is meant for use in few-photon imaging with an intensified Timepix camera. These systems output raw files in a [binary format](https://www.quantastro.bnl.gov/sites/default/files/2022-04/ASIServer%20TPX3%20manual%20V1.21.pdf), with a single photon detection being stored as a cluster of nearby pixel clicks. This package has functionality for parsing raw files, identifying clusters in an image based on spatiotemporal correlations, and analysis of the resulting data. In particular, it allows for photon positions to be calculated with arbitrary sub-pixel precision (whether this is useful is another question).

This code was written for use in a Timepix-based biphoton spectrometer, and so has analysis functions meant for this purpose, such as spectral calibration. Parsing and clustering functionality may be useful in more general contexts.

### Dependencies:
- `cupy-cuda12x` (requires a modern NVIDIA GPU)
- `numpy`
- `matplotlib`
- `scipy`
- `tqdm`

### Simple example
```python
from tpxpy.loader import TpxLoader, TpxImage

from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

fname = askopenfilename()

tpxl = TpxLoader()

# for best results, you should create a ToT calibration file, which is camera-specific. For example,
tpxl.generate_tot_calibration([fname,], 'tot_cal.txt') # can be generated from any .tpx3 image(s)
tpxl.set_tot_calibration('tot_cal.txt')

img = tpxl.load(fname)

hist2d = img.to_2d_image()
plt.imshow(hist2d)
plt.show()
```

If you encounter any errors, report them to the maintainer of the repository by [email](mailto:kjordan@uottawa.ca).

Related projects from our [lab](https://extremephotonics.com/):
- [PixGUI](https://github.com/baf57/PixGUI) and [tpx3_toolkit](https://github.com/baf57/tpx3_toolkit): A Python-based tool for ghost imaging, with more focus on spatial correlation analysis.
- [Spectral HOM Analysis Tool](https://github.com/k-m-jordan/JCEP-Spectral-HOM): A C++-based GUI for batch processing of Timepix files.

