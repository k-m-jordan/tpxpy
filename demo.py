import utils
from loader import TpxLoader
from analysis import TwinBeam, BiphotonSpectrum, yingwen_plot

import matplotlib.pyplot as plt
import numpy as np

dirname = "data/sinc 50x"
superres = 1 # scales the number of bins used for centroids; 1 = camera resolution (256x256); highly experimental
persist_cache = False

if __name__ == '__main__':

    tpxl = TpxLoader()
    tpxl.set_load_orientation(rotation=0)
    tpxl.set_tot_calibration("./tot_calibration.txt")
    tpxl.cache_all_files(dirname, use_existing_cache=True)

    # load a reference spectrum
    img = tpxl.load(dirname + '/scan_000000.tpx3')
    img.set_coincidence_window(0, 10)

    # create a mask for the images
    line1, line2 = img.fit_lines('horizontal', width_sigma=5, upscale_res=superres)
    beams = TwinBeam(img, line1, line2, superresolution=superres)

    # generate filter parameters (along the JSI diagonal) based on the reference
    bispec = BiphotonSpectrum(beams)
    bispec.load_calibration('./argon.calib.csv')
    filter_min, filter_max = bispec.filter_diagonal(5)

    delays, freqs, yplot = yingwen_plot(dirname, tpxl, line1, line2, superres, (0, 10), './argon.calib.csv', diag_filter=[filter_min, filter_max])

    plt.figure()
    plt.imshow(yplot, origin='lower', extent=[freqs[0], freqs[-1], delays[0], delays[-1]], aspect='auto')
    plt.xlabel('Frequency Difference [THz]')
    plt.ylabel('HOM Delay [um]')
    plt.savefig('./yingwen.png')

    # cleanup
    if not persist_cache:
        utils.clear_cluster_cache(dirname)