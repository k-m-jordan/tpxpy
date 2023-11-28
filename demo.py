import utils
from loader import TpxLoader
from analysis import TwinBeam, BiphotonSpectrum, yingwen_plot, gen_calibration

import matplotlib.pyplot as plt
import numpy as np

dirname = "E:/Projects/SpectralHOM/Timepix/2023-11-24 new alignment/sinc"
superres = 2 # scales the number of bins used for centroids; 1 = camera resolution (256x256); highly experimental
persist_cache = True

if __name__ == '__main__':

    tpxl = TpxLoader()
    tpxl.set_load_orientation(rotation=0)
    tpxl.set_tot_calibration("./tot_calibration.txt")
    tpxl.cache_all_files(dirname, use_existing_cache=True)

    calibration_name = './broadband.calib.csv'
    #gen_calibration(tpxl, './argon_calibration.tpx3', calibration_name, fit=[750.4, 763.5, 772.4, 794.8, 801.5, 810.4, 826.5, 840.8, 842.5, 852.1, 866.8])

    # load a reference spectrum
    img = tpxl.load(utils.all_tpx3_in_dir(dirname)[0])
    img.set_coincidence_window(0, 10)

    # create a mask for the images
    line1, line2 = img.fit_lines('horizontal', width_sigma=5, upscale_res=superres)
    beams = TwinBeam(img, line1, line2, superresolution=superres)

    # generate filter parameters (along the JSI diagonal) based on the reference
    bispec = BiphotonSpectrum(beams)
    bispec.load_calibration(calibration_name)
    diag_filter = bispec.filter_diagonal(5)

    delays, freqs, yplot = yingwen_plot(dirname, tpxl, line1, line2, superres, (0, 10), calibration_file=calibration_name, diag_filter=diag_filter, max_df=70805111622033.5)

    plt.figure(figsize=(12,6))
    plt.imshow(yplot, origin='lower', extent=[freqs[0], freqs[-1], delays[0], delays[-1]], aspect='auto')
    plt.xlabel('Frequency Difference [THz]')
    plt.ylabel('HOM Delay [um]')
    plt.savefig('./yingwen_sinc.png')

    # cleanup
    if not persist_cache:
        utils.clear_cluster_cache(dirname)