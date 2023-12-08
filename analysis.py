from tpxpy import utils
from tpxpy.loader import TpxLoader, TpxImage, TPX_SIZE

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.signal import find_peaks, peak_widths, savgol_filter

from numpy.fft import fft, fftshift, fftfreq, fft2, ifft2, ifftshift

from typing import Literal

from tqdm import tqdm

import os

def gen_calibration(loader : TpxLoader, fname, output, fit, orientation:Literal['horizontal', 'vertical']='horizontal', superresolution : int = 1):
    img = loader.load(fname)

    line1, line2 = img.fit_lines(orientation)
    beams = TwinBeam(img, line1, line2, superresolution, ignore_coincs=True)

    bispec = BiphotonSpectrum(beams)

    _, _, spec1, spec2 = bispec.singles_spectrum()

    peaks1 = find_peaks(spec1, height=np.mean(spec1), distance=3)[0]
    peaks2 = find_peaks(spec2, height=np.mean(spec2), distance=3)[0]

    if len(peaks1) != len(fit):
        raise RuntimeError(f"Found {len(peaks1)} peaks in line 1, expected {len(fit)}")
    if len(peaks2) != len(fit):
        raise RuntimeError(f"Found {len(peaks2)} peaks in line 2, expected {len(fit)}")

    with open(output, 'w') as f:
        f.write('Wavelength [nm], Bin 1 [px], Bin 2 [px]\n')
        for ix in range(len(fit)):
            f.write(f"{fit[ix]}, {peaks1[ix]/superresolution}, {peaks2[ix]/superresolution}\n")

class TwinBeam:

    # superresolution: how finely to bin the clusters (=1 for camera resolution (256x256))
    def __init__(self, image : TpxImage, channel_a_mask : np.ndarray = None, channel_b_mask : np.ndarray = None, superresolution : int = 1, ignore_coincs=False):
        if channel_a_mask is None and channel_b_mask is None:
            channel_a_mask, channel_b_mask = image.fit_lines(upscale_res=superresolution)

        if image.num_coincidences() == 0 and not ignore_coincs:
            raise ("Image has no coincidences; make sure coincidence window is set before calling TwinBeam()")
    
        sz = TPX_SIZE * superresolution

        if channel_a_mask.shape != (sz,sz) or channel_b_mask.shape != (sz,sz):
            raise ValueError(f"Channel masks must be of size ({sz},{sz})")

        self._image = image
        self._mask_a = channel_a_mask
        self._mask_b = channel_b_mask
        self._superres = superresolution
        self._size = sz
        self._coincs = self._image._coincidence_indices

        self._x_bins_float = self._image._centroid_x * superresolution
        self._y_bins_float = self._image._centroid_y * superresolution

        self._x_bins = np.floor(self._x_bins_float).astype(int)
        self._y_bins = np.floor(self._y_bins_float).astype(int)
        
        if not ignore_coincs:
            self._coincs_aa = np.logical_and(
                self._mask_a[self._x_bins[self._coincs[:,0]], self._y_bins[self._coincs[:,0]]],
                self._mask_a[self._x_bins[self._coincs[:,1]], self._y_bins[self._coincs[:,1]]]
            )
            self._coincs_bb = np.logical_and(
                self._mask_b[self._x_bins[self._coincs[:,0]], self._y_bins[self._coincs[:,0]]],
                self._mask_b[self._x_bins[self._coincs[:,1]], self._y_bins[self._coincs[:,1]]]
            )
            self._coincs_ab = np.logical_and(
                self._mask_a[self._x_bins[self._coincs[:,0]], self._y_bins[self._coincs[:,0]]],
                self._mask_b[self._x_bins[self._coincs[:,1]], self._y_bins[self._coincs[:,1]]]
            )
            self._coincs_ba = np.logical_and(
                self._mask_b[self._x_bins[self._coincs[:,0]], self._y_bins[self._coincs[:,0]]],
                self._mask_a[self._x_bins[self._coincs[:,1]], self._y_bins[self._coincs[:,1]]]
            )

    def coincidence_indices(self, channels:Literal["ab", "aa", "bb"] = "ab"):
        if channels == "ab":
            coinc1_ab = self._coincs[self._coincs_ab,0]
            coinc2_ab = self._coincs[self._coincs_ab,1]
            coinc1_ba = self._coincs[self._coincs_ba,0]
            coinc2_ba = self._coincs[self._coincs_ba,1]

            coinc1 = np.concatenate((coinc1_ab, coinc2_ba))
            coinc2 = np.concatenate((coinc2_ab, coinc1_ba))

            return coinc1, coinc2

        elif channels == "aa":
            return self._coincs[self._coincs_aa, 0], self._coincs[self._coincs_aa, 1]

        elif channels == "bb":
            return self._coincs[self._coincs_bb, 0], self._coincs[self._coincs_bb, 1]
    
        else:
            raise ValueError("channels must be either 'aa', 'bb', or 'ab'")
    
    def num_coincidences(self, channels:Literal["ab", "aa", "bb"] = "ab"):
        coinc1, _ = self.coincidence_indices(channels=channels)
        return len(coinc1)

    def coinc_image(self, channels:Literal["ab", "aa", "bb"] = "ab"):
        coinc1, coinc2 = self.coincidence_indices(channels=channels)
        x1 = self._x_bins[coinc1]
        y1 = self._y_bins[coinc1]
        x2 = self._x_bins[coinc2]
        y2 = self._y_bins[coinc2]

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))

        hist, _, _ = np.histogram2d(x, y, bins=self._size, range=[[0, self._size], [0, self._size]])
        return hist

    def correlation_image(self, type:Literal["x", "y"], channels:Literal["ab","aa","bb"]="ab"):
        coinc1, coinc2 = self.coincidence_indices(channels=channels)
        
        if type == "x":
            x1 = self._x_bins[coinc1]
            x2 = self._x_bins[coinc2]
            hist, _, _ = np.histogram2d(x1, x2, bins=self._size, range=[[0, self._size], [0, self._size]])
            return hist
        
        elif type == "y":
            y1 = self._y_bins[coinc1]
            y2 = self._y_bins[coinc2]
            hist, _, _ = np.histogram2d(y1, y2, bins=self._size, range=[[0, TPX_SIZE], [0, TPX_SIZE]])
            return hist
        
        else:
            return ValueError("type must be 'x' or 'y'")
        
class BiphotonCalibration:

    def __init__(self, pixel_to_wl_1 : callable = None, wl_to_pixel_1 : callable = None,
                       pixel_to_wl_2 : callable = None, wl_to_pixel_2 : callable = None):
        self._is_trivial = (pixel_to_wl_1 is None or wl_to_pixel_1 is None or pixel_to_wl_2 is None or wl_to_pixel_2 is None)

        if self._is_trivial: # +1 is so that no bin gives a wavelength of zero
            self._pixel_to_wl_1 = lambda x: x+1
            self._wl_to_pixel_1 = lambda x: x-1
            self._pixel_to_wl_2 = lambda x: x+1
            self._wl_to_pixel_2 = lambda x: x-1
        else:
            self._pixel_to_wl_1 = pixel_to_wl_1
            self._wl_to_pixel_1 = wl_to_pixel_1
            self._pixel_to_wl_2 = pixel_to_wl_2
            self._wl_to_pixel_2 = wl_to_pixel_2

    def is_trivial(self) -> bool:
        return self._is_trivial
        
class BiphotonSpectrum:

    def __init__(self, beams : TwinBeam):
        self._beams = beams
        self._image = self._beams._image
        self._size = self._beams._size
        self._calibration = BiphotonCalibration()
        self._diag_min = -np.inf # filter min and max along the diagonal direction, in Hz
        self._diag_max = np.inf

        # determine the orientation based on how the masks are oriented
        mask_coords = np.argwhere(beams._mask_a)
        x_min, y_min = mask_coords.min(axis=0)
        x_max, y_max = mask_coords.max(axis=0)
        
        width = x_max - x_min
        height = y_max - y_min

        self._horizontal = (width > height)
        if self._horizontal:
            self._par_axis = 0
            self._perp_axis = 1
        else:
            self._par_axis = 1
            self._perp_axis = 0

    def load_calibration(self, fname : str):
        calibration_data = np.loadtxt(fname, delimiter=',', skiprows=1)
        wls = calibration_data[:,0] * 1e-9 # meters
        bins1 = calibration_data[:,1]
        bins2 = calibration_data[:,2]

        fit1 = linregress(bins1, wls)
        fit2 = linregress(bins2, wls)
        
        pix_to_wl_1 = lambda p: (fit1.slope * p + fit1.intercept)
        pix_to_wl_2 = lambda p: (fit2.slope * p + fit2.intercept)

        wl_to_pix_1 = lambda f: ((f - fit1.intercept)/fit1.slope)
        wl_to_pix_2 = lambda f: ((f - fit2.intercept)/fit2.slope)

        self._calibration = BiphotonCalibration(pix_to_wl_1, wl_to_pix_1, pix_to_wl_2, wl_to_pix_2)

    def filter_diagonal(self, num_sigma = np.inf):
        self._diag_min = -np.inf
        self._diag_max = np.inf

        _, hist_sum, _, f_sum, _, _ = self.diagonal_projections()
        
        peak_ix = find_peaks(hist_sum, height=np.max(hist_sum)*0.9)[0][0]
        f0 = f_sum[peak_ix]
        _, _, f1, f2 = peak_widths(hist_sum, [peak_ix], rel_height=0.5)
        f_sigma = (f2-f1)*np.mean(np.diff(f_sum))/(2*np.sqrt(2*np.log(2)))
        f_sigma = f_sigma[0]

        self._diag_min = f0 - f_sigma*num_sigma
        self._diag_max = f0 + f_sigma*num_sigma

        return self._diag_min, self._diag_max

    def num_singles(self):
        mask_a = self._beams._mask_a
        mask_b = self._beams._mask_b
        x_bins = self._beams._x_bins
        y_bins = self._beams._y_bins
        singles = np.arange(self._image.num_clusters())
        
        clusters_in_a = np.argwhere(mask_a[x_bins[singles], y_bins[singles]])
        clusters_in_b = np.argwhere(mask_b[x_bins[singles], y_bins[singles]])

        return len(clusters_in_a), len(clusters_in_b)

    def singles_spectrum(self, type:Literal["frequency","wavelength"]="wavelength"):
        mask_a = self._beams._mask_a
        mask_b = self._beams._mask_b
        x_bins = self._beams._x_bins
        y_bins = self._beams._y_bins
        singles = np.arange(self._image.num_clusters())

        superresolution = (self._beams._superres)
        
        clusters_in_a = np.argwhere(mask_a[x_bins[singles], y_bins[singles]])
        clusters_in_b = np.argwhere(mask_b[x_bins[singles], y_bins[singles]])

        # coordinate parallel to lines
        par_pixels : np.ndarray
        if self._horizontal:
            par_pixels = x_bins.astype(np.float64) / superresolution
        else:
            par_pixels = y_bins.astype(np.float64) / superresolution

        pixels = np.linspace(-0.5/superresolution,TPX_SIZE-0.5/superresolution,self._size+1)
        par_indep_a = self._calibration._pixel_to_wl_1(pixels)
        par_indep_b = self._calibration._pixel_to_wl_2(pixels)

        bins_a = self._calibration._pixel_to_wl_1(par_pixels)
        bins_b = self._calibration._pixel_to_wl_2(par_pixels)

        c = utils.c

        if type == "wavelength":
            pass # nothing to do
        elif type == "frequency":
            par_indep_a = c / par_indep_a
            par_indep_b = c / par_indep_b

            bins_a = c / bins_a
            bins_b = c / bins_b
        else:
            raise ValueError("type must be either 'frequency' or 'wavelength'")
    
        par_indep_a = np.sort(par_indep_a)
        par_indep_b = np.sort(par_indep_b)

        hist_a, _ = np.histogram(bins_a[clusters_in_a], bins=par_indep_a)
        hist_b, _ = np.histogram(bins_b[clusters_in_b], bins=par_indep_b)

        # we want bin centers, not edges
        indep_a = (par_indep_a[1:]+par_indep_a[:-1])/2
        indep_b = (par_indep_b[1:]+par_indep_b[:-1])/2

        return indep_a, indep_b, hist_a, hist_b
    
    def joint_spectrum(self, channels:Literal["ab","aa","bb"]="ab", type:Literal["wavelength", "frequency"]="wavelength"):
        # determine the wavelengths/frequencies
        x_bins = self._beams._x_bins_float
        y_bins = self._beams._y_bins_float

        superresolution = (self._beams._superres)

        if self._horizontal:
            par_pixels = x_bins.astype(np.float64) / superresolution
        else:
            par_pixels = y_bins.astype(np.float64) / superresolution

        pixels = np.linspace(-0.5/superresolution,TPX_SIZE-0.5/superresolution,self._size+1)

        par_indep_a = self._calibration._pixel_to_wl_1(pixels)
        par_indep_b = self._calibration._pixel_to_wl_2(pixels)

        bins_a = self._calibration._pixel_to_wl_1(par_pixels)
        bins_b = self._calibration._pixel_to_wl_2(par_pixels)

        c = utils.c

        if type == "wavelength":
            pass # nothing to do
        elif type == "frequency":
            minwl_a = par_indep_a[0]
            maxwl_a = par_indep_a[-1]
            minwl_b = par_indep_b[0]
            maxwl_b = par_indep_b[-1]

            minf_a = c/maxwl_a
            maxf_a = c/minwl_a
            minf_b = c/maxwl_b
            maxf_b = c/minwl_b

            par_indep_a = np.linspace(minf_a, maxf_a, self._size+1)
            par_indep_b = np.linspace(minf_b, maxf_b, self._size+1)

            bins_a = c / bins_a
            bins_b = c / bins_b
        else:
            raise ValueError("type must be either 'frequency' or 'wavelength'")
    
        par_indep_a = np.sort(par_indep_a)
        par_indep_b = np.sort(par_indep_b)

        coinc1, coinc2 = self._beams.coincidence_indices(channels=channels)

        sum_bins = bins_b[coinc2] + bins_a[coinc1]
        coincs_within_filter = np.logical_and((sum_bins > self._diag_min), (sum_bins < self._diag_max))

        coinc1 = coinc1[coincs_within_filter]
        coinc2 = coinc2[coincs_within_filter]

        image : np.ndarray
        if channels == 'aa':
            image, _, _ = np.histogram2d(bins_a[coinc1], bins_a[coinc2], bins=[par_indep_a, par_indep_b])
        elif channels == 'bb':
            image, _, _ = np.histogram2d(bins_b[coinc1], bins_b[coinc2], bins=[par_indep_a, par_indep_b])
        elif channels == 'ab':
            image, _, _ = np.histogram2d(bins_a[coinc1], bins_b[coinc2], bins=[par_indep_a, par_indep_b])
        else:
            raise ValueError("channels must be either 'ab' or 'aa' or 'bb'")

        return par_indep_a, par_indep_b, image

    def jsi_fft(self, channels:Literal["ab","aa","bb"]="ab", type:Literal["wavelength", "frequency"]="frequency"):
        f1, f2, jsi = self.joint_spectrum(channels=channels, type=type)
        ft = fftshift(fft2(jsi))
        t1 = fftshift(fftfreq(len(f1), np.mean(np.diff(f1))))
        t2 = fftshift(fftfreq(len(f2), np.mean(np.diff(f2))))
        return t1, t2, ft
    
    def diagonal_projections(self, f_diff_edges=None, f_sum_edges=None):
        x_bins = self._beams._x_bins
        y_bins = self._beams._y_bins

        superresolution = (self._beams._superres)

        # coordinate parallel to lines
        par_pixels : np.ndarray
        if self._horizontal:
            par_pixels = x_bins.astype(np.float64) / superresolution
        else:
            par_pixels = y_bins.astype(np.float64) / superresolution

        bins_a = self._calibration._pixel_to_wl_1(par_pixels)
        bins_b = self._calibration._pixel_to_wl_2(par_pixels)

        c = utils.c

        bins_a = c / bins_a
        bins_b = c / bins_b

        coincs_a, coincs_b = self._beams.coincidence_indices()

        sum_bins = bins_b[coincs_b] + bins_a[coincs_a]
        coincs_within_filter = np.logical_and((sum_bins > self._diag_min), (sum_bins < self._diag_max))

        coincs_a = coincs_a[coincs_within_filter]
        coincs_b = coincs_b[coincs_within_filter]

        hist_diff = None
        diff_edges = None
        hist_sum = None
        sum_edges = None

        if f_diff_edges is None: 
            hist_diff, diff_edges = np.histogram(bins_a[coincs_a] - bins_b[coincs_b], bins=(self._size*2 - 1))
        else:
            hist_diff, diff_edges = np.histogram(bins_a[coincs_a] - bins_b[coincs_b], bins=f_diff_edges)
        
        if f_sum_edges is None:
            hist_sum, sum_edges = np.histogram(bins_a[coincs_a] + bins_b[coincs_b], bins=(self._size*2 - 1))
        else:
            hist_sum, sum_edges = np.histogram(bins_a[coincs_a] + bins_b[coincs_b], bins=f_sum_edges)

        return hist_diff, hist_sum, (diff_edges[1:] + diff_edges[:-1])/2, (sum_edges[1:] + sum_edges[:-1])/2, diff_edges, sum_edges

    def hom_hologram(self, channels:Literal['aa','bb','ab']='ab', savgol_len=21, savgol_order=7, sideband_width=2):
        f1, f2, _ = self.joint_spectrum(channels=channels, type='frequency')
        t1, t2, fft = self.jsi_fft(channels=channels, type='frequency')

        size = fft.shape[0]

        # sum along diagonal of FFT
        proj = np.zeros(2*size-1)
        norm = np.zeros(2*size-1)
        for row in range(size):
            proj[size-row-1:2*size-row-1] += np.abs(fft[row])
            norm[size-row-1:2*size-row-1] += 1

        proj /= norm
        proj = savgol_filter(proj, int(savgol_len*(size/TPX_SIZE)), savgol_order)

        proj_peaks = find_peaks(proj, height=max(proj)*0.3, rel_height=0.5)[0]
        proj_peak_widths = peak_widths(proj, proj_peaks, rel_height=0.5)[0]

        hologram_peak = proj_peaks[0]
        hologram_width = proj_peak_widths[0]

        mask_vals_x, mask_vals_y = np.mgrid[0:size, 0:size]
        mask_vals = (size-mask_vals_x) + mask_vals_y
        mask = np.logical_and(mask_vals < hologram_peak + sideband_width*hologram_width, mask_vals > hologram_peak - sideband_width*hologram_width)

        fft[np.logical_not(mask)] = 0

        return f1, f2, ifft2(ifftshift(fft))

def fit_osc(y, norm_freq):
    x = np.arange(len(y))
    
    # used to fit Yingwen plots (also fits frequency)
    def osc_fit_func_nofreq(x, amp, freq, phase, offset):
        return amp*np.cos(2*np.pi*freq*x + phase) + offset
    
    # used to fit Yingwen plots
    def osc_fit_func(x, amp, phase, offset):
        return amp*np.cos(2*np.pi*norm_freq*x + phase) + offset

    # use FFT to guess frequency
    phase_y = np.angle(fftshift(fft(y))[len(y)//2:])
    fft_y = np.abs(fftshift(fft(y))[len(y)//2:])
    fft_f = fftshift(fftfreq(len(y)))[len(y)//2:]
    fft_y[0] = 0
    ix = np.argwhere(fft_y == np.max(fft_y))[0][0]
    f0 = fft_f[ix]
    phase0 = phase_y[ix]

    # fit once to get better guesses (without using the known frequency)
    param_base, _ = curve_fit(osc_fit_func_nofreq, x, y, p0=(np.max(y)/2, f0, phase0, np.mean(y)), maxfev=100000)

    # now fit again to match the analytic frequency
    param, cov = curve_fit(osc_fit_func, x, y, p0=(param_base[0], param_base[1], param_base[2]), maxfev=100000)

    fit_y = osc_fit_func(x, param[0], param[1], param[2])
    residual = np.sum(np.abs(y - fit_y))/np.sum(0.01+np.sqrt(y))

    # ensure amplitudes are positive
    if param[0] < 0:
        param[0] = -param[0]
        param[1] += np.pi
    
    param[1] %= (2*np.pi)

    return param, cov, residual

class SRHomScan:
    def __init__(self, dirname, loader : TpxLoader, mask_a, mask_b, superresolution=1, coincidence_window = (0, 10), calibration_file = None, diag_filter = [-np.inf, np.inf]):
        file_list = utils.all_tpx3_in_dir(dirname)
        lines = []
        freqs = None

        has_config_file = os.path.exists(dirname + '/scan.config.txt')
        travel_dist : float
        if has_config_file:
            with open(dirname + '/scan.config.txt', "r") as f:
                file_lines = f.readlines()
                config_info = [l[:-1].split(':') for l in file_lines]
                stage_start = None
                stage_end = None
                for l in config_info:
                    if l[0] == 'Stage start (mm)':
                        stage_start = float(l[1])*1e-3
                    elif l[0] == 'Stage stop (mm)':
                        stage_end = float(l[1])*1e-3
                
                travel_dist = stage_end - stage_start
        
        if not has_config_file:
            print(f"Warning: directory {dirname} does not contain a scan.config.txt file; cannot determine HOM delays")

        for f in tqdm(file_list, desc="Processing SRHom Scan"):
            img = loader.load(f)
            img.set_coincidence_window(coincidence_window[0], coincidence_window[1])

            beams = TwinBeam(img, mask_a, mask_b, superresolution=superresolution)
            bispec = BiphotonSpectrum(beams)

            if calibration_file is not None:
                bispec.load_calibration(calibration_file)

            min_wl = min(bispec._calibration._pixel_to_wl_1(0), bispec._calibration._pixel_to_wl_2(0))
            max_wl = max(bispec._calibration._pixel_to_wl_1(TPX_SIZE-1), bispec._calibration._pixel_to_wl_2(TPX_SIZE-1))

            min_f = utils.c/max_wl
            max_f = utils.c/min_wl
            df = max_f - min_f
            
            bispec._diag_min = diag_filter[0]
            bispec._diag_max = diag_filter[1]

            hist_diff = None
            f_diff = None

            f_diff_edges = np.linspace(-df, df, bispec._size)

            hist_diff, _, f_diff, _, f_diff_edges, _ = bispec.diagonal_projections(f_diff_edges=f_diff_edges)

            lines.append(hist_diff)
            freqs = f_diff
        
        lines = np.array(lines)

        delays = None
        if has_config_file:
            delays = np.linspace(0, 1, len(file_list)) * travel_dist
        else:
            delays = np.arange(0, len(file_list))

        self._dirname = dirname
        self._loader = loader
        self._mask_a = mask_a
        self._mask_b = mask_b
        self._superresolution = superresolution
        self._coinc_window = coincidence_window
        self._calib_file = calibration_file
        self._diag_filter = diag_filter

        self._delays = delays
        self._freqs = freqs
        self._yplot = lines

    def yingwen_plot(self):
        return np.copy(self._delays), np.copy(self._freqs), np.copy(self._yplot)

    def fit_yingwen_plot(self):
        amp_list = np.empty(len(self._freqs))
        freq_list = np.empty(len(self._freqs))
        phase_list = np.empty(len(self._freqs))
        offset_list = np.empty(len(self._freqs))
        d_amp_list = np.empty(len(self._freqs))
        d_freq_list = np.empty(len(self._freqs))
        d_phase_list = np.empty(len(self._freqs))
        d_offset_list = np.empty(len(self._freqs))

        dz = np.mean(np.diff(self._delays))
        norm_freqs = 2*dz*self._freqs/utils.c # analytic frequency expected from HOM math

        for ix in range(len(self._freqs)): # iterate over all columns of Yingwen plot
            col = self._yplot[:,ix]
            p, cov, res = fit_osc(col, norm_freqs[ix])
            amp_list[ix] = p[0]
            phase_list[ix] = p[1]
            offset_list[ix] = p[2]
            d_amp_list[ix] = np.sqrt(cov[0,0])
            d_phase_list[ix] = np.sqrt(cov[1,1])
            d_offset_list[ix] = np.sqrt(cov[2,2])
        
        return amp_list, phase_list, offset_list, d_amp_list, d_phase_list, d_offset_list