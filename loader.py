from tpxpy import utils

import os
os.environ["TEMP"] = "C:/tmp"
os.environ["PATH"] += ";C:/ProgramData/CUDA/v12.3/bin"
import numpy as np

from typing import Literal

if "TPXPY_NO_CUDA" in os.environ.keys():
    import numpy as cp
    def asnumpy(x):
        return x
else:
    import cupy as cp
    from cupy import asnumpy

from collections import namedtuple

from tqdm.auto import tqdm

from scipy.signal import find_peaks, peak_widths

# unsigned little endian types
le_uint64 = cp.dtype(cp.uint64).newbyteorder('<')
le_uint16 = cp.dtype(cp.uint16).newbyteorder('<')
le_uint8 = cp.dtype(cp.uint8).newbyteorder('<')

# width and height of TPX3 sensor, in pixels
TPX_SIZE = 256
TPX_TOT_MAX = 1024

# used for raw data
PacketImage = namedtuple('PacketImage', 'x y toa tot')
ClusteredImage = namedtuple('ClusteredImage', 'x y toa tot cluster_indices centroid_x centroid_y centroid_t tot_calibration')

# version number for the cache files; increment this every time the cache format is changed to force the cache data to update
CACHE_VERSION = 1

class TpxImage:

    def __init__(self, clusters : ClusteredImage, fname : str):
        self._origin_fname = fname

        # data that is stored in the image
        if clusters is not None:
            self._x = clusters.x
            self._y = clusters.y
            self._toa = clusters.toa
            self._tot = clusters.tot
            self._cluster_indices = clusters.cluster_indices
            self._centroid_x = clusters.centroid_x
            self._centroid_y = clusters.centroid_y
            self._centroid_t = clusters.centroid_t
            self._tot_calibration = clusters.tot_calibration
        
        # information used to process the image; these are not set unless explicitly calculated after loading, and are not cached between runs
        self._coincidence_indices = None
    
    def save_cache(self, fname, compress=False):
        save_cmd = None
        if compress:
            save_cmd = cp.savez_compressed
        else:
            save_cmd = cp.savez
        
        # we first write to a temp file, then rename, so that crashes during saving won't be confused for completed saves
        cache_tmp_fname = fname + ".tmp"

        with open(cache_tmp_fname, "wb") as f:
            save_cmd(f,
                    cache_version=CACHE_VERSION,
                    fname=self._origin_fname,
                    x=self._x,
                    y=self._y,
                    toa=self._toa,
                    tot=self._tot,
                    cluster_indices=self._cluster_indices,
                    centroid_x = self._centroid_x,
                    centroid_y = self._centroid_y,
                    centroid_t = self._centroid_t,
                    tot_calibration = self._tot_calibration)
        
        os.rename(cache_tmp_fname, fname)
    
    @staticmethod
    def load_cache(fname):
        global CACHE_VERSION

        with open(fname, "rb") as f:
            data = np.load(f)

            version = data['cache_version']

            if version != CACHE_VERSION:
                raise RuntimeError(f"Tried to load an out-of-date cache file (version {version}, only {CACHE_VERSION} supported); re-cluster the data and try again")
            
            return TpxImage (
                ClusteredImage(
                    data['x'],
                    data['y'],
                    data['toa'],
                    data['tot'],
                    data['cluster_indices'],
                    data['centroid_x'],
                    data['centroid_y'],
                    data['centroid_t'],
                    data['tot_calibration']
                ),
                data['fname']
            )

    # Takes either a list of TpxImage or a list of string filenames
    @staticmethod
    def concatenate(image_list, loader=None, toa_spacing=1000):
        if type(image_list[0]) == str and loader is None:
            raise ValueError("Cannot load images when TpxLoader is None")

        if loader is not None:
            tpx_img_list = []
            for imgname in tqdm(image_list, desc="Loading Tpx3 Files"):
                tpx_img_list.append(loader.load(imgname))
            image_list = tpx_img_list

        num_packets = 0
        num_clusters = 0

        for img in image_list:
            num_packets += img.num_packets()
            num_clusters += img.num_clusters()

        x = np.empty(num_packets)
        y = np.empty(num_packets)
        toa = np.empty(num_packets)
        tot = np.empty(num_packets)
        cluster_indices = np.empty(num_packets)
        centroid_x = np.empty(num_clusters)
        centroid_y = np.empty(num_clusters)
        centroid_t = np.empty(num_clusters)

        tot_calibration = image_list[0]._tot_calibration

        packet_offset = 0
        cluster_offset = 0
        toa_offset = 0
        ix_offset = 0

        for img in tqdm(image_list, desc="Concatenating Tpx3 Files"):
            pix_start = packet_offset
            pix_end = packet_offset + img.num_packets()

            cix_start = cluster_offset
            cix_end = cluster_offset + img.num_clusters()

            x[pix_start:pix_end] = img._x
            y[pix_start:pix_end] = img._y
            toa[pix_start:pix_end] = img._toa + toa_offset
            tot[pix_start:pix_end] = img._tot
            cluster_indices[pix_start:pix_end] = img._cluster_indices + ix_offset

            centroid_x[cix_start:cix_end] = img._centroid_x
            centroid_y[cix_start:cix_end] = img._centroid_y
            centroid_t[cix_start:cix_end] = img._centroid_t + toa_offset

            packet_offset += img.num_packets()
            cluster_offset += img.num_clusters()
            toa_offset = toa[pix_end-1] + toa_spacing
            ix_offset = cluster_indices[pix_end-1] + 1

        return TpxImage(
            ClusteredImage(x, y, toa, tot, cluster_indices, centroid_x, centroid_y, centroid_t, tot_calibration),
            image_list[0]._origin_fname)
    
    def num_packets(self) -> int:
        return len(self._x)
    
    def num_clusters(self) -> int:
        return len(self._centroid_x)

    def cluster_sizes(self):
        cluster_start_ix = np.searchsorted(self._cluster_indices, np.arange(0, self.num_clusters()))
        cluster_stop_ix = np.concatenate((cluster_start_ix[1:], [self.num_packets()]))
        return cluster_stop_ix - cluster_start_ix

    def to_2d_image(self, type : Literal["raw", "singles", "coincidences"]="singles", bins=TPX_SIZE):
        x = None
        y = None

        if type == "raw":
            x = self._x
            y = self._y
        elif type == "singles":
            x = self._centroid_x
            y = self._centroid_y
        elif type == "coincidences":
            if self.num_coincidences() == 0:
                return None
            
            coinc_clusters = np.unique(self._coincidence_indices.flatten())
            x = self._centroid_x[coinc_clusters]
            y = self._centroid_y[coinc_clusters]
        else:
            raise ValueError("type needs to be either 'raw' (raw packet image) or 'singles' (cluster image) or 'coincidences' (coincidence image)")

        output, _, _ = np.histogram2d(x, y, bins=bins)
        
        return output

    # times gives the center of the bins, in ns
    def start_stop_histogram(self, bins=81, times=(0,100)):
        binsize = (times[1] - times[0])/bins
        times = (times[0] - binsize/2, times[1] + binsize/2)
        sorted_centroids = np.sort(self._centroid_t.astype(np.int64))
        delta_t = sorted_centroids[1:] - sorted_centroids[:-1]
        histogram, x = np.histogram(delta_t*1.25, bins=bins, range=times)
        histogram[0] *= 2 # double the density, since this bin is only half as big (values can't be negative)

        return (x[:-1] + x[1:])/2, histogram

    def timing_threshold_histogram(self):
        # for each packet, calculate the difference between packet toa and cluster toa as a function of packet tot
        histogram = np.zeros((TPX_TOT_MAX,))
        
        cluster_toa = self._centroid_t[self._cluster_indices]
        dt = self._toa - cluster_toa
        histogram[self._tot] += dt*1.25
        return histogram

    # min and max in ns
    # keeps searching for coincidences up to maxrange clusters apart, until newly found coincidences is less than threshold*(total coincidences)
    def set_coincidence_window(self, window_min : float, window_max : float, maxrange=250, threshold=0.0001):
        window_min /= 1.25
        window_max /= 1.25
        sort_ix = np.argsort(self._centroid_t)

        sort_times = self._centroid_t[sort_ix]

        coinc_pairs = []
        
        total_coincs = 0
        for offset in range(1,maxrange+1):
            dt = sort_times[offset:] - sort_times[:-offset]
            pairs = np.argwhere(np.logical_and(dt > window_min, dt < window_max))
            coinc_pairs.append((pairs, pairs+offset))

            num_within_window = len(pairs)
            total_coincs += num_within_window
            if num_within_window < threshold*total_coincs:
                break
        
        self._coincidence_indices = np.empty((total_coincs, 2), dtype=np.int32)
        ix = 0
        for result in coinc_pairs:
            num_coincs = len(result[0])
            self._coincidence_indices[ix:(ix+num_coincs),0] = result[0].flatten()
            self._coincidence_indices[ix:(ix+num_coincs),1] = result[1].flatten()
            ix += num_coincs
        
    def num_coincidences(self) -> int:
        if self._coincidence_indices is None:
            return 0
        else:
            return self._coincidence_indices.shape[0]

    def fit_lines(self, direction : Literal["horizontal", "vertical"]="horizontal", width_sigma=5, upscale_res:int=1):
        par = None # parallel and perpendicular directions
        perp = None

        if direction == 'vertical':
            par = self._centroid_y
            perp = self._centroid_x
        elif direction == 'horizontal':
            par = self._centroid_x
            perp = self._centroid_y
        else:
            raise ValueError("direction needs to be either 'horizontal' or 'vertical'")
        
        perp_hist, _ = np.histogram(perp, bins=TPX_SIZE, range=(0, TPX_SIZE))
        perp_max = np.max(perp_hist)

        peaks, _ = find_peaks(perp_hist, prominence=perp_max*0.5)
        _, _, w1, w2 = peak_widths(perp_hist, peaks)
        sigmas = ((w2 - w1)/(2*np.sqrt(2*np.log(2))))

        line1_mask = np.full((TPX_SIZE*upscale_res,TPX_SIZE*upscale_res), False)
        line2_mask = np.full((TPX_SIZE*upscale_res,TPX_SIZE*upscale_res), False)

        line1_bounds = np.array([peaks[0] - sigmas[0]*width_sigma, peaks[0] + sigmas[0]*width_sigma])*upscale_res
        line2_bounds = np.array([peaks[1] - sigmas[1]*width_sigma, peaks[1] + sigmas[1]*width_sigma])*upscale_res

        line1_bounds = [int(x) for x in line1_bounds]
        line2_bounds = [int(x) for x in line2_bounds]

        if direction == 'vertical':
            line1_mask[line1_bounds[0]:line1_bounds[1]+1,:] = True
            line2_mask[line2_bounds[0]:line2_bounds[1]+1,:] = True
        elif direction == 'horizontal':
            line1_mask[:,line1_bounds[0]:line1_bounds[1]+1] = True
            line2_mask[:,line2_bounds[0]:line2_bounds[1]+1] = True

        return line1_mask, line2_mask

class TpxLoader:

    def __init__(self, compress_cache = False):
        self._orientation_rotation = 0
        self._orientation_flip_we = False
        self._orientation_flip_ns = False

        self._compress_cache = compress_cache

        self._mask = np.full((TPX_SIZE,TPX_SIZE),True)

        self._cluster_range = 5
        self._space_window = 5
        self._time_window = 250

        self._tot_calibration_fname = None
        self._tot_calibration = np.zeros((TPX_TOT_MAX,))

    # rotation in degrees
    # flips are applied before rotation
    def set_load_orientation(self, flip_we=False, flip_ns=False, rotation=0):
        if rotation not in [0, 90, 180, 270]:
            raise ValueError("rotation should be 0, 90, 180, or 270")

        if rotation == 180:
            flip_we = not flip_we
            flip_ns = not flip_ns
            rotation = 0
        
        self._orientation_flip_we = flip_we
        self._orientation_flip_ns = flip_ns
        self._orientation_rotation = rotation

    
    def set_cluster_range(self, r) -> None:
        self._cluster_range = r
    

    def set_space_window(self, half_window) -> None:
        self._space_window = half_window


    def set_time_window(self, half_window) -> None:
        self._time_window = half_window
    

    def set_tot_calibration(self, fname) -> None:
        if not os.path.exists(fname):
            raise ValueError(f"Attempted to set ToT calibration file to {fname}, which does not exist")
        
        calibration = np.loadtxt(fname, delimiter=',')[:,1]

        self._tot_calibration_fname = fname
        self._tot_calibration[:len(calibration)] = calibration
        self._tot_calibration /= 1.25
    
    
    def set_cache_compressed(self, val : bool) -> None:
        self._compress_cache = val
    

    # parses a raw *.tpx3 file into a list of pixel addresses, time of arrivals, and time over thresholds
    def _loadtpx3_raw(self, fname, return_gpu=False, sort_toa=True) -> PacketImage:
        # initial pass to get the file size
        total_packets = 0

        with open(fname, "rb") as file:
            while True:
                # read a chunk
                header = file.read(4)

                if header == b'':
                    # we have hit the end of the file
                    break

                if not (header == b'TPX3'):
                    raise Exception(f"File {fname} is not a TPX3 file (chunk header {header})")
                
                chip_ix = file.read(1)
                file.read(1)
                chunk_size = int.from_bytes(file.read(2), byteorder='little')

                total_packets += chunk_size // 8

                file.read(chunk_size)

        # second pass to get the data
        cp_tot_calibration = cp.asarray(self._tot_calibration)

        chunk_data = []
        with open(fname, "rb") as file:
            while True:
                # read a chunk
                header = file.read(4)

                if header == b'':
                    # we have hit the end of the file
                    break

                if not (header == b'TPX3'):
                    raise Exception(f"File {fname} failed in second pass (chunk header {header})")
                
                chip_ix = file.read(1)
                file.read(1)
                chunk_size = int.from_bytes(file.read(2), byteorder='little')

                chunk_data.append(cp.frombuffer(file.read(chunk_size), le_uint64))
            
        # write all data to a cupy array
        raw_packets_gpu = cp.concatenate(chunk_data)

        # separate the pixel addresses
        pixel_addr = cp.empty(raw_packets_gpu.shape, dtype=le_uint16)
        cp.copyto(pixel_addr, (raw_packets_gpu & 0x0FFFF00000000000) >> 44, 'same_kind')

        # parse the x and y values from the address
        addr_x = cp.empty(raw_packets_gpu.shape, dtype=le_uint8)
        cp.copyto(addr_x, ((pixel_addr >> 1) & 0x00FC) | (pixel_addr & 0x0003), 'same_kind')

        addr_y = cp.empty(raw_packets_gpu.shape, dtype=le_uint8)
        cp.copyto(addr_y, ((pixel_addr >> 8) & 0x00FE) | ((pixel_addr >> 2) & 0x0001), 'same_kind')

        del pixel_addr

        # apply the mask to discard any unnecessary values
        mask_image = cp.asarray(self._mask)
        gpu_mask = mask_image[addr_x,addr_y]

        raw_packets_gpu = raw_packets_gpu[gpu_mask]
        addr_x = addr_x[gpu_mask]
        addr_y = addr_y[gpu_mask]

        del mask_image
        del gpu_mask

        if len(raw_packets_gpu) == 0:
            print(f"No packets found within mask for file {fname}")
            return PacketImage(None, None, None, None)

        # parse the ToT
        tot = cp.empty(raw_packets_gpu.shape, dtype=le_uint16)
        cp.copyto(tot, (raw_packets_gpu & 0x000000003FF00000) >> 20, 'same_kind')

        # parse the ToA
        toa = cp.empty(raw_packets_gpu.shape, dtype=le_uint16)
        cp.copyto(toa, (raw_packets_gpu & 0x00000FFFC0000000) >> 30, 'same_kind')

        # parse the FToA
        ftoa = cp.empty(raw_packets_gpu.shape, dtype=le_uint8)
        cp.copyto(ftoa, ((raw_packets_gpu & 0x00000000000F0000) >> 16) ^ 0x0F, 'same_kind')

        # parse the SPIDR time
        spidr_time = cp.empty(raw_packets_gpu.shape, dtype=le_uint16)
        cp.copyto(spidr_time, raw_packets_gpu & 0x000000000000FFFF, 'same_kind')

        del raw_packets_gpu

        # combine the three time values
        full_toa = cp.empty(toa.shape, dtype=le_uint64)
        cp.copyto(full_toa, spidr_time, 'same_kind')
        full_toa = (full_toa << 14) | toa
        full_toa = (full_toa << 4) | ftoa

        full_toa = (full_toa - cp_tot_calibration[tot]).astype(cp.uint64)

        if sort_toa:
            # ensure that ToA is ordered
            indices = cp.argsort(full_toa) # cupy can only do stable sorts
            addr_x = addr_x[indices]
            addr_y = addr_y[indices]
            tot = tot[indices]
            full_toa = full_toa[indices]

            del indices
        
        if self._orientation_flip_ns:
            addr_y = 255 - addr_y
        if self._orientation_flip_we:
            addr_x = 255 - addr_x
        if self._orientation_rotation == 90:
            addr_x, addr_y = addr_y, 255-addr_x
        if self._orientation_rotation == 270:
            addr_x, addr_y = 255-addr_y, addr_x
        
        # convert all values to signed integers (to prevent future math issues)
        addr_x = addr_x.astype(cp.int16)
        addr_y = addr_y.astype(cp.int16)
        tot = tot.astype(cp.int16)
        full_toa = full_toa.astype(cp.int64)

        pimage : PacketImage

        if return_gpu:
            pimage = PacketImage(
                addr_x,
                addr_y,
                full_toa,
                tot
            )
        else:
            pimage = PacketImage(
                asnumpy(addr_x),
                asnumpy(addr_y),
                asnumpy(full_toa),
                asnumpy(tot)
            )
        
        return pimage
    
    
    # parses a raw *.tpx3 file into raw packet data (see loadtpx3_raw) and a list of cluster indices and centroids
    def _loadtpx3(self, fname) -> TpxImage:
        x, y, toa, tot = self._loadtpx3_raw(fname, return_gpu=True, sort_toa=True)

        cluster_indices = cp.arange(len(x))
        
        for offset in range(1,self._cluster_range):
            # find out whether i and i+offset are neighbours:
            is_neighbour = cp.logical_and(
                cp.logical_and(
                    cp.abs(x[:-offset] - x[offset:]) <= self._space_window,    # neighbour along x?
                    cp.abs(y[:-offset] - y[offset:]) <= self._space_window),   # neighbour along y?
                    cp.abs(toa[:-offset] - toa[offset:]) <= (self._time_window // 1.25)) # neighbour along t?

            # neighbours belong to the same cluster
            cluster_indices[offset:][is_neighbour] = cluster_indices[:-offset][is_neighbour]

        # renumber cluster indices as 0, 1, 2, ...
        cluster_indices = cp.searchsorted(cp.unique(cluster_indices), cluster_indices)
        cluster_sort_ix = cp.argsort(cluster_indices)
        x = x[cluster_sort_ix]
        y = y[cluster_sort_ix]
        toa = toa[cluster_sort_ix]
        tot = tot[cluster_sort_ix]
        cluster_indices = cluster_indices[cluster_sort_ix]

        num_clusters = cluster_indices[-1] + 1

        # calculate cluster centroids
        weighted_x = x.astype(cp.int32) * tot
        weighted_y = y.astype(cp.int32) * tot

        # find indices dividing the difference clusters
        cluster_start_ix = cp.searchsorted(cluster_indices, cp.arange(0, num_clusters))

        # weighted sum over packet positions (weights are tot)
        sum_x = cp.add.reduceat(weighted_x, cluster_start_ix)
        sum_y = cp.add.reduceat(weighted_y, cluster_start_ix)
        sum_weight = cp.add.reduceat(tot, cluster_start_ix)

        # centroid x, y
        centroid_x = sum_x.astype(cp.float64) / sum_weight
        centroid_y = sum_y.astype(cp.float64) / sum_weight

        # the time of a centroid should be the toa of the packet with the largest tot
        tot_max = np.maximum.accumulate(asnumpy(tot.astype(cp.int64) + cluster_indices*TPX_TOT_MAX))
        cluster_stop_ix = np.zeros(cluster_start_ix.shape, dtype=cluster_start_ix.dtype)
        cluster_stop_ix[:-1] = asnumpy(cluster_start_ix)[1:] - 1
        cluster_stop_ix[-1] = tot_max.size - 1
        cluster_max_tot = tot_max[cluster_stop_ix]
        cluster_max_tot_ix = np.searchsorted(tot_max, cluster_max_tot)
        centroid_t = toa[cluster_max_tot_ix]
        
        data = ClusteredImage (
            asnumpy(x),
            asnumpy(y),
            asnumpy(toa),
            asnumpy(tot),
            asnumpy(cluster_indices),
            asnumpy(centroid_x),
            asnumpy(centroid_y),
            asnumpy(centroid_t),
            self._tot_calibration
        )
        
        return TpxImage(data, fname)
    

    def load(self, fname) -> TpxImage:
        is_cached = utils.is_file_cached(fname)
        
        if is_cached:
            cache_name = utils.get_cache_name(fname)
            return TpxImage.load_cache(cache_name)
        else:
            return self._loadtpx3(fname)


    def cache_all_files(self, dirname, use_existing_cache=True, show_pbar=True) -> None:
        global CACHE_VERSION

        file_list = None
        if use_existing_cache:
            file_list = utils.all_tpx3_in_dir(dirname, include='uncached')
        else:
            file_list = utils.all_tpx3_in_dir(dirname, include='cached')
        if len(file_list) == 0:
            return

        cache_dir = dirname + "/.tpxcache"

        os.makedirs(cache_dir, exist_ok=True)

        loop_iter = file_list
        if show_pbar:
            loop_iter = tqdm(file_list, miniters=1, position=0, leave=True, desc="Updating Tpx3 Cache")

        for fname in loop_iter:
            cache_fname = utils.get_cache_name(fname)
            cache_exists = os.path.exists(cache_fname)

            if cache_exists:
                if use_existing_cache:
                    continue # on to the next file
                else:
                    os.remove(cache_fname)
            
            img = self._loadtpx3(fname)
            img.save_cache(cache_fname, self._compress_cache)
    

    def generate_tot_calibration(self, file_list, output):
        timing_histogram = np.zeros((TPX_TOT_MAX,))
        num_files = 0

        for f in tqdm(file_list, desc="Generating ToT/ToA Calibration"):
            num_files += 1
            img = self.load(f)
            timing_histogram += img.timing_threshold_histogram() + img._tot_calibration # include any calibration that was corrected in processing
        timing_histogram /= num_files

        output_data = np.zeros((TPX_TOT_MAX, 2))
        output_data[:,0] = np.arange(TPX_TOT_MAX, dtype=np.int32)
        output_data[:,1] = timing_histogram

        np.savetxt(output, output_data, delimiter=',')