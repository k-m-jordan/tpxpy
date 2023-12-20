import os
import shutil
from glob import glob
from typing import Literal

import numpy as np

c = 299792458.0

def get_cache_name(fname) -> str:
    dirname, filename = os.path.split(fname)
    return dirname + "/.tpxcache/" + filename + ".cache.npz"

def is_file_cached(fname) -> bool:
    cache_fname = get_cache_name(fname)

    if not os.path.exists(cache_fname):
        return False

    from tpxpy.loader import CACHE_VERSION
    
    # check that the cache version number is up-to-date
    with np.load(cache_fname) as data:
        return data['cache_version'] == CACHE_VERSION

def is_dir_cached(dirname) -> bool:
    all_files_cached = True

    cache_dir = dirname + "/.tpxcache"
    file_list = glob(dirname + "/*.tpx3")

    for fname in file_list:
        all_files_cached = all_files_cached and is_file_cached(dirname, fname)

    return all_files_cached

def clear_cluster_cache(dirname) -> None:
    cache_dir = dirname + "/.tpxcache"
    shutil.rmtree(cache_dir, ignore_errors=True)

def all_tpx3_in_dir(dirname, include:Literal['cached','uncached','all']='all') -> list[str]:
    if include == 'all':
        return sorted(glob(dirname + "/*.tpx3"))
    else:
        files = sorted(glob(dirname + "/*.tpx3"))
        cached_files = []
        uncached_files = []
        for f in files:
            if is_file_cached(f):
                cached_files.append(f)
            else:
                uncached_files.append(f)

        if include == 'cached':
            return cached_files
        elif include == 'uncached':
            return uncached_files
        else:
            raise ValueError("include must be one of 'cached', 'uncached', or 'all")

# reorients a 2D ndarray so that it plots properly with pyplot.imshow(_, origin='lower')
def orient(data : np.ndarray):
    return data.transpose()

# file sizes get very large if cache_compressed is set to False
# (full size ~= 5 * compressed size)
# similar time to load
# the initial compression takes ~4 * more time than an uncompressed save
def load_dir_concatenated(loader, dirname, cached=True, cache_compressed=True, max_files=None):
    full_image_cache_name = dirname + "/.tpxcache/full.tpx3.cache.npz"

    from tpxpy.loader import TpxImage
    import tpxpy.utils as tpxutils

    img: TpxImage

    if cached and os.path.exists(full_image_cache_name):
        print("Loading cache...")
        img = TpxImage.load_cache(full_image_cache_name)
    else:
        tpx_list = tpxutils.all_tpx3_in_dir(dirname)
        if max_files is not None and len(tpx_list) > max_files:
            tpx_list = tpx_list[:max_files]

        img = TpxImage.concatenate(tpx_list, loader)
        if cached:
            print("Generating cache...")
            img.save_cache(full_image_cache_name, compress=cache_compressed)

    return img

def downsample(image, factor):
    result = np.zeros_like(image[0::factor,0::factor])
    for i in range(factor):
        for j in range(factor):
            result += image[i::factor,j::factor]

    return result
