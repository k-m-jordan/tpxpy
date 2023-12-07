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

def all_tpx3_in_dir(dirname, include:Literal['cached','uncached','all']='all'):
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