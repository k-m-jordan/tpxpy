import os
import shutil
from glob import glob

from tpxpy.loader import CACHE_VERSION

import numpy as np

c = 299792458.0

def get_cache_name(fname) -> str:
    dirname, filename = os.path.split(fname)
    return dirname + "/.tpxcache/" + filename + ".cache.npz"

def is_file_cached(fname) -> bool:
    cache_fname = get_cache_name(fname)

    if not os.path.exists(cache_fname):
        return False
    
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

def all_tpx3_in_dir(dirname):
    return sorted(glob(dirname + "/*.tpx3"))

# reorients a 2D ndarray so that it plots properly with pyplot.imshow(_, origin='lower')
def orient(data : np.ndarray):
    return data.transpose()