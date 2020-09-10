import numpy as np
import cbgen
import time
from pathlib import Path
from cbgen._env import BGEN_CACHE_HOME
from bgen_reader import open_bgen

def read_nth(nth, cb, offset_list, bgen2):

    bgen2_t0 = time.time()
    proball_bgen2 = bgen2.read(np.s_[::nth],order='F')
    #for ivariant in range(0,bgen2.nvariants,nth):
    #    prob1_bgen2 = bgen2.read(ivariant)#np.s_[::nth])
    diff_bgen2 = time.time()-bgen2_t0
    print(f"bgen2: reading every {nth}th variant takes {diff_bgen2} seconds")
    prob1_bgen2 = proball_bgen2[:,-1,:]

    cb_t0 = time.time()
    proball_cb = np.empty((bgen2.nsamples,len(offset_list[::nth]),3),order='F') # Much faster than order='C'
    for ivariant, offset in enumerate(offset_list[::nth]):
        proball_cb[:,ivariant,:] = cb.read_probability(offset)
    diff_cb = time.time()-cb_t0
    print(f"cbgen: reading every {nth}th variant takes {diff_cb} seconds")
    prob1_cb = proball_cb[:,-1,:]

    assert np.allclose(prob1_bgen2.reshape(-1,prob1_bgen2.shape[-1]),prob1_cb,equal_nan=True)



if __name__ == "__main__":
    filename = "merged_487400x220000.bgen"
    print(f"file {filename}")
    filepath = BGEN_CACHE_HOME / "test_data" / filename
    if not filepath.exists(): #For convenience, assume that any file is the right file
        filepath = cbgen.example.get(filename)

    cb = cbgen.bgen_file(filepath)

    cb_mf = filepath.parent / (filepath.stem + ".cb_mf")
    if not cb_mf.exists():
        cb.create_metafile(cb_mf, verbose=True)

    offset_list = []
    mf = cbgen.bgen_metafile(cb_mf)
    for ipartition in range(mf.npartitions):
        partition = mf.read_partition(ipartition)
        offset_list.extend(partition.variants.offset)

    bgen2 = open_bgen(filepath,verbose=False)

    read_nth(5000, cb, offset_list, bgen2)
    read_nth(500, cb, offset_list, bgen2)

    del bgen2
    del cb

    print("!!!cmk")