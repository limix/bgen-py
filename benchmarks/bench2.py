import numpy as np
import cbgen
import time
from pathlib import Path
from cbgen._env import BGEN_CACHE_HOME
from bgen_reader import open_bgen

# This uses the PySnpTools multithreading tools. 
# They use Python threading library under the covers
from pysnptools.util.mapreduce1 import map_reduce
from pysnptools.util.mapreduce1.runner import Local, LocalMultiThread


def read_nth(nth, cb, offset_list, bgen2):

    if False:
        bgen2_t0 = time.time()
        proball_bgen2 = bgen2.read(nth)
        diff_bgen2 = time.time()-bgen2_t0
        print(f"bgen2: reading every {nth}th variant takes {diff_bgen2} seconds")
        prob1_bgen2 = proball_bgen2[:,-1,:]

    if False:
        cb_t0 = time.time()
        proball_cb = np.empty((bgen2.nsamples,len(offset_list[nth]),3),order='F') # Much faster than order='C'
        for ivariant, offset in enumerate(offset_list[nth]):
            prob1 = cb.read_probability(offset)
            proball_cb[:,ivariant,:] = prob1
        diff_cb = time.time()-cb_t0
        print(f"cbgen: reading every {nth}th variant takes {diff_cb} seconds")
        prob1_cb = proball_cb[:,-1,:]
        assert np.allclose(prob1_bgen2.reshape(-1,prob1_bgen2.shape[-1]),prob1_cb,equal_nan=True)

    if True:
        for runner in [Local(), LocalMultiThread(8)]:
            mt_t0 = time.time()
            proball_mt = np.empty((bgen2.nsamples,len(offset_list[nth]),3),order='F') # Much faster than order='C'

            offset_slice = offset_list[nth]
            def mapper(ivariant):
                prob1 = cb.read_probability(offset_slice[ivariant])
                proball_mt[:,ivariant,:] = prob1
                return None
            map_reduce(range(len(offset_slice)),
                       mapper=mapper,
                       runner=runner)

            diff_mt = time.time()-mt_t0
            print(f"{runner}: reading every {nth}th variant takes {diff_mt} seconds")
            prob1_mt = proball_mt[:,-1,:]
            #assert np.allclose(prob1_mt,prob1_cb,equal_nan=True)


if __name__ == "__main__":
    #filename = "merged_487400x220000.bgen"
    filename = "1000x500000.bgen" #Copy from https://www.dropbox.com/sh/vuzozn39vsw8zcl/AAAZT2aRMB3V8kdz6CzlmdY-a?dl=0 bedToBgen

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

    #read_nth(np.s_[::5000], cb, offset_list, bgen2)
    #read_nth(np.s_[::500], cb, offset_list, bgen2)
    #read_nth(np.s_[::50], cb, offset_list, bgen2)
    #read_nth(np.s_[::10], cb, offset_list, bgen2)
    read_nth(np.s_[::10], cb, offset_list, bgen2)
    #read_nth(np.s_[::5], cb, offset_list, bgen2)
    #read_nth(np.s_[::500], cb, offset_list, bgen2)
    #read_nth(np.s_[200*1000:200*1000+50], cb, offset_list, bgen2)
    #read_nth(np.s_[200*1000:200*1000+500], cb, offset_list, bgen2)



    del bgen2
    del cb

    print("!!!cmk")