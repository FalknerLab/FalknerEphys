# import h5py
from nptdms import TdmsFile, TdmsWriter
from falknerephys.io.whitematter import tdms_to_h5
# slp_data = h5py.File('wm_demo.h5', 'r')
# track = slp_data['tracks']
# print(track.keys())

tdms_to_h5('wm_demo.tdms', 'test.h5', out_len_s=150)

