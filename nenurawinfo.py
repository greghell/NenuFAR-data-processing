# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:15:11 2019

@author: greg

code gives high level info about raw data set
call:
    python nenurawinfo.py filename.raw
"""


import sys
import numpy as np

dt_header = np.dtype([('nobpb', 'int32'),          # NUMBER_OF_BEAMLET_PER_BANK = number of channels
    ('nb_samples', 'int32'),     # NUMBER of SAMPLES (fftlen*nfft)
    ('bytespersample', 'int32'), # BYTES per SAMPLE (4/8 for 8/16bits data)
    ]);

with open(sys.argv[1],'rb') as fd_raw:
    header = np.frombuffer(fd_raw.read(dt_header.itemsize),
                           count=1,
                           dtype=dt_header,
                           )[0];

nobpb = header['nobpb']
nb_samples = header['nb_samples']
bytespersample = header['bytespersample']

print('\nreading file '+sys.argv[1]);

print('Number of beams : '+str(nobpb));
print('Number of samples per bloc : '+str(nb_samples));
print('Number of bytes per sample : '+str(bytespersample));

bytes_in = bytespersample * nb_samples * nobpb

dt_block = np.dtype([('eisb', 'uint64'),
   ('tsb', 'uint64'),
   ('bsnb', 'uint64'),
   ('data', 'int8', (bytes_in,)),
   ])   # data block structure

dt_lane_beam_chan = np.dtype([('lane', 'int32'),
  ('beam', 'int32'),
  ('chan', 'int32'),
  ])    # lane number, beam number, and channel number

dt_header = np.dtype([('nobpb', 'int32'),          # NUMBER_OF_BEAMLET_PER_BANK = number of channels
  ('nb_samples', 'int32'),     # NUMBER of SAMPLES (fftlen*nfft)
  ('bytespersample', 'int32'), # BYTES per SAMPLE (4/8 for 8/16bits data)
  ('lbc_alloc', dt_lane_beam_chan, (nobpb,)),
  ])    # header structure
    
with open(sys.argv[1],'rb') as fd_raw:
    header = np.frombuffer(fd_raw.read(dt_header.itemsize),
       count=1,
       dtype=dt_header,
       )[0]

data = np.memmap(sys.argv[1],
 dtype=dt_block,
 mode='r',
 offset=dt_header.itemsize,
 )

print(str(np.shape(data)[0])+' data bloc(s) found');
print('total number of samples : '+str(np.shape(data)[0]*nb_samples))
print('total data time span : '+str(np.shape(data)[0]*(nb_samples/(200./1024.*1e6)))+' s');
