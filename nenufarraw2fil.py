# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:08:01 2019

@author: greg
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import os
from astropy.time import Time

header_keyword_types = {
    b'telescope_id' : b'<l',
    b'machine_id'   : b'<l',
    b'data_type'    : b'<l',
    b'barycentric'  : b'<l',
    b'pulsarcentric': b'<l',
    b'nbits'        : b'<l',
    b'nsamples'     : b'<l',
    b'nchans'       : b'<l',
    b'nifs'         : b'<l',
    b'nbeams'       : b'<l',
    b'ibeam'        : b'<l',
    b'rawdatafile'  : b'str',
    b'source_name'  : b'str',
    b'az_start'     : b'<d',
    b'za_start'     : b'<d',
    b'tstart'       : b'<d',
    b'tsamp'        : b'<d',
    b'fch1'         : b'<d',
    b'foff'         : b'<d',
    b'refdm'        : b'<d',
    b'period'       : b'<d',
    b'src_raj'      : b'<d',
    b'src_dej'      : b'<d',
#    b'src_raj'      : b'angle',
#    b'src_dej'      : b'angle',
    }

def to_sigproc_keyword(keyword, value=None):
    """ Generate a serialized string for a sigproc keyword:value pair

    If value=None, just the keyword will be written with no payload.
    Data type is inferred by keyword name (via a lookup table)

    Args:
        keyword (str): Keyword to write
        value (None, float, str, double or angle): value to write to file

    Returns:
        value_str (str): serialized string to write to file.
    """

    keyword = bytes(keyword)

    if value is None:
        return np.int32(len(keyword)).tostring() + keyword
    else:
        dtype = header_keyword_types[keyword]

        dtype_to_type = {b'<l'  : np.int32,
                         b'str' : str,
                         b'<d'  : np.float64}#,
#                         b'angle' : to_sigproc_angle}

        value_dtype = dtype_to_type[dtype]

        if value_dtype is str:
            return np.int32(len(keyword)).tostring() + keyword + np.int32(len(value)).tostring() + value
        else:
            return np.int32(len(keyword)).tostring() + keyword + value_dtype(value).tostring()



parser = argparse.ArgumentParser(description="Converts NenuFAR-raw data to Filterbank file");
parser.add_argument('indir', type=str, help='NenuFAR raw files directory');
parser.add_argument("-f", type=float, dest='f_res', help="frequency resolution in Hz", default=1.);
parser.add_argument("-t", type=float, dest='t_res', help="time resolution in s", default=1.);
parser.add_argument("-o", type=str, dest='outdir', help="output directory");
parser.add_argument('-p', type=int, dest='polnum', choices=[0,1], help="one single polarization only (polarization number)");
parser.add_argument("-b", type=int, dest='bmch', help="beam number (for single beam extraction)");
args = parser.parse_args();

indir = Path(args.indir);
assert indir.exists(), "file(s) not found -- check path..."

fnames_raw = sorted(list(indir.glob("*.raw")));
fnames_parset = sorted(list(indir.glob("*.parset")));
nFiles = len(fnames_raw);
if nFiles < 1:
    print('no raw files found -- check path...');
    sys.exit();

nRes = int(2**np.round(np.log2(200./1024.*1e6 / args.f_res)));   # frequency samples
if nRes < 2:
    nRes = 2;

nInt = int(np.round((args.t_res) / (nRes / (200./1024.*1e6)))); # time integrations
if nInt < 1:
    nInt = 1;

# make sure there are at least nRes*nInt samples!!

print('\nprocessing data from : ' + args.indir);
print('found ' + str(nFiles) + ' raw file(s):');
for fname in fnames_raw:
    print('    '+(fname.name));
print('\nrequested frequency resolution : ' + str(args.f_res) + ' Hz');
print('requested time resolution : ' + str(args.t_res) + ' s');

print('\n# of frequency samples : ' + str(nRes));
print('frequency resolution : ' + str(200./1024.*1e6/nRes) + ' Hz');
print('# of spectrum integration : ' + str(nInt));
print('time integration : ' + str(nInt*nRes/(200./1024.*1e6)) + ' s');

os.chdir(args.indir);

dt_header = np.dtype([('nobpb', 'int32'),          # NUMBER_OF_BEAMLET_PER_BANK = number of channels
    ('nb_samples', 'int32'),     # NUMBER of SAMPLES (fftlen*nfft)
    ('bytespersample', 'int32'), # BYTES per SAMPLE (4/8 for 8/16bits data)
    ]);

# LOOP OVER FILES

with open(fnames_raw[0].name,'rb') as fd_raw:
    header = np.frombuffer(fd_raw.read(dt_header.itemsize),
                           count=1,
                           dtype=dt_header,
                           )[0];

nobpb = header['nobpb']
nb_samples = header['nb_samples']
bytespersample = header['bytespersample']

print('\n'+str(nobpb)+' beams found');
if args.bmch is not None:
    if args.bmch < 0 or args.bmch > nobpb-1:
        print('\n INVALID BEAM NUMBER -- exiting...');
        print('beam number should be between 0 and '+str(nobpb-1));
        sys.exit();

print(str(nb_samples)+' samples / beam = '+str(nb_samples/(200./1024.*1e6))+' s of data per bloc');

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
    
with open(fnames_raw[0].name,'rb') as fd_raw:
    header = np.frombuffer(fd_raw.read(dt_header.itemsize),
       count=1,
       dtype=dt_header,
       )[0]

data = np.memmap(fnames_raw[0].name,
 dtype=dt_block,
 mode='r',
 offset=dt_header.itemsize,
 )

print(str(np.shape(data)[0])+' data bloc(s) found');
print('total data time span : '+str(np.shape(data)[0]*(nb_samples/(200./1024.*1e6)))+' s');

if (np.shape(data)[0]*nb_samples) < nRes:
    print('requested frequency resolution higher than data set size... exiting...');
    sys.exit();

if (np.shape(data)[0]*(nb_samples/(200./1024.*1e6))) < args.t_res:
    print('requested time integration longer than data set... exiting...');
    sys.exit();


chan_start, chan_stop = header['lbc_alloc'][0][-1], header['lbc_alloc'][-1][-1];
print('frequency coverage : '+str(chan_start * 200.0/1024)+ ' - '+str(chan_stop * 200.0/1024)+' MHz');
    

with open(fnames_parset[0].name,'r') as fparset:
    currline = ' ';
    linenbr = 0;
    while len(currline) != 0:
        currline = fparset.readline();
        if 'Observation.startTime' in currline:
            timestr = currline[currline.find('=')+1:-1];
        if 'Output.hd_bitMode' in currline:
            numbits = int(currline[currline.find('=')+1:-1]);
        if 'Observation.contactName' in currline:
            observer = currline[currline.find('=')+1:-1];
        if 'AnaBeam[0].duration' in currline:
            scanlen = float(currline[currline.find('=')+1:-1]);
        if 'AnaBeam[0].directionType' in currline:
            srcname = currline[currline.find('=')+1:-1];
        linenbr += 1;

if args.outdir is not None:
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir);
    os.chdir(args.outdir);
else:
    os.chdir(args.indir);

if not os.path.exists('fil_files'):
    os.mkdir('fil_files');
os.chdir('fil_files');
print('\nwriting FIL files to '+os.getcwd());

# for data processing:
fftlen = 1;
nof_polcpx = 4;
nfft = len(data[0]['data']) // nobpb // fftlen // nof_polcpx;

if args.bmch is None:
    beams_to_process = range(nobpb);
else:
    beams_to_process = range(args.bmch,args.bmch+1);

for beam_nbr in beams_to_process:
    print('\n***********************');
    if args.bmch is None:
        print('processing beam # '+str(beam_nbr+1)+' / '+str(nobpb));
    else:
        print('processing beam # '+str(beam_nbr));
    if args.polnum is None:
        fname =  fnames_raw[0].name.strip('.raw') + '_beam_' + str(beam_nbr) + '_' + str(int(200./1024.*1e6/nRes)) + 'Hz_' + str(int(nInt*nRes/(200./1024.*1e6))) + 's.fil';
    else:
        fname =  fnames_raw[0].name.strip('.raw') + '_beam_' + str(beam_nbr) + '_' + str(int(200./1024.*1e6/nRes)) + 'Hz_' + str(int(nInt*nRes/(200./1024.*1e6))) + 's_pol_'+ str(args.polnum) +'.fil';
    print('writing data to '+fname);
    
    f = {b'telescope_id': b'66',    # NenuFAR
      b'az_start': b'0.0',  # not sure where to find
      b'nbits': str(numbits).encode(),       # TBD
      b'source_name': srcname.encode(),   # read in parset AnaBeam[0].directionType
      b'data_type': b'1',       # look into that
      b'nchans': str(nRes).encode(), # 2**18 for < 1Hz resolution at 200./1024. MHz sampling
      b'machine_id': b'99', # ??
      b'tsamp': str(1024./(200.*1e6)*nRes*nInt).encode(),
      b'foff': str(200./1024./nRes).encode(),    # 200./1024./2**18
      b'src_raj': b'181335.2',
      b'src_dej': b'-174958.1',
      b'tstart': str(Time(timestr).mjd).encode(),
      b'nbeams': b'1',
      b'fch1': str((chan_start+beam_nbr)* 200.0/1024 - 200./1024./2).encode(),   # b'34.08203125',   #(chan_start+beam_nbr)* 200.0/1024 + 200./1024./2.
      b'za_start': b'0.0',
      b'rawdatafile': fnames_raw[0].name.encode(),    # raw file name
      b'nifs': b'1'}

    header_string = b''
    header_string += to_sigproc_keyword(b'HEADER_START')
    
    for keyword in f.keys():
        if keyword not in header_keyword_types.keys():
            pass
        else:
            header_string += to_sigproc_keyword(keyword, f[keyword])
        
    header_string += to_sigproc_keyword(b'HEADER_END')

    outfile = open(fname, 'wb');
    outfile.write(header_string);
    
    full_sigXX = []; full_sigYY = [];
    
    for block_num in range(np.shape(data)[0]):    
        tmp = data[block_num]['data'];
        tmp.shape = (nfft, fftlen, nobpb, nof_polcpx);
        tmp = tmp.astype('float32').view('complex64');
        if args.polnum != 1:
            full_sigXX = np.append(full_sigXX,tmp[:,0,beam_nbr,0]);
        if args.polnum != 0:
            full_sigYY = np.append(full_sigYY,tmp[:,0,beam_nbr,1]);
        
    nSpec = max(int(np.floor(len(full_sigXX)/nRes/nInt)),int(np.floor(len(full_sigYY)/nRes/nInt)));
    
    if args.polnum != 1:
        full_sigXX = full_sigXX - np.mean(full_sigXX);
        full_sigXX = np.reshape(full_sigXX[:nRes*nInt*nSpec],(nRes,nInt*nSpec),order='F');
        full_sigXX = np.abs(np.fft.fft(full_sigXX,axis=0))**2;
        full_sigXX = np.fft.fftshift(full_sigXX,axes=0);
    
    if args.polnum != 0:
        full_sigYY = full_sigYY - np.mean(full_sigYY);
        full_sigYY = np.reshape(full_sigYY[:nRes*nInt*nSpec],(nRes,nInt*nSpec),order='F');
        full_sigYY = np.abs(np.fft.fft(full_sigYY,axis=0))**2;
        full_sigYY = np.fft.fftshift(full_sigYY,axes=0);
    
    stokesIdata = np.zeros((nRes,nSpec));
    
    for k in range(nSpec):
        if args.polnum is None:
            stokesIdata[:,k] = np.mean(full_sigXX[:,k*nInt:(k+1)*nInt], axis=1) + np.mean(full_sigYY[:,k*nInt:(k+1)*nInt], axis=1);
        elif args.polnum == 0:
            stokesIdata[:,k] = np.mean(full_sigXX[:,k*nInt:(k+1)*nInt], axis=1);
        elif args.polnum == 1:
            stokesIdata[:,k] = np.mean(full_sigYY[:,k*nInt:(k+1)*nInt], axis=1);
        
    maxtot = np.max(stokesIdata);
    mintot = np.min(stokesIdata);
    
    stokesIdata -= mintot;
    stokesIdata *= (250./(maxtot-mintot));
    
    for k in range(nSpec):
        stokesIdata[:,k].astype('uint8').tofile(outfile);
        
    outfile.close()
    
    
