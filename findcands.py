#://github.com/UCBerkeleySETI/blimpy/blob/master/blimpy/io/fil_reader.py
# plots filterbank file and extracts candidates

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy import stats
from pathlib import Path
import sigproc
import sys


#fname = '/datax2/devfil/beam_data/20201110104110_HAT-P-44C.fil';
fname = sys.argv[1];
fhead = sigproc.read_header(fname);
headlen = sigproc.len_header(fname);

nSpec = int((Path(fname).stat().st_size - headlen)/fhead['nchans']/4);

spec = np.zeros((fhead['nchans']));
for k in range(nSpec):
    print(str(k+1) + ' / ' + str(nSpec));
    spec += np.fromfile(fname, dtype=np.uint32, count=int(fhead['nchans']), offset=headlen+int(4*k*fhead['nchans']));

plt.figure();
plt.plot(np.linspace(fhead['fch1'],fhead['fch1']+fhead['foff']*fhead['nchans'],fhead['nchans']),10.*np.log10(spec));
plt.grid();
plt.xlabel('frequency [MHz]');
plt.ylabel('power [dB]');
plt.suptitle(fhead['source_name']);
plt.show();

speccorrec = np.copy(spec);
filtsize = 101;
nRes = int(fhead['nchans'] / fhead['nifs']);
thres = 20;

specfilt = np.zeros((fhead['nchans']));
cands = np.zeros((fhead['nchans']));
for k in range(fhead['nifs']):
    print(str(k+1) + ' / ' + str(fhead['nifs']));
    tmp = speccorrec[k*nRes:(k+1)*nRes];
    specfilt[k*nRes:(k+1)*nRes] = scipy.signal.medfilt(tmp,kernel_size=filtsize);
    speccorrec[k*nRes:(k+1)*nRes] = tmp - specfilt[k*nRes:(k+1)*nRes];
    specstd = stats.median_abs_deviation(speccorrec[k*nRes:(k+1)*nRes]);
    cands[np.argwhere(speccorrec[k*nRes:(k+1)*nRes] > thres*specstd) + k*nRes] = speccorrec[np.argwhere(speccorrec[k*nRes:(k+1)*nRes] > thres*specstd) + k*nRes];


freqsaxis = np.linspace(fhead['fch1'],fhead['fch1']+fhead['foff']*fhead['nchans'],fhead['nchans']);
plt.figure();
#plt.subplot(211);
plt.plot(freqsaxis, 10.*np.log10(spec), label='spectrum');
plt.plot(freqsaxis, 10.*np.log10(specfilt), label='baseline');
plt.plot(freqsaxis[np.nonzero(cands)[0]],10.*np.log10(spec[np.nonzero(cands)]),'or', label='candidates');
plt.legend();
plt.xlabel('frequency [MHz]');
plt.ylabel('power [dB]');
plt.grid();
# plt.subplot(212);
# plt.plot(freqsaxis, speccorrec, label='flattened spectrum');
# plt.plot(freqsaxis[np.nonzero(cands)[0]],cands[np.nonzero(cands)],'or', label='candidates');
# plt.legend();
# plt.xlabel('frequency [MHz]');
# plt.ylabel('power [arbitrary]');
# plt.grid();
plt.suptitle(fhead['source_name']);
plt.show();
