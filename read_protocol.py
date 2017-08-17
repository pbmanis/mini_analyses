"""
Read acq4 protocol directory

Add this to your .bash_profile
    export PYTHONPATH="path_to_acq4:${PYTHONPATH}"
For example:
    export PYTHONPATH="/Users/pbmanis/Desktop/acq4:${PYTHONPATH}"
"""


import sys
import sqlite3
import os.path
from collections import OrderedDict
import re
import pickle
import datetime
import numpy as np
from acq4.util.metaarray import MetaArray
from acq4.analysis.dataModels import PatchEPhys
import acq4.util.functions as functions
from acq4.util import DataManager

DM = DataManager

def readPhysProtocol(protocolFilename, records=None):
    dh = DM.getDirHandle(protocolFilename, create=False)
    PatchEPhys.cell_summary(dh)
    Clamps = PatchEPhys.GetClamps()
    ci = Clamps.getClampData(dh)
    info = dh.info
    DM.cleanup()
    return Clamps.traces, Clamps.time_base, Clamps.sample_interval

def readScannerProtocol(protocolFilename, records=None, sparsity=None):
    dh = DM.getDirHandle(protocolFilename, create=False)
    if records is None:
#       print 'protocolfile: ', protocolFilename
#        print 'info: ', dh.info()
        try:
            records = range(0, len(dh.info()['sequenceParams'][('Scanner', 'targets')]))
        except:
            raise StandardError("File not readable or not found: %s" % protocolFilename)
            exit()
    else:
        records = sorted(records)
    print 'Processing Protocol: %s' % protocolFilename
    (rest, mapnumber) = os.path.split(protocolFilename)
#    protocol = dh.name()
    PatchEPhys.cell_summary(dh)
    dirs = dh.subDirs()
    pars = {}
    if records is not None:
        pars['select'] = records
    Clamps = PatchEPhys.GetClamps()
    modes = []
    clampDevices = PatchEPhys.getClampDeviceNames(dh)
    # must handle multiple data formats, even in one experiment...
    if clampDevices is not None:
        data_mode = dh.info()['devices'][clampDevices[0]]['mode']  # get mode from top of protocol information
    else:  # try to set a data mode indirectly
        if 'devices' not in dh.info().keys():
            devices = 'not parsed'
        else:
            devices = dh.info()['devices'].keys()  # try to get clamp devices from another location
        for kc in PatchEPhys.knownClampNames():
            if kc in devices:
                clampDevices = [kc]
        try:
            data_mode = dh.info()['devices'][clampDevices[0]]['mode']
        except:
            data_mode = 'Unknown'
    if data_mode not in modes:
        modes.append(data_mode)
    sequence = PatchEPhys.listSequenceParams(dh)
    pars['sequence1'] = {}
    pars['sequence2'] = {}
    reps = sequence[('protocol', 'repetitions')]
    if sparsity is None:
        targets = range(len(sequence[('Scanner', 'targets')]))
    else:
        targets = range(0, len(sequence[('Scanner', 'targets')]), sparsity)
    pars['sequence1']['index'] = reps
    pars['sequence2']['index'] = targets
    try:
        del pars['select']
    except KeyError:
        pass
    ci = Clamps.getClampData(dh, pars)
    info = {}
    rep = 0
    tar = 0
    for i, directory_name in enumerate(dirs):  # dirs has the names of the runs within the protocol, is a LIST
#        if sparsity is not None and i % sparsity != 0:
#            continue
        data_dir_handle = dh[directory_name]  # get the directory within the protocol
        pd_file_handle = PatchEPhys.getNamedDeviceFile(data_dir_handle, 'Photodiode')
        pd_data = pd_file_handle.read()
        
        # if i == 0: # wait until we know the length of the traces
        #     data = np.zeros((len(reps), len(targets), len(Clamps.traces[i])))
        #     print Clamps.traces.shape
        # d = Clamps.traces[i]
        # data[rep, tar, :] = d
        info[(rep, tar)] = {'directory': directory_name, 'rep': rep, 'pos': data_dir_handle.info()['Scanner']['position']}
        tar = tar + 1
        if tar > len(targets):
            tar = 0
            rep = rep + 1
        DM.cleanup()  # close all opened files
    DM.cleanup()
    data = np.reshape(Clamps.traces, (len(reps), len(targets), Clamps.traces.shape[1]))
    return data, Clamps.time_base, pars, info
