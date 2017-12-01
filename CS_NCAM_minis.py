"""
Summary 14 mice
m#(1) means mouse #N, (#) is the number of cells
7 F/+  (m1(1), m2(3), m3(1), m4(1), m5(2), m9(2), m13(4)) = 14
7 F/F (m6(1), m7(4), m8(2), m10(5), m11(1), M12(1), m14(1)) = 15
"""
datasets = {'m1a': {'dir': '2017.04.25_000/slice_000/cell_001', 'prots': [0,1,3], 'thr': 1.75, 'rt': 0.35, 'decay': 6., 'G': 'F/+'},
            'm1b': {'dir': '2017.04.25_000/slice_000/cell_002', 'prots': [7], 'thr': 1.75, 'rt': 0.35, 'decay': 6., 'G': 'F/+'},
            'm2a': {'dir': '2017.05.02_000/slice_000/cell_000/', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.32, 'decay': 5., 'G': 'F/+'},
            'm2b': {'dir': '2017.05.02_000/slice_000/cell_001', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},
            'm2c': {'dir': '2017.05.02_000/slice_000/cell_002', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            'm3a': {'dir': '2017.05.04_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            'm4a': {'dir': '2017.05.05_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            'm5a': {'dir': '2017.05.11_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},
            'm5b': {'dir': '2017.05.11_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.5, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},
            'm6a': {'dir': '2017.07.05_000/slice_000/cell_001', 'prots': [2,3,4], 'thr': 2.5, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},  # unusually low rate
            'm7a': {'dir': '2017.07.06_000/slice_000/cell_000', 'prots': [1,2,3], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm7b': {'dir': '2017.07.06_000/slice_000/cell_001', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 6., 'G': 'F/F'},
            'm7c': {'dir': '2017.07.06_000/slice_000/cell_002', 'prots': [1,2], 'thr': 2.0, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm7d': {'dir': '2017.07.06_000/slice_000/cell_003', 'prots': [0,1,6], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
           # compared to others, e is an unstable recording
           # 'm7e': {'dir': '2017.07.06_000/slice_000/cell_004', 'prots': [0,1,2], 'thr': 2.5, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm8a': {'dir': '2017.07.07_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.5, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm8b': {'dir': '2017.07.07_000/slice_000/cell_001', 'prots': [0,1,2], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            # M9: some data has big noise, not acceptable
            #'m9a': {'dir': '2017.07.19_000/slice_000/cell_000', 'prots': [2,3,4], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            # m9b: protocols 0 and 2 have noise, not acceptable; 1 is ok
            'm9b': {'dir': '2017.07.19_000/slice_000/cell_001', 'prots': [1], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            'm9c': {'dir': '2017.07.19_000/slice_000/cell_002', 'prots': [0,1,2], 'thr': 1.5, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            # incomple data for m9d11:
            # 'm9d': {'dir': '2017.07.19_000/slice_000/cell_003', 'prots': [0], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/+'},
            # m10a: runs 1 & 2 have unacceptable noise
            'm10a': {'dir': '2017.07.27_000/slice_000/cell_000', 'prots': [0], 'thr': 2.0, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm10b': {'dir': '2017.07.27_000/slice_000/cell_001', 'prots': [0], 'thr': 1.75, 'rt': 0.35, 'decay': 5., 'G': 'F/F'},
            'm10c': {'dir': '2017.07.27_000/slice_000/cell_002', 'prots': [0], 'thr': 2.25, 'rt': 0.35, 'decay': 3.5, 'G': 'F/F'},
            # m10c, run 2: suspicious bursts
            'm10d': {'dir': '2017.07.27_000/slice_000/cell_003', 'prots': [0,1,2], 'thr': 1.5, 'rt': 0.35, 'decay': 4., 'G': 'F/F'},
            #'m10e': {'dir': '2017.07.27_000/slice_000/cell_004', 'prots': [1], 'thr': 1.5, 'rt': 0.35, 'decay': 4., 'G': 'F/F'},  # unstable and bursty
#
#  more:
#
            'm11a': {'dir': '2017.08.10_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.25, 'rt': 0.35, 'decay': 6, 'G': 'F/F'},
            'm12a': {'dir': '2017.08.11_000/slice_000/cell_000', 'prots': [0,1,2], 'thr': 1.0, 'rt': 0.35, 'decay': 4., 'G': 'F/F'},
            'm13b': {'dir': '2017.08.15_000/slice_000/cell_001', 'prots': [1,2], 'thr': 1.0, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},
            'm13c': {'dir': '2017.08.15_000/slice_000/cell_002', 'prots': [1,2], 'thr': 3.0, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},  # protocol minis_000 not very good - removed
            #'m13d': {'dir': '2017.08.15_000/slice_000/cell_003', 'prots': [0,1,2], 'thr': 2.0, 'rt': 0.35, 'decay': 4., 'G': 'F/+'}, # quite variable rate in runs 0 and 1 - dropped entire recording
            'm13e': {'dir': '2017.08.15_000/slice_000/cell_004', 'prots': [3], 'thr': 1.75, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},  # runs 1 and 2 had burstiness and instability - dropped
            # m13f has weird bursts - exclude
            #'m13f': {'dir': '2017.08.15_000/slice_000/cell_005', 'prots': [0,1,2], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'F/+'},
            # cells 006 and 007 are no good for 8/15
            # cells 000, 001 002 ng for 8/16
            'm14d': {'dir': '2017.08.16_000/slice_000/cell_003', 'prots': [1,2], 'thr': 1.5, 'rt': 0.35, 'decay': 4., 'G': 'F/F'},  # dropped run 0 - had unstable traces (baseline drift)
            
            }