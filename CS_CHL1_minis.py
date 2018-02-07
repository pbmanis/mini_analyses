basepath = '/Users/experimenters/Data/Chelsea/CHL1/'

datasets = {
	'm2a': {'dir': '2017.11.01_000/slice_000/cell_000', 'prots': [1, 2], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'WT'},
	'm2c': {'dir': '2017.11.01_000/slice_000/cell_006', 'prots': [0, 1, 2, 3], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'WT'},
    'm4a': {'dir': '2017.11.06_000/slice_000/cell_000', 'prots': [0, 1], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'WT'},
    'm4b': {'dir': '2017.11.06_000/slice_000/cell_001', 'prots': [0, 1, 2, 3], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'WT'},
    'm5a': {'dir': '2017.11.07_000/slice_000/cell_000', 'prots': [0, 1, 2, 3, 4, 5], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'CHL1'},
    'm5b': {'dir': '2017.11.07_000/slice_000/cell_001', 'prots': [0, 1, 2], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'CHL1'},
    'm5c': {'dir': '2017.11.07_000/slice_001/cell_000', 'prots': [0], 'thr': 2.0, 'rt': 0.4, 'decay': 5., 'G': 'CHL1'},
    'm6a': {'dir': '2017.11.09_000/slice_000/cell_000', 'prots': [0, 1, 2, 3, 4], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'CHL1'},
    'm7a': {'dir': '2017.11.20_000/slice_000/cell_000', 'prots': [0, 1, 2, 3, 4], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'CHL1'},
    'm7b': {'dir': '2017.11.20_000/slice_000/cell_001', 'prots': [3, 4, 5, 6, 7, 8], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'CHL1'},
    'm7c': {'dir': '2017.11.20_000/slice_000/cell_002', 'prots': [0], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'CHL1'},
    'm8a': {'dir': '2017.11.21_000/slice_000/cell_000', 'prots': [0, 1, 2], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'CHL1'},
    'm8b': {'dir': '2017.11.21_000/slice_000/cell_001', 'prots': [0, 1, 2, 3, 4], 'thr': 2.5, 'rt': 0.35, 'decay': 4., 'G': 'CHL1'},
    'm8c': {'dir': '2017.11.21_000/slice_000/cell_002', 'prots': [0, 1, 2], 'thr': 3, 'rt': 0.35, 'decay': 4., 'G': 'CHL1'},
}

excludes = {
    ('m5c', 0): [8],
    ('m5a', 1): [],

}
