import numpy as np

def print_n_samples_each_class(labels,label_type):
    unique_labels = np.unique(labels)
    classlen = []
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        classlen.append(n_samples)
    return np.array(classlen).reshape(1,label_type)

def load_npz_file(npz_file):
    with np.load(npz_file) as f:
        data = f["x"].reshape(-1,3000,1)
        labels = f["y"].reshape(-1,1)
    return data, labels