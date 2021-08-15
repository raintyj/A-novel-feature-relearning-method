import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第二块GPU（从0开始）
from sklearn.model_selection import StratifiedKFold


def load_npz_file(npz_file):
    with np.load(npz_file) as f:
        data = f["x"].reshape(-1,3000)
        labels = f["y"].reshape(-1,1)
    return data, labels

#标签合并
def label_merge(labels):
    mer_labels = []
    labels = labels.reshape(-1)
    for temp in labels:
        if temp==1 or temp == 4:
           mer_labels.append(1)
        else:
           mer_labels.append(temp)
    mer_labels = np.array(mer_labels)
    mer_labels = mer_labels.reshape(-1,1)
    return mer_labels

def load_npz_list_files(npz_files):
        data = np.zeros(shape=(0, 3000))
        labels = np.zeros(shape=(0, 1))
        merge_labels = np.zeros(shape=(0, 1))
        for npz_f in npz_files:
            tmp_data, tmp_labels = load_npz_file(npz_f)
            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.long)  # array
            tmp_merge_labels =  label_merge(tmp_labels)
            data = np.concatenate((data, tmp_data), axis=0)
            labels = np.concatenate((labels, tmp_labels), axis=0)
            merge_labels = np.concatenate((merge_labels, tmp_merge_labels), axis=0)
        return data, labels,merge_labels

def depart_files(data_dir):
        allfiles = os.listdir(data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(data_dir, f))
        npzfiles.sort()
        x,y,z = load_npz_list_files(npzfiles)
        return x,y,z


if __name__ == '__main__':
    fpz_dir = "/data/eeg_fpz_cz/"
    output_dir = "/data/fpz_5k/"

    data,labels,merge_labels = depart_files(fpz_dir)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)

    i = 0
    for train_index, test_index in skf.split(data, labels):
        i = i + 1
        data_train, data_test = data[train_index], data[test_index]
        label_train, label_test = labels[train_index], labels[test_index]
        merge_label_train, merge_label_test = merge_labels[train_index], merge_labels[test_index]
        np.savez(output_dir + "train_" + str(i) + ".npz",
                 data=data_train, label=label_train,merge_label =merge_label_train)
        np.savez(output_dir + "test_" + str(i) + ".npz",
                 data=data_test, label=label_test,merge_label =merge_label_test)
        print("finish {0}".format(i))
    print('---end---')