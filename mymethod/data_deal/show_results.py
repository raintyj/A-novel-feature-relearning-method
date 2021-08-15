import numpy as np
import  pandas as pd
import os
np.set_printoptions(suppress=True)

my_class = 4 # 4 , 5

def depart3datalist(data_dir):
    dataf1 = np.zeros((my_class, my_class))
    allfiles = os.listdir(data_dir)
    npzfiles = []
    for idx, f in enumerate(allfiles):
        if "4_view_test_confusion" in f:  #  4_view_test_confusion , all_view_test_confusion
            npzfiles.append(os.path.join(data_dir, f))
    npzfiles.sort()
    for npz_f in npzfiles:
        data = pd.DataFrame(pd.read_csv(npz_f, header=0)).values[:, 1:]
        dataf1 = dataf1 + data
    print(dataf1)
    print(np.sum(dataf1))
    return dataf1

if __name__ == '__main__':
    dir =  "/results/fpz_5k/"  # /results/pz_5k/
    sumlabel = depart3datalist(dir)
    accsum = 0
    for i in range(my_class):
        accsum += sumlabel[i][i]
    print("acc :{0}".format(accsum / np.sum(sumlabel)))

    # Precision
    pr = []
    for i in range(my_class):
        pr.append(sumlabel[i][i]/np.sum(sumlabel[:,i]))
    # Recall
    re = []
    for i in range(my_class):
        re.append(sumlabel[i][i]/np.sum(sumlabel[i,:]))
    # F1
    pr = np.array(pr)
    re = np.array(re)
    f1 = (2*pr*re) / (pr+re)
    print("re :{0}".format(re))
    print("pr :{0}".format(pr))
    print("f1 :{0}".format(f1))
    print("MF1 :{0}".format(np.sum(f1)/my_class))



