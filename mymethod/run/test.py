import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from model.my_model import model_part1,model_part2

if __name__ == '__main__':
    for test_time in range(1, 6):
        save_dir_4_test = "/data/fpz_5k/test_" + str(test_time) + ".npz"
        test_data = np.load(save_dir_4_test)
        data_test = test_data["data"].reshape(-1, 3000, 1)
        label_test_5 = test_data["label"].reshape(-1, 1)
        label_test_4 = test_data["merge_label"].reshape(-1, 1)
        # ------------------------------------test model1---------------------------------------------------
        model_file_path_4 = "/results/fpz_5k/4_modelsave_" + str(test_time) + ".h5"
        model_1 = model_part1()
        model_1.load_weights(model_file_path_4)
        y_pred = model_1.predict(data_test, batch_size=None)
        y_pred = np.array(np.argmax(y_pred, axis=1)).reshape(-1, 1)
        # ------------------ confusion ------------------
        conf_matrix = confusion_matrix(label_test_4, y_pred)
        pathconf = "/results/fpz_5k/4_view_test_confusion_" + str(test_time) + ".csv"
        pd.DataFrame(conf_matrix).to_csv(pathconf)
        # ------------------ F1 ------------------
        y_model1_true = label_test_4.tolist()
        y_model1_pred = y_pred.tolist()
        classification = classification_report(y_model1_true, y_model1_pred,
                                               target_names=['W', 'N 1-REM', 'N 2', 'N 3'],
                                               output_dict=True)
        classification_result = pd.DataFrame(classification).transpose()
        dir_class_result = "/results/fpz_5k/4_perF1_" + str(test_time) + ".csv"
        classification_result.to_csv(dir_class_result, index=True)
        # ------------------------------------test model2---------------------------------------------------
        y_pred_4_test = model_1.predict(data_test)
        y_pred_4_test = np.argmax(y_pred_4_test, axis=1)
        # ---------------Data selection is based on the prediction results of model 1 ----------------------
        need_index_2 = [i for i in range(len(y_pred_4_test)) if y_pred_4_test[i] == 1]
        data_in_model2 = data_test[need_index_2]
        model_2 = model_part2()
        model_file_path_2 = "/results/fpz_5k/2_modelsave_" + str(test_time) + ".h5"
        model_2.load_weights(model_file_path_2)
        y_pred_2 = model_2.predict(data_in_model2, batch_size=None)
        y_pred_2 = np.array(np.argmax(y_pred_2, axis=1)).reshape(-1, 1)
        # ------------------------------------Merge the results------------------------------------------------
        y_pred_2[(y_pred_2 == 0)] = 4
        y_pred_4_test = y_pred_4_test.reshape(-1, 1)
        y_pred_4_test[need_index_2] = y_pred_2
        # ------------------ confusion ------------------
        conf_matrix = confusion_matrix(label_test_5, y_pred_4_test)
        pathconf = "/results/fpz_5k/all_view_test_confusion_" + str(test_time) + ".csv"
        pd.DataFrame(conf_matrix).to_csv(pathconf)
        # ------------------ F1 ------------------
        y_model2_true = label_test_5.tolist()
        y_model2_pred = y_pred_4_test.tolist()
        classification = classification_report(y_model2_true, y_model2_pred,
                                               target_names=['W', 'N 1', 'N 2', 'N 3', 'REM'],
                                               output_dict=True)
        classification_result = pd.DataFrame(classification).transpose()
        dir_class_result = "/results/fpz_5k/all_perF1_" + str(test_time) + ".csv"
        classification_result.to_csv(dir_class_result, index=True)
        print("-------end {0}--------".format(test_time))
