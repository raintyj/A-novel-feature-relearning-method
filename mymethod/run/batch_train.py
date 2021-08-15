import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import RandomOverSampler
from model.my_model import model_part1,model_part2

if __name__ == '__main__':
    epochs = 2000
    batch_size = 128

    for train_time in range(1,6):
        # ----------------------------------------- train model1 ------------------------------------------
        save_dir_train = "/data/fpz_5k/train_" + str(train_time) + ".npz"
        train_data = np.load(save_dir_train)
        data_train = train_data["data"].reshape(-1, 3000, 1)
        label_train_5 = train_data["label"].reshape(-1, 1)
        label_train_4 = train_data["merge_label"].reshape(-1, 1)
        model_file_path_4 = "/results/fpz_5k/4_modelsave_" + str(train_time) + ".h5"
        model_1 = model_part1()
        checkpoint_4 = ModelCheckpoint(model_file_path_4, monitor='loss', mode="auto", verbose=1, save_best_only=True)
        early_1 = EarlyStopping(monitor="loss", mode="auto", patience=8, verbose=1)
        redonplat_4 = ReduceLROnPlateau(monitor="loss", mode="auto", patience=5, factor=0.5, verbose=2)
        callbacks_list_4 = [checkpoint_4, early_1, redonplat_4]
        history = model_1.fit(data_train, label_train_4,
                            batch_size=batch_size,
                            callbacks=callbacks_list_4,
                            epochs=epochs,
                            )
        # ---------------Data selection is based on the prediction results of model 1 ----------------------
        model_1.load_weights(model_file_path_4)
        y_pred_4_train = model_1.predict(data_train)
        y_pred_4_train = np.argmax(y_pred_4_train, axis=1)
        need_index_train = [i for i in range(len(y_pred_4_train)) if
                            y_pred_4_train[i] == 1 and y_pred_4_train[i] == label_train_4[i]]
        x_train_4 = data_train[need_index_train]
        y_train_true_5 = label_train_5[need_index_train] # 1 or 4
        y_train_true_5_copy = np.tile(y_train_true_5, 1)
        y_train_true_5_copy[(y_train_true_5_copy == 4)] = 0
        # -------------------------------------oversampled---------------------------------------------------
        ros = RandomOverSampler(random_state=2)
        x_train_4 = x_train_4.reshape(-1,3000)
        over_x_train_4, over_y = ros.fit_resample(x_train_4, y_train_true_5_copy)
        over_x_train_4 = over_x_train_4.reshape(-1,3000,1)
        # ------------------------------------train model2---------------------------------------------------
        model_2 = model_part2()
        model_file_path_2 = "/results/fpz_5k/2_modelsave_" + str(train_time) + ".h5"
        checkpoint_2 = ModelCheckpoint(model_file_path_2, monitor='loss', mode="auto", verbose=1, save_best_only=True)
        early_2 = EarlyStopping(monitor="loss", mode="auto", patience=8, verbose=1)
        redonplat_2 = ReduceLROnPlateau(monitor="loss", mode="auto", patience=5, factor=0.5, verbose=2)
        callbacks_list = [checkpoint_2, early_2, redonplat_2]
        history = model_2.fit(over_x_train_4 ,over_y,
                              batch_size=batch_size,
                              callbacks=callbacks_list,
                              epochs=epochs,
                              )
        print("-------end {0}--------".format(train_time))

