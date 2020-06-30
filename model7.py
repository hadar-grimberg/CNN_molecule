
# -*- coding: utf-8 -*-
"""
Hadar Grimberg
6/3/2020

"""


import os
from datetime import datetime
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, AveragePooling2D, MaxPooling2D, ZeroPadding2D, Flatten, ZeroPadding3D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.losses import mean_absolute_error as mae
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, r'C:\Users\Hadar Grimberg\PycharmProjects\ML\SMILESrepresentation')
from feature import *
import SCFPfunctions as Mf
import SCFPmodel as Mm
from sklearn.model_selection import KFold
from collections import Counter


def mol2mol_supplier (file=None,sanitize=True):
    mols={}
    with open(file, 'r') as f:
        line =f.readline()
        while not f.tell() == os.fstat(f.fileno()).st_size:
            if line.startswith("@<TRIPOS>MOLECULE"):
                mol = []
                mol.append(line)
                line = f.readline()
                while not line.startswith("@<TRIPOS>MOLECULE"):
                    mol.append(line)
                    line = f.readline()
                    if f.tell() == os.fstat(f.fileno()).st_size:
                        mol.append(line)
                        break
                mol[-1] = mol[-1].rstrip() # removes blank line at file end
                block = ",".join(mol).replace(',','')
                m=Chem.MolFromMol2Block(block,sanitize=sanitize)
            if m.GetProp('_Name') in mols.keys():
                del(mols[m.GetProp('_Name')])
            else:
                mols[m.GetProp('_Name')]=m
    return(mols)

filePath =r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\Data\aligned.mol2'
database=mol2mol_supplier(filePath,sanitize=True)

f = open('CV_log.txt', 'w')

max_len=0
F_list, T_list = [],[]
for mol in database.values():
    mol_h = Chem.AddHs(mol)
    if len(Chem.MolToSmiles(mol_h, kekuleSmiles=True, isomericSmiles=True)) > 400:
        f.write(mol_h.GetProp('_Name'), "too long mol was ignored\n")
    else:
        F_list.append(mol_to_feature(mol_h,-1,240))
        T_list.append(mol_h.GetProp('_Name') )
        if len(Chem.MolToSmiles(mol_h, kekuleSmiles=True, isomericSmiles=True)) > max_len:
            max_len=len(Chem.MolToSmiles(mol_h, kekuleSmiles=True, isomericSmiles=True))
print ("The longest smile containes: ", max_len)

f.write("Reshape the Dataset...\n")
Mf.random_list(F_list)
Mf.random_list(T_list)
data_t = np.asarray(T_list).reshape(-1,1)
data_f = np.asarray(F_list, dtype=np.float32).reshape(-1,1,240,42)
data_t=np.column_stack([data_t, np.zeros([len(data_t),1])])
f.write('{0}\t{1}\n'.format(data_t.shape, data_f.shape))

mol_scores= open(r"C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\Data\summary_2.0.sort", 'r')
for line in mol_scores:
    mol_name = line.split('_')[0].strip()
    val = float(line.split(',')[4].strip())
    if mol_name in data_t:
        data_t[np.where(data_t==mol_name)[0][0],1]=val

y=data_t[:,1].astype(float)
# np.save('x_data.npy',data_f)
# np.save('y_data.npy',y)
# X_train, X_test, y_train, y_test = train_test_split(data_f, y, test_size=.2, random_state=5)


def cnn_model7():
  initializer = RandomNormal(mean=0., stddev=0.01)

  input_shape=(1,240,42)

  model = Sequential()
  model.add(ZeroPadding2D((0,5),input_shape=input_shape))
  model.add(Conv2D(128, kernel_size=(11, 42), strides=(1, 1), activation='relu',
                  data_format='channels_first', kernel_initializer=initializer))
  model.add(BatchNormalization(axis=1))
  model.add(AveragePooling2D(pool_size=(5,1), padding='same', strides=1,data_format='channels_first'))
  model.add(Conv2D(64, kernel_size=(11, 1), strides=1,
                  padding='same', activation='relu', kernel_initializer=initializer,
                  data_format='channels_first'))
  model.add(BatchNormalization(axis=1))
  model.add(AveragePooling2D(pool_size=(5,1), padding='same', strides=1,data_format='channels_first'))
  model.add(MaxPooling2D(pool_size=(240,1),data_format='channels_first'))
  model.add(Flatten())
  model.add(Dense(32,activation='relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
  model.add(BatchNormalization(axis=1))
  model.add(Dense(1))

  # training
  print ("Starting to train")
  model.compile(loss=mae,
                optimizer=Adam(lr=0.00005),
                metrics=['mae'])

  return model

model = cnn_model7()
mae_per_fold7 = []
loss_per_fold7 = []
kfold = KFold(n_splits=5, shuffle=True, random_state=7)

# KFold_CNN=0
# for train, test in kfold.split(x_train, y_train):
#   model7 = cnn_model7()
#   KFold_CNN+=1
#   file_path = "weights_model7_{}.best.h5".format(KFold_CNN)
#   model_checkpoint7 = ModelCheckpoint(file_path, verbose=1, save_best_only=True, monitor='val_loss', mode='min')
#   tb7 = TensorBoard(r"/content/drive/My Drive/mols/TB-model7_{}_".format(KFold_CNN) + datetime.now().strftime("%Y%m%d-%H%M%S"),
#                  histogram_freq=5, write_images=True, update_freq=100)
#   history7 = model7.fit(x_train[train], y_train[train],
#               batch_size=70, validation_data=(x_train[test], y_train[test]),
#               epochs=1200, callbacks=[model_checkpoint7, tb7],
#               validation_split=0.2)
#
#   # Generate generalization metrics
#   scores = model7.evaluate(x_train[test], y_train[test], verbose=0)
#   print(f'Score for fold {fold_no}: {model7.metrics_names[0]} of {scores[0]}; {model7.metrics_names[1]} of {scores[1]}%')
#   mae_per_fold7.append(scores[1])
#   loss_per_fold7.append(scores[0])
#   # restart model
#   del(model7)
#     # Increase fold number
#   fold_no = fold_no + 1


# for i in range(0, len(mae_per_fold10)):
#     print('------------------------------------------------------------------------')
#     print(f'> Fold {i + 1} - Loss: {loss_per_fold10[i]} - MAE: {mae_per_fold10[i]}%')
# print('------------------------------------------------------------------------')
# print('Average scores for all folds:')
# print(f'> MAE: {np.mean(mae_per_fold10)} (+- {np.std(mae_per_fold10)})')
# print(f'> Loss: {np.mean(loss_per_fold10)}')
# print('------------------------------------------------------------------------')


def eval_new_20_mols(eval_data):
    max_len = 0
    F_list, T_list = [], []
    for mol in eval_data.values():
        mol_h = Chem.AddHs(mol)
        print(len(Chem.MolToSmiles(mol_h, kekuleSmiles=True, isomericSmiles=True)))
        if len(Chem.MolToSmiles(mol_h, kekuleSmiles=True, isomericSmiles=True)) > 240:
            print(mol_h.GetProp('_Name'), "too long mol was ignored\n")
            continue
        else:
            F_list.append(mol_to_feature(mol_h, -1, 240))
            T_list.append(mol_h.GetProp('_Name'))
            if len(Chem.MolToSmiles(mol_h, kekuleSmiles=True, isomericSmiles=True)) > max_len:
                max_len = len(Chem.MolToSmiles(mol_h, kekuleSmiles=True, isomericSmiles=True))
    print("The longest smile containes: ", max_len)
    data_t = np.asarray(T_list).reshape(-1, 1)
    data_f = np.asarray(F_list, dtype=np.float32).reshape(-1,1,240,42)
    return data_t,  data_f

def normalize_maxpooling_layer(model,x_data):
    get_7_layer_output = backend.function([model.layers[0].input],
                                    [model.layers[7].output])
    layer_output = (get_7_layer_output([x_data])[0]).squeeze()
    normalized_output_layer = np.zeros(layer_output.shape)
    for i in range(layer_output.shape[1]):
       normalized_output_layer[:, i] = (layer_output[:, i] - np.mean(layer_output[:, i])) / np.std(layer_output[:, i])
    return normalized_output_layer

def motif_freq(norm_maxpool):
    # nin_bla = norm_maxpool[np.where(np.logical_or(norm_maxpool > 2.58, norm_maxpool < -2.58))]
    # The most influancial motifs for each molecule
    indecies_2_58 = np.where(norm_maxpool > 2.58)
    indecies_dict = {}
    for i, j in zip(indecies_2_58[0], indecies_2_58[1]):
        try:
            indecies_dict[i].append(j)
        except KeyError:
            indecies_dict[i] = [j]
    # sig_norm_maxpool = [np.argmax(norm_maxpool[:,i]) for i in range(norm_maxpool.shape[1]) if np.max(norm_maxpool[:,i])>=np.mean(np.max(norm_maxpool,axis=0))]
    return indecies_dict

def top10_and_group_saperation(indecies_dict):
    no_inhibition = {}
    inhibitor = {}
    weak_inhibitor = {}
    no_inhib=0
    inhib=0
    weak_inhib=0
    model_predictions = model.predict(x)
    for i, j in enumerate(model_predictions):
        if j <= np.percentile(model_predictions, [5]):
            inhib+=1
            if i in indecies_dict.keys():
                inhibitor[i] = indecies_dict[i]
        elif j > np.mean(model_predictions):
            no_inhib+=1
            if i in indecies_dict.keys():
                no_inhibition[i] = indecies_dict[i]
        else:
            if i in indecies_dict.keys():
                weak_inhib+=1
                weak_inhibitor[i] = indecies_dict[i]

    inhibitor_cnt = Counter([j for i in inhibitor.values() for j in i])
    weak_inhibitor_cnt = Counter([j for i in weak_inhibitor.values() for j in i])
    no_inhibition_cnt = Counter([j for i in no_inhibition.values() for j in i])

    inhibitors_common_motifs = inhibitor_cnt.most_common(10)
    weak_inhibitor_common_motifs = weak_inhibitor_cnt.most_common(10)
    no_inhibition_common_motifs = no_inhibition_cnt.most_common(10)

    # for i in range(len(inhibitors_common_motifs), 0, -1):
    #     if inhibitors_common_motifs[i - 1][1] < np.round(len(inhibitor) / 2):
    #         del (inhibitors_common_motifs[i - 1])
    #     if weak_inhibitor_common_motifs[i - 1][1] < np.round(len(weak_inhibitor) / 2):
    #         del (weak_inhibitor_common_motifs[i - 1])
    #     if no_inhibition_common_motifs[i - 1][1] < np.round(len(no_inhibition) / 2):
    #         del (no_inhibition_common_motifs[i - 1])

    inhibitors_common_motifs1 = pd.DataFrame(np.array([inhibitors_common_motifs]).squeeze(),
                                             columns=['Inhibitor_mol', 'Inhibitor_count'])
    weak_inhibitor_common_motifs1 = pd.DataFrame(np.array([weak_inhibitor_common_motifs]).squeeze(),
                                                 columns=['Weak_mol', 'Weak_count'])
    no_inhibition_common_motifs1 = pd.DataFrame(np.array([no_inhibition_common_motifs]).squeeze(),
                                                columns=['No_inhibitor_mol', 'No_inhibitor_count'])

    top10 = pd.concat([inhibitors_common_motifs1, weak_inhibitor_common_motifs1, no_inhibition_common_motifs1], axis=1)

    for i in top10['Weak_mol']:
        for j in top10['No_inhibitor_mol']:
            if i == j:
                print("mol #", i, " in Weak_mol is: ", top10['Weak_count'][top10.index[top10['Weak_mol'] == i][0]],
                      " in No_inhibitor_mol is: ",
                      top10['No_inhibitor_count'][top10.index[top10['No_inhibitor_mol'] == i][0]])
        for j in top10['Inhibitor_mol']:
            if i == j:
                print("mol #", i, " in Weak_mol is: ", top10['Weak_count'][top10.index[top10['Weak_mol'] == i][0]],
                      " in inhibitor_mol is: ", top10['Inhibitor_count'][top10.index[top10['Inhibitor_mol'] == i][0]])

    for i in top10['Inhibitor_mol']:
        for j in top10['No_inhibitor_mol']:
            if i == j:
                print("mol #", i, " in Inhibitor_mol is: ",
                      top10['Inhibitor_count'][top10.index[top10['Inhibitor_mol'] == i][0]],
                      " in No_inhibitor_mol is: ",
                      top10['No_inhibitor_count'][top10.index[top10['No_inhibitor_mol'] == i][0]])
    return(inhibitor, weak_inhibitor,no_inhibition, top10)



if __name__=='__main__':
    model = cnn_model7()
    # model.load_weights(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\SMILE_CNN\model7\weights_model7.best.h5')
    model.load_weights(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\SMILE_CNN\model7\weights_model7_1_goodPadding.best.h5')
    evaluate = []
    mmm=Chem.MolFromMol2File(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\zinc_22812775.mol2')
    mm=Chem.MolFromMol2File(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\VS-28.mol2')
    dict1 = mol2mol_supplier(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\newmols.mol2', sanitize=True)
    dict2 = mol2mol_supplier(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\twisted.mol2', sanitize=True)
    dict1.update(dict2)
    dict1[mmm.GetProp('_Name')] = mmm
    dict1[mm.GetProp('_Name')] = mm
    newmols_names, newmols_data= eval_new_20_mols(dict1)

    x_train = np.load(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\X_train.npy')
    x_test = np.load(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\X_test.npy')
    y_train = np.load(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\y_train.npy').astype(float)
    y_test = np.load(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\y_test.npy').astype(float)

    x_train = np.delete(x_train, slice(240, 400), axis=2)
    x_test = np.delete(x_test, slice(240, 400), axis=2)

    # kfold = KFold(n_splits=5, shuffle=True, random_state=7)
    #
    # w4 = [r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\SMILE_CNN\model7\weights_model7_1_1.best.h5',
    #       r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\SMILE_CNN\model7\weights_model7_1_2.best.h5',
    #       r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\SMILE_CNN\model7\weights_model7_1_3.best.h5',
    #       r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\SMILE_CNN\model7\weights_model7_1_4.best.h5',
    #       r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\SMILE_CNN\model7\weights_model7_1_5.best.h5']
    #
    #
    # for i, (train, test) in enumerate(kfold.split(x_train, y_train)):
    #     model = cnn_model7()
    #     model.load_weights(w4[i])
    #     eval = model.evaluate(x_train[test], y_train[test])
    #     evaluate.append(eval[1])
    # print (np.mean(evaluate))
        # print("model: ",i+1, model.evaluate(data_f[test], y[test]))
        # # model.summary()
        # newmols_predictions = model.predict(newmols_data)
        # if i == 0:
        #     predictions = np.column_stack([newmols_names, newmols_predictions])
        # else:
        #     predictions = np.column_stack([predictions, newmols_predictions])
        # del (model)

    eval = model.evaluate(x_test, y_test)
    newmols_predictions = model.predict(newmols_data)
    # newmols_predictions2 = model.predict(newmols_data2)
    # predictions = np.column_stack([np.row_stack([newmols_names,newmols_names2]), np.row_stack([newmols_predictions,newmols_predictions2])])
    predictions = np.column_stack([newmols_names, newmols_predictions])
    predictions = np.column_stack([newmols_names, newmols_predictions])
    predictions = pd.DataFrame(predictions)
    # predictions.to_excel("model7_1_CV5_240x42.xlsx")
    x = np.concatenate((x_train, x_test))
    # normalize the max pooling layers
    norm_maxpool= normalize_maxpooling_layer(model,x)
    indecies_dict=motif_freq(norm_maxpool)
    inhibitor, weak_inhibitor,no_inhibition, top10=top10_and_group_saperation(indecies_dict)

    molsss20_norm_maxpool = normalize_maxpooling_layer(model, newmols_data)
    molsss20 = motif_freq(molsss20_norm_maxpool)

