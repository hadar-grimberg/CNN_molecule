# -*- coding: utf-8 -*-
"""
Hadar Grimberg
6/3/2020

"""

import os
from argparse import ArgumentError
from datetime import datetime
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, AveragePooling2D, MaxPooling2D, ZeroPadding2D, Flatten, ZeroPadding3D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import mean_absolute_error as mae
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from collections import Counter
import sys
sys.path.insert(1, r'C:\path\')
from feature import *
import SCFPfunctions as Mf



# read mol2 file and convert it to a dataframe contains mol objects
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

# read from dataframe and convert each molecule to a SMILES and then to a matrix made of one-hot vectors
def build_dataset(data_for_training):
    filePath =r'C:\path\mols.mol2'
    database=mol2mol_supplier(filePath,sanitize=True)

    max_len=0
    F_list, T_list, smiles = [],[],[]
    for mol in database.values():
        mol_h = Chem.AddHs(mol)
        if len(Chem.MolToSmiles(mol_h, kekuleSmiles=True, isomericSmiles=True)) > 240:
            print(mol_h.GetProp('_Name'), "too long mol was ignored\n")
        else:
            F_list.append(mol_to_feature(mol_h,-1,240))
            T_list.append(mol_h.GetProp('_Name') )
            if len(Chem.MolToSmiles(mol_h, kekuleSmiles=True, isomericSmiles=True)) > max_len:
                max_len=len(Chem.MolToSmiles(mol_h, kekuleSmiles=True, isomericSmiles=True))
        if data_for_training == False:
            try:
                isomerSMILES = Chem.MolToSmiles(mol_h, kekuleSmiles=True, isomericSmiles=True, rootedAtAtom=int(-1))
            except:
                isomerSMILES = Chem.MolToSmiles(mol_h, kekuleSmiles=True, isomericSmiles=True)
            smiles.append (isomerSMILES)
    print ("The longest smile containes: ", max_len)

    if data_for_training:
        Mf.random_list(F_list)
        Mf.random_list(T_list)
        data_f = np.asarray(F_list, dtype=np.float32).reshape(-1, 1, 240, 42)
    else:
        smiles = np.asarray(smiles).reshape(-1,1)
        data_f = np.asarray(F_list, dtype=np.float32).reshape(-1, 1, 240, 42)
    data_t = np.asarray(T_list).reshape(-1,1)
    data_t=np.column_stack([data_t, np.zeros([len(data_t),1]).astype(np.object)])

    # build y dataframe
    mol_scores= open(r"C:\path\summary.sort", 'r')
    for line in mol_scores:
        mol_name = line.split('_')[0].strip()
        val = float(line.split(',')[4].strip())
        if mol_name in data_t:
            data_t[np.where(data_t==mol_name)[0][0],1]=val

    if data_for_training:
        y=data_t[:,1]
        x_train, x_test, y_train, y_test = train_test_split(data_f, y, test_size=.2, random_state=5)
        return x_train, x_test, y_train, y_test
    else:
        return data_f, data_t, smiles


# CNN model used to train, validation, prediction and evaluation
def cnn_model7():
  initializer = RandomNormal(mean=0., stddev=0.01)
  input_shape=(1,240,42)

  model = Sequential()
  model.add(ZeroPadding2D((0,5),input_shape=input_shape, name="input_layer"))
  model.add(Conv2D(128, kernel_size=(11, 42), strides=(1, 1), activation='relu',
                  data_format='channels_first', kernel_initializer=initializer))
  model.add(BatchNormalization(axis=1))
  model.add(AveragePooling2D(pool_size=(5,1), padding='same', strides=1,data_format='channels_first'))
  model.add(Conv2D(64, kernel_size=(11, 1), strides=1,
                  padding='same', activation='relu', kernel_initializer=initializer,
                  data_format='channels_first'))
  model.add(BatchNormalization(axis=1))
  model.add(AveragePooling2D(pool_size=(5,1), padding='same', strides=1,data_format='channels_first'))
  model.add(MaxPooling2D(pool_size=(240,1),data_format='channels_first',name="maxpool"))
  model.add(Flatten())
  model.add(Dense(32,activation='relu', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
  model.add(BatchNormalization(axis=1))
  model.add(Dense(1))

  # compile
  model.compile(loss=mae,
                optimizer=Adam(lr=0.00005),
                metrics=['mae'])
  return model


# Train the model
def model_training_CV(x_train, y_train):
    mae_per_fold7 = []
    loss_per_fold7 = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=7)
    KFold_CNN = 0

    # Train with 5 cross validation groups
    for train, test in kfold.split(x_train, y_train):
      model=cnn_model7()
      KFold_CNN+=1
      file_path = "weights_model7_{}.best.h5".format(KFold_CNN)
      model_checkpoint7 = ModelCheckpoint(file_path, verbose=1, save_best_only=True, monitor='val_loss', mode='min')
      tb7 = TensorBoard(r"/content/drive/My Drive/mols/TB-model7_{}_".format(KFold_CNN) + datetime.now().strftime("%Y%m%d-%H%M%S"),
                     histogram_freq=5, write_images=True, update_freq=100)
      history7 = model.fit(x_train[train], y_train[train],
                  batch_size=70, validation_data=(x_train[test], y_train[test]),
                  epochs=1200, callbacks=[model_checkpoint7, tb7],
                  validation_split=0.2)

      # Generate generalization metrics
      scores = model.evaluate(x_train[test], y_train[test], verbose=0)
      print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}%')
      mae_per_fold7.append(scores[1])
      loss_per_fold7.append(scores[0])
      # restart model
      del(model)
        # Increase fold number
      fold_no = fold_no + 1

    for i in range(0, len(mae_per_fold7)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold7[i]} - MAE: {mae_per_fold7[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> MAE: {np.mean(mae_per_fold7)} (+- {np.std(mae_per_fold7)})')
    print(f'> Loss: {np.mean(loss_per_fold7)}')
    print('------------------------------------------------------------------------')

    return history7, np.mean(mae_per_fold7), np.mean(loss_per_fold7)


def model_training(x_train, y_train):
    model7 = cnn_model7()
    file_path = "weights_model7_1.best.h5"
    model_checkpoint7 = ModelCheckpoint(file_path, verbose=1, save_best_only=True, monitor='val_loss', mode='min')
    tb7 = TensorBoard(r"/content/drive/My Drive/mols/TB-model7_1_" + datetime.now().strftime("%Y%m%d-%H%M%S"),
                      histogram_freq=5, write_images=True, update_freq=100)
    model7.fit(x_train, y_train, batch_size=70, validation_data=(x_test, y_test),
               epochs=1200, callbacks=[model_checkpoint7, tb7])


# Build the new dataset of 20 molecules (SMILES converted to matrix of n-features x one-hot-vectors)
# These are test molecules without known score
def build_new_20_mols_dataset(twenty_mols_dict):
    max_len = 0
    F_list, T_list = [], []
    for mol in twenty_mols_dict.values():
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

# extract data from the maxpooling layer and normalize it into z-scores
# in order to find the most influencing filters on the score
def normalize_maxpooling_layer(model,x_data):
    get_7_layer_output = backend.function([model.layers[0].input],
                                    [model.layers[7].output])
    layer_output = (get_7_layer_output([x_data])[0]).squeeze()
    normalized_output_layer = np.zeros(layer_output.shape)
    for i in range(layer_output.shape[1]):
        if np.std(layer_output[:, i]) < 0.0001:
            normalized_output_layer[:, i] = 0
        else:
            normalized_output_layer[:, i] = (layer_output[:, i] - np.mean(layer_output[:, i])) / np.std(layer_output[:, i])
    return normalized_output_layer, layer_output

# find the relevant pixels in 2nd average-pooling layer and divide into 3 groups
def find_pixels(model,x_data, normalized_maxpool, maxpool_output,y):
    get_6_layer_output = backend.function([model.layers[0].input],
                                          [model.layers[6].output])
    pixels = pd.DataFrame(columns=["filter", "mol", "pixel"])
    avgpool_output = (get_6_layer_output([x_data])[0]).squeeze()
    for i in range(normalized_maxpool.shape[1]):
        for j in range(normalized_maxpool.shape[0]):
            if normalized_maxpool[j,i] >= 2.58:
                a = maxpool_output[j,i]
                feature= np.where(avgpool_output[j, i] == a)[0][0]
                pixels.loc[len(np.where(norm_maxpool > 2.58)[0])] = [i,j,feature]
                pixels.index = pixels.index - 1

    pixels['group'] = 0

    y = np.column_stack((y, np.zeros((y.shape[0], 1))))
    for i, j in enumerate(y[:,1]):
        # The top 5% classified as inhibitors
        if j < np.percentile(y[:,1], [5]):
            y[i,2]= 10 #'inhibitor'
        # The other molecules which had binding score of above the average, classified as weak inhibitors
        elif j < np.mean(y[:,1]):
            y[i, 2] = 5 # 'weak inhibitor'
        # The remaining molecules classified as non-inhibitors
        else:
            y[i,2]= 1 #'non-inhibitor'
    for i in pixels.index:
        pixels.loc[i,'group']= y[pixels.loc[i,'mol']][2]

    inhibitors_common_motifs = filters_data(pixels, group=10)
    weak_inhibitors_common_motifs = filters_data(pixels, group=5)
    no_inhibitors_common_motifs = filters_data(pixels, group=1)

    # print recurring motifs between groups
    for i in weak_inhibitors_common_motifs.index:
        for j in no_inhibitors_common_motifs.index:
            if i == j:
                print("mol #", i, " in Weak_mol is: ", weak_inhibitors_common_motifs.loc[i, 'Filter_count'],
                      " in No_inhibitor_mol is: ", no_inhibitors_common_motifs.loc[j, 'Filter_count'])
        for j in inhibitors_common_motifs.index:
            if i == j:
                print("mol #", i, " in Weak_mol is: ", weak_inhibitors_common_motifs.loc[i, 'Filter_count'],
                      " in inhibitor_mol is: ", inhibitors_common_motifs.loc[j, 'Filter_count'])

    for i in inhibitors_common_motifs.index:
        for j in no_inhibitors_common_motifs.index:
            if i == j:
                print("mol #", i, " in Inhibitor_mol is: ",
                      inhibitors_common_motifs.loc[i, 'Filter_count'],
                      " in No_inhibitor_mol is: ", no_inhibitors_common_motifs.loc[j, 'Filter_count'])

    return pixels, inhibitors_common_motifs, weak_inhibitors_common_motifs, no_inhibitors_common_motifs

# Take the top 10 most frequent filters in the group
# and build dataframe with count, mols and pixels for each filter
def filters_data(pixels, group):
    # Construct dictionaries for each filter with its mols and influential pixel
    mol_dict={}
    pixel_dict={}

    for filt in list(set([pixels.loc[i, 'filter'] for i in range(len(pixels.loc[:, 'filter'])) if pixels.loc[i, 'group'] == group])):
        mol_dict[filt]=[pixels.loc[i, 'mol'] for i in range(len(pixels.loc[:, 'mol'])) if pixels.loc[i, 'group'] == group
         and pixels.loc[i, 'filter'] == filt]
        pixel_dict[filt] = [pixels.loc[i, 'pixel'] for i in range(len(pixels.loc[:, 'mol'])) if pixels.loc[i, 'group'] == group
                          and pixels.loc[i, 'filter'] == filt]
    # Count mols for each filter
    filters_cnt=Counter()
    filters_cnt.update([keyy for keyy in mol_dict.keys() for i in range(len(mol_dict[keyy]))])

    # Exact the top 10 most common filter for each group and arrange them in a dataframe
    common_motifs = filters_cnt.most_common(10)

    # Insert everything into dataframes
    common_motifs = pd.DataFrame(np.array([common_motifs]).squeeze()[:,1],
                                    index=np.array([common_motifs]).squeeze()[:,0],
                                             columns=['Filter_count'])
    common_motifs['mols'] = 0
    common_motifs['pixels'] = 0
    common_motifs['mols'] = common_motifs['mols'].astype('object')
    common_motifs['pixels'] = common_motifs['pixels'].astype('object')
    for i in common_motifs.index:
        common_motifs.at[i,'mols']= mol_dict[i]
        common_motifs.at[i,'pixels']= pixel_dict[i]

    return common_motifs

# after normalizing choose only the top 1% of pixels from max-pooling
def motif_freq(norm_maxpool):
    # z-score >= 2.58 is 1% of the data
    indecies_2_58 = np.where(norm_maxpool > 2.58)
    indecies_dict = {}
                    # molecule         filter
    for i, j in zip(indecies_2_58[0], indecies_2_58[1]):
        try:
            indecies_dict[i].append(j)
        except KeyError:
            indecies_dict[i] = [j]
    return indecies_dict # {molecule: [filer/s]}


# Take the top 10 most frequent filters in each of the three groups
def top10_and_group_saperation(indecies_dict):
    no_inhibition = {}
    inhibitor = {}
    weak_inhibitor = {}
    no_inhib=0
    inhib=0
    weak_inhib=0

    # Classification of molecules into three groups inhibitors, weak inhibitors, non-inhibitors
    model_predictions = model.predict(x)
    for i, j in enumerate(model_predictions):
        # The top 5% classified as inhibitors
        if j <= np.percentile(model_predictions, [5]):
            inhib+=1
            if i in indecies_dict.keys():
                inhibitor[i] = indecies_dict[i]
        # The other molecules which had binding score of above the average, classified as weak inhibitors
        elif j > np.mean(model_predictions):
            weak_inhib += 1
            if i in indecies_dict.keys():
                weak_inhibitor[i] = indecies_dict[i]
        # The remaining molecules classified as non-inhibitors
        else:
            no_inhib += 1
            if i in indecies_dict.keys():
                no_inhibition[i] = indecies_dict[i]

    # Count the number of molecules that were influenced by each filter
    inhibitor_cnt = Counter([j for i in inhibitor.values() for j in i])
    weak_inhibitor_cnt = Counter([j for i in weak_inhibitor.values() for j in i])
    no_inhibition_cnt = Counter([j for i in no_inhibition.values() for j in i])

    # Exact the top 10 most common filter for each group and arrange them in a dataframe
    inhibitors_common_motifs = inhibitor_cnt.most_common(10)
    weak_inhibitor_common_motifs = weak_inhibitor_cnt.most_common(10)
    no_inhibition_common_motifs = no_inhibition_cnt.most_common(10)

    inhibitors_common_motifs1 = pd.DataFrame(np.array([inhibitors_common_motifs]).squeeze(),
                                             columns=['Inhibitor_mol', 'Inhibitor_count'])
    weak_inhibitor_common_motifs1 = pd.DataFrame(np.array([weak_inhibitor_common_motifs]).squeeze(),
                                                 columns=['Weak_mol', 'Weak_count'])
    no_inhibition_common_motifs1 = pd.DataFrame(np.array([no_inhibition_common_motifs]).squeeze(),
                                                columns=['No_inhibitor_mol', 'No_inhibitor_count'])

    top10 = pd.concat([inhibitors_common_motifs1, weak_inhibitor_common_motifs1, no_inhibition_common_motifs1], axis=1)

    # print recurring motifs between groups
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


"""
Calculate the receptive field according to this article:
https://distill.pub/2019/computing-receptive-fields/#computing-receptive-field-region
"""

# Compute receptive field bounderies for a single pixle in the 2nd average pooling layer
def receptive_field_compute(uL): #uL is the pixel location before maxpooling for receptive field calculation
    # find layers with padding and the input layer for max-pooling
    layers_p=[]
    for layer in range(len(model.layers) -1) :
        if "maxpool" in model.layers[layer].name:
            layer=layer-1
            break
        try:
            model.layers[layer].padding
        except AttributeError:
            continue
        layers_p.append(layer)

    # calculate padding and kernel size for each layer
    layers_p_p=[]
    kernels=[]
    for i in layers_p:
        if i==0:
            continue
        elif i==1:
            layers_p_p.append (((-model.layers[0].input_shape[2]) + model.layers[1].kernel_size[0] \
                    - (1 * 1) + (model.layers[1].output_shape[2] * 1)) / 2)
            kernels.append(model.layers[1].kernel_size[0])
        else:
            try:
                layers_p_p.append (((-model.layers[i].input_shape[2]) + model.layers[i].kernel_size[0] \
                 - (1 * 1) + (model.layers[i].output_shape[2] * 1)) / 2 )
                kernels.append(model.layers[i].kernel_size[0])
            except AttributeError:
                layers_p_p.append (((-model.layers[i].input_shape[2]) + model.layers[i].pool_size[0] \
                         -(1 * 1) + (model.layers[i].output_shape[2] * 1)) / 2)
                kernels.append(model.layers[i].pool_size[0])

    # Calculate effective padding for input image
    effective_padding = np.sum(layers_p_p)
    #r0 calculation:
    #r0 is the receptive field size in the input picture of a single pixle in average-pooling layer
    r0=0
    for k in kernels:
        r0+=(k-1)
    r0=r0+1
    #u0 is the hightest point in receptive field while v0 is the lowest one
    u0=-effective_padding+uL
    v0=u0+r0-1
    return u0, v0

def receptive_field_calculator(common_motifs,y):
    receptive_field = pd.DataFrame(columns=['Filter','Mol','Mol Name', 'u0', 'v0','Pixel'])
    a=0
    for filt in common_motifs.index:
        for mol in range(len(common_motifs.loc[filt,'mols'])):
            a+=1
            uL=common_motifs.loc[filt,'pixels'][mol]
            u0, v0 = receptive_field_compute(uL)
            receptive_field.loc[a, 'Filter'] = filt
            receptive_field.loc[a, 'u0'] = u0
            receptive_field.loc[a, 'v0'] = v0
            receptive_field.loc[a, 'Mol'] = common_motifs.loc[filt,'mols'][mol]
            receptive_field.loc[a, 'Mol Name'] = y[common_motifs.loc[filt,'mols'][mol],0]
            receptive_field.loc[a, 'Pixel'] = uL

    # unite receptive fields within the same mol
    receptive_field2 = pd.DataFrame(columns=['Filters', 'Mol', 'u0', 'v0', 'Pixels',
                                             'Filters_1', 'u0_1', 'v0_1', 'Pixels_1'])

    for i in receptive_field.loc[:,'Mol Name'].unique():
        lenm=len(receptive_field[receptive_field.loc[:,'Mol Name']==i].sort_values(by=['Pixel']))
        ind = receptive_field[receptive_field.loc[:, 'Mol Name'] == i].sort_values(by=['Pixel']).index[0]
        # Maximun distance of 5 lines to unite between receptive fields
        if lenm == 1:
            receptive_field2.loc[i,'Filters']= receptive_field.loc[ind, 'Filter']
            receptive_field2.loc[i,'Mol']= receptive_field.loc[ind, 'Mol']
            receptive_field2.loc[i,'u0']= receptive_field.loc[ind, 'u0']
            receptive_field2.loc[i,'v0']= receptive_field.loc[ind, 'v0']
            receptive_field2.loc[i,'Pixels']= receptive_field.loc[ind, 'Pixel']
        else:
            ln = 0
            while ln <= lenm-1:
                ind = receptive_field[receptive_field.loc[:, 'Mol Name'] == i].sort_values(by=['Pixel']).index[ln]
                if ln == lenm-1:
                    ind2 = receptive_field[receptive_field.loc[:, 'Mol Name'] == i].sort_values(by=['Pixel']).index[ln]
                else:
                    ind2 = receptive_field[receptive_field.loc[:, 'Mol Name'] == i].sort_values(by=['Pixel']).index[ln + 1]
                filts = set([receptive_field.loc[ind, 'Filter']])
                pix = set([receptive_field.loc[ind, 'Pixel']])
                new_u0 = receptive_field[receptive_field.loc[:, 'Mol Name'] == i].sort_values(by=['Pixel']).loc[
                    ind, 'u0']
                new_v0 = receptive_field[receptive_field.loc[:, 'Mol Name'] == i].sort_values(by=['Pixel']).loc[
                    ind, 'v0']
                while (receptive_field[receptive_field.loc[:,'Mol Name']==i].sort_values(by=['Pixel']).loc[ind2,'u0'] >=
                    receptive_field[receptive_field.loc[:,'Mol Name']==i].sort_values(by=['Pixel']).loc[ind,'u0']) and \
                    (receptive_field[receptive_field.loc[:, 'Mol Name'] == i].sort_values(by=['Pixel']).loc[ind2, 'u0'] <=
                    receptive_field[receptive_field.loc[:, 'Mol Name'] == i].sort_values(by=['Pixel']).loc[ind, 'v0'] + 5) \
                    and ln<=(lenm-1):
                        ind = receptive_field[receptive_field.loc[:, 'Mol Name'] == i].sort_values(by=['Pixel']).index[ln]
                        if ln == lenm - 1:
                            ind2 = receptive_field[receptive_field.loc[:, 'Mol Name'] == i].sort_values(by=['Pixel']).index[ln]
                        else:
                            ind2 = receptive_field[receptive_field.loc[:, 'Mol Name'] == i].sort_values(by=['Pixel']).index[ln + 1]
                        new_v0= receptive_field[receptive_field.loc[:,'Mol Name']==i].sort_values(by=['Pixel']).loc[ind2,'v0']
                        filts.add(receptive_field.loc[ind2, 'Filter'])
                        pix.add(receptive_field.loc[ind2, 'Pixel'])
                        ln+=1
                if receptive_field2.index.isin([i]).any():
                    receptive_field2.loc[i, 'Filters_1'] = [filts]
                    receptive_field2.loc[i, 'u0_1'] = new_u0
                    receptive_field2.loc[i, 'v0_1'] = new_v0
                    receptive_field2.loc[i, 'Pixels_1'] = [pix]
                else:
                    receptive_field2.loc[i, 'Filters'] = [filts]
                    receptive_field2.loc[i, 'Mol'] = receptive_field.loc[ind, 'Mol']
                    receptive_field2.loc[i, 'u0'] = new_u0
                    receptive_field2.loc[i, 'v0'] = new_v0
                    receptive_field2.loc[i, 'Pixels'] = [pix]
                ln += 1
    receptive_field2.loc[:, 'u0'] = receptive_field2.loc[:, 'u0'].astype(int)
    receptive_field2.loc[:, 'v0'] = receptive_field2.loc[:, 'v0'].astype(int)
    return receptive_field2, receptive_field


def get_fragment(receptive_field, smiles):
    for name in receptive_field.index:
        indecies=[]
        idx=0
        u0=receptive_field.loc[name, 'u0']
        v0=(receptive_field.loc[name, 'v0'])
        mol=receptive_field.loc[name, 'Mol']
        for i, c in enumerate(smiles[mol][0]):
            if islower(c) == True:
                continue
            elif isupper(c) == True:
                if c == 'H':
                    continue
                else:
                    idx = idx + 1
                    if i >= u0 and i <= v0:
                        indecies.append(idx)
            else:
                continue
        receptive_field.loc[name, 'SMILES'] = smiles[mol][0]
        try:
            receptive_field.loc[name, 'Fragment_smiles'] = Chem.MolFragmentToSmiles(
                Chem.MolFromSmiles(smiles[mol][0]), atomsToUse=indecies, allHsExplicit=True, isomericSmiles=True)
            receptive_field.loc[name,'Fragment']=Chem.MolToSmarts(Chem.MolFromSmiles(Chem.MolFragmentToSmiles(
                Chem.MolFromSmiles(smiles[mol][0]), atomsToUse=indecies, allHsExplicit=True, isomericSmiles=True)))
        except:
            continue
    return receptive_field

def draw_mol(receptive_field):
    for i in receptive_field.index:
        if pd.isnull(receptive_field.loc[i,"Fragment"]):
            continue
        else:
            mol = Chem.MolFromSmiles(receptive_field.loc[i,"SMILES"])
            patt = Chem.MolFromSmarts(receptive_field.loc[i,"Fragment"])
            hit_ats = list(mol.GetSubstructMatch(patt))
            hit_bonds = []
            for bond in patt.GetBonds():
               aid1 = hit_ats[bond.GetBeginAtomIdx()]
               aid2 = hit_ats[bond.GetEndAtomIdx()]
               hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
            d = rdMolDraw2D.MolDraw2DCairo(500, 500) # or MolDraw2DCairo to get PNGs
            d.drawOptions().legendFontSize = 20
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol, legend=i , highlightAtoms=hit_ats, highlightBonds=hit_bonds)
            output = d.GetDrawingText()
            with open(i +'image.png', 'wb') as pngf:
                   pngf.write(output)



if __name__=='__main__':
    # # Build the dataset, divide to 80% train and 20% test
    x_train, x_test, y_train, y_test = build_dataset(data_for_training=True)
    # Initialize the model
    model = cnn_model7()
    # Split the train dataset into 5 cross validation subgroups in order to choose a model
    history7, mean_MAE_CV, mean_loss_CV = model_training_CV(x_train, y_train)
    # After choosing a model, train the whole train dataset together, save weights and evaluate the test dataset
    model_training(x_train, y_train)
    model.summary()
    eval = model.evaluate(x_test, y_test)

    # After train and validation, examine which part of the picture (molecule sub-structure)
    # is contribute the most to the score
    x, y, smiles = build_dataset(data_for_training=False)
    # Normalize the max pooling layer's filters
    norm_maxpool, maxpool_output = normalize_maxpooling_layer(model, x)
    # after normalizing choose only the top 1% of pixels from max-pooling
    indecies_dict = motif_freq(norm_maxpool)
    # find the pixels of top 1% max-pooled variables, in previous layer
    pixels, inhibitors_common_motifs, weak_inhibitors_common_motifs, no_inhibitors_common_motifs = \
        find_pixels(model, x, norm_maxpool, maxpool_output, y)
    # Calculate receptive field and extract motifs
    receptive_field_inhibitors, receptive_field_inhibitors0 = receptive_field_calculator(inhibitors_common_motifs, y)
    receptive_field_weak_inhibitors, receptive_field_weak_inhibitors0 = receptive_field_calculator(
        weak_inhibitors_common_motifs, y)
    receptive_field_no_inhibitors, receptive_field_no_inhibitors0 = receptive_field_calculator(
        no_inhibitors_common_motifs, y)
    # Get the molecule sub-structure by identify the atoms that compose it.
    receptive_field_inhibitors = get_fragment(receptive_field_inhibitors, smiles)
    receptive_field_weak_inhibitors = get_fragment(receptive_field_weak_inhibitors, smiles)
    receptive_field_no_inhibitors = get_fragment(receptive_field_no_inhibitors, smiles)
    draw_mol(receptive_field_inhibitors)


    # Unsupervised - load new molecules and convert them to mol objects play with them
    dict1 = mol2mol_supplier(r'C:\path\newmols.mol2', sanitize=True)
    dict2 = mol2mol_supplier(r'C:\path\newmols2.mol2', sanitize=True)
    dict1.update(dict2)
    newmols_names, newmols_data= build_new_20_mols_dataset(dict1)

    # Predict score for new molecules and save to excel file
    newmols_predictions = model.predict(newmols_data)
    predictions = np.column_stack([newmols_names, newmols_predictions])
    predictions = pd.DataFrame(predictions)
    predictions.to_excel("model7_1_CV5_240x42.xlsx")


    # Play with the new 20 molecules
    molsss20_norm_maxpool, molsss20_layer_output = normalize_maxpooling_layer(model, newmols_data)
    molsss20 = motif_freq(molsss20_norm_maxpool)