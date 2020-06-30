# -*- coding: utf-8 -*-
"""
Hadar Grimberg
5/4/2020

"""

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
import numpy as np
from collections import Counter
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sn


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

def new20_mols():
    newmols1 = mol2mol_supplier(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\newmols.mol2')
    newmols2 = mol2mol_supplier(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\twisted.mol2')
    mmm = Chem.MolFromMol2File(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\zinc_22812775.mol2')
    mm = Chem.MolFromMol2File(r'C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\VS-28.mol2')
    newmols =  {**newmols1, **newmols2}
    newmols[mmm.GetProp('_Name')] = mmm
    newmols[mm.GetProp('_Name')] = mm
    test_mols = create_dataset(newmols, dset='test')
    test_mols = test_mols.drop(columns=['SMILES', 'num_of_H', 'NumAtoms', 'NumHeavyAtoms','num_bonds'])

    for i in test_mols.columns:
        if "num_of" in i:
            test_mols[i].fillna(0, inplace=True)
        try:
            test_mols[i + "_sqr"] = test_mols[i] ** 2
            pass
        except TypeError:
            print(i)
        test_mols[i + "_sqr"] = test_mols[i] ** 2
    return test_mols

def show_mol_images(database):
    no_none = [mol for mol in database if mol]  # None element can´t be drawn, this loop keep only valid entries
    [Chem.SanitizeMol(mol) for mol in no_none]
    images = Draw.MolsToGridImage(no_none, molsPerRow=7, subImgSize=(150, 150),
                                 legends=[mol.GetProp('_Name') for mol in no_none], maxMols=100)
    images.show()
    Draw.MolToImage(database[0]).show()
    return images

def atom_counter(name,mol,table):
    atomsList = [i.GetSymbol() for i in mol.GetAtoms()]
    atomsList = Counter(atomsList)
    for i in atomsList.keys():
        table.loc[name, f'num_of_{i}'] = atomsList[i]

def create_dataset(database,dset='train'):
    table=pd.DataFrame()
    index=0
    for mol in database.values():
        if mol:
            name=mol.GetProp('_Name')
            mol_h = Chem.AddHs(mol)
            AllChem.ComputeGasteigerCharges(mol_h)
            # table.loc[index,'Name']=mol.GetProp('_Name')
            table.loc[name,'NumAtoms']=mol_h.GetNumAtoms()
            table.loc[name,'NumHeavyAtoms']=mol_h.GetNumHeavyAtoms()
            table.loc[name,'SMILES']=Chem.MolToSmiles(mol_h)
            table.loc[name,'tpsa'] = Descriptors.TPSA(mol_h)
            table.loc[name,'mol_w'] = Descriptors.ExactMolWt(mol_h)  #The exact molecular weight of the molecule
            # table.loc[name,'num_radical_electrons'] = Descriptors.NumRadicalElectrons(mol_h) #The number of radical electrons the molecule has
            # table.loc[name,'num_valence_electrons'] = Descriptors.NumValenceElectrons(mol_h) #The number of valence electrons the molecule has
            table.loc[name,'num_heteroatoms'] = Descriptors.NumHeteroatoms(mol_h)
            table.loc[name,'num_rings'] = mol_h.GetRingInfo().NumRings()
            # table.loc[name,'num_conformers'] = mol_h.GetNumConformers()
            table.loc[name,'num_bonds'] = mol_h.GetNumBonds()
            table.loc[name,'gasteiger_charge'] = float(mol_h.GetAtomWithIdx(0).GetProp('_GasteigerCharge'))
            # number_of_atoms(['C', 'O', 'N', 'Cl'], table,mol_h)
            atom_counter(name,mol_h, table)
    table['HeavyAtoms_perc'] = table['NumHeavyAtoms'] / table['NumAtoms']
    # table['heteroatoms_perc'] = table['num_heteroatoms'] / table['NumAtoms']

    if dset=='train':
        mol_scores= open(r"C:\Users\Hadar Grimberg\PycharmProjects\ML\project\Team2\Data\summary_2.0.sort", 'r')
        for line in mol_scores:
            mol_name = line.split('_')[0].strip()
            val = float(line.split(',')[4].strip())
            if mol_name in table.index:
                table.loc[mol_name, 'Score'] =val
        return table
    else:
        return table

def arrange_data(dataframe):
    #Leave only features columns
    y = dataframe.loc[:,'Score']
    dataframe = dataframe.drop(columns=['SMILES', 'Score', 'num_of_I', 'num_of_H', 'NumAtoms', 'NumHeavyAtoms', 'num_of_Br','num_bonds'])

    for i in dataframe.columns:
       if "num_of" in i:
          dataframe[i].fillna(0, inplace=True)
       try:
            dataframe[i + "_sqr"] = dataframe[i]**2
            pass
       except TypeError:
           print (i)
       dataframe[i + "_sqr"] = dataframe[i]**2


    #Perform a train-test split. We'll use 20% of the data to evaluate the model while training on 80%
    X_train, X_test, y_train, y_test = train_test_split(dataframe, y, test_size=.2, random_state=5)
    return(X_train, X_test, y_train, y_test)


def evaluation(model, X_test, y_test):
    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    plt.figure(figsize=(15, 10))
    plt.plot(prediction[:100], "red", label="prediction", linewidth=1.0)
    plt.plot(y_test[:100], 'green', label="actual", linewidth=1.0)
    plt.legend()
    plt.ylabel('Score')
    plt.title("MAE {}, MSE {}".format(round(mae, 4), round(mse, 4)))
    plt.show()

    print('MAE score:', round(mae, 4))
    print('MSE score:', round(mse, 4))
    print('R2 score:', round(r2, 4))


if __name__ == '__main__':
    table = create_dataset(database,dset='train')
    X_train, X_test, y_train, y_test= arrange_data(table)
    ridge = Ridge(normalize=True)
    parameters = {'alpha': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 20], 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
    ridge_reg = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
    ridge_reg.fit(X_train, y_train)
    print("Best alpha is: ",ridge_reg.best_params_)
    print("Best training MSE is: ", np.abs(ridge_reg.best_score_))
    # Evaluate results
    evaluation(ridge_reg, X_test, y_test)
    #Check which parameters are the most influence on prediction
    ridge_df = pd.DataFrame({'variable': X_train.columns, 'estimate': ridge_reg.best_estimator_.coef_, 'beta': np.abs(ridge_reg.best_estimator_.coef_)})
    ridge_df2=ridge_df.sort_values(by=['beta'],ascending=False)
    ridge_df2.index=ridge_df2[['variable']]
    print("The parameters that have the most influence on prediction")
    print(ridge_df2['estimate'][ridge_df2['beta'] >= 1])
    ridge_df2.to_excel("RidgeRegression_estimate24.xlsx")

    new_table = table.drop(columns=['SMILES', 'num_of_I', 'num_of_H', 'NumAtoms', 'NumHeavyAtoms', 'num_heteroatoms', 'num_of_Br','num_bonds'])
    plt.figure()
    sn.heatmap(new_table.corr(), annot=True)
    plt.show()

    test_mols=new20_mols()
    df = pd.concat([X_train, test_mols])
    test_mols = df.iloc[-20:, :]
    for i in test_mols.columns:
       if "num_of" in i:
          test_mols[i].fillna(0, inplace=True)
    preds20 = pd.DataFrame(ridge_reg.predict(test_mols))
    preds20.index = test_mols.index
    preds20.to_excel("ridge_regression.xlsx")
