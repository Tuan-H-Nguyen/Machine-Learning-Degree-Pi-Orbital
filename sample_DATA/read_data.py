#%%
import glob
import pandas as pd

PATH = "*.csv"

def read_sample_data(path=PATH):
    files = glob.glob(path)
    for file in files:
        if file == "total_dataset.csv":
            continue
        elif "smiles" in file:
            try: smiles = pd.concat(
                [smiles,pd.read_csv(file,index_col = "id")])
            except NameError: 
                smiles = pd.read_csv(file,index_col = "id")
        else:
            try: data = pd.concat(
                [data,pd.read_csv(file,index_col = "No")])
            except NameError: 
                data = pd.read_csv(file,index_col = "No")
    smiles = smiles.rename({"id":"No"})
    data = pd.concat([data,smiles],axis=1)
    return data

def data_assembler(folder_path = ""):
    original_train_data = pd.concat([
        pd.read_csv('pah_train_set_v2.csv',index_col=['No']),
        pd.read_csv('thien_train_set.csv',index_col=['No'])
    ])

    original_test_data = pd.concat([
        pd.read_csv('pah_test_set.csv',index_col=['No']),
        pd.read_csv('thien_test_set.csv',index_col=['No'])
    ])

    data = pd.concat([original_train_data,original_test_data])

    smiles = pd.concat([
        pd.read_csv("PAH_smiles.csv",index_col=['id']),
        pd.read_csv("thienoacene_smiles.csv",index_col=['id'])
    ])

    smiles = smiles.rename(columns = {'id':'No'})
    
    data = pd.concat([data,smiles],axis=1)
    return data


data = data_assembler()
data.to_csv("total_dataset.csv",index_label = "No")

# %%
