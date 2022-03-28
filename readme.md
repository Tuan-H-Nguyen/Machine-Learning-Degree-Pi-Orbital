# ML-based DPO model

## Python and dependencies:

python 3   
argparse     
numpy   
matplotlib  
pandas  
sympy  
sklearn  

## Data.

Data for PAH and thienoacenes that are used in the project can be found in sample_DATA directory. They are datasets from ref 1 and 2 in csv format.

## Guide
### 1. Overall guide
Refer to **guide.ipynb**.

### 2. Running main.py
To make prediction of electronic properties of a certain PAH/thienoacenes from its SMILES string, run:  
  
    python main.py "C1(C=CS2)=C2C=C(C=CC3=C4C=CC5=C3C=CC6=C5C=CC7=C6C=CC8=C7C=CC=C8)C4=C1" -chk checkpoint.txt
  
where 
+ positional argument: SMILES string of PAH or thienoacene
+ -chk : path of file where parameters of the model can be retrieved.      

To train/test model and prepare the checkpoint file, run:  

    python main.py 0 -t 1 -chk checkpoint.txt -d "sample_DATA\total_dataset.csv" -n 10 -s 1052  

where:
+ 0 or anything inplace of SMILES string since this argument is not used during training but cannot be omitted.
+ -t : enable training mode.
+ -chk : path of txt file for saving model's parameters.
+ -d : path to data pool where data are drawed to assemble training and test sets.
+ -n : number of runs for training/testing.
+ -s : random seed.

## Highlighted components/scripts:
### 1. **DPO_generate** from poly_rings.DPO:  
Function that input SMILES of PAH and thienoacenes (string) and return DPO polynomials (string).

### 2. **GD_DPO** from gd_dpo.GDDPO.py:  
Python class for ML-based DPO model.

### 3. **sampling** from utils.stratified_splitting.py:  
Python function for splitting data to train set and test set.

### 4. **Trainer** from trainer.py:  
Class for quickly training and testing DPO model multiple times in a few lines of codes. Only required a generic data set (from which random training set and test set are assembled).

### 5. Paper results:  
paper_result_full_model, paper_result_trun_model, paper_result_robust for carrying out experiments reported in upcoming papers.

### 6. **main.py**:  
Script for running in command line. For either training and saving the model to a txt file; or make prediction from SMILES string using a pre-trained model.

## References
1. https://pubs.acs.org/doi/10.1021/acsomega.8b00870
2. https://pubs.acs.org/doi/full/10.1021/acsomega.9b00513
