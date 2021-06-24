# Body Part Regression 

The Body Part Regression (BPR) model translates the anatomy in a radiologic volume into a machine-readable form. 
Each axial slice maps to a slice score. The slice scores monotonously increase with patient height. 
With the help of a slice-score look-up table, the mapping between certain landmarks to slice scores can be checked. 
The BPR model learns in a completly self-supervised fashion. There is no need for annotated data for training the model instead of evaluation purposes. 

The BPR model can be used for sorting and labeling radiologic images by body parts. Moreover, it is useful for cropping specific body parts as a pre-processing or post-processing step of medical algorithms. If a body part is invalid for a certain medical algorithm, it can be cropped out before applying the algorithm to the volume. 

The Body Part Regression model in this repository is based on the SSBR model from [Yan et al.](https://arxiv.org/pdf/1707.03891.pdf) 
with a few modifications explained in the master thesis "Body Part Regression for CT images". 

For CT volumes you are able to load a body part regression model and apply it to your use case. If you want to train a BPR model for a different modality, the training procedure will be explained as well. 


## Applying the BPR model to CT volumes 
1. Load model parameters **TODO** 
2. nifti file -> json output 
3. Explain json output 
4. documents/notebooks/bpr-prediction.ipynb **TODO** 
5. documents/notebooks/estimate-body-part-examined.ipynb
6. documents/notebooks/valid-scope-sensitive-segmetnation.ipynb
7. documents/notebooks/data-sanity-checks.ipynb

Scope of the BPR modle: start of the pelvis to end of the head -> valid region in CT volumes
not in valid scope: kids and pregnant women, because of lag of data. 

### Estimate the Body Part Examined


### Valid scope sensitive prediction of clinical models 


### Data Sanity Checks 


## Training a BPR model 
For Training a BPR model it is important to have lots of different studies and volumes (preferably more than 1000 volumes). <br>
For demonstration a BPR model is trained with a small dataset and a small image size of 64 px x 64 px in the following notebook:  <br>
*documentation/notebooks/train-bpr-model.ipynb* <br>
Please read the paper from [Yan et al.](https://arxiv.org/pdf/1707.03891.pdf) and the Methods & Experiments section from the master thesis to 
understand the theoretical foundations behind the BPR model. The main pittfalls of training a body part regression model are: 
- using not enough data augmentation techniques, so that the model can overfitt on certain studies or patients
- using to less data 
- haveing to much data with a wrong axis ordering in the dataset 

A very valuable source for getting data for a body part regression model is [The Cancer Image Archive](https://www.cancerimagingarchive.net/). 

python package pip install ... **TODO** 
tests **TODO** 
master thesis link to arxiv **TODO** 