# Body Part Regression 

The Body Part Regression (BPR) model translates the anatomy in a radiologic volume into a machine-readable form. 
Each axial slice maps to a slice score. The slice scores monotonously increase with patient height. 
With the help of a slice-score look-up table, the mapping between certain landmarks to slice scores can be checked. 
The BPR model learns in a completly self-supervised fashion. There is no need for annotated data for training the model instead of evaluation purposes. 

The BPR model can be used for sorting and labeling radiologic images by body parts. Moreover, it is useful for cropping specific body parts as a pre-processing or post-processing step of medical algorithms. If a body part is invalid for a certain medical algorithm, it can be cropped out before applying the algorithm to the volume. 

The Body Part Regression model in this repository is based on the SSBR model from [Yan et al.](https://arxiv.org/pdf/1707.03891.pdf) 
with a few modifications explained in the master thesis "Body Part Regression for CT images". 

For CT volumes you are able to load a body part regression model and apply it to your use case. If you want to train a BPR model for a different modality, the training procedure will be explained as well. 

--------------------------------------------------------------
## Install package 
1. Create a new virtual environment with pip
2. Clone the bodypartregression repsoitory from phabricator 
3. Go into the bodypartregression/ folder and run: 
   
`pip install -e .`

--------------------------------------------------------------
## Analyze examined body parts
Provide the CT volumes in the nifti-format in one directory. 

Scope of the BPR modle: start of the pelvis to end of the head -> valid region in CT volumes
not in valid scope: kids and pregnant women, because of lag of data. #TODO 


Run the following command in the terminal: <br>
` bpreg_predict -i <input_path> -o <output_path>` <br>
For each nifti-file in the directory `input_path` a corresponding json-file 
gets created and saved in the `output_path`. <br>

Additional tags for the `bpreg_predict` command: <br>
--skip (bool): skip already created .json meta data files (default: 1) <br>
--model (str): specify model (default: public model from zenodo)

The json-file contains all the meta data regarding the examined body part of the nifti-file. 
It includes the following tags: 
- valid-zspacing
- cleaned slice-scores
- lookuptable 
- ... 

License and References of used public model: 

