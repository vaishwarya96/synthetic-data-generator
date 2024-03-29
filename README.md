Code for the paper "Usefulness of synthetic datasets for diatom automatic detection using a deep-learning approach" publsihed in Engineering Applications of Artificial Intelligence.

# Synthetic dataset generator

Contains the code to generate synthetic slide images. 

dataset_generator_multithread.py - Run this code for generating the synthetic dataset. It uses multithread approach to generate the images. 

## how to generate a Synthetic dataset

create a conda env with 
`conda env create --name synt_environment --file=environments.yml`

activate the conda env : 
`conda activate synt_environment`

Check the parameters in var.yaml

 generate the Synthetic dataset : 
 `python dataset_generator_multithread.py --var var.yaml`


## implementation details

Default variables are defined in : *diatom_codes/**utils/global_variables.py*** You can configure the variables in yaml files by adding these options : **--var ../var.yaml**. **var.yaml** is read by the python file *diatom_codes/synthetic_data_generator/**variables.py***.
The **var.yaml** configure the **INPUT** variables : 
- TAXON_FILTER_PATH: */dataset/model_id_map.csv*: TAXON_FILTER_PATH is a csv file associating taxon and id : taxon,id
- DATASET_PATH: */dataset/*:  DATASET_PATH **is a list of** folders where directory contain images of diatom. The name of each directory define the taxon of diatoms image inside 
- DATASET_DUST_PATH: *"synthetic_data/debris"*  : folder where directory contain images of debris
- IMAGE_REF: *synthetic_data/atlas/ref_img.png* # image of reference for synthetic image creation

Examples of files for TAXON_FILTER_PATH and IMAGE_REF are given in ```example_data/ ```

The **var.yaml** configure the **OUTPUT** variables : 
SAVE_PATH: *"ouput"* : output dir for results

# The **var.yaml** configure the **PARAM** variables : 
TOTAL_N_IMAGES: *20*  : output dir for results
PERCENTAGE_VAL: *0.1* : split for validation 

It uses generator.py for the entire generation process.\
generator.py - The entire generation pipeline is implemented here.  \
utils.py - Contains useful functions for generation \
global_variables.py and variables.py - These are config files to set the parameters for generation. 

The output is a train and val folder. In these folders the following are saved: 
1. images/ - The synthetic images 
2. annotations/ - The labels for horizontal bonding box detection in xml format 
3. annotations1/ - Labels for rotated bounding box detection 
4. annotations2/ - Labels for horizontal bounding box in YOLO format


## Citation
If you use this code please cite the following paper:

Aishwarya Venkataramanan, Pierre Faure-Giovagnoli, Cyril Regan, David Heudre, C´ecile Figus, Philippe Usseglio-Polatera, Cedric Pradalier, and Martin Laviale. Usefulness of synthetic datasets for diatom automatic detection using a deep-learning approach. Engineering Applications of Artificial Intelligence, 117:105594, 2023

```
@article{venkataramanan2023usefulness,
  title={Usefulness of synthetic datasets for diatom automatic detection using a deep-learning approach},
  author={Venkataramanan, Aishwarya and Faure-Giovagnoli, Pierre and Regan, Cyril and Heudre, David and Figus, C{\'e}cile and Usseglio-Polatera, Philippe and Pradalier, Cedric and Laviale, Martin},
  journal={Engineering Applications of Artificial Intelligence},
  volume={117},
  pages={105594},
  year={2023},
  publisher={Elsevier}
}
```






