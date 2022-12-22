# Image interpolation

## Context
This project is part of my engeneering school formation at INSA Rouen Normandie.
This project is optioal but mandatory.
## Aim
The goal of this project is to double the number of fps (frames per second). It could be applied to have a video smoother or a slow motion slower.
This repository is based on a keras implementation.

## Use
### Dataset
The dataset is downloaded from Youtube videos. Firstly, you need to use the "youtube_scrapper.ipynb" notebook and execute each cell.
You will have the directories corresponding to each video saved at your root folder. 
You need to create a directory with that sructure :
* youtube/
    * train/</br>
    Youtube video directories
    * valid/</br>
    Youtube video directories
### Training
To train and evaluate the models, you need to execute all the cells of the "train.ipynb" notebook.