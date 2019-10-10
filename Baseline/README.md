# BiDaF Baseline
To help the community start iterating on MSMARCO we have written a implementation of Bidaf in Pytorch. We will be refining and optimizing the model to make setup/deploy time minimal. We provide no guaretees for the code.
## Requirements
Python 3.5
CUDA 9.0 
Pytorch
h5py
nltk
## Setup
1. Install all dependencies and setup CUDA.
2. Ensure the mrcqa folder is in your python Path
~~~
export PYTHONPATH=${PYTHONPATH}:~/<Where you saved this git repo>/MSMARCO-Question-Answering/Baseline/mrcqa
~~~
2. Create an experiment folder and copy the config.yaml file from scripts. Currently, this will only train for 1 epoch and stop. This is useful for testing and debugging scripts. 
4. Get MSMARCO v2.1 data and pre-trained word embedings and save in your data folder. We recommend Glove's glove.840B.300d.txt
5. Try to train a model using the following script
~~~
python3 scripts/train.py <your experiment folder> <your datafolder>/train_v2.1.json --word_rep <your datafolder>/<chosen word embedding> --force_restart --cuda=True
~~~
6. If your model successfully finnished training then you are ready to start training a full model and experimenting



### Training
1. Modify the config.yaml file to match your desired paramaters such as training epochs, dropout rate, learning rate,etc
2. Run the following command. If you do not have CUDA set up the --cuda=True will be ignored. --force_restart is not strictly required but it is used to ignore any existing checkpoints in your exp folder. 
~~~
python3 scripts/train.py <your experiment folder> <your datafolder>/train_v2.1.json --word_rep <your datafolder>/<chosen word embedding> --force_restart --cuda=True
~~~
### Prediction
1. To generate a prediction file run the command below. This will load the model and data and predict for all answered questions where the answer is a span. Any new tokens/char will get a random embedding. In its current form the model cannot provide predictions for the held out eval set. In the future this dependency will be removed.
~~~
python3 scripts/predict.py $EXP/ $DATA/dev_v2.1_candidate.json prediction.json --cuda=True
~~~
To generate new embeddings from your embedding file instead instead of using random embeddings use the following command.
~~~
python3 scripts/predict.py $EXP/ $DATA/dev_v2.1_candidate.json prediction.json --word_rep <your datafolder>/<chosen word embedding> --cuda=True
~~~


