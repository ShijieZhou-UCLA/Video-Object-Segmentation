# Video Object Segmentation
COMS W4995: Applied Deep Learning

##Demos:

####1. Reproduce Part (OSVOS):

`parent_network_demo.ipynb`: train parent network

`test_network_demo.ipynb`: train test network and test the results

####2. Innovation Part (OSVOS + RCF):

`RCF_parent_network_demo.ipynb`: train parent network

`RCF_test_network_demo.ipynb`: train test network and test the results

##Codes

###Data loading:

`data_parent.py`: load data for parent network

`data_finetune.py`: load data for test network

###Neural Network
`cnn.py`: functions and architecture of OSVOS network

`cnn_rcf.py`: functions and architecture of OSVOS + RCF network

###Train:
`train_parent.py`: train function for parent network

`train_finetune.py`: train function for test network

`train_parent.txt`: list of training data

###Test:
`test_finetune.py`: test function for test network

##Others
`DAVIS`: dataset and result images

`models`: training models of reproduce part*

`models_RCF`: training models of innovation part*

`tensorboard_graph`: tensorboard graph screenshot

#####*: Due to the 100mb file restriction of Github, the files in these two folders cannot be uploaded. Please download [here](https://drive.google.com/open?id=1-llSA2tXeoJtWdlEqYZtMrQKqyCkKMg8) and replace them.


  



