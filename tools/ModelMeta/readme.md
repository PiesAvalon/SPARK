# ModelMeta     ISSTA2025

This is the open resposity of the paper "Improving Deep Learning Framework Testing with Model-Level Metamorphic Testing". Here is the structure of the result.



### Description

In this work, we propose ModelMeta, a model-level metamorphic testing method for DL frameworks with four MRs focused on model structure and calculation logic. ModelMeta inserts external structures to generate new models with consistent outputs, increasing interface diversity and detecting bugs without additional MRs. Besides, ModelMeta uses the QR-DQN strategy to guide model generation and then detects bugs from more fine-grained perspectives of training loss, memory usage, and execution time.


If you have any questions, please leave a message here to contact us. 


# Run

## Installation

Ensure you are using Python 3.9 and a Linux-64 platform:

```bash
$ conda create -n ModelMeta python=3.9
$ conda activate ModelMeta
$ pip install -r requirements.txt
```

## Dataset

We provide a few simple datasets`./dataset/`. Due to the large size of other datasets, we will provide a download link upon request. Please contact us to obtain the dataset.

## Usage

### Step 1: Run the master file

```bash
cd ./mindspore_mutation
python main_ms.py
```
```bash
cd ./torch_mutation
python main_torch.py
```
```bash
cd ./onnx_mutation
python main_onnx.py
```
### Step 2: Check Output

Results will be available in the `./mindspore_mutation/results/` ,`./torch_mutation/results/` or `./onnx_mutation/results/` directory.. This folder will contain two files:
- A `.json` file: Contains the log details.
- A `.xlsx` file: Records the results of the process, including coverage, distance, and other relevant metrics.


## Parameter Settings

For specific configuration changes, you can make them in `./config/rq3_exp1.yaml`.Then run `main_torch.py`, `main_ms.py` or `main_onnx.py`

Below are the adjustable parameters:
```python
execution_config:
  seed_model: "resnet" # Options: `resnet`,`UNet`,`vgg16`,`TextCNN`,`ssimae`...
  mutate_times: 100 # Number of epochs for training.
  ifeplison: 0.6 # Number of eplison threshold.
  ifapimut: False # Flag to determine whether to perform API mutation
  ifTompson: False # Flag to determine whether to perform Tompson mutation
  num_samples: 1 # Size of the batches for training.
  run_option: 0  # Mutation strategy. Options: 0:'random', 1:'MCMC', 2:'qrdqn'.
  MR: 0,1,2,3   # Mutation strategy. Options: 0:'SMR1', 1:'SMR2', 2:'SMR3', 3:'SMR4'.
  num_quantiles: 20 # QRDQN parameter
  device: -1 # Choose CPU or GPU

train_config: 
  loss_name: "CrossEntropy" # Options: yolov4loss CrossEntropy unetloss textcnnloss ssdmultix yololoss...
  opt_name: "SGD" # Options: adam SGD ...
  seed_model: "TextCNN" # Options: `resnet`,`UNet`,`vgg16`,`TextCNN`,`ssimae`...
```
