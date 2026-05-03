# Gandalf
Some codes and results about our paper, here is the structure of our directory:

- **/src:** High level interface of out implementation of model generator across frameworks by JSON

  - configuration: Some configurations and basic settings
  - dataset: Dataset across frameworks API
  - grammar: Grammar checker for model generation
  - interpreter: Model Interpreter for model generating across frameworks by JSON
  - metrics: Metrics across frameworks API
  - optimizers: Optimizers across frameworks API
  - test_process: Test process across frameworks API
  - train_process: Train process across frameworks API
  - model_json.py: Entrance implementation for model generation
  - *.json: Examples for JSON files

- **/torchsummary:** Open source package for the function of summary() for PyTorch

- **/trials:** Codes about Gandalf experiments

  - classical: An implementation of Audee with models of VGG16, Lenet-5 and Resnet-20

  - envs: The test environment for our testing process, currently, only the environment of CNN2d is supported

  - dqn: The main implementation of out method and random method

  - results: Some results of our experiments

    

Through our method, we have found some serious faults for Tensorflow and PyTorch, and we have submitted them to the issues as follow, most of them are accepted:

* https://github.com/keras-team/keras/issues/15666
* https://github.com/keras-team/keras/issues/15667
* https://github.com/keras-team/keras/issues/15677
* https://github.com/keras-team/keras/issues/15716
* https://github.com/keras-team/keras/issues/15717
* https://github.com/tensorflow/tensorflow/issues/53055
* https://github.com/tensorflow/tensorflow/issues/53107
* https://github.com/pytorch/pytorch/issues/68321
