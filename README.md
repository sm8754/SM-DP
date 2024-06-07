# SM-DP

## Introduction

This is a driver library for intelligent analysis, provided by the paper titled "Smart microscopy of the future: task-driven framework using large models for digital pathology".


## Prerequisites

The code is built with the following libraries:

- Python 3.9

- [Anaconda3](https://www.anaconda.com/)

- [Pytorch 1.13](https://pytorch.org/)

- Numpy 1.23

Or run the 'requirement.txt' file to install the dependent packages.


## Run

- Processing
  ```bash
  If needed, run the `split.py` file to divide the data randomly. The proportions can be adjusted as required.
  
  By running `pre_processing.py` to process useful tissues, you need to configure files before proceeding.
  ```
  
## Fine-tuned the model

- model
  ```bash
  `LM_train.py`
  `LM_test.py`
  ```

## Intelligent driving
- micro-dri
  ```bash
  Please set relevant instructions in the code based on the local device. Run `main.py` to perform the main analysis.
  ```
