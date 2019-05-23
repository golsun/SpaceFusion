# SpaceFusion
[SpaceFusion](https://arxiv.org/abs/1902.11205) is a learning paradigm proposed to align and structure the unstructured latent spaces learned by different models trained over different datasets. Of particular interest is its application to neural conversation modelling, where space fusion can be used to jointly optimize the relevance and diversity of generated responses. 

### News
* A Keras implementation is provided [here](https://github.com/golsun/SpaceFusion/blob/master/src)
* We published a [blog](https://www.microsoft.com/en-us/research/blog/spacefusion-structuring-the-unstructured-latent-space-for-conversational-ai/) to discuss the intuition and implication

## Requirement
the code is tested using Python 3.6 and Keras 2.2.4

## Dataset
* Reddit data is downloaded from a [third party](http://files.pushshift.io/reddit/comments/). However, as we don't own the data, we cannot release it here. But you use [this script](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction) to process the raw data.
* Switchboard data is downloaded from [this repo](https://github.com/snakeztc/NeuralDialog-CVAE)
* We provided a [toy dataset](https://github.com/golsun/SpaceFusion/blob/master/data) in this repo for debugging.

Tokenization functions and other preprocessing tools can be found [here](https://github.com/golsun/NLP-tools). 

## Usage
* To train a SpaceFusion model: `python src/main.py mtask train --data_name=toy`
* To visualize the learned latent space: `python src/vis.py --data_name=toy`
* To interact with the trained model: `python src/main.py interact_rand --data_name=toy`

## Discription
* `main.py` is the main file
* `model.py` defines the SpaceFusion model (see `class MTask`) and some baselines
* `vis.py` defines the function we used to visulize and analysis the latent space
* `dataset.py` defines the data feeder
* `shared.py` defines the default hyperparameters


![](https://github.com/golsun/SpaceFusion/blob/master/fig/intro_fig.PNG)

