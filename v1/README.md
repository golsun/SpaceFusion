This is the repo for the code used in the paper [Jointly Optimizing Diversity and Relevance in Neural Response Generation](https://arxiv.org/abs/1902.11205). 

## Requirement
the code is tested using Python 3.6 and Keras 2.2.4

## Discription
* `main.py` is the main file, you may start training a SpaceFusion model by `python main.py mtask train` with default parameters
* `model.py` defines the SpaceFusion model (see `class MTask`) and some baselines
* `vis.py` defines the function we used to visulize and analysis the latent space
* `dataset.py` defines the data feeder, which will be called by classes in `model.py`
* `shared.py` defines the default hyperparameters

## Dataset
we used Reddit data. Tokenization functions can be found [here](https://github.com/golsun/NLP-tools)
