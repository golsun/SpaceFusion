# SpaceFusion
SpaceFusion is a regularized multi-task learning paradigm proposed to align and structure the unstructured latent spaces learned by different models trained over different datasets. Of particular interest is its application to neural conversation modelling, where SpaceFusion is used to [jointly optimize the relevance and diversity of generated responses](https://arxiv.org/abs/1902.11205). 

### News
* provided [scripts](https://github.com/golsun/SpaceFusion/tree/master/data) to generate **Switchboard** and **Reddit** dataset.
* The [slides](https://github.com/golsun/SpaceFusion/blob/master/slides.pdf) presented at NAACL-HLT 2019 is available.
* We published a [blog](https://www.microsoft.com/en-us/research/blog/spacefusion-structuring-the-unstructured-latent-space-for-conversational-ai/) to discuss the intuition and implication
* A Keras implementation is provided

## Requirement
the code is tested using Python 3.6 and Keras 2.2.4

## Dataset
We provided [scripts](https://github.com/golsun/SpaceFusion/tree/master/data) to generate Switchboard and Reddit dataset as well as a [toy dataset](https://github.com/golsun/SpaceFusion/blob/master/data/toy) in this repo for debugging.

## Usage
* To train a SpaceFusion model: `python src/main.py mtask train --data_name=toy`
* To visualize the learned latent space: `python src/vis.py --data_name=toy`
* To interact with the trained model: `python src/main.py mtask interact_rand --data_name=toy`

## Discription
* `main.py` is the main file
* `model.py` defines the SpaceFusion model (see `class MTask`) and some baselines
* `vis.py` defines the function we used to visulize and analysis the latent space
* `dataset.py` defines the data feeder
* `shared.py` defines the default hyperparameters

## Citation
Please cite our NAACL paper if this repo inspired your work :)
```
@article{gao2019spacefusion,
  title={Jointly Optimizing Diversity and Relevance in Neural Response Generation},
  author={Gao, Xiang and Lee, Sungjin and Zhang, Yizhe and Brockett, Chris and Galley, Michel and Gao, Jianfeng and Dolan, Bill},
  journal={NAACL-HLT 2019},
  year={2019}
}
```

![](https://github.com/golsun/SpaceFusion/blob/master/fig/intro_fig.PNG)


