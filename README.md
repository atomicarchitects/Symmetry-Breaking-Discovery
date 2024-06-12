## Paper: 
[Discovering Symmetry Breaking in Physical Systems with Relaxed Group Convolution](https://arxiv.org/abs/2310.02299)

## Abstract:
Modeling symmetry breaking is essential for understanding the fundamental changes in the behaviors and properties of physical systems, from microscopic particle interactions to macroscopic phenomena like fluid dynamics and cosmic structures. Thus, identifying sources of asymmetry is an important tool for understanding physical systems. In this paper, we focus on learning asymmetries of data using relaxed group convolutions.  We provide both theoretical and empirical evidence that this flexible convolution technique allows the model to maintain the highest level of equivariance that is consistent with data and discover the subtle symmetry-breaking factors in various physical systems. We employ various relaxed group convolution architectures to uncover various symmetry-breaking factors that are interpretable and physically meaningful in different physical systems, including the phase transition of crystal structure, the isotropy and homogeneity breaking in turbulent flow, and the time-reversal symmetry breaking in pendulum systems.

## Dataset and Preprocessing
- Download the fluid simulation from [here](https://drive.google.com/file/d/1QVINJZ44Lm7EhK5iaSiisb-vT0Xs9LYF/view?usp=sharing) and Move it to 'data/fluid'.
- The code to generate the pendulum data is in TimeReversalPendulum.ipynb



## Description of Files
1. relaxed_gconv: implementation of relaxed rotation/translation/timereversal/octahedral group convolution layers.

2. utils: helper functions for training and computing group transformations. 

3. SquareRectangle.ipynb: a simple example to show how relaxed group convolution learns symmetry breaking factors. 
     
3. BaTiO3.ipynb: finding the symmetry breaking factors in the phase transtions of BaTiO3.

4. HomogeneityFluid.ipynb: finding the translation/homogeneity symmetry breaking factors in 2d channel flow. 

5. IsotropyFluid.ipynb: finding the scale where the eddies start to have rotation symmetry or isotropy. 

6. TimeReversalPendulum.ipynb: finding the time reversal symmetry breaking in the pendulum with frictions. 

7. superresolution: code for the 3d fluid superresolution experiment. 


## Requirements
- To install requirements
```
pip install -r requirements.txt
```

## Cite
```
@inproceedings{
wang2024discovering,
title={Discovering Symmetry Breaking in Physical Systems with Relaxed Group Convolution},
author={Rui Wang and Elyssa Hofgard and Han Gao and Robin Walters and Tess Smidt},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=59oXyDTLJv}
}
```
