# XAI612-project
This is a project from XAI612 course, Departments of artificial intelligence, Korea University

## Model
We used ResNet50, ResNet101. Tensorflow code was followed from https://github.com/Nyandwi/ModernConvNets.

## Improvement point
* Augmentation: HorizontalFlip, RandomRotation(-15,15), RandomZoom(0.0, 0.3)
* Custom earlystop: Since initial training is unstable, set up wait point for checkpoint and earlystop until 100 epochs
* ExponentialDecay on LearningSchedule


Colab link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DeTFm5gnW3lucOyaMJY945yzm_PorUQW?usp=sharing)
