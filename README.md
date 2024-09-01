<h1 align="center">
<img src="https://raw.githubusercontent.com/thienannguyen-cv/SMap/main/logo.png" width="300">
</h1><br>

-----------------

# SMap

| | |
| --- | --- |
| Testing | [![CI - Test](https://img.shields.io/badge/Unit_Tests-failed-red)]()|

## What is it?
An open source pytorch library for spatial mapping based on 2D representations, a concept in 3D inference that first appeared in the paper ["A Solution to the Fundamental Problem of 3D Inference Based on 2D Representations"](https://arxiv.org/abs/2211.04691) uploaded to arXiv. This project is the foundation of the "Dynamic 3D Inference" vision with three main pillars: 
 - Dynamic gradient flows. 
 - Platonic-representation based 3D inference. 
 - Countable vision. 

**Dynamic gradient flows**: A concept appears in backprobagation programming for neural network optimization when considering the differentiable rendering problem as a multi-objective optimization problem where each point corresponds to an objective. Thus, for each objective, the gradient flow will be controlled in a way that helps to translate an existing point to its corresponding screen position instead of “bubbling" a point at a certain screen position, what is a common process found in solutions based on image loss optimization. 
For more details, please read the paper: https://arxiv.org/abs/2211.04691. 

**Platonic-representation based 3D inference**: The projection of an object on an image is an 2D instance of the Platonic representation of that object. This project aims to perform 3D inference on these Platonic representations built during the training process. Unlike other approaches, the inverse rendering solution implemented in this project does not separate the 3D inference process from the training process but rather the training process is built on top of the inference. Accordingly, even if the target images are changed by a change in the 3D parameters such as increasing or decreasing the rotation angles of the object or translating the entire object, the training process will not have to restart from the beginning but will be similar to moving the object to the new position. 

**Countable vision**: This concept will be revealed in the second phase of the project. 

## Table of Contents

- [Main Features](#main-features)
- [License](#license)

## Main Features
🚧Under Construction🚧

## License
[BSD 3](LICENSE)

<hr>

[Go to Top](#table-of-contents)
