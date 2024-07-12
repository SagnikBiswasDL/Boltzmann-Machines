# Boltzmann Machines (BMs)

# High-Level Abstract
Boltzmann Machines are stochastic neural networks composed of binary neurons that toggle between on/off states. These neurons are interconnected through a network that learns complex probability distributions from data. Unlike traditional neural networks, which typically rely on backpropagation for training, Boltzmann Machines utilize Gibbs Sampling. This physics-based technique involves iterative sampling from a probability distribution using a Markov chain process derived from statistical mechanics. This method systematically adjusts weights to gradually reduce discrepancies between the observed and sampled data, making it crucial for the network to learn and replicate complex data patterns efficiently.

Boltzmann Machines are versatile, tackling tasks ranging from image recognition, where they excel in identifying and classifying visual data, to learning the notorious XOR logic function—a challenge that early neural networks struggled to overcome. Their capabilities extend into generative modeling, where they shine in creating new data instances that are statistically coherent with the original training data, demonstrating their robustness in both discriminative and generative tasks.

For a more technical understanding of Boltzmann Machines and how to tune them with appropriate hardware, please review this paper: https://arxiv.org/pdf/2303.10728 

# Understanding The Code

This repository includes Java program files simulating a Boltzmann Machine that is capable of learning logic:

### `IsingModel.java`
- **Purpose**: This file manages the learning process of the Boltzmann Machine which is implemented through Contrastive Divergence. 
### `Inference.java`
- **Purpose**: This file manages the state generation phase of the Boltzmann Machine. It handles the sampling process that updates the network states based on their probability distributions. 


### Data Files (`.txt` Files)
- **Purpose**: These text files contain the necessary data to training the Boltzmann Machine in IsingModel.java

