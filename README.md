# Boltzmann Machines (BMs)

# High-Level Abstract
Boltzmann Machines are stochastic neural networks composed of binary neurons that toggle between on/off states. These neurons are interconnected through a network that learns complex probability distributions from data. Unlike traditional neural networks, which typically rely on backpropagation for training, Boltzmann Machines utilize Gibbs Sampling. This physics-based technique involves iterative sampling from the probability distribution of the network, using a Markov chain process that derived from statistical mechanics. This method systematically adjusts weights to gradually reduce discrepancies between the observed and sampled data, making it crucial for the network to learn and replicate complex data patterns efficiently.

Boltzmann Machines are versatile, tackling tasks ranging from image recognition, where they excel in identifying and classifying visual data, to learning the notorious XOR logic function—a challenge that early neural networks struggled to overcome. Their capabilities extend into generative modeling, where they shine in creating new data instances that are statistically coherent with the original training data, demonstrating their robustness in both discriminative and generative tasks.

For a more technical understanding of Boltzmann Machines and how to properly train them with appropriate hardware, please review this paper: https://arxiv.org/pdf/2303.10728 

# Understanding of the Code

