# Boltzmann Machines (BMs)

# High-Level Abstract
Boltzmann Machines are stochastic neural networks composed of binary neurons that toggle between on/off states. These neurons are interconnected through a network that learns complex probability distributions from data. Unlike traditional neural networks, which typically rely on backpropagation for training, Boltzmann Machines utilize Gibbs Sampling. This physics-based technique involves iterative sampling from the probability distribution of the network, using a Markov chain process derived from statistical mechanics. This method systematically adjusts weights to gradually reduce discrepancies between the observed and sampled data, making it crucial for the network to learn and replicate complex data patterns efficiently.

Boltzmann Machines are versatile, tackling tasks ranging from image recognition, where they excel in identifying and classifying visual data, to learning the notorious XOR logic function—a challenge that early neural networks struggled to overcome. Their capabilities extend into generative modeling, where they shine in creating new data instances that are statistically coherent with the original training data, demonstrating their robustness in both discriminative and generative tasks.

For a more technical understanding of Boltzmann Machines and how to tune them with appropriate hardware, please review this paper: https://arxiv.org/pdf/2303.10728 

# Understanding of the Code

This repository is organized into Java program files and accompanying data files that together implement and utilize a Boltzmann Machine for various tasks. Here’s a breakdown of each file's purpose:

IsingModel.java: This file simulates the Ising model, which serves as a foundational concept for understanding the behavior of Boltzmann Machines. The simulation covers the dynamics of magnetic dipoles in a lattice, akin to how neurons in a Boltzmann Machine interact based on their binary states.
Key Functions: Includes routines for setting up the lattice, running the simulation at various temperatures, and visualizing the system's state. This helps demonstrate the energy minimization and equilibrium processes that are central to Gibbs Sampling in Boltzmann Machines.

Inference.java: Implements the core functionality of the Boltzmann Machine, focusing on the inference process using Gibbs Sampling. This file is crucial for understanding how the Boltzmann Machine uses learned data distributions to perform tasks such as data generation and pattern recognition.
Key Functions: Sets up the network architecture with initialized weights and biases, runs the Gibbs Sampling process to update neuron states, and adjusts parameters based on the divergence between sampled and actual data. It also includes methods to generate new data samples that reflect the trained model's understanding of the input data patterns.

Data Files (.txt Files)
Purpose: These text files contain the data used for training and testing the Boltzmann Machine. The data typically includes binary representations of images or other patterns that the network learns to model.
Usage: The data files are read by the Java programs at runtime to provide input for training the Boltzmann Machine. This includes both the input data for learning and validation data to test the accuracy of the network's inference capabilities.

