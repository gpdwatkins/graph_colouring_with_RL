# Generating a Graph Colouring Heuristic with Deep Q-Learning and Graph Neural Networks

This repository contains the code for the paper Generating a Graph Colouring Heuristic with Deep Q-Learning and Graph Neural Networks by Watkins et al.

## Training a model

A training run can be initalised by calling `main.py`. This will use the dataset saved in `datasets/training_dataset.pickle` to learn a heuristic for the graph colouring problem. The training dataset contains 1000 graphs with between 15 and 50 vertices, generated using a variety of mechansisms. 

The learned policy will periodically be used to colour a dataset of validation graphs, saved in `datasets/validation_dataset.pickle`. The validation dataset contains 100 graphs, generated using the same process as the training dataset.

The learned parameters of the trained policy are saved in `outputs/training`, together with some summary statistics of the training run.

## Testing a trained model

An example of a trained policy is provided in `trained_policies/learned_parameters_GN`. The policy can be tested on the dataset of 20 graphs from Lemos et al. (2019) by running `test_learned_policies_on_dataset.py`. 

## Further questions
Please send any further questions to george.watkins@warwick.ac.uk