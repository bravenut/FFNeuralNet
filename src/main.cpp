// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <chrono> // for time measurements
#include <vector>

// Library for matrix vector operations
#include <armadillo>

#include "FeedforwardNeuralNetwork.cpp"
#include "CostFunction.cpp"
#include "QuadraticCostFunction.cpp"
#include "CrossEntropyCostFunction.cpp"

int main() {
  // Initialize the random generator (should do this where things get randomized)
  // arma::arma_rng::set_seed(1);                    // for reproducible results
  arma::arma_rng::set_seed_random();      // for pseudo-random results

  Dataset mnistDataset;

  const int numInputs = 784;          // image pixels (uint8_t values)
  const int numOutputs = 10;          // numbers from 0-9 (uconst int8_t)
  const int numHiddenLayers = 3;      // number of hidden layers
  const int numHiddenNeurons = 20;    // number of neurons in each hidden layer

  const float eta = 3.0;        // learning rate (default: 3.0)
  const int batchsize = 60;      // batchsize to calculate gradients with (default: 20)
  const int epochs = 5;         // number of learning epochs (default: 6)

  // srand(1); // set seed for random number generator
  // const CostFunction costFunction = CostFunction::QUADRATIC;
  CostFunction* costFunction = new QuadraticCostFunction(); //{CostFunction::CostFunctionType::QUADRATIC};

  FeedforwardNeuralNetwork network(numInputs, numOutputs, numHiddenLayers, numHiddenNeurons, costFunction);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  network.train(mnistDataset, batchsize, eta, epochs);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "\nTraining finished! Took " << (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count())/1000.0
    << " s in total" << std::endl;
  std::cout << "Evaluating network on test dataset... ";

  double recognitionRate = network.evaluate(mnistDataset);

  std::cout << "Recognized " << recognitionRate*100
    << "% of the test data samples correctly. " << std::endl;

  return 0;
}
