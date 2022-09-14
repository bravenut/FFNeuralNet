#include "QuadraticCostFunction.h"

QuadraticCostFunction::QuadraticCostFunction() : CostFunction() {};

double QuadraticCostFunction::getCost(const Dataset& dataset, FeedforwardNeuralNetwork* network) const {
  // evaluates the quadratic cost function for the dataset
  int numSamples = dataset._testLabels.size();
  double cost = 0.0;
  double norm = 0.0;
  arma::vec desiredOutput;
  arma::vec diff;

  // loop over all data samples
  for (int i=0; i<numSamples; ++i) {
    // calculate output activation for given input image
    network->feedForward(dataset._testImages[i]);

    // calculate the desiredOutput vector (desiredOuput[label] = 1 , zero otherwise)
    desiredOutput = network->getDesiredOutputVectorFromLabel(dataset._testLabels[i]);

    // calculate the squared norm of the vector of differences of each neuron activation from the co>
    diff = desiredOutput - network->_activations[network->_numLayers-1];
    norm = arma::norm(diff);
    cost += norm*norm;
  }
  cost = 1/(2.0*numSamples) * cost;
  return cost;
};

