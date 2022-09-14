#include "CrossEntropyCostFunction.h"

CrossEntropyCostFunction::CrossEntropyCostFunction() : CostFunction() {};

double CrossEntropyCostFunction::getCost(const Dataset& dataset, FeedforwardNeuralNetwork* network) const {
  int numSamples = dataset._testLabels.size();
  arma::vec desiredOutput;
  arma::vec ones(10, arma::fill::ones);

  double cost = 0.0;

  // loop over all data samples
  for (int i=0; i<numSamples; ++i) {
    // calculate output activation for given input image
    network->feedForward(dataset._testImages[i]);

    // calculate the desiredOutput vector (desiredOuput[label] = 1 , zero otherwise)
    desiredOutput = network->getDesiredOutputVectorFromLabel(dataset._testLabels[i]);

    // calculate the squared norm of the vector of differences of each neuron activation from the co>
    double ylna = arma::dot(desiredOutput, arma::log(network->_activations[network->_numLayers-1]));
    double oneminusterm = arma::dot(ones-desiredOutput, arma::log(ones-network->_activations[network->_numLayers-1]));
    cost = cost + ylna + oneminusterm;
    //std::cout << "ylna and oneminusterm: " << ylna << " and " << oneminusterm << " (sum: " << (yln>
    //std::cout << "current cost: " << cost << std::endl;
  }
  cost = -1.0/numSamples * cost;
  return cost;
};
