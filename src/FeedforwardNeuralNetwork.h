#ifndef FEEDFORWARDNEURALNETWORK_H
#define FEEDFORWARDNEURALNETWORK_H

#include <vector>
#include <armadillo>

#include "Dataset.h"

class FeedforwardNeuralNetwork {
  public:
    FeedforwardNeuralNetwork(const int numInputs, const int numOutputs, const int numHiddenLayers,
        const int numHiddenNeurons);

    double evaluate(const Dataset& dataset);   // returns recognition rate
    double getCost(const Dataset& dataset);    // returns cost function

    void train(const Dataset& dataset, const int batchsize, const double eta, const int epochs);

  private:
    const size_t _numLayers;
    std::vector<int> _layerconfig;

    void feedForward(const arma::vec& input);
    void backpropagate(const uint8_t& label);

    inline const arma::vec activationFunction(const arma::vec& input) const;
    inline const arma::vec activationFunctionPrime(const arma::vec& input) const;

    inline const arma::uword getIndexMaxActivation() const;
    inline const arma::vec getDesiredOutputVectorFromLabel(const uint8_t& label) const;

    std::vector<arma::mat> _weights;
    std::vector<arma::vec> _biases;
    std::vector<arma::vec> _weightedInputs;
    std::vector<arma::vec> _activations;

    std::vector<arma::mat> _gradWeights;
    std::vector<arma::vec> _gradBiases;
    std::vector<arma::mat> _gradWeightsBatch;
    std::vector<arma::vec> _gradBiasesBatch;

    inline void zeroBatchGradients();

    void initialize(const int numInputs, const int numOutputs, const int numHiddenLayers,
        const int numHiddenNeurons);

};

#endif
