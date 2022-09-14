#ifndef QUADRATICCOSTFUNCTION_H
#define QUADRATICCOSTFUNCTION_H

#include "CostFunction.h"

class Dataset;
class FeedforwardNeuralNetwork;

class QuadraticCostFunction : public CostFunction {
    public:
        QuadraticCostFunction();
        double getCost(const Dataset& dataset, FeedforwardNeuralNetwork* network) const;
};

#endif
