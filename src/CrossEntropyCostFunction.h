#ifndef CROSSENTROPYCOSTFUNCTION_H
#define CROSSENTROPYCOSTFUNCTION_H

#include "CostFunction.h"
class Dataset;
class FeedforwardNeuralNetwork;

class CrossEntropyCostFunction : public CostFunction {
    public:
        CrossEntropyCostFunction();
        double getCost(const Dataset& dataset, FeedforwardNeuralNetwork* network) const;
};

#endif
