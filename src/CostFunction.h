#ifndef COSTFUNCTION_H
#define COSTFUNCTION_H

#include <armadillo>

class Dataset;
class FeedforwardNeuralNetwork;

class CostFunction {
    public:
        CostFunction();
        virtual double getCost(const Dataset& dataset, FeedforwardNeuralNetwork* network) const;
};

#endif
