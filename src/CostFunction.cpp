#include "CostFunction.h"

CostFunction::CostFunction() {};

double CostFunction::getCost(const Dataset& dataset, FeedforwardNeuralNetwork* network) const {
    return 1.0;
}
