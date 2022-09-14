#include "CostFunction.h"

CostFunction::CostFunction(CostFunctionType costFunctionType) : _costFunctionType(costFunctionType) {};

double CostFunction::getCost(const Dataset& dataset) const {
    return 2.0;
}

CostFunction::CostFunctionType CostFunction::getCostFunctionType() const {
    return _costFunctionType;
}
