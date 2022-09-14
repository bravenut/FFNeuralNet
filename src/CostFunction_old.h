#ifndef COSTFUNCTION_H
#define COSTFUNCTION_H

#include <armadillo>

class Dataset;

class CostFunction {
    public:
        enum CostFunctionType { QUADRATIC, CROSSENTROPY };
        CostFunction(CostFunctionType costFunctionType);

        double getCost(const Dataset& dataset) const;
        CostFunctionType getCostFunctionType() const;

    private:
        const CostFunctionType _costFunctionType;
        // double getGradBiasLastLayer(arma::vec activationLastLayer, 
};

#endif
