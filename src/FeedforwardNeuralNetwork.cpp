#include <math.h>
#include <assert.h>
#include <iomanip>  // for the use of std::setprecision
#include <chrono>   // timing features

#include "FeedforwardNeuralNetwork.h"

FeedforwardNeuralNetwork::FeedforwardNeuralNetwork(const int numInputs, const int numOutputs,
      const int numHiddenLayers, const int numHiddenNeurons, CostFunction* costFunction) :
  _numLayers(numHiddenLayers+2),
  _costFunction(costFunction) {
  // (total number of layers is number of hidden layers + input layer + output layer)

  // initialize the layer configuration (numbers of neurons for each layer) and
  // all member vectors (weights, biases, weightedInputs, activations, gradBiases,
  // gradWeights, gradBiasesBatch, gradWeightsBatch)
  FeedforwardNeuralNetwork::initialize(numInputs, numOutputs, numHiddenLayers, numHiddenNeurons);
}

// Train the network with a given batchsize for the dataset (MNIST)
void FeedforwardNeuralNetwork::train(const Dataset& dataset, const int batchsize, const double eta, const int epochs) {
  unsigned int numBatches = dataset._trainLabels.size() / batchsize;
  assert(dataset._trainLabels.size() % batchsize == 0);

  std::vector<bool> sampleUsed(dataset._trainLabels.size(), false);
  unsigned int randIdx;

//  std::cout << "Training network with a batch size of " << batchsize << 
//    " (number of batches: " << numBatches << ")" << 
//    " using cost function " << static_cast<std::underlying_type<CostFunction::CostFunctionType>::type>(_costFunction) << 
//    "." << std::endl;

  for(int e=0; e<epochs; e++) {
    std::chrono::steady_clock::time_point timeBeginEpoch = std::chrono::steady_clock::now();
    // loop over all batches
    for (int i=0; i<numBatches; ++i) {

      // set gradients for the current batch to zero
      FeedforwardNeuralNetwork::zeroBatchGradients();

     // loop over current mini batch (images and labels)
      for (int j=0; j<batchsize; ++j) {
        do {
          // make sure to make a new randIdx which is not used yet
          randIdx = rand() % dataset._trainLabels.size();
        } while(sampleUsed[randIdx] == 1);

        // Feedforward the random input image and backpropagate to
        // obtain the gradients for the current sample
        // results are stored in _gradBiases and _gradWeights
        FeedforwardNeuralNetwork::feedForward(dataset._trainImages[randIdx]);
        FeedforwardNeuralNetwork::backpropagate(dataset._trainLabels[randIdx]);

        // calculate sum of gradients for each layer
        for(int l=0; l<_numLayers-1; ++l) {
          _gradBiasesBatch[l] = _gradBiasesBatch[l] + _gradBiases[l];
          _gradWeightsBatch[l] = _gradWeightsBatch[l] + _gradWeights[l];
        }

        sampleUsed[randIdx] = 1;
      } // loop over samples in one batch

      // calculate new weights and biases
      for(int l=0; l<_numLayers-1; ++l) {
        _biases[l] = _biases[l] - eta/batchsize * _gradBiasesBatch[l];
        _weights[l] = _weights[l] - eta/batchsize * _gradWeightsBatch[l];
      }

      sampleUsed.assign(sampleUsed.size(), false);
    } // loop over all batches

    std::chrono::steady_clock::time_point timeEndEpoch = std::chrono::steady_clock::now();

    // Evaluating the cost each epoch is very costly and should be removed at some point
    double cost;
    /*switch(_costFunction._costFunctionType) {
        case CostFunction::QUADRATIC:
            cost = FeedforwardNeuralNetwork::getQuadraticCost(dataset);
        case CostFunction::CROSSENTROPY:
            cost = FeedforwardNeuralNetwork::getCrossEntropyCost(dataset);
    }
    */
    cost = FeedforwardNeuralNetwork::getCost(dataset);

    // Calculate recognition rate after each epoch
    double recRate = FeedforwardNeuralNetwork::evaluate(dataset);

    std::cout << "\r|> Finished epoch " << e+1 << "/" << epochs
      << " (took " << std::chrono::duration_cast<std::chrono::milliseconds>(timeEndEpoch - timeBeginEpoch).count()
      << " ms) | current recognition rate: " << std::setprecision(4) << recRate << " | cost: " << cost << std::flush;
  } // loop over epochs
}

// Calculate all the weightedInputs and activations for each layer for a given input,
// most importantly the ones for the last layer (output)
void FeedforwardNeuralNetwork::feedForward(const arma::vec& input) {
  _activations[0] = input;

  for(int i=0; i<_numLayers-1; ++i) {
    _weightedInputs[i] = _weights[i] * _activations[i] + _biases[i];
    _activations[i+1] = FeedforwardNeuralNetwork::activationFunction(_weightedInputs[i]);
  }
}

// Execute the backpropagation algorithm for one data sample to obtain
// the gradients of the cost function with respect to weights and biases
void FeedforwardNeuralNetwork::backpropagate(const uint8_t& label) {
  const int N = _numLayers - 2;   // numLayers=3 -> N=1

  // c.f. backpropagation algorithm with quadratic/cross-entropy cost functions
  arma::vec gradBiasLastLayer;
//  switch(_costFunction.getCostFunctionType()) {
//    case CostFunction::CostFunctionType::QUADRATIC:
      gradBiasLastLayer = (_activations[N+1] - FeedforwardNeuralNetwork::getDesiredOutputVectorFromLabel(label)) % activationFunctionPrime(_weightedInputs[N]);
//    case CostFunction::CostFunctionType::CROSSENTROPY:
//      gradBiasLastLayer = (_activations[N+1] - FeedforwardNeuralNetwork::getDesiredOutputVectorFromLabel(label));
//  }

  _gradBiases[N] = gradBiasLastLayer;
  _gradWeights[N] = gradBiasLastLayer * _activations[N].t();

  // recursively calculate the gradients of the weights and biases (c.f. backpropagation algorithm)
  for(int i=0; i<N; ++i) {
    arma::vec tmpGradBias = (_weights[N-i].t() * _gradBiases[N-i]) % activationFunctionPrime(_weightedInputs[N-1-i]);
    _gradBiases[N-1-i] = tmpGradBias;
    _gradWeights[N-1-i] = tmpGradBias * _activations[N-1-i].t();
  }
}

// returns vector of "activated" weighted inputs
inline const arma::vec FeedforwardNeuralNetwork::activationFunction(const arma::vec& input) const {
  return 1/(1+arma::exp(-input));
}

// returns vector of derivatives of the activation function evaluated on the weighted inputs
inline const arma::vec FeedforwardNeuralNetwork::activationFunctionPrime(const arma::vec& input) const {
  arma::vec ones(input.size(), arma::fill::ones);
  // % stands for element-wise multiplication (c.f. armadillo docs)
  return FeedforwardNeuralNetwork::activationFunction(input) % (ones - FeedforwardNeuralNetwork::activationFunction(input));
}

inline const arma::vec FeedforwardNeuralNetwork::getDesiredOutputVectorFromLabel(const uint8_t& label) const {
  arma::vec res(10, arma::fill::zeros);
  assert(label < 10);
  res[label] = 1.0;
  return res;
}

inline const arma::uword FeedforwardNeuralNetwork::getIndexMaxActivation() const {
  return _activations[_numLayers-1].index_max();
}

// evaluates the quadratic cost function for the dataset
double FeedforwardNeuralNetwork::getQuadraticCost(const Dataset& dataset) {
  int numSamples = dataset._testLabels.size();
  double cost = 0.0;
  double norm = 0.0;
  arma::vec desiredOutput;
  arma::vec diff;

  // loop over all data samples
  for (int i=0; i<numSamples; ++i) {
    // calculate output activation for given input image
    FeedforwardNeuralNetwork::feedForward(dataset._testImages[i]);

    // calculate the desiredOutput vector (desiredOuput[label] = 1 , zero otherwise)
    desiredOutput = FeedforwardNeuralNetwork::getDesiredOutputVectorFromLabel(dataset._testLabels[i]);

    // calculate the squared norm of the vector of differences of each neuron activation from the correct label (desiredOutput)
    diff = desiredOutput - _activations[_numLayers-1];
    norm = arma::norm(diff);
    cost += norm*norm;
  }
  cost = 1/(2.0*numSamples) * cost;
  return cost;
}

// evaluates the cross-entropy cost function for the dataset
double FeedforwardNeuralNetwork::getCrossEntropyCost(const Dataset& dataset) {
  int numSamples = dataset._testLabels.size();
  arma::vec desiredOutput;
  arma::vec ones(10, arma::fill::ones);

  double cost = 0.0;

  // loop over all data samples
  for (int i=0; i<numSamples; ++i) {
    // calculate output activation for given input image
    FeedforwardNeuralNetwork::feedForward(dataset._testImages[i]);

    // calculate the desiredOutput vector (desiredOuput[label] = 1 , zero otherwise)
    desiredOutput = FeedforwardNeuralNetwork::getDesiredOutputVectorFromLabel(dataset._testLabels[i]);

    // calculate the squared norm of the vector of differences of each neuron activation from the correct label (desiredOutput)
    double ylna = arma::dot(desiredOutput, arma::log(_activations[_numLayers-1]));
    double oneminusterm = arma::dot(ones-desiredOutput, arma::log(ones-_activations[_numLayers-1]));
    cost = cost + ylna + oneminusterm;
    //std::cout << "ylna and oneminusterm: " << ylna << " and " << oneminusterm << " (sum: " << (ylna+oneminusterm) << ")" << std::endl;
    //std::cout << "current cost: " << cost << std::endl;
  }
  cost = -1.0/numSamples * cost;
  return cost;
}

double FeedforwardNeuralNetwork::getCost(const Dataset& dataset) {
    return _costFunction->getCost(dataset,this);
}

// calculates recognition rate of the network for the given dataset
double FeedforwardNeuralNetwork::evaluate(const Dataset& dataset) {
  int numSamples = dataset._testLabels.size();
  arma::Col<uint8_t> diff(dataset._testLabels.size());

  // loop over data samples
  for (int i=0; i<numSamples; ++i) {
    FeedforwardNeuralNetwork::feedForward(dataset._testImages[i]);
    diff[i] = (uint8_t)(FeedforwardNeuralNetwork::getIndexMaxActivation()) - dataset._testLabels[i];
  }
  arma::uvec nonzeroDiffs = find(diff);

  double recognitionRate = (numSamples - nonzeroDiffs.size()) * 1.0/numSamples;
  return recognitionRate;
}

void FeedforwardNeuralNetwork::initialize(const int numInputs, const int numOutputs, const int numHiddenLayers,
    const int numHiddenNeurons)
{
  // setup vector with number of neurons in each layer
  _layerconfig.push_back(numInputs);
  for (int i=0; i<numHiddenLayers; ++i) {
    _layerconfig.push_back(numHiddenNeurons);
  }
  _layerconfig.push_back(numOutputs);

  // prepare to initialize weights, biases,
  // weightedInputs and activations vectors
  arma::mat weights;
  arma::vec biases;
  arma::vec weightedInputs;
  arma::vec activations;

  // initialize first activations vector aka input
  activations = arma::vec(_layerconfig[0], arma::fill::zeros);
  _activations.push_back(activations);

  // iterate over all layers
  for(int i=0; i<_layerconfig.size()-1; ++i) {
    const unsigned int numNeuronsInThisLayer = _layerconfig[i];
    const unsigned int numNeuronsInNextLayer = _layerconfig[i+1];

    // normally distributed random weights connecting layer i with layer i+1
    weights.randn(numNeuronsInNextLayer, numNeuronsInThisLayer);
    _weights.push_back(weights);

    // normally distributed random biases of layer i
    biases.randn(numNeuronsInNextLayer);
    _biases.push_back(biases);

    // initialze all elements of _weightedInputs to zero
    weightedInputs = arma::vec(numNeuronsInThisLayer, arma::fill::zeros);
    _weightedInputs.push_back(weightedInputs);

    // initialze all elements of _activations to zero
    activations = arma::vec(numNeuronsInNextLayer, arma::fill::zeros);
    _activations.push_back(activations);
  }

  // setup gradients of weights and biases for one sample aswell as for sample batches
  _gradBiases.resize(_numLayers-1);   // sets size and initializes all elements to zero
  _gradWeights.resize(_numLayers-1);  // sets size and initializes all elements to zero
  _gradBiasesBatch.resize(_numLayers-1);   // sets size and initializes all elements to zero
  _gradWeightsBatch.resize(_numLayers-1);  // sets size and initializes all elements to zero
}

inline void FeedforwardNeuralNetwork::zeroBatchGradients() {
  // iterate over all layers
  for(int i=0; i<_layerconfig.size()-1; ++i) {
    const unsigned int numNeuronsInThisLayer = _layerconfig[i];
    const unsigned int numNeuronsInNextLayer = _layerconfig[i+1];

    // initialize weights gradients for each layer to zero
    _gradWeightsBatch[i] = arma::zeros(numNeuronsInNextLayer, numNeuronsInThisLayer);

    // initialize biases gradients for each layer to zero
    _gradBiasesBatch[i] = arma::zeros(numNeuronsInNextLayer);
  }
}
