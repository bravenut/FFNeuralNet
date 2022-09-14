#ifndef DATASET_H
#define DATASET_H

// Include mnist data reader
#include "mnist/mnist_reader.hpp"

struct Dataset {
    Dataset() {
      // Load MNIST data and store in std::vectors, save pixels as uint8_t (range from 0 to 255)
      // (MNIST_DATA_LOCATION is set in CMakeLists.txt)
      mnist::MNIST_dataset<std::vector, arma::Col<uint8_t>, uint8_t> mnistDataset =
            mnist::read_dataset<std::vector, arma::Col, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

      for (int i=0; i<mnistDataset.training_images.size(); i++) {
        arma::vec image = arma::conv_to<arma::vec>::from(mnistDataset.training_images[i]);
        image = image * 1.0/255.0;
        _trainImages.push_back(image);
      }

      for (int i=0; i<mnistDataset.test_images.size(); i++) {
        arma::vec image = arma::conv_to<arma::vec>::from(mnistDataset.test_images[i]);
        image = image * 1.0/255.0;
        _testImages.push_back(image);
      }

      _trainLabels = mnistDataset.training_labels;
      _testLabels = mnistDataset.test_labels;
    };

    std::vector<arma::vec> _trainImages;
    std::vector<uint8_t> _trainLabels;
    std::vector<arma::vec> _testImages;
    std::vector<uint8_t> _testLabels;

};

#endif
