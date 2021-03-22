#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>

BOOST_AUTO_TEST_CASE( test_backprop, * boost::unit_test::tolerance(1e-15) ){
  std::cout << "testing backpropagation algorithm" << std::endl;
}

