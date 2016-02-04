/*
 * Utility functions
 * Hins Pan
 * 2016.2.2
 */

#ifndef _CPP_UTIL_
#define _CPP_UTIL_

#include<iostream>
#include<string>
#include<sstream>
#include<boost/numeric/ublas/matrix.hpp>
#include<boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

// refer to matrix row
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace ParameterServer
{
using namespace std;

template < class T> 
void convert_from_string(T& value, const string& s)
{
    stringstream ss(s);
    ss >> value;
}

double norm(const boost::numeric::ublas::vector<double>& v1, const boost::numeric::ublas::vector<double>& v2); 

double norm_1(const boost::numeric::ublas::vector<double>& v1, const boost::numeric::ublas::vector<double>& v2); 

double sigmoid(double x);

void lr_without_regularization(bool sgd,
        double learning_rate,
        double epsilon,
        int max_iteration,
        boost::numeric::ublas::matrix<double>& x,
        boost::numeric::ublas::vector<double>& y,
        boost::numeric::ublas::vector<double>& weight,
        double &loss
        );
}
#endif
