#include "util.h"

namespace ParameterServer
{
using namespace std;

//////////////////// utils/////////////////////////////////
double norm(const boost::numeric::ublas::vector<double>& v1, const boost::numeric::ublas::vector<double>& v2)
{
    assert (v1.size() == v2.size());
    double sum = 0;
    for (size_t i=0; i<v1.size(); ++i)
    {
        double minus = v1(i) - v2(i);
        double r = minus * minus;
        sum += r;
    }

    return sqrt(sum);
}


double norm_1(const boost::numeric::ublas::vector<double>& v1, const boost::numeric::ublas::vector<double>& v2)
{
    assert (v1.size() == v2.size());
    double sum = 0;
    for (size_t i=0; i<v1.size(); ++i)
    {
        double minus = abs(v1(i) - v2(i));
        sum += minus;
    }
    return sum;
}

double sigmoid(double x)
{
    double e = 2.718281828;
    return 1.0 / (1.0 + pow(e, -x));
}

void lr_without_regularization(bool sgd,
        double learning_rate,
        double epsilon,
        int max_iteration,
        boost::numeric::ublas::matrix<double>& x,
        boost::numeric::ublas::vector<double>& y,
        boost::numeric::ublas::vector<double>& weight,
        double &loss
        )
{

    int iter = 0;

    // Initialize predication vector;
    boost::numeric::ublas::vector<double> prediction(x.size1());
    for (size_t i = 0; i < prediction.size(); ++i)
    {
        prediction(i) = 0;
    }

    while (iter < max_iteration)
    {
        // Traverse features;
        for (size_t k=0; k<x.size2(); ++k)
        {
            double gradient = 0;
            // Traverse samples;
            for (size_t i=0; i<x.size1(); ++i)
            {
                double z_i = 0;
                // Traverse features;
                for (size_t j=0; j<weight.size(); ++j)
                {
                    z_i += weight(j) * x(i,j);
                }
                // SGD: update weight per sample;
                if (sgd)
                {
                    weight(k) += (y(i) - sigmoid(z_i)) * learning_rate * x(i, k);
                }
                // BGD: aggregate gradient change;
                else
                {
                    gradient += (y(i) - sigmoid(z_i)) * learning_rate * x(i,k); 
                }
            }
            // BGD: update weight by all samples
            if (!sgd)
            {
                weight(k) += gradient / x.size1();
            }
        }
        // Calcuate loss by square error
        for (size_t k = 0; k<x.size1(); ++k)
        {
            double temp = 0.0;
            for (size_t j = 0; j < x.size2(); ++j)
            {
                temp += weight(j) * x(k,j);
            }
            loss += pow(2, sigmoid(temp) - y(k));
        }
        loss /= (2 * x.size1());
        // Termination condition by epsilon;
        if (loss < epsilon)
        {
            break;
        }

        /*double dist = norm(weight_new, weight_old);
          if (dist < epsilon) {
              cout << "the best weight: " << weight_new << endl;
              break;
          }
          else {
               weight_old.swap(weight_new);
               //weight_old = weight_new;
        }*/

        iter++;

        //cout << "================================================" << endl;
        //cout << "The " << iter << " th iteration, loss: " << loss << ", weight: "<<endl;
        //cout << weight << endl << endl;
        //cout << "the diff between the old weight and the new weight: " << dist << endl << endl;
    }

    //cout << "The best weight:" << endl;
    //cout << weight << endl;
}
}
