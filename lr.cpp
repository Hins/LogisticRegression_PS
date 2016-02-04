/*
 * Model-distribution logistic regression without regularization items
 * Hins Pan
 * 2016.2.2
 */
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <map>

#include <stdio.h>
#include <stdlib.h>

#include <time.h>

using namespace std;
using namespace ParameterServer;
using namespace boost::numeric::ublas;

int main(int argc, char* argv[]) {
    // Parameter check;
    if (argc < 8) {
        cout<<"Usage: <server size> <feature size> <epsilon> <learning rate> <max iteration> <training set> <sgd flag>"<<endl;
        return -1;
    }

    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name,&namelen);
    cout<<"I'm rank "<<mpi_rank<<" of "<<mpi_procs<<" in "<<processor_name<<std::endl;

    int mpi_rank = 0, mpi_procs = 0, feature_size = 0, max_iteration = 0, server_size = 0, record_num = 0, sgd = 0;
    double epsilon = 0.0, learning_rate = 0.0;
    string file_path = "";
    
    // Convert parameter;
    string str = argv[1];
    convert_from_string(server_size, str);
    if (mpi_procs < server_size)
    {
        cout<<"Server size is more than mpi procs"<<endl;
        return -1;
    }
    str = argv[2];
    convert_from_string(feature_size, str);
    str = argv[3];
    convert_from_string(epsilon, str);
    str = argv[4];
    convert_from_string(learning_rate, str);
    str = argv[5];
    convert_from_string(max_iteration, str);
    file_path = argv[6];
    str = argv[7];
    convert_from_string(sgd, str);

    // Calculate sample size;
    string line;
    record_num = 0;
    ifstream in(file_path.c_str());
    if (in.is_open())
    {
        while (getline(in, line))
        {
            record_num++;
        }
    }
    in.close();

    // Not use Parameter-Server architecture;
    if (mpi_procs == 1)
    {
        // y is label;
        boost::numeric::ublas::vector<double> y(record_num);
        // x is training set;
        boost::numeric::ublas::matrix<double> x(record_num, feature_size + 1);
        SimpleDataLoader loader(record_num, feature_size + 1);
        loader.load_file(file_path.c_str(), y, x, 1, record_num);
        cout<<"Load file complete"<<endl;

        // Initialize weight into zero, also could initialize them with some distribution, as well as Gaussian;
        boost::numeric::ublas::vector<double> weight(x.size2());
        for (size_t i = 0; i < weight.size(); ++i)
        {
            weight(i) = 0;
        }
        double loss = 0.0;
        clock_t start = clock(), end;
        // Logistic regression training
        lr_without_regularization(sgd == 1, learning_rate, epsilon, max_iteration, x, y, weight, loss);
        end = clock();
        cout << "Runtime： " << (double)(end - start) / CLOCKS_PER_SEC << " s!" << endl;
    }
    else
    {
        // Server node;
        if (mpi_rank < server_size)
        {
            lr_server* lrServerObj = new lr_server(mpi_rank, feature_size, max_iteration, epsilon);
            lrServerObj->Run();
            delete lrServerObj;
        }
        // Worker node;
        else
        {
            lr_worker *lrWorkerObj = new lr_worker(file_path, mpi_rank, mpi_procs, server_size, record_num, feature_size, epsilon, learning_rate, max_iteration, sgd == 1);
            lrWorkerObj->Run();
            delete lrWorkerObj;
        }
    }
    MPI_Finalize();
    return 0;
}
