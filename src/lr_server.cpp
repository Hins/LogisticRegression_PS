#include "lr_server.h"

namespace ParameterServer
{
    void lr_server::Run()
    {
        MPI_Comm Reduce_comm;
        // Split all MPI nodes into one communication channel;
        if (m_mpi_rank == 0)
        {
            MPI_Comm_split(MPI_COMM_WORLD, 0, m_mpi_rank, &Reduce_comm);
        }
        // Initialize weight vector for reduce and broadcast;
        double *weight_old = new double[m_feature_size + 1];
        for (int k = 0; k < m_feature_size + 1; k++)
        {
            weight_old[k] = 0;
        }
        double *weight = new double[m_feature_size + 1];
        double preLoss = 0.0;
        int counter = 1;

        clock_t start, end;
        start = clock();
        while (counter <= m_max_iteration)
        {
            double *totalLoss = new double[1];
            double *send = new double[1];
            send[0] = 0;
            // Aggregate loss out of workers;
            MPI_Reduce(send, totalLoss, 1, MPI_DOUBLE, MPI_SUM, 0, Reduce_comm);
            // Broadcast total loss to workers;
            MPI_Bcast(totalLoss, 1, MPI_DOUBLE, 0, Reduce_comm);
            delete send;
            cout<<counter<<" server loss is "<<totalLoss[0]<<"; previous loss is "<<preLoss<<"; abs is "<<((preLoss - totalLoss[0]) > 0 ? (preLoss - totalLoss[0]) : (preLoss - totalLoss[0]) * -1.0)<<endl;
            // Check whether it would terminate learning by epsilon or loss difference;
            if (totalLoss[0] < m_epsilon ||
                ((preLoss - totalLoss[0]) > 0 ? (preLoss - totalLoss[0]) : (preLoss - totalLoss[0]) * -1.0) < m_epsilon)
            {
                delete totalLoss;
                break;
            }
            preLoss = totalLoss[0];
            // Aggregate weight out of workers;
            MPI_Reduce(weight_old, weight, m_feature_size + 1, MPI_DOUBLE, MPI_SUM, 0, Reduce_comm);
            // Broadcast weight to workers;
            MPI_Bcast(weight, m_feature_size + 1, MPI_DOUBLE, 0, Reduce_comm);
            delete totalLoss;
            counter++;
        }

        end = clock();
        cout << "Runtime： " << (double)(end - start) / CLOCKS_PER_SEC << " s!" << endl;
        delete weight;
        MPI_Comm_free(&Reduce_comm);
    }
}
