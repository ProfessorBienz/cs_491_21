#include "mpi.h"
#include <vector>
#include <numeric>

int inner_product(int n, std::vector<int>& x, std::vector<int>& y)
{
    int global_sum;
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += x[i]*y[i];
    MPI_Allreduce(&sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return global_sum;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int my_id, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc == 1)
    {
        printf("Requires command line argument for n\n");
        MPI_Finalize();
        return 0;
    }

    int N = (int) atoi(argv[1]);
    int n = N / num_procs;

    std::vector<int> x(n);
    std::vector<int> y(n);

    std::iota(x.begin(), x.end(), my_id*n);
    std::iota(y.begin(), y.end(), my_id*n);

    int global_sum = inner_product(n, x, y);
    if (my_id == 0) printf("Inner Product : %d\n", global_sum);

    MPI_Finalize();
    return 0;
}
