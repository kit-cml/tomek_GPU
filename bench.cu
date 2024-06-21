/**
 * @file main.cu
 * @brief Main entry point for the Drug Simulation using CUDA
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include "modules/glob_funct.hpp"
#include "modules/glob_type.hpp"
#include "modules/gpu.cuh"
#include "utils/constants.hpp"
#include "utils/file_operations.hpp"
#include "utils/gpu_operations.hpp"
#include "utils/timing.hpp"

/**
 * @brief Main function for running the drug simulation
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @return int Exit status of the program
 */
int main(int argc, char **argv) {
    param_t *p_param = new param_t();  // input data for CPU
    param_t *d_p_param;                // input data for GPU parsing
    p_param->init();
    edison_assign_params(argc, argv, p_param);

    double *ic50 = (double *)malloc(14 * sample_limit * sizeof(double));
    double *conc = (double *)malloc(sample_limit * sizeof(double));
    double *cvar = (double *)malloc(18 * sample_limit * sizeof(double));
    char *drug_name = get_drug_name(p_param->hill_file);
    double *d_ic50, *d_conc, *d_cvar, *d_ALGEBRAIC, *d_CONSTANTS, *d_RATES, *d_STATES, *d_STATES_RESULT, *d_STATES_init;
    // double *d_mec_ALGEBRAIC, *d_mec_CONSTANTS, *d_mec_RATES, *d_mec_STATES;
    double *time, *dt, *states, *ical, *inal, *cai_result, *ina, *ito, *ikr, *iks, *ik1;
    cipa_t *temp_result, *cipa_result;

    int sample_size = get_IC50_data_from_file(p_param->hill_file, ic50, conc);
    int blocksPerGrid = (sample_size + threadsPerBlock - 1) / threadsPerBlock;
    printf("Sample size: %d\nSet GPU Number: %d\n", sample_size, p_param->gpu_index);

    cudaSetDevice(p_param->gpu_index);

    if (p_param->is_cvar) {
        int cvar_sample = get_cvar_data_from_file(p_param->cvar_file, sample_size, cvar);
        printf("Reading: %d Conductance Variability samples\n", cvar_sample);
    }

    prepingGPUMemory(sample_size, d_ALGEBRAIC, d_CONSTANTS, d_RATES, d_STATES, d_p_param, temp_result, cipa_result, d_STATES_RESULT, d_ic50, ic50,
                     d_conc, conc, p_param);

    tic();

    printf("Timer started, doing simulation.... \n\n\nGPU Usage at this moment: \n");
    if (gpu_check(15 * sample_size * datapoint_size * sizeof(double) + sizeof(param_t)) == 1) {
        printf("GPU memory insufficient!\n");
        return 1;
    }
    printf("\n   Configuration: \n\n\tblock\t||\tthread\n---------------------------------------\n  \t%d\t||\t%d\n\n\n",
           blocksPerGrid, threadsPerBlock);

    kernel_DrugSimulation<<<blocksPerGrid, threadsPerBlock>>>(
        d_ic50, d_cvar, d_conc, d_CONSTANTS, d_STATES, d_STATES_init, d_RATES, d_ALGEBRAIC, d_STATES_RESULT,
        time, states, dt, cai_result, ina, inal, ical, ito,
        ikr, iks, ik1, sample_size, temp_result, cipa_result, d_p_param);
    cudaDeviceSynchronize();

    double *h_states = (double *)malloc(ORd_num_of_states * sample_size * sizeof(double));
    cipa_t *h_cipa_result = (cipa_t *)malloc(sample_size * sizeof(cipa_t));

    printf("copying the data back to the CPU \n");
    cudaMemcpy(h_states, d_STATES_RESULT, sample_size * ORd_num_of_states * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cipa_result, cipa_result, sample_size * sizeof(cipa_t), cudaMemcpyDeviceToHost);

    write_results_to_file("./result", drug_name, h_states, h_cipa_result, sample_size, ORd_num_of_states);

    freeingMemory(d_ALGEBRAIC, d_CONSTANTS, d_RATES, d_STATES, d_p_param, temp_result, cipa_result, d_STATES_RESULT, d_ic50, ic50, conc, h_states,
                  h_cipa_result, p_param);

    toc();

    return 0;
}
