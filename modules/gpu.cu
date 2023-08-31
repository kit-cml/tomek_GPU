#include "cellmodels/enums/enum_ord2011.hpp"


__device__ void kernel_ApplyDrugEffect(unsigned short offset, double conc, double *ic50, double epsilon, double *CONSTANTS)
{
  // int offset = threadIdx.x;
  // int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int num_of_constants = 146;

CONSTANTS[GK1+(offset * num_of_constants)] = CONSTANTS[GK1+(offset * num_of_constants)] * ((ic50[2 + (offset*14)] > epsilon && ic50[3+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[2+ (offset*14)],ic50[3+ (offset*14)])) : 1.);
CONSTANTS[GKr+(offset * num_of_constants)] = CONSTANTS[GKr+(offset * num_of_constants)] * ((ic50[12+ (offset*14)] > epsilon && ic50[13+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[12+ (offset*14)],ic50[13+ (offset*14)])) : 1.);
CONSTANTS[GKs+(offset * num_of_constants)] = CONSTANTS[GKs+(offset * num_of_constants)] * ((ic50[4 + (offset*14)] > epsilon && ic50[5+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[4+ (offset*14)],ic50[5+ (offset*14)])) : 1.);
CONSTANTS[GNaL+(offset * num_of_constants)] = CONSTANTS[GNaL+(offset * num_of_constants)] = CONSTANTS[GNaL+(offset * num_of_constants)] * ((ic50[8+ (offset*14)] > epsilon && ic50[9+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[8+ (offset*14)],ic50[9+ (offset*14)])) : 1.);
CONSTANTS[GNa+(offset * num_of_constants)] = CONSTANTS[GNa+(offset * num_of_constants)] * ((ic50[6 + (offset*14)] > epsilon && ic50[7+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[6+ (offset*14)],ic50[7+ (offset*14)])) : 1.);
CONSTANTS[Gto+(offset * num_of_constants)] = CONSTANTS[Gto+(offset * num_of_constants)] * ((ic50[10 + (offset*14)] > epsilon && ic50[11+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[10+ (offset*14)],ic50[11+ (offset*14)])) : 1.);
CONSTANTS[PCa+(offset * num_of_constants)] = CONSTANTS[PCa+(offset * num_of_constants)] * ( (ic50[0 + (offset*14)] > epsilon && ic50[1+ (offset*14)] > epsilon) ? 1./(1.+pow(conc/ic50[0+ (offset*14)],ic50[1+ (offset*14)])) : 1.);
}

__device__ double kernel_SetTimeStep(
    unsigned short offset, 
    double TIME,
    double time_point,
    double max_time_step,
    double* CONSTANTS,
    double* RATES) 

    {
    double time_step = 0.005;
    // int offset = threadIdx.x;
    // int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int num_of_constants = 146;
    int num_of_rates = 41; 
    if (TIME <= time_point || (TIME - floor(TIME / CONSTANTS[stim_period + (offset * num_of_constants)]) * CONSTANTS[stim_period + (offset * num_of_constants)] ) <= time_point) {
        return time_step;   
    }
    else {  
        if (std::abs(RATES[V + (offset * num_of_rates)] * time_step) <= 0.2) {//Slow changes in V
            time_step = std::abs(0.8 / RATES[V + (offset * num_of_rates)] );
            if (time_step < 0.005) {
                time_step = 0.005;
            }
            else if (time_step > max_time_step) {
                time_step = max_time_step;
            }
        }
        else if (std::abs(RATES[V + (offset * num_of_rates)] * time_step) >= 0.8) {//Fast changes in V
            time_step = std::abs(0.2 / RATES[V+ (offset * num_of_rates)]);
            while (std::abs(RATES[V+ (offset * num_of_rates)] * time_step) >= 0.8 && 0.005 < time_step && time_step < max_time_step) {
                time_step = time_step / 10.0;
            }
        }
        // __syncthreads(); //re investigate do we really need this?
        return time_step;
    }
}


__device__ void kernel_DoDrugSim(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, 
                                       double *d_ALGEBRAIC, double *time, double *out_dt, double *states,
                                       double *ical, double *inal, unsigned short sample_id, double *tcurr, 
                                       double *dt, unsigned int sample_size){
    
    unsigned int input_counter = 0;
    unsigned short cnt;

    int num_of_constants = 146;
    int num_of_states = 41;
    int num_of_algebraic = 199;
    // int num_of_rates = 41;

    tcurr[sample_id] = 0.000001;
    dt[sample_id] = 0.005;
    double tmax;
    double max_time_step = 1.0, time_point = 25.0;
    double dt_set;

    bool writen = false;

    // files for storing results
    // time-series result
    // FILE *fp_vm, *fp_inet, *fp_gate;

    // features
    // double inet, qnet;

    // looping counter
    // unsigned short idx;
  
    // simulation parameters
    // double dtw = 2.0;
    // const char *drug_name = "bepridil";
    const double bcl = 2000; // bcl is basic cycle length
    
    // const double inet_vm_threshold = -88.0;
    // const unsigned short pace_max = 300;
    const unsigned short pace_max = 1000;
    // const unsigned short celltype = 0.;
    // const unsigned short last_pace_print = 3;
    // const unsigned short last_drug_check_pace = 250;
    // const unsigned int print_freq = (1./dt) * dtw;
    // unsigned short pace_count = 0;
    // unsigned short pace_steepest = 0;
    // double conc = 243.0; //mmol
    double conc = 0.0;


    // printf("Core %d:\n",sample_id);
    initConsts(sample_id,d_CONSTANTS, d_STATES);

    kernel_ApplyDrugEffect(sample_id,conc,d_ic50,10E-14,d_CONSTANTS);

    d_CONSTANTS[stim_period + (sample_id * num_of_constants)] = bcl;

    // generate file for time-series output

    tmax = pace_max * bcl;
    int pace_count = 0;
  
    // printf("%d,%lf,%lf,%lf,%lf\n", sample_id, dt_set[sample_id], tcurr, d_STATES[V + (sample_id * num_of_states)],d_RATES[V + (sample_id * num_of_rates)]);

    while (tcurr[sample_id]<tmax){
        dt_set = kernel_SetTimeStep(sample_id, tcurr[sample_id], time_point, max_time_step, d_CONSTANTS, d_RATES); 
        computeRates(sample_id, tcurr[sample_id], d_CONSTANTS, d_RATES, d_STATES, d_ALGEBRAIC); 
        if (floor((tcurr[sample_id] + dt_set) / bcl) == floor(tcurr[sample_id] / bcl)) { 
          dt[sample_id] = dt_set;
        }
        else {
          dt[sample_id] = (floor(tcurr[sample_id] / bcl) + 1) * bcl - tcurr[sample_id];
          pace_count++;
          writen = false;
          // printf("core %d, pace_count: %d, tcurr: %lf\n", sample_id, pace_count, tcurr);
          // printf("timestep corrected in core %d \n", sample_id);
        }
        if(sample_id==0 && pace_count%10==0 && pace_count>99 && !writen){
        // printf("Calculating... watching core 0: %.2lf %% done\n",(tcurr[sample_id]/tmax)*100.0);
        printf("[");
        for (cnt=0; cnt<pace_count/10;cnt++){
          printf("=");
        }
        for (cnt=pace_count/10; cnt<pace_max/10;cnt++){
          printf("_");
        }
        printf("] %.2lf %% \n",(tcurr[sample_id]/tmax)*100.0);
        //mvaddch(0,pace_count,'=');
        //refresh();
        //system("clear");
        writen = true;
        }
        solveAnalytical(sample_id, dt[sample_id], d_CONSTANTS, d_RATES, d_STATES, d_ALGEBRAIC);
        tcurr[sample_id] = tcurr[sample_id] + dt[sample_id];
       
        if (pace_count > pace_max-2){
        time[input_counter + sample_id] = tcurr[sample_id];
        out_dt[input_counter + sample_id] = dt[sample_id];
        states[input_counter + sample_id] = d_STATES[V + (sample_id * num_of_states)];
        ical[input_counter + sample_id] = d_ALGEBRAIC[ICaL + (sample_id * num_of_algebraic)];
        inal[input_counter + sample_id] = d_ALGEBRAIC[INaL + (sample_id * num_of_algebraic)];
        input_counter = input_counter + sample_size;
        //printf("counter: %d core: %d\n",input_counter,sample_id);
        }
    }
    // __syncthreads();
    //avoid race condition with this? 
    //But the waiting become too long for 2< samples
}



__global__ void kernel_DrugSimulation(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, 
                                       double *d_ALGEBRAIC, double *time, double *out_dt, double *states,
                                       double *ical, double *inal, unsigned int sample_size){
    unsigned short sample_id;
    
    sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    double time_for_each_sample[56000];
    double dt_for_each_sample[56000];
    
    // printf("Calculating %d\n",sample_id);
    do_drug_sim_analytical(d_ic50, d_CONSTANTS, d_STATES, d_RATES, d_ALGEBRAIC, 
                          time, out_dt, states, ical, inal, sample_id, 
                          time_for_each_sample, dt_for_each_sample, sample_size);
                          // __syncthreads();
    // printf("Calculation for core %d done\n",sample_id);
    
  }