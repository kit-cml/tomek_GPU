// #include "cellmodels/enums/enum_Ohara_Rudy_2011.hpp"
#include "../cellmodels/Ohara_Rudy_2011.hpp"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "glob_funct.hpp"
#include "glob_type.hpp"
#include "gpu_glob_type.cuh"
#include "gpu.cuh"


/*
all kernel function has been moved. Unlike the previous GPU code, now we seperate everything into each modules.
all modules here are optimised for GPU and slightly different than the original component based code
differences are related to GPU offset calculations
*/

__device__ void kernel_DoDrugSim(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, double *d_ALGEBRAIC, 
                                       double *time, double *states, double *out_dt,  double *cai_result, 
                                       double *ina, double *inal,
                                       double *ical, double *ito,
                                       double *ikr, double *iks, 
                                       double *ik1,
                                       double *tcurr, double *dt, unsigned short sample_id, unsigned int sample_size,
                                       cipa_t *temp_result,
                                       param_t *p_param
                                       )
    {
    
    unsigned int input_counter = 0;
    unsigned short cnt;

    int num_of_constants = 146;
    int num_of_states = 41;
    int num_of_algebraic = 199;
    int num_of_rates = 41;


    // cipa_t cipa_result, 
    // cipa_t temp_result;

    tcurr[sample_id] = 0.000001;
    dt[sample_id] = p_param->dt;
    double tmax;
    double max_time_step = 1.0, time_point = 25.0;
    double dt_set;

    int datapoint_at_this_moment;

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
    // const double bcl = 2000; // bcl is basic cycle length
    const double bcl = p_param->bcl;
    
    // const double inet_vm_threshold = p_param->inet_vm_threshold;
    // const unsigned short pace_max = 300;
    // const unsigned short pace_max = 1000;
    const unsigned short pace_max = p_param->pace_max;
    // const unsigned short celltype = 0.;
    // const unsigned short last_pace_print = 3;
    const unsigned short last_drug_check_pace = 250;
    // const unsigned int print_freq = (1./dt) * dtw;
    // unsigned short pace_count = 0;
    // unsigned short pace_steepest = 0;
    double conc = 99.0; //mmol
    double type = p_param->celltype;
    bool dutta = p_param->is_dutta;
    double epsilon = 10E-14;

    // eligible AP shape means the Vm_peak > 0.
    bool is_eligible_AP;
    // Vm value at 30% repol, 50% repol, and 90% repol, respectively.
    double vm_repol30, vm_repol50, vm_repol90;
    double t_peak_capture = 0.0;

    // qnet_ap/inet_ap values
	  double inet_ap, qnet_ap, inet4_ap, qnet4_ap, inet_cl, qnet_cl, inet4_cl, qnet4_cl;
	  double inal_auc_ap, ical_auc_ap,inal_auc_cl, ical_auc_cl, qinward_cl;

    char buffer[255];

    // static const int CALCIUM_SCALING = 1000000;
	  // static const int CURRENT_SCALING = 1000;

    // printf("Core %d:\n",sample_id);
    initConsts(d_CONSTANTS, d_STATES, type, conc, d_ic50, dutta, sample_id);
    

    applyDrugEffect(d_CONSTANTS, conc, d_ic50, epsilon, sample_id);

    d_CONSTANTS[BCL + (sample_id * num_of_constants)] = bcl;

    // generate file for time-series output

    tmax = pace_max * bcl;
    int pace_count = 0;
    
  
    // printf("%d,%lf,%lf,%lf,%lf\n", sample_id, dt[sample_id], tcurr[sample_id], d_STATES[V + (sample_id * num_of_states)],d_RATES[V + (sample_id * num_of_rates)]);
    // printf("%lf,%lf,%lf,%lf,%lf\n", d_ic50[0 + (14*sample_id)], d_ic50[1+ (14*sample_id)], d_ic50[2+ (14*sample_id)], d_ic50[3+ (14*sample_id)], d_ic50[4+ (14*sample_id)]);

    while (tcurr[sample_id]<tmax){
        
        dt_set = set_time_step( tcurr[sample_id], time_point, max_time_step, 
        d_CONSTANTS, 
        d_RATES, 
        d_STATES, 
        d_ALGEBRAIC, 
        sample_id); 
        
        computeRates(tcurr[sample_id], d_CONSTANTS, d_RATES, d_STATES, d_ALGEBRAIC, sample_id); 
        
        if (floor((tcurr[sample_id] + dt_set) / bcl) == floor(tcurr[sample_id] / bcl)) { 
          dt[sample_id] = dt_set;
        }
        else {
          dt[sample_id] = (floor(tcurr[sample_id] / bcl) + 1) * bcl - tcurr[sample_id];
          pace_count++;
          writen = false;
        }

        //// progress bar starts ////
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
        // //// progress bar ends ////

        solveAnalytical(d_CONSTANTS, d_STATES, d_ALGEBRAIC, d_RATES,  dt[sample_id], sample_id);
        
        // __syncthreads();

        // begin the last 250 pace operations
        if (pace_count >= pace_max-last_drug_check_pace){

			// Find peak vm around 2 msecs and  40 msecs after stimulation
			// and when the sodium current reach 0
      // new codes start here
			if( tcurr[sample_id] > ((d_CONSTANTS[(sample_id * num_of_constants) +BCL]*pace_count)+(d_CONSTANTS[(sample_id * num_of_constants) +stim_start]+2)) && 
				tcurr[sample_id] < ((d_CONSTANTS[(sample_id * num_of_constants) +BCL]*pace_count)+(d_CONSTANTS[(sample_id * num_of_constants) +stim_start]+10)) && 
				abs(d_ALGEBRAIC[(sample_id * num_of_algebraic) +INa]) < 1){
				if( d_STATES[(sample_id * num_of_states) +V] > temp_result->vm_peak ){
					temp_result->vm_peak = d_STATES[(sample_id * num_of_states) +V];
					if(temp_result->vm_peak > 0){
						vm_repol30 = temp_result->vm_peak - (0.3 * (temp_result->vm_peak - temp_result->vm_valley));
						vm_repol50 = temp_result->vm_peak - (0.5 * (temp_result->vm_peak - temp_result->vm_valley));
						vm_repol90 = temp_result->vm_peak - (0.9 * (temp_result->vm_peak - temp_result->vm_valley));
						is_eligible_AP = true;
						t_peak_capture = tcurr[sample_id];
					}
					else is_eligible_AP = false;
				}
			}
			else if( tcurr[sample_id] > ((d_CONSTANTS[(sample_id * num_of_constants) +BCL]*pace_count)+(d_CONSTANTS[(sample_id * num_of_constants) +stim_start]+10)) && is_eligible_AP ){
				if( d_RATES[(sample_id * num_of_rates) +V] > temp_result->dvmdt_repol &&
					d_STATES[(sample_id * num_of_states) +V] <= vm_repol30 &&
					d_STATES[(sample_id * num_of_states) +V] >= vm_repol90 ){
					temp_result->dvmdt_repol = d_RATES[(sample_id * num_of_rates) +V];
				}
				
			}
			
			// calculate AP shape
			if(is_eligible_AP && d_STATES[(sample_id * num_of_states) +V] > vm_repol90){
				// inet_ap/qnet_ap under APD.
				inet_ap = (d_ALGEBRAIC[(sample_id * num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +Ito]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKs]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IK1]);
				inet4_ap = (d_ALGEBRAIC[(sample_id * num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +INa]);
				qnet_ap += (inet_ap * dt[sample_id])/1000.;
				qnet4_ap += (inet4_ap * dt[sample_id])/1000.;
				inal_auc_ap += (d_ALGEBRAIC[(sample_id * num_of_algebraic) +INaL]*dt[sample_id]);
				ical_auc_ap += (d_ALGEBRAIC[(sample_id * num_of_algebraic) +ICaL]*dt[sample_id]);
			}
			// inet_ap/qnet_ap under Cycle Length
			inet_cl = (d_ALGEBRAIC[(sample_id * num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +Ito]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKs]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IK1]);
			inet4_cl = (d_ALGEBRAIC[(sample_id * num_of_algebraic) +INaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +ICaL]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKr]+d_ALGEBRAIC[(sample_id * num_of_algebraic) +INa]);
			qnet_cl += (inet_cl * dt[sample_id])/1000.;
			qnet4_cl += (inet4_cl * dt[sample_id])/1000.;
			inal_auc_cl += (d_ALGEBRAIC[(sample_id * num_of_algebraic) +INaL]*dt[sample_id]);
			ical_auc_cl += (d_ALGEBRAIC[(sample_id * num_of_algebraic) +ICaL]*dt[sample_id]);
			
			
			// save temporary result
			if(pace_count >= pace_max-last_drug_check_pace){
        datapoint_at_this_moment = (int)tcurr[sample_id] - (pace_count * bcl);
				temp_result->cai_data[datapoint_at_this_moment] =  d_STATES[(sample_id * num_of_states) +cai] ;
        temp_result->cai_time[datapoint_at_this_moment] =  tcurr[sample_id];

				temp_result->vm_data[datapoint_at_this_moment] = d_STATES[(sample_id * num_of_states) +V];
        temp_result->vm_time[datapoint_at_this_moment] = tcurr[sample_id];

				temp_result->dvmdt_data[datapoint_at_this_moment] = d_RATES[(sample_id * num_of_rates) +V];
        temp_result->dvmdt_time[datapoint_at_this_moment] = tcurr[sample_id];

        // time series result

        time[input_counter + sample_id] = tcurr[sample_id];
        states[input_counter + sample_id] = d_STATES[V + (sample_id * num_of_states)];
        
        out_dt[input_counter + sample_id] = dt[sample_id];
        
        cai_result[input_counter + sample_id] = d_ALGEBRAIC[cai + (sample_id * num_of_algebraic)];

        ina[input_counter + sample_id] = d_ALGEBRAIC[INa + (sample_id * num_of_algebraic)] ;
        inal[input_counter + sample_id] = d_ALGEBRAIC[INaL + (sample_id * num_of_algebraic)] ;

        ical[input_counter + sample_id] = d_ALGEBRAIC[ICaL + (sample_id * num_of_algebraic)] ;
        ito[input_counter + sample_id] = d_ALGEBRAIC[Ito + (sample_id * num_of_algebraic)] ;

        ikr[input_counter + sample_id] = d_ALGEBRAIC[IKr + (sample_id * num_of_algebraic)] ;
        iks[input_counter + sample_id] = d_ALGEBRAIC[IKs + (sample_id * num_of_algebraic)] ;

        ik1[input_counter + sample_id] = d_ALGEBRAIC[IK1 + (sample_id * num_of_algebraic)] ;

        input_counter = input_counter + sample_size;

        //time series ends
			
				// snprintf( buffer, sizeof(buffer), "%.2lf,%.2lf,%.0lf,%.0lf,%.0lf,%.0lf,%0.lf,%.0lf,%.0lf,%.0lf",
				// 		d_STATES[(sample_id * num_of_states) +V], d_RATES[(sample_id * num_of_rates) +V], d_STATES[(sample_id * num_of_states) +cai]*CALCIUM_SCALING,
				// 		d_ALGEBRAIC[(sample_id * num_of_algebraic) +INa]*CURRENT_SCALING, d_ALGEBRAIC[(sample_id * num_of_algebraic) +INaL]*CURRENT_SCALING, 
				// 		d_ALGEBRAIC[(sample_id * num_of_algebraic) +ICaL]*CURRENT_SCALING, d_ALGEBRAIC[(sample_id * num_of_algebraic) +Ito]*CURRENT_SCALING,
				// 		d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKr]*CURRENT_SCALING, d_ALGEBRAIC[(sample_id * num_of_algebraic) +IKs]*CURRENT_SCALING, 
				// 		d_ALGEBRAIC[(sample_id * num_of_algebraic) +IK1]*CURRENT_SCALING);
				// temp_result.time_series_data.insert( std::pair<double, string> (tcurr[sample_id], string(buffer)) );
			}
          // new code ends here (last 250 pace operation)
          tcurr[sample_id] = tcurr[sample_id] + dt[sample_id];

         //temporary writing method
        // if (pace_count > pace_max-2){

        // time[input_counter + sample_id] = tcurr[sample_id];
        // states[input_counter + sample_id] = d_STATES[V + (sample_id * num_of_states)];
        
        // out_dt[input_counter + sample_id] = dt[sample_id];
        
        // cai_result[input_counter + sample_id] = d_ALGEBRAIC[cai + (sample_id * num_of_algebraic)];

        // ina[input_counter + sample_id] = d_ALGEBRAIC[INa + (sample_id * num_of_algebraic)] ;
        // inal[input_counter + sample_id] = d_ALGEBRAIC[INaL + (sample_id * num_of_algebraic)] ;

        // ical[input_counter + sample_id] = d_ALGEBRAIC[ICaL + (sample_id * num_of_algebraic)] ;
        // ito[input_counter + sample_id] = d_ALGEBRAIC[Ito + (sample_id * num_of_algebraic)] ;

        // ikr[input_counter + sample_id] = d_ALGEBRAIC[IKr + (sample_id * num_of_algebraic)] ;
        // iks[input_counter + sample_id] = d_ALGEBRAIC[IKs + (sample_id * num_of_algebraic)] ;

        // ik1[input_counter + sample_id] = d_ALGEBRAIC[IK1 + (sample_id * num_of_algebraic)] ;

        // input_counter = input_counter + sample_size;
        
        // } // temporary guard ends here

		} // end the last 250 pace operations
    

       
      } // while loop ends here 
    // __syncthreads();
}



__global__ void kernel_DrugSimulation(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, double *d_ALGEBRAIC, 
                                      double *time, double *states, double *out_dt,  double *cai_result, 
                                      double *ina, double *inal, 
                                      double *ical, double *ito,
                                      double *ikr, double *iks,
                                      double *ik1,
                                      unsigned int sample_size,
                                      cipa_t *temp_result,
                                      param_t *p_param
                                      )
  {
    unsigned short thread_id;
    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    double time_for_each_sample[2000];
    double dt_for_each_sample[2000];
    // printf("in\n");
    
    // printf("Calculating %d\n",thread_id);
    kernel_DoDrugSim(d_ic50, d_CONSTANTS, d_STATES, d_RATES, d_ALGEBRAIC, 
                          time, states, out_dt, cai_result,
                          ina, inal, 
                          ical, ito,
                          ikr, iks, 
                          ik1,
                          time_for_each_sample, dt_for_each_sample, thread_id, sample_size,
                          temp_result,
                          p_param
                          );
                          // __syncthreads();
    // printf("Calculation for core %d done\n",sample_id);
  }