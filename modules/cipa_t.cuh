#ifndef CIPA_T_HPP
#define CIPA_T_HPP

#include <map>
#include <string>

#include <cuda_runtime.h>

// using std::multimap;
// using std::string;
// case 1, what if we limit the datapoints to just 7000 as usual?
__global__ struct cipa_t{
   double qnet;
  double inal_auc;
  double ical_auc;
  double dvmdt_repol;
  double dvmdt_max;
  double vm_peak;
  double vm_valley;
  double vm_dia;
  double apd90;
  double apd50;
  double apd_tri;
  double ca_peak;
  double ca_valley;
  double ca_dia;
  double cad90;
  double cad50;
  double cad_tri;
  // multimap<double, double> vm_data;
  // multimap<double, double> dvmdt_data;
  // multimap<double, double> cai_data;
  // multimap<double, string> ires_data;
  
  // multimap<double, string> inet_data;
  // multimap<double, string> qnet_data;
  // multimap<double, string> inet4_data;
  // multimap<double, string> qnet4_data;
  
  // multimap<double, string> time_series_data;

  // temporary fix for this
  double vm_data[7500];
  double vm_time[7500];

  double dvmdt_data[7500];
  double dvmdt_time[7500];

  double cai_data[7500];
  double cai_time[7500];

  double ires_data[7500];
  double ires_time[7500];

  double inet_data[7500];
  double inet_time[7500];

  double qnet_data[7500];
  double qnet_time[7500];

  // double inet4_data[7500];
  // double inet4_time[7500];
  
  // double qnet4_data[7500];
  // double qnet4_time[7500];

  // double time_series_data[7000];
  // double time_series_time[7000];


  
  // __device__ cipa_t();
  // __device__ cipa_t( const cipa_t &source );
  // cipa_t& operator=(const cipa_t & source);
  // __device__ void copy(const cipa_t &source);
  // __device__ void init(const double vm_val);
  // __device__ void clear_time_result();


};


#endif
