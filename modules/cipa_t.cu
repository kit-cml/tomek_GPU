#include "cipa_t.cuh"
#include <cuda_runtime.h>

//  cipa_t::cipa_t()
// {

// }

//  cipa_t::cipa_t( const cipa_t &source )
// {
//   copy(source);
// }


// //  cipa_t& cipa_t::operator=(const cipa_t & source)
// // {
// //   if( this != &source ) copy(source);

// //   return *this;
// // }

// __device__ void cipa_t::copy(const cipa_t &source)
// {
//   qnet_ap = source.qnet_ap;
//   qnet4_ap = source.qnet4_ap;
//   inal_auc_ap = source.inal_auc_ap;
//   ical_auc_ap = source.ical_auc_ap;
//   qnet_cl = source.qnet_cl;
//   qnet4_cl = source.qnet4_cl;
//   inal_auc_cl = source.inal_auc_cl;
//   ical_auc_cl = source.ical_auc_cl;
  
//   dvmdt_repol = source.dvmdt_repol;
//   vm_peak = source.vm_peak;
//   vm_valley = source.vm_valley;
  
//   vm_data.clear();
//   cai_data.clear();
//   ires_data.clear();
//   dvmdt_data.clear();
//   inet_data.clear();
//   qnet_data.clear();
//   inet4_data.clear();
//   qnet4_data.clear();
  
//   time_series_data.clear();

//   vm_data.insert( (source.vm_data).begin(), (source.vm_data).end() );
//   dvmdt_data.insert( (source.dvmdt_data).begin(), (source.dvmdt_data).end() );
//   cai_data.insert( (source.cai_data).begin(), (source.cai_data).end() );
//   ires_data.insert( (source.ires_data).begin(), (source.ires_data).end() );
  
//   inet_data.insert( (source.inet_data).begin(), (source.inet_data).end() );
//   qnet_data.insert( (source.qnet_data).begin(), (source.qnet_data).end() );
//   inet4_data.insert( (source.inet4_data).begin(), (source.inet4_data).end() );
//   qnet4_data.insert( (source.qnet4_data).begin(), (source.qnet4_data).end() );
  
//   time_series_data.insert( (source.time_series_data).begin(), (source.time_series_data).end() );
// }

// __device__ void cipa_t::init(const double vm_val)
// {
//   qnet_ap = 0.;
//   qnet4_ap = 0.;
//   inal_auc_ap = 0.;
//   ical_auc_ap = 0.;
  
//   qnet_cl = 0.;
//   qnet4_cl = 0.;
//   inal_auc_cl = 0.;
//   ical_auc_cl = 0.;
  
//   dvmdt_repol = -999;
//   vm_peak = -999;
//   vm_valley = vm_val;
  
//   vm_data.clear();
//   dvmdt_data.clear();
//   cai_data.clear();
//   ires_data.clear();
  
//   inet_data.clear();
//   qnet_data.clear();
//   inet4_data.clear();
//   qnet4_data.clear();
  
//   time_series_data.clear();
// }
