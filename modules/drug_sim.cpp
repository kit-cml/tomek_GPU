#include "drug_sim.hpp"

#ifdef TOMEK_2019
#include "../cellmodels/Tomek_model.hpp"
#else
#include "../cellmodels/Ohara_Rudy_2011.hpp"
#endif

#include "glob_funct.hpp"

#include <cmath>
#include <cstdlib>

bool do_drug_sim(const double conc, row_data ic50, 
const param_t* p_param, const unsigned short sample_id, const unsigned short group_id, qinward_t *p_qin)
{
	bool is_ead = false;
	unsigned short idx;
	
	// to normalize the small values
	// so it can compressed the file size.
	// Information of the scaling should be
	// described in the header result.
	static const int CALCIUM_SCALING = 1000000;
	static const int CURRENT_SCALING = 1000;

	// cell object pointer
	Cellmodel* p_cell;

	// qnet_ap/inet_ap values
	double inet_ap, qnet_ap, inet4_ap, qnet4_ap, inet_cl, qnet_cl, inet4_cl, qnet4_cl;
	double inal_auc_ap, ical_auc_ap,inal_auc_cl, ical_auc_cl, qinward_cl;

	// variables for I/O
	char buffer[255];
	FILE* fp_dvmdt;
	FILE* fp_gate;
	FILE* fp_vm;
	FILE* fp_ires;
	FILE* fp_concs;
	FILE* fp_output;
	FILE* fp_inet;
	FILE* fp_qnet;
	FILE* fp_inet4;
	FILE* fp_qnet4;
	FILE* fp_qni;
	FILE* fp_vmdebug;
	
	FILE* fp_time_series;
	
	// simulation parameters
#ifdef DEBUG_MODE
	bool is_print_graph = true;
	bool is_dutta = false;
	const char *drug_name = "bepridil";
	const double bcl = 2000.;
	const double inet_vm_threshold = -88.0;
	const unsigned short pace_max = 1000;
	const unsigned short celltype = 0.;
	const unsigned short last_drug_check_pace = 250;
	const unsigned int print_freq = (1./dt) * dtw;
	unsigned short pace_count = 0;
	unsigned short pace_steepest = 0;
#else
	bool is_print_graph = p_param->is_print_graph;
	bool is_dutta = p_param->is_dutta;
	const char *drug_name = p_param->drug_name;
	const double bcl = p_param->bcl;
	const double inet_vm_threshold = p_param->inet_vm_threshold;
	const unsigned short pace_max = p_param->pace_max;
	const unsigned short celltype = p_param->celltype;
	const unsigned short last_drug_check_pace = 250;
	unsigned short pace_count = 0;
	unsigned short pace_steepest = 0;
#endif

	// variables to store features
	// temp_result is the result of features in 1 pace,
	// will be interchanged during the simulation.
	// cipa_result is the final result of the simulation.
	cipa_t cipa_result, temp_result;
	// eligible AP shape means the Vm_peak > 0.
	bool is_eligible_AP;
	// Vm value at 30% repol, 50% repol, and 90% repol, respectively.
	double vm_repol30, vm_repol50, vm_repol90;

	// // these values are from the supplementary materials of ORd2011
	double dt = 0.005;
	double max_time_step = 1.0;
	double time_point = 25.0;
	double tcurr = 0.0;
	double tmax = pace_max*bcl;
	double t_peak_capture = 0.0;
	double dt_set;

#ifdef TOMEK_2019
	printf("Using Tomek cell model\n");
	p_cell = new Tomek_model();
	p_cell->initConsts((double)celltype, conc, ic50.data);
#else
	printf("Using O'Hara Rudy cell model\n");
	p_cell = new Ohara_Rudy_2011();
	p_cell->initConsts((double)celltype, conc, ic50.data, is_dutta);
#endif
	p_cell->CONSTANTS[BCL] = bcl;

	FILE *fp_states;
	if( p_param->is_using_output > 0 ){
		fp_states = fopen("output_orudy.dat", "r");
		if( fp_states != NULL ){
			mpi_printf(0, "Using initial condition from steady state!\n");
			int idx = 0;
			while(fgets( buffer, sizeof(buffer), fp_states) != NULL){
				p_cell->STATES[idx++] = strtod(buffer, NULL);
			}
		}
		else{
			mpi_printf(0, "No initial file found! Skipped using initial file!\n");
		}
	}

#ifdef DEBUG_WRITE
	// generate file for time-series output
	snprintf(buffer, sizeof(buffer), "result/%.4lf/%s_%.4lf_vmcheck_smp%hu.plt", 
			conc, drug_name, conc, sample_id );
	fp_vm = fopen( buffer, "w" );
	snprintf(buffer, sizeof(buffer), "result/%.4lf/%s_%.4lf_dvmdt_smp%hu.plt", 
			conc, drug_name, conc, sample_id );
	fp_dvmdt = fopen( buffer, "w" );
	snprintf(buffer, sizeof(buffer), "result/%.4lf/%s_%.4lf_gates_smp%hu.plt",
			conc, drug_name, conc, sample_id );
	fp_gate = fopen(buffer, "w");
	snprintf(buffer, sizeof(buffer), "result/%.4lf/%s_%.4lf_ires_smp%hu.plt",
			conc, drug_name, conc, sample_id );
	fp_ires = fopen(buffer, "w");
	snprintf(buffer, sizeof(buffer), "result/%.4lf/%s_%.4lf_conc_smp%hu.plt",
			conc, drug_name, conc, sample_id );
	fp_concs = fopen(buffer, "w");
	snprintf(buffer, sizeof(buffer), "result/%.4lf/%s_%.4lf_inet_smp%hu.plt", 
			  conc, drug_name, conc, sample_id );
	fp_inet = fopen( buffer, "w" );	
	snprintf(buffer, sizeof(buffer), "result/%.4lf/%s_%.4lf_qnet_smp%hu.plt", 
			  conc, drug_name, conc, sample_id );
	fp_qnet = fopen( buffer, "w" );
	snprintf(buffer, sizeof(buffer), "result/%.4lf/%s_%.4lf_inet4_smp%hu.plt", 
			  conc, drug_name, conc, sample_id );
	fp_inet4 = fopen( buffer, "w" );	
	snprintf(buffer, sizeof(buffer), "result/%.4lf/%s_%.4lf_qnet4_smp%hu.plt", 
			  conc, drug_name, conc, sample_id );
	fp_qnet4 = fopen( buffer, "w" );
#endif
	
	snprintf(buffer, sizeof(buffer), "result/%.4lf/%s_%.4lf_vmdebug_smp%hu.plt", 
			  conc, drug_name, conc, sample_id );
	fp_vmdebug = fopen( buffer, "w" );
	snprintf(buffer, sizeof(buffer), "result/%.4lf/%s_%.4lf_qni_proc%hu.plt", 
			  conc, drug_name, conc, mympi::rank );
	fp_qni = fopen( buffer, "a" );
	
	snprintf(buffer, sizeof(buffer), "result/%.4lf/%s_%.4lf_time_series_result_smp%hu.plt", 
			  conc, drug_name, conc, sample_id );
	fp_time_series = fopen( buffer, "w" );	

#ifdef DEBUG_WRITE
	fprintf(fp_vm, "%s,%s\n", "Time", "Vm");
	fprintf(fp_dvmdt, "%s,%s\n", "Time", "dVm/dt");
	//fprintf(fp_gate, "Time,%s\n", p_cell->gates_header);
	fprintf(fp_ires, "%s,%s,%s,%s,%s,%s,%s\n","Time","INa","IKr","IKs","IK1","Ito","ICaL");
	fprintf(fp_concs, "Time,cai\n");
	//fprintf(fp_concs, "Time,%s\n", p_cell->concs_header);
	fprintf(fp_inet, "%s,%s,%s\n", "Time", "Inet_AP","Inet_CL");
	fprintf(fp_qnet, "%s,%s,%s\n", "Time", "Qnet_AP","Qnet_CL");
	fprintf(fp_inet4, "%s,%s,%s\n", "Time", "Inet4_AP","Inet4_CL");
	fprintf(fp_qnet4, "%s,%s,%s\n", "Time", "Qnet4_AP","Qnet_CL");
#endif
	if(group_id == 0){
		fprintf( fp_qni, "%s,%s,%s,%s,%s,%s,%s\n","Sample_ID", "Qnet_AP", "Qnet4_AP", "Qinward_CL","Qnet_CL", "Qnet4_CL", "Qinward_CL");
	}

	fprintf(fp_vmdebug, "%s,%s,%s,%s,%s,%s,%s\n", "Pace","T_Peak", "Vmpeak","Vm_repol30","Vm_repol50","Vm_repol90","dVmdt_repol");
	fprintf(fp_time_series,"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
			"Time","Vm","dVm/dt","Cai(x1.000.000)(milliM->kiloM)",
			"INa(x1.000)(microA->milliA)","INaL(x1.000)(microA->milliA)","ICaL(x1.000)(microA->milliA)",
			"IKs(x1.000)(microA->milliA)","IKr(x1.000)(microA->milliA)","IK1(x1.000)(microA->milliA)",
			"Ito(x1.000)(microA->milliA)");
	
	inet_ap = 0.;
	qnet_ap = 0.;
	inet4_ap = 0.;
	qnet4_ap = 0.;
	inal_auc_ap = 0.;
	ical_auc_ap = 0.;
	qinward_cl = 0.;
	inet_cl = 0.;
	qnet_cl = 0.;
	inet4_cl = 0.;
	qnet4_cl = 0.;
	inal_auc_cl = 0.;
	ical_auc_cl = 0.;
	qinward_cl = 0.;
	pace_count = 0;
	
	
	cipa_result.init( p_cell->STATES[V]);
	temp_result.init( p_cell->STATES[V]);	
	
	while(tcurr < tmax)
	{
		// compute ODE at tcurr
		p_cell->computeRates(tcurr,
					p_cell->CONSTANTS,
					p_cell->RATES,
					p_cell->STATES,
					p_cell->ALGEBRAIC);
		dt_set = Ohara_Rudy_2011::set_time_step(tcurr,
						time_point,
						max_time_step,
						p_cell->CONSTANTS,
						p_cell->RATES,
						p_cell->STATES,
						p_cell->ALGEBRAIC);
		// compute accepted timestep
		if (floor((tcurr + dt_set) / bcl) == floor(tcurr / bcl)) {
		  dt = dt_set;
		}
		// new cycle length code.
		// this is the place for comparing current and previous paces.
		// if the AP shape is eligible and dvmdt_repol is bigger,
		// that AP shape become the resultant pace.
		// also for re-initializing stuffs.
		else {
			dt = (floor(tcurr / bcl) + 1) * bcl - tcurr;
			if( is_eligible_AP && pace_count >= pace_max-last_drug_check_pace) {
				temp_result.qnet_ap = qnet_ap;
				temp_result.qnet4_ap = qnet4_ap;
				temp_result.inal_auc_ap = inal_auc_ap;
				temp_result.ical_auc_ap = ical_auc_ap;
				temp_result.qnet_cl = qnet_cl;
				temp_result.qnet4_cl = qnet4_cl;
				temp_result.inal_auc_cl = inal_auc_cl;
				temp_result.ical_auc_cl = ical_auc_cl;
				fprintf(fp_vmdebug, "%hu,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf,%.2lf\n", pace_count,t_peak_capture,temp_result.vm_peak,vm_repol30,vm_repol50,vm_repol90,temp_result.dvmdt_repol);
				// replace result with steeper repolarization AP or first pace from the last 250 paces
				if( temp_result.dvmdt_repol > cipa_result.dvmdt_repol ) {
					pace_steepest = pace_count;
					cipa_result = temp_result;
				}
			};
			inet_ap = 0.;
			qnet_ap = 0.;
			inet4_ap = 0.;
			qnet4_ap = 0.;
			inal_auc_ap = 0.;
			ical_auc_ap = 0.;
			inet_cl = 0.;
			qnet_cl = 0.;
			inet4_cl = 0.;
			qnet4_cl = 0.;
			inal_auc_cl = 0.;
			ical_auc_cl = 0.;
			t_peak_capture = 0.;
			temp_result.init( p_cell->STATES[V]);	
			pace_count++;
			is_eligible_AP = false;
		}

		//Compute the analytical solution
		p_cell->solveAnalytical(dt);
		
		
		// begin the last 250 pace operations
		if(pace_count >= pace_max-last_drug_check_pace){
		
			// Find peak vm around 2 msecs and  40 msecs after stimulation
			// and when the sodium current reach 0
			if( tcurr > ((p_cell->CONSTANTS[BCL]*pace_count)+(p_cell->CONSTANTS[stim_start]+2)) && 
				tcurr < ((p_cell->CONSTANTS[BCL]*pace_count)+(p_cell->CONSTANTS[stim_start]+10)) && 
				abs(p_cell->ALGEBRAIC[INa]) < 1){
				if( p_cell->STATES[V] > temp_result.vm_peak ){
					temp_result.vm_peak = p_cell->STATES[V];
					if(temp_result.vm_peak > 0){
						vm_repol30 = temp_result.vm_peak - (0.3 * (temp_result.vm_peak - temp_result.vm_valley));
						vm_repol50 = temp_result.vm_peak - (0.5 * (temp_result.vm_peak - temp_result.vm_valley));
						vm_repol90 = temp_result.vm_peak - (0.9 * (temp_result.vm_peak - temp_result.vm_valley));
						is_eligible_AP = true;
						t_peak_capture = tcurr;
					}
					else is_eligible_AP = false;
				}
			}
			else if( tcurr > ((p_cell->CONSTANTS[BCL]*pace_count)+(p_cell->CONSTANTS[stim_start]+10)) && is_eligible_AP ){
				if( p_cell->RATES[V] > temp_result.dvmdt_repol &&
					p_cell->STATES[V] <= vm_repol30 &&
					p_cell->STATES[V] >= vm_repol90 ){
					temp_result.dvmdt_repol = p_cell->RATES[V];
				}
				
			}
			
			// calculate AP shape
			if(is_eligible_AP && p_cell->STATES[V] > vm_repol90){
				// inet_ap/qnet_ap under APD.
				inet_ap = (p_cell->ALGEBRAIC[INaL]+p_cell->ALGEBRAIC[ICaL]+p_cell->ALGEBRAIC[Ito]+p_cell->ALGEBRAIC[IKr]+p_cell->ALGEBRAIC[IKs]+p_cell->ALGEBRAIC[IK1]);
				inet4_ap = (p_cell->ALGEBRAIC[INaL]+p_cell->ALGEBRAIC[ICaL]+p_cell->ALGEBRAIC[IKr]+p_cell->ALGEBRAIC[INa]);
				qnet_ap += (inet_ap * dt)/1000.;
				qnet4_ap += (inet4_ap * dt)/1000.;
				inal_auc_ap += (p_cell->ALGEBRAIC[INaL]*dt);
				ical_auc_ap += (p_cell->ALGEBRAIC[ICaL]*dt);
			}
			// inet_ap/qnet_ap under Cycle Length
			inet_cl = (p_cell->ALGEBRAIC[INaL]+p_cell->ALGEBRAIC[ICaL]+p_cell->ALGEBRAIC[Ito]+p_cell->ALGEBRAIC[IKr]+p_cell->ALGEBRAIC[IKs]+p_cell->ALGEBRAIC[IK1]);
			inet4_cl = (p_cell->ALGEBRAIC[INaL]+p_cell->ALGEBRAIC[ICaL]+p_cell->ALGEBRAIC[IKr]+p_cell->ALGEBRAIC[INa]);
			qnet_cl += (inet_cl * dt)/1000.;
			qnet4_cl += (inet4_cl * dt)/1000.;
			inal_auc_cl += (p_cell->ALGEBRAIC[INaL]*dt);
			ical_auc_cl += (p_cell->ALGEBRAIC[ICaL]*dt);
			
			
			// save temporary result
			if(pace_count >= pace_max-last_drug_check_pace){
				temp_result.cai_data.insert( std::pair<double, double> (tcurr, p_cell->STATES[cai]) );
				temp_result.vm_data.insert( std::pair<double, double> (tcurr, p_cell->STATES[V]) );
				temp_result.dvmdt_data.insert( std::pair<double, double> (tcurr, p_cell->RATES[V]) );
#ifdef DEBUG_WRITE				
				snprintf( buffer, sizeof(buffer), "%lf,%lf",inet_ap,inet_cl);
				temp_result.inet_data.insert( std::pair<double, string> (tcurr, buffer) );
				snprintf( buffer, sizeof(buffer), "%lf,%lf",qnet_ap,qnet_cl);
				temp_result.qnet_data.insert( std::pair<double, string> (tcurr, buffer) );
				snprintf( buffer, sizeof(buffer), "%lf,%lf",inet4_ap,inet4_cl);
				temp_result.inet4_data.insert( std::pair<double, string> (tcurr, buffer) );
				snprintf( buffer, sizeof(buffer), "%lf,%lf",qnet4_ap,qnet4_cl);
				temp_result.qnet4_data.insert( std::pair<double, string> (tcurr, buffer) );
				snprintf( buffer, sizeof(buffer), "%lf,%lf,%lf,%lf,%lf,%lf,%lf",
					  p_cell->ALGEBRAIC[INa], p_cell->ALGEBRAIC[INaL], p_cell->ALGEBRAIC[ICaL], p_cell->ALGEBRAIC[Ito],
					  p_cell->ALGEBRAIC[IKr], p_cell->ALGEBRAIC[IKs], p_cell->ALGEBRAIC[IK1] );
				temp_result.ires_data.insert( std::pair<double, string> (tcurr, string(buffer)) );
#endif				
				snprintf( buffer, sizeof(buffer), "%.2lf,%.2lf,%.0lf,%.0lf,%.0lf,%.0lf,%0.lf,%.0lf,%.0lf,%.0lf",
						p_cell->STATES[V], p_cell->RATES[V], p_cell->STATES[cai]*CALCIUM_SCALING,
						p_cell->ALGEBRAIC[INa]*CURRENT_SCALING, p_cell->ALGEBRAIC[INaL]*CURRENT_SCALING, 
						p_cell->ALGEBRAIC[ICaL]*CURRENT_SCALING, p_cell->ALGEBRAIC[Ito]*CURRENT_SCALING,
						p_cell->ALGEBRAIC[IKr]*CURRENT_SCALING, p_cell->ALGEBRAIC[IKs]*CURRENT_SCALING, 
						p_cell->ALGEBRAIC[IK1]*CURRENT_SCALING);
				temp_result.time_series_data.insert( std::pair<double, string> (tcurr, string(buffer)) );
				 
				
			}

		} // end the last 250 pace operations
		
		tcurr += dt;
	}

	if( cipa_result.dvmdt_repol > 0 ) is_ead = true;
	
	// print result from the map to the file.
	// first_itrmap is used to normalize the time unit.
	std::multimap<double, double>::iterator first_itrmap = cipa_result.cai_data.begin();
	std::multimap<double, string>::iterator first_itrmap_str = cipa_result.ires_data.begin();
#ifdef DEBUG_WRITE
	for(std::multimap<double, double>::iterator itrmap = cipa_result.cai_data.begin(); itrmap != cipa_result.cai_data.end() ; itrmap++ ){
		fprintf(fp_concs, "%lf,%lf\n", itrmap->first-first_itrmap->first, itrmap->second);
	}
	first_itrmap = cipa_result.vm_data.begin();
	for(std::multimap<double, double>::iterator itrmap = cipa_result.vm_data.begin(); itrmap != cipa_result.vm_data.end() ; itrmap++ ){
		fprintf(fp_vm, "%lf,%lf\n", itrmap->first-first_itrmap->first, itrmap->second);
	}
	first_itrmap = cipa_result.dvmdt_data.begin();
	for(std::multimap<double, double>::iterator itrmap = cipa_result.dvmdt_data.begin(); itrmap != cipa_result.dvmdt_data.end() ; itrmap++ ){
		fprintf(fp_dvmdt, "%lf,%lf\n", itrmap->first-first_itrmap->first, itrmap->second);
	}
	
	
	for(std::multimap<double, string>::iterator itrmap = cipa_result.ires_data.begin(); itrmap != cipa_result.ires_data.end() ; itrmap++ ){
		fprintf(fp_ires, "%lf,%s\n", itrmap->first-first_itrmap_str->first, (itrmap->second).c_str());
	}
	first_itrmap_str = cipa_result.inet_data.begin();
	for(std::multimap<double, string>::iterator itrmap = cipa_result.inet_data.begin(); itrmap != cipa_result.inet_data.end() ; itrmap++ ){
		fprintf(fp_inet, "%lf,%s\n", itrmap->first-first_itrmap_str->first, (itrmap->second).c_str());
	}
	first_itrmap_str = cipa_result.qnet_data.begin();
	for(std::multimap<double, string>::iterator itrmap = cipa_result.qnet_data.begin(); itrmap != cipa_result.qnet_data.end() ; itrmap++ ){
		fprintf(fp_qnet, "%lf,%s\n", itrmap->first-first_itrmap_str->first, (itrmap->second).c_str());
	}
	first_itrmap_str = cipa_result.inet4_data.begin();
	for(std::multimap<double, string>::iterator itrmap = cipa_result.inet4_data.begin(); itrmap != cipa_result.inet4_data.end() ; itrmap++ ){
		fprintf(fp_inet4, "%lf,%s\n", itrmap->first-first_itrmap_str->first, (itrmap->second).c_str());
	}
	first_itrmap_str = cipa_result.qnet4_data.begin();
	for(std::multimap<double, string>::iterator itrmap = cipa_result.qnet4_data.begin(); itrmap != cipa_result.qnet4_data.end() ; itrmap++ ){
		fprintf(fp_qnet4, "%lf,%s\n", itrmap->first-first_itrmap_str->first, (itrmap->second).c_str());
	}
#endif
	first_itrmap_str = cipa_result.time_series_data.begin();
	for(std::multimap<double, string>::iterator itrmap = cipa_result.time_series_data.begin(); itrmap != cipa_result.time_series_data.end() ; itrmap++ ){
		fprintf(fp_time_series, "%.4lf,%s\n", itrmap->first-first_itrmap_str->first, (itrmap->second).c_str());
	}

	fprintf(fp_vmdebug,"Steepest pace: %hu\n", pace_steepest);
	
	// for qnet_ap and qinward_cl output
    if( (int)ceil(conc) == 0 ) {
      p_qin->inal_auc_control = cipa_result.inal_auc_cl;
      p_qin->ical_auc_control = cipa_result.ical_auc_cl;
      fprintf( fp_qni, "%hu,%lf,%lf,%lf,%lf,%lf,%lf\n", sample_id, cipa_result.qnet_ap, cipa_result.qnet4_ap, 0.0, cipa_result.qnet_cl, cipa_result.qnet4_cl, 0.0);
    }
    else{
      p_qin->inal_auc_drug = cipa_result.inal_auc_cl;
      p_qin->ical_auc_drug = cipa_result.ical_auc_cl;
      qinward_cl =  ( (p_qin->inal_auc_drug/p_qin->inal_auc_control) + (p_qin->ical_auc_drug/p_qin->ical_auc_control) ) * 0.5;
      fprintf( fp_qni, "%hu,%lf,%lf,%lf,%lf,%lf,%lf\n", sample_id, cipa_result.qnet_ap, cipa_result.qnet4_ap, qinward_cl, cipa_result.qnet_cl, cipa_result.qnet4_cl, qinward_cl );
    }


	// clean the memories
	fclose(fp_time_series);
	fclose(fp_vmdebug);
	fclose(fp_qni);

#ifdef DEBUG_WRITE
	fclose(fp_vm);
	fclose(fp_ires);
	fclose(fp_concs);
	fclose(fp_dvmdt);
	fclose(fp_qnet);
	fclose(fp_inet);
	fclose(fp_qnet4);
	fclose(fp_inet4);
	fclose(fp_gate);
#endif
	delete p_cell;


	return is_ead;
}