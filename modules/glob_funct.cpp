#include "glob_funct.hpp"
// #include "../libs/zip.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// to make it more "portable" between OSes.
#if defined _WIN32
  #include <direct.h>
  #define snprintf _snprintf
  #define vsnprintf _vsnprintf
  #define strcasecmp _stricmp
  #define strncasecmp _strnicmp
#else
  #include <dirent.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


void mpi_printf(unsigned short node_id, const char *fmt, ...)
{

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);	
}


void edison_assign_params(int argc, char *argv[], param_t *p_param)
{
  bool is_default;
  char buffer[100];
  char key[100];
  char value[100];
  char file_name[150];
  FILE *fp_inputdeck;

  // parameters from arguments
  for (int idx = 1; idx < argc; idx += 2) {
    if (!strcasecmp(argv[idx], "-input_deck"))
      strncpy(file_name, argv[idx + 1], sizeof(file_name));
    else if (!strcasecmp(argv[idx], "-hill_file"))
      strncpy(p_param->hill_file, argv[idx + 1], sizeof(p_param->hill_file));
    else if (!strcasecmp(argv[idx], "-init_file"))
      strncpy(p_param->cache_file, argv[idx + 1], sizeof(p_param->cache_file));
    else if (!strcasecmp(argv[idx], "-cvar_file"))
      strncpy(p_param->cvar_file, argv[idx + 1], sizeof(p_param->cvar_file));
  }  

  is_default = false;
  fp_inputdeck = fopen( file_name, "r");
  if(fp_inputdeck == NULL){
    fprintf(stderr, "Cannot open input deck file %s!!!\nUse default value as the failsafe.\n", file_name);
    is_default = true;
  }

  // read input_deck line by line
  // and store each line to the buffer
  while ( is_default == false && fgets( buffer, 100, fp_inputdeck ) != NULL ) {
    sscanf( buffer, "%s %*s %s", key, value );
    if (strcasecmp(key, "Simulation_Mode") == 0) {
      p_param->simulation_mode = strtod( value, NULL );
    }
    else if (strcasecmp(key, "Celltype") == 0) {
      p_param->celltype = strtod( value, NULL );
    }
    else if (strcasecmp(key, "Is_Post_Processing") == 0) {
      p_param->is_time_series = strtod( value, NULL );
    }
    else if (strcasecmp(key, "Is_Dutta") == 0) {
      p_param->is_dutta = strtol( value, NULL, 10 );
    }
    else if (strcasecmp(key, "Use_Conductance_Variability") == 0) {
      p_param->is_cvar = strtol( value, NULL, 10 );
    }
    else if (strcasecmp(key, "Pace_Find_Steepest") == 0) {
      p_param->find_steepest_start = strtol( value, NULL, 10);
    }
    else if (strcasecmp(key, "GPU_Index") == 0) {
      p_param->gpu_index = strtod( value, NULL);
    }
    else if (strcasecmp(key, "Basic_Cycle_Length") == 0) {
      p_param->bcl = strtod( value, NULL );
    }
    else if (strcasecmp(key, "Number_of_Pacing") == 0) {
      p_param->pace_max = strtol( value, NULL, 10 );
    }
    else if (strcasecmp(key, "Time_Step") == 0) {
      p_param->dt = strtod( value, NULL );
    }
    else if (strcasecmp(key, "Drug_Name") == 0) {
      strncpy( p_param->drug_name, value, sizeof(p_param->concs) );
    }
    else if (strcasecmp(key, "Concentrations") == 0) {
      p_param->conc = strtod( value, NULL);
    }

  }

  if( is_default == false ) fclose( fp_inputdeck );
}


