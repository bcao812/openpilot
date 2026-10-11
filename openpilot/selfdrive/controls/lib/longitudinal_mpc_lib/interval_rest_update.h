#ifndef LONG_INTERVAL_REST_UPDATE_H
#define LONG_INTERVAL_REST_UPDATE_H

#include "acados_solver_long.h"

int custom_update_init_function(long_solver_capsule *capsule);
int custom_update_function(long_solver_capsule *capsule, double *data, int data_len);
int custom_update_terminate_function(long_solver_capsule *capsule);

#endif
