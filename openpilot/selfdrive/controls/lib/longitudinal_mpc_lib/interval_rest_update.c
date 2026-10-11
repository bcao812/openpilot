#include <math.h>
#include <stddef.h>

#include "interval_rest_update.h"

int custom_update_init_function(long_solver_capsule *capsule) {
  (void)capsule;
  return 0;
}

int custom_update_function(long_solver_capsule *capsule, double *data, int data_len) {
  // Protocol 1 changes only the first motion integration duration. The Python
  // caller admits a shorter duration only with a verified rest event and hard
  // zero v/a thereafter. Public observation times and other intervals stay put:
  // the omitted part of interval zero is an explicit stationary hold.
  const double first_interval = 10.0 / (LONG_N * LONG_N);
  if (capsule == NULL || data == NULL || data_len != 2 || data[0] != 1.0 ||
      !isfinite(data[1]) || data[1] <= 0.0 || data[1] > first_interval) {
    return 1;
  }
  ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, 0, "Ts", &data[1]);
  // Jerk acts only during this ramp. The following hold has zero jerk.
  ocp_nlp_cost_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, 0, "scaling", &data[1]);
  return 0;
}

int custom_update_terminate_function(long_solver_capsule *capsule) {
  (void)capsule;
  return 0;
}
