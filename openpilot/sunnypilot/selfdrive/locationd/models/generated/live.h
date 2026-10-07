#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void live_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_9(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_12(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_35(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_32(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_33(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_H(double *in_vec, double *out_5593381341199216691);
void live_err_fun(double *nom_x, double *delta_x, double *out_2784555889927626890);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_1314296965077161696);
void live_H_mod_fun(double *state, double *out_4048706759284590994);
void live_f_fun(double *state, double dt, double *out_7599182337541360880);
void live_F_fun(double *state, double dt, double *out_2615700754434387943);
void live_h_4(double *state, double *unused, double *out_7425949073379995405);
void live_H_4(double *state, double *unused, double *out_2277114388930639403);
void live_h_9(double *state, double *unused, double *out_2032233054044910849);
void live_H_9(double *state, double *unused, double *out_2035924742301048758);
void live_h_10(double *state, double *unused, double *out_6729102152069705987);
void live_H_10(double *state, double *unused, double *out_4672329931393971644);
void live_h_12(double *state, double *unused, double *out_3798928622617683135);
void live_H_12(double *state, double *unused, double *out_2742342019101322392);
void live_h_35(double *state, double *unused, double *out_7042753523488060433);
void live_H_35(double *state, double *unused, double *out_5487905051426336101);
void live_h_32(double *state, double *unused, double *out_3481072224209903822);
void live_H_32(double *state, double *unused, double *out_5287425487142384012);
void live_h_13(double *state, double *unused, double *out_6825947050274631683);
void live_H_13(double *state, double *unused, double *out_5624671507879931283);
void live_h_14(double *state, double *unused, double *out_2032233054044910849);
void live_H_14(double *state, double *unused, double *out_2035924742301048758);
void live_h_33(double *state, double *unused, double *out_5262065714789910684);
void live_H_33(double *state, double *unused, double *out_8638462056065193705);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}