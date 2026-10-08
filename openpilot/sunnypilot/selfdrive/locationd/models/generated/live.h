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
void live_H(double *in_vec, double *out_721620894149363910);
void live_err_fun(double *nom_x, double *delta_x, double *out_3514531385097126235);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_4846562840790671580);
void live_H_mod_fun(double *state, double *out_7258895626954848658);
void live_f_fun(double *state, double dt, double *out_7218955641699668044);
void live_F_fun(double *state, double dt, double *out_4504003870660893186);
void live_h_4(double *state, double *unused, double *out_5454830580018947692);
void live_H_4(double *state, double *unused, double *out_4289064323014300081);
void live_h_9(double *state, double *unused, double *out_3939251600139818825);
void live_H_9(double *state, double *unused, double *out_4047874676384709436);
void live_h_10(double *state, double *unused, double *out_6827854678537120624);
void live_H_10(double *state, double *unused, double *out_7232399403443211980);
void live_h_12(double *state, double *unused, double *out_8390839773306270708);
void live_H_12(double *state, double *unused, double *out_730392085017661714);
void live_h_35(double *state, double *unused, double *out_8788208117462245608);
void live_H_35(double *state, double *unused, double *out_3475955117342675423);
void live_h_32(double *state, double *unused, double *out_4187698664317266671);
void live_H_32(double *state, double *unused, double *out_7837266769335809061);
void live_h_13(double *state, double *unused, double *out_5513072798526060030);
void live_H_13(double *state, double *unused, double *out_5815377977510766913);
void live_h_14(double *state, double *unused, double *out_3939251600139818825);
void live_H_14(double *state, double *unused, double *out_4047874676384709436);
void live_h_33(double *state, double *unused, double *out_5062058866136538258);
void live_H_33(double *state, double *unused, double *out_2228154738997164899);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}