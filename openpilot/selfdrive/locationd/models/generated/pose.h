#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_7822607448529633229);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_4249233298537579143);
void pose_H_mod_fun(double *state, double *out_3177263273874987825);
void pose_f_fun(double *state, double dt, double *out_691237182481573573);
void pose_F_fun(double *state, double dt, double *out_5721468400791985323);
void pose_h_4(double *state, double *unused, double *out_3930645532014992442);
void pose_H_4(double *state, double *unused, double *out_1979315791992089721);
void pose_h_10(double *state, double *unused, double *out_7266140832989971302);
void pose_H_10(double *state, double *unused, double *out_5742539690088144716);
void pose_h_13(double *state, double *unused, double *out_3951674911255468860);
void pose_H_13(double *state, double *unused, double *out_5813071255294613745);
void pose_h_14(double *state, double *unused, double *out_8796949500966616670);
void pose_H_14(double *state, double *unused, double *out_5062104224287462017);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}