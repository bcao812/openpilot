#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_err_fun(double *nom_x, double *delta_x, double *out_4680453271178883722);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2158585798385581189);
void car_H_mod_fun(double *state, double *out_5438563696312539105);
void car_f_fun(double *state, double dt, double *out_5226637712802323280);
void car_F_fun(double *state, double dt, double *out_1299444942662215767);
void car_h_25(double *state, double *unused, double *out_3872552225024340379);
void car_H_25(double *state, double *unused, double *out_4134224586120567485);
void car_h_24(double *state, double *unused, double *out_9027181674952805450);
void car_H_24(double *state, double *unused, double *out_2899580928508487625);
void car_h_30(double *state, double *unused, double *out_7734235701452842031);
void car_H_30(double *state, double *unused, double *out_4004885638977327415);
void car_h_26(double *state, double *unused, double *out_1730539200249868820);
void car_H_26(double *state, double *unused, double *out_4791078650230879389);
void car_h_27(double *state, double *unused, double *out_9098459440859665326);
void car_H_27(double *state, double *unused, double *out_1830122327176902504);
void car_h_29(double *state, double *unused, double *out_5710835104595863940);
void car_H_29(double *state, double *unused, double *out_4515116983291719599);
void car_h_28(double *state, double *unused, double *out_6061673389415332641);
void car_H_28(double *state, double *unused, double *out_567282033777810975);
void car_h_31(double *state, double *unused, double *out_3821592110914313792);
void car_H_31(double *state, double *unused, double *out_4164870547997527913);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}