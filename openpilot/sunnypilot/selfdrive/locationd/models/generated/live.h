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
void live_H(double *in_vec, double *out_47804104617978308);
void live_err_fun(double *nom_x, double *delta_x, double *out_3044735197278814730);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_1197418404864356272);
void live_H_mod_fun(double *state, double *out_4480637621689338596);
void live_f_fun(double *state, double dt, double *out_7252882908104132962);
void live_F_fun(double *state, double dt, double *out_8296291579486998584);
void live_h_4(double *state, double *unused, double *out_74524857013721926);
void live_H_4(double *state, double *unused, double *out_1894733852524476939);
void live_h_9(double *state, double *unused, double *out_6538024057642950278);
void live_H_9(double *state, double *unused, double *out_5392485082739970531);
void live_h_10(double *state, double *unused, double *out_3055894436910046000);
void live_H_10(double *state, double *unused, double *out_4874795283076868029);
void live_h_12(double *state, double *unused, double *out_5593242706850216274);
void live_H_12(double *state, double *unused, double *out_8275992229567209935);
void live_h_35(double *state, double *unused, double *out_4109558175979041229);
void live_H_35(double *state, double *unused, double *out_1471928204848130437);
void live_h_32(double *state, double *unused, double *out_2434572600123179808);
void live_H_32(double *state, double *unused, double *out_1603092989788870906);
void live_h_13(double *state, double *unused, double *out_8296318041343750975);
void live_H_13(double *state, double *unused, double *out_8810526556892616038);
void live_h_14(double *state, double *unused, double *out_6538024057642950278);
void live_H_14(double *state, double *unused, double *out_5392485082739970531);
void live_h_33(double *state, double *unused, double *out_3199735378693798971);
void live_H_33(double *state, double *unused, double *out_4622485209486988041);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}