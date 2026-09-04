#include "car.h"

namespace {
#define DIM 9
#define EDIM 9
#define MEDIM 9
typedef void (*Hfun)(double *, double *, double *);

double mass;

void set_mass(double x){ mass = x;}

double rotational_inertia;

void set_rotational_inertia(double x){ rotational_inertia = x;}

double center_to_front;

void set_center_to_front(double x){ center_to_front = x;}

double center_to_rear;

void set_center_to_rear(double x){ center_to_rear = x;}

double stiffness_front;

void set_stiffness_front(double x){ stiffness_front = x;}

double stiffness_rear;

void set_stiffness_rear(double x){ stiffness_rear = x;}
const static double MAHA_THRESH_25 = 3.8414588206941227;
const static double MAHA_THRESH_24 = 5.991464547107981;
const static double MAHA_THRESH_30 = 3.8414588206941227;
const static double MAHA_THRESH_26 = 3.8414588206941227;
const static double MAHA_THRESH_27 = 3.8414588206941227;
const static double MAHA_THRESH_29 = 3.8414588206941227;
const static double MAHA_THRESH_28 = 3.8414588206941227;
const static double MAHA_THRESH_31 = 3.8414588206941227;

/******************************************************************************
 *                      Code generated with SymPy 1.14.0                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_4680453271178883722) {
   out_4680453271178883722[0] = delta_x[0] + nom_x[0];
   out_4680453271178883722[1] = delta_x[1] + nom_x[1];
   out_4680453271178883722[2] = delta_x[2] + nom_x[2];
   out_4680453271178883722[3] = delta_x[3] + nom_x[3];
   out_4680453271178883722[4] = delta_x[4] + nom_x[4];
   out_4680453271178883722[5] = delta_x[5] + nom_x[5];
   out_4680453271178883722[6] = delta_x[6] + nom_x[6];
   out_4680453271178883722[7] = delta_x[7] + nom_x[7];
   out_4680453271178883722[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2158585798385581189) {
   out_2158585798385581189[0] = -nom_x[0] + true_x[0];
   out_2158585798385581189[1] = -nom_x[1] + true_x[1];
   out_2158585798385581189[2] = -nom_x[2] + true_x[2];
   out_2158585798385581189[3] = -nom_x[3] + true_x[3];
   out_2158585798385581189[4] = -nom_x[4] + true_x[4];
   out_2158585798385581189[5] = -nom_x[5] + true_x[5];
   out_2158585798385581189[6] = -nom_x[6] + true_x[6];
   out_2158585798385581189[7] = -nom_x[7] + true_x[7];
   out_2158585798385581189[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_5438563696312539105) {
   out_5438563696312539105[0] = 1.0;
   out_5438563696312539105[1] = 0.0;
   out_5438563696312539105[2] = 0.0;
   out_5438563696312539105[3] = 0.0;
   out_5438563696312539105[4] = 0.0;
   out_5438563696312539105[5] = 0.0;
   out_5438563696312539105[6] = 0.0;
   out_5438563696312539105[7] = 0.0;
   out_5438563696312539105[8] = 0.0;
   out_5438563696312539105[9] = 0.0;
   out_5438563696312539105[10] = 1.0;
   out_5438563696312539105[11] = 0.0;
   out_5438563696312539105[12] = 0.0;
   out_5438563696312539105[13] = 0.0;
   out_5438563696312539105[14] = 0.0;
   out_5438563696312539105[15] = 0.0;
   out_5438563696312539105[16] = 0.0;
   out_5438563696312539105[17] = 0.0;
   out_5438563696312539105[18] = 0.0;
   out_5438563696312539105[19] = 0.0;
   out_5438563696312539105[20] = 1.0;
   out_5438563696312539105[21] = 0.0;
   out_5438563696312539105[22] = 0.0;
   out_5438563696312539105[23] = 0.0;
   out_5438563696312539105[24] = 0.0;
   out_5438563696312539105[25] = 0.0;
   out_5438563696312539105[26] = 0.0;
   out_5438563696312539105[27] = 0.0;
   out_5438563696312539105[28] = 0.0;
   out_5438563696312539105[29] = 0.0;
   out_5438563696312539105[30] = 1.0;
   out_5438563696312539105[31] = 0.0;
   out_5438563696312539105[32] = 0.0;
   out_5438563696312539105[33] = 0.0;
   out_5438563696312539105[34] = 0.0;
   out_5438563696312539105[35] = 0.0;
   out_5438563696312539105[36] = 0.0;
   out_5438563696312539105[37] = 0.0;
   out_5438563696312539105[38] = 0.0;
   out_5438563696312539105[39] = 0.0;
   out_5438563696312539105[40] = 1.0;
   out_5438563696312539105[41] = 0.0;
   out_5438563696312539105[42] = 0.0;
   out_5438563696312539105[43] = 0.0;
   out_5438563696312539105[44] = 0.0;
   out_5438563696312539105[45] = 0.0;
   out_5438563696312539105[46] = 0.0;
   out_5438563696312539105[47] = 0.0;
   out_5438563696312539105[48] = 0.0;
   out_5438563696312539105[49] = 0.0;
   out_5438563696312539105[50] = 1.0;
   out_5438563696312539105[51] = 0.0;
   out_5438563696312539105[52] = 0.0;
   out_5438563696312539105[53] = 0.0;
   out_5438563696312539105[54] = 0.0;
   out_5438563696312539105[55] = 0.0;
   out_5438563696312539105[56] = 0.0;
   out_5438563696312539105[57] = 0.0;
   out_5438563696312539105[58] = 0.0;
   out_5438563696312539105[59] = 0.0;
   out_5438563696312539105[60] = 1.0;
   out_5438563696312539105[61] = 0.0;
   out_5438563696312539105[62] = 0.0;
   out_5438563696312539105[63] = 0.0;
   out_5438563696312539105[64] = 0.0;
   out_5438563696312539105[65] = 0.0;
   out_5438563696312539105[66] = 0.0;
   out_5438563696312539105[67] = 0.0;
   out_5438563696312539105[68] = 0.0;
   out_5438563696312539105[69] = 0.0;
   out_5438563696312539105[70] = 1.0;
   out_5438563696312539105[71] = 0.0;
   out_5438563696312539105[72] = 0.0;
   out_5438563696312539105[73] = 0.0;
   out_5438563696312539105[74] = 0.0;
   out_5438563696312539105[75] = 0.0;
   out_5438563696312539105[76] = 0.0;
   out_5438563696312539105[77] = 0.0;
   out_5438563696312539105[78] = 0.0;
   out_5438563696312539105[79] = 0.0;
   out_5438563696312539105[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_5226637712802323280) {
   out_5226637712802323280[0] = state[0];
   out_5226637712802323280[1] = state[1];
   out_5226637712802323280[2] = state[2];
   out_5226637712802323280[3] = state[3];
   out_5226637712802323280[4] = state[4];
   out_5226637712802323280[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8100000000000005*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_5226637712802323280[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_5226637712802323280[7] = state[7];
   out_5226637712802323280[8] = state[8];
}
void F_fun(double *state, double dt, double *out_1299444942662215767) {
   out_1299444942662215767[0] = 1;
   out_1299444942662215767[1] = 0;
   out_1299444942662215767[2] = 0;
   out_1299444942662215767[3] = 0;
   out_1299444942662215767[4] = 0;
   out_1299444942662215767[5] = 0;
   out_1299444942662215767[6] = 0;
   out_1299444942662215767[7] = 0;
   out_1299444942662215767[8] = 0;
   out_1299444942662215767[9] = 0;
   out_1299444942662215767[10] = 1;
   out_1299444942662215767[11] = 0;
   out_1299444942662215767[12] = 0;
   out_1299444942662215767[13] = 0;
   out_1299444942662215767[14] = 0;
   out_1299444942662215767[15] = 0;
   out_1299444942662215767[16] = 0;
   out_1299444942662215767[17] = 0;
   out_1299444942662215767[18] = 0;
   out_1299444942662215767[19] = 0;
   out_1299444942662215767[20] = 1;
   out_1299444942662215767[21] = 0;
   out_1299444942662215767[22] = 0;
   out_1299444942662215767[23] = 0;
   out_1299444942662215767[24] = 0;
   out_1299444942662215767[25] = 0;
   out_1299444942662215767[26] = 0;
   out_1299444942662215767[27] = 0;
   out_1299444942662215767[28] = 0;
   out_1299444942662215767[29] = 0;
   out_1299444942662215767[30] = 1;
   out_1299444942662215767[31] = 0;
   out_1299444942662215767[32] = 0;
   out_1299444942662215767[33] = 0;
   out_1299444942662215767[34] = 0;
   out_1299444942662215767[35] = 0;
   out_1299444942662215767[36] = 0;
   out_1299444942662215767[37] = 0;
   out_1299444942662215767[38] = 0;
   out_1299444942662215767[39] = 0;
   out_1299444942662215767[40] = 1;
   out_1299444942662215767[41] = 0;
   out_1299444942662215767[42] = 0;
   out_1299444942662215767[43] = 0;
   out_1299444942662215767[44] = 0;
   out_1299444942662215767[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_1299444942662215767[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_1299444942662215767[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_1299444942662215767[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_1299444942662215767[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_1299444942662215767[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_1299444942662215767[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_1299444942662215767[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_1299444942662215767[53] = -9.8100000000000005*dt;
   out_1299444942662215767[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_1299444942662215767[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_1299444942662215767[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1299444942662215767[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1299444942662215767[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_1299444942662215767[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_1299444942662215767[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_1299444942662215767[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_1299444942662215767[62] = 0;
   out_1299444942662215767[63] = 0;
   out_1299444942662215767[64] = 0;
   out_1299444942662215767[65] = 0;
   out_1299444942662215767[66] = 0;
   out_1299444942662215767[67] = 0;
   out_1299444942662215767[68] = 0;
   out_1299444942662215767[69] = 0;
   out_1299444942662215767[70] = 1;
   out_1299444942662215767[71] = 0;
   out_1299444942662215767[72] = 0;
   out_1299444942662215767[73] = 0;
   out_1299444942662215767[74] = 0;
   out_1299444942662215767[75] = 0;
   out_1299444942662215767[76] = 0;
   out_1299444942662215767[77] = 0;
   out_1299444942662215767[78] = 0;
   out_1299444942662215767[79] = 0;
   out_1299444942662215767[80] = 1;
}
void h_25(double *state, double *unused, double *out_3872552225024340379) {
   out_3872552225024340379[0] = state[6];
}
void H_25(double *state, double *unused, double *out_4134224586120567485) {
   out_4134224586120567485[0] = 0;
   out_4134224586120567485[1] = 0;
   out_4134224586120567485[2] = 0;
   out_4134224586120567485[3] = 0;
   out_4134224586120567485[4] = 0;
   out_4134224586120567485[5] = 0;
   out_4134224586120567485[6] = 1;
   out_4134224586120567485[7] = 0;
   out_4134224586120567485[8] = 0;
}
void h_24(double *state, double *unused, double *out_9027181674952805450) {
   out_9027181674952805450[0] = state[4];
   out_9027181674952805450[1] = state[5];
}
void H_24(double *state, double *unused, double *out_2899580928508487625) {
   out_2899580928508487625[0] = 0;
   out_2899580928508487625[1] = 0;
   out_2899580928508487625[2] = 0;
   out_2899580928508487625[3] = 0;
   out_2899580928508487625[4] = 1;
   out_2899580928508487625[5] = 0;
   out_2899580928508487625[6] = 0;
   out_2899580928508487625[7] = 0;
   out_2899580928508487625[8] = 0;
   out_2899580928508487625[9] = 0;
   out_2899580928508487625[10] = 0;
   out_2899580928508487625[11] = 0;
   out_2899580928508487625[12] = 0;
   out_2899580928508487625[13] = 0;
   out_2899580928508487625[14] = 1;
   out_2899580928508487625[15] = 0;
   out_2899580928508487625[16] = 0;
   out_2899580928508487625[17] = 0;
}
void h_30(double *state, double *unused, double *out_7734235701452842031) {
   out_7734235701452842031[0] = state[4];
}
void H_30(double *state, double *unused, double *out_4004885638977327415) {
   out_4004885638977327415[0] = 0;
   out_4004885638977327415[1] = 0;
   out_4004885638977327415[2] = 0;
   out_4004885638977327415[3] = 0;
   out_4004885638977327415[4] = 1;
   out_4004885638977327415[5] = 0;
   out_4004885638977327415[6] = 0;
   out_4004885638977327415[7] = 0;
   out_4004885638977327415[8] = 0;
}
void h_26(double *state, double *unused, double *out_1730539200249868820) {
   out_1730539200249868820[0] = state[7];
}
void H_26(double *state, double *unused, double *out_4791078650230879389) {
   out_4791078650230879389[0] = 0;
   out_4791078650230879389[1] = 0;
   out_4791078650230879389[2] = 0;
   out_4791078650230879389[3] = 0;
   out_4791078650230879389[4] = 0;
   out_4791078650230879389[5] = 0;
   out_4791078650230879389[6] = 0;
   out_4791078650230879389[7] = 1;
   out_4791078650230879389[8] = 0;
}
void h_27(double *state, double *unused, double *out_9098459440859665326) {
   out_9098459440859665326[0] = state[3];
}
void H_27(double *state, double *unused, double *out_1830122327176902504) {
   out_1830122327176902504[0] = 0;
   out_1830122327176902504[1] = 0;
   out_1830122327176902504[2] = 0;
   out_1830122327176902504[3] = 1;
   out_1830122327176902504[4] = 0;
   out_1830122327176902504[5] = 0;
   out_1830122327176902504[6] = 0;
   out_1830122327176902504[7] = 0;
   out_1830122327176902504[8] = 0;
}
void h_29(double *state, double *unused, double *out_5710835104595863940) {
   out_5710835104595863940[0] = state[1];
}
void H_29(double *state, double *unused, double *out_4515116983291719599) {
   out_4515116983291719599[0] = 0;
   out_4515116983291719599[1] = 1;
   out_4515116983291719599[2] = 0;
   out_4515116983291719599[3] = 0;
   out_4515116983291719599[4] = 0;
   out_4515116983291719599[5] = 0;
   out_4515116983291719599[6] = 0;
   out_4515116983291719599[7] = 0;
   out_4515116983291719599[8] = 0;
}
void h_28(double *state, double *unused, double *out_6061673389415332641) {
   out_6061673389415332641[0] = state[0];
}
void H_28(double *state, double *unused, double *out_567282033777810975) {
   out_567282033777810975[0] = 1;
   out_567282033777810975[1] = 0;
   out_567282033777810975[2] = 0;
   out_567282033777810975[3] = 0;
   out_567282033777810975[4] = 0;
   out_567282033777810975[5] = 0;
   out_567282033777810975[6] = 0;
   out_567282033777810975[7] = 0;
   out_567282033777810975[8] = 0;
}
void h_31(double *state, double *unused, double *out_3821592110914313792) {
   out_3821592110914313792[0] = state[8];
}
void H_31(double *state, double *unused, double *out_4164870547997527913) {
   out_4164870547997527913[0] = 0;
   out_4164870547997527913[1] = 0;
   out_4164870547997527913[2] = 0;
   out_4164870547997527913[3] = 0;
   out_4164870547997527913[4] = 0;
   out_4164870547997527913[5] = 0;
   out_4164870547997527913[6] = 0;
   out_4164870547997527913[7] = 0;
   out_4164870547997527913[8] = 1;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_25, H_25, NULL, in_z, in_R, in_ea, MAHA_THRESH_25);
}
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<2, 3, 0>(in_x, in_P, h_24, H_24, NULL, in_z, in_R, in_ea, MAHA_THRESH_24);
}
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_30, H_30, NULL, in_z, in_R, in_ea, MAHA_THRESH_30);
}
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_26, H_26, NULL, in_z, in_R, in_ea, MAHA_THRESH_26);
}
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_27, H_27, NULL, in_z, in_R, in_ea, MAHA_THRESH_27);
}
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_29, H_29, NULL, in_z, in_R, in_ea, MAHA_THRESH_29);
}
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_28, H_28, NULL, in_z, in_R, in_ea, MAHA_THRESH_28);
}
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_31, H_31, NULL, in_z, in_R, in_ea, MAHA_THRESH_31);
}
void car_err_fun(double *nom_x, double *delta_x, double *out_4680453271178883722) {
  err_fun(nom_x, delta_x, out_4680453271178883722);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2158585798385581189) {
  inv_err_fun(nom_x, true_x, out_2158585798385581189);
}
void car_H_mod_fun(double *state, double *out_5438563696312539105) {
  H_mod_fun(state, out_5438563696312539105);
}
void car_f_fun(double *state, double dt, double *out_5226637712802323280) {
  f_fun(state,  dt, out_5226637712802323280);
}
void car_F_fun(double *state, double dt, double *out_1299444942662215767) {
  F_fun(state,  dt, out_1299444942662215767);
}
void car_h_25(double *state, double *unused, double *out_3872552225024340379) {
  h_25(state, unused, out_3872552225024340379);
}
void car_H_25(double *state, double *unused, double *out_4134224586120567485) {
  H_25(state, unused, out_4134224586120567485);
}
void car_h_24(double *state, double *unused, double *out_9027181674952805450) {
  h_24(state, unused, out_9027181674952805450);
}
void car_H_24(double *state, double *unused, double *out_2899580928508487625) {
  H_24(state, unused, out_2899580928508487625);
}
void car_h_30(double *state, double *unused, double *out_7734235701452842031) {
  h_30(state, unused, out_7734235701452842031);
}
void car_H_30(double *state, double *unused, double *out_4004885638977327415) {
  H_30(state, unused, out_4004885638977327415);
}
void car_h_26(double *state, double *unused, double *out_1730539200249868820) {
  h_26(state, unused, out_1730539200249868820);
}
void car_H_26(double *state, double *unused, double *out_4791078650230879389) {
  H_26(state, unused, out_4791078650230879389);
}
void car_h_27(double *state, double *unused, double *out_9098459440859665326) {
  h_27(state, unused, out_9098459440859665326);
}
void car_H_27(double *state, double *unused, double *out_1830122327176902504) {
  H_27(state, unused, out_1830122327176902504);
}
void car_h_29(double *state, double *unused, double *out_5710835104595863940) {
  h_29(state, unused, out_5710835104595863940);
}
void car_H_29(double *state, double *unused, double *out_4515116983291719599) {
  H_29(state, unused, out_4515116983291719599);
}
void car_h_28(double *state, double *unused, double *out_6061673389415332641) {
  h_28(state, unused, out_6061673389415332641);
}
void car_H_28(double *state, double *unused, double *out_567282033777810975) {
  H_28(state, unused, out_567282033777810975);
}
void car_h_31(double *state, double *unused, double *out_3821592110914313792) {
  h_31(state, unused, out_3821592110914313792);
}
void car_H_31(double *state, double *unused, double *out_4164870547997527913) {
  H_31(state, unused, out_4164870547997527913);
}
void car_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
void car_set_mass(double x) {
  set_mass(x);
}
void car_set_rotational_inertia(double x) {
  set_rotational_inertia(x);
}
void car_set_center_to_front(double x) {
  set_center_to_front(x);
}
void car_set_center_to_rear(double x) {
  set_center_to_rear(x);
}
void car_set_stiffness_front(double x) {
  set_stiffness_front(x);
}
void car_set_stiffness_rear(double x) {
  set_stiffness_rear(x);
}
}

const EKF car = {
  .name = "car",
  .kinds = { 25, 24, 30, 26, 27, 29, 28, 31 },
  .feature_kinds = {  },
  .f_fun = car_f_fun,
  .F_fun = car_F_fun,
  .err_fun = car_err_fun,
  .inv_err_fun = car_inv_err_fun,
  .H_mod_fun = car_H_mod_fun,
  .predict = car_predict,
  .hs = {
    { 25, car_h_25 },
    { 24, car_h_24 },
    { 30, car_h_30 },
    { 26, car_h_26 },
    { 27, car_h_27 },
    { 29, car_h_29 },
    { 28, car_h_28 },
    { 31, car_h_31 },
  },
  .Hs = {
    { 25, car_H_25 },
    { 24, car_H_24 },
    { 30, car_H_30 },
    { 26, car_H_26 },
    { 27, car_H_27 },
    { 29, car_H_29 },
    { 28, car_H_28 },
    { 31, car_H_31 },
  },
  .updates = {
    { 25, car_update_25 },
    { 24, car_update_24 },
    { 30, car_update_30 },
    { 26, car_update_26 },
    { 27, car_update_27 },
    { 29, car_update_29 },
    { 28, car_update_28 },
    { 31, car_update_31 },
  },
  .Hes = {
  },
  .sets = {
    { "mass", car_set_mass },
    { "rotational_inertia", car_set_rotational_inertia },
    { "center_to_front", car_set_center_to_front },
    { "center_to_rear", car_set_center_to_rear },
    { "stiffness_front", car_set_stiffness_front },
    { "stiffness_rear", car_set_stiffness_rear },
  },
  .extra_routines = {
  },
};

ekf_lib_init(car)
