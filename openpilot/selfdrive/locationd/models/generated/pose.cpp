#include "pose.h"

namespace {
#define DIM 18
#define EDIM 18
#define MEDIM 18
typedef void (*Hfun)(double *, double *, double *);
const static double MAHA_THRESH_4 = 7.814727903251177;
const static double MAHA_THRESH_10 = 7.814727903251177;
const static double MAHA_THRESH_13 = 7.814727903251177;
const static double MAHA_THRESH_14 = 7.814727903251177;

/******************************************************************************
 *                      Code generated with SymPy 1.14.0                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_7822607448529633229) {
   out_7822607448529633229[0] = delta_x[0] + nom_x[0];
   out_7822607448529633229[1] = delta_x[1] + nom_x[1];
   out_7822607448529633229[2] = delta_x[2] + nom_x[2];
   out_7822607448529633229[3] = delta_x[3] + nom_x[3];
   out_7822607448529633229[4] = delta_x[4] + nom_x[4];
   out_7822607448529633229[5] = delta_x[5] + nom_x[5];
   out_7822607448529633229[6] = delta_x[6] + nom_x[6];
   out_7822607448529633229[7] = delta_x[7] + nom_x[7];
   out_7822607448529633229[8] = delta_x[8] + nom_x[8];
   out_7822607448529633229[9] = delta_x[9] + nom_x[9];
   out_7822607448529633229[10] = delta_x[10] + nom_x[10];
   out_7822607448529633229[11] = delta_x[11] + nom_x[11];
   out_7822607448529633229[12] = delta_x[12] + nom_x[12];
   out_7822607448529633229[13] = delta_x[13] + nom_x[13];
   out_7822607448529633229[14] = delta_x[14] + nom_x[14];
   out_7822607448529633229[15] = delta_x[15] + nom_x[15];
   out_7822607448529633229[16] = delta_x[16] + nom_x[16];
   out_7822607448529633229[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_4249233298537579143) {
   out_4249233298537579143[0] = -nom_x[0] + true_x[0];
   out_4249233298537579143[1] = -nom_x[1] + true_x[1];
   out_4249233298537579143[2] = -nom_x[2] + true_x[2];
   out_4249233298537579143[3] = -nom_x[3] + true_x[3];
   out_4249233298537579143[4] = -nom_x[4] + true_x[4];
   out_4249233298537579143[5] = -nom_x[5] + true_x[5];
   out_4249233298537579143[6] = -nom_x[6] + true_x[6];
   out_4249233298537579143[7] = -nom_x[7] + true_x[7];
   out_4249233298537579143[8] = -nom_x[8] + true_x[8];
   out_4249233298537579143[9] = -nom_x[9] + true_x[9];
   out_4249233298537579143[10] = -nom_x[10] + true_x[10];
   out_4249233298537579143[11] = -nom_x[11] + true_x[11];
   out_4249233298537579143[12] = -nom_x[12] + true_x[12];
   out_4249233298537579143[13] = -nom_x[13] + true_x[13];
   out_4249233298537579143[14] = -nom_x[14] + true_x[14];
   out_4249233298537579143[15] = -nom_x[15] + true_x[15];
   out_4249233298537579143[16] = -nom_x[16] + true_x[16];
   out_4249233298537579143[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_3177263273874987825) {
   out_3177263273874987825[0] = 1.0;
   out_3177263273874987825[1] = 0.0;
   out_3177263273874987825[2] = 0.0;
   out_3177263273874987825[3] = 0.0;
   out_3177263273874987825[4] = 0.0;
   out_3177263273874987825[5] = 0.0;
   out_3177263273874987825[6] = 0.0;
   out_3177263273874987825[7] = 0.0;
   out_3177263273874987825[8] = 0.0;
   out_3177263273874987825[9] = 0.0;
   out_3177263273874987825[10] = 0.0;
   out_3177263273874987825[11] = 0.0;
   out_3177263273874987825[12] = 0.0;
   out_3177263273874987825[13] = 0.0;
   out_3177263273874987825[14] = 0.0;
   out_3177263273874987825[15] = 0.0;
   out_3177263273874987825[16] = 0.0;
   out_3177263273874987825[17] = 0.0;
   out_3177263273874987825[18] = 0.0;
   out_3177263273874987825[19] = 1.0;
   out_3177263273874987825[20] = 0.0;
   out_3177263273874987825[21] = 0.0;
   out_3177263273874987825[22] = 0.0;
   out_3177263273874987825[23] = 0.0;
   out_3177263273874987825[24] = 0.0;
   out_3177263273874987825[25] = 0.0;
   out_3177263273874987825[26] = 0.0;
   out_3177263273874987825[27] = 0.0;
   out_3177263273874987825[28] = 0.0;
   out_3177263273874987825[29] = 0.0;
   out_3177263273874987825[30] = 0.0;
   out_3177263273874987825[31] = 0.0;
   out_3177263273874987825[32] = 0.0;
   out_3177263273874987825[33] = 0.0;
   out_3177263273874987825[34] = 0.0;
   out_3177263273874987825[35] = 0.0;
   out_3177263273874987825[36] = 0.0;
   out_3177263273874987825[37] = 0.0;
   out_3177263273874987825[38] = 1.0;
   out_3177263273874987825[39] = 0.0;
   out_3177263273874987825[40] = 0.0;
   out_3177263273874987825[41] = 0.0;
   out_3177263273874987825[42] = 0.0;
   out_3177263273874987825[43] = 0.0;
   out_3177263273874987825[44] = 0.0;
   out_3177263273874987825[45] = 0.0;
   out_3177263273874987825[46] = 0.0;
   out_3177263273874987825[47] = 0.0;
   out_3177263273874987825[48] = 0.0;
   out_3177263273874987825[49] = 0.0;
   out_3177263273874987825[50] = 0.0;
   out_3177263273874987825[51] = 0.0;
   out_3177263273874987825[52] = 0.0;
   out_3177263273874987825[53] = 0.0;
   out_3177263273874987825[54] = 0.0;
   out_3177263273874987825[55] = 0.0;
   out_3177263273874987825[56] = 0.0;
   out_3177263273874987825[57] = 1.0;
   out_3177263273874987825[58] = 0.0;
   out_3177263273874987825[59] = 0.0;
   out_3177263273874987825[60] = 0.0;
   out_3177263273874987825[61] = 0.0;
   out_3177263273874987825[62] = 0.0;
   out_3177263273874987825[63] = 0.0;
   out_3177263273874987825[64] = 0.0;
   out_3177263273874987825[65] = 0.0;
   out_3177263273874987825[66] = 0.0;
   out_3177263273874987825[67] = 0.0;
   out_3177263273874987825[68] = 0.0;
   out_3177263273874987825[69] = 0.0;
   out_3177263273874987825[70] = 0.0;
   out_3177263273874987825[71] = 0.0;
   out_3177263273874987825[72] = 0.0;
   out_3177263273874987825[73] = 0.0;
   out_3177263273874987825[74] = 0.0;
   out_3177263273874987825[75] = 0.0;
   out_3177263273874987825[76] = 1.0;
   out_3177263273874987825[77] = 0.0;
   out_3177263273874987825[78] = 0.0;
   out_3177263273874987825[79] = 0.0;
   out_3177263273874987825[80] = 0.0;
   out_3177263273874987825[81] = 0.0;
   out_3177263273874987825[82] = 0.0;
   out_3177263273874987825[83] = 0.0;
   out_3177263273874987825[84] = 0.0;
   out_3177263273874987825[85] = 0.0;
   out_3177263273874987825[86] = 0.0;
   out_3177263273874987825[87] = 0.0;
   out_3177263273874987825[88] = 0.0;
   out_3177263273874987825[89] = 0.0;
   out_3177263273874987825[90] = 0.0;
   out_3177263273874987825[91] = 0.0;
   out_3177263273874987825[92] = 0.0;
   out_3177263273874987825[93] = 0.0;
   out_3177263273874987825[94] = 0.0;
   out_3177263273874987825[95] = 1.0;
   out_3177263273874987825[96] = 0.0;
   out_3177263273874987825[97] = 0.0;
   out_3177263273874987825[98] = 0.0;
   out_3177263273874987825[99] = 0.0;
   out_3177263273874987825[100] = 0.0;
   out_3177263273874987825[101] = 0.0;
   out_3177263273874987825[102] = 0.0;
   out_3177263273874987825[103] = 0.0;
   out_3177263273874987825[104] = 0.0;
   out_3177263273874987825[105] = 0.0;
   out_3177263273874987825[106] = 0.0;
   out_3177263273874987825[107] = 0.0;
   out_3177263273874987825[108] = 0.0;
   out_3177263273874987825[109] = 0.0;
   out_3177263273874987825[110] = 0.0;
   out_3177263273874987825[111] = 0.0;
   out_3177263273874987825[112] = 0.0;
   out_3177263273874987825[113] = 0.0;
   out_3177263273874987825[114] = 1.0;
   out_3177263273874987825[115] = 0.0;
   out_3177263273874987825[116] = 0.0;
   out_3177263273874987825[117] = 0.0;
   out_3177263273874987825[118] = 0.0;
   out_3177263273874987825[119] = 0.0;
   out_3177263273874987825[120] = 0.0;
   out_3177263273874987825[121] = 0.0;
   out_3177263273874987825[122] = 0.0;
   out_3177263273874987825[123] = 0.0;
   out_3177263273874987825[124] = 0.0;
   out_3177263273874987825[125] = 0.0;
   out_3177263273874987825[126] = 0.0;
   out_3177263273874987825[127] = 0.0;
   out_3177263273874987825[128] = 0.0;
   out_3177263273874987825[129] = 0.0;
   out_3177263273874987825[130] = 0.0;
   out_3177263273874987825[131] = 0.0;
   out_3177263273874987825[132] = 0.0;
   out_3177263273874987825[133] = 1.0;
   out_3177263273874987825[134] = 0.0;
   out_3177263273874987825[135] = 0.0;
   out_3177263273874987825[136] = 0.0;
   out_3177263273874987825[137] = 0.0;
   out_3177263273874987825[138] = 0.0;
   out_3177263273874987825[139] = 0.0;
   out_3177263273874987825[140] = 0.0;
   out_3177263273874987825[141] = 0.0;
   out_3177263273874987825[142] = 0.0;
   out_3177263273874987825[143] = 0.0;
   out_3177263273874987825[144] = 0.0;
   out_3177263273874987825[145] = 0.0;
   out_3177263273874987825[146] = 0.0;
   out_3177263273874987825[147] = 0.0;
   out_3177263273874987825[148] = 0.0;
   out_3177263273874987825[149] = 0.0;
   out_3177263273874987825[150] = 0.0;
   out_3177263273874987825[151] = 0.0;
   out_3177263273874987825[152] = 1.0;
   out_3177263273874987825[153] = 0.0;
   out_3177263273874987825[154] = 0.0;
   out_3177263273874987825[155] = 0.0;
   out_3177263273874987825[156] = 0.0;
   out_3177263273874987825[157] = 0.0;
   out_3177263273874987825[158] = 0.0;
   out_3177263273874987825[159] = 0.0;
   out_3177263273874987825[160] = 0.0;
   out_3177263273874987825[161] = 0.0;
   out_3177263273874987825[162] = 0.0;
   out_3177263273874987825[163] = 0.0;
   out_3177263273874987825[164] = 0.0;
   out_3177263273874987825[165] = 0.0;
   out_3177263273874987825[166] = 0.0;
   out_3177263273874987825[167] = 0.0;
   out_3177263273874987825[168] = 0.0;
   out_3177263273874987825[169] = 0.0;
   out_3177263273874987825[170] = 0.0;
   out_3177263273874987825[171] = 1.0;
   out_3177263273874987825[172] = 0.0;
   out_3177263273874987825[173] = 0.0;
   out_3177263273874987825[174] = 0.0;
   out_3177263273874987825[175] = 0.0;
   out_3177263273874987825[176] = 0.0;
   out_3177263273874987825[177] = 0.0;
   out_3177263273874987825[178] = 0.0;
   out_3177263273874987825[179] = 0.0;
   out_3177263273874987825[180] = 0.0;
   out_3177263273874987825[181] = 0.0;
   out_3177263273874987825[182] = 0.0;
   out_3177263273874987825[183] = 0.0;
   out_3177263273874987825[184] = 0.0;
   out_3177263273874987825[185] = 0.0;
   out_3177263273874987825[186] = 0.0;
   out_3177263273874987825[187] = 0.0;
   out_3177263273874987825[188] = 0.0;
   out_3177263273874987825[189] = 0.0;
   out_3177263273874987825[190] = 1.0;
   out_3177263273874987825[191] = 0.0;
   out_3177263273874987825[192] = 0.0;
   out_3177263273874987825[193] = 0.0;
   out_3177263273874987825[194] = 0.0;
   out_3177263273874987825[195] = 0.0;
   out_3177263273874987825[196] = 0.0;
   out_3177263273874987825[197] = 0.0;
   out_3177263273874987825[198] = 0.0;
   out_3177263273874987825[199] = 0.0;
   out_3177263273874987825[200] = 0.0;
   out_3177263273874987825[201] = 0.0;
   out_3177263273874987825[202] = 0.0;
   out_3177263273874987825[203] = 0.0;
   out_3177263273874987825[204] = 0.0;
   out_3177263273874987825[205] = 0.0;
   out_3177263273874987825[206] = 0.0;
   out_3177263273874987825[207] = 0.0;
   out_3177263273874987825[208] = 0.0;
   out_3177263273874987825[209] = 1.0;
   out_3177263273874987825[210] = 0.0;
   out_3177263273874987825[211] = 0.0;
   out_3177263273874987825[212] = 0.0;
   out_3177263273874987825[213] = 0.0;
   out_3177263273874987825[214] = 0.0;
   out_3177263273874987825[215] = 0.0;
   out_3177263273874987825[216] = 0.0;
   out_3177263273874987825[217] = 0.0;
   out_3177263273874987825[218] = 0.0;
   out_3177263273874987825[219] = 0.0;
   out_3177263273874987825[220] = 0.0;
   out_3177263273874987825[221] = 0.0;
   out_3177263273874987825[222] = 0.0;
   out_3177263273874987825[223] = 0.0;
   out_3177263273874987825[224] = 0.0;
   out_3177263273874987825[225] = 0.0;
   out_3177263273874987825[226] = 0.0;
   out_3177263273874987825[227] = 0.0;
   out_3177263273874987825[228] = 1.0;
   out_3177263273874987825[229] = 0.0;
   out_3177263273874987825[230] = 0.0;
   out_3177263273874987825[231] = 0.0;
   out_3177263273874987825[232] = 0.0;
   out_3177263273874987825[233] = 0.0;
   out_3177263273874987825[234] = 0.0;
   out_3177263273874987825[235] = 0.0;
   out_3177263273874987825[236] = 0.0;
   out_3177263273874987825[237] = 0.0;
   out_3177263273874987825[238] = 0.0;
   out_3177263273874987825[239] = 0.0;
   out_3177263273874987825[240] = 0.0;
   out_3177263273874987825[241] = 0.0;
   out_3177263273874987825[242] = 0.0;
   out_3177263273874987825[243] = 0.0;
   out_3177263273874987825[244] = 0.0;
   out_3177263273874987825[245] = 0.0;
   out_3177263273874987825[246] = 0.0;
   out_3177263273874987825[247] = 1.0;
   out_3177263273874987825[248] = 0.0;
   out_3177263273874987825[249] = 0.0;
   out_3177263273874987825[250] = 0.0;
   out_3177263273874987825[251] = 0.0;
   out_3177263273874987825[252] = 0.0;
   out_3177263273874987825[253] = 0.0;
   out_3177263273874987825[254] = 0.0;
   out_3177263273874987825[255] = 0.0;
   out_3177263273874987825[256] = 0.0;
   out_3177263273874987825[257] = 0.0;
   out_3177263273874987825[258] = 0.0;
   out_3177263273874987825[259] = 0.0;
   out_3177263273874987825[260] = 0.0;
   out_3177263273874987825[261] = 0.0;
   out_3177263273874987825[262] = 0.0;
   out_3177263273874987825[263] = 0.0;
   out_3177263273874987825[264] = 0.0;
   out_3177263273874987825[265] = 0.0;
   out_3177263273874987825[266] = 1.0;
   out_3177263273874987825[267] = 0.0;
   out_3177263273874987825[268] = 0.0;
   out_3177263273874987825[269] = 0.0;
   out_3177263273874987825[270] = 0.0;
   out_3177263273874987825[271] = 0.0;
   out_3177263273874987825[272] = 0.0;
   out_3177263273874987825[273] = 0.0;
   out_3177263273874987825[274] = 0.0;
   out_3177263273874987825[275] = 0.0;
   out_3177263273874987825[276] = 0.0;
   out_3177263273874987825[277] = 0.0;
   out_3177263273874987825[278] = 0.0;
   out_3177263273874987825[279] = 0.0;
   out_3177263273874987825[280] = 0.0;
   out_3177263273874987825[281] = 0.0;
   out_3177263273874987825[282] = 0.0;
   out_3177263273874987825[283] = 0.0;
   out_3177263273874987825[284] = 0.0;
   out_3177263273874987825[285] = 1.0;
   out_3177263273874987825[286] = 0.0;
   out_3177263273874987825[287] = 0.0;
   out_3177263273874987825[288] = 0.0;
   out_3177263273874987825[289] = 0.0;
   out_3177263273874987825[290] = 0.0;
   out_3177263273874987825[291] = 0.0;
   out_3177263273874987825[292] = 0.0;
   out_3177263273874987825[293] = 0.0;
   out_3177263273874987825[294] = 0.0;
   out_3177263273874987825[295] = 0.0;
   out_3177263273874987825[296] = 0.0;
   out_3177263273874987825[297] = 0.0;
   out_3177263273874987825[298] = 0.0;
   out_3177263273874987825[299] = 0.0;
   out_3177263273874987825[300] = 0.0;
   out_3177263273874987825[301] = 0.0;
   out_3177263273874987825[302] = 0.0;
   out_3177263273874987825[303] = 0.0;
   out_3177263273874987825[304] = 1.0;
   out_3177263273874987825[305] = 0.0;
   out_3177263273874987825[306] = 0.0;
   out_3177263273874987825[307] = 0.0;
   out_3177263273874987825[308] = 0.0;
   out_3177263273874987825[309] = 0.0;
   out_3177263273874987825[310] = 0.0;
   out_3177263273874987825[311] = 0.0;
   out_3177263273874987825[312] = 0.0;
   out_3177263273874987825[313] = 0.0;
   out_3177263273874987825[314] = 0.0;
   out_3177263273874987825[315] = 0.0;
   out_3177263273874987825[316] = 0.0;
   out_3177263273874987825[317] = 0.0;
   out_3177263273874987825[318] = 0.0;
   out_3177263273874987825[319] = 0.0;
   out_3177263273874987825[320] = 0.0;
   out_3177263273874987825[321] = 0.0;
   out_3177263273874987825[322] = 0.0;
   out_3177263273874987825[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_691237182481573573) {
   out_691237182481573573[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_691237182481573573[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_691237182481573573[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_691237182481573573[3] = dt*state[12] + state[3];
   out_691237182481573573[4] = dt*state[13] + state[4];
   out_691237182481573573[5] = dt*state[14] + state[5];
   out_691237182481573573[6] = state[6];
   out_691237182481573573[7] = state[7];
   out_691237182481573573[8] = state[8];
   out_691237182481573573[9] = state[9];
   out_691237182481573573[10] = state[10];
   out_691237182481573573[11] = state[11];
   out_691237182481573573[12] = state[12];
   out_691237182481573573[13] = state[13];
   out_691237182481573573[14] = state[14];
   out_691237182481573573[15] = state[15];
   out_691237182481573573[16] = state[16];
   out_691237182481573573[17] = state[17];
}
void F_fun(double *state, double dt, double *out_5721468400791985323) {
   out_5721468400791985323[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5721468400791985323[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5721468400791985323[2] = 0;
   out_5721468400791985323[3] = 0;
   out_5721468400791985323[4] = 0;
   out_5721468400791985323[5] = 0;
   out_5721468400791985323[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5721468400791985323[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5721468400791985323[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_5721468400791985323[9] = 0;
   out_5721468400791985323[10] = 0;
   out_5721468400791985323[11] = 0;
   out_5721468400791985323[12] = 0;
   out_5721468400791985323[13] = 0;
   out_5721468400791985323[14] = 0;
   out_5721468400791985323[15] = 0;
   out_5721468400791985323[16] = 0;
   out_5721468400791985323[17] = 0;
   out_5721468400791985323[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_5721468400791985323[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_5721468400791985323[20] = 0;
   out_5721468400791985323[21] = 0;
   out_5721468400791985323[22] = 0;
   out_5721468400791985323[23] = 0;
   out_5721468400791985323[24] = 0;
   out_5721468400791985323[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_5721468400791985323[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_5721468400791985323[27] = 0;
   out_5721468400791985323[28] = 0;
   out_5721468400791985323[29] = 0;
   out_5721468400791985323[30] = 0;
   out_5721468400791985323[31] = 0;
   out_5721468400791985323[32] = 0;
   out_5721468400791985323[33] = 0;
   out_5721468400791985323[34] = 0;
   out_5721468400791985323[35] = 0;
   out_5721468400791985323[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5721468400791985323[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5721468400791985323[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5721468400791985323[39] = 0;
   out_5721468400791985323[40] = 0;
   out_5721468400791985323[41] = 0;
   out_5721468400791985323[42] = 0;
   out_5721468400791985323[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5721468400791985323[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_5721468400791985323[45] = 0;
   out_5721468400791985323[46] = 0;
   out_5721468400791985323[47] = 0;
   out_5721468400791985323[48] = 0;
   out_5721468400791985323[49] = 0;
   out_5721468400791985323[50] = 0;
   out_5721468400791985323[51] = 0;
   out_5721468400791985323[52] = 0;
   out_5721468400791985323[53] = 0;
   out_5721468400791985323[54] = 0;
   out_5721468400791985323[55] = 0;
   out_5721468400791985323[56] = 0;
   out_5721468400791985323[57] = 1;
   out_5721468400791985323[58] = 0;
   out_5721468400791985323[59] = 0;
   out_5721468400791985323[60] = 0;
   out_5721468400791985323[61] = 0;
   out_5721468400791985323[62] = 0;
   out_5721468400791985323[63] = 0;
   out_5721468400791985323[64] = 0;
   out_5721468400791985323[65] = 0;
   out_5721468400791985323[66] = dt;
   out_5721468400791985323[67] = 0;
   out_5721468400791985323[68] = 0;
   out_5721468400791985323[69] = 0;
   out_5721468400791985323[70] = 0;
   out_5721468400791985323[71] = 0;
   out_5721468400791985323[72] = 0;
   out_5721468400791985323[73] = 0;
   out_5721468400791985323[74] = 0;
   out_5721468400791985323[75] = 0;
   out_5721468400791985323[76] = 1;
   out_5721468400791985323[77] = 0;
   out_5721468400791985323[78] = 0;
   out_5721468400791985323[79] = 0;
   out_5721468400791985323[80] = 0;
   out_5721468400791985323[81] = 0;
   out_5721468400791985323[82] = 0;
   out_5721468400791985323[83] = 0;
   out_5721468400791985323[84] = 0;
   out_5721468400791985323[85] = dt;
   out_5721468400791985323[86] = 0;
   out_5721468400791985323[87] = 0;
   out_5721468400791985323[88] = 0;
   out_5721468400791985323[89] = 0;
   out_5721468400791985323[90] = 0;
   out_5721468400791985323[91] = 0;
   out_5721468400791985323[92] = 0;
   out_5721468400791985323[93] = 0;
   out_5721468400791985323[94] = 0;
   out_5721468400791985323[95] = 1;
   out_5721468400791985323[96] = 0;
   out_5721468400791985323[97] = 0;
   out_5721468400791985323[98] = 0;
   out_5721468400791985323[99] = 0;
   out_5721468400791985323[100] = 0;
   out_5721468400791985323[101] = 0;
   out_5721468400791985323[102] = 0;
   out_5721468400791985323[103] = 0;
   out_5721468400791985323[104] = dt;
   out_5721468400791985323[105] = 0;
   out_5721468400791985323[106] = 0;
   out_5721468400791985323[107] = 0;
   out_5721468400791985323[108] = 0;
   out_5721468400791985323[109] = 0;
   out_5721468400791985323[110] = 0;
   out_5721468400791985323[111] = 0;
   out_5721468400791985323[112] = 0;
   out_5721468400791985323[113] = 0;
   out_5721468400791985323[114] = 1;
   out_5721468400791985323[115] = 0;
   out_5721468400791985323[116] = 0;
   out_5721468400791985323[117] = 0;
   out_5721468400791985323[118] = 0;
   out_5721468400791985323[119] = 0;
   out_5721468400791985323[120] = 0;
   out_5721468400791985323[121] = 0;
   out_5721468400791985323[122] = 0;
   out_5721468400791985323[123] = 0;
   out_5721468400791985323[124] = 0;
   out_5721468400791985323[125] = 0;
   out_5721468400791985323[126] = 0;
   out_5721468400791985323[127] = 0;
   out_5721468400791985323[128] = 0;
   out_5721468400791985323[129] = 0;
   out_5721468400791985323[130] = 0;
   out_5721468400791985323[131] = 0;
   out_5721468400791985323[132] = 0;
   out_5721468400791985323[133] = 1;
   out_5721468400791985323[134] = 0;
   out_5721468400791985323[135] = 0;
   out_5721468400791985323[136] = 0;
   out_5721468400791985323[137] = 0;
   out_5721468400791985323[138] = 0;
   out_5721468400791985323[139] = 0;
   out_5721468400791985323[140] = 0;
   out_5721468400791985323[141] = 0;
   out_5721468400791985323[142] = 0;
   out_5721468400791985323[143] = 0;
   out_5721468400791985323[144] = 0;
   out_5721468400791985323[145] = 0;
   out_5721468400791985323[146] = 0;
   out_5721468400791985323[147] = 0;
   out_5721468400791985323[148] = 0;
   out_5721468400791985323[149] = 0;
   out_5721468400791985323[150] = 0;
   out_5721468400791985323[151] = 0;
   out_5721468400791985323[152] = 1;
   out_5721468400791985323[153] = 0;
   out_5721468400791985323[154] = 0;
   out_5721468400791985323[155] = 0;
   out_5721468400791985323[156] = 0;
   out_5721468400791985323[157] = 0;
   out_5721468400791985323[158] = 0;
   out_5721468400791985323[159] = 0;
   out_5721468400791985323[160] = 0;
   out_5721468400791985323[161] = 0;
   out_5721468400791985323[162] = 0;
   out_5721468400791985323[163] = 0;
   out_5721468400791985323[164] = 0;
   out_5721468400791985323[165] = 0;
   out_5721468400791985323[166] = 0;
   out_5721468400791985323[167] = 0;
   out_5721468400791985323[168] = 0;
   out_5721468400791985323[169] = 0;
   out_5721468400791985323[170] = 0;
   out_5721468400791985323[171] = 1;
   out_5721468400791985323[172] = 0;
   out_5721468400791985323[173] = 0;
   out_5721468400791985323[174] = 0;
   out_5721468400791985323[175] = 0;
   out_5721468400791985323[176] = 0;
   out_5721468400791985323[177] = 0;
   out_5721468400791985323[178] = 0;
   out_5721468400791985323[179] = 0;
   out_5721468400791985323[180] = 0;
   out_5721468400791985323[181] = 0;
   out_5721468400791985323[182] = 0;
   out_5721468400791985323[183] = 0;
   out_5721468400791985323[184] = 0;
   out_5721468400791985323[185] = 0;
   out_5721468400791985323[186] = 0;
   out_5721468400791985323[187] = 0;
   out_5721468400791985323[188] = 0;
   out_5721468400791985323[189] = 0;
   out_5721468400791985323[190] = 1;
   out_5721468400791985323[191] = 0;
   out_5721468400791985323[192] = 0;
   out_5721468400791985323[193] = 0;
   out_5721468400791985323[194] = 0;
   out_5721468400791985323[195] = 0;
   out_5721468400791985323[196] = 0;
   out_5721468400791985323[197] = 0;
   out_5721468400791985323[198] = 0;
   out_5721468400791985323[199] = 0;
   out_5721468400791985323[200] = 0;
   out_5721468400791985323[201] = 0;
   out_5721468400791985323[202] = 0;
   out_5721468400791985323[203] = 0;
   out_5721468400791985323[204] = 0;
   out_5721468400791985323[205] = 0;
   out_5721468400791985323[206] = 0;
   out_5721468400791985323[207] = 0;
   out_5721468400791985323[208] = 0;
   out_5721468400791985323[209] = 1;
   out_5721468400791985323[210] = 0;
   out_5721468400791985323[211] = 0;
   out_5721468400791985323[212] = 0;
   out_5721468400791985323[213] = 0;
   out_5721468400791985323[214] = 0;
   out_5721468400791985323[215] = 0;
   out_5721468400791985323[216] = 0;
   out_5721468400791985323[217] = 0;
   out_5721468400791985323[218] = 0;
   out_5721468400791985323[219] = 0;
   out_5721468400791985323[220] = 0;
   out_5721468400791985323[221] = 0;
   out_5721468400791985323[222] = 0;
   out_5721468400791985323[223] = 0;
   out_5721468400791985323[224] = 0;
   out_5721468400791985323[225] = 0;
   out_5721468400791985323[226] = 0;
   out_5721468400791985323[227] = 0;
   out_5721468400791985323[228] = 1;
   out_5721468400791985323[229] = 0;
   out_5721468400791985323[230] = 0;
   out_5721468400791985323[231] = 0;
   out_5721468400791985323[232] = 0;
   out_5721468400791985323[233] = 0;
   out_5721468400791985323[234] = 0;
   out_5721468400791985323[235] = 0;
   out_5721468400791985323[236] = 0;
   out_5721468400791985323[237] = 0;
   out_5721468400791985323[238] = 0;
   out_5721468400791985323[239] = 0;
   out_5721468400791985323[240] = 0;
   out_5721468400791985323[241] = 0;
   out_5721468400791985323[242] = 0;
   out_5721468400791985323[243] = 0;
   out_5721468400791985323[244] = 0;
   out_5721468400791985323[245] = 0;
   out_5721468400791985323[246] = 0;
   out_5721468400791985323[247] = 1;
   out_5721468400791985323[248] = 0;
   out_5721468400791985323[249] = 0;
   out_5721468400791985323[250] = 0;
   out_5721468400791985323[251] = 0;
   out_5721468400791985323[252] = 0;
   out_5721468400791985323[253] = 0;
   out_5721468400791985323[254] = 0;
   out_5721468400791985323[255] = 0;
   out_5721468400791985323[256] = 0;
   out_5721468400791985323[257] = 0;
   out_5721468400791985323[258] = 0;
   out_5721468400791985323[259] = 0;
   out_5721468400791985323[260] = 0;
   out_5721468400791985323[261] = 0;
   out_5721468400791985323[262] = 0;
   out_5721468400791985323[263] = 0;
   out_5721468400791985323[264] = 0;
   out_5721468400791985323[265] = 0;
   out_5721468400791985323[266] = 1;
   out_5721468400791985323[267] = 0;
   out_5721468400791985323[268] = 0;
   out_5721468400791985323[269] = 0;
   out_5721468400791985323[270] = 0;
   out_5721468400791985323[271] = 0;
   out_5721468400791985323[272] = 0;
   out_5721468400791985323[273] = 0;
   out_5721468400791985323[274] = 0;
   out_5721468400791985323[275] = 0;
   out_5721468400791985323[276] = 0;
   out_5721468400791985323[277] = 0;
   out_5721468400791985323[278] = 0;
   out_5721468400791985323[279] = 0;
   out_5721468400791985323[280] = 0;
   out_5721468400791985323[281] = 0;
   out_5721468400791985323[282] = 0;
   out_5721468400791985323[283] = 0;
   out_5721468400791985323[284] = 0;
   out_5721468400791985323[285] = 1;
   out_5721468400791985323[286] = 0;
   out_5721468400791985323[287] = 0;
   out_5721468400791985323[288] = 0;
   out_5721468400791985323[289] = 0;
   out_5721468400791985323[290] = 0;
   out_5721468400791985323[291] = 0;
   out_5721468400791985323[292] = 0;
   out_5721468400791985323[293] = 0;
   out_5721468400791985323[294] = 0;
   out_5721468400791985323[295] = 0;
   out_5721468400791985323[296] = 0;
   out_5721468400791985323[297] = 0;
   out_5721468400791985323[298] = 0;
   out_5721468400791985323[299] = 0;
   out_5721468400791985323[300] = 0;
   out_5721468400791985323[301] = 0;
   out_5721468400791985323[302] = 0;
   out_5721468400791985323[303] = 0;
   out_5721468400791985323[304] = 1;
   out_5721468400791985323[305] = 0;
   out_5721468400791985323[306] = 0;
   out_5721468400791985323[307] = 0;
   out_5721468400791985323[308] = 0;
   out_5721468400791985323[309] = 0;
   out_5721468400791985323[310] = 0;
   out_5721468400791985323[311] = 0;
   out_5721468400791985323[312] = 0;
   out_5721468400791985323[313] = 0;
   out_5721468400791985323[314] = 0;
   out_5721468400791985323[315] = 0;
   out_5721468400791985323[316] = 0;
   out_5721468400791985323[317] = 0;
   out_5721468400791985323[318] = 0;
   out_5721468400791985323[319] = 0;
   out_5721468400791985323[320] = 0;
   out_5721468400791985323[321] = 0;
   out_5721468400791985323[322] = 0;
   out_5721468400791985323[323] = 1;
}
void h_4(double *state, double *unused, double *out_3930645532014992442) {
   out_3930645532014992442[0] = state[6] + state[9];
   out_3930645532014992442[1] = state[7] + state[10];
   out_3930645532014992442[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_1979315791992089721) {
   out_1979315791992089721[0] = 0;
   out_1979315791992089721[1] = 0;
   out_1979315791992089721[2] = 0;
   out_1979315791992089721[3] = 0;
   out_1979315791992089721[4] = 0;
   out_1979315791992089721[5] = 0;
   out_1979315791992089721[6] = 1;
   out_1979315791992089721[7] = 0;
   out_1979315791992089721[8] = 0;
   out_1979315791992089721[9] = 1;
   out_1979315791992089721[10] = 0;
   out_1979315791992089721[11] = 0;
   out_1979315791992089721[12] = 0;
   out_1979315791992089721[13] = 0;
   out_1979315791992089721[14] = 0;
   out_1979315791992089721[15] = 0;
   out_1979315791992089721[16] = 0;
   out_1979315791992089721[17] = 0;
   out_1979315791992089721[18] = 0;
   out_1979315791992089721[19] = 0;
   out_1979315791992089721[20] = 0;
   out_1979315791992089721[21] = 0;
   out_1979315791992089721[22] = 0;
   out_1979315791992089721[23] = 0;
   out_1979315791992089721[24] = 0;
   out_1979315791992089721[25] = 1;
   out_1979315791992089721[26] = 0;
   out_1979315791992089721[27] = 0;
   out_1979315791992089721[28] = 1;
   out_1979315791992089721[29] = 0;
   out_1979315791992089721[30] = 0;
   out_1979315791992089721[31] = 0;
   out_1979315791992089721[32] = 0;
   out_1979315791992089721[33] = 0;
   out_1979315791992089721[34] = 0;
   out_1979315791992089721[35] = 0;
   out_1979315791992089721[36] = 0;
   out_1979315791992089721[37] = 0;
   out_1979315791992089721[38] = 0;
   out_1979315791992089721[39] = 0;
   out_1979315791992089721[40] = 0;
   out_1979315791992089721[41] = 0;
   out_1979315791992089721[42] = 0;
   out_1979315791992089721[43] = 0;
   out_1979315791992089721[44] = 1;
   out_1979315791992089721[45] = 0;
   out_1979315791992089721[46] = 0;
   out_1979315791992089721[47] = 1;
   out_1979315791992089721[48] = 0;
   out_1979315791992089721[49] = 0;
   out_1979315791992089721[50] = 0;
   out_1979315791992089721[51] = 0;
   out_1979315791992089721[52] = 0;
   out_1979315791992089721[53] = 0;
}
void h_10(double *state, double *unused, double *out_7266140832989971302) {
   out_7266140832989971302[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_7266140832989971302[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_7266140832989971302[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_5742539690088144716) {
   out_5742539690088144716[0] = 0;
   out_5742539690088144716[1] = 9.8100000000000005*cos(state[1]);
   out_5742539690088144716[2] = 0;
   out_5742539690088144716[3] = 0;
   out_5742539690088144716[4] = -state[8];
   out_5742539690088144716[5] = state[7];
   out_5742539690088144716[6] = 0;
   out_5742539690088144716[7] = state[5];
   out_5742539690088144716[8] = -state[4];
   out_5742539690088144716[9] = 0;
   out_5742539690088144716[10] = 0;
   out_5742539690088144716[11] = 0;
   out_5742539690088144716[12] = 1;
   out_5742539690088144716[13] = 0;
   out_5742539690088144716[14] = 0;
   out_5742539690088144716[15] = 1;
   out_5742539690088144716[16] = 0;
   out_5742539690088144716[17] = 0;
   out_5742539690088144716[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_5742539690088144716[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_5742539690088144716[20] = 0;
   out_5742539690088144716[21] = state[8];
   out_5742539690088144716[22] = 0;
   out_5742539690088144716[23] = -state[6];
   out_5742539690088144716[24] = -state[5];
   out_5742539690088144716[25] = 0;
   out_5742539690088144716[26] = state[3];
   out_5742539690088144716[27] = 0;
   out_5742539690088144716[28] = 0;
   out_5742539690088144716[29] = 0;
   out_5742539690088144716[30] = 0;
   out_5742539690088144716[31] = 1;
   out_5742539690088144716[32] = 0;
   out_5742539690088144716[33] = 0;
   out_5742539690088144716[34] = 1;
   out_5742539690088144716[35] = 0;
   out_5742539690088144716[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_5742539690088144716[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_5742539690088144716[38] = 0;
   out_5742539690088144716[39] = -state[7];
   out_5742539690088144716[40] = state[6];
   out_5742539690088144716[41] = 0;
   out_5742539690088144716[42] = state[4];
   out_5742539690088144716[43] = -state[3];
   out_5742539690088144716[44] = 0;
   out_5742539690088144716[45] = 0;
   out_5742539690088144716[46] = 0;
   out_5742539690088144716[47] = 0;
   out_5742539690088144716[48] = 0;
   out_5742539690088144716[49] = 0;
   out_5742539690088144716[50] = 1;
   out_5742539690088144716[51] = 0;
   out_5742539690088144716[52] = 0;
   out_5742539690088144716[53] = 1;
}
void h_13(double *state, double *unused, double *out_3951674911255468860) {
   out_3951674911255468860[0] = state[3];
   out_3951674911255468860[1] = state[4];
   out_3951674911255468860[2] = state[5];
}
void H_13(double *state, double *unused, double *out_5813071255294613745) {
   out_5813071255294613745[0] = 0;
   out_5813071255294613745[1] = 0;
   out_5813071255294613745[2] = 0;
   out_5813071255294613745[3] = 1;
   out_5813071255294613745[4] = 0;
   out_5813071255294613745[5] = 0;
   out_5813071255294613745[6] = 0;
   out_5813071255294613745[7] = 0;
   out_5813071255294613745[8] = 0;
   out_5813071255294613745[9] = 0;
   out_5813071255294613745[10] = 0;
   out_5813071255294613745[11] = 0;
   out_5813071255294613745[12] = 0;
   out_5813071255294613745[13] = 0;
   out_5813071255294613745[14] = 0;
   out_5813071255294613745[15] = 0;
   out_5813071255294613745[16] = 0;
   out_5813071255294613745[17] = 0;
   out_5813071255294613745[18] = 0;
   out_5813071255294613745[19] = 0;
   out_5813071255294613745[20] = 0;
   out_5813071255294613745[21] = 0;
   out_5813071255294613745[22] = 1;
   out_5813071255294613745[23] = 0;
   out_5813071255294613745[24] = 0;
   out_5813071255294613745[25] = 0;
   out_5813071255294613745[26] = 0;
   out_5813071255294613745[27] = 0;
   out_5813071255294613745[28] = 0;
   out_5813071255294613745[29] = 0;
   out_5813071255294613745[30] = 0;
   out_5813071255294613745[31] = 0;
   out_5813071255294613745[32] = 0;
   out_5813071255294613745[33] = 0;
   out_5813071255294613745[34] = 0;
   out_5813071255294613745[35] = 0;
   out_5813071255294613745[36] = 0;
   out_5813071255294613745[37] = 0;
   out_5813071255294613745[38] = 0;
   out_5813071255294613745[39] = 0;
   out_5813071255294613745[40] = 0;
   out_5813071255294613745[41] = 1;
   out_5813071255294613745[42] = 0;
   out_5813071255294613745[43] = 0;
   out_5813071255294613745[44] = 0;
   out_5813071255294613745[45] = 0;
   out_5813071255294613745[46] = 0;
   out_5813071255294613745[47] = 0;
   out_5813071255294613745[48] = 0;
   out_5813071255294613745[49] = 0;
   out_5813071255294613745[50] = 0;
   out_5813071255294613745[51] = 0;
   out_5813071255294613745[52] = 0;
   out_5813071255294613745[53] = 0;
}
void h_14(double *state, double *unused, double *out_8796949500966616670) {
   out_8796949500966616670[0] = state[6];
   out_8796949500966616670[1] = state[7];
   out_8796949500966616670[2] = state[8];
}
void H_14(double *state, double *unused, double *out_5062104224287462017) {
   out_5062104224287462017[0] = 0;
   out_5062104224287462017[1] = 0;
   out_5062104224287462017[2] = 0;
   out_5062104224287462017[3] = 0;
   out_5062104224287462017[4] = 0;
   out_5062104224287462017[5] = 0;
   out_5062104224287462017[6] = 1;
   out_5062104224287462017[7] = 0;
   out_5062104224287462017[8] = 0;
   out_5062104224287462017[9] = 0;
   out_5062104224287462017[10] = 0;
   out_5062104224287462017[11] = 0;
   out_5062104224287462017[12] = 0;
   out_5062104224287462017[13] = 0;
   out_5062104224287462017[14] = 0;
   out_5062104224287462017[15] = 0;
   out_5062104224287462017[16] = 0;
   out_5062104224287462017[17] = 0;
   out_5062104224287462017[18] = 0;
   out_5062104224287462017[19] = 0;
   out_5062104224287462017[20] = 0;
   out_5062104224287462017[21] = 0;
   out_5062104224287462017[22] = 0;
   out_5062104224287462017[23] = 0;
   out_5062104224287462017[24] = 0;
   out_5062104224287462017[25] = 1;
   out_5062104224287462017[26] = 0;
   out_5062104224287462017[27] = 0;
   out_5062104224287462017[28] = 0;
   out_5062104224287462017[29] = 0;
   out_5062104224287462017[30] = 0;
   out_5062104224287462017[31] = 0;
   out_5062104224287462017[32] = 0;
   out_5062104224287462017[33] = 0;
   out_5062104224287462017[34] = 0;
   out_5062104224287462017[35] = 0;
   out_5062104224287462017[36] = 0;
   out_5062104224287462017[37] = 0;
   out_5062104224287462017[38] = 0;
   out_5062104224287462017[39] = 0;
   out_5062104224287462017[40] = 0;
   out_5062104224287462017[41] = 0;
   out_5062104224287462017[42] = 0;
   out_5062104224287462017[43] = 0;
   out_5062104224287462017[44] = 1;
   out_5062104224287462017[45] = 0;
   out_5062104224287462017[46] = 0;
   out_5062104224287462017[47] = 0;
   out_5062104224287462017[48] = 0;
   out_5062104224287462017[49] = 0;
   out_5062104224287462017[50] = 0;
   out_5062104224287462017[51] = 0;
   out_5062104224287462017[52] = 0;
   out_5062104224287462017[53] = 0;
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

void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_4, H_4, NULL, in_z, in_R, in_ea, MAHA_THRESH_4);
}
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_10, H_10, NULL, in_z, in_R, in_ea, MAHA_THRESH_10);
}
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_13, H_13, NULL, in_z, in_R, in_ea, MAHA_THRESH_13);
}
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_14, H_14, NULL, in_z, in_R, in_ea, MAHA_THRESH_14);
}
void pose_err_fun(double *nom_x, double *delta_x, double *out_7822607448529633229) {
  err_fun(nom_x, delta_x, out_7822607448529633229);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_4249233298537579143) {
  inv_err_fun(nom_x, true_x, out_4249233298537579143);
}
void pose_H_mod_fun(double *state, double *out_3177263273874987825) {
  H_mod_fun(state, out_3177263273874987825);
}
void pose_f_fun(double *state, double dt, double *out_691237182481573573) {
  f_fun(state,  dt, out_691237182481573573);
}
void pose_F_fun(double *state, double dt, double *out_5721468400791985323) {
  F_fun(state,  dt, out_5721468400791985323);
}
void pose_h_4(double *state, double *unused, double *out_3930645532014992442) {
  h_4(state, unused, out_3930645532014992442);
}
void pose_H_4(double *state, double *unused, double *out_1979315791992089721) {
  H_4(state, unused, out_1979315791992089721);
}
void pose_h_10(double *state, double *unused, double *out_7266140832989971302) {
  h_10(state, unused, out_7266140832989971302);
}
void pose_H_10(double *state, double *unused, double *out_5742539690088144716) {
  H_10(state, unused, out_5742539690088144716);
}
void pose_h_13(double *state, double *unused, double *out_3951674911255468860) {
  h_13(state, unused, out_3951674911255468860);
}
void pose_H_13(double *state, double *unused, double *out_5813071255294613745) {
  H_13(state, unused, out_5813071255294613745);
}
void pose_h_14(double *state, double *unused, double *out_8796949500966616670) {
  h_14(state, unused, out_8796949500966616670);
}
void pose_H_14(double *state, double *unused, double *out_5062104224287462017) {
  H_14(state, unused, out_5062104224287462017);
}
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
}

const EKF pose = {
  .name = "pose",
  .kinds = { 4, 10, 13, 14 },
  .feature_kinds = {  },
  .f_fun = pose_f_fun,
  .F_fun = pose_F_fun,
  .err_fun = pose_err_fun,
  .inv_err_fun = pose_inv_err_fun,
  .H_mod_fun = pose_H_mod_fun,
  .predict = pose_predict,
  .hs = {
    { 4, pose_h_4 },
    { 10, pose_h_10 },
    { 13, pose_h_13 },
    { 14, pose_h_14 },
  },
  .Hs = {
    { 4, pose_H_4 },
    { 10, pose_H_10 },
    { 13, pose_H_13 },
    { 14, pose_H_14 },
  },
  .updates = {
    { 4, pose_update_4 },
    { 10, pose_update_10 },
    { 13, pose_update_13 },
    { 14, pose_update_14 },
  },
  .Hes = {
  },
  .sets = {
  },
  .extra_routines = {
  },
};

ekf_lib_init(pose)
