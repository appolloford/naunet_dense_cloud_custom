#include <math.h>
/* */
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>  // access to dense SUNMatrix
/* */
#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_ode.h"
#include "naunet_physics.h"

#ifdef USE_CUDA
#define NVEC_CUDA_CONTENT(x) ((N_VectorContent_Cuda)(x->content))
#define NVEC_CUDA_STREAM(x) (NVEC_CUDA_CONTENT(x)->stream_exec_policy->stream())
#define NVEC_CUDA_BLOCKSIZE(x) \
    (NVEC_CUDA_CONTENT(x)->stream_exec_policy->blockSize())
#define NVEC_CUDA_GRIDSIZE(x, n) \
    (NVEC_CUDA_CONTENT(x)->stream_exec_policy->gridSize(n))
#endif

/* */

int Jac(realtype t, N_Vector u, N_Vector fu, SUNMatrix jmatrix, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    /* */
    realtype *y            = N_VGetArrayPointer(u);
    NaunetData *u_data     = (NaunetData *)user_data;
    // clang-format off
    realtype nH = u_data->nH;
    realtype Tgas = u_data->Tgas;
    realtype zeta = u_data->zeta;
    realtype Av = u_data->Av;
    realtype omega = u_data->omega;
    realtype G0 = u_data->G0;
    realtype gdens = u_data->gdens;
    realtype rG = u_data->rG;
    realtype sites = u_data->sites;
    realtype fr = u_data->fr;
    realtype opt_crd = u_data->opt_crd;
    realtype opt_uvd = u_data->opt_uvd;
    realtype opt_h2d = u_data->opt_h2d;
    realtype eb_crd = u_data->eb_crd;
    realtype eb_uvd = u_data->eb_uvd;
    realtype eb_h2d = u_data->eb_h2d;
    realtype crdeseff = u_data->crdeseff;
    realtype uvcreff = u_data->uvcreff;
    realtype h2deseff = u_data->h2deseff;
    realtype opt_thd = u_data->opt_thd;
    realtype ksp = u_data->ksp;
    
        
#if (NHEATPROCS || NCOOLPROCS)
    if (mu < 0) mu = GetMu(y);
    if (gamma < 0) gamma = GetGamma(y);
#endif

    // clang-format on

    realtype k[NREACTIONS] = {0.0};
    EvalRates(k, y, u_data);

#if NHEATPROCS
    realtype kh[NHEATPROCS] = {0.0};
    EvalHeatingRates(kh, y, u_data);
#endif

#if NCOOLPROCS
    realtype kc[NCOOLPROCS] = {0.0};
    EvalCoolingRates(kc, y, u_data);
#endif

    SUNMatZero(jmatrix);

    // clang-format off
    IJth(jmatrix, 0, 0) = 0.0 - k[1339] - k[1340] - k[1341] - k[1342];
    IJth(jmatrix, 0, 23) = 0.0 + k[1298];
    IJth(jmatrix, 1, 1) = 0.0 - k[1327] - k[1328] - k[1329] - k[1330];
    IJth(jmatrix, 1, 63) = 0.0 + k[1305];
    IJth(jmatrix, 2, 2) = 0.0 - k[1387] - k[1388] - k[1389] - k[1390];
    IJth(jmatrix, 2, 39) = 0.0 + k[1293];
    IJth(jmatrix, 3, 3) = 0.0 - k[1379] - k[1380] - k[1381] - k[1382];
    IJth(jmatrix, 4, 4) = 0.0 - k[1331] - k[1332] - k[1333] - k[1334];
    IJth(jmatrix, 4, 68) = 0.0 + k[1275];
    IJth(jmatrix, 4, 111) = 0.0 + k[1251];
    IJth(jmatrix, 5, 5) = 0.0 - k[1375] - k[1376] - k[1377] - k[1378];
    IJth(jmatrix, 5, 28) = 0.0 + k[1224];
    IJth(jmatrix, 5, 42) = 0.0 + k[1225];
    IJth(jmatrix, 6, 6) = 0.0 - k[1319] - k[1320] - k[1321] - k[1322];
    IJth(jmatrix, 6, 53) = 0.0 + k[1300];
    IJth(jmatrix, 6, 54) = 0.0 + k[1299];
    IJth(jmatrix, 7, 7) = 0.0 - k[1343] - k[1344] - k[1345] - k[1346];
    IJth(jmatrix, 7, 84) = 0.0 + k[1277];
    IJth(jmatrix, 7, 101) = 0.0 + k[1255];
    IJth(jmatrix, 8, 8) = 0.0 - k[1355] - k[1356] - k[1357] - k[1358];
    IJth(jmatrix, 8, 78) = 0.0 + k[1270];
    IJth(jmatrix, 8, 104) = 0.0 + k[1292];
    IJth(jmatrix, 9, 9) = 0.0 - k[1367] - k[1368] - k[1369] - k[1370];
    IJth(jmatrix, 9, 37) = 0.0 + k[1303];
    IJth(jmatrix, 9, 60) = 0.0 + k[1297];
    IJth(jmatrix, 10, 10) = 0.0 - k[1371] - k[1372] - k[1373] - k[1374];
    IJth(jmatrix, 10, 36) = 0.0 + k[1241];
    IJth(jmatrix, 10, 38) = 0.0 + k[1239];
    IJth(jmatrix, 11, 11) = 0.0 - k[1395] - k[1396] - k[1397] - k[1398];
    IJth(jmatrix, 11, 30) = 0.0 + k[1242];
    IJth(jmatrix, 11, 32) = 0.0 + k[1240];
    IJth(jmatrix, 12, 12) = 0.0 - k[1399] - k[1400] - k[1401] - k[1402];
    IJth(jmatrix, 12, 22) = 0.0 + k[1244];
    IJth(jmatrix, 12, 33) = 0.0 + k[1243];
    IJth(jmatrix, 13, 13) = 0.0 - k[1359] - k[1360] - k[1361] - k[1362];
    IJth(jmatrix, 13, 52) = 0.0 + k[1223];
    IJth(jmatrix, 13, 66) = 0.0 + k[1249];
    IJth(jmatrix, 13, 111) = 0.0 + k[1304];
    IJth(jmatrix, 14, 14) = 0.0 - k[1383] - k[1384] - k[1385] - k[1386];
    IJth(jmatrix, 14, 50) = 0.0 + k[1287];
    IJth(jmatrix, 14, 72) = 0.0 + k[1258];
    IJth(jmatrix, 15, 15) = 0.0 - k[1391] - k[1392] - k[1393] - k[1394];
    IJth(jmatrix, 15, 26) = 0.0 + k[1247];
    IJth(jmatrix, 15, 44) = 0.0 + k[1246];
    IJth(jmatrix, 15, 49) = 0.0 + k[1245];
    IJth(jmatrix, 15, 56) = 0.0 + k[1229];
    IJth(jmatrix, 16, 16) = 0.0 - k[1351] - k[1352] - k[1353] - k[1354];
    IJth(jmatrix, 16, 25) = 0.0 + k[1296];
    IJth(jmatrix, 16, 51) = 0.0 + k[1294];
    IJth(jmatrix, 16, 64) = 0.0 + k[1295];
    IJth(jmatrix, 17, 17) = 0.0 - k[1335] - k[1336] - k[1337] - k[1338];
    IJth(jmatrix, 17, 59) = 0.0 + k[1302];
    IJth(jmatrix, 17, 65) = 0.0 + k[1271];
    IJth(jmatrix, 17, 86) = 0.0 + k[1262];
    IJth(jmatrix, 18, 18) = 0.0 - k[1347] - k[1348] - k[1349] - k[1350];
    IJth(jmatrix, 18, 29) = 0.0 + k[1226];
    IJth(jmatrix, 18, 91) = 0.0 + k[1284];
    IJth(jmatrix, 18, 95) = 0.0 + k[1252];
    IJth(jmatrix, 18, 96) = 0.0 + k[1261];
    IJth(jmatrix, 18, 107) = 0.0 + k[1281];
    IJth(jmatrix, 18, 111) = 0.0 + k[1227];
    IJth(jmatrix, 19, 19) = 0.0 - k[1323] - k[1324] - k[1325] - k[1326];
    IJth(jmatrix, 19, 57) = 0.0 + k[1276];
    IJth(jmatrix, 19, 58) = 0.0 + k[1301];
    IJth(jmatrix, 19, 75) = 0.0 + k[1282];
    IJth(jmatrix, 19, 88) = 0.0 + k[1266];
    IJth(jmatrix, 19, 94) = 0.0 + k[1263];
    IJth(jmatrix, 20, 20) = 0.0 - k[1315] - k[1316] - k[1317] - k[1318];
    IJth(jmatrix, 20, 74) = 0.0 + k[1269];
    IJth(jmatrix, 20, 82) = 0.0 + k[1280];
    IJth(jmatrix, 20, 85) = 0.0 + k[1286];
    IJth(jmatrix, 20, 93) = 0.0 + k[1274];
    IJth(jmatrix, 20, 103) = 0.0 + k[1254];
    IJth(jmatrix, 20, 108) = 0.0 + k[1257];
    IJth(jmatrix, 20, 109) = 0.0 + k[1291];
    IJth(jmatrix, 21, 21) = 0.0 - k[1311] - k[1312] - k[1313] - k[1314];
    IJth(jmatrix, 21, 70) = 0.0 + k[1267];
    IJth(jmatrix, 21, 73) = 0.0 + k[1268];
    IJth(jmatrix, 21, 76) = 0.0 + k[1279];
    IJth(jmatrix, 21, 77) = 0.0 + k[1273];
    IJth(jmatrix, 21, 80) = 0.0 + k[1289];
    IJth(jmatrix, 21, 83) = 0.0 + k[1283];
    IJth(jmatrix, 21, 92) = 0.0 + k[1265];
    IJth(jmatrix, 21, 102) = 0.0 + k[1290];
    IJth(jmatrix, 22, 22) = 0.0 - k[351]*y[IDX_EM] - k[1244];
    IJth(jmatrix, 22, 33) = 0.0 + k[23]*y[IDX_CII] + k[93]*y[IDX_HII];
    IJth(jmatrix, 22, 87) = 0.0 + k[23]*y[IDX_SiC3I];
    IJth(jmatrix, 22, 106) = 0.0 + k[93]*y[IDX_SiC3I];
    IJth(jmatrix, 22, 110) = 0.0 - k[351]*y[IDX_SiC3II];
    IJth(jmatrix, 23, 0) = 0.0 + k[1339] + k[1340] + k[1341] + k[1342];
    IJth(jmatrix, 23, 23) = 0.0 - k[254] - k[973]*y[IDX_HI] - k[1013]*y[IDX_NI] -
                        k[1060]*y[IDX_OI] - k[1135] - k[1298];
    IJth(jmatrix, 23, 71) = 0.0 + k[1008]*y[IDX_NI];
    IJth(jmatrix, 23, 102) = 0.0 + k[1008]*y[IDX_CH3I] - k[1013]*y[IDX_H2CNI];
    IJth(jmatrix, 23, 109) = 0.0 - k[1060]*y[IDX_H2CNI];
    IJth(jmatrix, 23, 113) = 0.0 - k[973]*y[IDX_H2CNI];
    IJth(jmatrix, 24, 24) = 0.0 - k[1307] - k[1308] - k[1309] - k[1310];
    IJth(jmatrix, 24, 55) = 0.0 + k[1288];
    IJth(jmatrix, 24, 67) = 0.0 + k[1260];
    IJth(jmatrix, 24, 71) = 0.0 + k[1259];
    IJth(jmatrix, 24, 79) = 0.0 + k[1285];
    IJth(jmatrix, 24, 81) = 0.0 + k[1278];
    IJth(jmatrix, 24, 87) = 0.0 + k[1264];
    IJth(jmatrix, 24, 89) = 0.0 + k[1272];
    IJth(jmatrix, 24, 90) = 0.0 + k[1256];
    IJth(jmatrix, 24, 98) = 0.0 + k[1253];
    IJth(jmatrix, 24, 105) = 0.0 + k[1250];
    IJth(jmatrix, 25, 25) = 0.0 - k[310]*y[IDX_EM] - k[311]*y[IDX_EM] - k[1296];
    IJth(jmatrix, 25, 51) = 0.0 + k[590]*y[IDX_H3II];
    IJth(jmatrix, 25, 76) = 0.0 + k[777]*y[IDX_O2I];
    IJth(jmatrix, 25, 99) = 0.0 + k[590]*y[IDX_HNOI];
    IJth(jmatrix, 25, 104) = 0.0 + k[777]*y[IDX_NH2II];
    IJth(jmatrix, 25, 110) = 0.0 - k[310]*y[IDX_H2NOII] - k[311]*y[IDX_H2NOII];
    IJth(jmatrix, 26, 15) = 0.0 + k[1391] + k[1392] + k[1393] + k[1394];
    IJth(jmatrix, 26, 26) = 0.0 - k[257] - k[499]*y[IDX_HII] - k[675]*y[IDX_HeII] -
                        k[1143] - k[1144] - k[1247];
    IJth(jmatrix, 26, 48) = 0.0 + k[1087]*y[IDX_OI];
    IJth(jmatrix, 26, 97) = 0.0 - k[675]*y[IDX_H2SiOI];
    IJth(jmatrix, 26, 106) = 0.0 - k[499]*y[IDX_H2SiOI];
    IJth(jmatrix, 26, 109) = 0.0 + k[1087]*y[IDX_SiH3I];
    IJth(jmatrix, 27, 27) = 0.0 - k[336]*y[IDX_EM] - k[537]*y[IDX_H2I] -
                        k[618]*y[IDX_HI];
    IJth(jmatrix, 27, 69) = 0.0 + k[520]*y[IDX_HeI];
    IJth(jmatrix, 27, 96) = 0.0 + k[681]*y[IDX_HeII];
    IJth(jmatrix, 27, 97) = 0.0 + k[681]*y[IDX_HCOI];
    IJth(jmatrix, 27, 100) = 0.0 + k[520]*y[IDX_H2II] + k[1198]*y[IDX_HII];
    IJth(jmatrix, 27, 106) = 0.0 + k[1198]*y[IDX_HeI];
    IJth(jmatrix, 27, 110) = 0.0 - k[336]*y[IDX_HeHII];
    IJth(jmatrix, 27, 112) = 0.0 - k[537]*y[IDX_HeHII];
    IJth(jmatrix, 27, 113) = 0.0 - k[618]*y[IDX_HeHII];
    IJth(jmatrix, 28, 5) = 0.0 + k[1375] + k[1376] + k[1377] + k[1378];
    IJth(jmatrix, 28, 28) = 0.0 - k[263] - k[502]*y[IDX_HII] - k[1004]*y[IDX_CI] -
                        k[1152] - k[1224];
    IJth(jmatrix, 28, 90) = 0.0 + k[888]*y[IDX_NOI];
    IJth(jmatrix, 28, 101) = 0.0 + k[888]*y[IDX_CH2I];
    IJth(jmatrix, 28, 105) = 0.0 - k[1004]*y[IDX_HNCOI];
    IJth(jmatrix, 28, 106) = 0.0 - k[502]*y[IDX_HNCOI];
    IJth(jmatrix, 29, 29) = 0.0 - k[5]*y[IDX_H2I] - k[335]*y[IDX_EM] - k[1226];
    IJth(jmatrix, 29, 68) = 0.0 + k[533]*y[IDX_H2I];
    IJth(jmatrix, 29, 87) = 0.0 + k[371]*y[IDX_H2OI];
    IJth(jmatrix, 29, 99) = 0.0 + k[584]*y[IDX_COI];
    IJth(jmatrix, 29, 108) = 0.0 + k[371]*y[IDX_CII];
    IJth(jmatrix, 29, 110) = 0.0 - k[335]*y[IDX_HOCII];
    IJth(jmatrix, 29, 111) = 0.0 + k[584]*y[IDX_H3II];
    IJth(jmatrix, 29, 112) = 0.0 - k[5]*y[IDX_HOCII] + k[533]*y[IDX_COII];
    IJth(jmatrix, 30, 30) = 0.0 - k[350]*y[IDX_EM] - k[1242];
    IJth(jmatrix, 30, 32) = 0.0 + k[22]*y[IDX_CII] + k[92]*y[IDX_HII];
    IJth(jmatrix, 30, 33) = 0.0 + k[700]*y[IDX_HeII];
    IJth(jmatrix, 30, 87) = 0.0 + k[22]*y[IDX_SiC2I];
    IJth(jmatrix, 30, 97) = 0.0 + k[700]*y[IDX_SiC3I];
    IJth(jmatrix, 30, 106) = 0.0 + k[92]*y[IDX_SiC2I];
    IJth(jmatrix, 30, 110) = 0.0 - k[350]*y[IDX_SiC2II];
    IJth(jmatrix, 31, 31) = 0.0 - k[1363] - k[1364] - k[1365] - k[1366];
    IJth(jmatrix, 31, 34) = 0.0 + k[1248];
    IJth(jmatrix, 31, 35) = 0.0 + k[1238];
    IJth(jmatrix, 31, 40) = 0.0 + k[1236];
    IJth(jmatrix, 31, 41) = 0.0 + k[1234];
    IJth(jmatrix, 31, 43) = 0.0 + k[1233];
    IJth(jmatrix, 31, 45) = 0.0 + k[1232];
    IJth(jmatrix, 31, 46) = 0.0 + k[1237];
    IJth(jmatrix, 31, 47) = 0.0 + k[1230];
    IJth(jmatrix, 31, 48) = 0.0 + k[1235];
    IJth(jmatrix, 31, 61) = 0.0 + k[1231];
    IJth(jmatrix, 31, 62) = 0.0 + k[1228];
    IJth(jmatrix, 32, 11) = 0.0 + k[1395] + k[1396] + k[1397] + k[1398];
    IJth(jmatrix, 32, 22) = 0.0 + k[351]*y[IDX_EM];
    IJth(jmatrix, 32, 32) = 0.0 - k[22]*y[IDX_CII] - k[92]*y[IDX_HII] - k[286] -
                        k[1081]*y[IDX_OI] - k[1240];
    IJth(jmatrix, 32, 33) = 0.0 + k[287] + k[1082]*y[IDX_OI] + k[1177];
    IJth(jmatrix, 32, 87) = 0.0 - k[22]*y[IDX_SiC2I];
    IJth(jmatrix, 32, 106) = 0.0 - k[92]*y[IDX_SiC2I];
    IJth(jmatrix, 32, 109) = 0.0 - k[1081]*y[IDX_SiC2I] + k[1082]*y[IDX_SiC3I];
    IJth(jmatrix, 32, 110) = 0.0 + k[351]*y[IDX_SiC3II];
    IJth(jmatrix, 33, 12) = 0.0 + k[1399] + k[1400] + k[1401] + k[1402];
    IJth(jmatrix, 33, 33) = 0.0 - k[23]*y[IDX_CII] - k[93]*y[IDX_HII] - k[287] -
                        k[700]*y[IDX_HeII] - k[1082]*y[IDX_OI] - k[1177] -
                        k[1243];
    IJth(jmatrix, 33, 87) = 0.0 - k[23]*y[IDX_SiC3I];
    IJth(jmatrix, 33, 97) = 0.0 - k[700]*y[IDX_SiC3I];
    IJth(jmatrix, 33, 106) = 0.0 - k[93]*y[IDX_SiC3I];
    IJth(jmatrix, 33, 109) = 0.0 - k[1082]*y[IDX_SiC3I];
    IJth(jmatrix, 34, 34) = 0.0 - k[360]*y[IDX_EM] - k[361]*y[IDX_EM] -
                        k[575]*y[IDX_H2OI] - k[1248];
    IJth(jmatrix, 34, 35) = 0.0 + k[546]*y[IDX_H2I];
    IJth(jmatrix, 34, 40) = 0.0 + k[1204]*y[IDX_H2I];
    IJth(jmatrix, 34, 46) = 0.0 + k[604]*y[IDX_H3II] + k[638]*y[IDX_HCOII];
    IJth(jmatrix, 34, 99) = 0.0 + k[604]*y[IDX_SiH4I];
    IJth(jmatrix, 34, 107) = 0.0 + k[638]*y[IDX_SiH4I];
    IJth(jmatrix, 34, 108) = 0.0 - k[575]*y[IDX_SiH5II];
    IJth(jmatrix, 34, 110) = 0.0 - k[360]*y[IDX_SiH5II] - k[361]*y[IDX_SiH5II];
    IJth(jmatrix, 34, 112) = 0.0 + k[546]*y[IDX_SiH4II] + k[1204]*y[IDX_SiH3II];
    IJth(jmatrix, 35, 35) = 0.0 - k[358]*y[IDX_EM] - k[359]*y[IDX_EM] -
                        k[489]*y[IDX_COI] - k[546]*y[IDX_H2I] -
                        k[574]*y[IDX_H2OI] - k[1238];
    IJth(jmatrix, 35, 46) = 0.0 + k[97]*y[IDX_HII];
    IJth(jmatrix, 35, 48) = 0.0 + k[603]*y[IDX_H3II];
    IJth(jmatrix, 35, 99) = 0.0 + k[603]*y[IDX_SiH3I];
    IJth(jmatrix, 35, 106) = 0.0 + k[97]*y[IDX_SiH4I];
    IJth(jmatrix, 35, 108) = 0.0 - k[574]*y[IDX_SiH4II];
    IJth(jmatrix, 35, 110) = 0.0 - k[358]*y[IDX_SiH4II] - k[359]*y[IDX_SiH4II];
    IJth(jmatrix, 35, 111) = 0.0 - k[489]*y[IDX_SiH4II];
    IJth(jmatrix, 35, 112) = 0.0 - k[546]*y[IDX_SiH4II];
    IJth(jmatrix, 36, 36) = 0.0 - k[349]*y[IDX_EM] - k[744]*y[IDX_NI] -
                        k[831]*y[IDX_OI] - k[1241];
    IJth(jmatrix, 36, 38) = 0.0 + k[24]*y[IDX_CII] + k[94]*y[IDX_HII];
    IJth(jmatrix, 36, 43) = 0.0 + k[380]*y[IDX_CII];
    IJth(jmatrix, 36, 45) = 0.0 + k[394]*y[IDX_CI];
    IJth(jmatrix, 36, 47) = 0.0 + k[381]*y[IDX_CII];
    IJth(jmatrix, 36, 61) = 0.0 + k[476]*y[IDX_CHI];
    IJth(jmatrix, 36, 87) = 0.0 + k[24]*y[IDX_SiCI] + k[380]*y[IDX_SiH2I] +
                        k[381]*y[IDX_SiHI];
    IJth(jmatrix, 36, 98) = 0.0 + k[476]*y[IDX_SiII];
    IJth(jmatrix, 36, 102) = 0.0 - k[744]*y[IDX_SiCII];
    IJth(jmatrix, 36, 105) = 0.0 + k[394]*y[IDX_SiHII];
    IJth(jmatrix, 36, 106) = 0.0 + k[94]*y[IDX_SiCI];
    IJth(jmatrix, 36, 109) = 0.0 - k[831]*y[IDX_SiCII];
    IJth(jmatrix, 36, 110) = 0.0 - k[349]*y[IDX_SiCII];
    IJth(jmatrix, 37, 9) = 0.0 + k[1367] + k[1368] + k[1369] + k[1370];
    IJth(jmatrix, 37, 37) = 0.0 - k[281] - k[914]*y[IDX_CH3I] - k[937]*y[IDX_CHI] -
                        k[938]*y[IDX_CHI] - k[954]*y[IDX_COI] - k[990]*y[IDX_HI]
                        - k[991]*y[IDX_HI] - k[992]*y[IDX_HI] -
                        k[1003]*y[IDX_HCOI] - k[1024]*y[IDX_NI] -
                        k[1077]*y[IDX_OI] - k[1100]*y[IDX_OHI] - k[1170] -
                        k[1171] - k[1303];
    IJth(jmatrix, 37, 67) = 0.0 + k[921]*y[IDX_O2I];
    IJth(jmatrix, 37, 71) = 0.0 + k[913]*y[IDX_O2I] - k[914]*y[IDX_O2HI];
    IJth(jmatrix, 37, 91) = 0.0 + k[549]*y[IDX_O2I];
    IJth(jmatrix, 37, 96) = 0.0 + k[1002]*y[IDX_O2I] - k[1003]*y[IDX_O2HI];
    IJth(jmatrix, 37, 98) = 0.0 - k[937]*y[IDX_O2HI] - k[938]*y[IDX_O2HI];
    IJth(jmatrix, 37, 102) = 0.0 - k[1024]*y[IDX_O2HI];
    IJth(jmatrix, 37, 103) = 0.0 - k[1100]*y[IDX_O2HI];
    IJth(jmatrix, 37, 104) = 0.0 + k[549]*y[IDX_H2COII] + k[913]*y[IDX_CH3I] +
                        k[921]*y[IDX_CH4I] + k[963]*y[IDX_H2I] +
                        k[1002]*y[IDX_HCOI];
    IJth(jmatrix, 37, 109) = 0.0 - k[1077]*y[IDX_O2HI];
    IJth(jmatrix, 37, 111) = 0.0 - k[954]*y[IDX_O2HI];
    IJth(jmatrix, 37, 112) = 0.0 + k[963]*y[IDX_O2I];
    IJth(jmatrix, 37, 113) = 0.0 - k[990]*y[IDX_O2HI] - k[991]*y[IDX_O2HI] -
                        k[992]*y[IDX_O2HI];
    IJth(jmatrix, 38, 10) = 0.0 + k[1371] + k[1372] + k[1373] + k[1374];
    IJth(jmatrix, 38, 30) = 0.0 + k[350]*y[IDX_EM];
    IJth(jmatrix, 38, 32) = 0.0 + k[286] + k[1081]*y[IDX_OI];
    IJth(jmatrix, 38, 38) = 0.0 - k[24]*y[IDX_CII] - k[94]*y[IDX_HII] - k[288] -
                        k[701]*y[IDX_HeII] - k[702]*y[IDX_HeII] -
                        k[1027]*y[IDX_NI] - k[1083]*y[IDX_OI] -
                        k[1084]*y[IDX_OI] - k[1178] - k[1239];
    IJth(jmatrix, 38, 47) = 0.0 + k[877]*y[IDX_CI];
    IJth(jmatrix, 38, 87) = 0.0 - k[24]*y[IDX_SiCI];
    IJth(jmatrix, 38, 97) = 0.0 - k[701]*y[IDX_SiCI] - k[702]*y[IDX_SiCI];
    IJth(jmatrix, 38, 102) = 0.0 - k[1027]*y[IDX_SiCI];
    IJth(jmatrix, 38, 105) = 0.0 + k[877]*y[IDX_SiHI];
    IJth(jmatrix, 38, 106) = 0.0 - k[94]*y[IDX_SiCI];
    IJth(jmatrix, 38, 109) = 0.0 + k[1081]*y[IDX_SiC2I] - k[1083]*y[IDX_SiCI] -
                        k[1084]*y[IDX_SiCI];
    IJth(jmatrix, 38, 110) = 0.0 + k[350]*y[IDX_SiC2II];
    IJth(jmatrix, 39, 2) = 0.0 + k[1387] + k[1388] + k[1389] + k[1390];
    IJth(jmatrix, 39, 39) = 0.0 - k[276] - k[504]*y[IDX_HII] - k[595]*y[IDX_H3II] -
                        k[818]*y[IDX_OII] - k[885]*y[IDX_CH2I] -
                        k[909]*y[IDX_CH3I] - k[945]*y[IDX_CNI] -
                        k[952]*y[IDX_COI] - k[986]*y[IDX_HI] - k[1019]*y[IDX_NI]
                        - k[1020]*y[IDX_NI] - k[1021]*y[IDX_NI] -
                        k[1041]*y[IDX_NHI] - k[1075]*y[IDX_OI] - k[1164] -
                        k[1293];
    IJth(jmatrix, 39, 42) = 0.0 + k[1055]*y[IDX_O2I];
    IJth(jmatrix, 39, 51) = 0.0 + k[1068]*y[IDX_OI];
    IJth(jmatrix, 39, 71) = 0.0 - k[909]*y[IDX_NO2I];
    IJth(jmatrix, 39, 74) = 0.0 - k[818]*y[IDX_NO2I];
    IJth(jmatrix, 39, 90) = 0.0 - k[885]*y[IDX_NO2I];
    IJth(jmatrix, 39, 92) = 0.0 - k[1041]*y[IDX_NO2I];
    IJth(jmatrix, 39, 94) = 0.0 - k[945]*y[IDX_NO2I];
    IJth(jmatrix, 39, 99) = 0.0 - k[595]*y[IDX_NO2I];
    IJth(jmatrix, 39, 101) = 0.0 + k[1052]*y[IDX_O2I] + k[1099]*y[IDX_OHI];
    IJth(jmatrix, 39, 102) = 0.0 - k[1019]*y[IDX_NO2I] - k[1020]*y[IDX_NO2I] -
                        k[1021]*y[IDX_NO2I];
    IJth(jmatrix, 39, 103) = 0.0 + k[1099]*y[IDX_NOI];
    IJth(jmatrix, 39, 104) = 0.0 + k[1052]*y[IDX_NOI] + k[1055]*y[IDX_OCNI];
    IJth(jmatrix, 39, 106) = 0.0 - k[504]*y[IDX_NO2I];
    IJth(jmatrix, 39, 109) = 0.0 + k[1068]*y[IDX_HNOI] - k[1075]*y[IDX_NO2I];
    IJth(jmatrix, 39, 111) = 0.0 - k[952]*y[IDX_NO2I];
    IJth(jmatrix, 39, 113) = 0.0 - k[986]*y[IDX_NO2I];
    IJth(jmatrix, 40, 40) = 0.0 - k[356]*y[IDX_EM] - k[357]*y[IDX_EM] -
                        k[834]*y[IDX_OI] - k[1204]*y[IDX_H2I] - k[1236];
    IJth(jmatrix, 40, 43) = 0.0 + k[602]*y[IDX_H3II] + k[611]*y[IDX_H3OII] +
                        k[637]*y[IDX_HCOII];
    IJth(jmatrix, 40, 45) = 0.0 + k[1203]*y[IDX_H2I];
    IJth(jmatrix, 40, 46) = 0.0 + k[446]*y[IDX_CH3II] + k[507]*y[IDX_HII];
    IJth(jmatrix, 40, 48) = 0.0 + k[26]*y[IDX_CII] + k[96]*y[IDX_HII] + k[1183];
    IJth(jmatrix, 40, 79) = 0.0 + k[446]*y[IDX_SiH4I];
    IJth(jmatrix, 40, 85) = 0.0 + k[611]*y[IDX_SiH2I];
    IJth(jmatrix, 40, 87) = 0.0 + k[26]*y[IDX_SiH3I];
    IJth(jmatrix, 40, 99) = 0.0 + k[602]*y[IDX_SiH2I];
    IJth(jmatrix, 40, 106) = 0.0 + k[96]*y[IDX_SiH3I] + k[507]*y[IDX_SiH4I];
    IJth(jmatrix, 40, 107) = 0.0 + k[637]*y[IDX_SiH2I];
    IJth(jmatrix, 40, 109) = 0.0 - k[834]*y[IDX_SiH3II];
    IJth(jmatrix, 40, 110) = 0.0 - k[356]*y[IDX_SiH3II] - k[357]*y[IDX_SiH3II];
    IJth(jmatrix, 40, 112) = 0.0 + k[1203]*y[IDX_SiHII] - k[1204]*y[IDX_SiH3II];
    IJth(jmatrix, 41, 41) = 0.0 - k[353]*y[IDX_EM] - k[354]*y[IDX_EM] -
                        k[355]*y[IDX_EM] - k[833]*y[IDX_OI] - k[862]*y[IDX_O2I]
                        - k[1234];
    IJth(jmatrix, 41, 43) = 0.0 + k[25]*y[IDX_CII] + k[95]*y[IDX_HII] + k[1180];
    IJth(jmatrix, 41, 47) = 0.0 + k[605]*y[IDX_H3II] + k[612]*y[IDX_H3OII] +
                        k[639]*y[IDX_HCOII] + k[849]*y[IDX_OHII];
    IJth(jmatrix, 41, 48) = 0.0 + k[506]*y[IDX_HII] + k[706]*y[IDX_HeII];
    IJth(jmatrix, 41, 61) = 0.0 + k[1202]*y[IDX_H2I];
    IJth(jmatrix, 41, 85) = 0.0 + k[612]*y[IDX_SiHI];
    IJth(jmatrix, 41, 87) = 0.0 + k[25]*y[IDX_SiH2I];
    IJth(jmatrix, 41, 93) = 0.0 + k[849]*y[IDX_SiHI];
    IJth(jmatrix, 41, 97) = 0.0 + k[706]*y[IDX_SiH3I];
    IJth(jmatrix, 41, 99) = 0.0 + k[605]*y[IDX_SiHI];
    IJth(jmatrix, 41, 104) = 0.0 - k[862]*y[IDX_SiH2II];
    IJth(jmatrix, 41, 106) = 0.0 + k[95]*y[IDX_SiH2I] + k[506]*y[IDX_SiH3I];
    IJth(jmatrix, 41, 107) = 0.0 + k[639]*y[IDX_SiHI];
    IJth(jmatrix, 41, 109) = 0.0 - k[833]*y[IDX_SiH2II];
    IJth(jmatrix, 41, 110) = 0.0 - k[353]*y[IDX_SiH2II] - k[354]*y[IDX_SiH2II] -
                        k[355]*y[IDX_SiH2II];
    IJth(jmatrix, 41, 112) = 0.0 + k[1202]*y[IDX_SiII];
    IJth(jmatrix, 42, 23) = 0.0 + k[1060]*y[IDX_OI];
    IJth(jmatrix, 42, 39) = 0.0 + k[945]*y[IDX_CNI];
    IJth(jmatrix, 42, 42) = 0.0 - k[283] - k[378]*y[IDX_CII] - k[697]*y[IDX_HeII] -
                        k[698]*y[IDX_HeII] - k[874]*y[IDX_CI] - k[993]*y[IDX_HI]
                        - k[994]*y[IDX_HI] - k[995]*y[IDX_HI] -
                        k[1053]*y[IDX_NOI] - k[1054]*y[IDX_O2I] -
                        k[1055]*y[IDX_O2I] - k[1078]*y[IDX_OI] -
                        k[1079]*y[IDX_OI] - k[1172] - k[1225];
    IJth(jmatrix, 42, 87) = 0.0 - k[378]*y[IDX_OCNI];
    IJth(jmatrix, 42, 88) = 0.0 + k[1065]*y[IDX_OI];
    IJth(jmatrix, 42, 94) = 0.0 + k[945]*y[IDX_NO2I] + k[947]*y[IDX_NOI] +
                        k[949]*y[IDX_O2I] + k[1091]*y[IDX_OHI];
    IJth(jmatrix, 42, 96) = 0.0 + k[1016]*y[IDX_NI];
    IJth(jmatrix, 42, 97) = 0.0 - k[697]*y[IDX_OCNI] - k[698]*y[IDX_OCNI];
    IJth(jmatrix, 42, 98) = 0.0 + k[932]*y[IDX_NOI];
    IJth(jmatrix, 42, 101) = 0.0 + k[932]*y[IDX_CHI] + k[947]*y[IDX_CNI] -
                        k[1053]*y[IDX_OCNI];
    IJth(jmatrix, 42, 102) = 0.0 + k[1016]*y[IDX_HCOI];
    IJth(jmatrix, 42, 103) = 0.0 + k[1091]*y[IDX_CNI];
    IJth(jmatrix, 42, 104) = 0.0 + k[949]*y[IDX_CNI] - k[1054]*y[IDX_OCNI] -
                        k[1055]*y[IDX_OCNI];
    IJth(jmatrix, 42, 105) = 0.0 - k[874]*y[IDX_OCNI];
    IJth(jmatrix, 42, 109) = 0.0 + k[1060]*y[IDX_H2CNI] + k[1065]*y[IDX_HCNI] -
                        k[1078]*y[IDX_OCNI] - k[1079]*y[IDX_OCNI];
    IJth(jmatrix, 42, 113) = 0.0 - k[993]*y[IDX_OCNI] - k[994]*y[IDX_OCNI] -
                        k[995]*y[IDX_OCNI];
    IJth(jmatrix, 43, 35) = 0.0 + k[358]*y[IDX_EM];
    IJth(jmatrix, 43, 40) = 0.0 + k[356]*y[IDX_EM];
    IJth(jmatrix, 43, 43) = 0.0 - k[25]*y[IDX_CII] - k[95]*y[IDX_HII] - k[289] -
                        k[380]*y[IDX_CII] - k[505]*y[IDX_HII] -
                        k[602]*y[IDX_H3II] - k[611]*y[IDX_H3OII] -
                        k[637]*y[IDX_HCOII] - k[703]*y[IDX_HeII] -
                        k[704]*y[IDX_HeII] - k[1085]*y[IDX_OI] -
                        k[1086]*y[IDX_OI] - k[1180] - k[1181] - k[1233];
    IJth(jmatrix, 43, 46) = 0.0 + k[291] + k[1185];
    IJth(jmatrix, 43, 48) = 0.0 + k[290] + k[1182];
    IJth(jmatrix, 43, 85) = 0.0 - k[611]*y[IDX_SiH2I];
    IJth(jmatrix, 43, 87) = 0.0 - k[25]*y[IDX_SiH2I] - k[380]*y[IDX_SiH2I];
    IJth(jmatrix, 43, 97) = 0.0 - k[703]*y[IDX_SiH2I] - k[704]*y[IDX_SiH2I];
    IJth(jmatrix, 43, 99) = 0.0 - k[602]*y[IDX_SiH2I];
    IJth(jmatrix, 43, 106) = 0.0 - k[95]*y[IDX_SiH2I] - k[505]*y[IDX_SiH2I];
    IJth(jmatrix, 43, 107) = 0.0 - k[637]*y[IDX_SiH2I];
    IJth(jmatrix, 43, 109) = 0.0 - k[1085]*y[IDX_SiH2I] - k[1086]*y[IDX_SiH2I];
    IJth(jmatrix, 43, 110) = 0.0 + k[356]*y[IDX_SiH3II] + k[358]*y[IDX_SiH4II];
    IJth(jmatrix, 44, 26) = 0.0 + k[499]*y[IDX_HII] + k[675]*y[IDX_HeII];
    IJth(jmatrix, 44, 40) = 0.0 + k[834]*y[IDX_OI];
    IJth(jmatrix, 44, 41) = 0.0 + k[833]*y[IDX_OI] + k[862]*y[IDX_O2I];
    IJth(jmatrix, 44, 44) = 0.0 - k[363]*y[IDX_EM] - k[364]*y[IDX_EM] - k[1246];
    IJth(jmatrix, 44, 49) = 0.0 + k[547]*y[IDX_H2I];
    IJth(jmatrix, 44, 52) = 0.0 + k[860]*y[IDX_SiII];
    IJth(jmatrix, 44, 56) = 0.0 + k[606]*y[IDX_H3II] + k[613]*y[IDX_H3OII] +
                        k[640]*y[IDX_HCOII] + k[850]*y[IDX_OHII];
    IJth(jmatrix, 44, 61) = 0.0 + k[572]*y[IDX_H2OI] + k[860]*y[IDX_CH3OHI];
    IJth(jmatrix, 44, 85) = 0.0 + k[613]*y[IDX_SiOI];
    IJth(jmatrix, 44, 93) = 0.0 + k[850]*y[IDX_SiOI];
    IJth(jmatrix, 44, 97) = 0.0 + k[675]*y[IDX_H2SiOI];
    IJth(jmatrix, 44, 99) = 0.0 + k[606]*y[IDX_SiOI];
    IJth(jmatrix, 44, 104) = 0.0 + k[862]*y[IDX_SiH2II];
    IJth(jmatrix, 44, 106) = 0.0 + k[499]*y[IDX_H2SiOI];
    IJth(jmatrix, 44, 107) = 0.0 + k[640]*y[IDX_SiOI];
    IJth(jmatrix, 44, 108) = 0.0 + k[572]*y[IDX_SiII];
    IJth(jmatrix, 44, 109) = 0.0 + k[833]*y[IDX_SiH2II] + k[834]*y[IDX_SiH3II];
    IJth(jmatrix, 44, 110) = 0.0 - k[363]*y[IDX_SiOHII] - k[364]*y[IDX_SiOHII];
    IJth(jmatrix, 44, 112) = 0.0 + k[547]*y[IDX_SiOII];
    IJth(jmatrix, 45, 43) = 0.0 + k[505]*y[IDX_HII] + k[704]*y[IDX_HeII];
    IJth(jmatrix, 45, 45) = 0.0 - k[352]*y[IDX_EM] - k[394]*y[IDX_CI] -
                        k[477]*y[IDX_CHI] - k[573]*y[IDX_H2OI] -
                        k[619]*y[IDX_HI] - k[832]*y[IDX_OI] - k[1179] -
                        k[1203]*y[IDX_H2I] - k[1232];
    IJth(jmatrix, 45, 46) = 0.0 + k[708]*y[IDX_HeII];
    IJth(jmatrix, 45, 47) = 0.0 + k[98]*y[IDX_HII];
    IJth(jmatrix, 45, 48) = 0.0 + k[705]*y[IDX_HeII];
    IJth(jmatrix, 45, 61) = 0.0 + k[1209]*y[IDX_HI];
    IJth(jmatrix, 45, 62) = 0.0 + k[601]*y[IDX_H3II] + k[610]*y[IDX_H3OII] +
                        k[848]*y[IDX_OHII] + k[861]*y[IDX_HCOII];
    IJth(jmatrix, 45, 85) = 0.0 + k[610]*y[IDX_SiI];
    IJth(jmatrix, 45, 93) = 0.0 + k[848]*y[IDX_SiI];
    IJth(jmatrix, 45, 97) = 0.0 + k[704]*y[IDX_SiH2I] + k[705]*y[IDX_SiH3I] +
                        k[708]*y[IDX_SiH4I];
    IJth(jmatrix, 45, 98) = 0.0 - k[477]*y[IDX_SiHII];
    IJth(jmatrix, 45, 99) = 0.0 + k[601]*y[IDX_SiI];
    IJth(jmatrix, 45, 105) = 0.0 - k[394]*y[IDX_SiHII];
    IJth(jmatrix, 45, 106) = 0.0 + k[98]*y[IDX_SiHI] + k[505]*y[IDX_SiH2I];
    IJth(jmatrix, 45, 107) = 0.0 + k[861]*y[IDX_SiI];
    IJth(jmatrix, 45, 108) = 0.0 - k[573]*y[IDX_SiHII];
    IJth(jmatrix, 45, 109) = 0.0 - k[832]*y[IDX_SiHII];
    IJth(jmatrix, 45, 110) = 0.0 - k[352]*y[IDX_SiHII];
    IJth(jmatrix, 45, 112) = 0.0 - k[1203]*y[IDX_SiHII];
    IJth(jmatrix, 45, 113) = 0.0 - k[619]*y[IDX_SiHII] + k[1209]*y[IDX_SiII];
    IJth(jmatrix, 46, 31) = 0.0 + k[1363] + k[1364] + k[1365] + k[1366];
    IJth(jmatrix, 46, 34) = 0.0 + k[361]*y[IDX_EM] + k[575]*y[IDX_H2OI];
    IJth(jmatrix, 46, 46) = 0.0 - k[97]*y[IDX_HII] - k[291] - k[446]*y[IDX_CH3II] -
                        k[507]*y[IDX_HII] - k[604]*y[IDX_H3II] -
                        k[638]*y[IDX_HCOII] - k[707]*y[IDX_HeII] -
                        k[708]*y[IDX_HeII] - k[950]*y[IDX_CNI] -
                        k[1088]*y[IDX_OI] - k[1185] - k[1186] - k[1187] -
                        k[1237];
    IJth(jmatrix, 46, 79) = 0.0 - k[446]*y[IDX_SiH4I];
    IJth(jmatrix, 46, 94) = 0.0 - k[950]*y[IDX_SiH4I];
    IJth(jmatrix, 46, 97) = 0.0 - k[707]*y[IDX_SiH4I] - k[708]*y[IDX_SiH4I];
    IJth(jmatrix, 46, 99) = 0.0 - k[604]*y[IDX_SiH4I];
    IJth(jmatrix, 46, 106) = 0.0 - k[97]*y[IDX_SiH4I] - k[507]*y[IDX_SiH4I];
    IJth(jmatrix, 46, 107) = 0.0 - k[638]*y[IDX_SiH4I];
    IJth(jmatrix, 46, 108) = 0.0 + k[575]*y[IDX_SiH5II];
    IJth(jmatrix, 46, 109) = 0.0 - k[1088]*y[IDX_SiH4I];
    IJth(jmatrix, 46, 110) = 0.0 + k[361]*y[IDX_SiH5II];
    IJth(jmatrix, 47, 40) = 0.0 + k[357]*y[IDX_EM];
    IJth(jmatrix, 47, 41) = 0.0 + k[355]*y[IDX_EM];
    IJth(jmatrix, 47, 43) = 0.0 + k[289] + k[1181];
    IJth(jmatrix, 47, 46) = 0.0 + k[1187];
    IJth(jmatrix, 47, 47) = 0.0 - k[98]*y[IDX_HII] - k[292] - k[381]*y[IDX_CII] -
                        k[508]*y[IDX_HII] - k[605]*y[IDX_H3II] -
                        k[612]*y[IDX_H3OII] - k[639]*y[IDX_HCOII] -
                        k[709]*y[IDX_HeII] - k[849]*y[IDX_OHII] -
                        k[877]*y[IDX_CI] - k[1089]*y[IDX_OI] - k[1188] - k[1230];
    IJth(jmatrix, 47, 48) = 0.0 + k[1184];
    IJth(jmatrix, 47, 85) = 0.0 - k[612]*y[IDX_SiHI];
    IJth(jmatrix, 47, 87) = 0.0 - k[381]*y[IDX_SiHI];
    IJth(jmatrix, 47, 93) = 0.0 - k[849]*y[IDX_SiHI];
    IJth(jmatrix, 47, 97) = 0.0 - k[709]*y[IDX_SiHI];
    IJth(jmatrix, 47, 99) = 0.0 - k[605]*y[IDX_SiHI];
    IJth(jmatrix, 47, 105) = 0.0 - k[877]*y[IDX_SiHI];
    IJth(jmatrix, 47, 106) = 0.0 - k[98]*y[IDX_SiHI] - k[508]*y[IDX_SiHI];
    IJth(jmatrix, 47, 107) = 0.0 - k[639]*y[IDX_SiHI];
    IJth(jmatrix, 47, 109) = 0.0 - k[1089]*y[IDX_SiHI];
    IJth(jmatrix, 47, 110) = 0.0 + k[355]*y[IDX_SiH2II] + k[357]*y[IDX_SiH3II];
    IJth(jmatrix, 48, 34) = 0.0 + k[360]*y[IDX_EM];
    IJth(jmatrix, 48, 35) = 0.0 + k[359]*y[IDX_EM] + k[489]*y[IDX_COI] +
                        k[574]*y[IDX_H2OI];
    IJth(jmatrix, 48, 46) = 0.0 + k[950]*y[IDX_CNI] + k[1088]*y[IDX_OI] + k[1186];
    IJth(jmatrix, 48, 48) = 0.0 - k[26]*y[IDX_CII] - k[96]*y[IDX_HII] - k[290] -
                        k[506]*y[IDX_HII] - k[603]*y[IDX_H3II] -
                        k[705]*y[IDX_HeII] - k[706]*y[IDX_HeII] -
                        k[1087]*y[IDX_OI] - k[1182] - k[1183] - k[1184] -
                        k[1235];
    IJth(jmatrix, 48, 87) = 0.0 - k[26]*y[IDX_SiH3I];
    IJth(jmatrix, 48, 94) = 0.0 + k[950]*y[IDX_SiH4I];
    IJth(jmatrix, 48, 97) = 0.0 - k[705]*y[IDX_SiH3I] - k[706]*y[IDX_SiH3I];
    IJth(jmatrix, 48, 99) = 0.0 - k[603]*y[IDX_SiH3I];
    IJth(jmatrix, 48, 106) = 0.0 - k[96]*y[IDX_SiH3I] - k[506]*y[IDX_SiH3I];
    IJth(jmatrix, 48, 108) = 0.0 + k[574]*y[IDX_SiH4II];
    IJth(jmatrix, 48, 109) = 0.0 - k[1087]*y[IDX_SiH3I] + k[1088]*y[IDX_SiH4I];
    IJth(jmatrix, 48, 110) = 0.0 + k[359]*y[IDX_SiH4II] + k[360]*y[IDX_SiH5II];
    IJth(jmatrix, 48, 111) = 0.0 + k[489]*y[IDX_SiH4II];
    IJth(jmatrix, 49, 36) = 0.0 + k[831]*y[IDX_OI];
    IJth(jmatrix, 49, 45) = 0.0 + k[832]*y[IDX_OI];
    IJth(jmatrix, 49, 49) = 0.0 - k[138]*y[IDX_HCOI] - k[154]*y[IDX_MgI] -
                        k[206]*y[IDX_NOI] - k[362]*y[IDX_EM] - k[395]*y[IDX_CI]
                        - k[438]*y[IDX_CH2I] - k[478]*y[IDX_CHI] -
                        k[490]*y[IDX_COI] - k[547]*y[IDX_H2I] - k[745]*y[IDX_NI]
                        - k[746]*y[IDX_NI] - k[835]*y[IDX_OI] - k[1189] -
                        k[1245];
    IJth(jmatrix, 49, 53) = 0.0 - k[154]*y[IDX_SiOII];
    IJth(jmatrix, 49, 56) = 0.0 + k[99]*y[IDX_HII] + k[1191];
    IJth(jmatrix, 49, 61) = 0.0 + k[859]*y[IDX_OHI] + k[1212]*y[IDX_OI];
    IJth(jmatrix, 49, 90) = 0.0 - k[438]*y[IDX_SiOII];
    IJth(jmatrix, 49, 96) = 0.0 - k[138]*y[IDX_SiOII];
    IJth(jmatrix, 49, 98) = 0.0 - k[478]*y[IDX_SiOII];
    IJth(jmatrix, 49, 101) = 0.0 - k[206]*y[IDX_SiOII];
    IJth(jmatrix, 49, 102) = 0.0 - k[745]*y[IDX_SiOII] - k[746]*y[IDX_SiOII];
    IJth(jmatrix, 49, 103) = 0.0 + k[859]*y[IDX_SiII];
    IJth(jmatrix, 49, 105) = 0.0 - k[395]*y[IDX_SiOII];
    IJth(jmatrix, 49, 106) = 0.0 + k[99]*y[IDX_SiOI];
    IJth(jmatrix, 49, 109) = 0.0 + k[831]*y[IDX_SiCII] + k[832]*y[IDX_SiHII] -
                        k[835]*y[IDX_SiOII] + k[1212]*y[IDX_SiII];
    IJth(jmatrix, 49, 110) = 0.0 - k[362]*y[IDX_SiOII];
    IJth(jmatrix, 49, 111) = 0.0 - k[490]*y[IDX_SiOII];
    IJth(jmatrix, 49, 112) = 0.0 - k[547]*y[IDX_SiOII];
    IJth(jmatrix, 50, 50) = 0.0 - k[331]*y[IDX_EM] - k[332]*y[IDX_EM] -
                        k[333]*y[IDX_EM] - k[387]*y[IDX_CI] - k[485]*y[IDX_COI]
                        - k[567]*y[IDX_H2OI] - k[824]*y[IDX_OI] - k[1287];
    IJth(jmatrix, 50, 55) = 0.0 + k[447]*y[IDX_CO2I];
    IJth(jmatrix, 50, 59) = 0.0 + k[734]*y[IDX_CO2I];
    IJth(jmatrix, 50, 60) = 0.0 + k[821]*y[IDX_CO2I];
    IJth(jmatrix, 50, 64) = 0.0 + k[652]*y[IDX_CO2I];
    IJth(jmatrix, 50, 69) = 0.0 + k[514]*y[IDX_CO2I];
    IJth(jmatrix, 50, 72) = 0.0 + k[447]*y[IDX_CH4II] + k[514]*y[IDX_H2II] +
                        k[582]*y[IDX_H3II] + k[620]*y[IDX_HCNII] +
                        k[652]*y[IDX_HNOII] + k[734]*y[IDX_N2HII] +
                        k[748]*y[IDX_NHII] + k[821]*y[IDX_O2HII] +
                        k[837]*y[IDX_OHII];
    IJth(jmatrix, 50, 75) = 0.0 + k[620]*y[IDX_CO2I];
    IJth(jmatrix, 50, 77) = 0.0 + k[748]*y[IDX_CO2I];
    IJth(jmatrix, 50, 93) = 0.0 + k[837]*y[IDX_CO2I];
    IJth(jmatrix, 50, 99) = 0.0 + k[582]*y[IDX_CO2I];
    IJth(jmatrix, 50, 103) = 0.0 + k[855]*y[IDX_HCOII];
    IJth(jmatrix, 50, 105) = 0.0 - k[387]*y[IDX_HCO2II];
    IJth(jmatrix, 50, 107) = 0.0 + k[855]*y[IDX_OHI];
    IJth(jmatrix, 50, 108) = 0.0 - k[567]*y[IDX_HCO2II];
    IJth(jmatrix, 50, 109) = 0.0 - k[824]*y[IDX_HCO2II];
    IJth(jmatrix, 50, 110) = 0.0 - k[331]*y[IDX_HCO2II] - k[332]*y[IDX_HCO2II] -
                        k[333]*y[IDX_HCO2II];
    IJth(jmatrix, 50, 111) = 0.0 - k[485]*y[IDX_HCO2II];
    IJth(jmatrix, 51, 16) = 0.0 + k[1351] + k[1352] + k[1353] + k[1354];
    IJth(jmatrix, 51, 25) = 0.0 + k[310]*y[IDX_EM];
    IJth(jmatrix, 51, 39) = 0.0 + k[909]*y[IDX_CH3I] + k[1041]*y[IDX_NHI];
    IJth(jmatrix, 51, 51) = 0.0 - k[264] - k[503]*y[IDX_HII] - k[590]*y[IDX_H3II] -
                        k[686]*y[IDX_HeII] - k[687]*y[IDX_HeII] -
                        k[883]*y[IDX_CH2I] - k[906]*y[IDX_CH3I] -
                        k[926]*y[IDX_CHI] - k[944]*y[IDX_CNI] -
                        k[951]*y[IDX_COI] - k[980]*y[IDX_HI] - k[981]*y[IDX_HI]
                        - k[982]*y[IDX_HI] - k[999]*y[IDX_HCOI] -
                        k[1017]*y[IDX_NI] - k[1068]*y[IDX_OI] -
                        k[1069]*y[IDX_OI] - k[1070]*y[IDX_OI] -
                        k[1097]*y[IDX_OHI] - k[1153] - k[1294];
    IJth(jmatrix, 51, 64) = 0.0 + k[204]*y[IDX_NOI];
    IJth(jmatrix, 51, 71) = 0.0 - k[906]*y[IDX_HNOI] + k[909]*y[IDX_NO2I];
    IJth(jmatrix, 51, 80) = 0.0 + k[1072]*y[IDX_OI];
    IJth(jmatrix, 51, 90) = 0.0 - k[883]*y[IDX_HNOI];
    IJth(jmatrix, 51, 92) = 0.0 + k[1041]*y[IDX_NO2I] + k[1044]*y[IDX_O2I] +
                        k[1049]*y[IDX_OHI];
    IJth(jmatrix, 51, 94) = 0.0 - k[944]*y[IDX_HNOI];
    IJth(jmatrix, 51, 96) = 0.0 - k[999]*y[IDX_HNOI] + k[1000]*y[IDX_NOI];
    IJth(jmatrix, 51, 97) = 0.0 - k[686]*y[IDX_HNOI] - k[687]*y[IDX_HNOI];
    IJth(jmatrix, 51, 98) = 0.0 - k[926]*y[IDX_HNOI];
    IJth(jmatrix, 51, 99) = 0.0 - k[590]*y[IDX_HNOI];
    IJth(jmatrix, 51, 101) = 0.0 + k[204]*y[IDX_HNOII] + k[1000]*y[IDX_HCOI];
    IJth(jmatrix, 51, 102) = 0.0 - k[1017]*y[IDX_HNOI];
    IJth(jmatrix, 51, 103) = 0.0 + k[1049]*y[IDX_NHI] - k[1097]*y[IDX_HNOI];
    IJth(jmatrix, 51, 104) = 0.0 + k[1044]*y[IDX_NHI];
    IJth(jmatrix, 51, 106) = 0.0 - k[503]*y[IDX_HNOI];
    IJth(jmatrix, 51, 109) = 0.0 - k[1068]*y[IDX_HNOI] - k[1069]*y[IDX_HNOI] -
                        k[1070]*y[IDX_HNOI] + k[1072]*y[IDX_NH2I];
    IJth(jmatrix, 51, 110) = 0.0 + k[310]*y[IDX_H2NOII];
    IJth(jmatrix, 51, 111) = 0.0 - k[951]*y[IDX_HNOI];
    IJth(jmatrix, 51, 113) = 0.0 - k[980]*y[IDX_HNOI] - k[981]*y[IDX_HNOI] -
                        k[982]*y[IDX_HNOI];
    IJth(jmatrix, 52, 13) = 0.0 + k[1359] + k[1360] + k[1361] + k[1362];
    IJth(jmatrix, 52, 52) = 0.0 - k[247] - k[248] - k[365]*y[IDX_CII] -
                        k[366]*y[IDX_CII] - k[396]*y[IDX_CHII] -
                        k[397]*y[IDX_CHII] - k[439]*y[IDX_CH3II] -
                        k[492]*y[IDX_HII] - k[493]*y[IDX_HII] -
                        k[494]*y[IDX_HII] - k[579]*y[IDX_H3II] -
                        k[656]*y[IDX_HeII] - k[657]*y[IDX_HeII] -
                        k[712]*y[IDX_NII] - k[713]*y[IDX_NII] -
                        k[714]*y[IDX_NII] - k[715]*y[IDX_NII] -
                        k[808]*y[IDX_OII] - k[809]*y[IDX_OII] -
                        k[820]*y[IDX_O2II] - k[860]*y[IDX_SiII] - k[1119] -
                        k[1120] - k[1121] - k[1223];
    IJth(jmatrix, 52, 61) = 0.0 - k[860]*y[IDX_CH3OHI];
    IJth(jmatrix, 52, 73) = 0.0 - k[712]*y[IDX_CH3OHI] - k[713]*y[IDX_CH3OHI] -
                        k[714]*y[IDX_CH3OHI] - k[715]*y[IDX_CH3OHI];
    IJth(jmatrix, 52, 74) = 0.0 - k[808]*y[IDX_CH3OHI] - k[809]*y[IDX_CH3OHI];
    IJth(jmatrix, 52, 78) = 0.0 - k[820]*y[IDX_CH3OHI];
    IJth(jmatrix, 52, 79) = 0.0 - k[439]*y[IDX_CH3OHI];
    IJth(jmatrix, 52, 87) = 0.0 - k[365]*y[IDX_CH3OHI] - k[366]*y[IDX_CH3OHI];
    IJth(jmatrix, 52, 89) = 0.0 - k[396]*y[IDX_CH3OHI] - k[397]*y[IDX_CH3OHI];
    IJth(jmatrix, 52, 97) = 0.0 - k[656]*y[IDX_CH3OHI] - k[657]*y[IDX_CH3OHI];
    IJth(jmatrix, 52, 99) = 0.0 - k[579]*y[IDX_CH3OHI];
    IJth(jmatrix, 52, 106) = 0.0 - k[492]*y[IDX_CH3OHI] - k[493]*y[IDX_CH3OHI] -
                        k[494]*y[IDX_CH3OHI];
    IJth(jmatrix, 53, 6) = 0.0 + k[1319] + k[1320] + k[1321] + k[1322];
    IJth(jmatrix, 53, 49) = 0.0 - k[154]*y[IDX_MgI];
    IJth(jmatrix, 53, 53) = 0.0 - k[18]*y[IDX_CII] - k[32]*y[IDX_CHII] -
                        k[47]*y[IDX_CH3II] - k[83]*y[IDX_HII] -
                        k[119]*y[IDX_H2OII] - k[148]*y[IDX_H2COII] -
                        k[149]*y[IDX_HCOII] - k[150]*y[IDX_N2II] -
                        k[151]*y[IDX_NOII] - k[152]*y[IDX_O2II] -
                        k[153]*y[IDX_SiII] - k[154]*y[IDX_SiOII] -
                        k[163]*y[IDX_NII] - k[190]*y[IDX_NH3II] - k[266] -
                        k[591]*y[IDX_H3II] - k[1154] - k[1300];
    IJth(jmatrix, 53, 54) = 0.0 + k[1219]*y[IDX_EM];
    IJth(jmatrix, 53, 61) = 0.0 - k[153]*y[IDX_MgI];
    IJth(jmatrix, 53, 65) = 0.0 - k[150]*y[IDX_MgI];
    IJth(jmatrix, 53, 73) = 0.0 - k[163]*y[IDX_MgI];
    IJth(jmatrix, 53, 78) = 0.0 - k[152]*y[IDX_MgI];
    IJth(jmatrix, 53, 79) = 0.0 - k[47]*y[IDX_MgI];
    IJth(jmatrix, 53, 82) = 0.0 - k[119]*y[IDX_MgI];
    IJth(jmatrix, 53, 83) = 0.0 - k[190]*y[IDX_MgI];
    IJth(jmatrix, 53, 84) = 0.0 - k[151]*y[IDX_MgI];
    IJth(jmatrix, 53, 87) = 0.0 - k[18]*y[IDX_MgI];
    IJth(jmatrix, 53, 89) = 0.0 - k[32]*y[IDX_MgI];
    IJth(jmatrix, 53, 91) = 0.0 - k[148]*y[IDX_MgI];
    IJth(jmatrix, 53, 99) = 0.0 - k[591]*y[IDX_MgI];
    IJth(jmatrix, 53, 106) = 0.0 - k[83]*y[IDX_MgI];
    IJth(jmatrix, 53, 107) = 0.0 - k[149]*y[IDX_MgI];
    IJth(jmatrix, 53, 110) = 0.0 + k[1219]*y[IDX_MgII];
    IJth(jmatrix, 54, 49) = 0.0 + k[154]*y[IDX_MgI];
    IJth(jmatrix, 54, 53) = 0.0 + k[18]*y[IDX_CII] + k[32]*y[IDX_CHII] +
                        k[47]*y[IDX_CH3II] + k[83]*y[IDX_HII] +
                        k[119]*y[IDX_H2OII] + k[148]*y[IDX_H2COII] +
                        k[149]*y[IDX_HCOII] + k[150]*y[IDX_N2II] +
                        k[151]*y[IDX_NOII] + k[152]*y[IDX_O2II] +
                        k[153]*y[IDX_SiII] + k[154]*y[IDX_SiOII] +
                        k[163]*y[IDX_NII] + k[190]*y[IDX_NH3II] + k[266] +
                        k[591]*y[IDX_H3II] + k[1154];
    IJth(jmatrix, 54, 54) = 0.0 - k[1219]*y[IDX_EM] - k[1299];
    IJth(jmatrix, 54, 61) = 0.0 + k[153]*y[IDX_MgI];
    IJth(jmatrix, 54, 65) = 0.0 + k[150]*y[IDX_MgI];
    IJth(jmatrix, 54, 73) = 0.0 + k[163]*y[IDX_MgI];
    IJth(jmatrix, 54, 78) = 0.0 + k[152]*y[IDX_MgI];
    IJth(jmatrix, 54, 79) = 0.0 + k[47]*y[IDX_MgI];
    IJth(jmatrix, 54, 82) = 0.0 + k[119]*y[IDX_MgI];
    IJth(jmatrix, 54, 83) = 0.0 + k[190]*y[IDX_MgI];
    IJth(jmatrix, 54, 84) = 0.0 + k[151]*y[IDX_MgI];
    IJth(jmatrix, 54, 87) = 0.0 + k[18]*y[IDX_MgI];
    IJth(jmatrix, 54, 89) = 0.0 + k[32]*y[IDX_MgI];
    IJth(jmatrix, 54, 91) = 0.0 + k[148]*y[IDX_MgI];
    IJth(jmatrix, 54, 99) = 0.0 + k[591]*y[IDX_MgI];
    IJth(jmatrix, 54, 106) = 0.0 + k[83]*y[IDX_MgI];
    IJth(jmatrix, 54, 107) = 0.0 + k[149]*y[IDX_MgI];
    IJth(jmatrix, 54, 110) = 0.0 - k[1219]*y[IDX_MgII];
    IJth(jmatrix, 55, 55) = 0.0 - k[49]*y[IDX_H2COI] - k[50]*y[IDX_NH3I] -
                        k[51]*y[IDX_O2I] - k[301]*y[IDX_EM] - k[302]*y[IDX_EM] -
                        k[447]*y[IDX_CO2I] - k[448]*y[IDX_COI] -
                        k[449]*y[IDX_H2COI] - k[450]*y[IDX_H2OI] -
                        k[617]*y[IDX_HI] - k[822]*y[IDX_OI] - k[1122] - k[1123]
                        - k[1288];
    IJth(jmatrix, 55, 67) = 0.0 + k[52]*y[IDX_COII] + k[77]*y[IDX_HII] +
                        k[101]*y[IDX_H2II] + k[140]*y[IDX_HeII] +
                        k[156]*y[IDX_NII] + k[207]*y[IDX_OII] + k[1126];
    IJth(jmatrix, 55, 68) = 0.0 + k[52]*y[IDX_CH4I];
    IJth(jmatrix, 55, 69) = 0.0 + k[101]*y[IDX_CH4I];
    IJth(jmatrix, 55, 70) = 0.0 - k[50]*y[IDX_CH4II];
    IJth(jmatrix, 55, 71) = 0.0 + k[578]*y[IDX_H3II];
    IJth(jmatrix, 55, 72) = 0.0 - k[447]*y[IDX_CH4II];
    IJth(jmatrix, 55, 73) = 0.0 + k[156]*y[IDX_CH4I];
    IJth(jmatrix, 55, 74) = 0.0 + k[207]*y[IDX_CH4I];
    IJth(jmatrix, 55, 79) = 0.0 + k[441]*y[IDX_HCOI];
    IJth(jmatrix, 55, 95) = 0.0 - k[49]*y[IDX_CH4II] - k[449]*y[IDX_CH4II];
    IJth(jmatrix, 55, 96) = 0.0 + k[441]*y[IDX_CH3II];
    IJth(jmatrix, 55, 97) = 0.0 + k[140]*y[IDX_CH4I];
    IJth(jmatrix, 55, 99) = 0.0 + k[578]*y[IDX_CH3I];
    IJth(jmatrix, 55, 104) = 0.0 - k[51]*y[IDX_CH4II];
    IJth(jmatrix, 55, 106) = 0.0 + k[77]*y[IDX_CH4I];
    IJth(jmatrix, 55, 108) = 0.0 - k[450]*y[IDX_CH4II];
    IJth(jmatrix, 55, 109) = 0.0 - k[822]*y[IDX_CH4II];
    IJth(jmatrix, 55, 110) = 0.0 - k[301]*y[IDX_CH4II] - k[302]*y[IDX_CH4II];
    IJth(jmatrix, 55, 111) = 0.0 - k[448]*y[IDX_CH4II];
    IJth(jmatrix, 55, 113) = 0.0 - k[617]*y[IDX_CH4II];
    IJth(jmatrix, 56, 3) = 0.0 + k[1379] + k[1380] + k[1381] + k[1382];
    IJth(jmatrix, 56, 26) = 0.0 + k[257] + k[1143] + k[1144];
    IJth(jmatrix, 56, 38) = 0.0 + k[1084]*y[IDX_OI];
    IJth(jmatrix, 56, 43) = 0.0 + k[1085]*y[IDX_OI] + k[1086]*y[IDX_OI];
    IJth(jmatrix, 56, 44) = 0.0 + k[364]*y[IDX_EM];
    IJth(jmatrix, 56, 47) = 0.0 + k[1089]*y[IDX_OI];
    IJth(jmatrix, 56, 49) = 0.0 + k[138]*y[IDX_HCOI] + k[154]*y[IDX_MgI] +
                        k[206]*y[IDX_NOI];
    IJth(jmatrix, 56, 53) = 0.0 + k[154]*y[IDX_SiOII];
    IJth(jmatrix, 56, 56) = 0.0 - k[99]*y[IDX_HII] - k[293] - k[382]*y[IDX_CII] -
                        k[606]*y[IDX_H3II] - k[613]*y[IDX_H3OII] -
                        k[640]*y[IDX_HCOII] - k[710]*y[IDX_HeII] -
                        k[711]*y[IDX_HeII] - k[850]*y[IDX_OHII] - k[1190] -
                        k[1191] - k[1229];
    IJth(jmatrix, 56, 62) = 0.0 + k[1102]*y[IDX_OHI] + k[1103]*y[IDX_CO2I] +
                        k[1104]*y[IDX_COI] + k[1105]*y[IDX_NOI] +
                        k[1106]*y[IDX_O2I] + k[1213]*y[IDX_OI];
    IJth(jmatrix, 56, 72) = 0.0 + k[1103]*y[IDX_SiI];
    IJth(jmatrix, 56, 85) = 0.0 - k[613]*y[IDX_SiOI];
    IJth(jmatrix, 56, 87) = 0.0 - k[382]*y[IDX_SiOI];
    IJth(jmatrix, 56, 93) = 0.0 - k[850]*y[IDX_SiOI];
    IJth(jmatrix, 56, 96) = 0.0 + k[138]*y[IDX_SiOII];
    IJth(jmatrix, 56, 97) = 0.0 - k[710]*y[IDX_SiOI] - k[711]*y[IDX_SiOI];
    IJth(jmatrix, 56, 99) = 0.0 - k[606]*y[IDX_SiOI];
    IJth(jmatrix, 56, 101) = 0.0 + k[206]*y[IDX_SiOII] + k[1105]*y[IDX_SiI];
    IJth(jmatrix, 56, 103) = 0.0 + k[1102]*y[IDX_SiI];
    IJth(jmatrix, 56, 104) = 0.0 + k[1106]*y[IDX_SiI];
    IJth(jmatrix, 56, 106) = 0.0 - k[99]*y[IDX_SiOI];
    IJth(jmatrix, 56, 107) = 0.0 - k[640]*y[IDX_SiOI];
    IJth(jmatrix, 56, 109) = 0.0 + k[1084]*y[IDX_SiCI] + k[1085]*y[IDX_SiH2I] +
                        k[1086]*y[IDX_SiH2I] + k[1089]*y[IDX_SiHI] +
                        k[1213]*y[IDX_SiI];
    IJth(jmatrix, 56, 110) = 0.0 + k[364]*y[IDX_SiOHII];
    IJth(jmatrix, 56, 111) = 0.0 + k[1104]*y[IDX_SiI];
    IJth(jmatrix, 57, 42) = 0.0 + k[697]*y[IDX_HeII];
    IJth(jmatrix, 57, 57) = 0.0 - k[27]*y[IDX_CI] - k[37]*y[IDX_CH2I] -
                        k[53]*y[IDX_CHI] - k[63]*y[IDX_COI] - k[64]*y[IDX_H2COI]
                        - k[65]*y[IDX_HCNI] - k[66]*y[IDX_HCOI] -
                        k[67]*y[IDX_NOI] - k[68]*y[IDX_O2I] - k[126]*y[IDX_HI] -
                        k[183]*y[IDX_NH2I] - k[199]*y[IDX_NHI] -
                        k[216]*y[IDX_OI] - k[225]*y[IDX_OHI] - k[303]*y[IDX_EM]
                        - k[479]*y[IDX_H2COI] - k[480]*y[IDX_HCOI] -
                        k[481]*y[IDX_O2I] - k[531]*y[IDX_H2I] -
                        k[560]*y[IDX_H2OI] - k[561]*y[IDX_H2OI] -
                        k[737]*y[IDX_NI] - k[1276];
    IJth(jmatrix, 57, 63) = 0.0 + k[683]*y[IDX_HeII];
    IJth(jmatrix, 57, 65) = 0.0 + k[69]*y[IDX_CNI];
    IJth(jmatrix, 57, 69) = 0.0 + k[103]*y[IDX_CNI];
    IJth(jmatrix, 57, 73) = 0.0 + k[157]*y[IDX_CNI] + k[468]*y[IDX_CHI];
    IJth(jmatrix, 57, 80) = 0.0 - k[183]*y[IDX_CNII];
    IJth(jmatrix, 57, 87) = 0.0 + k[375]*y[IDX_NHI] + k[1192]*y[IDX_NI];
    IJth(jmatrix, 57, 88) = 0.0 - k[65]*y[IDX_CNII] + k[676]*y[IDX_HeII];
    IJth(jmatrix, 57, 89) = 0.0 + k[408]*y[IDX_NI] + k[410]*y[IDX_NHI];
    IJth(jmatrix, 57, 90) = 0.0 - k[37]*y[IDX_CNII];
    IJth(jmatrix, 57, 92) = 0.0 - k[199]*y[IDX_CNII] + k[375]*y[IDX_CII] +
                        k[410]*y[IDX_CHII];
    IJth(jmatrix, 57, 94) = 0.0 + k[69]*y[IDX_N2II] + k[103]*y[IDX_H2II] +
                        k[157]*y[IDX_NII];
    IJth(jmatrix, 57, 95) = 0.0 - k[64]*y[IDX_CNII] - k[479]*y[IDX_CNII];
    IJth(jmatrix, 57, 96) = 0.0 - k[66]*y[IDX_CNII] - k[480]*y[IDX_CNII];
    IJth(jmatrix, 57, 97) = 0.0 + k[676]*y[IDX_HCNI] + k[683]*y[IDX_HNCI] +
                        k[697]*y[IDX_OCNI];
    IJth(jmatrix, 57, 98) = 0.0 - k[53]*y[IDX_CNII] + k[468]*y[IDX_NII];
    IJth(jmatrix, 57, 101) = 0.0 - k[67]*y[IDX_CNII];
    IJth(jmatrix, 57, 102) = 0.0 + k[408]*y[IDX_CHII] - k[737]*y[IDX_CNII] +
                        k[1192]*y[IDX_CII];
    IJth(jmatrix, 57, 103) = 0.0 - k[225]*y[IDX_CNII];
    IJth(jmatrix, 57, 104) = 0.0 - k[68]*y[IDX_CNII] - k[481]*y[IDX_CNII];
    IJth(jmatrix, 57, 105) = 0.0 - k[27]*y[IDX_CNII];
    IJth(jmatrix, 57, 108) = 0.0 - k[560]*y[IDX_CNII] - k[561]*y[IDX_CNII];
    IJth(jmatrix, 57, 109) = 0.0 - k[216]*y[IDX_CNII];
    IJth(jmatrix, 57, 110) = 0.0 - k[303]*y[IDX_CNII];
    IJth(jmatrix, 57, 111) = 0.0 - k[63]*y[IDX_CNII];
    IJth(jmatrix, 57, 112) = 0.0 - k[531]*y[IDX_CNII];
    IJth(jmatrix, 57, 113) = 0.0 - k[126]*y[IDX_CNII];
    IJth(jmatrix, 58, 58) = 0.0 - k[327]*y[IDX_EM] - k[328]*y[IDX_EM] -
                        k[329]*y[IDX_EM] - k[427]*y[IDX_CH2I] -
                        k[428]*y[IDX_CH2I] - k[464]*y[IDX_CHI] -
                        k[465]*y[IDX_CHI] - k[633]*y[IDX_H2COI] -
                        k[634]*y[IDX_H2COI] - k[785]*y[IDX_NH2I] -
                        k[786]*y[IDX_NH2I] - k[1301];
    IJth(jmatrix, 58, 59) = 0.0 + k[631]*y[IDX_HCNI] + k[650]*y[IDX_HNCI];
    IJth(jmatrix, 58, 60) = 0.0 + k[632]*y[IDX_HCNI] + k[651]*y[IDX_HNCI];
    IJth(jmatrix, 58, 63) = 0.0 + k[407]*y[IDX_CHII] + k[559]*y[IDX_H2OII] +
                        k[589]*y[IDX_H3II] + k[609]*y[IDX_H3OII] +
                        k[626]*y[IDX_HCNII] + k[646]*y[IDX_H2COII] +
                        k[647]*y[IDX_H3COII] + k[648]*y[IDX_HCOII] +
                        k[649]*y[IDX_HNOII] + k[650]*y[IDX_N2HII] +
                        k[651]*y[IDX_O2HII] + k[760]*y[IDX_NHII] +
                        k[775]*y[IDX_NH2II] + k[844]*y[IDX_OHII];
    IJth(jmatrix, 58, 64) = 0.0 + k[630]*y[IDX_HCNI] + k[649]*y[IDX_HNCI];
    IJth(jmatrix, 58, 66) = 0.0 + k[628]*y[IDX_HCNI] + k[647]*y[IDX_HNCI];
    IJth(jmatrix, 58, 67) = 0.0 + k[454]*y[IDX_HCNII] + k[718]*y[IDX_NII];
    IJth(jmatrix, 58, 70) = 0.0 + k[793]*y[IDX_HCNII];
    IJth(jmatrix, 58, 73) = 0.0 + k[718]*y[IDX_CH4I];
    IJth(jmatrix, 58, 75) = 0.0 + k[454]*y[IDX_CH4I] + k[535]*y[IDX_H2I] +
                        k[623]*y[IDX_HCNI] + k[625]*y[IDX_HCOI] +
                        k[626]*y[IDX_HNCI] + k[793]*y[IDX_NH3I];
    IJth(jmatrix, 58, 76) = 0.0 + k[773]*y[IDX_HCNI] + k[775]*y[IDX_HNCI];
    IJth(jmatrix, 58, 77) = 0.0 + k[758]*y[IDX_HCNI] + k[760]*y[IDX_HNCI];
    IJth(jmatrix, 58, 79) = 0.0 + k[794]*y[IDX_NHI];
    IJth(jmatrix, 58, 80) = 0.0 - k[785]*y[IDX_HCNHII] - k[786]*y[IDX_HCNHII];
    IJth(jmatrix, 58, 82) = 0.0 + k[556]*y[IDX_HCNI] + k[559]*y[IDX_HNCI];
    IJth(jmatrix, 58, 85) = 0.0 + k[608]*y[IDX_HCNI] + k[609]*y[IDX_HNCI];
    IJth(jmatrix, 58, 88) = 0.0 + k[405]*y[IDX_CHII] + k[556]*y[IDX_H2OII] +
                        k[587]*y[IDX_H3II] + k[608]*y[IDX_H3OII] +
                        k[623]*y[IDX_HCNII] + k[627]*y[IDX_H2COII] +
                        k[628]*y[IDX_H3COII] + k[629]*y[IDX_HCOII] +
                        k[630]*y[IDX_HNOII] + k[631]*y[IDX_N2HII] +
                        k[632]*y[IDX_O2HII] + k[758]*y[IDX_NHII] +
                        k[773]*y[IDX_NH2II] + k[841]*y[IDX_OHII];
    IJth(jmatrix, 58, 89) = 0.0 + k[405]*y[IDX_HCNI] + k[407]*y[IDX_HNCI];
    IJth(jmatrix, 58, 90) = 0.0 - k[427]*y[IDX_HCNHII] - k[428]*y[IDX_HCNHII];
    IJth(jmatrix, 58, 91) = 0.0 + k[627]*y[IDX_HCNI] + k[646]*y[IDX_HNCI];
    IJth(jmatrix, 58, 92) = 0.0 + k[794]*y[IDX_CH3II];
    IJth(jmatrix, 58, 93) = 0.0 + k[841]*y[IDX_HCNI] + k[844]*y[IDX_HNCI];
    IJth(jmatrix, 58, 95) = 0.0 - k[633]*y[IDX_HCNHII] - k[634]*y[IDX_HCNHII];
    IJth(jmatrix, 58, 96) = 0.0 + k[625]*y[IDX_HCNII];
    IJth(jmatrix, 58, 98) = 0.0 - k[464]*y[IDX_HCNHII] - k[465]*y[IDX_HCNHII];
    IJth(jmatrix, 58, 99) = 0.0 + k[587]*y[IDX_HCNI] + k[589]*y[IDX_HNCI];
    IJth(jmatrix, 58, 107) = 0.0 + k[629]*y[IDX_HCNI] + k[648]*y[IDX_HNCI];
    IJth(jmatrix, 58, 110) = 0.0 - k[327]*y[IDX_HCNHII] - k[328]*y[IDX_HCNHII] -
                        k[329]*y[IDX_HCNHII];
    IJth(jmatrix, 58, 112) = 0.0 + k[535]*y[IDX_HCNII];
    IJth(jmatrix, 59, 59) = 0.0 - k[338]*y[IDX_EM] - k[339]*y[IDX_EM] -
                        k[389]*y[IDX_CI] - k[431]*y[IDX_CH2I] -
                        k[469]*y[IDX_CHI] - k[487]*y[IDX_COI] -
                        k[570]*y[IDX_H2OI] - k[631]*y[IDX_HCNI] -
                        k[643]*y[IDX_HCOI] - k[650]*y[IDX_HNCI] -
                        k[734]*y[IDX_CO2I] - k[735]*y[IDX_H2COI] -
                        k[789]*y[IDX_NH2I] - k[801]*y[IDX_NHI] -
                        k[826]*y[IDX_OI] - k[857]*y[IDX_OHI] - k[1302];
    IJth(jmatrix, 59, 60) = 0.0 + k[733]*y[IDX_N2I];
    IJth(jmatrix, 59, 63) = 0.0 - k[650]*y[IDX_N2HII];
    IJth(jmatrix, 59, 64) = 0.0 + k[732]*y[IDX_N2I];
    IJth(jmatrix, 59, 65) = 0.0 + k[539]*y[IDX_H2I] + k[569]*y[IDX_H2OI] +
                        k[731]*y[IDX_HCOI];
    IJth(jmatrix, 59, 69) = 0.0 + k[521]*y[IDX_N2I];
    IJth(jmatrix, 59, 70) = 0.0 + k[724]*y[IDX_NII];
    IJth(jmatrix, 59, 72) = 0.0 - k[734]*y[IDX_N2HII];
    IJth(jmatrix, 59, 73) = 0.0 + k[724]*y[IDX_NH3I];
    IJth(jmatrix, 59, 76) = 0.0 + k[741]*y[IDX_NI];
    IJth(jmatrix, 59, 77) = 0.0 + k[761]*y[IDX_N2I] + k[764]*y[IDX_NOI];
    IJth(jmatrix, 59, 80) = 0.0 - k[789]*y[IDX_N2HII];
    IJth(jmatrix, 59, 86) = 0.0 + k[521]*y[IDX_H2II] + k[592]*y[IDX_H3II] +
                        k[732]*y[IDX_HNOII] + k[733]*y[IDX_O2HII] +
                        k[761]*y[IDX_NHII] + k[845]*y[IDX_OHII];
    IJth(jmatrix, 59, 88) = 0.0 - k[631]*y[IDX_N2HII];
    IJth(jmatrix, 59, 90) = 0.0 - k[431]*y[IDX_N2HII];
    IJth(jmatrix, 59, 92) = 0.0 - k[801]*y[IDX_N2HII];
    IJth(jmatrix, 59, 93) = 0.0 + k[845]*y[IDX_N2I];
    IJth(jmatrix, 59, 95) = 0.0 - k[735]*y[IDX_N2HII];
    IJth(jmatrix, 59, 96) = 0.0 - k[643]*y[IDX_N2HII] + k[731]*y[IDX_N2II];
    IJth(jmatrix, 59, 98) = 0.0 - k[469]*y[IDX_N2HII];
    IJth(jmatrix, 59, 99) = 0.0 + k[592]*y[IDX_N2I];
    IJth(jmatrix, 59, 101) = 0.0 + k[764]*y[IDX_NHII];
    IJth(jmatrix, 59, 102) = 0.0 + k[741]*y[IDX_NH2II];
    IJth(jmatrix, 59, 103) = 0.0 - k[857]*y[IDX_N2HII];
    IJth(jmatrix, 59, 105) = 0.0 - k[389]*y[IDX_N2HII];
    IJth(jmatrix, 59, 108) = 0.0 + k[569]*y[IDX_N2II] - k[570]*y[IDX_N2HII];
    IJth(jmatrix, 59, 109) = 0.0 - k[826]*y[IDX_N2HII];
    IJth(jmatrix, 59, 110) = 0.0 - k[338]*y[IDX_N2HII] - k[339]*y[IDX_N2HII];
    IJth(jmatrix, 59, 111) = 0.0 - k[487]*y[IDX_N2HII];
    IJth(jmatrix, 59, 112) = 0.0 + k[539]*y[IDX_N2II];
    IJth(jmatrix, 60, 60) = 0.0 - k[347]*y[IDX_EM] - k[392]*y[IDX_CI] -
                        k[436]*y[IDX_CH2I] - k[474]*y[IDX_CHI] -
                        k[483]*y[IDX_CNI] - k[488]*y[IDX_COI] -
                        k[544]*y[IDX_H2I] - k[552]*y[IDX_H2COI] -
                        k[571]*y[IDX_H2OI] - k[632]*y[IDX_HCNI] -
                        k[645]*y[IDX_HCOI] - k[651]*y[IDX_HNCI] -
                        k[733]*y[IDX_N2I] - k[790]*y[IDX_NH2I] -
                        k[805]*y[IDX_NHI] - k[807]*y[IDX_NOI] -
                        k[821]*y[IDX_CO2I] - k[829]*y[IDX_OI] -
                        k[858]*y[IDX_OHI] - k[1297];
    IJth(jmatrix, 60, 63) = 0.0 - k[651]*y[IDX_O2HII];
    IJth(jmatrix, 60, 69) = 0.0 + k[525]*y[IDX_O2I];
    IJth(jmatrix, 60, 72) = 0.0 - k[821]*y[IDX_O2HII];
    IJth(jmatrix, 60, 77) = 0.0 + k[766]*y[IDX_O2I];
    IJth(jmatrix, 60, 78) = 0.0 + k[644]*y[IDX_HCOI];
    IJth(jmatrix, 60, 80) = 0.0 - k[790]*y[IDX_O2HII];
    IJth(jmatrix, 60, 86) = 0.0 - k[733]*y[IDX_O2HII];
    IJth(jmatrix, 60, 88) = 0.0 - k[632]*y[IDX_O2HII];
    IJth(jmatrix, 60, 90) = 0.0 - k[436]*y[IDX_O2HII];
    IJth(jmatrix, 60, 92) = 0.0 - k[805]*y[IDX_O2HII];
    IJth(jmatrix, 60, 94) = 0.0 - k[483]*y[IDX_O2HII];
    IJth(jmatrix, 60, 95) = 0.0 - k[552]*y[IDX_O2HII];
    IJth(jmatrix, 60, 96) = 0.0 + k[644]*y[IDX_O2II] - k[645]*y[IDX_O2HII];
    IJth(jmatrix, 60, 98) = 0.0 - k[474]*y[IDX_O2HII];
    IJth(jmatrix, 60, 99) = 0.0 + k[597]*y[IDX_O2I];
    IJth(jmatrix, 60, 101) = 0.0 - k[807]*y[IDX_O2HII];
    IJth(jmatrix, 60, 103) = 0.0 - k[858]*y[IDX_O2HII];
    IJth(jmatrix, 60, 104) = 0.0 + k[525]*y[IDX_H2II] + k[597]*y[IDX_H3II] +
                        k[766]*y[IDX_NHII];
    IJth(jmatrix, 60, 105) = 0.0 - k[392]*y[IDX_O2HII];
    IJth(jmatrix, 60, 108) = 0.0 - k[571]*y[IDX_O2HII];
    IJth(jmatrix, 60, 109) = 0.0 - k[829]*y[IDX_O2HII];
    IJth(jmatrix, 60, 110) = 0.0 - k[347]*y[IDX_O2HII];
    IJth(jmatrix, 60, 111) = 0.0 - k[488]*y[IDX_O2HII];
    IJth(jmatrix, 60, 112) = 0.0 - k[544]*y[IDX_O2HII];
    IJth(jmatrix, 61, 36) = 0.0 + k[744]*y[IDX_NI];
    IJth(jmatrix, 61, 38) = 0.0 + k[701]*y[IDX_HeII];
    IJth(jmatrix, 61, 43) = 0.0 + k[703]*y[IDX_HeII];
    IJth(jmatrix, 61, 45) = 0.0 + k[619]*y[IDX_HI] + k[1179];
    IJth(jmatrix, 61, 46) = 0.0 + k[707]*y[IDX_HeII];
    IJth(jmatrix, 61, 47) = 0.0 + k[508]*y[IDX_HII] + k[709]*y[IDX_HeII];
    IJth(jmatrix, 61, 49) = 0.0 + k[395]*y[IDX_CI] + k[438]*y[IDX_CH2I] +
                        k[490]*y[IDX_COI] + k[746]*y[IDX_NI] + k[835]*y[IDX_OI]
                        + k[1189];
    IJth(jmatrix, 61, 52) = 0.0 - k[860]*y[IDX_SiII];
    IJth(jmatrix, 61, 53) = 0.0 - k[153]*y[IDX_SiII];
    IJth(jmatrix, 61, 56) = 0.0 + k[382]*y[IDX_CII] + k[710]*y[IDX_HeII];
    IJth(jmatrix, 61, 61) = 0.0 - k[153]*y[IDX_MgI] - k[476]*y[IDX_CHI] -
                        k[572]*y[IDX_H2OI] - k[859]*y[IDX_OHI] -
                        k[860]*y[IDX_CH3OHI] - k[1202]*y[IDX_H2I] -
                        k[1209]*y[IDX_HI] - k[1212]*y[IDX_OI] -
                        k[1222]*y[IDX_EM] - k[1231];
    IJth(jmatrix, 61, 62) = 0.0 + k[21]*y[IDX_CII] + k[35]*y[IDX_CHII] +
                        k[91]*y[IDX_HII] + k[122]*y[IDX_H2OII] +
                        k[147]*y[IDX_HeII] + k[192]*y[IDX_NH3II] +
                        k[228]*y[IDX_H2COII] + k[229]*y[IDX_NOII] +
                        k[230]*y[IDX_O2II] + k[285] + k[1176];
    IJth(jmatrix, 61, 78) = 0.0 + k[230]*y[IDX_SiI];
    IJth(jmatrix, 61, 82) = 0.0 + k[122]*y[IDX_SiI];
    IJth(jmatrix, 61, 83) = 0.0 + k[192]*y[IDX_SiI];
    IJth(jmatrix, 61, 84) = 0.0 + k[229]*y[IDX_SiI];
    IJth(jmatrix, 61, 87) = 0.0 + k[21]*y[IDX_SiI] + k[382]*y[IDX_SiOI];
    IJth(jmatrix, 61, 89) = 0.0 + k[35]*y[IDX_SiI];
    IJth(jmatrix, 61, 90) = 0.0 + k[438]*y[IDX_SiOII];
    IJth(jmatrix, 61, 91) = 0.0 + k[228]*y[IDX_SiI];
    IJth(jmatrix, 61, 97) = 0.0 + k[147]*y[IDX_SiI] + k[701]*y[IDX_SiCI] +
                        k[703]*y[IDX_SiH2I] + k[707]*y[IDX_SiH4I] +
                        k[709]*y[IDX_SiHI] + k[710]*y[IDX_SiOI];
    IJth(jmatrix, 61, 98) = 0.0 - k[476]*y[IDX_SiII];
    IJth(jmatrix, 61, 102) = 0.0 + k[744]*y[IDX_SiCII] + k[746]*y[IDX_SiOII];
    IJth(jmatrix, 61, 103) = 0.0 - k[859]*y[IDX_SiII];
    IJth(jmatrix, 61, 105) = 0.0 + k[395]*y[IDX_SiOII];
    IJth(jmatrix, 61, 106) = 0.0 + k[91]*y[IDX_SiI] + k[508]*y[IDX_SiHI];
    IJth(jmatrix, 61, 108) = 0.0 - k[572]*y[IDX_SiII];
    IJth(jmatrix, 61, 109) = 0.0 + k[835]*y[IDX_SiOII] - k[1212]*y[IDX_SiII];
    IJth(jmatrix, 61, 110) = 0.0 - k[1222]*y[IDX_SiII];
    IJth(jmatrix, 61, 111) = 0.0 + k[490]*y[IDX_SiOII];
    IJth(jmatrix, 61, 112) = 0.0 - k[1202]*y[IDX_SiII];
    IJth(jmatrix, 61, 113) = 0.0 + k[619]*y[IDX_SiHII] - k[1209]*y[IDX_SiII];
    IJth(jmatrix, 62, 36) = 0.0 + k[349]*y[IDX_EM];
    IJth(jmatrix, 62, 38) = 0.0 + k[288] + k[702]*y[IDX_HeII] + k[1027]*y[IDX_NI] +
                        k[1083]*y[IDX_OI] + k[1178];
    IJth(jmatrix, 62, 41) = 0.0 + k[353]*y[IDX_EM] + k[354]*y[IDX_EM];
    IJth(jmatrix, 62, 44) = 0.0 + k[363]*y[IDX_EM];
    IJth(jmatrix, 62, 45) = 0.0 + k[352]*y[IDX_EM] + k[477]*y[IDX_CHI] +
                        k[573]*y[IDX_H2OI];
    IJth(jmatrix, 62, 47) = 0.0 + k[292] + k[1188];
    IJth(jmatrix, 62, 49) = 0.0 + k[362]*y[IDX_EM] + k[478]*y[IDX_CHI] +
                        k[745]*y[IDX_NI];
    IJth(jmatrix, 62, 53) = 0.0 + k[153]*y[IDX_SiII];
    IJth(jmatrix, 62, 56) = 0.0 + k[293] + k[711]*y[IDX_HeII] + k[1190];
    IJth(jmatrix, 62, 61) = 0.0 + k[153]*y[IDX_MgI] + k[1222]*y[IDX_EM];
    IJth(jmatrix, 62, 62) = 0.0 - k[21]*y[IDX_CII] - k[35]*y[IDX_CHII] -
                        k[91]*y[IDX_HII] - k[122]*y[IDX_H2OII] -
                        k[147]*y[IDX_HeII] - k[192]*y[IDX_NH3II] -
                        k[228]*y[IDX_H2COII] - k[229]*y[IDX_NOII] -
                        k[230]*y[IDX_O2II] - k[285] - k[601]*y[IDX_H3II] -
                        k[610]*y[IDX_H3OII] - k[848]*y[IDX_OHII] -
                        k[861]*y[IDX_HCOII] - k[1102]*y[IDX_OHI] -
                        k[1103]*y[IDX_CO2I] - k[1104]*y[IDX_COI] -
                        k[1105]*y[IDX_NOI] - k[1106]*y[IDX_O2I] - k[1176] -
                        k[1213]*y[IDX_OI] - k[1228];
    IJth(jmatrix, 62, 72) = 0.0 - k[1103]*y[IDX_SiI];
    IJth(jmatrix, 62, 78) = 0.0 - k[230]*y[IDX_SiI];
    IJth(jmatrix, 62, 82) = 0.0 - k[122]*y[IDX_SiI];
    IJth(jmatrix, 62, 83) = 0.0 - k[192]*y[IDX_SiI];
    IJth(jmatrix, 62, 84) = 0.0 - k[229]*y[IDX_SiI];
    IJth(jmatrix, 62, 85) = 0.0 - k[610]*y[IDX_SiI];
    IJth(jmatrix, 62, 87) = 0.0 - k[21]*y[IDX_SiI];
    IJth(jmatrix, 62, 89) = 0.0 - k[35]*y[IDX_SiI];
    IJth(jmatrix, 62, 91) = 0.0 - k[228]*y[IDX_SiI];
    IJth(jmatrix, 62, 93) = 0.0 - k[848]*y[IDX_SiI];
    IJth(jmatrix, 62, 97) = 0.0 - k[147]*y[IDX_SiI] + k[702]*y[IDX_SiCI] +
                        k[711]*y[IDX_SiOI];
    IJth(jmatrix, 62, 98) = 0.0 + k[477]*y[IDX_SiHII] + k[478]*y[IDX_SiOII];
    IJth(jmatrix, 62, 99) = 0.0 - k[601]*y[IDX_SiI];
    IJth(jmatrix, 62, 101) = 0.0 - k[1105]*y[IDX_SiI];
    IJth(jmatrix, 62, 102) = 0.0 + k[745]*y[IDX_SiOII] + k[1027]*y[IDX_SiCI];
    IJth(jmatrix, 62, 103) = 0.0 - k[1102]*y[IDX_SiI];
    IJth(jmatrix, 62, 104) = 0.0 - k[1106]*y[IDX_SiI];
    IJth(jmatrix, 62, 106) = 0.0 - k[91]*y[IDX_SiI];
    IJth(jmatrix, 62, 107) = 0.0 - k[861]*y[IDX_SiI];
    IJth(jmatrix, 62, 108) = 0.0 + k[573]*y[IDX_SiHII];
    IJth(jmatrix, 62, 109) = 0.0 + k[1083]*y[IDX_SiCI] - k[1213]*y[IDX_SiI];
    IJth(jmatrix, 62, 110) = 0.0 + k[349]*y[IDX_SiCII] + k[352]*y[IDX_SiHII] +
                        k[353]*y[IDX_SiH2II] + k[354]*y[IDX_SiH2II] +
                        k[362]*y[IDX_SiOII] + k[363]*y[IDX_SiOHII] +
                        k[1222]*y[IDX_SiII];
    IJth(jmatrix, 62, 111) = 0.0 - k[1104]*y[IDX_SiI];
    IJth(jmatrix, 63, 1) = 0.0 + k[1327] + k[1328] + k[1329] + k[1330];
    IJth(jmatrix, 63, 28) = 0.0 + k[1004]*y[IDX_CI];
    IJth(jmatrix, 63, 58) = 0.0 + k[329]*y[IDX_EM] + k[428]*y[IDX_CH2I] +
                        k[465]*y[IDX_CHI] + k[634]*y[IDX_H2COI] +
                        k[786]*y[IDX_NH2I];
    IJth(jmatrix, 63, 59) = 0.0 - k[650]*y[IDX_HNCI];
    IJth(jmatrix, 63, 60) = 0.0 - k[651]*y[IDX_HNCI];
    IJth(jmatrix, 63, 63) = 0.0 - k[1]*y[IDX_HII] - k[262] - k[407]*y[IDX_CHII] -
                        k[559]*y[IDX_H2OII] - k[589]*y[IDX_H3II] -
                        k[609]*y[IDX_H3OII] - k[626]*y[IDX_HCNII] -
                        k[646]*y[IDX_H2COII] - k[647]*y[IDX_H3COII] -
                        k[648]*y[IDX_HCOII] - k[649]*y[IDX_HNOII] -
                        k[650]*y[IDX_N2HII] - k[651]*y[IDX_O2HII] -
                        k[683]*y[IDX_HeII] - k[684]*y[IDX_HeII] -
                        k[685]*y[IDX_HeII] - k[760]*y[IDX_NHII] -
                        k[775]*y[IDX_NH2II] - k[844]*y[IDX_OHII] -
                        k[979]*y[IDX_HI] - k[1151] - k[1305];
    IJth(jmatrix, 63, 64) = 0.0 - k[649]*y[IDX_HNCI];
    IJth(jmatrix, 63, 66) = 0.0 - k[647]*y[IDX_HNCI];
    IJth(jmatrix, 63, 75) = 0.0 - k[626]*y[IDX_HNCI];
    IJth(jmatrix, 63, 76) = 0.0 - k[775]*y[IDX_HNCI];
    IJth(jmatrix, 63, 77) = 0.0 - k[760]*y[IDX_HNCI];
    IJth(jmatrix, 63, 80) = 0.0 + k[786]*y[IDX_HCNHII] + k[867]*y[IDX_CI];
    IJth(jmatrix, 63, 82) = 0.0 - k[559]*y[IDX_HNCI];
    IJth(jmatrix, 63, 85) = 0.0 - k[609]*y[IDX_HNCI];
    IJth(jmatrix, 63, 89) = 0.0 - k[407]*y[IDX_HNCI];
    IJth(jmatrix, 63, 90) = 0.0 + k[428]*y[IDX_HCNHII] + k[1006]*y[IDX_NI];
    IJth(jmatrix, 63, 91) = 0.0 - k[646]*y[IDX_HNCI];
    IJth(jmatrix, 63, 93) = 0.0 - k[844]*y[IDX_HNCI];
    IJth(jmatrix, 63, 95) = 0.0 + k[634]*y[IDX_HCNHII];
    IJth(jmatrix, 63, 97) = 0.0 - k[683]*y[IDX_HNCI] - k[684]*y[IDX_HNCI] -
                        k[685]*y[IDX_HNCI];
    IJth(jmatrix, 63, 98) = 0.0 + k[465]*y[IDX_HCNHII];
    IJth(jmatrix, 63, 99) = 0.0 - k[589]*y[IDX_HNCI];
    IJth(jmatrix, 63, 102) = 0.0 + k[1006]*y[IDX_CH2I];
    IJth(jmatrix, 63, 105) = 0.0 + k[867]*y[IDX_NH2I] + k[1004]*y[IDX_HNCOI];
    IJth(jmatrix, 63, 106) = 0.0 - k[1]*y[IDX_HNCI];
    IJth(jmatrix, 63, 107) = 0.0 - k[648]*y[IDX_HNCI];
    IJth(jmatrix, 63, 110) = 0.0 + k[329]*y[IDX_HCNHII];
    IJth(jmatrix, 63, 113) = 0.0 - k[979]*y[IDX_HNCI];
    IJth(jmatrix, 64, 60) = 0.0 + k[807]*y[IDX_NOI];
    IJth(jmatrix, 64, 63) = 0.0 - k[649]*y[IDX_HNOII];
    IJth(jmatrix, 64, 64) = 0.0 - k[204]*y[IDX_NOI] - k[334]*y[IDX_EM] -
                        k[388]*y[IDX_CI] - k[430]*y[IDX_CH2I] -
                        k[467]*y[IDX_CHI] - k[482]*y[IDX_CNI] -
                        k[486]*y[IDX_COI] - k[550]*y[IDX_H2COI] -
                        k[568]*y[IDX_H2OI] - k[630]*y[IDX_HCNI] -
                        k[642]*y[IDX_HCOI] - k[649]*y[IDX_HNCI] -
                        k[652]*y[IDX_CO2I] - k[732]*y[IDX_N2I] -
                        k[788]*y[IDX_NH2I] - k[800]*y[IDX_NHI] -
                        k[856]*y[IDX_OHI] - k[1295];
    IJth(jmatrix, 64, 69) = 0.0 + k[524]*y[IDX_NOI];
    IJth(jmatrix, 64, 72) = 0.0 - k[652]*y[IDX_HNOII] + k[749]*y[IDX_NHII];
    IJth(jmatrix, 64, 76) = 0.0 + k[778]*y[IDX_O2I] + k[827]*y[IDX_OI];
    IJth(jmatrix, 64, 77) = 0.0 + k[749]*y[IDX_CO2I] + k[755]*y[IDX_H2OI];
    IJth(jmatrix, 64, 78) = 0.0 + k[804]*y[IDX_NHI];
    IJth(jmatrix, 64, 80) = 0.0 - k[788]*y[IDX_HNOII];
    IJth(jmatrix, 64, 82) = 0.0 + k[738]*y[IDX_NI];
    IJth(jmatrix, 64, 83) = 0.0 + k[828]*y[IDX_OI];
    IJth(jmatrix, 64, 86) = 0.0 - k[732]*y[IDX_HNOII];
    IJth(jmatrix, 64, 88) = 0.0 - k[630]*y[IDX_HNOII];
    IJth(jmatrix, 64, 90) = 0.0 - k[430]*y[IDX_HNOII];
    IJth(jmatrix, 64, 92) = 0.0 - k[800]*y[IDX_HNOII] + k[804]*y[IDX_O2II];
    IJth(jmatrix, 64, 93) = 0.0 + k[846]*y[IDX_NOI];
    IJth(jmatrix, 64, 94) = 0.0 - k[482]*y[IDX_HNOII];
    IJth(jmatrix, 64, 95) = 0.0 - k[550]*y[IDX_HNOII];
    IJth(jmatrix, 64, 96) = 0.0 - k[642]*y[IDX_HNOII];
    IJth(jmatrix, 64, 98) = 0.0 - k[467]*y[IDX_HNOII];
    IJth(jmatrix, 64, 99) = 0.0 + k[596]*y[IDX_NOI];
    IJth(jmatrix, 64, 101) = 0.0 - k[204]*y[IDX_HNOII] + k[524]*y[IDX_H2II] +
                        k[596]*y[IDX_H3II] + k[807]*y[IDX_O2HII] +
                        k[846]*y[IDX_OHII];
    IJth(jmatrix, 64, 102) = 0.0 + k[738]*y[IDX_H2OII];
    IJth(jmatrix, 64, 103) = 0.0 - k[856]*y[IDX_HNOII];
    IJth(jmatrix, 64, 104) = 0.0 + k[778]*y[IDX_NH2II];
    IJth(jmatrix, 64, 105) = 0.0 - k[388]*y[IDX_HNOII];
    IJth(jmatrix, 64, 108) = 0.0 - k[568]*y[IDX_HNOII] + k[755]*y[IDX_NHII];
    IJth(jmatrix, 64, 109) = 0.0 + k[827]*y[IDX_NH2II] + k[828]*y[IDX_NH3II];
    IJth(jmatrix, 64, 110) = 0.0 - k[334]*y[IDX_HNOII];
    IJth(jmatrix, 64, 111) = 0.0 - k[486]*y[IDX_HNOII];
    IJth(jmatrix, 65, 53) = 0.0 - k[150]*y[IDX_N2II];
    IJth(jmatrix, 65, 57) = 0.0 + k[737]*y[IDX_NI];
    IJth(jmatrix, 65, 65) = 0.0 - k[29]*y[IDX_CI] - k[41]*y[IDX_CH2I] -
                        k[58]*y[IDX_CHI] - k[69]*y[IDX_CNI] - k[74]*y[IDX_COI] -
                        k[125]*y[IDX_H2OI] - k[135]*y[IDX_HCNI] -
                        k[150]*y[IDX_MgI] - k[170]*y[IDX_H2COI] -
                        k[171]*y[IDX_HCOI] - k[172]*y[IDX_NOI] -
                        k[173]*y[IDX_O2I] - k[174]*y[IDX_NI] -
                        k[186]*y[IDX_NH2I] - k[197]*y[IDX_NH3I] -
                        k[201]*y[IDX_NHI] - k[218]*y[IDX_OI] - k[227]*y[IDX_OHI]
                        - k[337]*y[IDX_EM] - k[455]*y[IDX_CH4I] -
                        k[456]*y[IDX_CH4I] - k[539]*y[IDX_H2I] -
                        k[569]*y[IDX_H2OI] - k[730]*y[IDX_H2COI] -
                        k[731]*y[IDX_HCOI] - k[825]*y[IDX_OI] - k[1271];
    IJth(jmatrix, 65, 67) = 0.0 - k[455]*y[IDX_N2II] - k[456]*y[IDX_N2II];
    IJth(jmatrix, 65, 70) = 0.0 - k[197]*y[IDX_N2II];
    IJth(jmatrix, 65, 73) = 0.0 + k[726]*y[IDX_NHI] + k[727]*y[IDX_NOI] +
                        k[1210]*y[IDX_NI];
    IJth(jmatrix, 65, 77) = 0.0 + k[740]*y[IDX_NI];
    IJth(jmatrix, 65, 80) = 0.0 - k[186]*y[IDX_N2II];
    IJth(jmatrix, 65, 86) = 0.0 + k[144]*y[IDX_HeII];
    IJth(jmatrix, 65, 88) = 0.0 - k[135]*y[IDX_N2II];
    IJth(jmatrix, 65, 90) = 0.0 - k[41]*y[IDX_N2II];
    IJth(jmatrix, 65, 92) = 0.0 - k[201]*y[IDX_N2II] + k[726]*y[IDX_NII];
    IJth(jmatrix, 65, 94) = 0.0 - k[69]*y[IDX_N2II];
    IJth(jmatrix, 65, 95) = 0.0 - k[170]*y[IDX_N2II] - k[730]*y[IDX_N2II];
    IJth(jmatrix, 65, 96) = 0.0 - k[171]*y[IDX_N2II] - k[731]*y[IDX_N2II];
    IJth(jmatrix, 65, 97) = 0.0 + k[144]*y[IDX_N2I];
    IJth(jmatrix, 65, 98) = 0.0 - k[58]*y[IDX_N2II];
    IJth(jmatrix, 65, 101) = 0.0 - k[172]*y[IDX_N2II] + k[727]*y[IDX_NII];
    IJth(jmatrix, 65, 102) = 0.0 - k[174]*y[IDX_N2II] + k[737]*y[IDX_CNII] +
                        k[740]*y[IDX_NHII] + k[1210]*y[IDX_NII];
    IJth(jmatrix, 65, 103) = 0.0 - k[227]*y[IDX_N2II];
    IJth(jmatrix, 65, 104) = 0.0 - k[173]*y[IDX_N2II];
    IJth(jmatrix, 65, 105) = 0.0 - k[29]*y[IDX_N2II];
    IJth(jmatrix, 65, 108) = 0.0 - k[125]*y[IDX_N2II] - k[569]*y[IDX_N2II];
    IJth(jmatrix, 65, 109) = 0.0 - k[218]*y[IDX_N2II] - k[825]*y[IDX_N2II];
    IJth(jmatrix, 65, 110) = 0.0 - k[337]*y[IDX_N2II];
    IJth(jmatrix, 65, 111) = 0.0 - k[74]*y[IDX_N2II];
    IJth(jmatrix, 65, 112) = 0.0 - k[539]*y[IDX_N2II];
    IJth(jmatrix, 66, 52) = 0.0 + k[365]*y[IDX_CII] + k[397]*y[IDX_CHII] +
                        k[439]*y[IDX_CH3II] + k[493]*y[IDX_HII] +
                        k[713]*y[IDX_NII] + k[809]*y[IDX_OII] +
                        k[820]*y[IDX_O2II] + k[1120];
    IJth(jmatrix, 66, 55) = 0.0 + k[449]*y[IDX_H2COI];
    IJth(jmatrix, 66, 58) = 0.0 + k[633]*y[IDX_H2COI] + k[634]*y[IDX_H2COI];
    IJth(jmatrix, 66, 59) = 0.0 + k[735]*y[IDX_H2COI];
    IJth(jmatrix, 66, 60) = 0.0 + k[552]*y[IDX_H2COI];
    IJth(jmatrix, 66, 63) = 0.0 - k[647]*y[IDX_H3COII];
    IJth(jmatrix, 66, 64) = 0.0 + k[550]*y[IDX_H2COI];
    IJth(jmatrix, 66, 66) = 0.0 - k[317]*y[IDX_EM] - k[318]*y[IDX_EM] -
                        k[319]*y[IDX_EM] - k[320]*y[IDX_EM] - k[321]*y[IDX_EM] -
                        k[461]*y[IDX_CHI] - k[564]*y[IDX_H2OI] -
                        k[628]*y[IDX_HCNI] - k[647]*y[IDX_HNCI] -
                        k[782]*y[IDX_NH2I] - k[1249];
    IJth(jmatrix, 66, 67) = 0.0 + k[452]*y[IDX_H2COII];
    IJth(jmatrix, 66, 73) = 0.0 + k[713]*y[IDX_CH3OHI];
    IJth(jmatrix, 66, 74) = 0.0 + k[809]*y[IDX_CH3OHI];
    IJth(jmatrix, 66, 75) = 0.0 + k[622]*y[IDX_H2COI];
    IJth(jmatrix, 66, 76) = 0.0 + k[769]*y[IDX_H2COI];
    IJth(jmatrix, 66, 77) = 0.0 + k[752]*y[IDX_H2COI];
    IJth(jmatrix, 66, 78) = 0.0 + k[820]*y[IDX_CH3OHI];
    IJth(jmatrix, 66, 79) = 0.0 + k[439]*y[IDX_CH3OHI] + k[442]*y[IDX_O2I];
    IJth(jmatrix, 66, 80) = 0.0 - k[782]*y[IDX_H3COII];
    IJth(jmatrix, 66, 81) = 0.0 + k[418]*y[IDX_H2OI];
    IJth(jmatrix, 66, 82) = 0.0 + k[554]*y[IDX_H2COI];
    IJth(jmatrix, 66, 85) = 0.0 + k[607]*y[IDX_H2COI];
    IJth(jmatrix, 66, 87) = 0.0 + k[365]*y[IDX_CH3OHI];
    IJth(jmatrix, 66, 88) = 0.0 - k[628]*y[IDX_H3COII];
    IJth(jmatrix, 66, 89) = 0.0 + k[397]*y[IDX_CH3OHI] + k[400]*y[IDX_H2COI];
    IJth(jmatrix, 66, 91) = 0.0 + k[452]*y[IDX_CH4I] + k[548]*y[IDX_H2COI] +
                        k[641]*y[IDX_HCOI] + k[796]*y[IDX_NHI];
    IJth(jmatrix, 66, 92) = 0.0 + k[796]*y[IDX_H2COII];
    IJth(jmatrix, 66, 93) = 0.0 + k[839]*y[IDX_H2COI];
    IJth(jmatrix, 66, 95) = 0.0 + k[400]*y[IDX_CHII] + k[449]*y[IDX_CH4II] +
                        k[548]*y[IDX_H2COII] + k[550]*y[IDX_HNOII] +
                        k[552]*y[IDX_O2HII] + k[554]*y[IDX_H2OII] +
                        k[585]*y[IDX_H3II] + k[607]*y[IDX_H3OII] +
                        k[622]*y[IDX_HCNII] + k[633]*y[IDX_HCNHII] +
                        k[634]*y[IDX_HCNHII] + k[635]*y[IDX_HCOII] +
                        k[735]*y[IDX_N2HII] + k[752]*y[IDX_NHII] +
                        k[769]*y[IDX_NH2II] + k[839]*y[IDX_OHII];
    IJth(jmatrix, 66, 96) = 0.0 + k[641]*y[IDX_H2COII];
    IJth(jmatrix, 66, 98) = 0.0 - k[461]*y[IDX_H3COII];
    IJth(jmatrix, 66, 99) = 0.0 + k[585]*y[IDX_H2COI];
    IJth(jmatrix, 66, 104) = 0.0 + k[442]*y[IDX_CH3II];
    IJth(jmatrix, 66, 106) = 0.0 + k[493]*y[IDX_CH3OHI];
    IJth(jmatrix, 66, 107) = 0.0 + k[635]*y[IDX_H2COI];
    IJth(jmatrix, 66, 108) = 0.0 + k[418]*y[IDX_CH2II] - k[564]*y[IDX_H3COII];
    IJth(jmatrix, 66, 110) = 0.0 - k[317]*y[IDX_H3COII] - k[318]*y[IDX_H3COII] -
                        k[319]*y[IDX_H3COII] - k[320]*y[IDX_H3COII] -
                        k[321]*y[IDX_H3COII];
    IJth(jmatrix, 67, 24) = 0.0 + k[1307] + k[1308] + k[1309] + k[1310];
    IJth(jmatrix, 67, 37) = 0.0 + k[914]*y[IDX_CH3I];
    IJth(jmatrix, 67, 46) = 0.0 + k[446]*y[IDX_CH3II];
    IJth(jmatrix, 67, 51) = 0.0 + k[906]*y[IDX_CH3I];
    IJth(jmatrix, 67, 52) = 0.0 + k[439]*y[IDX_CH3II];
    IJth(jmatrix, 67, 55) = 0.0 + k[49]*y[IDX_H2COI] + k[50]*y[IDX_NH3I] +
                        k[51]*y[IDX_O2I];
    IJth(jmatrix, 67, 65) = 0.0 - k[455]*y[IDX_CH4I] - k[456]*y[IDX_CH4I];
    IJth(jmatrix, 67, 67) = 0.0 - k[52]*y[IDX_COII] - k[77]*y[IDX_HII] -
                        k[101]*y[IDX_H2II] - k[140]*y[IDX_HeII] -
                        k[156]*y[IDX_NII] - k[207]*y[IDX_OII] - k[249] -
                        k[451]*y[IDX_COII] - k[452]*y[IDX_H2COII] -
                        k[453]*y[IDX_H2OII] - k[454]*y[IDX_HCNII] -
                        k[455]*y[IDX_N2II] - k[456]*y[IDX_N2II] -
                        k[457]*y[IDX_OHII] - k[495]*y[IDX_HII] -
                        k[511]*y[IDX_H2II] - k[658]*y[IDX_HeII] -
                        k[659]*y[IDX_HeII] - k[660]*y[IDX_HeII] -
                        k[661]*y[IDX_HeII] - k[716]*y[IDX_NII] -
                        k[717]*y[IDX_NII] - k[718]*y[IDX_NII] -
                        k[810]*y[IDX_OII] - k[879]*y[IDX_CH2I] -
                        k[920]*y[IDX_CNI] - k[921]*y[IDX_O2I] -
                        k[922]*y[IDX_OHI] - k[969]*y[IDX_HI] -
                        k[1028]*y[IDX_NH2I] - k[1034]*y[IDX_NHI] -
                        k[1056]*y[IDX_OI] - k[1124] - k[1125] - k[1126] -
                        k[1127] - k[1260];
    IJth(jmatrix, 67, 68) = 0.0 - k[52]*y[IDX_CH4I] - k[451]*y[IDX_CH4I];
    IJth(jmatrix, 67, 69) = 0.0 - k[101]*y[IDX_CH4I] - k[511]*y[IDX_CH4I];
    IJth(jmatrix, 67, 70) = 0.0 + k[50]*y[IDX_CH4II] + k[908]*y[IDX_CH3I];
    IJth(jmatrix, 67, 71) = 0.0 + k[901]*y[IDX_CH3I] + k[901]*y[IDX_CH3I] +
                        k[903]*y[IDX_H2COI] + k[904]*y[IDX_H2OI] +
                        k[905]*y[IDX_HCOI] + k[906]*y[IDX_HNOI] +
                        k[907]*y[IDX_NH2I] + k[908]*y[IDX_NH3I] +
                        k[914]*y[IDX_O2HI] + k[917]*y[IDX_OHI] +
                        k[957]*y[IDX_H2I];
    IJth(jmatrix, 67, 73) = 0.0 - k[156]*y[IDX_CH4I] - k[716]*y[IDX_CH4I] -
                        k[717]*y[IDX_CH4I] - k[718]*y[IDX_CH4I];
    IJth(jmatrix, 67, 74) = 0.0 - k[207]*y[IDX_CH4I] - k[810]*y[IDX_CH4I];
    IJth(jmatrix, 67, 75) = 0.0 - k[454]*y[IDX_CH4I];
    IJth(jmatrix, 67, 79) = 0.0 + k[439]*y[IDX_CH3OHI] + k[440]*y[IDX_H2COI] +
                        k[446]*y[IDX_SiH4I];
    IJth(jmatrix, 67, 80) = 0.0 + k[907]*y[IDX_CH3I] - k[1028]*y[IDX_CH4I];
    IJth(jmatrix, 67, 82) = 0.0 - k[453]*y[IDX_CH4I];
    IJth(jmatrix, 67, 90) = 0.0 - k[879]*y[IDX_CH4I];
    IJth(jmatrix, 67, 91) = 0.0 - k[452]*y[IDX_CH4I];
    IJth(jmatrix, 67, 92) = 0.0 - k[1034]*y[IDX_CH4I];
    IJth(jmatrix, 67, 93) = 0.0 - k[457]*y[IDX_CH4I];
    IJth(jmatrix, 67, 94) = 0.0 - k[920]*y[IDX_CH4I];
    IJth(jmatrix, 67, 95) = 0.0 + k[49]*y[IDX_CH4II] + k[440]*y[IDX_CH3II] +
                        k[903]*y[IDX_CH3I];
    IJth(jmatrix, 67, 96) = 0.0 + k[905]*y[IDX_CH3I];
    IJth(jmatrix, 67, 97) = 0.0 - k[140]*y[IDX_CH4I] - k[658]*y[IDX_CH4I] -
                        k[659]*y[IDX_CH4I] - k[660]*y[IDX_CH4I] -
                        k[661]*y[IDX_CH4I];
    IJth(jmatrix, 67, 103) = 0.0 + k[917]*y[IDX_CH3I] - k[922]*y[IDX_CH4I];
    IJth(jmatrix, 67, 104) = 0.0 + k[51]*y[IDX_CH4II] - k[921]*y[IDX_CH4I];
    IJth(jmatrix, 67, 106) = 0.0 - k[77]*y[IDX_CH4I] - k[495]*y[IDX_CH4I];
    IJth(jmatrix, 67, 108) = 0.0 + k[904]*y[IDX_CH3I];
    IJth(jmatrix, 67, 109) = 0.0 - k[1056]*y[IDX_CH4I];
    IJth(jmatrix, 67, 112) = 0.0 + k[957]*y[IDX_CH3I];
    IJth(jmatrix, 67, 113) = 0.0 - k[969]*y[IDX_CH4I];
    IJth(jmatrix, 68, 42) = 0.0 + k[378]*y[IDX_CII];
    IJth(jmatrix, 68, 57) = 0.0 + k[63]*y[IDX_COI];
    IJth(jmatrix, 68, 65) = 0.0 + k[74]*y[IDX_COI];
    IJth(jmatrix, 68, 67) = 0.0 - k[52]*y[IDX_COII] - k[451]*y[IDX_COII];
    IJth(jmatrix, 68, 68) = 0.0 - k[28]*y[IDX_CI] - k[38]*y[IDX_CH2I] -
                        k[52]*y[IDX_CH4I] - k[54]*y[IDX_CHI] -
                        k[70]*y[IDX_H2COI] - k[71]*y[IDX_HCOI] -
                        k[72]*y[IDX_NOI] - k[73]*y[IDX_O2I] - k[123]*y[IDX_H2OI]
                        - k[127]*y[IDX_HI] - k[134]*y[IDX_HCNI] -
                        k[184]*y[IDX_NH2I] - k[193]*y[IDX_NH3I] -
                        k[200]*y[IDX_NHI] - k[217]*y[IDX_OI] - k[226]*y[IDX_OHI]
                        - k[304]*y[IDX_EM] - k[422]*y[IDX_CH2I] -
                        k[451]*y[IDX_CH4I] - k[458]*y[IDX_CHI] -
                        k[484]*y[IDX_H2COI] - k[532]*y[IDX_H2I] -
                        k[533]*y[IDX_H2I] - k[562]*y[IDX_H2OI] -
                        k[779]*y[IDX_NH2I] - k[792]*y[IDX_NH3I] -
                        k[795]*y[IDX_NHI] - k[851]*y[IDX_OHI] - k[1131] -
                        k[1275];
    IJth(jmatrix, 68, 69) = 0.0 + k[104]*y[IDX_COI];
    IJth(jmatrix, 68, 70) = 0.0 - k[193]*y[IDX_COII] - k[792]*y[IDX_COII];
    IJth(jmatrix, 68, 72) = 0.0 + k[367]*y[IDX_CII] + k[665]*y[IDX_HeII] +
                        k[719]*y[IDX_NII];
    IJth(jmatrix, 68, 73) = 0.0 + k[158]*y[IDX_COI] + k[719]*y[IDX_CO2I];
    IJth(jmatrix, 68, 74) = 0.0 + k[208]*y[IDX_COI] + k[472]*y[IDX_CHI] +
                        k[1195]*y[IDX_CI];
    IJth(jmatrix, 68, 78) = 0.0 + k[391]*y[IDX_CI];
    IJth(jmatrix, 68, 80) = 0.0 - k[184]*y[IDX_COII] - k[779]*y[IDX_COII];
    IJth(jmatrix, 68, 87) = 0.0 + k[367]*y[IDX_CO2I] + k[376]*y[IDX_O2I] +
                        k[378]*y[IDX_OCNI] + k[379]*y[IDX_OHI] +
                        k[1193]*y[IDX_OI];
    IJth(jmatrix, 68, 88) = 0.0 - k[134]*y[IDX_COII];
    IJth(jmatrix, 68, 89) = 0.0 + k[411]*y[IDX_O2I] + k[414]*y[IDX_OI] +
                        k[415]*y[IDX_OHI];
    IJth(jmatrix, 68, 90) = 0.0 - k[38]*y[IDX_COII] - k[422]*y[IDX_COII];
    IJth(jmatrix, 68, 92) = 0.0 - k[200]*y[IDX_COII] - k[795]*y[IDX_COII];
    IJth(jmatrix, 68, 95) = 0.0 - k[70]*y[IDX_COII] - k[484]*y[IDX_COII] +
                        k[497]*y[IDX_HII] + k[670]*y[IDX_HeII];
    IJth(jmatrix, 68, 96) = 0.0 - k[71]*y[IDX_COII] + k[500]*y[IDX_HII] +
                        k[680]*y[IDX_HeII];
    IJth(jmatrix, 68, 97) = 0.0 + k[665]*y[IDX_CO2I] + k[670]*y[IDX_H2COI] +
                        k[680]*y[IDX_HCOI];
    IJth(jmatrix, 68, 98) = 0.0 - k[54]*y[IDX_COII] - k[458]*y[IDX_COII] +
                        k[472]*y[IDX_OII];
    IJth(jmatrix, 68, 101) = 0.0 - k[72]*y[IDX_COII];
    IJth(jmatrix, 68, 103) = 0.0 - k[226]*y[IDX_COII] + k[379]*y[IDX_CII] +
                        k[415]*y[IDX_CHII] - k[851]*y[IDX_COII];
    IJth(jmatrix, 68, 104) = 0.0 - k[73]*y[IDX_COII] + k[376]*y[IDX_CII] +
                        k[411]*y[IDX_CHII];
    IJth(jmatrix, 68, 105) = 0.0 - k[28]*y[IDX_COII] + k[391]*y[IDX_O2II] +
                        k[1195]*y[IDX_OII];
    IJth(jmatrix, 68, 106) = 0.0 + k[497]*y[IDX_H2COI] + k[500]*y[IDX_HCOI];
    IJth(jmatrix, 68, 107) = 0.0 + k[1148];
    IJth(jmatrix, 68, 108) = 0.0 - k[123]*y[IDX_COII] - k[562]*y[IDX_COII];
    IJth(jmatrix, 68, 109) = 0.0 - k[217]*y[IDX_COII] + k[414]*y[IDX_CHII] +
                        k[1193]*y[IDX_CII];
    IJth(jmatrix, 68, 110) = 0.0 - k[304]*y[IDX_COII];
    IJth(jmatrix, 68, 111) = 0.0 + k[63]*y[IDX_CNII] + k[74]*y[IDX_N2II] +
                        k[104]*y[IDX_H2II] + k[158]*y[IDX_NII] +
                        k[208]*y[IDX_OII] + k[232];
    IJth(jmatrix, 68, 112) = 0.0 - k[532]*y[IDX_COII] - k[533]*y[IDX_COII];
    IJth(jmatrix, 68, 113) = 0.0 - k[127]*y[IDX_COII];
    IJth(jmatrix, 69, 27) = 0.0 + k[618]*y[IDX_HI];
    IJth(jmatrix, 69, 67) = 0.0 - k[101]*y[IDX_H2II] - k[511]*y[IDX_H2II];
    IJth(jmatrix, 69, 69) = 0.0 - k[100]*y[IDX_CH2I] - k[101]*y[IDX_CH4I] -
                        k[102]*y[IDX_CHI] - k[103]*y[IDX_CNI] -
                        k[104]*y[IDX_COI] - k[105]*y[IDX_H2COI] -
                        k[106]*y[IDX_H2OI] - k[107]*y[IDX_HCNI] -
                        k[108]*y[IDX_HCOI] - k[109]*y[IDX_NH2I] -
                        k[110]*y[IDX_NH3I] - k[111]*y[IDX_NHI] -
                        k[112]*y[IDX_NOI] - k[113]*y[IDX_O2I] -
                        k[114]*y[IDX_OHI] - k[128]*y[IDX_HI] - k[305]*y[IDX_EM]
                        - k[509]*y[IDX_CI] - k[510]*y[IDX_CH2I] -
                        k[511]*y[IDX_CH4I] - k[512]*y[IDX_CHI] -
                        k[513]*y[IDX_CNI] - k[514]*y[IDX_CO2I] -
                        k[515]*y[IDX_COI] - k[516]*y[IDX_H2I] -
                        k[517]*y[IDX_H2COI] - k[518]*y[IDX_H2OI] -
                        k[519]*y[IDX_HCOI] - k[520]*y[IDX_HeI] -
                        k[521]*y[IDX_N2I] - k[522]*y[IDX_NI] - k[523]*y[IDX_NHI]
                        - k[524]*y[IDX_NOI] - k[525]*y[IDX_O2I] -
                        k[526]*y[IDX_OI] - k[527]*y[IDX_OHI] - k[1134];
    IJth(jmatrix, 69, 70) = 0.0 - k[110]*y[IDX_H2II];
    IJth(jmatrix, 69, 72) = 0.0 - k[514]*y[IDX_H2II];
    IJth(jmatrix, 69, 80) = 0.0 - k[109]*y[IDX_H2II];
    IJth(jmatrix, 69, 86) = 0.0 - k[521]*y[IDX_H2II];
    IJth(jmatrix, 69, 88) = 0.0 - k[107]*y[IDX_H2II];
    IJth(jmatrix, 69, 90) = 0.0 - k[100]*y[IDX_H2II] - k[510]*y[IDX_H2II];
    IJth(jmatrix, 69, 92) = 0.0 - k[111]*y[IDX_H2II] - k[523]*y[IDX_H2II];
    IJth(jmatrix, 69, 94) = 0.0 - k[103]*y[IDX_H2II] - k[513]*y[IDX_H2II];
    IJth(jmatrix, 69, 95) = 0.0 - k[105]*y[IDX_H2II] - k[517]*y[IDX_H2II];
    IJth(jmatrix, 69, 96) = 0.0 - k[108]*y[IDX_H2II] + k[501]*y[IDX_HII] -
                        k[519]*y[IDX_H2II];
    IJth(jmatrix, 69, 97) = 0.0 + k[115]*y[IDX_H2I];
    IJth(jmatrix, 69, 98) = 0.0 - k[102]*y[IDX_H2II] - k[512]*y[IDX_H2II];
    IJth(jmatrix, 69, 99) = 0.0 + k[1145];
    IJth(jmatrix, 69, 100) = 0.0 - k[520]*y[IDX_H2II];
    IJth(jmatrix, 69, 101) = 0.0 - k[112]*y[IDX_H2II] - k[524]*y[IDX_H2II];
    IJth(jmatrix, 69, 102) = 0.0 - k[522]*y[IDX_H2II];
    IJth(jmatrix, 69, 103) = 0.0 - k[114]*y[IDX_H2II] - k[527]*y[IDX_H2II];
    IJth(jmatrix, 69, 104) = 0.0 - k[113]*y[IDX_H2II] - k[525]*y[IDX_H2II];
    IJth(jmatrix, 69, 105) = 0.0 - k[509]*y[IDX_H2II];
    IJth(jmatrix, 69, 106) = 0.0 + k[501]*y[IDX_HCOI] + k[1197]*y[IDX_HI];
    IJth(jmatrix, 69, 108) = 0.0 - k[106]*y[IDX_H2II] - k[518]*y[IDX_H2II];
    IJth(jmatrix, 69, 109) = 0.0 - k[526]*y[IDX_H2II];
    IJth(jmatrix, 69, 110) = 0.0 - k[305]*y[IDX_H2II];
    IJth(jmatrix, 69, 111) = 0.0 - k[104]*y[IDX_H2II] - k[515]*y[IDX_H2II];
    IJth(jmatrix, 69, 112) = 0.0 + k[115]*y[IDX_HeII] + k[234] - k[516]*y[IDX_H2II];
    IJth(jmatrix, 69, 113) = 0.0 - k[128]*y[IDX_H2II] + k[618]*y[IDX_HeHII] +
                        k[1197]*y[IDX_HII];
    IJth(jmatrix, 70, 21) = 0.0 + k[1311] + k[1312] + k[1313] + k[1314];
    IJth(jmatrix, 70, 53) = 0.0 + k[190]*y[IDX_NH3II];
    IJth(jmatrix, 70, 55) = 0.0 - k[50]*y[IDX_NH3I];
    IJth(jmatrix, 70, 62) = 0.0 + k[192]*y[IDX_NH3II];
    IJth(jmatrix, 70, 65) = 0.0 - k[197]*y[IDX_NH3I];
    IJth(jmatrix, 70, 67) = 0.0 + k[1028]*y[IDX_NH2I];
    IJth(jmatrix, 70, 68) = 0.0 - k[193]*y[IDX_NH3I] - k[792]*y[IDX_NH3I];
    IJth(jmatrix, 70, 69) = 0.0 - k[110]*y[IDX_NH3I];
    IJth(jmatrix, 70, 70) = 0.0 - k[19]*y[IDX_CII] - k[33]*y[IDX_CHII] -
                        k[50]*y[IDX_CH4II] - k[85]*y[IDX_HII] -
                        k[110]*y[IDX_H2II] - k[145]*y[IDX_HeII] -
                        k[165]*y[IDX_NII] - k[177]*y[IDX_NHII] -
                        k[181]*y[IDX_NH2II] - k[193]*y[IDX_COII] -
                        k[194]*y[IDX_H2COII] - k[195]*y[IDX_H2OII] -
                        k[196]*y[IDX_HCNII] - k[197]*y[IDX_N2II] -
                        k[198]*y[IDX_O2II] - k[213]*y[IDX_OII] -
                        k[222]*y[IDX_OHII] - k[271] - k[272] - k[273] -
                        k[374]*y[IDX_CII] - k[691]*y[IDX_HeII] -
                        k[692]*y[IDX_HeII] - k[724]*y[IDX_NII] -
                        k[725]*y[IDX_NII] - k[792]*y[IDX_COII] -
                        k[793]*y[IDX_HCNII] - k[908]*y[IDX_CH3I] -
                        k[984]*y[IDX_HI] - k[1033]*y[IDX_CNI] -
                        k[1037]*y[IDX_NHI] - k[1074]*y[IDX_OI] -
                        k[1098]*y[IDX_OHI] - k[1159] - k[1160] - k[1161] -
                        k[1267];
    IJth(jmatrix, 70, 71) = 0.0 - k[908]*y[IDX_NH3I];
    IJth(jmatrix, 70, 73) = 0.0 - k[165]*y[IDX_NH3I] - k[724]*y[IDX_NH3I] -
                        k[725]*y[IDX_NH3I];
    IJth(jmatrix, 70, 74) = 0.0 - k[213]*y[IDX_NH3I];
    IJth(jmatrix, 70, 75) = 0.0 - k[196]*y[IDX_NH3I] - k[793]*y[IDX_NH3I];
    IJth(jmatrix, 70, 76) = 0.0 - k[181]*y[IDX_NH3I];
    IJth(jmatrix, 70, 77) = 0.0 - k[177]*y[IDX_NH3I];
    IJth(jmatrix, 70, 78) = 0.0 - k[198]*y[IDX_NH3I];
    IJth(jmatrix, 70, 80) = 0.0 + k[961]*y[IDX_H2I] + k[1028]*y[IDX_CH4I] +
                        k[1032]*y[IDX_OHI];
    IJth(jmatrix, 70, 82) = 0.0 - k[195]*y[IDX_NH3I];
    IJth(jmatrix, 70, 83) = 0.0 + k[189]*y[IDX_HCOI] + k[190]*y[IDX_MgI] +
                        k[191]*y[IDX_NOI] + k[192]*y[IDX_SiI];
    IJth(jmatrix, 70, 87) = 0.0 - k[19]*y[IDX_NH3I] - k[374]*y[IDX_NH3I];
    IJth(jmatrix, 70, 89) = 0.0 - k[33]*y[IDX_NH3I];
    IJth(jmatrix, 70, 91) = 0.0 - k[194]*y[IDX_NH3I];
    IJth(jmatrix, 70, 92) = 0.0 - k[1037]*y[IDX_NH3I];
    IJth(jmatrix, 70, 93) = 0.0 - k[222]*y[IDX_NH3I];
    IJth(jmatrix, 70, 94) = 0.0 - k[1033]*y[IDX_NH3I];
    IJth(jmatrix, 70, 96) = 0.0 + k[189]*y[IDX_NH3II];
    IJth(jmatrix, 70, 97) = 0.0 - k[145]*y[IDX_NH3I] - k[691]*y[IDX_NH3I] -
                        k[692]*y[IDX_NH3I];
    IJth(jmatrix, 70, 101) = 0.0 + k[191]*y[IDX_NH3II];
    IJth(jmatrix, 70, 103) = 0.0 + k[1032]*y[IDX_NH2I] - k[1098]*y[IDX_NH3I];
    IJth(jmatrix, 70, 106) = 0.0 - k[85]*y[IDX_NH3I];
    IJth(jmatrix, 70, 109) = 0.0 - k[1074]*y[IDX_NH3I];
    IJth(jmatrix, 70, 112) = 0.0 + k[961]*y[IDX_NH2I];
    IJth(jmatrix, 70, 113) = 0.0 - k[984]*y[IDX_NH3I];
    IJth(jmatrix, 71, 37) = 0.0 - k[914]*y[IDX_CH3I];
    IJth(jmatrix, 71, 39) = 0.0 - k[909]*y[IDX_CH3I];
    IJth(jmatrix, 71, 51) = 0.0 + k[883]*y[IDX_CH2I] - k[906]*y[IDX_CH3I];
    IJth(jmatrix, 71, 52) = 0.0 + k[248] + k[656]*y[IDX_HeII] + k[714]*y[IDX_NII] +
                        k[860]*y[IDX_SiII] + k[1121];
    IJth(jmatrix, 71, 53) = 0.0 + k[47]*y[IDX_CH3II];
    IJth(jmatrix, 71, 55) = 0.0 + k[302]*y[IDX_EM] + k[447]*y[IDX_CO2I] +
                        k[448]*y[IDX_COI] + k[449]*y[IDX_H2COI] +
                        k[450]*y[IDX_H2OI];
    IJth(jmatrix, 71, 61) = 0.0 + k[860]*y[IDX_CH3OHI];
    IJth(jmatrix, 71, 67) = 0.0 + k[451]*y[IDX_COII] + k[452]*y[IDX_H2COII] +
                        k[453]*y[IDX_H2OII] + k[454]*y[IDX_HCNII] +
                        k[661]*y[IDX_HeII] + k[879]*y[IDX_CH2I] +
                        k[879]*y[IDX_CH2I] + k[920]*y[IDX_CNI] +
                        k[921]*y[IDX_O2I] + k[922]*y[IDX_OHI] + k[969]*y[IDX_HI]
                        + k[1028]*y[IDX_NH2I] + k[1034]*y[IDX_NHI] +
                        k[1056]*y[IDX_OI] + k[1125];
    IJth(jmatrix, 71, 68) = 0.0 + k[451]*y[IDX_CH4I];
    IJth(jmatrix, 71, 70) = 0.0 - k[908]*y[IDX_CH3I];
    IJth(jmatrix, 71, 71) = 0.0 - k[76]*y[IDX_HII] - k[244] - k[245] - k[246] -
                        k[578]*y[IDX_H3II] - k[655]*y[IDX_HeII] -
                        k[901]*y[IDX_CH3I] - k[901]*y[IDX_CH3I] -
                        k[901]*y[IDX_CH3I] - k[901]*y[IDX_CH3I] -
                        k[902]*y[IDX_CNI] - k[903]*y[IDX_H2COI] -
                        k[904]*y[IDX_H2OI] - k[905]*y[IDX_HCOI] -
                        k[906]*y[IDX_HNOI] - k[907]*y[IDX_NH2I] -
                        k[908]*y[IDX_NH3I] - k[909]*y[IDX_NO2I] -
                        k[910]*y[IDX_NOI] - k[911]*y[IDX_O2I] -
                        k[912]*y[IDX_O2I] - k[913]*y[IDX_O2I] -
                        k[914]*y[IDX_O2HI] - k[915]*y[IDX_OI] - k[916]*y[IDX_OI]
                        - k[917]*y[IDX_OHI] - k[918]*y[IDX_OHI] -
                        k[919]*y[IDX_OHI] - k[957]*y[IDX_H2I] - k[968]*y[IDX_HI]
                        - k[1008]*y[IDX_NI] - k[1009]*y[IDX_NI] -
                        k[1010]*y[IDX_NI] - k[1116] - k[1117] - k[1118] -
                        k[1259];
    IJth(jmatrix, 71, 72) = 0.0 + k[447]*y[IDX_CH4II];
    IJth(jmatrix, 71, 73) = 0.0 + k[714]*y[IDX_CH3OHI];
    IJth(jmatrix, 71, 75) = 0.0 + k[454]*y[IDX_CH4I];
    IJth(jmatrix, 71, 79) = 0.0 + k[46]*y[IDX_HCOI] + k[47]*y[IDX_MgI] +
                        k[48]*y[IDX_NOI] + k[1215]*y[IDX_EM];
    IJth(jmatrix, 71, 80) = 0.0 - k[907]*y[IDX_CH3I] + k[1028]*y[IDX_CH4I];
    IJth(jmatrix, 71, 81) = 0.0 + k[417]*y[IDX_H2COI];
    IJth(jmatrix, 71, 82) = 0.0 + k[453]*y[IDX_CH4I];
    IJth(jmatrix, 71, 90) = 0.0 + k[878]*y[IDX_CH2I] + k[878]*y[IDX_CH2I] +
                        k[879]*y[IDX_CH4I] + k[879]*y[IDX_CH4I] +
                        k[881]*y[IDX_H2COI] + k[882]*y[IDX_HCOI] +
                        k[883]*y[IDX_HNOI] + k[900]*y[IDX_OHI] +
                        k[956]*y[IDX_H2I];
    IJth(jmatrix, 71, 91) = 0.0 + k[452]*y[IDX_CH4I];
    IJth(jmatrix, 71, 92) = 0.0 + k[1034]*y[IDX_CH4I];
    IJth(jmatrix, 71, 94) = 0.0 - k[902]*y[IDX_CH3I] + k[920]*y[IDX_CH4I];
    IJth(jmatrix, 71, 95) = 0.0 + k[417]*y[IDX_CH2II] + k[449]*y[IDX_CH4II] +
                        k[881]*y[IDX_CH2I] - k[903]*y[IDX_CH3I];
    IJth(jmatrix, 71, 96) = 0.0 + k[46]*y[IDX_CH3II] + k[882]*y[IDX_CH2I] -
                        k[905]*y[IDX_CH3I];
    IJth(jmatrix, 71, 97) = 0.0 - k[655]*y[IDX_CH3I] + k[656]*y[IDX_CH3OHI] +
                        k[661]*y[IDX_CH4I];
    IJth(jmatrix, 71, 98) = 0.0 + k[1201]*y[IDX_H2I];
    IJth(jmatrix, 71, 99) = 0.0 - k[578]*y[IDX_CH3I];
    IJth(jmatrix, 71, 101) = 0.0 + k[48]*y[IDX_CH3II] - k[910]*y[IDX_CH3I];
    IJth(jmatrix, 71, 102) = 0.0 - k[1008]*y[IDX_CH3I] - k[1009]*y[IDX_CH3I] -
                        k[1010]*y[IDX_CH3I];
    IJth(jmatrix, 71, 103) = 0.0 + k[900]*y[IDX_CH2I] - k[917]*y[IDX_CH3I] -
                        k[918]*y[IDX_CH3I] - k[919]*y[IDX_CH3I] +
                        k[922]*y[IDX_CH4I];
    IJth(jmatrix, 71, 104) = 0.0 - k[911]*y[IDX_CH3I] - k[912]*y[IDX_CH3I] -
                        k[913]*y[IDX_CH3I] + k[921]*y[IDX_CH4I];
    IJth(jmatrix, 71, 106) = 0.0 - k[76]*y[IDX_CH3I];
    IJth(jmatrix, 71, 108) = 0.0 + k[450]*y[IDX_CH4II] - k[904]*y[IDX_CH3I];
    IJth(jmatrix, 71, 109) = 0.0 - k[915]*y[IDX_CH3I] - k[916]*y[IDX_CH3I] +
                        k[1056]*y[IDX_CH4I];
    IJth(jmatrix, 71, 110) = 0.0 + k[302]*y[IDX_CH4II] + k[1215]*y[IDX_CH3II];
    IJth(jmatrix, 71, 111) = 0.0 + k[448]*y[IDX_CH4II];
    IJth(jmatrix, 71, 112) = 0.0 + k[956]*y[IDX_CH2I] - k[957]*y[IDX_CH3I] +
                        k[1201]*y[IDX_CHI];
    IJth(jmatrix, 71, 113) = 0.0 - k[968]*y[IDX_CH3I] + k[969]*y[IDX_CH4I];
    IJth(jmatrix, 72, 14) = 0.0 + k[1383] + k[1384] + k[1385] + k[1386];
    IJth(jmatrix, 72, 37) = 0.0 + k[954]*y[IDX_COI];
    IJth(jmatrix, 72, 39) = 0.0 + k[952]*y[IDX_COI];
    IJth(jmatrix, 72, 42) = 0.0 + k[1053]*y[IDX_NOI] + k[1054]*y[IDX_O2I];
    IJth(jmatrix, 72, 49) = 0.0 + k[490]*y[IDX_COI];
    IJth(jmatrix, 72, 50) = 0.0 + k[331]*y[IDX_EM] + k[387]*y[IDX_CI] +
                        k[485]*y[IDX_COI] + k[567]*y[IDX_H2OI];
    IJth(jmatrix, 72, 51) = 0.0 + k[951]*y[IDX_COI];
    IJth(jmatrix, 72, 55) = 0.0 - k[447]*y[IDX_CO2I];
    IJth(jmatrix, 72, 59) = 0.0 - k[734]*y[IDX_CO2I];
    IJth(jmatrix, 72, 60) = 0.0 - k[821]*y[IDX_CO2I];
    IJth(jmatrix, 72, 62) = 0.0 - k[1103]*y[IDX_CO2I];
    IJth(jmatrix, 72, 64) = 0.0 - k[652]*y[IDX_CO2I];
    IJth(jmatrix, 72, 69) = 0.0 - k[514]*y[IDX_CO2I];
    IJth(jmatrix, 72, 72) = 0.0 - k[252] - k[367]*y[IDX_CII] - k[398]*y[IDX_CHII] -
                        k[416]*y[IDX_CH2II] - k[447]*y[IDX_CH4II] -
                        k[496]*y[IDX_HII] - k[514]*y[IDX_H2II] -
                        k[582]*y[IDX_H3II] - k[620]*y[IDX_HCNII] -
                        k[652]*y[IDX_HNOII] - k[665]*y[IDX_HeII] -
                        k[666]*y[IDX_HeII] - k[667]*y[IDX_HeII] -
                        k[668]*y[IDX_HeII] - k[719]*y[IDX_NII] -
                        k[734]*y[IDX_N2HII] - k[748]*y[IDX_NHII] -
                        k[749]*y[IDX_NHII] - k[750]*y[IDX_NHII] -
                        k[812]*y[IDX_OII] - k[821]*y[IDX_O2HII] -
                        k[837]*y[IDX_OHII] - k[923]*y[IDX_CHI] -
                        k[971]*y[IDX_HI] - k[1012]*y[IDX_NI] - k[1059]*y[IDX_OI]
                        - k[1103]*y[IDX_SiI] - k[1132] - k[1258];
    IJth(jmatrix, 72, 73) = 0.0 - k[719]*y[IDX_CO2I];
    IJth(jmatrix, 72, 74) = 0.0 - k[812]*y[IDX_CO2I];
    IJth(jmatrix, 72, 75) = 0.0 - k[620]*y[IDX_CO2I];
    IJth(jmatrix, 72, 77) = 0.0 - k[748]*y[IDX_CO2I] - k[749]*y[IDX_CO2I] -
                        k[750]*y[IDX_CO2I];
    IJth(jmatrix, 72, 81) = 0.0 - k[416]*y[IDX_CO2I];
    IJth(jmatrix, 72, 87) = 0.0 - k[367]*y[IDX_CO2I];
    IJth(jmatrix, 72, 89) = 0.0 - k[398]*y[IDX_CO2I];
    IJth(jmatrix, 72, 90) = 0.0 + k[889]*y[IDX_O2I] + k[890]*y[IDX_O2I];
    IJth(jmatrix, 72, 93) = 0.0 - k[837]*y[IDX_CO2I];
    IJth(jmatrix, 72, 96) = 0.0 + k[1001]*y[IDX_O2I] + k[1066]*y[IDX_OI];
    IJth(jmatrix, 72, 97) = 0.0 - k[665]*y[IDX_CO2I] - k[666]*y[IDX_CO2I] -
                        k[667]*y[IDX_CO2I] - k[668]*y[IDX_CO2I];
    IJth(jmatrix, 72, 98) = 0.0 - k[923]*y[IDX_CO2I] + k[933]*y[IDX_O2I];
    IJth(jmatrix, 72, 99) = 0.0 - k[582]*y[IDX_CO2I];
    IJth(jmatrix, 72, 101) = 0.0 + k[1053]*y[IDX_OCNI];
    IJth(jmatrix, 72, 102) = 0.0 - k[1012]*y[IDX_CO2I];
    IJth(jmatrix, 72, 103) = 0.0 + k[1092]*y[IDX_COI];
    IJth(jmatrix, 72, 104) = 0.0 + k[889]*y[IDX_CH2I] + k[890]*y[IDX_CH2I] +
                        k[933]*y[IDX_CHI] + k[953]*y[IDX_COI] +
                        k[1001]*y[IDX_HCOI] + k[1054]*y[IDX_OCNI];
    IJth(jmatrix, 72, 105) = 0.0 + k[387]*y[IDX_HCO2II];
    IJth(jmatrix, 72, 106) = 0.0 - k[496]*y[IDX_CO2I];
    IJth(jmatrix, 72, 108) = 0.0 + k[567]*y[IDX_HCO2II];
    IJth(jmatrix, 72, 109) = 0.0 - k[1059]*y[IDX_CO2I] + k[1066]*y[IDX_HCOI];
    IJth(jmatrix, 72, 110) = 0.0 + k[331]*y[IDX_HCO2II];
    IJth(jmatrix, 72, 111) = 0.0 + k[485]*y[IDX_HCO2II] + k[490]*y[IDX_SiOII] +
                        k[951]*y[IDX_HNOI] + k[952]*y[IDX_NO2I] +
                        k[953]*y[IDX_O2I] + k[954]*y[IDX_O2HI] +
                        k[1092]*y[IDX_OHI];
    IJth(jmatrix, 72, 113) = 0.0 - k[971]*y[IDX_CO2I];
    IJth(jmatrix, 73, 52) = 0.0 - k[712]*y[IDX_NII] - k[713]*y[IDX_NII] -
                        k[714]*y[IDX_NII] - k[715]*y[IDX_NII];
    IJth(jmatrix, 73, 53) = 0.0 - k[163]*y[IDX_NII];
    IJth(jmatrix, 73, 65) = 0.0 + k[174]*y[IDX_NI];
    IJth(jmatrix, 73, 67) = 0.0 - k[156]*y[IDX_NII] - k[716]*y[IDX_NII] -
                        k[717]*y[IDX_NII] - k[718]*y[IDX_NII];
    IJth(jmatrix, 73, 70) = 0.0 - k[165]*y[IDX_NII] - k[724]*y[IDX_NII] -
                        k[725]*y[IDX_NII];
    IJth(jmatrix, 73, 72) = 0.0 - k[719]*y[IDX_NII];
    IJth(jmatrix, 73, 73) = 0.0 - k[57]*y[IDX_CHI] - k[155]*y[IDX_CH2I] -
                        k[156]*y[IDX_CH4I] - k[157]*y[IDX_CNI] -
                        k[158]*y[IDX_COI] - k[159]*y[IDX_H2COI] -
                        k[160]*y[IDX_H2OI] - k[161]*y[IDX_HCNI] -
                        k[162]*y[IDX_HCOI] - k[163]*y[IDX_MgI] -
                        k[164]*y[IDX_NH2I] - k[165]*y[IDX_NH3I] -
                        k[166]*y[IDX_NHI] - k[167]*y[IDX_NOI] -
                        k[168]*y[IDX_O2I] - k[169]*y[IDX_OHI] -
                        k[468]*y[IDX_CHI] - k[538]*y[IDX_H2I] -
                        k[712]*y[IDX_CH3OHI] - k[713]*y[IDX_CH3OHI] -
                        k[714]*y[IDX_CH3OHI] - k[715]*y[IDX_CH3OHI] -
                        k[716]*y[IDX_CH4I] - k[717]*y[IDX_CH4I] -
                        k[718]*y[IDX_CH4I] - k[719]*y[IDX_CO2I] -
                        k[720]*y[IDX_COI] - k[721]*y[IDX_H2COI] -
                        k[722]*y[IDX_H2COI] - k[723]*y[IDX_HCOI] -
                        k[724]*y[IDX_NH3I] - k[725]*y[IDX_NH3I] -
                        k[726]*y[IDX_NHI] - k[727]*y[IDX_NOI] -
                        k[728]*y[IDX_O2I] - k[729]*y[IDX_O2I] -
                        k[1210]*y[IDX_NI] - k[1220]*y[IDX_EM] - k[1268];
    IJth(jmatrix, 73, 80) = 0.0 - k[164]*y[IDX_NII] + k[689]*y[IDX_HeII];
    IJth(jmatrix, 73, 86) = 0.0 + k[688]*y[IDX_HeII];
    IJth(jmatrix, 73, 88) = 0.0 - k[161]*y[IDX_NII] + k[677]*y[IDX_HeII];
    IJth(jmatrix, 73, 90) = 0.0 - k[155]*y[IDX_NII];
    IJth(jmatrix, 73, 92) = 0.0 - k[166]*y[IDX_NII] + k[693]*y[IDX_HeII] -
                        k[726]*y[IDX_NII];
    IJth(jmatrix, 73, 94) = 0.0 - k[157]*y[IDX_NII] + k[663]*y[IDX_HeII];
    IJth(jmatrix, 73, 95) = 0.0 - k[159]*y[IDX_NII] - k[721]*y[IDX_NII] -
                        k[722]*y[IDX_NII];
    IJth(jmatrix, 73, 96) = 0.0 - k[162]*y[IDX_NII] - k[723]*y[IDX_NII];
    IJth(jmatrix, 73, 97) = 0.0 + k[663]*y[IDX_CNI] + k[677]*y[IDX_HCNI] +
                        k[688]*y[IDX_N2I] + k[689]*y[IDX_NH2I] +
                        k[693]*y[IDX_NHI] + k[695]*y[IDX_NOI];
    IJth(jmatrix, 73, 98) = 0.0 - k[57]*y[IDX_NII] - k[468]*y[IDX_NII];
    IJth(jmatrix, 73, 101) = 0.0 - k[167]*y[IDX_NII] + k[695]*y[IDX_HeII] -
                        k[727]*y[IDX_NII];
    IJth(jmatrix, 73, 102) = 0.0 + k[174]*y[IDX_N2II] + k[238] + k[268] -
                        k[1210]*y[IDX_NII];
    IJth(jmatrix, 73, 103) = 0.0 - k[169]*y[IDX_NII];
    IJth(jmatrix, 73, 104) = 0.0 - k[168]*y[IDX_NII] - k[728]*y[IDX_NII] -
                        k[729]*y[IDX_NII];
    IJth(jmatrix, 73, 108) = 0.0 - k[160]*y[IDX_NII];
    IJth(jmatrix, 73, 110) = 0.0 - k[1220]*y[IDX_NII];
    IJth(jmatrix, 73, 111) = 0.0 - k[158]*y[IDX_NII] - k[720]*y[IDX_NII];
    IJth(jmatrix, 73, 112) = 0.0 - k[538]*y[IDX_NII];
    IJth(jmatrix, 74, 39) = 0.0 - k[818]*y[IDX_OII];
    IJth(jmatrix, 74, 42) = 0.0 + k[698]*y[IDX_HeII];
    IJth(jmatrix, 74, 52) = 0.0 - k[808]*y[IDX_OII] - k[809]*y[IDX_OII];
    IJth(jmatrix, 74, 56) = 0.0 + k[711]*y[IDX_HeII];
    IJth(jmatrix, 74, 57) = 0.0 + k[216]*y[IDX_OI];
    IJth(jmatrix, 74, 65) = 0.0 + k[218]*y[IDX_OI];
    IJth(jmatrix, 74, 67) = 0.0 - k[207]*y[IDX_OII] - k[810]*y[IDX_OII];
    IJth(jmatrix, 74, 68) = 0.0 + k[217]*y[IDX_OI];
    IJth(jmatrix, 74, 70) = 0.0 - k[213]*y[IDX_OII];
    IJth(jmatrix, 74, 72) = 0.0 + k[666]*y[IDX_HeII] - k[812]*y[IDX_OII];
    IJth(jmatrix, 74, 73) = 0.0 + k[729]*y[IDX_O2I];
    IJth(jmatrix, 74, 74) = 0.0 - k[43]*y[IDX_CH2I] - k[60]*y[IDX_CHI] -
                        k[131]*y[IDX_HI] - k[202]*y[IDX_NHI] -
                        k[207]*y[IDX_CH4I] - k[208]*y[IDX_COI] -
                        k[209]*y[IDX_H2COI] - k[210]*y[IDX_H2OI] -
                        k[211]*y[IDX_HCOI] - k[212]*y[IDX_NH2I] -
                        k[213]*y[IDX_NH3I] - k[214]*y[IDX_O2I] -
                        k[215]*y[IDX_OHI] - k[472]*y[IDX_CHI] -
                        k[543]*y[IDX_H2I] - k[803]*y[IDX_NHI] -
                        k[808]*y[IDX_CH3OHI] - k[809]*y[IDX_CH3OHI] -
                        k[810]*y[IDX_CH4I] - k[811]*y[IDX_CNI] -
                        k[812]*y[IDX_CO2I] - k[813]*y[IDX_H2COI] -
                        k[814]*y[IDX_HCNI] - k[815]*y[IDX_HCNI] -
                        k[816]*y[IDX_HCOI] - k[817]*y[IDX_N2I] -
                        k[818]*y[IDX_NO2I] - k[819]*y[IDX_OHI] -
                        k[1195]*y[IDX_CI] - k[1221]*y[IDX_EM] - k[1269];
    IJth(jmatrix, 74, 78) = 0.0 + k[1167];
    IJth(jmatrix, 74, 80) = 0.0 - k[212]*y[IDX_OII];
    IJth(jmatrix, 74, 86) = 0.0 - k[817]*y[IDX_OII];
    IJth(jmatrix, 74, 87) = 0.0 + k[377]*y[IDX_O2I];
    IJth(jmatrix, 74, 88) = 0.0 - k[814]*y[IDX_OII] - k[815]*y[IDX_OII];
    IJth(jmatrix, 74, 89) = 0.0 + k[413]*y[IDX_O2I];
    IJth(jmatrix, 74, 90) = 0.0 - k[43]*y[IDX_OII];
    IJth(jmatrix, 74, 92) = 0.0 - k[202]*y[IDX_OII] - k[803]*y[IDX_OII];
    IJth(jmatrix, 74, 93) = 0.0 + k[1173];
    IJth(jmatrix, 74, 94) = 0.0 - k[811]*y[IDX_OII];
    IJth(jmatrix, 74, 95) = 0.0 - k[209]*y[IDX_OII] - k[813]*y[IDX_OII];
    IJth(jmatrix, 74, 96) = 0.0 - k[211]*y[IDX_OII] - k[816]*y[IDX_OII];
    IJth(jmatrix, 74, 97) = 0.0 + k[666]*y[IDX_CO2I] + k[694]*y[IDX_NOI] +
                        k[696]*y[IDX_O2I] + k[698]*y[IDX_OCNI] +
                        k[699]*y[IDX_OHI] + k[711]*y[IDX_SiOI];
    IJth(jmatrix, 74, 98) = 0.0 - k[60]*y[IDX_OII] - k[472]*y[IDX_OII];
    IJth(jmatrix, 74, 101) = 0.0 + k[694]*y[IDX_HeII];
    IJth(jmatrix, 74, 103) = 0.0 - k[215]*y[IDX_OII] + k[699]*y[IDX_HeII] -
                        k[819]*y[IDX_OII];
    IJth(jmatrix, 74, 104) = 0.0 - k[214]*y[IDX_OII] + k[377]*y[IDX_CII] +
                        k[413]*y[IDX_CHII] + k[696]*y[IDX_HeII] +
                        k[729]*y[IDX_NII];
    IJth(jmatrix, 74, 105) = 0.0 - k[1195]*y[IDX_OII];
    IJth(jmatrix, 74, 106) = 0.0 + k[89]*y[IDX_OI];
    IJth(jmatrix, 74, 108) = 0.0 - k[210]*y[IDX_OII];
    IJth(jmatrix, 74, 109) = 0.0 + k[89]*y[IDX_HII] + k[216]*y[IDX_CNII] +
                        k[217]*y[IDX_COII] + k[218]*y[IDX_N2II] + k[239] +
                        k[282];
    IJth(jmatrix, 74, 110) = 0.0 - k[1221]*y[IDX_OII];
    IJth(jmatrix, 74, 111) = 0.0 - k[208]*y[IDX_OII];
    IJth(jmatrix, 74, 112) = 0.0 - k[543]*y[IDX_OII];
    IJth(jmatrix, 74, 113) = 0.0 - k[131]*y[IDX_OII];
    IJth(jmatrix, 75, 57) = 0.0 + k[65]*y[IDX_HCNI] + k[480]*y[IDX_HCOI] +
                        k[531]*y[IDX_H2I] + k[560]*y[IDX_H2OI];
    IJth(jmatrix, 75, 60) = 0.0 + k[483]*y[IDX_CNI];
    IJth(jmatrix, 75, 63) = 0.0 - k[626]*y[IDX_HCNII];
    IJth(jmatrix, 75, 64) = 0.0 + k[482]*y[IDX_CNI];
    IJth(jmatrix, 75, 65) = 0.0 + k[135]*y[IDX_HCNI];
    IJth(jmatrix, 75, 67) = 0.0 - k[454]*y[IDX_HCNII] + k[717]*y[IDX_NII];
    IJth(jmatrix, 75, 68) = 0.0 + k[134]*y[IDX_HCNI];
    IJth(jmatrix, 75, 69) = 0.0 + k[107]*y[IDX_HCNI] + k[513]*y[IDX_CNI];
    IJth(jmatrix, 75, 70) = 0.0 - k[196]*y[IDX_HCNII] + k[374]*y[IDX_CII] -
                        k[793]*y[IDX_HCNII];
    IJth(jmatrix, 75, 72) = 0.0 - k[620]*y[IDX_HCNII];
    IJth(jmatrix, 75, 73) = 0.0 + k[161]*y[IDX_HCNI] + k[717]*y[IDX_CH4I];
    IJth(jmatrix, 75, 75) = 0.0 - k[124]*y[IDX_H2OI] - k[129]*y[IDX_HI] -
                        k[132]*y[IDX_NOI] - k[133]*y[IDX_O2I] -
                        k[196]*y[IDX_NH3I] - k[326]*y[IDX_EM] - k[385]*y[IDX_CI]
                        - k[426]*y[IDX_CH2I] - k[454]*y[IDX_CH4I] -
                        k[463]*y[IDX_CHI] - k[535]*y[IDX_H2I] -
                        k[565]*y[IDX_H2OI] - k[620]*y[IDX_CO2I] -
                        k[621]*y[IDX_COI] - k[622]*y[IDX_H2COI] -
                        k[623]*y[IDX_HCNI] - k[624]*y[IDX_HCOI] -
                        k[625]*y[IDX_HCOI] - k[626]*y[IDX_HNCI] -
                        k[784]*y[IDX_NH2I] - k[793]*y[IDX_NH3I] -
                        k[798]*y[IDX_NHI] - k[853]*y[IDX_OHI] - k[1282];
    IJth(jmatrix, 75, 77) = 0.0 + k[747]*y[IDX_CNI];
    IJth(jmatrix, 75, 80) = 0.0 + k[373]*y[IDX_CII] + k[409]*y[IDX_CHII] -
                        k[784]*y[IDX_HCNII];
    IJth(jmatrix, 75, 81) = 0.0 + k[736]*y[IDX_NI];
    IJth(jmatrix, 75, 87) = 0.0 + k[373]*y[IDX_NH2I] + k[374]*y[IDX_NH3I];
    IJth(jmatrix, 75, 88) = 0.0 + k[65]*y[IDX_CNII] + k[81]*y[IDX_HII] +
                        k[107]*y[IDX_H2II] + k[134]*y[IDX_COII] +
                        k[135]*y[IDX_N2II] + k[161]*y[IDX_NII] -
                        k[623]*y[IDX_HCNII];
    IJth(jmatrix, 75, 89) = 0.0 + k[409]*y[IDX_NH2I];
    IJth(jmatrix, 75, 90) = 0.0 - k[426]*y[IDX_HCNII];
    IJth(jmatrix, 75, 92) = 0.0 - k[798]*y[IDX_HCNII];
    IJth(jmatrix, 75, 93) = 0.0 + k[836]*y[IDX_CNI];
    IJth(jmatrix, 75, 94) = 0.0 + k[482]*y[IDX_HNOII] + k[483]*y[IDX_O2HII] +
                        k[513]*y[IDX_H2II] + k[581]*y[IDX_H3II] +
                        k[747]*y[IDX_NHII] + k[836]*y[IDX_OHII];
    IJth(jmatrix, 75, 95) = 0.0 - k[622]*y[IDX_HCNII];
    IJth(jmatrix, 75, 96) = 0.0 + k[480]*y[IDX_CNII] - k[624]*y[IDX_HCNII] -
                        k[625]*y[IDX_HCNII];
    IJth(jmatrix, 75, 98) = 0.0 - k[463]*y[IDX_HCNII];
    IJth(jmatrix, 75, 99) = 0.0 + k[581]*y[IDX_CNI];
    IJth(jmatrix, 75, 101) = 0.0 - k[132]*y[IDX_HCNII];
    IJth(jmatrix, 75, 102) = 0.0 + k[736]*y[IDX_CH2II];
    IJth(jmatrix, 75, 103) = 0.0 - k[853]*y[IDX_HCNII];
    IJth(jmatrix, 75, 104) = 0.0 - k[133]*y[IDX_HCNII];
    IJth(jmatrix, 75, 105) = 0.0 - k[385]*y[IDX_HCNII];
    IJth(jmatrix, 75, 106) = 0.0 + k[81]*y[IDX_HCNI];
    IJth(jmatrix, 75, 108) = 0.0 - k[124]*y[IDX_HCNII] + k[560]*y[IDX_CNII] -
                        k[565]*y[IDX_HCNII];
    IJth(jmatrix, 75, 110) = 0.0 - k[326]*y[IDX_HCNII];
    IJth(jmatrix, 75, 111) = 0.0 - k[621]*y[IDX_HCNII];
    IJth(jmatrix, 75, 112) = 0.0 + k[531]*y[IDX_CNII] - k[535]*y[IDX_HCNII];
    IJth(jmatrix, 75, 113) = 0.0 - k[129]*y[IDX_HCNII];
    IJth(jmatrix, 76, 28) = 0.0 + k[502]*y[IDX_HII];
    IJth(jmatrix, 76, 57) = 0.0 + k[183]*y[IDX_NH2I];
    IJth(jmatrix, 76, 59) = 0.0 + k[801]*y[IDX_NHI];
    IJth(jmatrix, 76, 60) = 0.0 + k[805]*y[IDX_NHI];
    IJth(jmatrix, 76, 63) = 0.0 - k[775]*y[IDX_NH2II];
    IJth(jmatrix, 76, 64) = 0.0 + k[800]*y[IDX_NHI];
    IJth(jmatrix, 76, 65) = 0.0 + k[186]*y[IDX_NH2I];
    IJth(jmatrix, 76, 68) = 0.0 + k[184]*y[IDX_NH2I];
    IJth(jmatrix, 76, 69) = 0.0 + k[109]*y[IDX_NH2I] + k[523]*y[IDX_NHI];
    IJth(jmatrix, 76, 70) = 0.0 - k[181]*y[IDX_NH2II] + k[692]*y[IDX_HeII] +
                        k[725]*y[IDX_NII];
    IJth(jmatrix, 76, 73) = 0.0 + k[164]*y[IDX_NH2I] + k[725]*y[IDX_NH3I];
    IJth(jmatrix, 76, 74) = 0.0 + k[212]*y[IDX_NH2I];
    IJth(jmatrix, 76, 75) = 0.0 + k[798]*y[IDX_NHI];
    IJth(jmatrix, 76, 76) = 0.0 - k[42]*y[IDX_CH2I] - k[59]*y[IDX_CHI] -
                        k[180]*y[IDX_HCOI] - k[181]*y[IDX_NH3I] -
                        k[182]*y[IDX_NOI] - k[341]*y[IDX_EM] - k[342]*y[IDX_EM]
                        - k[433]*y[IDX_CH2I] - k[471]*y[IDX_CHI] -
                        k[542]*y[IDX_H2I] - k[741]*y[IDX_NI] -
                        k[769]*y[IDX_H2COI] - k[770]*y[IDX_H2COI] -
                        k[771]*y[IDX_H2OI] - k[772]*y[IDX_H2OI] -
                        k[773]*y[IDX_HCNI] - k[774]*y[IDX_HCOI] -
                        k[775]*y[IDX_HNCI] - k[776]*y[IDX_NH2I] -
                        k[777]*y[IDX_O2I] - k[778]*y[IDX_O2I] -
                        k[802]*y[IDX_NHI] - k[827]*y[IDX_OI] - k[1279];
    IJth(jmatrix, 76, 77) = 0.0 + k[541]*y[IDX_H2I] + k[757]*y[IDX_H2OI] +
                        k[763]*y[IDX_NHI];
    IJth(jmatrix, 76, 78) = 0.0 + k[187]*y[IDX_NH2I];
    IJth(jmatrix, 76, 80) = 0.0 + k[84]*y[IDX_HII] + k[109]*y[IDX_H2II] +
                        k[164]*y[IDX_NII] + k[183]*y[IDX_CNII] +
                        k[184]*y[IDX_COII] + k[185]*y[IDX_H2OII] +
                        k[186]*y[IDX_N2II] + k[187]*y[IDX_O2II] +
                        k[188]*y[IDX_OHII] + k[212]*y[IDX_OII] + k[269] -
                        k[776]*y[IDX_NH2II] + k[1157];
    IJth(jmatrix, 76, 82) = 0.0 + k[185]*y[IDX_NH2I];
    IJth(jmatrix, 76, 88) = 0.0 - k[773]*y[IDX_NH2II];
    IJth(jmatrix, 76, 90) = 0.0 - k[42]*y[IDX_NH2II] - k[433]*y[IDX_NH2II];
    IJth(jmatrix, 76, 92) = 0.0 + k[523]*y[IDX_H2II] + k[594]*y[IDX_H3II] +
                        k[763]*y[IDX_NHII] + k[798]*y[IDX_HCNII] +
                        k[799]*y[IDX_HCOII] + k[800]*y[IDX_HNOII] +
                        k[801]*y[IDX_N2HII] - k[802]*y[IDX_NH2II] +
                        k[805]*y[IDX_O2HII] + k[806]*y[IDX_OHII];
    IJth(jmatrix, 76, 93) = 0.0 + k[188]*y[IDX_NH2I] + k[806]*y[IDX_NHI];
    IJth(jmatrix, 76, 95) = 0.0 - k[769]*y[IDX_NH2II] - k[770]*y[IDX_NH2II];
    IJth(jmatrix, 76, 96) = 0.0 - k[180]*y[IDX_NH2II] - k[774]*y[IDX_NH2II];
    IJth(jmatrix, 76, 97) = 0.0 + k[692]*y[IDX_NH3I];
    IJth(jmatrix, 76, 98) = 0.0 - k[59]*y[IDX_NH2II] - k[471]*y[IDX_NH2II];
    IJth(jmatrix, 76, 99) = 0.0 + k[594]*y[IDX_NHI];
    IJth(jmatrix, 76, 101) = 0.0 - k[182]*y[IDX_NH2II];
    IJth(jmatrix, 76, 102) = 0.0 - k[741]*y[IDX_NH2II];
    IJth(jmatrix, 76, 104) = 0.0 - k[777]*y[IDX_NH2II] - k[778]*y[IDX_NH2II];
    IJth(jmatrix, 76, 106) = 0.0 + k[84]*y[IDX_NH2I] + k[502]*y[IDX_HNCOI];
    IJth(jmatrix, 76, 107) = 0.0 + k[799]*y[IDX_NHI];
    IJth(jmatrix, 76, 108) = 0.0 + k[757]*y[IDX_NHII] - k[771]*y[IDX_NH2II] -
                        k[772]*y[IDX_NH2II];
    IJth(jmatrix, 76, 109) = 0.0 - k[827]*y[IDX_NH2II];
    IJth(jmatrix, 76, 110) = 0.0 - k[341]*y[IDX_NH2II] - k[342]*y[IDX_NH2II];
    IJth(jmatrix, 76, 112) = 0.0 + k[541]*y[IDX_NHII] - k[542]*y[IDX_NH2II];
    IJth(jmatrix, 77, 57) = 0.0 + k[199]*y[IDX_NHI];
    IJth(jmatrix, 77, 63) = 0.0 + k[685]*y[IDX_HeII] - k[760]*y[IDX_NHII];
    IJth(jmatrix, 77, 65) = 0.0 + k[201]*y[IDX_NHI];
    IJth(jmatrix, 77, 68) = 0.0 + k[200]*y[IDX_NHI];
    IJth(jmatrix, 77, 69) = 0.0 + k[111]*y[IDX_NHI] + k[522]*y[IDX_NI];
    IJth(jmatrix, 77, 70) = 0.0 - k[177]*y[IDX_NHII] + k[691]*y[IDX_HeII];
    IJth(jmatrix, 77, 72) = 0.0 - k[748]*y[IDX_NHII] - k[749]*y[IDX_NHII] -
                        k[750]*y[IDX_NHII];
    IJth(jmatrix, 77, 73) = 0.0 + k[166]*y[IDX_NHI] + k[538]*y[IDX_H2I] +
                        k[723]*y[IDX_HCOI];
    IJth(jmatrix, 77, 74) = 0.0 + k[202]*y[IDX_NHI];
    IJth(jmatrix, 77, 77) = 0.0 - k[175]*y[IDX_H2COI] - k[176]*y[IDX_H2OI] -
                        k[177]*y[IDX_NH3I] - k[178]*y[IDX_NOI] -
                        k[179]*y[IDX_O2I] - k[340]*y[IDX_EM] - k[390]*y[IDX_CI]
                        - k[432]*y[IDX_CH2I] - k[470]*y[IDX_CHI] -
                        k[540]*y[IDX_H2I] - k[541]*y[IDX_H2I] - k[740]*y[IDX_NI]
                        - k[747]*y[IDX_CNI] - k[748]*y[IDX_CO2I] -
                        k[749]*y[IDX_CO2I] - k[750]*y[IDX_CO2I] -
                        k[751]*y[IDX_COI] - k[752]*y[IDX_H2COI] -
                        k[753]*y[IDX_H2COI] - k[754]*y[IDX_H2OI] -
                        k[755]*y[IDX_H2OI] - k[756]*y[IDX_H2OI] -
                        k[757]*y[IDX_H2OI] - k[758]*y[IDX_HCNI] -
                        k[759]*y[IDX_HCOI] - k[760]*y[IDX_HNCI] -
                        k[761]*y[IDX_N2I] - k[762]*y[IDX_NH2I] -
                        k[763]*y[IDX_NHI] - k[764]*y[IDX_NOI] -
                        k[765]*y[IDX_O2I] - k[766]*y[IDX_O2I] - k[767]*y[IDX_OI]
                        - k[768]*y[IDX_OHI] - k[1156] - k[1273];
    IJth(jmatrix, 77, 80) = 0.0 + k[690]*y[IDX_HeII] - k[762]*y[IDX_NHII];
    IJth(jmatrix, 77, 86) = 0.0 - k[761]*y[IDX_NHII];
    IJth(jmatrix, 77, 88) = 0.0 - k[758]*y[IDX_NHII];
    IJth(jmatrix, 77, 90) = 0.0 - k[432]*y[IDX_NHII];
    IJth(jmatrix, 77, 92) = 0.0 + k[86]*y[IDX_HII] + k[111]*y[IDX_H2II] +
                        k[166]*y[IDX_NII] + k[199]*y[IDX_CNII] +
                        k[200]*y[IDX_COII] + k[201]*y[IDX_N2II] +
                        k[202]*y[IDX_OII] + k[275] - k[763]*y[IDX_NHII] +
                        k[1163];
    IJth(jmatrix, 77, 94) = 0.0 - k[747]*y[IDX_NHII];
    IJth(jmatrix, 77, 95) = 0.0 - k[175]*y[IDX_NHII] - k[752]*y[IDX_NHII] -
                        k[753]*y[IDX_NHII];
    IJth(jmatrix, 77, 96) = 0.0 + k[723]*y[IDX_NII] - k[759]*y[IDX_NHII];
    IJth(jmatrix, 77, 97) = 0.0 + k[685]*y[IDX_HNCI] + k[690]*y[IDX_NH2I] +
                        k[691]*y[IDX_NH3I];
    IJth(jmatrix, 77, 98) = 0.0 - k[470]*y[IDX_NHII];
    IJth(jmatrix, 77, 101) = 0.0 - k[178]*y[IDX_NHII] - k[764]*y[IDX_NHII];
    IJth(jmatrix, 77, 102) = 0.0 + k[522]*y[IDX_H2II] - k[740]*y[IDX_NHII];
    IJth(jmatrix, 77, 103) = 0.0 - k[768]*y[IDX_NHII];
    IJth(jmatrix, 77, 104) = 0.0 - k[179]*y[IDX_NHII] - k[765]*y[IDX_NHII] -
                        k[766]*y[IDX_NHII];
    IJth(jmatrix, 77, 105) = 0.0 - k[390]*y[IDX_NHII];
    IJth(jmatrix, 77, 106) = 0.0 + k[86]*y[IDX_NHI];
    IJth(jmatrix, 77, 108) = 0.0 - k[176]*y[IDX_NHII] - k[754]*y[IDX_NHII] -
                        k[755]*y[IDX_NHII] - k[756]*y[IDX_NHII] -
                        k[757]*y[IDX_NHII];
    IJth(jmatrix, 77, 109) = 0.0 - k[767]*y[IDX_NHII];
    IJth(jmatrix, 77, 110) = 0.0 - k[340]*y[IDX_NHII];
    IJth(jmatrix, 77, 111) = 0.0 - k[751]*y[IDX_NHII];
    IJth(jmatrix, 77, 112) = 0.0 + k[538]*y[IDX_NII] - k[540]*y[IDX_NHII] -
                        k[541]*y[IDX_NHII];
    IJth(jmatrix, 78, 52) = 0.0 - k[820]*y[IDX_O2II];
    IJth(jmatrix, 78, 53) = 0.0 - k[152]*y[IDX_O2II];
    IJth(jmatrix, 78, 55) = 0.0 + k[51]*y[IDX_O2I];
    IJth(jmatrix, 78, 57) = 0.0 + k[68]*y[IDX_O2I];
    IJth(jmatrix, 78, 62) = 0.0 - k[230]*y[IDX_O2II];
    IJth(jmatrix, 78, 65) = 0.0 + k[173]*y[IDX_O2I];
    IJth(jmatrix, 78, 68) = 0.0 + k[73]*y[IDX_O2I];
    IJth(jmatrix, 78, 69) = 0.0 + k[113]*y[IDX_O2I];
    IJth(jmatrix, 78, 70) = 0.0 - k[198]*y[IDX_O2II];
    IJth(jmatrix, 78, 72) = 0.0 + k[667]*y[IDX_HeII] + k[812]*y[IDX_OII];
    IJth(jmatrix, 78, 73) = 0.0 + k[168]*y[IDX_O2I];
    IJth(jmatrix, 78, 74) = 0.0 + k[214]*y[IDX_O2I] + k[812]*y[IDX_CO2I] +
                        k[819]*y[IDX_OHI];
    IJth(jmatrix, 78, 75) = 0.0 + k[133]*y[IDX_O2I];
    IJth(jmatrix, 78, 77) = 0.0 + k[179]*y[IDX_O2I];
    IJth(jmatrix, 78, 78) = 0.0 - k[30]*y[IDX_CI] - k[44]*y[IDX_CH2I] -
                        k[61]*y[IDX_CHI] - k[116]*y[IDX_H2COI] -
                        k[137]*y[IDX_HCOI] - k[152]*y[IDX_MgI] -
                        k[187]*y[IDX_NH2I] - k[198]*y[IDX_NH3I] -
                        k[205]*y[IDX_NOI] - k[230]*y[IDX_SiI] - k[346]*y[IDX_EM]
                        - k[391]*y[IDX_CI] - k[435]*y[IDX_CH2I] -
                        k[473]*y[IDX_CHI] - k[551]*y[IDX_H2COI] -
                        k[644]*y[IDX_HCOI] - k[742]*y[IDX_NI] -
                        k[804]*y[IDX_NHI] - k[820]*y[IDX_CH3OHI] - k[1167] -
                        k[1270];
    IJth(jmatrix, 78, 80) = 0.0 - k[187]*y[IDX_O2II];
    IJth(jmatrix, 78, 82) = 0.0 + k[121]*y[IDX_O2I] + k[823]*y[IDX_OI];
    IJth(jmatrix, 78, 90) = 0.0 - k[44]*y[IDX_O2II] - k[435]*y[IDX_O2II];
    IJth(jmatrix, 78, 92) = 0.0 - k[804]*y[IDX_O2II];
    IJth(jmatrix, 78, 93) = 0.0 + k[224]*y[IDX_O2I] + k[830]*y[IDX_OI];
    IJth(jmatrix, 78, 95) = 0.0 - k[116]*y[IDX_O2II] - k[551]*y[IDX_O2II];
    IJth(jmatrix, 78, 96) = 0.0 - k[137]*y[IDX_O2II] - k[644]*y[IDX_O2II];
    IJth(jmatrix, 78, 97) = 0.0 + k[146]*y[IDX_O2I] + k[667]*y[IDX_CO2I];
    IJth(jmatrix, 78, 98) = 0.0 - k[61]*y[IDX_O2II] - k[473]*y[IDX_O2II];
    IJth(jmatrix, 78, 101) = 0.0 - k[205]*y[IDX_O2II];
    IJth(jmatrix, 78, 102) = 0.0 - k[742]*y[IDX_O2II];
    IJth(jmatrix, 78, 103) = 0.0 + k[819]*y[IDX_OII];
    IJth(jmatrix, 78, 104) = 0.0 + k[51]*y[IDX_CH4II] + k[68]*y[IDX_CNII] +
                        k[73]*y[IDX_COII] + k[88]*y[IDX_HII] +
                        k[113]*y[IDX_H2II] + k[121]*y[IDX_H2OII] +
                        k[133]*y[IDX_HCNII] + k[146]*y[IDX_HeII] +
                        k[168]*y[IDX_NII] + k[173]*y[IDX_N2II] +
                        k[179]*y[IDX_NHII] + k[214]*y[IDX_OII] +
                        k[224]*y[IDX_OHII] + k[279] + k[1168];
    IJth(jmatrix, 78, 105) = 0.0 - k[30]*y[IDX_O2II] - k[391]*y[IDX_O2II];
    IJth(jmatrix, 78, 106) = 0.0 + k[88]*y[IDX_O2I];
    IJth(jmatrix, 78, 109) = 0.0 + k[823]*y[IDX_H2OII] + k[830]*y[IDX_OHII];
    IJth(jmatrix, 78, 110) = 0.0 - k[346]*y[IDX_O2II];
    IJth(jmatrix, 79, 46) = 0.0 - k[446]*y[IDX_CH3II];
    IJth(jmatrix, 79, 52) = 0.0 + k[366]*y[IDX_CII] + k[396]*y[IDX_CHII] -
                        k[439]*y[IDX_CH3II] + k[492]*y[IDX_HII] +
                        k[579]*y[IDX_H3II] + k[657]*y[IDX_HeII] +
                        k[715]*y[IDX_NII];
    IJth(jmatrix, 79, 53) = 0.0 - k[47]*y[IDX_CH3II];
    IJth(jmatrix, 79, 55) = 0.0 + k[617]*y[IDX_HI] + k[822]*y[IDX_OI] + k[1123];
    IJth(jmatrix, 79, 58) = 0.0 + k[427]*y[IDX_CH2I] + k[428]*y[IDX_CH2I];
    IJth(jmatrix, 79, 59) = 0.0 + k[431]*y[IDX_CH2I];
    IJth(jmatrix, 79, 60) = 0.0 + k[436]*y[IDX_CH2I];
    IJth(jmatrix, 79, 64) = 0.0 + k[430]*y[IDX_CH2I];
    IJth(jmatrix, 79, 65) = 0.0 + k[456]*y[IDX_CH4I];
    IJth(jmatrix, 79, 67) = 0.0 + k[456]*y[IDX_N2II] + k[495]*y[IDX_HII] +
                        k[511]*y[IDX_H2II] + k[660]*y[IDX_HeII] +
                        k[716]*y[IDX_NII] + k[810]*y[IDX_OII];
    IJth(jmatrix, 79, 69) = 0.0 + k[510]*y[IDX_CH2I] + k[511]*y[IDX_CH4I];
    IJth(jmatrix, 79, 71) = 0.0 + k[76]*y[IDX_HII] + k[245] + k[1117];
    IJth(jmatrix, 79, 73) = 0.0 + k[715]*y[IDX_CH3OHI] + k[716]*y[IDX_CH4I];
    IJth(jmatrix, 79, 74) = 0.0 + k[810]*y[IDX_CH4I];
    IJth(jmatrix, 79, 75) = 0.0 + k[426]*y[IDX_CH2I];
    IJth(jmatrix, 79, 76) = 0.0 + k[433]*y[IDX_CH2I];
    IJth(jmatrix, 79, 77) = 0.0 + k[432]*y[IDX_CH2I];
    IJth(jmatrix, 79, 79) = 0.0 - k[46]*y[IDX_HCOI] - k[47]*y[IDX_MgI] -
                        k[48]*y[IDX_NOI] - k[298]*y[IDX_EM] - k[299]*y[IDX_EM] -
                        k[300]*y[IDX_EM] - k[439]*y[IDX_CH3OHI] -
                        k[440]*y[IDX_H2COI] - k[441]*y[IDX_HCOI] -
                        k[442]*y[IDX_O2I] - k[443]*y[IDX_OI] - k[444]*y[IDX_OI]
                        - k[445]*y[IDX_OHI] - k[446]*y[IDX_SiH4I] -
                        k[616]*y[IDX_HI] - k[794]*y[IDX_NHI] - k[1114] - k[1115]
                        - k[1215]*y[IDX_EM] - k[1285];
    IJth(jmatrix, 79, 81) = 0.0 + k[419]*y[IDX_HCOI] + k[530]*y[IDX_H2I];
    IJth(jmatrix, 79, 82) = 0.0 + k[424]*y[IDX_CH2I];
    IJth(jmatrix, 79, 83) = 0.0 + k[434]*y[IDX_CH2I];
    IJth(jmatrix, 79, 85) = 0.0 + k[425]*y[IDX_CH2I];
    IJth(jmatrix, 79, 87) = 0.0 + k[366]*y[IDX_CH3OHI];
    IJth(jmatrix, 79, 89) = 0.0 + k[396]*y[IDX_CH3OHI] + k[399]*y[IDX_H2COI];
    IJth(jmatrix, 79, 90) = 0.0 + k[423]*y[IDX_H2COII] + k[424]*y[IDX_H2OII] +
                        k[425]*y[IDX_H3OII] + k[426]*y[IDX_HCNII] +
                        k[427]*y[IDX_HCNHII] + k[428]*y[IDX_HCNHII] +
                        k[429]*y[IDX_HCOII] + k[430]*y[IDX_HNOII] +
                        k[431]*y[IDX_N2HII] + k[432]*y[IDX_NHII] +
                        k[433]*y[IDX_NH2II] + k[434]*y[IDX_NH3II] +
                        k[436]*y[IDX_O2HII] + k[437]*y[IDX_OHII] +
                        k[510]*y[IDX_H2II] + k[577]*y[IDX_H3II];
    IJth(jmatrix, 79, 91) = 0.0 + k[423]*y[IDX_CH2I];
    IJth(jmatrix, 79, 92) = 0.0 - k[794]*y[IDX_CH3II];
    IJth(jmatrix, 79, 93) = 0.0 + k[437]*y[IDX_CH2I];
    IJth(jmatrix, 79, 95) = 0.0 + k[399]*y[IDX_CHII] - k[440]*y[IDX_CH3II];
    IJth(jmatrix, 79, 96) = 0.0 - k[46]*y[IDX_CH3II] + k[419]*y[IDX_CH2II] -
                        k[441]*y[IDX_CH3II];
    IJth(jmatrix, 79, 97) = 0.0 + k[657]*y[IDX_CH3OHI] + k[660]*y[IDX_CH4I];
    IJth(jmatrix, 79, 99) = 0.0 + k[577]*y[IDX_CH2I] + k[579]*y[IDX_CH3OHI];
    IJth(jmatrix, 79, 101) = 0.0 - k[48]*y[IDX_CH3II];
    IJth(jmatrix, 79, 103) = 0.0 - k[445]*y[IDX_CH3II];
    IJth(jmatrix, 79, 104) = 0.0 - k[442]*y[IDX_CH3II];
    IJth(jmatrix, 79, 106) = 0.0 + k[76]*y[IDX_CH3I] + k[492]*y[IDX_CH3OHI] +
                        k[495]*y[IDX_CH4I];
    IJth(jmatrix, 79, 107) = 0.0 + k[429]*y[IDX_CH2I];
    IJth(jmatrix, 79, 109) = 0.0 - k[443]*y[IDX_CH3II] - k[444]*y[IDX_CH3II] +
                        k[822]*y[IDX_CH4II];
    IJth(jmatrix, 79, 110) = 0.0 - k[298]*y[IDX_CH3II] - k[299]*y[IDX_CH3II] -
                        k[300]*y[IDX_CH3II] - k[1215]*y[IDX_CH3II];
    IJth(jmatrix, 79, 112) = 0.0 + k[530]*y[IDX_CH2II];
    IJth(jmatrix, 79, 113) = 0.0 - k[616]*y[IDX_CH3II] + k[617]*y[IDX_CH4II];
    IJth(jmatrix, 80, 51) = 0.0 + k[980]*y[IDX_HI];
    IJth(jmatrix, 80, 57) = 0.0 - k[183]*y[IDX_NH2I];
    IJth(jmatrix, 80, 58) = 0.0 - k[785]*y[IDX_NH2I] - k[786]*y[IDX_NH2I];
    IJth(jmatrix, 80, 59) = 0.0 - k[789]*y[IDX_NH2I];
    IJth(jmatrix, 80, 60) = 0.0 - k[790]*y[IDX_NH2I];
    IJth(jmatrix, 80, 64) = 0.0 - k[788]*y[IDX_NH2I];
    IJth(jmatrix, 80, 65) = 0.0 - k[186]*y[IDX_NH2I];
    IJth(jmatrix, 80, 66) = 0.0 - k[782]*y[IDX_NH2I];
    IJth(jmatrix, 80, 67) = 0.0 - k[1028]*y[IDX_NH2I] + k[1034]*y[IDX_NHI];
    IJth(jmatrix, 80, 68) = 0.0 - k[184]*y[IDX_NH2I] - k[779]*y[IDX_NH2I] +
                        k[792]*y[IDX_NH3I];
    IJth(jmatrix, 80, 69) = 0.0 - k[109]*y[IDX_NH2I];
    IJth(jmatrix, 80, 70) = 0.0 + k[181]*y[IDX_NH2II] + k[271] + k[792]*y[IDX_COII]
                        + k[793]*y[IDX_HCNII] + k[908]*y[IDX_CH3I] +
                        k[984]*y[IDX_HI] + k[1033]*y[IDX_CNI] +
                        k[1037]*y[IDX_NHI] + k[1037]*y[IDX_NHI] +
                        k[1074]*y[IDX_OI] + k[1098]*y[IDX_OHI] + k[1159];
    IJth(jmatrix, 80, 71) = 0.0 - k[907]*y[IDX_NH2I] + k[908]*y[IDX_NH3I];
    IJth(jmatrix, 80, 73) = 0.0 - k[164]*y[IDX_NH2I];
    IJth(jmatrix, 80, 74) = 0.0 - k[212]*y[IDX_NH2I];
    IJth(jmatrix, 80, 75) = 0.0 - k[784]*y[IDX_NH2I] + k[793]*y[IDX_NH3I];
    IJth(jmatrix, 80, 76) = 0.0 + k[42]*y[IDX_CH2I] + k[59]*y[IDX_CHI] +
                        k[180]*y[IDX_HCOI] + k[181]*y[IDX_NH3I] +
                        k[182]*y[IDX_NOI] - k[776]*y[IDX_NH2I];
    IJth(jmatrix, 80, 77) = 0.0 + k[753]*y[IDX_H2COI] - k[762]*y[IDX_NH2I];
    IJth(jmatrix, 80, 78) = 0.0 - k[187]*y[IDX_NH2I];
    IJth(jmatrix, 80, 80) = 0.0 - k[84]*y[IDX_HII] - k[109]*y[IDX_H2II] -
                        k[164]*y[IDX_NII] - k[183]*y[IDX_CNII] -
                        k[184]*y[IDX_COII] - k[185]*y[IDX_H2OII] -
                        k[186]*y[IDX_N2II] - k[187]*y[IDX_O2II] -
                        k[188]*y[IDX_OHII] - k[212]*y[IDX_OII] - k[269] - k[270]
                        - k[373]*y[IDX_CII] - k[409]*y[IDX_CHII] -
                        k[593]*y[IDX_H3II] - k[689]*y[IDX_HeII] -
                        k[690]*y[IDX_HeII] - k[762]*y[IDX_NHII] -
                        k[776]*y[IDX_NH2II] - k[779]*y[IDX_COII] -
                        k[780]*y[IDX_H2COII] - k[781]*y[IDX_H2OII] -
                        k[782]*y[IDX_H3COII] - k[783]*y[IDX_H3OII] -
                        k[784]*y[IDX_HCNII] - k[785]*y[IDX_HCNHII] -
                        k[786]*y[IDX_HCNHII] - k[787]*y[IDX_HCOII] -
                        k[788]*y[IDX_HNOII] - k[789]*y[IDX_N2HII] -
                        k[790]*y[IDX_O2HII] - k[791]*y[IDX_OHII] -
                        k[866]*y[IDX_CI] - k[867]*y[IDX_CI] - k[868]*y[IDX_CI] -
                        k[907]*y[IDX_CH3I] - k[961]*y[IDX_H2I] -
                        k[983]*y[IDX_HI] - k[1028]*y[IDX_CH4I] -
                        k[1029]*y[IDX_NOI] - k[1030]*y[IDX_NOI] -
                        k[1031]*y[IDX_OHI] - k[1032]*y[IDX_OHI] -
                        k[1072]*y[IDX_OI] - k[1073]*y[IDX_OI] - k[1157] -
                        k[1158] - k[1289];
    IJth(jmatrix, 80, 82) = 0.0 - k[185]*y[IDX_NH2I] - k[781]*y[IDX_NH2I];
    IJth(jmatrix, 80, 83) = 0.0 + k[343]*y[IDX_EM] + k[434]*y[IDX_CH2I];
    IJth(jmatrix, 80, 85) = 0.0 - k[783]*y[IDX_NH2I];
    IJth(jmatrix, 80, 87) = 0.0 - k[373]*y[IDX_NH2I];
    IJth(jmatrix, 80, 88) = 0.0 + k[1095]*y[IDX_OHI];
    IJth(jmatrix, 80, 89) = 0.0 - k[409]*y[IDX_NH2I];
    IJth(jmatrix, 80, 90) = 0.0 + k[42]*y[IDX_NH2II] + k[434]*y[IDX_NH3II];
    IJth(jmatrix, 80, 91) = 0.0 - k[780]*y[IDX_NH2I];
    IJth(jmatrix, 80, 92) = 0.0 + k[962]*y[IDX_H2I] + k[1034]*y[IDX_CH4I] +
                        k[1036]*y[IDX_H2OI] + k[1037]*y[IDX_NH3I] +
                        k[1037]*y[IDX_NH3I] + k[1040]*y[IDX_NHI] +
                        k[1040]*y[IDX_NHI] + k[1050]*y[IDX_OHI];
    IJth(jmatrix, 80, 93) = 0.0 - k[188]*y[IDX_NH2I] - k[791]*y[IDX_NH2I];
    IJth(jmatrix, 80, 94) = 0.0 + k[1033]*y[IDX_NH3I];
    IJth(jmatrix, 80, 95) = 0.0 + k[753]*y[IDX_NHII];
    IJth(jmatrix, 80, 96) = 0.0 + k[180]*y[IDX_NH2II];
    IJth(jmatrix, 80, 97) = 0.0 - k[689]*y[IDX_NH2I] - k[690]*y[IDX_NH2I];
    IJth(jmatrix, 80, 98) = 0.0 + k[59]*y[IDX_NH2II];
    IJth(jmatrix, 80, 99) = 0.0 - k[593]*y[IDX_NH2I];
    IJth(jmatrix, 80, 101) = 0.0 + k[182]*y[IDX_NH2II] - k[1029]*y[IDX_NH2I] -
                        k[1030]*y[IDX_NH2I];
    IJth(jmatrix, 80, 103) = 0.0 - k[1031]*y[IDX_NH2I] - k[1032]*y[IDX_NH2I] +
                        k[1050]*y[IDX_NHI] + k[1095]*y[IDX_HCNI] +
                        k[1098]*y[IDX_NH3I];
    IJth(jmatrix, 80, 105) = 0.0 - k[866]*y[IDX_NH2I] - k[867]*y[IDX_NH2I] -
                        k[868]*y[IDX_NH2I];
    IJth(jmatrix, 80, 106) = 0.0 - k[84]*y[IDX_NH2I];
    IJth(jmatrix, 80, 107) = 0.0 - k[787]*y[IDX_NH2I];
    IJth(jmatrix, 80, 108) = 0.0 + k[1036]*y[IDX_NHI];
    IJth(jmatrix, 80, 109) = 0.0 - k[1072]*y[IDX_NH2I] - k[1073]*y[IDX_NH2I] +
                        k[1074]*y[IDX_NH3I];
    IJth(jmatrix, 80, 110) = 0.0 + k[343]*y[IDX_NH3II];
    IJth(jmatrix, 80, 112) = 0.0 - k[961]*y[IDX_NH2I] + k[962]*y[IDX_NHI];
    IJth(jmatrix, 80, 113) = 0.0 + k[980]*y[IDX_HNOI] - k[983]*y[IDX_NH2I] +
                        k[984]*y[IDX_NH3I];
    IJth(jmatrix, 81, 45) = 0.0 + k[477]*y[IDX_CHI];
    IJth(jmatrix, 81, 55) = 0.0 + k[1122];
    IJth(jmatrix, 81, 57) = 0.0 + k[37]*y[IDX_CH2I];
    IJth(jmatrix, 81, 58) = 0.0 + k[464]*y[IDX_CHI] + k[465]*y[IDX_CHI];
    IJth(jmatrix, 81, 59) = 0.0 + k[469]*y[IDX_CHI];
    IJth(jmatrix, 81, 60) = 0.0 + k[474]*y[IDX_CHI];
    IJth(jmatrix, 81, 64) = 0.0 + k[467]*y[IDX_CHI];
    IJth(jmatrix, 81, 65) = 0.0 + k[41]*y[IDX_CH2I] + k[455]*y[IDX_CH4I];
    IJth(jmatrix, 81, 66) = 0.0 + k[461]*y[IDX_CHI];
    IJth(jmatrix, 81, 67) = 0.0 + k[455]*y[IDX_N2II] + k[659]*y[IDX_HeII];
    IJth(jmatrix, 81, 68) = 0.0 + k[38]*y[IDX_CH2I];
    IJth(jmatrix, 81, 69) = 0.0 + k[100]*y[IDX_CH2I] + k[512]*y[IDX_CHI];
    IJth(jmatrix, 81, 72) = 0.0 - k[416]*y[IDX_CH2II];
    IJth(jmatrix, 81, 73) = 0.0 + k[155]*y[IDX_CH2I];
    IJth(jmatrix, 81, 74) = 0.0 + k[43]*y[IDX_CH2I];
    IJth(jmatrix, 81, 75) = 0.0 + k[463]*y[IDX_CHI];
    IJth(jmatrix, 81, 76) = 0.0 + k[42]*y[IDX_CH2I] + k[471]*y[IDX_CHI];
    IJth(jmatrix, 81, 77) = 0.0 + k[470]*y[IDX_CHI];
    IJth(jmatrix, 81, 78) = 0.0 + k[44]*y[IDX_CH2I];
    IJth(jmatrix, 81, 79) = 0.0 + k[616]*y[IDX_HI] + k[1115];
    IJth(jmatrix, 81, 81) = 0.0 - k[36]*y[IDX_NOI] - k[295]*y[IDX_EM] -
                        k[296]*y[IDX_EM] - k[297]*y[IDX_EM] - k[416]*y[IDX_CO2I]
                        - k[417]*y[IDX_H2COI] - k[418]*y[IDX_H2OI] -
                        k[419]*y[IDX_HCOI] - k[420]*y[IDX_O2I] -
                        k[421]*y[IDX_OI] - k[530]*y[IDX_H2I] - k[615]*y[IDX_HI]
                        - k[736]*y[IDX_NI] - k[1109] - k[1110] - k[1111] -
                        k[1278];
    IJth(jmatrix, 81, 82) = 0.0 + k[40]*y[IDX_CH2I] + k[460]*y[IDX_CHI];
    IJth(jmatrix, 81, 85) = 0.0 + k[462]*y[IDX_CHI];
    IJth(jmatrix, 81, 87) = 0.0 + k[14]*y[IDX_CH2I] + k[368]*y[IDX_H2COI] +
                        k[1199]*y[IDX_H2I];
    IJth(jmatrix, 81, 89) = 0.0 + k[406]*y[IDX_HCOI] + k[529]*y[IDX_H2I];
    IJth(jmatrix, 81, 90) = 0.0 + k[14]*y[IDX_CII] + k[37]*y[IDX_CNII] +
                        k[38]*y[IDX_COII] + k[39]*y[IDX_H2COII] +
                        k[40]*y[IDX_H2OII] + k[41]*y[IDX_N2II] +
                        k[42]*y[IDX_NH2II] + k[43]*y[IDX_OII] +
                        k[44]*y[IDX_O2II] + k[45]*y[IDX_OHII] + k[75]*y[IDX_HII]
                        + k[100]*y[IDX_H2II] + k[155]*y[IDX_NII] + k[242] +
                        k[1112];
    IJth(jmatrix, 81, 91) = 0.0 + k[39]*y[IDX_CH2I] + k[459]*y[IDX_CHI];
    IJth(jmatrix, 81, 93) = 0.0 + k[45]*y[IDX_CH2I] + k[475]*y[IDX_CHI];
    IJth(jmatrix, 81, 95) = 0.0 + k[368]*y[IDX_CII] - k[417]*y[IDX_CH2II] +
                        k[672]*y[IDX_HeII];
    IJth(jmatrix, 81, 96) = 0.0 + k[406]*y[IDX_CHII] - k[419]*y[IDX_CH2II];
    IJth(jmatrix, 81, 97) = 0.0 + k[659]*y[IDX_CH4I] + k[672]*y[IDX_H2COI];
    IJth(jmatrix, 81, 98) = 0.0 + k[459]*y[IDX_H2COII] + k[460]*y[IDX_H2OII] +
                        k[461]*y[IDX_H3COII] + k[462]*y[IDX_H3OII] +
                        k[463]*y[IDX_HCNII] + k[464]*y[IDX_HCNHII] +
                        k[465]*y[IDX_HCNHII] + k[466]*y[IDX_HCOII] +
                        k[467]*y[IDX_HNOII] + k[469]*y[IDX_N2HII] +
                        k[470]*y[IDX_NHII] + k[471]*y[IDX_NH2II] +
                        k[474]*y[IDX_O2HII] + k[475]*y[IDX_OHII] +
                        k[477]*y[IDX_SiHII] + k[512]*y[IDX_H2II] +
                        k[580]*y[IDX_H3II];
    IJth(jmatrix, 81, 99) = 0.0 + k[580]*y[IDX_CHI];
    IJth(jmatrix, 81, 101) = 0.0 - k[36]*y[IDX_CH2II];
    IJth(jmatrix, 81, 102) = 0.0 - k[736]*y[IDX_CH2II];
    IJth(jmatrix, 81, 104) = 0.0 - k[420]*y[IDX_CH2II];
    IJth(jmatrix, 81, 106) = 0.0 + k[75]*y[IDX_CH2I];
    IJth(jmatrix, 81, 107) = 0.0 + k[466]*y[IDX_CHI];
    IJth(jmatrix, 81, 108) = 0.0 - k[418]*y[IDX_CH2II];
    IJth(jmatrix, 81, 109) = 0.0 - k[421]*y[IDX_CH2II];
    IJth(jmatrix, 81, 110) = 0.0 - k[295]*y[IDX_CH2II] - k[296]*y[IDX_CH2II] -
                        k[297]*y[IDX_CH2II];
    IJth(jmatrix, 81, 112) = 0.0 + k[529]*y[IDX_CHII] - k[530]*y[IDX_CH2II] +
                        k[1199]*y[IDX_CII];
    IJth(jmatrix, 81, 113) = 0.0 - k[615]*y[IDX_CH2II] + k[616]*y[IDX_CH3II];
    IJth(jmatrix, 82, 53) = 0.0 - k[119]*y[IDX_H2OII];
    IJth(jmatrix, 82, 59) = 0.0 + k[857]*y[IDX_OHI];
    IJth(jmatrix, 82, 60) = 0.0 + k[858]*y[IDX_OHI];
    IJth(jmatrix, 82, 62) = 0.0 - k[122]*y[IDX_H2OII];
    IJth(jmatrix, 82, 63) = 0.0 - k[559]*y[IDX_H2OII];
    IJth(jmatrix, 82, 64) = 0.0 + k[856]*y[IDX_OHI];
    IJth(jmatrix, 82, 65) = 0.0 + k[125]*y[IDX_H2OI];
    IJth(jmatrix, 82, 67) = 0.0 - k[453]*y[IDX_H2OII];
    IJth(jmatrix, 82, 68) = 0.0 + k[123]*y[IDX_H2OI];
    IJth(jmatrix, 82, 69) = 0.0 + k[106]*y[IDX_H2OI] + k[527]*y[IDX_OHI];
    IJth(jmatrix, 82, 70) = 0.0 - k[195]*y[IDX_H2OII];
    IJth(jmatrix, 82, 73) = 0.0 + k[160]*y[IDX_H2OI];
    IJth(jmatrix, 82, 74) = 0.0 + k[210]*y[IDX_H2OI];
    IJth(jmatrix, 82, 75) = 0.0 + k[124]*y[IDX_H2OI] + k[853]*y[IDX_OHI];
    IJth(jmatrix, 82, 77) = 0.0 + k[176]*y[IDX_H2OI] + k[768]*y[IDX_OHI];
    IJth(jmatrix, 82, 80) = 0.0 - k[185]*y[IDX_H2OII] - k[781]*y[IDX_H2OII];
    IJth(jmatrix, 82, 82) = 0.0 - k[40]*y[IDX_CH2I] - k[56]*y[IDX_CHI] -
                        k[117]*y[IDX_H2COI] - k[118]*y[IDX_HCOI] -
                        k[119]*y[IDX_MgI] - k[120]*y[IDX_NOI] -
                        k[121]*y[IDX_O2I] - k[122]*y[IDX_SiI] -
                        k[185]*y[IDX_NH2I] - k[195]*y[IDX_NH3I] -
                        k[312]*y[IDX_EM] - k[313]*y[IDX_EM] - k[314]*y[IDX_EM] -
                        k[383]*y[IDX_CI] - k[424]*y[IDX_CH2I] -
                        k[453]*y[IDX_CH4I] - k[460]*y[IDX_CHI] -
                        k[534]*y[IDX_H2I] - k[553]*y[IDX_COI] -
                        k[554]*y[IDX_H2COI] - k[555]*y[IDX_H2OI] -
                        k[556]*y[IDX_HCNI] - k[557]*y[IDX_HCOI] -
                        k[558]*y[IDX_HCOI] - k[559]*y[IDX_HNCI] -
                        k[738]*y[IDX_NI] - k[739]*y[IDX_NI] - k[781]*y[IDX_NH2I]
                        - k[797]*y[IDX_NHI] - k[823]*y[IDX_OI] -
                        k[852]*y[IDX_OHI] - k[1140] - k[1280];
    IJth(jmatrix, 82, 88) = 0.0 - k[556]*y[IDX_H2OII];
    IJth(jmatrix, 82, 90) = 0.0 - k[40]*y[IDX_H2OII] - k[424]*y[IDX_H2OII];
    IJth(jmatrix, 82, 92) = 0.0 - k[797]*y[IDX_H2OII];
    IJth(jmatrix, 82, 93) = 0.0 + k[220]*y[IDX_H2OI] + k[545]*y[IDX_H2I] +
                        k[842]*y[IDX_HCOI] + k[847]*y[IDX_OHI];
    IJth(jmatrix, 82, 95) = 0.0 - k[117]*y[IDX_H2OII] - k[554]*y[IDX_H2OII];
    IJth(jmatrix, 82, 96) = 0.0 - k[118]*y[IDX_H2OII] - k[557]*y[IDX_H2OII] -
                        k[558]*y[IDX_H2OII] + k[842]*y[IDX_OHII];
    IJth(jmatrix, 82, 97) = 0.0 + k[143]*y[IDX_H2OI];
    IJth(jmatrix, 82, 98) = 0.0 - k[56]*y[IDX_H2OII] - k[460]*y[IDX_H2OII];
    IJth(jmatrix, 82, 99) = 0.0 + k[598]*y[IDX_OI] + k[600]*y[IDX_OHI];
    IJth(jmatrix, 82, 101) = 0.0 - k[120]*y[IDX_H2OII];
    IJth(jmatrix, 82, 102) = 0.0 - k[738]*y[IDX_H2OII] - k[739]*y[IDX_H2OII];
    IJth(jmatrix, 82, 103) = 0.0 + k[527]*y[IDX_H2II] + k[600]*y[IDX_H3II] +
                        k[768]*y[IDX_NHII] + k[847]*y[IDX_OHII] -
                        k[852]*y[IDX_H2OII] + k[853]*y[IDX_HCNII] +
                        k[854]*y[IDX_HCOII] + k[856]*y[IDX_HNOII] +
                        k[857]*y[IDX_N2HII] + k[858]*y[IDX_O2HII];
    IJth(jmatrix, 82, 104) = 0.0 - k[121]*y[IDX_H2OII];
    IJth(jmatrix, 82, 105) = 0.0 - k[383]*y[IDX_H2OII];
    IJth(jmatrix, 82, 106) = 0.0 + k[80]*y[IDX_H2OI];
    IJth(jmatrix, 82, 107) = 0.0 + k[854]*y[IDX_OHI];
    IJth(jmatrix, 82, 108) = 0.0 + k[80]*y[IDX_HII] + k[106]*y[IDX_H2II] +
                        k[123]*y[IDX_COII] + k[124]*y[IDX_HCNII] +
                        k[125]*y[IDX_N2II] + k[143]*y[IDX_HeII] +
                        k[160]*y[IDX_NII] + k[176]*y[IDX_NHII] +
                        k[210]*y[IDX_OII] + k[220]*y[IDX_OHII] -
                        k[555]*y[IDX_H2OII] + k[1141];
    IJth(jmatrix, 82, 109) = 0.0 + k[598]*y[IDX_H3II] - k[823]*y[IDX_H2OII];
    IJth(jmatrix, 82, 110) = 0.0 - k[312]*y[IDX_H2OII] - k[313]*y[IDX_H2OII] -
                        k[314]*y[IDX_H2OII];
    IJth(jmatrix, 82, 111) = 0.0 - k[553]*y[IDX_H2OII];
    IJth(jmatrix, 82, 112) = 0.0 - k[534]*y[IDX_H2OII] + k[545]*y[IDX_OHII];
    IJth(jmatrix, 83, 53) = 0.0 - k[190]*y[IDX_NH3II];
    IJth(jmatrix, 83, 55) = 0.0 + k[50]*y[IDX_NH3I];
    IJth(jmatrix, 83, 58) = 0.0 + k[785]*y[IDX_NH2I] + k[786]*y[IDX_NH2I];
    IJth(jmatrix, 83, 59) = 0.0 + k[789]*y[IDX_NH2I];
    IJth(jmatrix, 83, 60) = 0.0 + k[790]*y[IDX_NH2I];
    IJth(jmatrix, 83, 62) = 0.0 - k[192]*y[IDX_NH3II];
    IJth(jmatrix, 83, 64) = 0.0 + k[788]*y[IDX_NH2I];
    IJth(jmatrix, 83, 65) = 0.0 + k[197]*y[IDX_NH3I];
    IJth(jmatrix, 83, 66) = 0.0 + k[782]*y[IDX_NH2I];
    IJth(jmatrix, 83, 68) = 0.0 + k[193]*y[IDX_NH3I];
    IJth(jmatrix, 83, 69) = 0.0 + k[110]*y[IDX_NH3I];
    IJth(jmatrix, 83, 70) = 0.0 + k[19]*y[IDX_CII] + k[33]*y[IDX_CHII] +
                        k[50]*y[IDX_CH4II] + k[85]*y[IDX_HII] +
                        k[110]*y[IDX_H2II] + k[145]*y[IDX_HeII] +
                        k[165]*y[IDX_NII] + k[177]*y[IDX_NHII] +
                        k[181]*y[IDX_NH2II] + k[193]*y[IDX_COII] +
                        k[194]*y[IDX_H2COII] + k[195]*y[IDX_H2OII] +
                        k[196]*y[IDX_HCNII] + k[197]*y[IDX_N2II] +
                        k[198]*y[IDX_O2II] + k[213]*y[IDX_OII] +
                        k[222]*y[IDX_OHII] + k[272] + k[1160];
    IJth(jmatrix, 83, 73) = 0.0 + k[165]*y[IDX_NH3I];
    IJth(jmatrix, 83, 74) = 0.0 + k[213]*y[IDX_NH3I];
    IJth(jmatrix, 83, 75) = 0.0 + k[196]*y[IDX_NH3I] + k[784]*y[IDX_NH2I];
    IJth(jmatrix, 83, 76) = 0.0 + k[181]*y[IDX_NH3I] + k[542]*y[IDX_H2I] +
                        k[770]*y[IDX_H2COI] + k[772]*y[IDX_H2OI] +
                        k[776]*y[IDX_NH2I] + k[802]*y[IDX_NHI];
    IJth(jmatrix, 83, 77) = 0.0 + k[177]*y[IDX_NH3I] + k[756]*y[IDX_H2OI] +
                        k[762]*y[IDX_NH2I];
    IJth(jmatrix, 83, 78) = 0.0 + k[198]*y[IDX_NH3I];
    IJth(jmatrix, 83, 80) = 0.0 + k[593]*y[IDX_H3II] + k[762]*y[IDX_NHII] +
                        k[776]*y[IDX_NH2II] + k[780]*y[IDX_H2COII] +
                        k[781]*y[IDX_H2OII] + k[782]*y[IDX_H3COII] +
                        k[783]*y[IDX_H3OII] + k[784]*y[IDX_HCNII] +
                        k[785]*y[IDX_HCNHII] + k[786]*y[IDX_HCNHII] +
                        k[787]*y[IDX_HCOII] + k[788]*y[IDX_HNOII] +
                        k[789]*y[IDX_N2HII] + k[790]*y[IDX_O2HII] +
                        k[791]*y[IDX_OHII];
    IJth(jmatrix, 83, 82) = 0.0 + k[195]*y[IDX_NH3I] + k[781]*y[IDX_NH2I];
    IJth(jmatrix, 83, 83) = 0.0 - k[189]*y[IDX_HCOI] - k[190]*y[IDX_MgI] -
                        k[191]*y[IDX_NOI] - k[192]*y[IDX_SiI] - k[343]*y[IDX_EM]
                        - k[344]*y[IDX_EM] - k[434]*y[IDX_CH2I] -
                        k[828]*y[IDX_OI] - k[1283];
    IJth(jmatrix, 83, 85) = 0.0 + k[783]*y[IDX_NH2I];
    IJth(jmatrix, 83, 87) = 0.0 + k[19]*y[IDX_NH3I];
    IJth(jmatrix, 83, 89) = 0.0 + k[33]*y[IDX_NH3I];
    IJth(jmatrix, 83, 90) = 0.0 - k[434]*y[IDX_NH3II];
    IJth(jmatrix, 83, 91) = 0.0 + k[194]*y[IDX_NH3I] + k[780]*y[IDX_NH2I];
    IJth(jmatrix, 83, 92) = 0.0 + k[802]*y[IDX_NH2II];
    IJth(jmatrix, 83, 93) = 0.0 + k[222]*y[IDX_NH3I] + k[791]*y[IDX_NH2I];
    IJth(jmatrix, 83, 95) = 0.0 + k[770]*y[IDX_NH2II];
    IJth(jmatrix, 83, 96) = 0.0 - k[189]*y[IDX_NH3II];
    IJth(jmatrix, 83, 97) = 0.0 + k[145]*y[IDX_NH3I];
    IJth(jmatrix, 83, 99) = 0.0 + k[593]*y[IDX_NH2I];
    IJth(jmatrix, 83, 101) = 0.0 - k[191]*y[IDX_NH3II];
    IJth(jmatrix, 83, 106) = 0.0 + k[85]*y[IDX_NH3I];
    IJth(jmatrix, 83, 107) = 0.0 + k[787]*y[IDX_NH2I];
    IJth(jmatrix, 83, 108) = 0.0 + k[756]*y[IDX_NHII] + k[772]*y[IDX_NH2II];
    IJth(jmatrix, 83, 109) = 0.0 - k[828]*y[IDX_NH3II];
    IJth(jmatrix, 83, 110) = 0.0 - k[343]*y[IDX_NH3II] - k[344]*y[IDX_NH3II];
    IJth(jmatrix, 83, 112) = 0.0 + k[542]*y[IDX_NH2II];
    IJth(jmatrix, 84, 39) = 0.0 + k[504]*y[IDX_HII] + k[595]*y[IDX_H3II] +
                        k[818]*y[IDX_OII];
    IJth(jmatrix, 84, 49) = 0.0 + k[206]*y[IDX_NOI] + k[745]*y[IDX_NI];
    IJth(jmatrix, 84, 51) = 0.0 + k[503]*y[IDX_HII] + k[686]*y[IDX_HeII];
    IJth(jmatrix, 84, 52) = 0.0 + k[714]*y[IDX_NII];
    IJth(jmatrix, 84, 53) = 0.0 - k[151]*y[IDX_NOII];
    IJth(jmatrix, 84, 57) = 0.0 + k[67]*y[IDX_NOI] + k[481]*y[IDX_O2I];
    IJth(jmatrix, 84, 62) = 0.0 - k[229]*y[IDX_NOII];
    IJth(jmatrix, 84, 64) = 0.0 + k[204]*y[IDX_NOI];
    IJth(jmatrix, 84, 65) = 0.0 + k[172]*y[IDX_NOI] + k[825]*y[IDX_OI];
    IJth(jmatrix, 84, 68) = 0.0 + k[72]*y[IDX_NOI];
    IJth(jmatrix, 84, 69) = 0.0 + k[112]*y[IDX_NOI];
    IJth(jmatrix, 84, 72) = 0.0 + k[750]*y[IDX_NHII];
    IJth(jmatrix, 84, 73) = 0.0 + k[167]*y[IDX_NOI] + k[714]*y[IDX_CH3OHI] +
                        k[720]*y[IDX_COI] + k[722]*y[IDX_H2COI] +
                        k[728]*y[IDX_O2I];
    IJth(jmatrix, 84, 74) = 0.0 + k[803]*y[IDX_NHI] + k[811]*y[IDX_CNI] +
                        k[815]*y[IDX_HCNI] + k[817]*y[IDX_N2I] +
                        k[818]*y[IDX_NO2I];
    IJth(jmatrix, 84, 75) = 0.0 + k[132]*y[IDX_NOI];
    IJth(jmatrix, 84, 76) = 0.0 + k[182]*y[IDX_NOI];
    IJth(jmatrix, 84, 77) = 0.0 + k[178]*y[IDX_NOI] + k[750]*y[IDX_CO2I] +
                        k[765]*y[IDX_O2I];
    IJth(jmatrix, 84, 78) = 0.0 + k[205]*y[IDX_NOI] + k[742]*y[IDX_NI];
    IJth(jmatrix, 84, 79) = 0.0 + k[48]*y[IDX_NOI];
    IJth(jmatrix, 84, 81) = 0.0 + k[36]*y[IDX_NOI];
    IJth(jmatrix, 84, 82) = 0.0 + k[120]*y[IDX_NOI] + k[739]*y[IDX_NI];
    IJth(jmatrix, 84, 83) = 0.0 + k[191]*y[IDX_NOI];
    IJth(jmatrix, 84, 84) = 0.0 - k[151]*y[IDX_MgI] - k[229]*y[IDX_SiI] -
                        k[345]*y[IDX_EM] - k[1277];
    IJth(jmatrix, 84, 86) = 0.0 + k[817]*y[IDX_OII];
    IJth(jmatrix, 84, 87) = 0.0 + k[20]*y[IDX_NOI];
    IJth(jmatrix, 84, 88) = 0.0 + k[815]*y[IDX_OII];
    IJth(jmatrix, 84, 89) = 0.0 + k[34]*y[IDX_NOI];
    IJth(jmatrix, 84, 91) = 0.0 + k[203]*y[IDX_NOI];
    IJth(jmatrix, 84, 92) = 0.0 + k[803]*y[IDX_OII];
    IJth(jmatrix, 84, 93) = 0.0 + k[223]*y[IDX_NOI] + k[743]*y[IDX_NI];
    IJth(jmatrix, 84, 94) = 0.0 + k[811]*y[IDX_OII];
    IJth(jmatrix, 84, 95) = 0.0 + k[722]*y[IDX_NII];
    IJth(jmatrix, 84, 97) = 0.0 + k[686]*y[IDX_HNOI];
    IJth(jmatrix, 84, 99) = 0.0 + k[595]*y[IDX_NO2I];
    IJth(jmatrix, 84, 101) = 0.0 + k[20]*y[IDX_CII] + k[34]*y[IDX_CHII] +
                        k[36]*y[IDX_CH2II] + k[48]*y[IDX_CH3II] +
                        k[67]*y[IDX_CNII] + k[72]*y[IDX_COII] + k[87]*y[IDX_HII]
                        + k[112]*y[IDX_H2II] + k[120]*y[IDX_H2OII] +
                        k[132]*y[IDX_HCNII] + k[167]*y[IDX_NII] +
                        k[172]*y[IDX_N2II] + k[178]*y[IDX_NHII] +
                        k[182]*y[IDX_NH2II] + k[191]*y[IDX_NH3II] +
                        k[203]*y[IDX_H2COII] + k[204]*y[IDX_HNOII] +
                        k[205]*y[IDX_O2II] + k[206]*y[IDX_SiOII] +
                        k[223]*y[IDX_OHII] + k[277] + k[1165];
    IJth(jmatrix, 84, 102) = 0.0 + k[739]*y[IDX_H2OII] + k[742]*y[IDX_O2II] +
                        k[743]*y[IDX_OHII] + k[745]*y[IDX_SiOII];
    IJth(jmatrix, 84, 104) = 0.0 + k[481]*y[IDX_CNII] + k[728]*y[IDX_NII] +
                        k[765]*y[IDX_NHII];
    IJth(jmatrix, 84, 106) = 0.0 + k[87]*y[IDX_NOI] + k[503]*y[IDX_HNOI] +
                        k[504]*y[IDX_NO2I];
    IJth(jmatrix, 84, 109) = 0.0 + k[825]*y[IDX_N2II];
    IJth(jmatrix, 84, 110) = 0.0 - k[345]*y[IDX_NOII];
    IJth(jmatrix, 84, 111) = 0.0 + k[720]*y[IDX_NII];
    IJth(jmatrix, 85, 34) = 0.0 + k[575]*y[IDX_H2OI];
    IJth(jmatrix, 85, 35) = 0.0 + k[574]*y[IDX_H2OI];
    IJth(jmatrix, 85, 43) = 0.0 - k[611]*y[IDX_H3OII];
    IJth(jmatrix, 85, 45) = 0.0 + k[573]*y[IDX_H2OI];
    IJth(jmatrix, 85, 47) = 0.0 - k[612]*y[IDX_H3OII];
    IJth(jmatrix, 85, 50) = 0.0 + k[567]*y[IDX_H2OI];
    IJth(jmatrix, 85, 55) = 0.0 + k[450]*y[IDX_H2OI];
    IJth(jmatrix, 85, 56) = 0.0 - k[613]*y[IDX_H3OII];
    IJth(jmatrix, 85, 59) = 0.0 + k[570]*y[IDX_H2OI];
    IJth(jmatrix, 85, 60) = 0.0 + k[571]*y[IDX_H2OI];
    IJth(jmatrix, 85, 62) = 0.0 - k[610]*y[IDX_H3OII];
    IJth(jmatrix, 85, 63) = 0.0 - k[609]*y[IDX_H3OII];
    IJth(jmatrix, 85, 64) = 0.0 + k[568]*y[IDX_H2OI];
    IJth(jmatrix, 85, 66) = 0.0 + k[564]*y[IDX_H2OI];
    IJth(jmatrix, 85, 67) = 0.0 + k[453]*y[IDX_H2OII] + k[457]*y[IDX_OHII];
    IJth(jmatrix, 85, 69) = 0.0 + k[518]*y[IDX_H2OI];
    IJth(jmatrix, 85, 75) = 0.0 + k[565]*y[IDX_H2OI];
    IJth(jmatrix, 85, 76) = 0.0 + k[771]*y[IDX_H2OI];
    IJth(jmatrix, 85, 77) = 0.0 + k[754]*y[IDX_H2OI];
    IJth(jmatrix, 85, 80) = 0.0 - k[783]*y[IDX_H3OII];
    IJth(jmatrix, 85, 82) = 0.0 + k[453]*y[IDX_CH4I] + k[534]*y[IDX_H2I] +
                        k[555]*y[IDX_H2OI] + k[557]*y[IDX_HCOI] +
                        k[797]*y[IDX_NHI] + k[852]*y[IDX_OHI];
    IJth(jmatrix, 85, 85) = 0.0 - k[322]*y[IDX_EM] - k[323]*y[IDX_EM] -
                        k[324]*y[IDX_EM] - k[325]*y[IDX_EM] - k[384]*y[IDX_CI] -
                        k[425]*y[IDX_CH2I] - k[462]*y[IDX_CHI] -
                        k[607]*y[IDX_H2COI] - k[608]*y[IDX_HCNI] -
                        k[609]*y[IDX_HNCI] - k[610]*y[IDX_SiI] -
                        k[611]*y[IDX_SiH2I] - k[612]*y[IDX_SiHI] -
                        k[613]*y[IDX_SiOI] - k[783]*y[IDX_NH2I] - k[1286];
    IJth(jmatrix, 85, 88) = 0.0 - k[608]*y[IDX_H3OII];
    IJth(jmatrix, 85, 89) = 0.0 + k[403]*y[IDX_H2OI];
    IJth(jmatrix, 85, 90) = 0.0 - k[425]*y[IDX_H3OII];
    IJth(jmatrix, 85, 91) = 0.0 + k[563]*y[IDX_H2OI];
    IJth(jmatrix, 85, 92) = 0.0 + k[797]*y[IDX_H2OII];
    IJth(jmatrix, 85, 93) = 0.0 + k[457]*y[IDX_CH4I] + k[840]*y[IDX_H2OI];
    IJth(jmatrix, 85, 95) = 0.0 - k[607]*y[IDX_H3OII];
    IJth(jmatrix, 85, 96) = 0.0 + k[557]*y[IDX_H2OII];
    IJth(jmatrix, 85, 98) = 0.0 - k[462]*y[IDX_H3OII];
    IJth(jmatrix, 85, 99) = 0.0 + k[586]*y[IDX_H2OI];
    IJth(jmatrix, 85, 103) = 0.0 + k[852]*y[IDX_H2OII];
    IJth(jmatrix, 85, 105) = 0.0 - k[384]*y[IDX_H3OII];
    IJth(jmatrix, 85, 107) = 0.0 + k[566]*y[IDX_H2OI];
    IJth(jmatrix, 85, 108) = 0.0 + k[403]*y[IDX_CHII] + k[450]*y[IDX_CH4II] +
                        k[518]*y[IDX_H2II] + k[555]*y[IDX_H2OII] +
                        k[563]*y[IDX_H2COII] + k[564]*y[IDX_H3COII] +
                        k[565]*y[IDX_HCNII] + k[566]*y[IDX_HCOII] +
                        k[567]*y[IDX_HCO2II] + k[568]*y[IDX_HNOII] +
                        k[570]*y[IDX_N2HII] + k[571]*y[IDX_O2HII] +
                        k[573]*y[IDX_SiHII] + k[574]*y[IDX_SiH4II] +
                        k[575]*y[IDX_SiH5II] + k[586]*y[IDX_H3II] +
                        k[754]*y[IDX_NHII] + k[771]*y[IDX_NH2II] +
                        k[840]*y[IDX_OHII];
    IJth(jmatrix, 85, 110) = 0.0 - k[322]*y[IDX_H3OII] - k[323]*y[IDX_H3OII] -
                        k[324]*y[IDX_H3OII] - k[325]*y[IDX_H3OII];
    IJth(jmatrix, 85, 112) = 0.0 + k[534]*y[IDX_H2OII];
    IJth(jmatrix, 86, 17) = 0.0 + k[1335] + k[1336] + k[1337] + k[1338];
    IJth(jmatrix, 86, 39) = 0.0 + k[1019]*y[IDX_NI] + k[1021]*y[IDX_NI];
    IJth(jmatrix, 86, 42) = 0.0 + k[1053]*y[IDX_NOI];
    IJth(jmatrix, 86, 53) = 0.0 + k[150]*y[IDX_N2II];
    IJth(jmatrix, 86, 59) = 0.0 + k[338]*y[IDX_EM] + k[389]*y[IDX_CI] +
                        k[431]*y[IDX_CH2I] + k[469]*y[IDX_CHI] +
                        k[487]*y[IDX_COI] + k[570]*y[IDX_H2OI] +
                        k[631]*y[IDX_HCNI] + k[643]*y[IDX_HCOI] +
                        k[650]*y[IDX_HNCI] + k[734]*y[IDX_CO2I] +
                        k[735]*y[IDX_H2COI] + k[789]*y[IDX_NH2I] +
                        k[801]*y[IDX_NHI] + k[826]*y[IDX_OI] + k[857]*y[IDX_OHI];
    IJth(jmatrix, 86, 60) = 0.0 - k[733]*y[IDX_N2I];
    IJth(jmatrix, 86, 63) = 0.0 + k[650]*y[IDX_N2HII];
    IJth(jmatrix, 86, 64) = 0.0 - k[732]*y[IDX_N2I];
    IJth(jmatrix, 86, 65) = 0.0 + k[29]*y[IDX_CI] + k[41]*y[IDX_CH2I] +
                        k[58]*y[IDX_CHI] + k[69]*y[IDX_CNI] + k[74]*y[IDX_COI] +
                        k[125]*y[IDX_H2OI] + k[135]*y[IDX_HCNI] +
                        k[150]*y[IDX_MgI] + k[170]*y[IDX_H2COI] +
                        k[171]*y[IDX_HCOI] + k[172]*y[IDX_NOI] +
                        k[173]*y[IDX_O2I] + k[174]*y[IDX_NI] +
                        k[186]*y[IDX_NH2I] + k[197]*y[IDX_NH3I] +
                        k[201]*y[IDX_NHI] + k[218]*y[IDX_OI] + k[227]*y[IDX_OHI]
                        + k[455]*y[IDX_CH4I] + k[456]*y[IDX_CH4I] +
                        k[730]*y[IDX_H2COI];
    IJth(jmatrix, 86, 67) = 0.0 + k[455]*y[IDX_N2II] + k[456]*y[IDX_N2II];
    IJth(jmatrix, 86, 69) = 0.0 - k[521]*y[IDX_N2I];
    IJth(jmatrix, 86, 70) = 0.0 + k[197]*y[IDX_N2II];
    IJth(jmatrix, 86, 72) = 0.0 + k[734]*y[IDX_N2HII];
    IJth(jmatrix, 86, 74) = 0.0 - k[817]*y[IDX_N2I];
    IJth(jmatrix, 86, 77) = 0.0 - k[761]*y[IDX_N2I];
    IJth(jmatrix, 86, 80) = 0.0 + k[186]*y[IDX_N2II] + k[789]*y[IDX_N2HII] +
                        k[1029]*y[IDX_NOI] + k[1030]*y[IDX_NOI];
    IJth(jmatrix, 86, 86) = 0.0 - k[144]*y[IDX_HeII] - k[267] - k[521]*y[IDX_H2II] -
                        k[592]*y[IDX_H3II] - k[688]*y[IDX_HeII] -
                        k[732]*y[IDX_HNOII] - k[733]*y[IDX_O2HII] -
                        k[761]*y[IDX_NHII] - k[817]*y[IDX_OII] -
                        k[845]*y[IDX_OHII] - k[865]*y[IDX_CI] -
                        k[884]*y[IDX_CH2I] - k[927]*y[IDX_CHI] -
                        k[1071]*y[IDX_OI] - k[1155] - k[1262];
    IJth(jmatrix, 86, 88) = 0.0 + k[135]*y[IDX_N2II] + k[631]*y[IDX_N2HII];
    IJth(jmatrix, 86, 90) = 0.0 + k[41]*y[IDX_N2II] + k[431]*y[IDX_N2HII] -
                        k[884]*y[IDX_N2I];
    IJth(jmatrix, 86, 92) = 0.0 + k[201]*y[IDX_N2II] + k[801]*y[IDX_N2HII] +
                        k[1018]*y[IDX_NI] + k[1038]*y[IDX_NHI] +
                        k[1038]*y[IDX_NHI] + k[1039]*y[IDX_NHI] +
                        k[1039]*y[IDX_NHI] + k[1042]*y[IDX_NOI] +
                        k[1043]*y[IDX_NOI];
    IJth(jmatrix, 86, 93) = 0.0 - k[845]*y[IDX_N2I];
    IJth(jmatrix, 86, 94) = 0.0 + k[69]*y[IDX_N2II] + k[946]*y[IDX_NOI] +
                        k[1011]*y[IDX_NI];
    IJth(jmatrix, 86, 95) = 0.0 + k[170]*y[IDX_N2II] + k[730]*y[IDX_N2II] +
                        k[735]*y[IDX_N2HII];
    IJth(jmatrix, 86, 96) = 0.0 + k[171]*y[IDX_N2II] + k[643]*y[IDX_N2HII];
    IJth(jmatrix, 86, 97) = 0.0 - k[144]*y[IDX_N2I] - k[688]*y[IDX_N2I];
    IJth(jmatrix, 86, 98) = 0.0 + k[58]*y[IDX_N2II] + k[469]*y[IDX_N2HII] -
                        k[927]*y[IDX_N2I];
    IJth(jmatrix, 86, 99) = 0.0 - k[592]*y[IDX_N2I];
    IJth(jmatrix, 86, 101) = 0.0 + k[172]*y[IDX_N2II] + k[946]*y[IDX_CNI] +
                        k[1022]*y[IDX_NI] + k[1029]*y[IDX_NH2I] +
                        k[1030]*y[IDX_NH2I] + k[1042]*y[IDX_NHI] +
                        k[1043]*y[IDX_NHI] + k[1051]*y[IDX_NOI] +
                        k[1051]*y[IDX_NOI] + k[1053]*y[IDX_OCNI];
    IJth(jmatrix, 86, 102) = 0.0 + k[174]*y[IDX_N2II] + k[1011]*y[IDX_CNI] +
                        k[1018]*y[IDX_NHI] + k[1019]*y[IDX_NO2I] +
                        k[1021]*y[IDX_NO2I] + k[1022]*y[IDX_NOI];
    IJth(jmatrix, 86, 103) = 0.0 + k[227]*y[IDX_N2II] + k[857]*y[IDX_N2HII];
    IJth(jmatrix, 86, 104) = 0.0 + k[173]*y[IDX_N2II];
    IJth(jmatrix, 86, 105) = 0.0 + k[29]*y[IDX_N2II] + k[389]*y[IDX_N2HII] -
                        k[865]*y[IDX_N2I];
    IJth(jmatrix, 86, 108) = 0.0 + k[125]*y[IDX_N2II] + k[570]*y[IDX_N2HII];
    IJth(jmatrix, 86, 109) = 0.0 + k[218]*y[IDX_N2II] + k[826]*y[IDX_N2HII] -
                        k[1071]*y[IDX_N2I];
    IJth(jmatrix, 86, 110) = 0.0 + k[338]*y[IDX_N2HII];
    IJth(jmatrix, 86, 111) = 0.0 + k[74]*y[IDX_N2II] + k[487]*y[IDX_N2HII];
    IJth(jmatrix, 87, 32) = 0.0 - k[22]*y[IDX_CII];
    IJth(jmatrix, 87, 33) = 0.0 - k[23]*y[IDX_CII];
    IJth(jmatrix, 87, 38) = 0.0 - k[24]*y[IDX_CII] + k[702]*y[IDX_HeII];
    IJth(jmatrix, 87, 42) = 0.0 - k[378]*y[IDX_CII];
    IJth(jmatrix, 87, 43) = 0.0 - k[25]*y[IDX_CII] - k[380]*y[IDX_CII];
    IJth(jmatrix, 87, 47) = 0.0 - k[381]*y[IDX_CII];
    IJth(jmatrix, 87, 48) = 0.0 - k[26]*y[IDX_CII];
    IJth(jmatrix, 87, 52) = 0.0 - k[365]*y[IDX_CII] - k[366]*y[IDX_CII];
    IJth(jmatrix, 87, 53) = 0.0 - k[18]*y[IDX_CII];
    IJth(jmatrix, 87, 56) = 0.0 - k[382]*y[IDX_CII];
    IJth(jmatrix, 87, 57) = 0.0 + k[27]*y[IDX_CI];
    IJth(jmatrix, 87, 62) = 0.0 - k[21]*y[IDX_CII];
    IJth(jmatrix, 87, 63) = 0.0 + k[684]*y[IDX_HeII];
    IJth(jmatrix, 87, 65) = 0.0 + k[29]*y[IDX_CI];
    IJth(jmatrix, 87, 68) = 0.0 + k[28]*y[IDX_CI] + k[1131];
    IJth(jmatrix, 87, 70) = 0.0 - k[19]*y[IDX_CII] - k[374]*y[IDX_CII];
    IJth(jmatrix, 87, 72) = 0.0 - k[367]*y[IDX_CII] + k[668]*y[IDX_HeII];
    IJth(jmatrix, 87, 78) = 0.0 + k[30]*y[IDX_CI];
    IJth(jmatrix, 87, 80) = 0.0 - k[373]*y[IDX_CII];
    IJth(jmatrix, 87, 81) = 0.0 + k[1109];
    IJth(jmatrix, 87, 87) = 0.0 - k[14]*y[IDX_CH2I] - k[15]*y[IDX_CHI] -
                        k[16]*y[IDX_H2COI] - k[17]*y[IDX_HCOI] -
                        k[18]*y[IDX_MgI] - k[19]*y[IDX_NH3I] - k[20]*y[IDX_NOI]
                        - k[21]*y[IDX_SiI] - k[22]*y[IDX_SiC2I] -
                        k[23]*y[IDX_SiC3I] - k[24]*y[IDX_SiCI] -
                        k[25]*y[IDX_SiH2I] - k[26]*y[IDX_SiH3I] -
                        k[365]*y[IDX_CH3OHI] - k[366]*y[IDX_CH3OHI] -
                        k[367]*y[IDX_CO2I] - k[368]*y[IDX_H2COI] -
                        k[369]*y[IDX_H2COI] - k[370]*y[IDX_H2OI] -
                        k[371]*y[IDX_H2OI] - k[372]*y[IDX_HCOI] -
                        k[373]*y[IDX_NH2I] - k[374]*y[IDX_NH3I] -
                        k[375]*y[IDX_NHI] - k[376]*y[IDX_O2I] -
                        k[377]*y[IDX_O2I] - k[378]*y[IDX_OCNI] -
                        k[379]*y[IDX_OHI] - k[380]*y[IDX_SiH2I] -
                        k[381]*y[IDX_SiHI] - k[382]*y[IDX_SiOI] -
                        k[528]*y[IDX_H2I] - k[1192]*y[IDX_NI] -
                        k[1193]*y[IDX_OI] - k[1199]*y[IDX_H2I] -
                        k[1205]*y[IDX_HI] - k[1214]*y[IDX_EM] - k[1264];
    IJth(jmatrix, 87, 88) = 0.0 + k[678]*y[IDX_HeII];
    IJth(jmatrix, 87, 89) = 0.0 + k[241] + k[614]*y[IDX_HI];
    IJth(jmatrix, 87, 90) = 0.0 - k[14]*y[IDX_CII] + k[653]*y[IDX_HeII];
    IJth(jmatrix, 87, 92) = 0.0 - k[375]*y[IDX_CII];
    IJth(jmatrix, 87, 94) = 0.0 + k[664]*y[IDX_HeII];
    IJth(jmatrix, 87, 95) = 0.0 - k[16]*y[IDX_CII] - k[368]*y[IDX_CII] -
                        k[369]*y[IDX_CII];
    IJth(jmatrix, 87, 96) = 0.0 - k[17]*y[IDX_CII] - k[372]*y[IDX_CII];
    IJth(jmatrix, 87, 97) = 0.0 + k[139]*y[IDX_CI] + k[653]*y[IDX_CH2I] +
                        k[662]*y[IDX_CHI] + k[664]*y[IDX_CNI] +
                        k[668]*y[IDX_CO2I] + k[669]*y[IDX_COI] +
                        k[678]*y[IDX_HCNI] + k[684]*y[IDX_HNCI] +
                        k[702]*y[IDX_SiCI];
    IJth(jmatrix, 87, 98) = 0.0 - k[15]*y[IDX_CII] + k[662]*y[IDX_HeII];
    IJth(jmatrix, 87, 101) = 0.0 - k[20]*y[IDX_CII];
    IJth(jmatrix, 87, 102) = 0.0 - k[1192]*y[IDX_CII];
    IJth(jmatrix, 87, 103) = 0.0 - k[379]*y[IDX_CII];
    IJth(jmatrix, 87, 104) = 0.0 - k[376]*y[IDX_CII] - k[377]*y[IDX_CII];
    IJth(jmatrix, 87, 105) = 0.0 + k[27]*y[IDX_CNII] + k[28]*y[IDX_COII] +
                        k[29]*y[IDX_N2II] + k[30]*y[IDX_O2II] +
                        k[139]*y[IDX_HeII] + k[231] + k[240] + k[1107];
    IJth(jmatrix, 87, 108) = 0.0 - k[370]*y[IDX_CII] - k[371]*y[IDX_CII];
    IJth(jmatrix, 87, 109) = 0.0 - k[1193]*y[IDX_CII];
    IJth(jmatrix, 87, 110) = 0.0 - k[1214]*y[IDX_CII];
    IJth(jmatrix, 87, 111) = 0.0 + k[669]*y[IDX_HeII];
    IJth(jmatrix, 87, 112) = 0.0 - k[528]*y[IDX_CII] - k[1199]*y[IDX_CII];
    IJth(jmatrix, 87, 113) = 0.0 + k[614]*y[IDX_CHII] - k[1205]*y[IDX_CII];
    IJth(jmatrix, 88, 19) = 0.0 + k[1323] + k[1324] + k[1325] + k[1326];
    IJth(jmatrix, 88, 23) = 0.0 + k[254] + k[973]*y[IDX_HI] + k[1013]*y[IDX_NI] +
                        k[1135];
    IJth(jmatrix, 88, 42) = 0.0 + k[993]*y[IDX_HI];
    IJth(jmatrix, 88, 46) = 0.0 + k[950]*y[IDX_CNI];
    IJth(jmatrix, 88, 51) = 0.0 + k[944]*y[IDX_CNI];
    IJth(jmatrix, 88, 57) = 0.0 - k[65]*y[IDX_HCNI] + k[479]*y[IDX_H2COI];
    IJth(jmatrix, 88, 58) = 0.0 + k[328]*y[IDX_EM] + k[427]*y[IDX_CH2I] +
                        k[464]*y[IDX_CHI] + k[633]*y[IDX_H2COI] +
                        k[785]*y[IDX_NH2I];
    IJth(jmatrix, 88, 59) = 0.0 - k[631]*y[IDX_HCNI];
    IJth(jmatrix, 88, 60) = 0.0 - k[632]*y[IDX_HCNI];
    IJth(jmatrix, 88, 63) = 0.0 + k[1]*y[IDX_HII] + k[979]*y[IDX_HI];
    IJth(jmatrix, 88, 64) = 0.0 - k[630]*y[IDX_HCNI];
    IJth(jmatrix, 88, 65) = 0.0 - k[135]*y[IDX_HCNI];
    IJth(jmatrix, 88, 66) = 0.0 - k[628]*y[IDX_HCNI];
    IJth(jmatrix, 88, 67) = 0.0 + k[920]*y[IDX_CNI];
    IJth(jmatrix, 88, 68) = 0.0 - k[134]*y[IDX_HCNI];
    IJth(jmatrix, 88, 69) = 0.0 - k[107]*y[IDX_HCNI];
    IJth(jmatrix, 88, 70) = 0.0 + k[196]*y[IDX_HCNII] + k[1033]*y[IDX_CNI];
    IJth(jmatrix, 88, 71) = 0.0 + k[902]*y[IDX_CNI] + k[910]*y[IDX_NOI] +
                        k[1009]*y[IDX_NI] + k[1010]*y[IDX_NI];
    IJth(jmatrix, 88, 73) = 0.0 - k[161]*y[IDX_HCNI];
    IJth(jmatrix, 88, 74) = 0.0 - k[814]*y[IDX_HCNI] - k[815]*y[IDX_HCNI];
    IJth(jmatrix, 88, 75) = 0.0 + k[124]*y[IDX_H2OI] + k[129]*y[IDX_HI] +
                        k[132]*y[IDX_NOI] + k[133]*y[IDX_O2I] +
                        k[196]*y[IDX_NH3I] - k[623]*y[IDX_HCNI];
    IJth(jmatrix, 88, 76) = 0.0 - k[773]*y[IDX_HCNI];
    IJth(jmatrix, 88, 77) = 0.0 - k[758]*y[IDX_HCNI];
    IJth(jmatrix, 88, 80) = 0.0 + k[785]*y[IDX_HCNHII] + k[866]*y[IDX_CI];
    IJth(jmatrix, 88, 82) = 0.0 - k[556]*y[IDX_HCNI];
    IJth(jmatrix, 88, 85) = 0.0 - k[608]*y[IDX_HCNI];
    IJth(jmatrix, 88, 86) = 0.0 + k[884]*y[IDX_CH2I] + k[927]*y[IDX_CHI];
    IJth(jmatrix, 88, 88) = 0.0 - k[65]*y[IDX_CNII] - k[81]*y[IDX_HII] -
                        k[107]*y[IDX_H2II] - k[134]*y[IDX_COII] -
                        k[135]*y[IDX_N2II] - k[161]*y[IDX_NII] - k[259] -
                        k[405]*y[IDX_CHII] - k[556]*y[IDX_H2OII] -
                        k[587]*y[IDX_H3II] - k[608]*y[IDX_H3OII] -
                        k[623]*y[IDX_HCNII] - k[627]*y[IDX_H2COII] -
                        k[628]*y[IDX_H3COII] - k[629]*y[IDX_HCOII] -
                        k[630]*y[IDX_HNOII] - k[631]*y[IDX_N2HII] -
                        k[632]*y[IDX_O2HII] - k[676]*y[IDX_HeII] -
                        k[677]*y[IDX_HeII] - k[678]*y[IDX_HeII] -
                        k[679]*y[IDX_HeII] - k[758]*y[IDX_NHII] -
                        k[773]*y[IDX_NH2II] - k[814]*y[IDX_OII] -
                        k[815]*y[IDX_OII] - k[841]*y[IDX_OHII] -
                        k[976]*y[IDX_HI] - k[1063]*y[IDX_OI] - k[1064]*y[IDX_OI]
                        - k[1065]*y[IDX_OI] - k[1094]*y[IDX_OHI] -
                        k[1095]*y[IDX_OHI] - k[1147] - k[1266];
    IJth(jmatrix, 88, 89) = 0.0 - k[405]*y[IDX_HCNI];
    IJth(jmatrix, 88, 90) = 0.0 + k[427]*y[IDX_HCNHII] + k[880]*y[IDX_CNI] +
                        k[884]*y[IDX_N2I] + k[887]*y[IDX_NOI] +
                        k[1005]*y[IDX_NI];
    IJth(jmatrix, 88, 91) = 0.0 - k[627]*y[IDX_HCNI];
    IJth(jmatrix, 88, 92) = 0.0 + k[1035]*y[IDX_CNI];
    IJth(jmatrix, 88, 93) = 0.0 - k[841]*y[IDX_HCNI];
    IJth(jmatrix, 88, 94) = 0.0 + k[880]*y[IDX_CH2I] + k[902]*y[IDX_CH3I] +
                        k[920]*y[IDX_CH4I] + k[942]*y[IDX_H2COI] +
                        k[943]*y[IDX_HCOI] + k[944]*y[IDX_HNOI] +
                        k[950]*y[IDX_SiH4I] + k[959]*y[IDX_H2I] +
                        k[1033]*y[IDX_NH3I] + k[1035]*y[IDX_NHI] +
                        k[1090]*y[IDX_OHI];
    IJth(jmatrix, 88, 95) = 0.0 + k[479]*y[IDX_CNII] + k[633]*y[IDX_HCNHII] +
                        k[942]*y[IDX_CNI];
    IJth(jmatrix, 88, 96) = 0.0 + k[943]*y[IDX_CNI] + k[1015]*y[IDX_NI];
    IJth(jmatrix, 88, 97) = 0.0 - k[676]*y[IDX_HCNI] - k[677]*y[IDX_HCNI] -
                        k[678]*y[IDX_HCNI] - k[679]*y[IDX_HCNI];
    IJth(jmatrix, 88, 98) = 0.0 + k[464]*y[IDX_HCNHII] + k[927]*y[IDX_N2I] +
                        k[930]*y[IDX_NOI];
    IJth(jmatrix, 88, 99) = 0.0 - k[587]*y[IDX_HCNI];
    IJth(jmatrix, 88, 101) = 0.0 + k[132]*y[IDX_HCNII] + k[887]*y[IDX_CH2I] +
                        k[910]*y[IDX_CH3I] + k[930]*y[IDX_CHI];
    IJth(jmatrix, 88, 102) = 0.0 + k[1005]*y[IDX_CH2I] + k[1009]*y[IDX_CH3I] +
                        k[1010]*y[IDX_CH3I] + k[1013]*y[IDX_H2CNI] +
                        k[1015]*y[IDX_HCOI];
    IJth(jmatrix, 88, 103) = 0.0 + k[1090]*y[IDX_CNI] - k[1094]*y[IDX_HCNI] -
                        k[1095]*y[IDX_HCNI];
    IJth(jmatrix, 88, 104) = 0.0 + k[133]*y[IDX_HCNII];
    IJth(jmatrix, 88, 105) = 0.0 + k[866]*y[IDX_NH2I];
    IJth(jmatrix, 88, 106) = 0.0 + k[1]*y[IDX_HNCI] - k[81]*y[IDX_HCNI];
    IJth(jmatrix, 88, 107) = 0.0 - k[629]*y[IDX_HCNI];
    IJth(jmatrix, 88, 108) = 0.0 + k[124]*y[IDX_HCNII];
    IJth(jmatrix, 88, 109) = 0.0 - k[1063]*y[IDX_HCNI] - k[1064]*y[IDX_HCNI] -
                        k[1065]*y[IDX_HCNI];
    IJth(jmatrix, 88, 110) = 0.0 + k[328]*y[IDX_HCNHII];
    IJth(jmatrix, 88, 112) = 0.0 + k[959]*y[IDX_CNI];
    IJth(jmatrix, 88, 113) = 0.0 + k[129]*y[IDX_HCNII] + k[973]*y[IDX_H2CNI] -
                        k[976]*y[IDX_HCNI] + k[979]*y[IDX_HNCI] +
                        k[993]*y[IDX_OCNI];
    IJth(jmatrix, 89, 50) = 0.0 + k[387]*y[IDX_CI];
    IJth(jmatrix, 89, 52) = 0.0 - k[396]*y[IDX_CHII] - k[397]*y[IDX_CHII];
    IJth(jmatrix, 89, 53) = 0.0 - k[32]*y[IDX_CHII];
    IJth(jmatrix, 89, 57) = 0.0 + k[53]*y[IDX_CHI];
    IJth(jmatrix, 89, 59) = 0.0 + k[389]*y[IDX_CI];
    IJth(jmatrix, 89, 60) = 0.0 + k[392]*y[IDX_CI];
    IJth(jmatrix, 89, 62) = 0.0 - k[35]*y[IDX_CHII];
    IJth(jmatrix, 89, 63) = 0.0 - k[407]*y[IDX_CHII];
    IJth(jmatrix, 89, 64) = 0.0 + k[388]*y[IDX_CI];
    IJth(jmatrix, 89, 65) = 0.0 + k[58]*y[IDX_CHI];
    IJth(jmatrix, 89, 67) = 0.0 + k[658]*y[IDX_HeII];
    IJth(jmatrix, 89, 68) = 0.0 + k[54]*y[IDX_CHI];
    IJth(jmatrix, 89, 69) = 0.0 + k[102]*y[IDX_CHI] + k[509]*y[IDX_CI];
    IJth(jmatrix, 89, 70) = 0.0 - k[33]*y[IDX_CHII];
    IJth(jmatrix, 89, 71) = 0.0 + k[655]*y[IDX_HeII];
    IJth(jmatrix, 89, 72) = 0.0 - k[398]*y[IDX_CHII];
    IJth(jmatrix, 89, 73) = 0.0 + k[57]*y[IDX_CHI];
    IJth(jmatrix, 89, 74) = 0.0 + k[60]*y[IDX_CHI];
    IJth(jmatrix, 89, 75) = 0.0 + k[385]*y[IDX_CI];
    IJth(jmatrix, 89, 76) = 0.0 + k[59]*y[IDX_CHI];
    IJth(jmatrix, 89, 77) = 0.0 + k[390]*y[IDX_CI];
    IJth(jmatrix, 89, 78) = 0.0 + k[61]*y[IDX_CHI];
    IJth(jmatrix, 89, 79) = 0.0 + k[1114];
    IJth(jmatrix, 89, 80) = 0.0 - k[409]*y[IDX_CHII];
    IJth(jmatrix, 89, 81) = 0.0 + k[615]*y[IDX_HI] + k[1110];
    IJth(jmatrix, 89, 82) = 0.0 + k[56]*y[IDX_CHI] + k[383]*y[IDX_CI];
    IJth(jmatrix, 89, 87) = 0.0 + k[15]*y[IDX_CHI] + k[372]*y[IDX_HCOI] +
                        k[528]*y[IDX_H2I] + k[1205]*y[IDX_HI];
    IJth(jmatrix, 89, 88) = 0.0 - k[405]*y[IDX_CHII] + k[679]*y[IDX_HeII];
    IJth(jmatrix, 89, 89) = 0.0 - k[31]*y[IDX_HCOI] - k[32]*y[IDX_MgI] -
                        k[33]*y[IDX_NH3I] - k[34]*y[IDX_NOI] - k[35]*y[IDX_SiI]
                        - k[241] - k[294]*y[IDX_EM] - k[396]*y[IDX_CH3OHI] -
                        k[397]*y[IDX_CH3OHI] - k[398]*y[IDX_CO2I] -
                        k[399]*y[IDX_H2COI] - k[400]*y[IDX_H2COI] -
                        k[401]*y[IDX_H2COI] - k[402]*y[IDX_H2OI] -
                        k[403]*y[IDX_H2OI] - k[404]*y[IDX_H2OI] -
                        k[405]*y[IDX_HCNI] - k[406]*y[IDX_HCOI] -
                        k[407]*y[IDX_HNCI] - k[408]*y[IDX_NI] -
                        k[409]*y[IDX_NH2I] - k[410]*y[IDX_NHI] -
                        k[411]*y[IDX_O2I] - k[412]*y[IDX_O2I] -
                        k[413]*y[IDX_O2I] - k[414]*y[IDX_OI] - k[415]*y[IDX_OHI]
                        - k[529]*y[IDX_H2I] - k[614]*y[IDX_HI] - k[1108] -
                        k[1272];
    IJth(jmatrix, 89, 90) = 0.0 + k[491]*y[IDX_HII] + k[654]*y[IDX_HeII];
    IJth(jmatrix, 89, 91) = 0.0 + k[55]*y[IDX_CHI];
    IJth(jmatrix, 89, 92) = 0.0 - k[410]*y[IDX_CHII];
    IJth(jmatrix, 89, 93) = 0.0 + k[62]*y[IDX_CHI] + k[393]*y[IDX_CI];
    IJth(jmatrix, 89, 95) = 0.0 - k[399]*y[IDX_CHII] - k[400]*y[IDX_CHII] -
                        k[401]*y[IDX_CHII];
    IJth(jmatrix, 89, 96) = 0.0 - k[31]*y[IDX_CHII] + k[372]*y[IDX_CII] -
                        k[406]*y[IDX_CHII] + k[682]*y[IDX_HeII];
    IJth(jmatrix, 89, 97) = 0.0 + k[141]*y[IDX_CHI] + k[654]*y[IDX_CH2I] +
                        k[655]*y[IDX_CH3I] + k[658]*y[IDX_CH4I] +
                        k[679]*y[IDX_HCNI] + k[682]*y[IDX_HCOI];
    IJth(jmatrix, 89, 98) = 0.0 + k[15]*y[IDX_CII] + k[53]*y[IDX_CNII] +
                        k[54]*y[IDX_COII] + k[55]*y[IDX_H2COII] +
                        k[56]*y[IDX_H2OII] + k[57]*y[IDX_NII] +
                        k[58]*y[IDX_N2II] + k[59]*y[IDX_NH2II] +
                        k[60]*y[IDX_OII] + k[61]*y[IDX_O2II] + k[62]*y[IDX_OHII]
                        + k[78]*y[IDX_HII] + k[102]*y[IDX_H2II] +
                        k[141]*y[IDX_HeII] + k[1129];
    IJth(jmatrix, 89, 99) = 0.0 + k[576]*y[IDX_CI];
    IJth(jmatrix, 89, 101) = 0.0 - k[34]*y[IDX_CHII];
    IJth(jmatrix, 89, 102) = 0.0 - k[408]*y[IDX_CHII];
    IJth(jmatrix, 89, 103) = 0.0 - k[415]*y[IDX_CHII];
    IJth(jmatrix, 89, 104) = 0.0 - k[411]*y[IDX_CHII] - k[412]*y[IDX_CHII] -
                        k[413]*y[IDX_CHII];
    IJth(jmatrix, 89, 105) = 0.0 + k[383]*y[IDX_H2OII] + k[385]*y[IDX_HCNII] +
                        k[386]*y[IDX_HCOII] + k[387]*y[IDX_HCO2II] +
                        k[388]*y[IDX_HNOII] + k[389]*y[IDX_N2HII] +
                        k[390]*y[IDX_NHII] + k[392]*y[IDX_O2HII] +
                        k[393]*y[IDX_OHII] + k[509]*y[IDX_H2II] +
                        k[576]*y[IDX_H3II];
    IJth(jmatrix, 89, 106) = 0.0 + k[78]*y[IDX_CHI] + k[491]*y[IDX_CH2I];
    IJth(jmatrix, 89, 107) = 0.0 + k[386]*y[IDX_CI];
    IJth(jmatrix, 89, 108) = 0.0 - k[402]*y[IDX_CHII] - k[403]*y[IDX_CHII] -
                        k[404]*y[IDX_CHII];
    IJth(jmatrix, 89, 109) = 0.0 - k[414]*y[IDX_CHII];
    IJth(jmatrix, 89, 110) = 0.0 - k[294]*y[IDX_CHII];
    IJth(jmatrix, 89, 112) = 0.0 + k[528]*y[IDX_CII] - k[529]*y[IDX_CHII];
    IJth(jmatrix, 89, 113) = 0.0 - k[614]*y[IDX_CHII] + k[615]*y[IDX_CH2II] +
                        k[1205]*y[IDX_CII];
    IJth(jmatrix, 90, 37) = 0.0 + k[938]*y[IDX_CHI];
    IJth(jmatrix, 90, 39) = 0.0 - k[885]*y[IDX_CH2I];
    IJth(jmatrix, 90, 49) = 0.0 - k[438]*y[IDX_CH2I];
    IJth(jmatrix, 90, 51) = 0.0 - k[883]*y[IDX_CH2I] + k[926]*y[IDX_CHI];
    IJth(jmatrix, 90, 52) = 0.0 + k[397]*y[IDX_CHII];
    IJth(jmatrix, 90, 55) = 0.0 + k[301]*y[IDX_EM];
    IJth(jmatrix, 90, 57) = 0.0 - k[37]*y[IDX_CH2I];
    IJth(jmatrix, 90, 58) = 0.0 - k[427]*y[IDX_CH2I] - k[428]*y[IDX_CH2I];
    IJth(jmatrix, 90, 59) = 0.0 - k[431]*y[IDX_CH2I];
    IJth(jmatrix, 90, 60) = 0.0 - k[436]*y[IDX_CH2I];
    IJth(jmatrix, 90, 64) = 0.0 - k[430]*y[IDX_CH2I];
    IJth(jmatrix, 90, 65) = 0.0 - k[41]*y[IDX_CH2I];
    IJth(jmatrix, 90, 66) = 0.0 + k[317]*y[IDX_EM];
    IJth(jmatrix, 90, 67) = 0.0 + k[249] + k[457]*y[IDX_OHII] - k[879]*y[IDX_CH2I] +
                        k[1124];
    IJth(jmatrix, 90, 68) = 0.0 - k[38]*y[IDX_CH2I] - k[422]*y[IDX_CH2I];
    IJth(jmatrix, 90, 69) = 0.0 - k[100]*y[IDX_CH2I] - k[510]*y[IDX_CH2I];
    IJth(jmatrix, 90, 71) = 0.0 + k[244] + k[901]*y[IDX_CH3I] + k[901]*y[IDX_CH3I] +
                        k[902]*y[IDX_CNI] + k[913]*y[IDX_O2I] +
                        k[919]*y[IDX_OHI] + k[968]*y[IDX_HI] + k[1116];
    IJth(jmatrix, 90, 73) = 0.0 - k[155]*y[IDX_CH2I] + k[722]*y[IDX_H2COI];
    IJth(jmatrix, 90, 74) = 0.0 - k[43]*y[IDX_CH2I];
    IJth(jmatrix, 90, 75) = 0.0 - k[426]*y[IDX_CH2I];
    IJth(jmatrix, 90, 76) = 0.0 - k[42]*y[IDX_CH2I] - k[433]*y[IDX_CH2I];
    IJth(jmatrix, 90, 77) = 0.0 - k[432]*y[IDX_CH2I];
    IJth(jmatrix, 90, 78) = 0.0 - k[44]*y[IDX_CH2I] - k[435]*y[IDX_CH2I];
    IJth(jmatrix, 90, 79) = 0.0 + k[298]*y[IDX_EM];
    IJth(jmatrix, 90, 81) = 0.0 + k[36]*y[IDX_NOI];
    IJth(jmatrix, 90, 82) = 0.0 - k[40]*y[IDX_CH2I] - k[424]*y[IDX_CH2I];
    IJth(jmatrix, 90, 83) = 0.0 - k[434]*y[IDX_CH2I];
    IJth(jmatrix, 90, 85) = 0.0 - k[425]*y[IDX_CH2I];
    IJth(jmatrix, 90, 86) = 0.0 - k[884]*y[IDX_CH2I];
    IJth(jmatrix, 90, 87) = 0.0 - k[14]*y[IDX_CH2I];
    IJth(jmatrix, 90, 89) = 0.0 + k[397]*y[IDX_CH3OHI] + k[401]*y[IDX_H2COI];
    IJth(jmatrix, 90, 90) = 0.0 - k[14]*y[IDX_CII] - k[37]*y[IDX_CNII] -
                        k[38]*y[IDX_COII] - k[39]*y[IDX_H2COII] -
                        k[40]*y[IDX_H2OII] - k[41]*y[IDX_N2II] -
                        k[42]*y[IDX_NH2II] - k[43]*y[IDX_OII] -
                        k[44]*y[IDX_O2II] - k[45]*y[IDX_OHII] - k[75]*y[IDX_HII]
                        - k[100]*y[IDX_H2II] - k[155]*y[IDX_NII] - k[242] -
                        k[243] - k[422]*y[IDX_COII] - k[423]*y[IDX_H2COII] -
                        k[424]*y[IDX_H2OII] - k[425]*y[IDX_H3OII] -
                        k[426]*y[IDX_HCNII] - k[427]*y[IDX_HCNHII] -
                        k[428]*y[IDX_HCNHII] - k[429]*y[IDX_HCOII] -
                        k[430]*y[IDX_HNOII] - k[431]*y[IDX_N2HII] -
                        k[432]*y[IDX_NHII] - k[433]*y[IDX_NH2II] -
                        k[434]*y[IDX_NH3II] - k[435]*y[IDX_O2II] -
                        k[436]*y[IDX_O2HII] - k[437]*y[IDX_OHII] -
                        k[438]*y[IDX_SiOII] - k[491]*y[IDX_HII] -
                        k[510]*y[IDX_H2II] - k[577]*y[IDX_H3II] -
                        k[653]*y[IDX_HeII] - k[654]*y[IDX_HeII] -
                        k[863]*y[IDX_CI] - k[878]*y[IDX_CH2I] -
                        k[878]*y[IDX_CH2I] - k[878]*y[IDX_CH2I] -
                        k[878]*y[IDX_CH2I] - k[879]*y[IDX_CH4I] -
                        k[880]*y[IDX_CNI] - k[881]*y[IDX_H2COI] -
                        k[882]*y[IDX_HCOI] - k[883]*y[IDX_HNOI] -
                        k[884]*y[IDX_N2I] - k[885]*y[IDX_NO2I] -
                        k[886]*y[IDX_NOI] - k[887]*y[IDX_NOI] -
                        k[888]*y[IDX_NOI] - k[889]*y[IDX_O2I] -
                        k[890]*y[IDX_O2I] - k[891]*y[IDX_O2I] -
                        k[892]*y[IDX_O2I] - k[893]*y[IDX_O2I] - k[894]*y[IDX_OI]
                        - k[895]*y[IDX_OI] - k[896]*y[IDX_OI] - k[897]*y[IDX_OI]
                        - k[898]*y[IDX_OHI] - k[899]*y[IDX_OHI] -
                        k[900]*y[IDX_OHI] - k[956]*y[IDX_H2I] - k[967]*y[IDX_HI]
                        - k[1005]*y[IDX_NI] - k[1006]*y[IDX_NI] -
                        k[1007]*y[IDX_NI] - k[1112] - k[1113] - k[1256];
    IJth(jmatrix, 90, 91) = 0.0 - k[39]*y[IDX_CH2I] + k[306]*y[IDX_EM] -
                        k[423]*y[IDX_CH2I];
    IJth(jmatrix, 90, 93) = 0.0 - k[45]*y[IDX_CH2I] - k[437]*y[IDX_CH2I] +
                        k[457]*y[IDX_CH4I];
    IJth(jmatrix, 90, 94) = 0.0 - k[880]*y[IDX_CH2I] + k[902]*y[IDX_CH3I];
    IJth(jmatrix, 90, 95) = 0.0 + k[401]*y[IDX_CHII] + k[722]*y[IDX_NII] -
                        k[881]*y[IDX_CH2I] + k[924]*y[IDX_CHI];
    IJth(jmatrix, 90, 96) = 0.0 - k[882]*y[IDX_CH2I] + k[925]*y[IDX_CHI] +
                        k[978]*y[IDX_HI];
    IJth(jmatrix, 90, 97) = 0.0 - k[653]*y[IDX_CH2I] - k[654]*y[IDX_CH2I];
    IJth(jmatrix, 90, 98) = 0.0 + k[924]*y[IDX_H2COI] + k[925]*y[IDX_HCOI] +
                        k[926]*y[IDX_HNOI] + k[938]*y[IDX_O2HI] +
                        k[958]*y[IDX_H2I];
    IJth(jmatrix, 90, 99) = 0.0 - k[577]*y[IDX_CH2I];
    IJth(jmatrix, 90, 101) = 0.0 + k[36]*y[IDX_CH2II] - k[886]*y[IDX_CH2I] -
                        k[887]*y[IDX_CH2I] - k[888]*y[IDX_CH2I];
    IJth(jmatrix, 90, 102) = 0.0 - k[1005]*y[IDX_CH2I] - k[1006]*y[IDX_CH2I] -
                        k[1007]*y[IDX_CH2I];
    IJth(jmatrix, 90, 103) = 0.0 - k[898]*y[IDX_CH2I] - k[899]*y[IDX_CH2I] -
                        k[900]*y[IDX_CH2I] + k[919]*y[IDX_CH3I];
    IJth(jmatrix, 90, 104) = 0.0 - k[889]*y[IDX_CH2I] - k[890]*y[IDX_CH2I] -
                        k[891]*y[IDX_CH2I] - k[892]*y[IDX_CH2I] -
                        k[893]*y[IDX_CH2I] + k[913]*y[IDX_CH3I];
    IJth(jmatrix, 90, 105) = 0.0 - k[863]*y[IDX_CH2I] + k[1200]*y[IDX_H2I];
    IJth(jmatrix, 90, 106) = 0.0 - k[75]*y[IDX_CH2I] - k[491]*y[IDX_CH2I];
    IJth(jmatrix, 90, 107) = 0.0 - k[429]*y[IDX_CH2I];
    IJth(jmatrix, 90, 109) = 0.0 - k[894]*y[IDX_CH2I] - k[895]*y[IDX_CH2I] -
                        k[896]*y[IDX_CH2I] - k[897]*y[IDX_CH2I];
    IJth(jmatrix, 90, 110) = 0.0 + k[298]*y[IDX_CH3II] + k[301]*y[IDX_CH4II] +
                        k[306]*y[IDX_H2COII] + k[317]*y[IDX_H3COII];
    IJth(jmatrix, 90, 112) = 0.0 - k[956]*y[IDX_CH2I] + k[958]*y[IDX_CHI] +
                        k[1200]*y[IDX_CI];
    IJth(jmatrix, 90, 113) = 0.0 - k[967]*y[IDX_CH2I] + k[968]*y[IDX_CH3I] +
                        k[978]*y[IDX_HCOI];
    IJth(jmatrix, 91, 52) = 0.0 + k[712]*y[IDX_NII] + k[808]*y[IDX_OII];
    IJth(jmatrix, 91, 53) = 0.0 - k[148]*y[IDX_H2COII];
    IJth(jmatrix, 91, 55) = 0.0 + k[49]*y[IDX_H2COI];
    IJth(jmatrix, 91, 57) = 0.0 + k[64]*y[IDX_H2COI];
    IJth(jmatrix, 91, 59) = 0.0 + k[643]*y[IDX_HCOI];
    IJth(jmatrix, 91, 60) = 0.0 + k[645]*y[IDX_HCOI];
    IJth(jmatrix, 91, 62) = 0.0 - k[228]*y[IDX_H2COII];
    IJth(jmatrix, 91, 63) = 0.0 - k[646]*y[IDX_H2COII];
    IJth(jmatrix, 91, 64) = 0.0 + k[642]*y[IDX_HCOI];
    IJth(jmatrix, 91, 65) = 0.0 + k[170]*y[IDX_H2COI];
    IJth(jmatrix, 91, 67) = 0.0 - k[452]*y[IDX_H2COII];
    IJth(jmatrix, 91, 68) = 0.0 + k[70]*y[IDX_H2COI];
    IJth(jmatrix, 91, 69) = 0.0 + k[105]*y[IDX_H2COI];
    IJth(jmatrix, 91, 70) = 0.0 - k[194]*y[IDX_H2COII];
    IJth(jmatrix, 91, 72) = 0.0 + k[416]*y[IDX_CH2II];
    IJth(jmatrix, 91, 73) = 0.0 + k[159]*y[IDX_H2COI] + k[712]*y[IDX_CH3OHI];
    IJth(jmatrix, 91, 74) = 0.0 + k[209]*y[IDX_H2COI] + k[808]*y[IDX_CH3OHI];
    IJth(jmatrix, 91, 75) = 0.0 + k[624]*y[IDX_HCOI];
    IJth(jmatrix, 91, 76) = 0.0 + k[774]*y[IDX_HCOI];
    IJth(jmatrix, 91, 77) = 0.0 + k[175]*y[IDX_H2COI] + k[759]*y[IDX_HCOI];
    IJth(jmatrix, 91, 78) = 0.0 + k[116]*y[IDX_H2COI] + k[435]*y[IDX_CH2I];
    IJth(jmatrix, 91, 79) = 0.0 + k[443]*y[IDX_OI] + k[445]*y[IDX_OHI];
    IJth(jmatrix, 91, 80) = 0.0 - k[780]*y[IDX_H2COII];
    IJth(jmatrix, 91, 81) = 0.0 + k[416]*y[IDX_CO2I];
    IJth(jmatrix, 91, 82) = 0.0 + k[117]*y[IDX_H2COI] + k[558]*y[IDX_HCOI];
    IJth(jmatrix, 91, 87) = 0.0 + k[16]*y[IDX_H2COI];
    IJth(jmatrix, 91, 88) = 0.0 - k[627]*y[IDX_H2COII];
    IJth(jmatrix, 91, 89) = 0.0 + k[402]*y[IDX_H2OI];
    IJth(jmatrix, 91, 90) = 0.0 - k[39]*y[IDX_H2COII] - k[423]*y[IDX_H2COII] +
                        k[435]*y[IDX_O2II];
    IJth(jmatrix, 91, 91) = 0.0 - k[39]*y[IDX_CH2I] - k[55]*y[IDX_CHI] -
                        k[136]*y[IDX_HCOI] - k[148]*y[IDX_MgI] -
                        k[194]*y[IDX_NH3I] - k[203]*y[IDX_NOI] -
                        k[228]*y[IDX_SiI] - k[306]*y[IDX_EM] - k[307]*y[IDX_EM]
                        - k[308]*y[IDX_EM] - k[309]*y[IDX_EM] -
                        k[423]*y[IDX_CH2I] - k[452]*y[IDX_CH4I] -
                        k[459]*y[IDX_CHI] - k[548]*y[IDX_H2COI] -
                        k[549]*y[IDX_O2I] - k[563]*y[IDX_H2OI] -
                        k[627]*y[IDX_HCNI] - k[641]*y[IDX_HCOI] -
                        k[646]*y[IDX_HNCI] - k[780]*y[IDX_NH2I] -
                        k[796]*y[IDX_NHI] - k[1217]*y[IDX_EM] - k[1284];
    IJth(jmatrix, 91, 92) = 0.0 - k[796]*y[IDX_H2COII];
    IJth(jmatrix, 91, 93) = 0.0 + k[219]*y[IDX_H2COI] + k[843]*y[IDX_HCOI];
    IJth(jmatrix, 91, 95) = 0.0 + k[16]*y[IDX_CII] + k[49]*y[IDX_CH4II] +
                        k[64]*y[IDX_CNII] + k[70]*y[IDX_COII] + k[79]*y[IDX_HII]
                        + k[105]*y[IDX_H2II] + k[116]*y[IDX_O2II] +
                        k[117]*y[IDX_H2OII] + k[142]*y[IDX_HeII] +
                        k[159]*y[IDX_NII] + k[170]*y[IDX_N2II] +
                        k[175]*y[IDX_NHII] + k[209]*y[IDX_OII] +
                        k[219]*y[IDX_OHII] - k[548]*y[IDX_H2COII] + k[1138];
    IJth(jmatrix, 91, 96) = 0.0 - k[136]*y[IDX_H2COII] + k[558]*y[IDX_H2OII] +
                        k[588]*y[IDX_H3II] + k[624]*y[IDX_HCNII] +
                        k[636]*y[IDX_HCOII] - k[641]*y[IDX_H2COII] +
                        k[642]*y[IDX_HNOII] + k[643]*y[IDX_N2HII] +
                        k[645]*y[IDX_O2HII] + k[759]*y[IDX_NHII] +
                        k[774]*y[IDX_NH2II] + k[843]*y[IDX_OHII];
    IJth(jmatrix, 91, 97) = 0.0 + k[142]*y[IDX_H2COI];
    IJth(jmatrix, 91, 98) = 0.0 - k[55]*y[IDX_H2COII] - k[459]*y[IDX_H2COII];
    IJth(jmatrix, 91, 99) = 0.0 + k[588]*y[IDX_HCOI];
    IJth(jmatrix, 91, 101) = 0.0 - k[203]*y[IDX_H2COII];
    IJth(jmatrix, 91, 103) = 0.0 + k[445]*y[IDX_CH3II];
    IJth(jmatrix, 91, 104) = 0.0 - k[549]*y[IDX_H2COII];
    IJth(jmatrix, 91, 106) = 0.0 + k[79]*y[IDX_H2COI];
    IJth(jmatrix, 91, 107) = 0.0 + k[636]*y[IDX_HCOI];
    IJth(jmatrix, 91, 108) = 0.0 + k[402]*y[IDX_CHII] - k[563]*y[IDX_H2COII];
    IJth(jmatrix, 91, 109) = 0.0 + k[443]*y[IDX_CH3II];
    IJth(jmatrix, 91, 110) = 0.0 - k[306]*y[IDX_H2COII] - k[307]*y[IDX_H2COII] -
                        k[308]*y[IDX_H2COII] - k[309]*y[IDX_H2COII] -
                        k[1217]*y[IDX_H2COII];
    IJth(jmatrix, 92, 23) = 0.0 + k[1013]*y[IDX_NI];
    IJth(jmatrix, 92, 28) = 0.0 + k[263] + k[1152];
    IJth(jmatrix, 92, 37) = 0.0 + k[1024]*y[IDX_NI];
    IJth(jmatrix, 92, 39) = 0.0 - k[1041]*y[IDX_NHI];
    IJth(jmatrix, 92, 42) = 0.0 + k[994]*y[IDX_HI];
    IJth(jmatrix, 92, 51) = 0.0 + k[951]*y[IDX_COI] + k[982]*y[IDX_HI] +
                        k[1017]*y[IDX_NI] + k[1070]*y[IDX_OI];
    IJth(jmatrix, 92, 52) = 0.0 + k[712]*y[IDX_NII] + k[713]*y[IDX_NII];
    IJth(jmatrix, 92, 57) = 0.0 - k[199]*y[IDX_NHI] + k[561]*y[IDX_H2OI];
    IJth(jmatrix, 92, 59) = 0.0 + k[339]*y[IDX_EM] - k[801]*y[IDX_NHI];
    IJth(jmatrix, 92, 60) = 0.0 - k[805]*y[IDX_NHI];
    IJth(jmatrix, 92, 63) = 0.0 + k[775]*y[IDX_NH2II];
    IJth(jmatrix, 92, 64) = 0.0 - k[800]*y[IDX_NHI];
    IJth(jmatrix, 92, 65) = 0.0 - k[201]*y[IDX_NHI];
    IJth(jmatrix, 92, 67) = 0.0 - k[1034]*y[IDX_NHI];
    IJth(jmatrix, 92, 68) = 0.0 - k[200]*y[IDX_NHI] + k[779]*y[IDX_NH2I] -
                        k[795]*y[IDX_NHI];
    IJth(jmatrix, 92, 69) = 0.0 - k[111]*y[IDX_NHI] - k[523]*y[IDX_NHI];
    IJth(jmatrix, 92, 70) = 0.0 + k[177]*y[IDX_NHII] + k[273] + k[725]*y[IDX_NII] -
                        k[1037]*y[IDX_NHI] + k[1161];
    IJth(jmatrix, 92, 71) = 0.0 + k[907]*y[IDX_NH2I];
    IJth(jmatrix, 92, 73) = 0.0 - k[166]*y[IDX_NHI] + k[712]*y[IDX_CH3OHI] +
                        k[713]*y[IDX_CH3OHI] + k[721]*y[IDX_H2COI] +
                        k[725]*y[IDX_NH3I] - k[726]*y[IDX_NHI];
    IJth(jmatrix, 92, 74) = 0.0 - k[202]*y[IDX_NHI] - k[803]*y[IDX_NHI];
    IJth(jmatrix, 92, 75) = 0.0 - k[798]*y[IDX_NHI];
    IJth(jmatrix, 92, 76) = 0.0 + k[342]*y[IDX_EM] + k[433]*y[IDX_CH2I] +
                        k[471]*y[IDX_CHI] + k[769]*y[IDX_H2COI] +
                        k[771]*y[IDX_H2OI] + k[773]*y[IDX_HCNI] +
                        k[774]*y[IDX_HCOI] + k[775]*y[IDX_HNCI] +
                        k[776]*y[IDX_NH2I] - k[802]*y[IDX_NHI];
    IJth(jmatrix, 92, 77) = 0.0 + k[175]*y[IDX_H2COI] + k[176]*y[IDX_H2OI] +
                        k[177]*y[IDX_NH3I] + k[178]*y[IDX_NOI] +
                        k[179]*y[IDX_O2I] - k[763]*y[IDX_NHI];
    IJth(jmatrix, 92, 78) = 0.0 - k[804]*y[IDX_NHI];
    IJth(jmatrix, 92, 79) = 0.0 - k[794]*y[IDX_NHI];
    IJth(jmatrix, 92, 80) = 0.0 + k[270] + k[776]*y[IDX_NH2II] + k[779]*y[IDX_COII]
                        + k[868]*y[IDX_CI] + k[907]*y[IDX_CH3I] +
                        k[983]*y[IDX_HI] + k[1031]*y[IDX_OHI] +
                        k[1073]*y[IDX_OI] + k[1158];
    IJth(jmatrix, 92, 82) = 0.0 - k[797]*y[IDX_NHI];
    IJth(jmatrix, 92, 83) = 0.0 + k[344]*y[IDX_EM];
    IJth(jmatrix, 92, 86) = 0.0 + k[884]*y[IDX_CH2I];
    IJth(jmatrix, 92, 87) = 0.0 - k[375]*y[IDX_NHI];
    IJth(jmatrix, 92, 88) = 0.0 + k[773]*y[IDX_NH2II] + k[1064]*y[IDX_OI];
    IJth(jmatrix, 92, 89) = 0.0 - k[410]*y[IDX_NHI];
    IJth(jmatrix, 92, 90) = 0.0 + k[433]*y[IDX_NH2II] + k[884]*y[IDX_N2I] +
                        k[1007]*y[IDX_NI];
    IJth(jmatrix, 92, 91) = 0.0 - k[796]*y[IDX_NHI];
    IJth(jmatrix, 92, 92) = 0.0 - k[86]*y[IDX_HII] - k[111]*y[IDX_H2II] -
                        k[166]*y[IDX_NII] - k[199]*y[IDX_CNII] -
                        k[200]*y[IDX_COII] - k[201]*y[IDX_N2II] -
                        k[202]*y[IDX_OII] - k[274] - k[275] - k[375]*y[IDX_CII]
                        - k[410]*y[IDX_CHII] - k[523]*y[IDX_H2II] -
                        k[594]*y[IDX_H3II] - k[693]*y[IDX_HeII] -
                        k[726]*y[IDX_NII] - k[763]*y[IDX_NHII] -
                        k[794]*y[IDX_CH3II] - k[795]*y[IDX_COII] -
                        k[796]*y[IDX_H2COII] - k[797]*y[IDX_H2OII] -
                        k[798]*y[IDX_HCNII] - k[799]*y[IDX_HCOII] -
                        k[800]*y[IDX_HNOII] - k[801]*y[IDX_N2HII] -
                        k[802]*y[IDX_NH2II] - k[803]*y[IDX_OII] -
                        k[804]*y[IDX_O2II] - k[805]*y[IDX_O2HII] -
                        k[806]*y[IDX_OHII] - k[869]*y[IDX_CI] - k[870]*y[IDX_CI]
                        - k[962]*y[IDX_H2I] - k[985]*y[IDX_HI] -
                        k[1018]*y[IDX_NI] - k[1034]*y[IDX_CH4I] -
                        k[1035]*y[IDX_CNI] - k[1036]*y[IDX_H2OI] -
                        k[1037]*y[IDX_NH3I] - k[1038]*y[IDX_NHI] -
                        k[1038]*y[IDX_NHI] - k[1038]*y[IDX_NHI] -
                        k[1038]*y[IDX_NHI] - k[1039]*y[IDX_NHI] -
                        k[1039]*y[IDX_NHI] - k[1039]*y[IDX_NHI] -
                        k[1039]*y[IDX_NHI] - k[1040]*y[IDX_NHI] -
                        k[1040]*y[IDX_NHI] - k[1040]*y[IDX_NHI] -
                        k[1040]*y[IDX_NHI] - k[1041]*y[IDX_NO2I] -
                        k[1042]*y[IDX_NOI] - k[1043]*y[IDX_NOI] -
                        k[1044]*y[IDX_O2I] - k[1045]*y[IDX_O2I] -
                        k[1046]*y[IDX_OI] - k[1047]*y[IDX_OI] -
                        k[1048]*y[IDX_OHI] - k[1049]*y[IDX_OHI] -
                        k[1050]*y[IDX_OHI] - k[1162] - k[1163] - k[1265];
    IJth(jmatrix, 92, 93) = 0.0 - k[806]*y[IDX_NHI];
    IJth(jmatrix, 92, 94) = 0.0 - k[1035]*y[IDX_NHI];
    IJth(jmatrix, 92, 95) = 0.0 + k[175]*y[IDX_NHII] + k[721]*y[IDX_NII] +
                        k[769]*y[IDX_NH2II];
    IJth(jmatrix, 92, 96) = 0.0 + k[774]*y[IDX_NH2II] + k[1014]*y[IDX_NI];
    IJth(jmatrix, 92, 97) = 0.0 - k[693]*y[IDX_NHI];
    IJth(jmatrix, 92, 98) = 0.0 + k[471]*y[IDX_NH2II] + k[929]*y[IDX_NI];
    IJth(jmatrix, 92, 99) = 0.0 - k[594]*y[IDX_NHI];
    IJth(jmatrix, 92, 101) = 0.0 + k[178]*y[IDX_NHII] + k[987]*y[IDX_HI] -
                        k[1042]*y[IDX_NHI] - k[1043]*y[IDX_NHI];
    IJth(jmatrix, 92, 102) = 0.0 + k[929]*y[IDX_CHI] + k[960]*y[IDX_H2I] +
                        k[1007]*y[IDX_CH2I] + k[1013]*y[IDX_H2CNI] +
                        k[1014]*y[IDX_HCOI] + k[1017]*y[IDX_HNOI] -
                        k[1018]*y[IDX_NHI] + k[1024]*y[IDX_O2HI] +
                        k[1026]*y[IDX_OHI];
    IJth(jmatrix, 92, 103) = 0.0 + k[1026]*y[IDX_NI] + k[1031]*y[IDX_NH2I] -
                        k[1048]*y[IDX_NHI] - k[1049]*y[IDX_NHI] -
                        k[1050]*y[IDX_NHI];
    IJth(jmatrix, 92, 104) = 0.0 + k[179]*y[IDX_NHII] - k[1044]*y[IDX_NHI] -
                        k[1045]*y[IDX_NHI];
    IJth(jmatrix, 92, 105) = 0.0 + k[868]*y[IDX_NH2I] - k[869]*y[IDX_NHI] -
                        k[870]*y[IDX_NHI];
    IJth(jmatrix, 92, 106) = 0.0 - k[86]*y[IDX_NHI];
    IJth(jmatrix, 92, 107) = 0.0 - k[799]*y[IDX_NHI];
    IJth(jmatrix, 92, 108) = 0.0 + k[176]*y[IDX_NHII] + k[561]*y[IDX_CNII] +
                        k[771]*y[IDX_NH2II] - k[1036]*y[IDX_NHI];
    IJth(jmatrix, 92, 109) = 0.0 - k[1046]*y[IDX_NHI] - k[1047]*y[IDX_NHI] +
                        k[1064]*y[IDX_HCNI] + k[1070]*y[IDX_HNOI] +
                        k[1073]*y[IDX_NH2I];
    IJth(jmatrix, 92, 110) = 0.0 + k[339]*y[IDX_N2HII] + k[342]*y[IDX_NH2II] +
                        k[344]*y[IDX_NH3II];
    IJth(jmatrix, 92, 111) = 0.0 + k[951]*y[IDX_HNOI];
    IJth(jmatrix, 92, 112) = 0.0 + k[960]*y[IDX_NI] - k[962]*y[IDX_NHI];
    IJth(jmatrix, 92, 113) = 0.0 + k[982]*y[IDX_HNOI] + k[983]*y[IDX_NH2I] -
                        k[985]*y[IDX_NHI] + k[987]*y[IDX_NOI] +
                        k[994]*y[IDX_OCNI];
    IJth(jmatrix, 93, 47) = 0.0 - k[849]*y[IDX_OHII];
    IJth(jmatrix, 93, 52) = 0.0 + k[656]*y[IDX_HeII];
    IJth(jmatrix, 93, 56) = 0.0 - k[850]*y[IDX_OHII];
    IJth(jmatrix, 93, 57) = 0.0 + k[225]*y[IDX_OHI];
    IJth(jmatrix, 93, 59) = 0.0 + k[826]*y[IDX_OI];
    IJth(jmatrix, 93, 60) = 0.0 + k[829]*y[IDX_OI];
    IJth(jmatrix, 93, 62) = 0.0 - k[848]*y[IDX_OHII];
    IJth(jmatrix, 93, 63) = 0.0 - k[844]*y[IDX_OHII];
    IJth(jmatrix, 93, 65) = 0.0 + k[227]*y[IDX_OHI];
    IJth(jmatrix, 93, 67) = 0.0 - k[457]*y[IDX_OHII];
    IJth(jmatrix, 93, 68) = 0.0 + k[226]*y[IDX_OHI];
    IJth(jmatrix, 93, 69) = 0.0 + k[114]*y[IDX_OHI] + k[526]*y[IDX_OI];
    IJth(jmatrix, 93, 70) = 0.0 - k[222]*y[IDX_OHII];
    IJth(jmatrix, 93, 72) = 0.0 - k[837]*y[IDX_OHII];
    IJth(jmatrix, 93, 73) = 0.0 + k[169]*y[IDX_OHI];
    IJth(jmatrix, 93, 74) = 0.0 + k[215]*y[IDX_OHI] + k[543]*y[IDX_H2I] +
                        k[816]*y[IDX_HCOI];
    IJth(jmatrix, 93, 77) = 0.0 + k[767]*y[IDX_OI];
    IJth(jmatrix, 93, 80) = 0.0 - k[188]*y[IDX_OHII] - k[791]*y[IDX_OHII];
    IJth(jmatrix, 93, 82) = 0.0 + k[1140];
    IJth(jmatrix, 93, 86) = 0.0 - k[845]*y[IDX_OHII];
    IJth(jmatrix, 93, 88) = 0.0 - k[841]*y[IDX_OHII];
    IJth(jmatrix, 93, 90) = 0.0 - k[45]*y[IDX_OHII] - k[437]*y[IDX_OHII];
    IJth(jmatrix, 93, 92) = 0.0 - k[806]*y[IDX_OHII];
    IJth(jmatrix, 93, 93) = 0.0 - k[45]*y[IDX_CH2I] - k[62]*y[IDX_CHI] -
                        k[188]*y[IDX_NH2I] - k[219]*y[IDX_H2COI] -
                        k[220]*y[IDX_H2OI] - k[221]*y[IDX_HCOI] -
                        k[222]*y[IDX_NH3I] - k[223]*y[IDX_NOI] -
                        k[224]*y[IDX_O2I] - k[348]*y[IDX_EM] - k[393]*y[IDX_CI]
                        - k[437]*y[IDX_CH2I] - k[457]*y[IDX_CH4I] -
                        k[475]*y[IDX_CHI] - k[545]*y[IDX_H2I] - k[743]*y[IDX_NI]
                        - k[791]*y[IDX_NH2I] - k[806]*y[IDX_NHI] -
                        k[830]*y[IDX_OI] - k[836]*y[IDX_CNI] -
                        k[837]*y[IDX_CO2I] - k[838]*y[IDX_COI] -
                        k[839]*y[IDX_H2COI] - k[840]*y[IDX_H2OI] -
                        k[841]*y[IDX_HCNI] - k[842]*y[IDX_HCOI] -
                        k[843]*y[IDX_HCOI] - k[844]*y[IDX_HNCI] -
                        k[845]*y[IDX_N2I] - k[846]*y[IDX_NOI] -
                        k[847]*y[IDX_OHI] - k[848]*y[IDX_SiI] -
                        k[849]*y[IDX_SiHI] - k[850]*y[IDX_SiOI] - k[1173] -
                        k[1274];
    IJth(jmatrix, 93, 94) = 0.0 - k[836]*y[IDX_OHII];
    IJth(jmatrix, 93, 95) = 0.0 - k[219]*y[IDX_OHII] - k[839]*y[IDX_OHII];
    IJth(jmatrix, 93, 96) = 0.0 - k[221]*y[IDX_OHII] + k[816]*y[IDX_OII] -
                        k[842]*y[IDX_OHII] - k[843]*y[IDX_OHII];
    IJth(jmatrix, 93, 97) = 0.0 + k[656]*y[IDX_CH3OHI] + k[673]*y[IDX_H2OI];
    IJth(jmatrix, 93, 98) = 0.0 - k[62]*y[IDX_OHII] - k[475]*y[IDX_OHII];
    IJth(jmatrix, 93, 99) = 0.0 + k[599]*y[IDX_OI];
    IJth(jmatrix, 93, 101) = 0.0 - k[223]*y[IDX_OHII] - k[846]*y[IDX_OHII];
    IJth(jmatrix, 93, 102) = 0.0 - k[743]*y[IDX_OHII];
    IJth(jmatrix, 93, 103) = 0.0 + k[90]*y[IDX_HII] + k[114]*y[IDX_H2II] +
                        k[169]*y[IDX_NII] + k[215]*y[IDX_OII] +
                        k[225]*y[IDX_CNII] + k[226]*y[IDX_COII] +
                        k[227]*y[IDX_N2II] - k[847]*y[IDX_OHII] + k[1175];
    IJth(jmatrix, 93, 104) = 0.0 - k[224]*y[IDX_OHII];
    IJth(jmatrix, 93, 105) = 0.0 - k[393]*y[IDX_OHII];
    IJth(jmatrix, 93, 106) = 0.0 + k[90]*y[IDX_OHI];
    IJth(jmatrix, 93, 108) = 0.0 - k[220]*y[IDX_OHII] + k[673]*y[IDX_HeII] -
                        k[840]*y[IDX_OHII];
    IJth(jmatrix, 93, 109) = 0.0 + k[526]*y[IDX_H2II] + k[599]*y[IDX_H3II] +
                        k[767]*y[IDX_NHII] + k[826]*y[IDX_N2HII] +
                        k[829]*y[IDX_O2HII] - k[830]*y[IDX_OHII];
    IJth(jmatrix, 93, 110) = 0.0 - k[348]*y[IDX_OHII];
    IJth(jmatrix, 93, 111) = 0.0 - k[838]*y[IDX_OHII];
    IJth(jmatrix, 93, 112) = 0.0 + k[543]*y[IDX_OII] - k[545]*y[IDX_OHII];
    IJth(jmatrix, 94, 36) = 0.0 + k[744]*y[IDX_NI];
    IJth(jmatrix, 94, 38) = 0.0 + k[1027]*y[IDX_NI];
    IJth(jmatrix, 94, 39) = 0.0 - k[945]*y[IDX_CNI];
    IJth(jmatrix, 94, 42) = 0.0 + k[283] + k[378]*y[IDX_CII] + k[698]*y[IDX_HeII] +
                        k[874]*y[IDX_CI] + k[995]*y[IDX_HI] + k[1079]*y[IDX_OI]
                        + k[1172];
    IJth(jmatrix, 94, 46) = 0.0 - k[950]*y[IDX_CNI];
    IJth(jmatrix, 94, 51) = 0.0 - k[944]*y[IDX_CNI];
    IJth(jmatrix, 94, 57) = 0.0 + k[27]*y[IDX_CI] + k[37]*y[IDX_CH2I] +
                        k[53]*y[IDX_CHI] + k[63]*y[IDX_COI] + k[64]*y[IDX_H2COI]
                        + k[65]*y[IDX_HCNI] + k[66]*y[IDX_HCOI] +
                        k[67]*y[IDX_NOI] + k[68]*y[IDX_O2I] + k[126]*y[IDX_HI] +
                        k[183]*y[IDX_NH2I] + k[199]*y[IDX_NHI] +
                        k[216]*y[IDX_OI] + k[225]*y[IDX_OHI];
    IJth(jmatrix, 94, 58) = 0.0 + k[327]*y[IDX_EM];
    IJth(jmatrix, 94, 60) = 0.0 - k[483]*y[IDX_CNI];
    IJth(jmatrix, 94, 63) = 0.0 + k[262] + k[626]*y[IDX_HCNII] + k[1151];
    IJth(jmatrix, 94, 64) = 0.0 - k[482]*y[IDX_CNI];
    IJth(jmatrix, 94, 65) = 0.0 - k[69]*y[IDX_CNI];
    IJth(jmatrix, 94, 67) = 0.0 - k[920]*y[IDX_CNI];
    IJth(jmatrix, 94, 69) = 0.0 - k[103]*y[IDX_CNI] - k[513]*y[IDX_CNI];
    IJth(jmatrix, 94, 70) = 0.0 - k[1033]*y[IDX_CNI];
    IJth(jmatrix, 94, 71) = 0.0 - k[902]*y[IDX_CNI];
    IJth(jmatrix, 94, 72) = 0.0 + k[620]*y[IDX_HCNII];
    IJth(jmatrix, 94, 73) = 0.0 - k[157]*y[IDX_CNI];
    IJth(jmatrix, 94, 74) = 0.0 - k[811]*y[IDX_CNI];
    IJth(jmatrix, 94, 75) = 0.0 + k[326]*y[IDX_EM] + k[385]*y[IDX_CI] +
                        k[426]*y[IDX_CH2I] + k[463]*y[IDX_CHI] +
                        k[565]*y[IDX_H2OI] + k[620]*y[IDX_CO2I] +
                        k[621]*y[IDX_COI] + k[622]*y[IDX_H2COI] +
                        k[623]*y[IDX_HCNI] + k[624]*y[IDX_HCOI] +
                        k[626]*y[IDX_HNCI] + k[784]*y[IDX_NH2I] +
                        k[798]*y[IDX_NHI] + k[853]*y[IDX_OHI];
    IJth(jmatrix, 94, 77) = 0.0 - k[747]*y[IDX_CNI];
    IJth(jmatrix, 94, 80) = 0.0 + k[183]*y[IDX_CNII] + k[784]*y[IDX_HCNII];
    IJth(jmatrix, 94, 86) = 0.0 + k[865]*y[IDX_CI];
    IJth(jmatrix, 94, 87) = 0.0 + k[378]*y[IDX_OCNI];
    IJth(jmatrix, 94, 88) = 0.0 + k[65]*y[IDX_CNII] + k[259] + k[623]*y[IDX_HCNII] +
                        k[976]*y[IDX_HI] + k[1063]*y[IDX_OI] +
                        k[1094]*y[IDX_OHI] + k[1147];
    IJth(jmatrix, 94, 90) = 0.0 + k[37]*y[IDX_CNII] + k[426]*y[IDX_HCNII] -
                        k[880]*y[IDX_CNI];
    IJth(jmatrix, 94, 92) = 0.0 + k[199]*y[IDX_CNII] + k[798]*y[IDX_HCNII] +
                        k[869]*y[IDX_CI] - k[1035]*y[IDX_CNI];
    IJth(jmatrix, 94, 93) = 0.0 - k[836]*y[IDX_CNI];
    IJth(jmatrix, 94, 94) = 0.0 - k[69]*y[IDX_N2II] - k[103]*y[IDX_H2II] -
                        k[157]*y[IDX_NII] - k[251] - k[482]*y[IDX_HNOII] -
                        k[483]*y[IDX_O2HII] - k[513]*y[IDX_H2II] -
                        k[581]*y[IDX_H3II] - k[663]*y[IDX_HeII] -
                        k[664]*y[IDX_HeII] - k[747]*y[IDX_NHII] -
                        k[811]*y[IDX_OII] - k[836]*y[IDX_OHII] -
                        k[880]*y[IDX_CH2I] - k[902]*y[IDX_CH3I] -
                        k[920]*y[IDX_CH4I] - k[942]*y[IDX_H2COI] -
                        k[943]*y[IDX_HCOI] - k[944]*y[IDX_HNOI] -
                        k[945]*y[IDX_NO2I] - k[946]*y[IDX_NOI] -
                        k[947]*y[IDX_NOI] - k[948]*y[IDX_O2I] -
                        k[949]*y[IDX_O2I] - k[950]*y[IDX_SiH4I] -
                        k[959]*y[IDX_H2I] - k[1011]*y[IDX_NI] -
                        k[1033]*y[IDX_NH3I] - k[1035]*y[IDX_NHI] -
                        k[1057]*y[IDX_OI] - k[1058]*y[IDX_OI] -
                        k[1090]*y[IDX_OHI] - k[1091]*y[IDX_OHI] - k[1130] -
                        k[1263];
    IJth(jmatrix, 94, 95) = 0.0 + k[64]*y[IDX_CNII] + k[622]*y[IDX_HCNII] -
                        k[942]*y[IDX_CNI];
    IJth(jmatrix, 94, 96) = 0.0 + k[66]*y[IDX_CNII] + k[624]*y[IDX_HCNII] -
                        k[943]*y[IDX_CNI];
    IJth(jmatrix, 94, 97) = 0.0 - k[663]*y[IDX_CNI] - k[664]*y[IDX_CNI] +
                        k[698]*y[IDX_OCNI];
    IJth(jmatrix, 94, 98) = 0.0 + k[53]*y[IDX_CNII] + k[463]*y[IDX_HCNII] +
                        k[928]*y[IDX_NI];
    IJth(jmatrix, 94, 99) = 0.0 - k[581]*y[IDX_CNI];
    IJth(jmatrix, 94, 101) = 0.0 + k[67]*y[IDX_CNII] + k[871]*y[IDX_CI] -
                        k[946]*y[IDX_CNI] - k[947]*y[IDX_CNI];
    IJth(jmatrix, 94, 102) = 0.0 + k[744]*y[IDX_SiCII] + k[928]*y[IDX_CHI] -
                        k[1011]*y[IDX_CNI] + k[1027]*y[IDX_SiCI] +
                        k[1194]*y[IDX_CI];
    IJth(jmatrix, 94, 103) = 0.0 + k[225]*y[IDX_CNII] + k[853]*y[IDX_HCNII] -
                        k[1090]*y[IDX_CNI] - k[1091]*y[IDX_CNI] +
                        k[1094]*y[IDX_HCNI];
    IJth(jmatrix, 94, 104) = 0.0 + k[68]*y[IDX_CNII] - k[948]*y[IDX_CNI] -
                        k[949]*y[IDX_CNI];
    IJth(jmatrix, 94, 105) = 0.0 + k[27]*y[IDX_CNII] + k[385]*y[IDX_HCNII] +
                        k[865]*y[IDX_N2I] + k[869]*y[IDX_NHI] +
                        k[871]*y[IDX_NOI] + k[874]*y[IDX_OCNI] +
                        k[1194]*y[IDX_NI];
    IJth(jmatrix, 94, 108) = 0.0 + k[565]*y[IDX_HCNII];
    IJth(jmatrix, 94, 109) = 0.0 + k[216]*y[IDX_CNII] - k[1057]*y[IDX_CNI] -
                        k[1058]*y[IDX_CNI] + k[1063]*y[IDX_HCNI] +
                        k[1079]*y[IDX_OCNI];
    IJth(jmatrix, 94, 110) = 0.0 + k[326]*y[IDX_HCNII] + k[327]*y[IDX_HCNHII];
    IJth(jmatrix, 94, 111) = 0.0 + k[63]*y[IDX_CNII] + k[621]*y[IDX_HCNII];
    IJth(jmatrix, 94, 112) = 0.0 - k[959]*y[IDX_CNI];
    IJth(jmatrix, 94, 113) = 0.0 + k[126]*y[IDX_CNII] + k[976]*y[IDX_HCNI] +
                        k[995]*y[IDX_OCNI];
    IJth(jmatrix, 95, 18) = 0.0 + k[1347] + k[1348] + k[1349] + k[1350];
    IJth(jmatrix, 95, 37) = 0.0 + k[1003]*y[IDX_HCOI];
    IJth(jmatrix, 95, 39) = 0.0 + k[885]*y[IDX_CH2I] + k[909]*y[IDX_CH3I];
    IJth(jmatrix, 95, 49) = 0.0 + k[438]*y[IDX_CH2I];
    IJth(jmatrix, 95, 51) = 0.0 + k[999]*y[IDX_HCOI];
    IJth(jmatrix, 95, 52) = 0.0 + k[247] + k[396]*y[IDX_CHII] + k[1119];
    IJth(jmatrix, 95, 53) = 0.0 + k[148]*y[IDX_H2COII];
    IJth(jmatrix, 95, 55) = 0.0 - k[49]*y[IDX_H2COI] - k[449]*y[IDX_H2COI];
    IJth(jmatrix, 95, 57) = 0.0 - k[64]*y[IDX_H2COI] - k[479]*y[IDX_H2COI];
    IJth(jmatrix, 95, 58) = 0.0 - k[633]*y[IDX_H2COI] - k[634]*y[IDX_H2COI];
    IJth(jmatrix, 95, 59) = 0.0 - k[735]*y[IDX_H2COI];
    IJth(jmatrix, 95, 60) = 0.0 - k[552]*y[IDX_H2COI];
    IJth(jmatrix, 95, 62) = 0.0 + k[228]*y[IDX_H2COII];
    IJth(jmatrix, 95, 63) = 0.0 + k[647]*y[IDX_H3COII];
    IJth(jmatrix, 95, 64) = 0.0 - k[550]*y[IDX_H2COI];
    IJth(jmatrix, 95, 65) = 0.0 - k[170]*y[IDX_H2COI] - k[730]*y[IDX_H2COI];
    IJth(jmatrix, 95, 66) = 0.0 + k[320]*y[IDX_EM] + k[461]*y[IDX_CHI] +
                        k[564]*y[IDX_H2OI] + k[628]*y[IDX_HCNI] +
                        k[647]*y[IDX_HNCI] + k[782]*y[IDX_NH2I];
    IJth(jmatrix, 95, 68) = 0.0 - k[70]*y[IDX_H2COI] - k[484]*y[IDX_H2COI];
    IJth(jmatrix, 95, 69) = 0.0 - k[105]*y[IDX_H2COI] - k[517]*y[IDX_H2COI];
    IJth(jmatrix, 95, 70) = 0.0 + k[194]*y[IDX_H2COII];
    IJth(jmatrix, 95, 71) = 0.0 - k[903]*y[IDX_H2COI] + k[909]*y[IDX_NO2I] +
                        k[911]*y[IDX_O2I] + k[916]*y[IDX_OI] + k[918]*y[IDX_OHI];
    IJth(jmatrix, 95, 73) = 0.0 - k[159]*y[IDX_H2COI] - k[721]*y[IDX_H2COI] -
                        k[722]*y[IDX_H2COI];
    IJth(jmatrix, 95, 74) = 0.0 - k[209]*y[IDX_H2COI] - k[813]*y[IDX_H2COI];
    IJth(jmatrix, 95, 75) = 0.0 - k[622]*y[IDX_H2COI];
    IJth(jmatrix, 95, 76) = 0.0 - k[769]*y[IDX_H2COI] - k[770]*y[IDX_H2COI];
    IJth(jmatrix, 95, 77) = 0.0 - k[175]*y[IDX_H2COI] - k[752]*y[IDX_H2COI] -
                        k[753]*y[IDX_H2COI];
    IJth(jmatrix, 95, 78) = 0.0 - k[116]*y[IDX_H2COI] - k[551]*y[IDX_H2COI];
    IJth(jmatrix, 95, 79) = 0.0 - k[440]*y[IDX_H2COI];
    IJth(jmatrix, 95, 80) = 0.0 + k[782]*y[IDX_H3COII];
    IJth(jmatrix, 95, 81) = 0.0 - k[417]*y[IDX_H2COI];
    IJth(jmatrix, 95, 82) = 0.0 - k[117]*y[IDX_H2COI] - k[554]*y[IDX_H2COI];
    IJth(jmatrix, 95, 85) = 0.0 - k[607]*y[IDX_H2COI];
    IJth(jmatrix, 95, 87) = 0.0 - k[16]*y[IDX_H2COI] - k[368]*y[IDX_H2COI] -
                        k[369]*y[IDX_H2COI];
    IJth(jmatrix, 95, 88) = 0.0 + k[628]*y[IDX_H3COII];
    IJth(jmatrix, 95, 89) = 0.0 + k[396]*y[IDX_CH3OHI] - k[399]*y[IDX_H2COI] -
                        k[400]*y[IDX_H2COI] - k[401]*y[IDX_H2COI];
    IJth(jmatrix, 95, 90) = 0.0 + k[39]*y[IDX_H2COII] + k[438]*y[IDX_SiOII] -
                        k[881]*y[IDX_H2COI] + k[885]*y[IDX_NO2I] +
                        k[886]*y[IDX_NOI] + k[892]*y[IDX_O2I] +
                        k[898]*y[IDX_OHI];
    IJth(jmatrix, 95, 91) = 0.0 + k[39]*y[IDX_CH2I] + k[55]*y[IDX_CHI] +
                        k[136]*y[IDX_HCOI] + k[148]*y[IDX_MgI] +
                        k[194]*y[IDX_NH3I] + k[203]*y[IDX_NOI] +
                        k[228]*y[IDX_SiI] - k[548]*y[IDX_H2COI] +
                        k[1217]*y[IDX_EM];
    IJth(jmatrix, 95, 93) = 0.0 - k[219]*y[IDX_H2COI] - k[839]*y[IDX_H2COI];
    IJth(jmatrix, 95, 94) = 0.0 - k[942]*y[IDX_H2COI];
    IJth(jmatrix, 95, 95) = 0.0 - k[16]*y[IDX_CII] - k[49]*y[IDX_CH4II] -
                        k[64]*y[IDX_CNII] - k[70]*y[IDX_COII] - k[79]*y[IDX_HII]
                        - k[105]*y[IDX_H2II] - k[116]*y[IDX_O2II] -
                        k[117]*y[IDX_H2OII] - k[142]*y[IDX_HeII] -
                        k[159]*y[IDX_NII] - k[170]*y[IDX_N2II] -
                        k[175]*y[IDX_NHII] - k[209]*y[IDX_OII] -
                        k[219]*y[IDX_OHII] - k[255] - k[368]*y[IDX_CII] -
                        k[369]*y[IDX_CII] - k[399]*y[IDX_CHII] -
                        k[400]*y[IDX_CHII] - k[401]*y[IDX_CHII] -
                        k[417]*y[IDX_CH2II] - k[440]*y[IDX_CH3II] -
                        k[449]*y[IDX_CH4II] - k[479]*y[IDX_CNII] -
                        k[484]*y[IDX_COII] - k[497]*y[IDX_HII] -
                        k[498]*y[IDX_HII] - k[517]*y[IDX_H2II] -
                        k[548]*y[IDX_H2COII] - k[550]*y[IDX_HNOII] -
                        k[551]*y[IDX_O2II] - k[552]*y[IDX_O2HII] -
                        k[554]*y[IDX_H2OII] - k[585]*y[IDX_H3II] -
                        k[607]*y[IDX_H3OII] - k[622]*y[IDX_HCNII] -
                        k[633]*y[IDX_HCNHII] - k[634]*y[IDX_HCNHII] -
                        k[635]*y[IDX_HCOII] - k[670]*y[IDX_HeII] -
                        k[671]*y[IDX_HeII] - k[672]*y[IDX_HeII] -
                        k[721]*y[IDX_NII] - k[722]*y[IDX_NII] -
                        k[730]*y[IDX_N2II] - k[735]*y[IDX_N2HII] -
                        k[752]*y[IDX_NHII] - k[753]*y[IDX_NHII] -
                        k[769]*y[IDX_NH2II] - k[770]*y[IDX_NH2II] -
                        k[813]*y[IDX_OII] - k[839]*y[IDX_OHII] -
                        k[881]*y[IDX_CH2I] - k[903]*y[IDX_CH3I] -
                        k[924]*y[IDX_CHI] - k[942]*y[IDX_CNI] - k[974]*y[IDX_HI]
                        - k[1061]*y[IDX_OI] - k[1093]*y[IDX_OHI] - k[1136] -
                        k[1137] - k[1138] - k[1139] - k[1252];
    IJth(jmatrix, 95, 96) = 0.0 + k[136]*y[IDX_H2COII] + k[998]*y[IDX_HCOI] +
                        k[998]*y[IDX_HCOI] + k[999]*y[IDX_HNOI] +
                        k[1003]*y[IDX_O2HI];
    IJth(jmatrix, 95, 97) = 0.0 - k[142]*y[IDX_H2COI] - k[670]*y[IDX_H2COI] -
                        k[671]*y[IDX_H2COI] - k[672]*y[IDX_H2COI];
    IJth(jmatrix, 95, 98) = 0.0 + k[55]*y[IDX_H2COII] + k[461]*y[IDX_H3COII] -
                        k[924]*y[IDX_H2COI];
    IJth(jmatrix, 95, 99) = 0.0 - k[585]*y[IDX_H2COI];
    IJth(jmatrix, 95, 101) = 0.0 + k[203]*y[IDX_H2COII] + k[886]*y[IDX_CH2I];
    IJth(jmatrix, 95, 103) = 0.0 + k[898]*y[IDX_CH2I] + k[918]*y[IDX_CH3I] -
                        k[1093]*y[IDX_H2COI];
    IJth(jmatrix, 95, 104) = 0.0 + k[892]*y[IDX_CH2I] + k[911]*y[IDX_CH3I];
    IJth(jmatrix, 95, 106) = 0.0 - k[79]*y[IDX_H2COI] - k[497]*y[IDX_H2COI] -
                        k[498]*y[IDX_H2COI];
    IJth(jmatrix, 95, 107) = 0.0 - k[635]*y[IDX_H2COI];
    IJth(jmatrix, 95, 108) = 0.0 + k[564]*y[IDX_H3COII];
    IJth(jmatrix, 95, 109) = 0.0 + k[916]*y[IDX_CH3I] - k[1061]*y[IDX_H2COI];
    IJth(jmatrix, 95, 110) = 0.0 + k[320]*y[IDX_H3COII] + k[1217]*y[IDX_H2COII];
    IJth(jmatrix, 95, 113) = 0.0 - k[974]*y[IDX_H2COI];
    IJth(jmatrix, 96, 37) = 0.0 + k[937]*y[IDX_CHI] - k[1003]*y[IDX_HCOI];
    IJth(jmatrix, 96, 49) = 0.0 - k[138]*y[IDX_HCOI];
    IJth(jmatrix, 96, 51) = 0.0 - k[999]*y[IDX_HCOI];
    IJth(jmatrix, 96, 52) = 0.0 + k[366]*y[IDX_CII];
    IJth(jmatrix, 96, 53) = 0.0 + k[149]*y[IDX_HCOII];
    IJth(jmatrix, 96, 57) = 0.0 - k[66]*y[IDX_HCOI] - k[480]*y[IDX_HCOI];
    IJth(jmatrix, 96, 59) = 0.0 - k[643]*y[IDX_HCOI];
    IJth(jmatrix, 96, 60) = 0.0 - k[645]*y[IDX_HCOI];
    IJth(jmatrix, 96, 63) = 0.0 + k[646]*y[IDX_H2COII];
    IJth(jmatrix, 96, 64) = 0.0 - k[642]*y[IDX_HCOI];
    IJth(jmatrix, 96, 65) = 0.0 - k[171]*y[IDX_HCOI] - k[731]*y[IDX_HCOI];
    IJth(jmatrix, 96, 66) = 0.0 + k[321]*y[IDX_EM];
    IJth(jmatrix, 96, 68) = 0.0 - k[71]*y[IDX_HCOI] + k[484]*y[IDX_H2COI];
    IJth(jmatrix, 96, 69) = 0.0 - k[108]*y[IDX_HCOI] - k[519]*y[IDX_HCOI];
    IJth(jmatrix, 96, 71) = 0.0 + k[903]*y[IDX_H2COI] - k[905]*y[IDX_HCOI] +
                        k[912]*y[IDX_O2I];
    IJth(jmatrix, 96, 72) = 0.0 + k[750]*y[IDX_NHII] + k[923]*y[IDX_CHI];
    IJth(jmatrix, 96, 73) = 0.0 - k[162]*y[IDX_HCOI] - k[723]*y[IDX_HCOI];
    IJth(jmatrix, 96, 74) = 0.0 - k[211]*y[IDX_HCOI] - k[816]*y[IDX_HCOI];
    IJth(jmatrix, 96, 75) = 0.0 - k[624]*y[IDX_HCOI] - k[625]*y[IDX_HCOI];
    IJth(jmatrix, 96, 76) = 0.0 - k[180]*y[IDX_HCOI] + k[770]*y[IDX_H2COI] -
                        k[774]*y[IDX_HCOI];
    IJth(jmatrix, 96, 77) = 0.0 + k[750]*y[IDX_CO2I] - k[759]*y[IDX_HCOI];
    IJth(jmatrix, 96, 78) = 0.0 - k[137]*y[IDX_HCOI] - k[644]*y[IDX_HCOI];
    IJth(jmatrix, 96, 79) = 0.0 - k[46]*y[IDX_HCOI] - k[441]*y[IDX_HCOI];
    IJth(jmatrix, 96, 80) = 0.0 + k[780]*y[IDX_H2COII];
    IJth(jmatrix, 96, 81) = 0.0 - k[419]*y[IDX_HCOI];
    IJth(jmatrix, 96, 82) = 0.0 - k[118]*y[IDX_HCOI] - k[557]*y[IDX_HCOI] -
                        k[558]*y[IDX_HCOI];
    IJth(jmatrix, 96, 83) = 0.0 - k[189]*y[IDX_HCOI];
    IJth(jmatrix, 96, 87) = 0.0 - k[17]*y[IDX_HCOI] + k[366]*y[IDX_CH3OHI] -
                        k[372]*y[IDX_HCOI];
    IJth(jmatrix, 96, 88) = 0.0 + k[627]*y[IDX_H2COII];
    IJth(jmatrix, 96, 89) = 0.0 - k[31]*y[IDX_HCOI] - k[406]*y[IDX_HCOI] +
                        k[413]*y[IDX_O2I];
    IJth(jmatrix, 96, 90) = 0.0 + k[423]*y[IDX_H2COII] + k[881]*y[IDX_H2COI] -
                        k[882]*y[IDX_HCOI] + k[893]*y[IDX_O2I] +
                        k[896]*y[IDX_OI];
    IJth(jmatrix, 96, 91) = 0.0 - k[136]*y[IDX_HCOI] + k[309]*y[IDX_EM] +
                        k[423]*y[IDX_CH2I] + k[459]*y[IDX_CHI] +
                        k[548]*y[IDX_H2COI] + k[563]*y[IDX_H2OI] +
                        k[627]*y[IDX_HCNI] - k[641]*y[IDX_HCOI] +
                        k[646]*y[IDX_HNCI] + k[780]*y[IDX_NH2I];
    IJth(jmatrix, 96, 93) = 0.0 - k[221]*y[IDX_HCOI] - k[842]*y[IDX_HCOI] -
                        k[843]*y[IDX_HCOI];
    IJth(jmatrix, 96, 94) = 0.0 + k[942]*y[IDX_H2COI] - k[943]*y[IDX_HCOI];
    IJth(jmatrix, 96, 95) = 0.0 + k[484]*y[IDX_COII] + k[548]*y[IDX_H2COII] +
                        k[770]*y[IDX_NH2II] + k[881]*y[IDX_CH2I] +
                        k[903]*y[IDX_CH3I] + k[924]*y[IDX_CHI] +
                        k[942]*y[IDX_CNI] + k[974]*y[IDX_HI] + k[1061]*y[IDX_OI]
                        + k[1093]*y[IDX_OHI];
    IJth(jmatrix, 96, 96) = 0.0 - k[17]*y[IDX_CII] - k[31]*y[IDX_CHII] -
                        k[46]*y[IDX_CH3II] - k[66]*y[IDX_CNII] -
                        k[71]*y[IDX_COII] - k[82]*y[IDX_HII] -
                        k[108]*y[IDX_H2II] - k[118]*y[IDX_H2OII] -
                        k[136]*y[IDX_H2COII] - k[137]*y[IDX_O2II] -
                        k[138]*y[IDX_SiOII] - k[162]*y[IDX_NII] -
                        k[171]*y[IDX_N2II] - k[180]*y[IDX_NH2II] -
                        k[189]*y[IDX_NH3II] - k[211]*y[IDX_OII] -
                        k[221]*y[IDX_OHII] - k[260] - k[261] - k[372]*y[IDX_CII]
                        - k[406]*y[IDX_CHII] - k[419]*y[IDX_CH2II] -
                        k[441]*y[IDX_CH3II] - k[480]*y[IDX_CNII] -
                        k[500]*y[IDX_HII] - k[501]*y[IDX_HII] -
                        k[519]*y[IDX_H2II] - k[557]*y[IDX_H2OII] -
                        k[558]*y[IDX_H2OII] - k[588]*y[IDX_H3II] -
                        k[624]*y[IDX_HCNII] - k[625]*y[IDX_HCNII] -
                        k[636]*y[IDX_HCOII] - k[641]*y[IDX_H2COII] -
                        k[642]*y[IDX_HNOII] - k[643]*y[IDX_N2HII] -
                        k[644]*y[IDX_O2II] - k[645]*y[IDX_O2HII] -
                        k[680]*y[IDX_HeII] - k[681]*y[IDX_HeII] -
                        k[682]*y[IDX_HeII] - k[723]*y[IDX_NII] -
                        k[731]*y[IDX_N2II] - k[759]*y[IDX_NHII] -
                        k[774]*y[IDX_NH2II] - k[816]*y[IDX_OII] -
                        k[842]*y[IDX_OHII] - k[843]*y[IDX_OHII] -
                        k[864]*y[IDX_CI] - k[882]*y[IDX_CH2I] -
                        k[905]*y[IDX_CH3I] - k[925]*y[IDX_CHI] -
                        k[943]*y[IDX_CNI] - k[977]*y[IDX_HI] - k[978]*y[IDX_HI]
                        - k[997]*y[IDX_HCOI] - k[997]*y[IDX_HCOI] -
                        k[997]*y[IDX_HCOI] - k[997]*y[IDX_HCOI] -
                        k[998]*y[IDX_HCOI] - k[998]*y[IDX_HCOI] -
                        k[998]*y[IDX_HCOI] - k[998]*y[IDX_HCOI] -
                        k[999]*y[IDX_HNOI] - k[1000]*y[IDX_NOI] -
                        k[1001]*y[IDX_O2I] - k[1002]*y[IDX_O2I] -
                        k[1003]*y[IDX_O2HI] - k[1014]*y[IDX_NI] -
                        k[1015]*y[IDX_NI] - k[1016]*y[IDX_NI] -
                        k[1066]*y[IDX_OI] - k[1067]*y[IDX_OI] -
                        k[1096]*y[IDX_OHI] - k[1149] - k[1150] - k[1261];
    IJth(jmatrix, 96, 97) = 0.0 - k[680]*y[IDX_HCOI] - k[681]*y[IDX_HCOI] -
                        k[682]*y[IDX_HCOI];
    IJth(jmatrix, 96, 98) = 0.0 + k[459]*y[IDX_H2COII] + k[923]*y[IDX_CO2I] +
                        k[924]*y[IDX_H2COI] - k[925]*y[IDX_HCOI] +
                        k[931]*y[IDX_NOI] + k[936]*y[IDX_O2I] +
                        k[937]*y[IDX_O2HI] + k[941]*y[IDX_OHI];
    IJth(jmatrix, 96, 99) = 0.0 - k[588]*y[IDX_HCOI];
    IJth(jmatrix, 96, 101) = 0.0 + k[931]*y[IDX_CHI] - k[1000]*y[IDX_HCOI];
    IJth(jmatrix, 96, 102) = 0.0 - k[1014]*y[IDX_HCOI] - k[1015]*y[IDX_HCOI] -
                        k[1016]*y[IDX_HCOI];
    IJth(jmatrix, 96, 103) = 0.0 + k[941]*y[IDX_CHI] + k[1093]*y[IDX_H2COI] -
                        k[1096]*y[IDX_HCOI];
    IJth(jmatrix, 96, 104) = 0.0 + k[413]*y[IDX_CHII] + k[893]*y[IDX_CH2I] +
                        k[912]*y[IDX_CH3I] + k[936]*y[IDX_CHI] -
                        k[1001]*y[IDX_HCOI] - k[1002]*y[IDX_HCOI];
    IJth(jmatrix, 96, 105) = 0.0 - k[864]*y[IDX_HCOI];
    IJth(jmatrix, 96, 106) = 0.0 - k[82]*y[IDX_HCOI] - k[500]*y[IDX_HCOI] -
                        k[501]*y[IDX_HCOI];
    IJth(jmatrix, 96, 107) = 0.0 + k[149]*y[IDX_MgI] - k[636]*y[IDX_HCOI];
    IJth(jmatrix, 96, 108) = 0.0 + k[563]*y[IDX_H2COII];
    IJth(jmatrix, 96, 109) = 0.0 + k[896]*y[IDX_CH2I] + k[1061]*y[IDX_H2COI] -
                        k[1066]*y[IDX_HCOI] - k[1067]*y[IDX_HCOI];
    IJth(jmatrix, 96, 110) = 0.0 + k[309]*y[IDX_H2COII] + k[321]*y[IDX_H3COII];
    IJth(jmatrix, 96, 113) = 0.0 + k[974]*y[IDX_H2COI] - k[977]*y[IDX_HCOI] -
                        k[978]*y[IDX_HCOI];
    IJth(jmatrix, 97, 26) = 0.0 - k[675]*y[IDX_HeII];
    IJth(jmatrix, 97, 33) = 0.0 - k[700]*y[IDX_HeII];
    IJth(jmatrix, 97, 38) = 0.0 - k[701]*y[IDX_HeII] - k[702]*y[IDX_HeII];
    IJth(jmatrix, 97, 42) = 0.0 - k[697]*y[IDX_HeII] - k[698]*y[IDX_HeII];
    IJth(jmatrix, 97, 43) = 0.0 - k[703]*y[IDX_HeII] - k[704]*y[IDX_HeII];
    IJth(jmatrix, 97, 46) = 0.0 - k[707]*y[IDX_HeII] - k[708]*y[IDX_HeII];
    IJth(jmatrix, 97, 47) = 0.0 - k[709]*y[IDX_HeII];
    IJth(jmatrix, 97, 48) = 0.0 - k[705]*y[IDX_HeII] - k[706]*y[IDX_HeII];
    IJth(jmatrix, 97, 51) = 0.0 - k[686]*y[IDX_HeII] - k[687]*y[IDX_HeII];
    IJth(jmatrix, 97, 52) = 0.0 - k[656]*y[IDX_HeII] - k[657]*y[IDX_HeII];
    IJth(jmatrix, 97, 56) = 0.0 - k[710]*y[IDX_HeII] - k[711]*y[IDX_HeII];
    IJth(jmatrix, 97, 62) = 0.0 - k[147]*y[IDX_HeII];
    IJth(jmatrix, 97, 63) = 0.0 - k[683]*y[IDX_HeII] - k[684]*y[IDX_HeII] -
                        k[685]*y[IDX_HeII];
    IJth(jmatrix, 97, 67) = 0.0 - k[140]*y[IDX_HeII] - k[658]*y[IDX_HeII] -
                        k[659]*y[IDX_HeII] - k[660]*y[IDX_HeII] -
                        k[661]*y[IDX_HeII];
    IJth(jmatrix, 97, 70) = 0.0 - k[145]*y[IDX_HeII] - k[691]*y[IDX_HeII] -
                        k[692]*y[IDX_HeII];
    IJth(jmatrix, 97, 71) = 0.0 - k[655]*y[IDX_HeII];
    IJth(jmatrix, 97, 72) = 0.0 - k[665]*y[IDX_HeII] - k[666]*y[IDX_HeII] -
                        k[667]*y[IDX_HeII] - k[668]*y[IDX_HeII];
    IJth(jmatrix, 97, 80) = 0.0 - k[689]*y[IDX_HeII] - k[690]*y[IDX_HeII];
    IJth(jmatrix, 97, 86) = 0.0 - k[144]*y[IDX_HeII] - k[688]*y[IDX_HeII];
    IJth(jmatrix, 97, 88) = 0.0 - k[676]*y[IDX_HeII] - k[677]*y[IDX_HeII] -
                        k[678]*y[IDX_HeII] - k[679]*y[IDX_HeII];
    IJth(jmatrix, 97, 90) = 0.0 - k[653]*y[IDX_HeII] - k[654]*y[IDX_HeII];
    IJth(jmatrix, 97, 92) = 0.0 - k[693]*y[IDX_HeII];
    IJth(jmatrix, 97, 94) = 0.0 - k[663]*y[IDX_HeII] - k[664]*y[IDX_HeII];
    IJth(jmatrix, 97, 95) = 0.0 - k[142]*y[IDX_HeII] - k[670]*y[IDX_HeII] -
                        k[671]*y[IDX_HeII] - k[672]*y[IDX_HeII];
    IJth(jmatrix, 97, 96) = 0.0 - k[680]*y[IDX_HeII] - k[681]*y[IDX_HeII] -
                        k[682]*y[IDX_HeII];
    IJth(jmatrix, 97, 97) = 0.0 - k[115]*y[IDX_H2I] - k[130]*y[IDX_HI] -
                        k[139]*y[IDX_CI] - k[140]*y[IDX_CH4I] -
                        k[141]*y[IDX_CHI] - k[142]*y[IDX_H2COI] -
                        k[143]*y[IDX_H2OI] - k[144]*y[IDX_N2I] -
                        k[145]*y[IDX_NH3I] - k[146]*y[IDX_O2I] -
                        k[147]*y[IDX_SiI] - k[536]*y[IDX_H2I] -
                        k[653]*y[IDX_CH2I] - k[654]*y[IDX_CH2I] -
                        k[655]*y[IDX_CH3I] - k[656]*y[IDX_CH3OHI] -
                        k[657]*y[IDX_CH3OHI] - k[658]*y[IDX_CH4I] -
                        k[659]*y[IDX_CH4I] - k[660]*y[IDX_CH4I] -
                        k[661]*y[IDX_CH4I] - k[662]*y[IDX_CHI] -
                        k[663]*y[IDX_CNI] - k[664]*y[IDX_CNI] -
                        k[665]*y[IDX_CO2I] - k[666]*y[IDX_CO2I] -
                        k[667]*y[IDX_CO2I] - k[668]*y[IDX_CO2I] -
                        k[669]*y[IDX_COI] - k[670]*y[IDX_H2COI] -
                        k[671]*y[IDX_H2COI] - k[672]*y[IDX_H2COI] -
                        k[673]*y[IDX_H2OI] - k[674]*y[IDX_H2OI] -
                        k[675]*y[IDX_H2SiOI] - k[676]*y[IDX_HCNI] -
                        k[677]*y[IDX_HCNI] - k[678]*y[IDX_HCNI] -
                        k[679]*y[IDX_HCNI] - k[680]*y[IDX_HCOI] -
                        k[681]*y[IDX_HCOI] - k[682]*y[IDX_HCOI] -
                        k[683]*y[IDX_HNCI] - k[684]*y[IDX_HNCI] -
                        k[685]*y[IDX_HNCI] - k[686]*y[IDX_HNOI] -
                        k[687]*y[IDX_HNOI] - k[688]*y[IDX_N2I] -
                        k[689]*y[IDX_NH2I] - k[690]*y[IDX_NH2I] -
                        k[691]*y[IDX_NH3I] - k[692]*y[IDX_NH3I] -
                        k[693]*y[IDX_NHI] - k[694]*y[IDX_NOI] -
                        k[695]*y[IDX_NOI] - k[696]*y[IDX_O2I] -
                        k[697]*y[IDX_OCNI] - k[698]*y[IDX_OCNI] -
                        k[699]*y[IDX_OHI] - k[700]*y[IDX_SiC3I] -
                        k[701]*y[IDX_SiCI] - k[702]*y[IDX_SiCI] -
                        k[703]*y[IDX_SiH2I] - k[704]*y[IDX_SiH2I] -
                        k[705]*y[IDX_SiH3I] - k[706]*y[IDX_SiH3I] -
                        k[707]*y[IDX_SiH4I] - k[708]*y[IDX_SiH4I] -
                        k[709]*y[IDX_SiHI] - k[710]*y[IDX_SiOI] -
                        k[711]*y[IDX_SiOI] - k[1218]*y[IDX_EM];
    IJth(jmatrix, 97, 98) = 0.0 - k[141]*y[IDX_HeII] - k[662]*y[IDX_HeII];
    IJth(jmatrix, 97, 100) = 0.0 + k[237] + k[265];
    IJth(jmatrix, 97, 101) = 0.0 - k[694]*y[IDX_HeII] - k[695]*y[IDX_HeII];
    IJth(jmatrix, 97, 103) = 0.0 - k[699]*y[IDX_HeII];
    IJth(jmatrix, 97, 104) = 0.0 - k[146]*y[IDX_HeII] - k[696]*y[IDX_HeII];
    IJth(jmatrix, 97, 105) = 0.0 - k[139]*y[IDX_HeII];
    IJth(jmatrix, 97, 108) = 0.0 - k[143]*y[IDX_HeII] - k[673]*y[IDX_HeII] -
                        k[674]*y[IDX_HeII];
    IJth(jmatrix, 97, 110) = 0.0 - k[1218]*y[IDX_HeII];
    IJth(jmatrix, 97, 111) = 0.0 - k[669]*y[IDX_HeII];
    IJth(jmatrix, 97, 112) = 0.0 - k[115]*y[IDX_HeII] - k[536]*y[IDX_HeII];
    IJth(jmatrix, 97, 113) = 0.0 - k[130]*y[IDX_HeII];
    IJth(jmatrix, 98, 37) = 0.0 - k[937]*y[IDX_CHI] - k[938]*y[IDX_CHI];
    IJth(jmatrix, 98, 45) = 0.0 - k[477]*y[IDX_CHI];
    IJth(jmatrix, 98, 49) = 0.0 - k[478]*y[IDX_CHI];
    IJth(jmatrix, 98, 51) = 0.0 - k[926]*y[IDX_CHI];
    IJth(jmatrix, 98, 52) = 0.0 + k[365]*y[IDX_CII];
    IJth(jmatrix, 98, 53) = 0.0 + k[32]*y[IDX_CHII];
    IJth(jmatrix, 98, 57) = 0.0 - k[53]*y[IDX_CHI];
    IJth(jmatrix, 98, 58) = 0.0 - k[464]*y[IDX_CHI] - k[465]*y[IDX_CHI];
    IJth(jmatrix, 98, 59) = 0.0 - k[469]*y[IDX_CHI];
    IJth(jmatrix, 98, 60) = 0.0 - k[474]*y[IDX_CHI];
    IJth(jmatrix, 98, 61) = 0.0 - k[476]*y[IDX_CHI];
    IJth(jmatrix, 98, 62) = 0.0 + k[35]*y[IDX_CHII];
    IJth(jmatrix, 98, 64) = 0.0 - k[467]*y[IDX_CHI];
    IJth(jmatrix, 98, 65) = 0.0 - k[58]*y[IDX_CHI];
    IJth(jmatrix, 98, 66) = 0.0 + k[318]*y[IDX_EM] - k[461]*y[IDX_CHI];
    IJth(jmatrix, 98, 67) = 0.0 + k[1127];
    IJth(jmatrix, 98, 68) = 0.0 - k[54]*y[IDX_CHI] + k[422]*y[IDX_CH2I] -
                        k[458]*y[IDX_CHI];
    IJth(jmatrix, 98, 69) = 0.0 - k[102]*y[IDX_CHI] - k[512]*y[IDX_CHI];
    IJth(jmatrix, 98, 70) = 0.0 + k[33]*y[IDX_CHII];
    IJth(jmatrix, 98, 71) = 0.0 + k[246] + k[1118];
    IJth(jmatrix, 98, 72) = 0.0 - k[923]*y[IDX_CHI];
    IJth(jmatrix, 98, 73) = 0.0 - k[57]*y[IDX_CHI] - k[468]*y[IDX_CHI];
    IJth(jmatrix, 98, 74) = 0.0 - k[60]*y[IDX_CHI] - k[472]*y[IDX_CHI] +
                        k[815]*y[IDX_HCNI];
    IJth(jmatrix, 98, 75) = 0.0 - k[463]*y[IDX_CHI];
    IJth(jmatrix, 98, 76) = 0.0 - k[59]*y[IDX_CHI] - k[471]*y[IDX_CHI];
    IJth(jmatrix, 98, 77) = 0.0 - k[470]*y[IDX_CHI];
    IJth(jmatrix, 98, 78) = 0.0 - k[61]*y[IDX_CHI] - k[473]*y[IDX_CHI];
    IJth(jmatrix, 98, 79) = 0.0 + k[299]*y[IDX_EM] + k[300]*y[IDX_EM];
    IJth(jmatrix, 98, 80) = 0.0 + k[868]*y[IDX_CI];
    IJth(jmatrix, 98, 81) = 0.0 + k[297]*y[IDX_EM] + k[1111];
    IJth(jmatrix, 98, 82) = 0.0 - k[56]*y[IDX_CHI] - k[460]*y[IDX_CHI];
    IJth(jmatrix, 98, 85) = 0.0 - k[462]*y[IDX_CHI];
    IJth(jmatrix, 98, 86) = 0.0 - k[927]*y[IDX_CHI];
    IJth(jmatrix, 98, 87) = 0.0 - k[15]*y[IDX_CHI] + k[365]*y[IDX_CH3OHI] +
                        k[369]*y[IDX_H2COI];
    IJth(jmatrix, 98, 88) = 0.0 + k[677]*y[IDX_HeII] + k[815]*y[IDX_OII];
    IJth(jmatrix, 98, 89) = 0.0 + k[31]*y[IDX_HCOI] + k[32]*y[IDX_MgI] +
                        k[33]*y[IDX_NH3I] + k[34]*y[IDX_NOI] + k[35]*y[IDX_SiI];
    IJth(jmatrix, 98, 90) = 0.0 + k[243] + k[422]*y[IDX_COII] + k[863]*y[IDX_CI] +
                        k[863]*y[IDX_CI] + k[878]*y[IDX_CH2I] +
                        k[878]*y[IDX_CH2I] + k[880]*y[IDX_CNI] +
                        k[897]*y[IDX_OI] + k[899]*y[IDX_OHI] + k[967]*y[IDX_HI]
                        + k[1007]*y[IDX_NI] + k[1113];
    IJth(jmatrix, 98, 91) = 0.0 - k[55]*y[IDX_CHI] - k[459]*y[IDX_CHI];
    IJth(jmatrix, 98, 92) = 0.0 + k[870]*y[IDX_CI];
    IJth(jmatrix, 98, 93) = 0.0 - k[62]*y[IDX_CHI] - k[475]*y[IDX_CHI];
    IJth(jmatrix, 98, 94) = 0.0 + k[880]*y[IDX_CH2I];
    IJth(jmatrix, 98, 95) = 0.0 + k[369]*y[IDX_CII] - k[924]*y[IDX_CHI];
    IJth(jmatrix, 98, 96) = 0.0 + k[31]*y[IDX_CHII] + k[864]*y[IDX_CI] -
                        k[925]*y[IDX_CHI];
    IJth(jmatrix, 98, 97) = 0.0 - k[141]*y[IDX_CHI] - k[662]*y[IDX_CHI] +
                        k[677]*y[IDX_HCNI];
    IJth(jmatrix, 98, 98) = 0.0 - k[0]*y[IDX_OI] - k[2]*y[IDX_H2I] - k[9]*y[IDX_HI]
                        - k[15]*y[IDX_CII] - k[53]*y[IDX_CNII] -
                        k[54]*y[IDX_COII] - k[55]*y[IDX_H2COII] -
                        k[56]*y[IDX_H2OII] - k[57]*y[IDX_NII] -
                        k[58]*y[IDX_N2II] - k[59]*y[IDX_NH2II] -
                        k[60]*y[IDX_OII] - k[61]*y[IDX_O2II] - k[62]*y[IDX_OHII]
                        - k[78]*y[IDX_HII] - k[102]*y[IDX_H2II] -
                        k[141]*y[IDX_HeII] - k[250] - k[458]*y[IDX_COII] -
                        k[459]*y[IDX_H2COII] - k[460]*y[IDX_H2OII] -
                        k[461]*y[IDX_H3COII] - k[462]*y[IDX_H3OII] -
                        k[463]*y[IDX_HCNII] - k[464]*y[IDX_HCNHII] -
                        k[465]*y[IDX_HCNHII] - k[466]*y[IDX_HCOII] -
                        k[467]*y[IDX_HNOII] - k[468]*y[IDX_NII] -
                        k[469]*y[IDX_N2HII] - k[470]*y[IDX_NHII] -
                        k[471]*y[IDX_NH2II] - k[472]*y[IDX_OII] -
                        k[473]*y[IDX_O2II] - k[474]*y[IDX_O2HII] -
                        k[475]*y[IDX_OHII] - k[476]*y[IDX_SiII] -
                        k[477]*y[IDX_SiHII] - k[478]*y[IDX_SiOII] -
                        k[512]*y[IDX_H2II] - k[580]*y[IDX_H3II] -
                        k[662]*y[IDX_HeII] - k[923]*y[IDX_CO2I] -
                        k[924]*y[IDX_H2COI] - k[925]*y[IDX_HCOI] -
                        k[926]*y[IDX_HNOI] - k[927]*y[IDX_N2I] -
                        k[928]*y[IDX_NI] - k[929]*y[IDX_NI] - k[930]*y[IDX_NOI]
                        - k[931]*y[IDX_NOI] - k[932]*y[IDX_NOI] -
                        k[933]*y[IDX_O2I] - k[934]*y[IDX_O2I] -
                        k[935]*y[IDX_O2I] - k[936]*y[IDX_O2I] -
                        k[937]*y[IDX_O2HI] - k[938]*y[IDX_O2HI] -
                        k[939]*y[IDX_OI] - k[940]*y[IDX_OI] - k[941]*y[IDX_OHI]
                        - k[958]*y[IDX_H2I] - k[970]*y[IDX_HI] - k[1128] -
                        k[1129] - k[1201]*y[IDX_H2I] - k[1253];
    IJth(jmatrix, 98, 99) = 0.0 - k[580]*y[IDX_CHI];
    IJth(jmatrix, 98, 101) = 0.0 + k[34]*y[IDX_CHII] - k[930]*y[IDX_CHI] -
                        k[931]*y[IDX_CHI] - k[932]*y[IDX_CHI];
    IJth(jmatrix, 98, 102) = 0.0 - k[928]*y[IDX_CHI] - k[929]*y[IDX_CHI] +
                        k[1007]*y[IDX_CH2I];
    IJth(jmatrix, 98, 103) = 0.0 + k[876]*y[IDX_CI] + k[899]*y[IDX_CH2I] -
                        k[941]*y[IDX_CHI];
    IJth(jmatrix, 98, 104) = 0.0 - k[933]*y[IDX_CHI] - k[934]*y[IDX_CHI] -
                        k[935]*y[IDX_CHI] - k[936]*y[IDX_CHI];
    IJth(jmatrix, 98, 105) = 0.0 + k[863]*y[IDX_CH2I] + k[863]*y[IDX_CH2I] +
                        k[864]*y[IDX_HCOI] + k[868]*y[IDX_NH2I] +
                        k[870]*y[IDX_NHI] + k[876]*y[IDX_OHI] +
                        k[955]*y[IDX_H2I] + k[1206]*y[IDX_HI];
    IJth(jmatrix, 98, 106) = 0.0 - k[78]*y[IDX_CHI];
    IJth(jmatrix, 98, 107) = 0.0 - k[466]*y[IDX_CHI];
    IJth(jmatrix, 98, 109) = 0.0 - k[0]*y[IDX_CHI] + k[897]*y[IDX_CH2I] -
                        k[939]*y[IDX_CHI] - k[940]*y[IDX_CHI];
    IJth(jmatrix, 98, 110) = 0.0 + k[297]*y[IDX_CH2II] + k[299]*y[IDX_CH3II] +
                        k[300]*y[IDX_CH3II] + k[318]*y[IDX_H3COII];
    IJth(jmatrix, 98, 112) = 0.0 - k[2]*y[IDX_CHI] + k[955]*y[IDX_CI] -
                        k[958]*y[IDX_CHI] - k[1201]*y[IDX_CHI];
    IJth(jmatrix, 98, 113) = 0.0 - k[9]*y[IDX_CHI] + k[967]*y[IDX_CH2I] -
                        k[970]*y[IDX_CHI] + k[1206]*y[IDX_CI];
    IJth(jmatrix, 99, 27) = 0.0 + k[537]*y[IDX_H2I];
    IJth(jmatrix, 99, 39) = 0.0 - k[595]*y[IDX_H3II];
    IJth(jmatrix, 99, 43) = 0.0 - k[602]*y[IDX_H3II];
    IJth(jmatrix, 99, 46) = 0.0 - k[604]*y[IDX_H3II];
    IJth(jmatrix, 99, 47) = 0.0 - k[605]*y[IDX_H3II];
    IJth(jmatrix, 99, 48) = 0.0 - k[603]*y[IDX_H3II];
    IJth(jmatrix, 99, 51) = 0.0 - k[590]*y[IDX_H3II];
    IJth(jmatrix, 99, 52) = 0.0 - k[579]*y[IDX_H3II];
    IJth(jmatrix, 99, 53) = 0.0 - k[591]*y[IDX_H3II];
    IJth(jmatrix, 99, 56) = 0.0 - k[606]*y[IDX_H3II];
    IJth(jmatrix, 99, 60) = 0.0 + k[544]*y[IDX_H2I];
    IJth(jmatrix, 99, 62) = 0.0 - k[601]*y[IDX_H3II];
    IJth(jmatrix, 99, 63) = 0.0 - k[589]*y[IDX_H3II];
    IJth(jmatrix, 99, 69) = 0.0 + k[516]*y[IDX_H2I] + k[519]*y[IDX_HCOI];
    IJth(jmatrix, 99, 71) = 0.0 - k[578]*y[IDX_H3II];
    IJth(jmatrix, 99, 72) = 0.0 - k[582]*y[IDX_H3II];
    IJth(jmatrix, 99, 77) = 0.0 + k[540]*y[IDX_H2I];
    IJth(jmatrix, 99, 80) = 0.0 - k[593]*y[IDX_H3II];
    IJth(jmatrix, 99, 86) = 0.0 - k[592]*y[IDX_H3II];
    IJth(jmatrix, 99, 88) = 0.0 - k[587]*y[IDX_H3II];
    IJth(jmatrix, 99, 90) = 0.0 - k[577]*y[IDX_H3II];
    IJth(jmatrix, 99, 92) = 0.0 - k[594]*y[IDX_H3II];
    IJth(jmatrix, 99, 94) = 0.0 - k[581]*y[IDX_H3II];
    IJth(jmatrix, 99, 95) = 0.0 - k[585]*y[IDX_H3II];
    IJth(jmatrix, 99, 96) = 0.0 + k[519]*y[IDX_H2II] - k[588]*y[IDX_H3II];
    IJth(jmatrix, 99, 98) = 0.0 - k[580]*y[IDX_H3II];
    IJth(jmatrix, 99, 99) = 0.0 - k[315]*y[IDX_EM] - k[316]*y[IDX_EM] -
                        k[576]*y[IDX_CI] - k[577]*y[IDX_CH2I] -
                        k[578]*y[IDX_CH3I] - k[579]*y[IDX_CH3OHI] -
                        k[580]*y[IDX_CHI] - k[581]*y[IDX_CNI] -
                        k[582]*y[IDX_CO2I] - k[583]*y[IDX_COI] -
                        k[584]*y[IDX_COI] - k[585]*y[IDX_H2COI] -
                        k[586]*y[IDX_H2OI] - k[587]*y[IDX_HCNI] -
                        k[588]*y[IDX_HCOI] - k[589]*y[IDX_HNCI] -
                        k[590]*y[IDX_HNOI] - k[591]*y[IDX_MgI] -
                        k[592]*y[IDX_N2I] - k[593]*y[IDX_NH2I] -
                        k[594]*y[IDX_NHI] - k[595]*y[IDX_NO2I] -
                        k[596]*y[IDX_NOI] - k[597]*y[IDX_O2I] - k[598]*y[IDX_OI]
                        - k[599]*y[IDX_OI] - k[600]*y[IDX_OHI] -
                        k[601]*y[IDX_SiI] - k[602]*y[IDX_SiH2I] -
                        k[603]*y[IDX_SiH3I] - k[604]*y[IDX_SiH4I] -
                        k[605]*y[IDX_SiHI] - k[606]*y[IDX_SiOI] - k[1145] -
                        k[1146];
    IJth(jmatrix, 99, 101) = 0.0 - k[596]*y[IDX_H3II];
    IJth(jmatrix, 99, 103) = 0.0 - k[600]*y[IDX_H3II];
    IJth(jmatrix, 99, 104) = 0.0 - k[597]*y[IDX_H3II];
    IJth(jmatrix, 99, 105) = 0.0 - k[576]*y[IDX_H3II];
    IJth(jmatrix, 99, 108) = 0.0 - k[586]*y[IDX_H3II];
    IJth(jmatrix, 99, 109) = 0.0 - k[598]*y[IDX_H3II] - k[599]*y[IDX_H3II];
    IJth(jmatrix, 99, 110) = 0.0 - k[315]*y[IDX_H3II] - k[316]*y[IDX_H3II];
    IJth(jmatrix, 99, 111) = 0.0 - k[583]*y[IDX_H3II] - k[584]*y[IDX_H3II];
    IJth(jmatrix, 99, 112) = 0.0 + k[516]*y[IDX_H2II] + k[537]*y[IDX_HeHII] +
                        k[540]*y[IDX_NHII] + k[544]*y[IDX_O2HII];
    IJth(jmatrix, 100, 26) = 0.0 + k[675]*y[IDX_HeII];
    IJth(jmatrix, 100, 27) = 0.0 + k[336]*y[IDX_EM] + k[537]*y[IDX_H2I] +
                        k[618]*y[IDX_HI];
    IJth(jmatrix, 100, 33) = 0.0 + k[700]*y[IDX_HeII];
    IJth(jmatrix, 100, 38) = 0.0 + k[701]*y[IDX_HeII] + k[702]*y[IDX_HeII];
    IJth(jmatrix, 100, 42) = 0.0 + k[697]*y[IDX_HeII] + k[698]*y[IDX_HeII];
    IJth(jmatrix, 100, 43) = 0.0 + k[703]*y[IDX_HeII] + k[704]*y[IDX_HeII];
    IJth(jmatrix, 100, 46) = 0.0 + k[707]*y[IDX_HeII] + k[708]*y[IDX_HeII];
    IJth(jmatrix, 100, 47) = 0.0 + k[709]*y[IDX_HeII];
    IJth(jmatrix, 100, 48) = 0.0 + k[705]*y[IDX_HeII] + k[706]*y[IDX_HeII];
    IJth(jmatrix, 100, 51) = 0.0 + k[686]*y[IDX_HeII] + k[687]*y[IDX_HeII];
    IJth(jmatrix, 100, 52) = 0.0 + k[656]*y[IDX_HeII] + k[657]*y[IDX_HeII];
    IJth(jmatrix, 100, 56) = 0.0 + k[710]*y[IDX_HeII] + k[711]*y[IDX_HeII];
    IJth(jmatrix, 100, 62) = 0.0 + k[147]*y[IDX_HeII];
    IJth(jmatrix, 100, 63) = 0.0 + k[683]*y[IDX_HeII] + k[684]*y[IDX_HeII] +
                        k[685]*y[IDX_HeII];
    IJth(jmatrix, 100, 67) = 0.0 + k[140]*y[IDX_HeII] + k[658]*y[IDX_HeII] +
                        k[659]*y[IDX_HeII] + k[660]*y[IDX_HeII] +
                        k[661]*y[IDX_HeII];
    IJth(jmatrix, 100, 69) = 0.0 - k[520]*y[IDX_HeI];
    IJth(jmatrix, 100, 70) = 0.0 + k[145]*y[IDX_HeII] + k[691]*y[IDX_HeII] +
                        k[692]*y[IDX_HeII];
    IJth(jmatrix, 100, 71) = 0.0 + k[655]*y[IDX_HeII];
    IJth(jmatrix, 100, 72) = 0.0 + k[665]*y[IDX_HeII] + k[666]*y[IDX_HeII] +
                        k[667]*y[IDX_HeII] + k[668]*y[IDX_HeII];
    IJth(jmatrix, 100, 80) = 0.0 + k[689]*y[IDX_HeII] + k[690]*y[IDX_HeII];
    IJth(jmatrix, 100, 86) = 0.0 + k[144]*y[IDX_HeII] + k[688]*y[IDX_HeII];
    IJth(jmatrix, 100, 88) = 0.0 + k[676]*y[IDX_HeII] + k[677]*y[IDX_HeII] +
                        k[678]*y[IDX_HeII] + k[679]*y[IDX_HeII];
    IJth(jmatrix, 100, 90) = 0.0 + k[653]*y[IDX_HeII] + k[654]*y[IDX_HeII];
    IJth(jmatrix, 100, 92) = 0.0 + k[693]*y[IDX_HeII];
    IJth(jmatrix, 100, 94) = 0.0 + k[663]*y[IDX_HeII] + k[664]*y[IDX_HeII];
    IJth(jmatrix, 100, 95) = 0.0 + k[142]*y[IDX_HeII] + k[670]*y[IDX_HeII] +
                        k[671]*y[IDX_HeII] + k[672]*y[IDX_HeII];
    IJth(jmatrix, 100, 96) = 0.0 + k[680]*y[IDX_HeII] + k[682]*y[IDX_HeII];
    IJth(jmatrix, 100, 97) = 0.0 + k[115]*y[IDX_H2I] + k[130]*y[IDX_HI] +
                        k[139]*y[IDX_CI] + k[140]*y[IDX_CH4I] +
                        k[141]*y[IDX_CHI] + k[142]*y[IDX_H2COI] +
                        k[143]*y[IDX_H2OI] + k[144]*y[IDX_N2I] +
                        k[145]*y[IDX_NH3I] + k[146]*y[IDX_O2I] +
                        k[147]*y[IDX_SiI] + k[536]*y[IDX_H2I] +
                        k[653]*y[IDX_CH2I] + k[654]*y[IDX_CH2I] +
                        k[655]*y[IDX_CH3I] + k[656]*y[IDX_CH3OHI] +
                        k[657]*y[IDX_CH3OHI] + k[658]*y[IDX_CH4I] +
                        k[659]*y[IDX_CH4I] + k[660]*y[IDX_CH4I] +
                        k[661]*y[IDX_CH4I] + k[662]*y[IDX_CHI] +
                        k[663]*y[IDX_CNI] + k[664]*y[IDX_CNI] +
                        k[665]*y[IDX_CO2I] + k[666]*y[IDX_CO2I] +
                        k[667]*y[IDX_CO2I] + k[668]*y[IDX_CO2I] +
                        k[669]*y[IDX_COI] + k[670]*y[IDX_H2COI] +
                        k[671]*y[IDX_H2COI] + k[672]*y[IDX_H2COI] +
                        k[673]*y[IDX_H2OI] + k[674]*y[IDX_H2OI] +
                        k[675]*y[IDX_H2SiOI] + k[676]*y[IDX_HCNI] +
                        k[677]*y[IDX_HCNI] + k[678]*y[IDX_HCNI] +
                        k[679]*y[IDX_HCNI] + k[680]*y[IDX_HCOI] +
                        k[682]*y[IDX_HCOI] + k[683]*y[IDX_HNCI] +
                        k[684]*y[IDX_HNCI] + k[685]*y[IDX_HNCI] +
                        k[686]*y[IDX_HNOI] + k[687]*y[IDX_HNOI] +
                        k[688]*y[IDX_N2I] + k[689]*y[IDX_NH2I] +
                        k[690]*y[IDX_NH2I] + k[691]*y[IDX_NH3I] +
                        k[692]*y[IDX_NH3I] + k[693]*y[IDX_NHI] +
                        k[694]*y[IDX_NOI] + k[695]*y[IDX_NOI] +
                        k[696]*y[IDX_O2I] + k[697]*y[IDX_OCNI] +
                        k[698]*y[IDX_OCNI] + k[699]*y[IDX_OHI] +
                        k[700]*y[IDX_SiC3I] + k[701]*y[IDX_SiCI] +
                        k[702]*y[IDX_SiCI] + k[703]*y[IDX_SiH2I] +
                        k[704]*y[IDX_SiH2I] + k[705]*y[IDX_SiH3I] +
                        k[706]*y[IDX_SiH3I] + k[707]*y[IDX_SiH4I] +
                        k[708]*y[IDX_SiH4I] + k[709]*y[IDX_SiHI] +
                        k[710]*y[IDX_SiOI] + k[711]*y[IDX_SiOI] +
                        k[1218]*y[IDX_EM];
    IJth(jmatrix, 100, 98) = 0.0 + k[141]*y[IDX_HeII] + k[662]*y[IDX_HeII];
    IJth(jmatrix, 100, 100) = 0.0 - k[237] - k[265] - k[520]*y[IDX_H2II] -
                        k[1198]*y[IDX_HII];
    IJth(jmatrix, 100, 101) = 0.0 + k[694]*y[IDX_HeII] + k[695]*y[IDX_HeII];
    IJth(jmatrix, 100, 103) = 0.0 + k[699]*y[IDX_HeII];
    IJth(jmatrix, 100, 104) = 0.0 + k[146]*y[IDX_HeII] + k[696]*y[IDX_HeII];
    IJth(jmatrix, 100, 105) = 0.0 + k[139]*y[IDX_HeII];
    IJth(jmatrix, 100, 106) = 0.0 - k[1198]*y[IDX_HeI];
    IJth(jmatrix, 100, 108) = 0.0 + k[143]*y[IDX_HeII] + k[673]*y[IDX_HeII] +
                        k[674]*y[IDX_HeII];
    IJth(jmatrix, 100, 110) = 0.0 + k[336]*y[IDX_HeHII] + k[1218]*y[IDX_HeII];
    IJth(jmatrix, 100, 111) = 0.0 + k[669]*y[IDX_HeII];
    IJth(jmatrix, 100, 112) = 0.0 + k[115]*y[IDX_HeII] + k[536]*y[IDX_HeII] +
                        k[537]*y[IDX_HeHII];
    IJth(jmatrix, 100, 113) = 0.0 + k[130]*y[IDX_HeII] + k[618]*y[IDX_HeHII];
    IJth(jmatrix, 101, 7) = 0.0 + k[1343] + k[1344] + k[1345] + k[1346];
    IJth(jmatrix, 101, 25) = 0.0 + k[311]*y[IDX_EM];
    IJth(jmatrix, 101, 39) = 0.0 + k[276] + k[885]*y[IDX_CH2I] + k[945]*y[IDX_CNI] +
                        k[952]*y[IDX_COI] + k[986]*y[IDX_HI] + k[1020]*y[IDX_NI]
                        + k[1020]*y[IDX_NI] + k[1041]*y[IDX_NHI] +
                        k[1075]*y[IDX_OI] + k[1164];
    IJth(jmatrix, 101, 42) = 0.0 - k[1053]*y[IDX_NOI] + k[1054]*y[IDX_O2I] +
                        k[1078]*y[IDX_OI];
    IJth(jmatrix, 101, 49) = 0.0 - k[206]*y[IDX_NOI] + k[746]*y[IDX_NI];
    IJth(jmatrix, 101, 51) = 0.0 + k[264] + k[687]*y[IDX_HeII] + k[883]*y[IDX_CH2I] +
                        k[906]*y[IDX_CH3I] + k[926]*y[IDX_CHI] +
                        k[944]*y[IDX_CNI] + k[981]*y[IDX_HI] +
                        k[999]*y[IDX_HCOI] + k[1017]*y[IDX_NI] +
                        k[1069]*y[IDX_OI] + k[1097]*y[IDX_OHI] + k[1153];
    IJth(jmatrix, 101, 52) = 0.0 + k[715]*y[IDX_NII];
    IJth(jmatrix, 101, 53) = 0.0 + k[151]*y[IDX_NOII];
    IJth(jmatrix, 101, 57) = 0.0 - k[67]*y[IDX_NOI];
    IJth(jmatrix, 101, 60) = 0.0 - k[807]*y[IDX_NOI];
    IJth(jmatrix, 101, 62) = 0.0 + k[229]*y[IDX_NOII] - k[1105]*y[IDX_NOI];
    IJth(jmatrix, 101, 63) = 0.0 + k[649]*y[IDX_HNOII];
    IJth(jmatrix, 101, 64) = 0.0 - k[204]*y[IDX_NOI] + k[334]*y[IDX_EM] +
                        k[388]*y[IDX_CI] + k[430]*y[IDX_CH2I] +
                        k[467]*y[IDX_CHI] + k[482]*y[IDX_CNI] +
                        k[486]*y[IDX_COI] + k[550]*y[IDX_H2COI] +
                        k[568]*y[IDX_H2OI] + k[630]*y[IDX_HCNI] +
                        k[642]*y[IDX_HCOI] + k[649]*y[IDX_HNCI] +
                        k[652]*y[IDX_CO2I] + k[732]*y[IDX_N2I] +
                        k[788]*y[IDX_NH2I] + k[800]*y[IDX_NHI] +
                        k[856]*y[IDX_OHI];
    IJth(jmatrix, 101, 65) = 0.0 - k[172]*y[IDX_NOI];
    IJth(jmatrix, 101, 68) = 0.0 - k[72]*y[IDX_NOI];
    IJth(jmatrix, 101, 69) = 0.0 - k[112]*y[IDX_NOI] - k[524]*y[IDX_NOI];
    IJth(jmatrix, 101, 71) = 0.0 + k[906]*y[IDX_HNOI] - k[910]*y[IDX_NOI];
    IJth(jmatrix, 101, 72) = 0.0 + k[652]*y[IDX_HNOII] + k[719]*y[IDX_NII] +
                        k[1012]*y[IDX_NI];
    IJth(jmatrix, 101, 73) = 0.0 - k[167]*y[IDX_NOI] + k[715]*y[IDX_CH3OHI] +
                        k[719]*y[IDX_CO2I] - k[727]*y[IDX_NOI] +
                        k[729]*y[IDX_O2I];
    IJth(jmatrix, 101, 75) = 0.0 - k[132]*y[IDX_NOI];
    IJth(jmatrix, 101, 76) = 0.0 - k[182]*y[IDX_NOI];
    IJth(jmatrix, 101, 77) = 0.0 - k[178]*y[IDX_NOI] - k[764]*y[IDX_NOI];
    IJth(jmatrix, 101, 78) = 0.0 - k[205]*y[IDX_NOI];
    IJth(jmatrix, 101, 79) = 0.0 - k[48]*y[IDX_NOI];
    IJth(jmatrix, 101, 80) = 0.0 + k[788]*y[IDX_HNOII] - k[1029]*y[IDX_NOI] -
                        k[1030]*y[IDX_NOI];
    IJth(jmatrix, 101, 81) = 0.0 - k[36]*y[IDX_NOI];
    IJth(jmatrix, 101, 82) = 0.0 - k[120]*y[IDX_NOI];
    IJth(jmatrix, 101, 83) = 0.0 - k[191]*y[IDX_NOI];
    IJth(jmatrix, 101, 84) = 0.0 + k[151]*y[IDX_MgI] + k[229]*y[IDX_SiI];
    IJth(jmatrix, 101, 86) = 0.0 + k[732]*y[IDX_HNOII] + k[1071]*y[IDX_OI];
    IJth(jmatrix, 101, 87) = 0.0 - k[20]*y[IDX_NOI];
    IJth(jmatrix, 101, 88) = 0.0 + k[630]*y[IDX_HNOII];
    IJth(jmatrix, 101, 89) = 0.0 - k[34]*y[IDX_NOI];
    IJth(jmatrix, 101, 90) = 0.0 + k[430]*y[IDX_HNOII] + k[883]*y[IDX_HNOI] +
                        k[885]*y[IDX_NO2I] - k[886]*y[IDX_NOI] -
                        k[887]*y[IDX_NOI] - k[888]*y[IDX_NOI];
    IJth(jmatrix, 101, 91) = 0.0 - k[203]*y[IDX_NOI];
    IJth(jmatrix, 101, 92) = 0.0 + k[800]*y[IDX_HNOII] + k[1041]*y[IDX_NO2I] -
                        k[1042]*y[IDX_NOI] - k[1043]*y[IDX_NOI] +
                        k[1045]*y[IDX_O2I] + k[1046]*y[IDX_OI];
    IJth(jmatrix, 101, 93) = 0.0 - k[223]*y[IDX_NOI] - k[846]*y[IDX_NOI];
    IJth(jmatrix, 101, 94) = 0.0 + k[482]*y[IDX_HNOII] + k[944]*y[IDX_HNOI] +
                        k[945]*y[IDX_NO2I] - k[946]*y[IDX_NOI] -
                        k[947]*y[IDX_NOI] + k[948]*y[IDX_O2I] +
                        k[1058]*y[IDX_OI];
    IJth(jmatrix, 101, 95) = 0.0 + k[550]*y[IDX_HNOII];
    IJth(jmatrix, 101, 96) = 0.0 + k[642]*y[IDX_HNOII] + k[999]*y[IDX_HNOI] -
                        k[1000]*y[IDX_NOI];
    IJth(jmatrix, 101, 97) = 0.0 + k[687]*y[IDX_HNOI] - k[694]*y[IDX_NOI] -
                        k[695]*y[IDX_NOI];
    IJth(jmatrix, 101, 98) = 0.0 + k[467]*y[IDX_HNOII] + k[926]*y[IDX_HNOI] -
                        k[930]*y[IDX_NOI] - k[931]*y[IDX_NOI] -
                        k[932]*y[IDX_NOI];
    IJth(jmatrix, 101, 99) = 0.0 - k[596]*y[IDX_NOI];
    IJth(jmatrix, 101, 101) = 0.0 - k[20]*y[IDX_CII] - k[34]*y[IDX_CHII] -
                        k[36]*y[IDX_CH2II] - k[48]*y[IDX_CH3II] -
                        k[67]*y[IDX_CNII] - k[72]*y[IDX_COII] - k[87]*y[IDX_HII]
                        - k[112]*y[IDX_H2II] - k[120]*y[IDX_H2OII] -
                        k[132]*y[IDX_HCNII] - k[167]*y[IDX_NII] -
                        k[172]*y[IDX_N2II] - k[178]*y[IDX_NHII] -
                        k[182]*y[IDX_NH2II] - k[191]*y[IDX_NH3II] -
                        k[203]*y[IDX_H2COII] - k[204]*y[IDX_HNOII] -
                        k[205]*y[IDX_O2II] - k[206]*y[IDX_SiOII] -
                        k[223]*y[IDX_OHII] - k[277] - k[278] -
                        k[524]*y[IDX_H2II] - k[596]*y[IDX_H3II] -
                        k[694]*y[IDX_HeII] - k[695]*y[IDX_HeII] -
                        k[727]*y[IDX_NII] - k[764]*y[IDX_NHII] -
                        k[807]*y[IDX_O2HII] - k[846]*y[IDX_OHII] -
                        k[871]*y[IDX_CI] - k[872]*y[IDX_CI] - k[886]*y[IDX_CH2I]
                        - k[887]*y[IDX_CH2I] - k[888]*y[IDX_CH2I] -
                        k[910]*y[IDX_CH3I] - k[930]*y[IDX_CHI] -
                        k[931]*y[IDX_CHI] - k[932]*y[IDX_CHI] -
                        k[946]*y[IDX_CNI] - k[947]*y[IDX_CNI] - k[987]*y[IDX_HI]
                        - k[988]*y[IDX_HI] - k[1000]*y[IDX_HCOI] -
                        k[1022]*y[IDX_NI] - k[1029]*y[IDX_NH2I] -
                        k[1030]*y[IDX_NH2I] - k[1042]*y[IDX_NHI] -
                        k[1043]*y[IDX_NHI] - k[1051]*y[IDX_NOI] -
                        k[1051]*y[IDX_NOI] - k[1051]*y[IDX_NOI] -
                        k[1051]*y[IDX_NOI] - k[1052]*y[IDX_O2I] -
                        k[1053]*y[IDX_OCNI] - k[1076]*y[IDX_OI] -
                        k[1099]*y[IDX_OHI] - k[1105]*y[IDX_SiI] - k[1165] -
                        k[1166] - k[1255];
    IJth(jmatrix, 101, 102) = 0.0 + k[746]*y[IDX_SiOII] + k[1012]*y[IDX_CO2I] +
                        k[1017]*y[IDX_HNOI] + k[1020]*y[IDX_NO2I] +
                        k[1020]*y[IDX_NO2I] - k[1022]*y[IDX_NOI] +
                        k[1023]*y[IDX_O2I] + k[1025]*y[IDX_OHI];
    IJth(jmatrix, 101, 103) = 0.0 + k[856]*y[IDX_HNOII] + k[1025]*y[IDX_NI] +
                        k[1097]*y[IDX_HNOI] - k[1099]*y[IDX_NOI];
    IJth(jmatrix, 101, 104) = 0.0 + k[729]*y[IDX_NII] + k[948]*y[IDX_CNI] +
                        k[1023]*y[IDX_NI] + k[1045]*y[IDX_NHI] -
                        k[1052]*y[IDX_NOI] + k[1054]*y[IDX_OCNI];
    IJth(jmatrix, 101, 105) = 0.0 + k[388]*y[IDX_HNOII] - k[871]*y[IDX_NOI] -
                        k[872]*y[IDX_NOI];
    IJth(jmatrix, 101, 106) = 0.0 - k[87]*y[IDX_NOI];
    IJth(jmatrix, 101, 108) = 0.0 + k[568]*y[IDX_HNOII];
    IJth(jmatrix, 101, 109) = 0.0 + k[1046]*y[IDX_NHI] + k[1058]*y[IDX_CNI] +
                        k[1069]*y[IDX_HNOI] + k[1071]*y[IDX_N2I] +
                        k[1075]*y[IDX_NO2I] - k[1076]*y[IDX_NOI] +
                        k[1078]*y[IDX_OCNI];
    IJth(jmatrix, 101, 110) = 0.0 + k[311]*y[IDX_H2NOII] + k[334]*y[IDX_HNOII];
    IJth(jmatrix, 101, 111) = 0.0 + k[486]*y[IDX_HNOII] + k[952]*y[IDX_NO2I];
    IJth(jmatrix, 101, 113) = 0.0 + k[981]*y[IDX_HNOI] + k[986]*y[IDX_NO2I] -
                        k[987]*y[IDX_NOI] - k[988]*y[IDX_NOI];
    IJth(jmatrix, 102, 23) = 0.0 - k[1013]*y[IDX_NI];
    IJth(jmatrix, 102, 36) = 0.0 - k[744]*y[IDX_NI];
    IJth(jmatrix, 102, 37) = 0.0 - k[1024]*y[IDX_NI];
    IJth(jmatrix, 102, 38) = 0.0 - k[1027]*y[IDX_NI];
    IJth(jmatrix, 102, 39) = 0.0 - k[1019]*y[IDX_NI] - k[1020]*y[IDX_NI] -
                        k[1021]*y[IDX_NI];
    IJth(jmatrix, 102, 49) = 0.0 - k[745]*y[IDX_NI] - k[746]*y[IDX_NI];
    IJth(jmatrix, 102, 51) = 0.0 - k[1017]*y[IDX_NI];
    IJth(jmatrix, 102, 53) = 0.0 + k[163]*y[IDX_NII];
    IJth(jmatrix, 102, 57) = 0.0 + k[303]*y[IDX_EM] - k[737]*y[IDX_NI];
    IJth(jmatrix, 102, 59) = 0.0 + k[339]*y[IDX_EM];
    IJth(jmatrix, 102, 62) = 0.0 + k[1105]*y[IDX_NOI];
    IJth(jmatrix, 102, 63) = 0.0 + k[684]*y[IDX_HeII] + k[760]*y[IDX_NHII];
    IJth(jmatrix, 102, 65) = 0.0 - k[174]*y[IDX_NI] + k[337]*y[IDX_EM] +
                        k[337]*y[IDX_EM] + k[825]*y[IDX_OI];
    IJth(jmatrix, 102, 67) = 0.0 + k[156]*y[IDX_NII] + k[716]*y[IDX_NII];
    IJth(jmatrix, 102, 68) = 0.0 + k[795]*y[IDX_NHI];
    IJth(jmatrix, 102, 69) = 0.0 - k[522]*y[IDX_NI];
    IJth(jmatrix, 102, 70) = 0.0 + k[165]*y[IDX_NII];
    IJth(jmatrix, 102, 71) = 0.0 - k[1008]*y[IDX_NI] - k[1009]*y[IDX_NI] -
                        k[1010]*y[IDX_NI];
    IJth(jmatrix, 102, 72) = 0.0 + k[748]*y[IDX_NHII] - k[1012]*y[IDX_NI];
    IJth(jmatrix, 102, 73) = 0.0 + k[57]*y[IDX_CHI] + k[155]*y[IDX_CH2I] +
                        k[156]*y[IDX_CH4I] + k[157]*y[IDX_CNI] +
                        k[158]*y[IDX_COI] + k[159]*y[IDX_H2COI] +
                        k[160]*y[IDX_H2OI] + k[161]*y[IDX_HCNI] +
                        k[162]*y[IDX_HCOI] + k[163]*y[IDX_MgI] +
                        k[164]*y[IDX_NH2I] + k[165]*y[IDX_NH3I] +
                        k[166]*y[IDX_NHI] + k[167]*y[IDX_NOI] +
                        k[168]*y[IDX_O2I] + k[169]*y[IDX_OHI] +
                        k[716]*y[IDX_CH4I] - k[1210]*y[IDX_NI] +
                        k[1220]*y[IDX_EM];
    IJth(jmatrix, 102, 74) = 0.0 + k[814]*y[IDX_HCNI] + k[817]*y[IDX_N2I];
    IJth(jmatrix, 102, 76) = 0.0 + k[341]*y[IDX_EM] - k[741]*y[IDX_NI] +
                        k[802]*y[IDX_NHI];
    IJth(jmatrix, 102, 77) = 0.0 + k[340]*y[IDX_EM] + k[390]*y[IDX_CI] +
                        k[432]*y[IDX_CH2I] + k[470]*y[IDX_CHI] +
                        k[540]*y[IDX_H2I] - k[740]*y[IDX_NI] + k[747]*y[IDX_CNI]
                        + k[748]*y[IDX_CO2I] + k[751]*y[IDX_COI] +
                        k[752]*y[IDX_H2COI] + k[754]*y[IDX_H2OI] +
                        k[758]*y[IDX_HCNI] + k[759]*y[IDX_HCOI] +
                        k[760]*y[IDX_HNCI] + k[761]*y[IDX_N2I] +
                        k[762]*y[IDX_NH2I] + k[763]*y[IDX_NHI] +
                        k[766]*y[IDX_O2I] + k[767]*y[IDX_OI] + k[768]*y[IDX_OHI]
                        + k[1156];
    IJth(jmatrix, 102, 78) = 0.0 - k[742]*y[IDX_NI];
    IJth(jmatrix, 102, 80) = 0.0 + k[164]*y[IDX_NII] + k[762]*y[IDX_NHII];
    IJth(jmatrix, 102, 81) = 0.0 - k[736]*y[IDX_NI];
    IJth(jmatrix, 102, 82) = 0.0 - k[738]*y[IDX_NI] - k[739]*y[IDX_NI] +
                        k[797]*y[IDX_NHI];
    IJth(jmatrix, 102, 84) = 0.0 + k[345]*y[IDX_EM];
    IJth(jmatrix, 102, 86) = 0.0 + k[267] + k[267] + k[688]*y[IDX_HeII] +
                        k[761]*y[IDX_NHII] + k[817]*y[IDX_OII] +
                        k[865]*y[IDX_CI] + k[927]*y[IDX_CHI] + k[1071]*y[IDX_OI]
                        + k[1155] + k[1155];
    IJth(jmatrix, 102, 87) = 0.0 - k[1192]*y[IDX_NI];
    IJth(jmatrix, 102, 88) = 0.0 + k[161]*y[IDX_NII] + k[678]*y[IDX_HeII] +
                        k[679]*y[IDX_HeII] + k[758]*y[IDX_NHII] +
                        k[814]*y[IDX_OII];
    IJth(jmatrix, 102, 89) = 0.0 - k[408]*y[IDX_NI];
    IJth(jmatrix, 102, 90) = 0.0 + k[155]*y[IDX_NII] + k[432]*y[IDX_NHII] +
                        k[886]*y[IDX_NOI] - k[1005]*y[IDX_NI] -
                        k[1006]*y[IDX_NI] - k[1007]*y[IDX_NI];
    IJth(jmatrix, 102, 91) = 0.0 + k[796]*y[IDX_NHI];
    IJth(jmatrix, 102, 92) = 0.0 + k[166]*y[IDX_NII] + k[274] + k[763]*y[IDX_NHII] +
                        k[795]*y[IDX_COII] + k[796]*y[IDX_H2COII] +
                        k[797]*y[IDX_H2OII] + k[802]*y[IDX_NH2II] +
                        k[870]*y[IDX_CI] + k[985]*y[IDX_HI] - k[1018]*y[IDX_NI]
                        + k[1035]*y[IDX_CNI] + k[1040]*y[IDX_NHI] +
                        k[1040]*y[IDX_NHI] + k[1047]*y[IDX_OI] +
                        k[1048]*y[IDX_OHI] + k[1162];
    IJth(jmatrix, 102, 93) = 0.0 - k[743]*y[IDX_NI];
    IJth(jmatrix, 102, 94) = 0.0 + k[157]*y[IDX_NII] + k[251] + k[664]*y[IDX_HeII] +
                        k[747]*y[IDX_NHII] + k[947]*y[IDX_NOI] -
                        k[1011]*y[IDX_NI] + k[1035]*y[IDX_NHI] +
                        k[1057]*y[IDX_OI] + k[1130];
    IJth(jmatrix, 102, 95) = 0.0 + k[159]*y[IDX_NII] + k[752]*y[IDX_NHII];
    IJth(jmatrix, 102, 96) = 0.0 + k[162]*y[IDX_NII] + k[759]*y[IDX_NHII] -
                        k[1014]*y[IDX_NI] - k[1015]*y[IDX_NI] -
                        k[1016]*y[IDX_NI];
    IJth(jmatrix, 102, 97) = 0.0 + k[664]*y[IDX_CNI] + k[678]*y[IDX_HCNI] +
                        k[679]*y[IDX_HCNI] + k[684]*y[IDX_HNCI] +
                        k[688]*y[IDX_N2I] + k[694]*y[IDX_NOI];
    IJth(jmatrix, 102, 98) = 0.0 + k[57]*y[IDX_NII] + k[470]*y[IDX_NHII] +
                        k[927]*y[IDX_N2I] - k[928]*y[IDX_NI] - k[929]*y[IDX_NI]
                        + k[931]*y[IDX_NOI];
    IJth(jmatrix, 102, 101) = 0.0 + k[167]*y[IDX_NII] + k[278] + k[694]*y[IDX_HeII] +
                        k[872]*y[IDX_CI] + k[886]*y[IDX_CH2I] +
                        k[931]*y[IDX_CHI] + k[947]*y[IDX_CNI] + k[988]*y[IDX_HI]
                        - k[1022]*y[IDX_NI] + k[1076]*y[IDX_OI] +
                        k[1105]*y[IDX_SiI] + k[1166];
    IJth(jmatrix, 102, 102) = 0.0 - k[174]*y[IDX_N2II] - k[238] - k[268] -
                        k[408]*y[IDX_CHII] - k[522]*y[IDX_H2II] -
                        k[736]*y[IDX_CH2II] - k[737]*y[IDX_CNII] -
                        k[738]*y[IDX_H2OII] - k[739]*y[IDX_H2OII] -
                        k[740]*y[IDX_NHII] - k[741]*y[IDX_NH2II] -
                        k[742]*y[IDX_O2II] - k[743]*y[IDX_OHII] -
                        k[744]*y[IDX_SiCII] - k[745]*y[IDX_SiOII] -
                        k[746]*y[IDX_SiOII] - k[928]*y[IDX_CHI] -
                        k[929]*y[IDX_CHI] - k[960]*y[IDX_H2I] -
                        k[1005]*y[IDX_CH2I] - k[1006]*y[IDX_CH2I] -
                        k[1007]*y[IDX_CH2I] - k[1008]*y[IDX_CH3I] -
                        k[1009]*y[IDX_CH3I] - k[1010]*y[IDX_CH3I] -
                        k[1011]*y[IDX_CNI] - k[1012]*y[IDX_CO2I] -
                        k[1013]*y[IDX_H2CNI] - k[1014]*y[IDX_HCOI] -
                        k[1015]*y[IDX_HCOI] - k[1016]*y[IDX_HCOI] -
                        k[1017]*y[IDX_HNOI] - k[1018]*y[IDX_NHI] -
                        k[1019]*y[IDX_NO2I] - k[1020]*y[IDX_NO2I] -
                        k[1021]*y[IDX_NO2I] - k[1022]*y[IDX_NOI] -
                        k[1023]*y[IDX_O2I] - k[1024]*y[IDX_O2HI] -
                        k[1025]*y[IDX_OHI] - k[1026]*y[IDX_OHI] -
                        k[1027]*y[IDX_SiCI] - k[1192]*y[IDX_CII] -
                        k[1194]*y[IDX_CI] - k[1210]*y[IDX_NII] - k[1290];
    IJth(jmatrix, 102, 103) = 0.0 + k[169]*y[IDX_NII] + k[768]*y[IDX_NHII] -
                        k[1025]*y[IDX_NI] - k[1026]*y[IDX_NI] +
                        k[1048]*y[IDX_NHI];
    IJth(jmatrix, 102, 104) = 0.0 + k[168]*y[IDX_NII] + k[766]*y[IDX_NHII] -
                        k[1023]*y[IDX_NI];
    IJth(jmatrix, 102, 105) = 0.0 + k[390]*y[IDX_NHII] + k[865]*y[IDX_N2I] +
                        k[870]*y[IDX_NHI] + k[872]*y[IDX_NOI] -
                        k[1194]*y[IDX_NI];
    IJth(jmatrix, 102, 108) = 0.0 + k[160]*y[IDX_NII] + k[754]*y[IDX_NHII];
    IJth(jmatrix, 102, 109) = 0.0 + k[767]*y[IDX_NHII] + k[825]*y[IDX_N2II] +
                        k[1047]*y[IDX_NHI] + k[1057]*y[IDX_CNI] +
                        k[1071]*y[IDX_N2I] + k[1076]*y[IDX_NOI];
    IJth(jmatrix, 102, 110) = 0.0 + k[303]*y[IDX_CNII] + k[337]*y[IDX_N2II] +
                        k[337]*y[IDX_N2II] + k[339]*y[IDX_N2HII] +
                        k[340]*y[IDX_NHII] + k[341]*y[IDX_NH2II] +
                        k[345]*y[IDX_NOII] + k[1220]*y[IDX_NII];
    IJth(jmatrix, 102, 111) = 0.0 + k[158]*y[IDX_NII] + k[751]*y[IDX_NHII];
    IJth(jmatrix, 102, 112) = 0.0 + k[540]*y[IDX_NHII] - k[960]*y[IDX_NI];
    IJth(jmatrix, 102, 113) = 0.0 + k[985]*y[IDX_NHI] + k[988]*y[IDX_NOI];
    IJth(jmatrix, 103, 37) = 0.0 + k[937]*y[IDX_CHI] + k[954]*y[IDX_COI] +
                        k[992]*y[IDX_HI] + k[992]*y[IDX_HI] + k[1077]*y[IDX_OI]
                        - k[1100]*y[IDX_OHI] + k[1171];
    IJth(jmatrix, 103, 39) = 0.0 + k[504]*y[IDX_HII] + k[595]*y[IDX_H3II] +
                        k[986]*y[IDX_HI];
    IJth(jmatrix, 103, 41) = 0.0 + k[862]*y[IDX_O2I];
    IJth(jmatrix, 103, 42) = 0.0 + k[995]*y[IDX_HI];
    IJth(jmatrix, 103, 44) = 0.0 + k[363]*y[IDX_EM];
    IJth(jmatrix, 103, 46) = 0.0 + k[1088]*y[IDX_OI];
    IJth(jmatrix, 103, 50) = 0.0 + k[333]*y[IDX_EM];
    IJth(jmatrix, 103, 51) = 0.0 + k[982]*y[IDX_HI] + k[1069]*y[IDX_OI] -
                        k[1097]*y[IDX_OHI];
    IJth(jmatrix, 103, 52) = 0.0 + k[248] + k[657]*y[IDX_HeII] + k[809]*y[IDX_OII] +
                        k[1121];
    IJth(jmatrix, 103, 55) = 0.0 + k[822]*y[IDX_OI];
    IJth(jmatrix, 103, 57) = 0.0 - k[225]*y[IDX_OHI] + k[560]*y[IDX_H2OI];
    IJth(jmatrix, 103, 59) = 0.0 - k[857]*y[IDX_OHI];
    IJth(jmatrix, 103, 60) = 0.0 - k[858]*y[IDX_OHI];
    IJth(jmatrix, 103, 61) = 0.0 - k[859]*y[IDX_OHI];
    IJth(jmatrix, 103, 62) = 0.0 - k[1102]*y[IDX_OHI];
    IJth(jmatrix, 103, 63) = 0.0 + k[559]*y[IDX_H2OII];
    IJth(jmatrix, 103, 64) = 0.0 - k[856]*y[IDX_OHI];
    IJth(jmatrix, 103, 65) = 0.0 - k[227]*y[IDX_OHI] + k[569]*y[IDX_H2OI];
    IJth(jmatrix, 103, 66) = 0.0 + k[317]*y[IDX_EM];
    IJth(jmatrix, 103, 67) = 0.0 + k[810]*y[IDX_OII] - k[922]*y[IDX_OHI] +
                        k[1056]*y[IDX_OI];
    IJth(jmatrix, 103, 68) = 0.0 - k[226]*y[IDX_OHI] + k[562]*y[IDX_H2OI] -
                        k[851]*y[IDX_OHI];
    IJth(jmatrix, 103, 69) = 0.0 - k[114]*y[IDX_OHI] - k[527]*y[IDX_OHI];
    IJth(jmatrix, 103, 70) = 0.0 + k[222]*y[IDX_OHII] + k[1074]*y[IDX_OI] -
                        k[1098]*y[IDX_OHI];
    IJth(jmatrix, 103, 71) = 0.0 + k[904]*y[IDX_H2OI] + k[911]*y[IDX_O2I] -
                        k[917]*y[IDX_OHI] - k[918]*y[IDX_OHI] -
                        k[919]*y[IDX_OHI];
    IJth(jmatrix, 103, 72) = 0.0 + k[971]*y[IDX_HI];
    IJth(jmatrix, 103, 73) = 0.0 - k[169]*y[IDX_OHI];
    IJth(jmatrix, 103, 74) = 0.0 - k[215]*y[IDX_OHI] + k[809]*y[IDX_CH3OHI] +
                        k[810]*y[IDX_CH4I] + k[813]*y[IDX_H2COI] -
                        k[819]*y[IDX_OHI];
    IJth(jmatrix, 103, 75) = 0.0 - k[853]*y[IDX_OHI];
    IJth(jmatrix, 103, 76) = 0.0 + k[772]*y[IDX_H2OI] + k[778]*y[IDX_O2I];
    IJth(jmatrix, 103, 77) = 0.0 + k[757]*y[IDX_H2OI] + k[765]*y[IDX_O2I] -
                        k[768]*y[IDX_OHI];
    IJth(jmatrix, 103, 79) = 0.0 - k[445]*y[IDX_OHI];
    IJth(jmatrix, 103, 80) = 0.0 + k[188]*y[IDX_OHII] + k[781]*y[IDX_H2OII] +
                        k[1030]*y[IDX_NOI] - k[1031]*y[IDX_OHI] -
                        k[1032]*y[IDX_OHI] + k[1073]*y[IDX_OI];
    IJth(jmatrix, 103, 81) = 0.0 + k[420]*y[IDX_O2I];
    IJth(jmatrix, 103, 82) = 0.0 + k[314]*y[IDX_EM] + k[383]*y[IDX_CI] +
                        k[424]*y[IDX_CH2I] + k[460]*y[IDX_CHI] +
                        k[553]*y[IDX_COI] + k[554]*y[IDX_H2COI] +
                        k[555]*y[IDX_H2OI] + k[556]*y[IDX_HCNI] +
                        k[558]*y[IDX_HCOI] + k[559]*y[IDX_HNCI] +
                        k[781]*y[IDX_NH2I] - k[852]*y[IDX_OHI];
    IJth(jmatrix, 103, 85) = 0.0 + k[324]*y[IDX_EM] + k[325]*y[IDX_EM];
    IJth(jmatrix, 103, 87) = 0.0 - k[379]*y[IDX_OHI];
    IJth(jmatrix, 103, 88) = 0.0 + k[556]*y[IDX_H2OII] + k[1063]*y[IDX_OI] -
                        k[1094]*y[IDX_OHI] - k[1095]*y[IDX_OHI];
    IJth(jmatrix, 103, 89) = 0.0 + k[411]*y[IDX_O2I] - k[415]*y[IDX_OHI];
    IJth(jmatrix, 103, 90) = 0.0 + k[45]*y[IDX_OHII] + k[424]*y[IDX_H2OII] +
                        k[887]*y[IDX_NOI] + k[893]*y[IDX_O2I] + k[897]*y[IDX_OI]
                        - k[898]*y[IDX_OHI] - k[899]*y[IDX_OHI] -
                        k[900]*y[IDX_OHI];
    IJth(jmatrix, 103, 92) = 0.0 + k[1036]*y[IDX_H2OI] + k[1043]*y[IDX_NOI] +
                        k[1045]*y[IDX_O2I] + k[1047]*y[IDX_OI] -
                        k[1048]*y[IDX_OHI] - k[1049]*y[IDX_OHI] -
                        k[1050]*y[IDX_OHI];
    IJth(jmatrix, 103, 93) = 0.0 + k[45]*y[IDX_CH2I] + k[62]*y[IDX_CHI] +
                        k[188]*y[IDX_NH2I] + k[219]*y[IDX_H2COI] +
                        k[220]*y[IDX_H2OI] + k[221]*y[IDX_HCOI] +
                        k[222]*y[IDX_NH3I] + k[223]*y[IDX_NOI] +
                        k[224]*y[IDX_O2I] - k[847]*y[IDX_OHI];
    IJth(jmatrix, 103, 94) = 0.0 - k[1090]*y[IDX_OHI] - k[1091]*y[IDX_OHI];
    IJth(jmatrix, 103, 95) = 0.0 + k[219]*y[IDX_OHII] + k[554]*y[IDX_H2OII] +
                        k[813]*y[IDX_OII] + k[1061]*y[IDX_OI] -
                        k[1093]*y[IDX_OHI];
    IJth(jmatrix, 103, 96) = 0.0 + k[221]*y[IDX_OHII] + k[558]*y[IDX_H2OII] +
                        k[1001]*y[IDX_O2I] + k[1067]*y[IDX_OI] -
                        k[1096]*y[IDX_OHI];
    IJth(jmatrix, 103, 97) = 0.0 + k[657]*y[IDX_CH3OHI] + k[674]*y[IDX_H2OI] -
                        k[699]*y[IDX_OHI];
    IJth(jmatrix, 103, 98) = 0.0 + k[62]*y[IDX_OHII] + k[460]*y[IDX_H2OII] +
                        k[935]*y[IDX_O2I] + k[937]*y[IDX_O2HI] +
                        k[940]*y[IDX_OI] - k[941]*y[IDX_OHI];
    IJth(jmatrix, 103, 99) = 0.0 + k[595]*y[IDX_NO2I] - k[600]*y[IDX_OHI];
    IJth(jmatrix, 103, 101) = 0.0 + k[223]*y[IDX_OHII] + k[887]*y[IDX_CH2I] +
                        k[988]*y[IDX_HI] + k[1030]*y[IDX_NH2I] +
                        k[1043]*y[IDX_NHI] - k[1099]*y[IDX_OHI];
    IJth(jmatrix, 103, 102) = 0.0 - k[1025]*y[IDX_OHI] - k[1026]*y[IDX_OHI];
    IJth(jmatrix, 103, 103) = 0.0 - k[7]*y[IDX_H2I] - k[13]*y[IDX_HI] -
                        k[90]*y[IDX_HII] - k[114]*y[IDX_H2II] -
                        k[169]*y[IDX_NII] - k[215]*y[IDX_OII] -
                        k[225]*y[IDX_CNII] - k[226]*y[IDX_COII] -
                        k[227]*y[IDX_N2II] - k[284] - k[379]*y[IDX_CII] -
                        k[415]*y[IDX_CHII] - k[445]*y[IDX_CH3II] -
                        k[527]*y[IDX_H2II] - k[600]*y[IDX_H3II] -
                        k[699]*y[IDX_HeII] - k[768]*y[IDX_NHII] -
                        k[819]*y[IDX_OII] - k[847]*y[IDX_OHII] -
                        k[851]*y[IDX_COII] - k[852]*y[IDX_H2OII] -
                        k[853]*y[IDX_HCNII] - k[854]*y[IDX_HCOII] -
                        k[855]*y[IDX_HCOII] - k[856]*y[IDX_HNOII] -
                        k[857]*y[IDX_N2HII] - k[858]*y[IDX_O2HII] -
                        k[859]*y[IDX_SiII] - k[875]*y[IDX_CI] - k[876]*y[IDX_CI]
                        - k[898]*y[IDX_CH2I] - k[899]*y[IDX_CH2I] -
                        k[900]*y[IDX_CH2I] - k[917]*y[IDX_CH3I] -
                        k[918]*y[IDX_CH3I] - k[919]*y[IDX_CH3I] -
                        k[922]*y[IDX_CH4I] - k[941]*y[IDX_CHI] -
                        k[966]*y[IDX_H2I] - k[996]*y[IDX_HI] - k[1025]*y[IDX_NI]
                        - k[1026]*y[IDX_NI] - k[1031]*y[IDX_NH2I] -
                        k[1032]*y[IDX_NH2I] - k[1048]*y[IDX_NHI] -
                        k[1049]*y[IDX_NHI] - k[1050]*y[IDX_NHI] -
                        k[1080]*y[IDX_OI] - k[1090]*y[IDX_CNI] -
                        k[1091]*y[IDX_CNI] - k[1092]*y[IDX_COI] -
                        k[1093]*y[IDX_H2COI] - k[1094]*y[IDX_HCNI] -
                        k[1095]*y[IDX_HCNI] - k[1096]*y[IDX_HCOI] -
                        k[1097]*y[IDX_HNOI] - k[1098]*y[IDX_NH3I] -
                        k[1099]*y[IDX_NOI] - k[1100]*y[IDX_O2HI] -
                        k[1101]*y[IDX_OHI] - k[1101]*y[IDX_OHI] -
                        k[1101]*y[IDX_OHI] - k[1101]*y[IDX_OHI] -
                        k[1102]*y[IDX_SiI] - k[1174] - k[1175] -
                        k[1208]*y[IDX_HI] - k[1254];
    IJth(jmatrix, 103, 104) = 0.0 + k[224]*y[IDX_OHII] + k[411]*y[IDX_CHII] +
                        k[420]*y[IDX_CH2II] + k[765]*y[IDX_NHII] +
                        k[778]*y[IDX_NH2II] + k[862]*y[IDX_SiH2II] +
                        k[893]*y[IDX_CH2I] + k[911]*y[IDX_CH3I] +
                        k[935]*y[IDX_CHI] + k[964]*y[IDX_H2I] +
                        k[964]*y[IDX_H2I] + k[989]*y[IDX_HI] +
                        k[1001]*y[IDX_HCOI] + k[1045]*y[IDX_NHI];
    IJth(jmatrix, 103, 105) = 0.0 + k[383]*y[IDX_H2OII] - k[875]*y[IDX_OHI] -
                        k[876]*y[IDX_OHI];
    IJth(jmatrix, 103, 106) = 0.0 - k[90]*y[IDX_OHI] + k[504]*y[IDX_NO2I];
    IJth(jmatrix, 103, 107) = 0.0 - k[854]*y[IDX_OHI] - k[855]*y[IDX_OHI];
    IJth(jmatrix, 103, 108) = 0.0 + k[4]*y[IDX_H2I] + k[11]*y[IDX_HI] +
                        k[220]*y[IDX_OHII] + k[256] + k[555]*y[IDX_H2OII] +
                        k[560]*y[IDX_CNII] + k[562]*y[IDX_COII] +
                        k[569]*y[IDX_N2II] + k[674]*y[IDX_HeII] +
                        k[757]*y[IDX_NHII] + k[772]*y[IDX_NH2II] +
                        k[904]*y[IDX_CH3I] + k[975]*y[IDX_HI] +
                        k[1036]*y[IDX_NHI] + k[1062]*y[IDX_OI] +
                        k[1062]*y[IDX_OI] + k[1142];
    IJth(jmatrix, 103, 109) = 0.0 + k[822]*y[IDX_CH4II] + k[897]*y[IDX_CH2I] +
                        k[940]*y[IDX_CHI] + k[965]*y[IDX_H2I] +
                        k[1047]*y[IDX_NHI] + k[1056]*y[IDX_CH4I] +
                        k[1061]*y[IDX_H2COI] + k[1062]*y[IDX_H2OI] +
                        k[1062]*y[IDX_H2OI] + k[1063]*y[IDX_HCNI] +
                        k[1067]*y[IDX_HCOI] + k[1069]*y[IDX_HNOI] +
                        k[1073]*y[IDX_NH2I] + k[1074]*y[IDX_NH3I] +
                        k[1077]*y[IDX_O2HI] - k[1080]*y[IDX_OHI] +
                        k[1088]*y[IDX_SiH4I] + k[1207]*y[IDX_HI];
    IJth(jmatrix, 103, 110) = 0.0 + k[314]*y[IDX_H2OII] + k[317]*y[IDX_H3COII] +
                        k[324]*y[IDX_H3OII] + k[325]*y[IDX_H3OII] +
                        k[333]*y[IDX_HCO2II] + k[363]*y[IDX_SiOHII];
    IJth(jmatrix, 103, 111) = 0.0 + k[553]*y[IDX_H2OII] + k[954]*y[IDX_O2HI] +
                        k[972]*y[IDX_HI] - k[1092]*y[IDX_OHI];
    IJth(jmatrix, 103, 112) = 0.0 + k[4]*y[IDX_H2OI] - k[7]*y[IDX_OHI] +
                        k[964]*y[IDX_O2I] + k[964]*y[IDX_O2I] + k[965]*y[IDX_OI]
                        - k[966]*y[IDX_OHI];
    IJth(jmatrix, 103, 113) = 0.0 + k[11]*y[IDX_H2OI] - k[13]*y[IDX_OHI] +
                        k[971]*y[IDX_CO2I] + k[972]*y[IDX_COI] +
                        k[975]*y[IDX_H2OI] + k[982]*y[IDX_HNOI] +
                        k[986]*y[IDX_NO2I] + k[988]*y[IDX_NOI] +
                        k[989]*y[IDX_O2I] + k[992]*y[IDX_O2HI] +
                        k[992]*y[IDX_O2HI] + k[995]*y[IDX_OCNI] -
                        k[996]*y[IDX_OHI] + k[1207]*y[IDX_OI] -
                        k[1208]*y[IDX_OHI];
    IJth(jmatrix, 104, 8) = 0.0 + k[1355] + k[1356] + k[1357] + k[1358];
    IJth(jmatrix, 104, 37) = 0.0 + k[281] + k[914]*y[IDX_CH3I] + k[938]*y[IDX_CHI] +
                        k[991]*y[IDX_HI] + k[1003]*y[IDX_HCOI] +
                        k[1024]*y[IDX_NI] + k[1077]*y[IDX_OI] +
                        k[1100]*y[IDX_OHI] + k[1170];
    IJth(jmatrix, 104, 39) = 0.0 + k[818]*y[IDX_OII] + k[1021]*y[IDX_NI] +
                        k[1075]*y[IDX_OI];
    IJth(jmatrix, 104, 41) = 0.0 - k[862]*y[IDX_O2I];
    IJth(jmatrix, 104, 42) = 0.0 - k[1054]*y[IDX_O2I] - k[1055]*y[IDX_O2I] +
                        k[1079]*y[IDX_OI];
    IJth(jmatrix, 104, 49) = 0.0 + k[835]*y[IDX_OI];
    IJth(jmatrix, 104, 50) = 0.0 + k[824]*y[IDX_OI];
    IJth(jmatrix, 104, 51) = 0.0 + k[1070]*y[IDX_OI];
    IJth(jmatrix, 104, 52) = 0.0 + k[820]*y[IDX_O2II];
    IJth(jmatrix, 104, 53) = 0.0 + k[152]*y[IDX_O2II];
    IJth(jmatrix, 104, 55) = 0.0 - k[51]*y[IDX_O2I];
    IJth(jmatrix, 104, 57) = 0.0 - k[68]*y[IDX_O2I] - k[481]*y[IDX_O2I];
    IJth(jmatrix, 104, 60) = 0.0 + k[347]*y[IDX_EM] + k[392]*y[IDX_CI] +
                        k[436]*y[IDX_CH2I] + k[474]*y[IDX_CHI] +
                        k[483]*y[IDX_CNI] + k[488]*y[IDX_COI] +
                        k[544]*y[IDX_H2I] + k[552]*y[IDX_H2COI] +
                        k[571]*y[IDX_H2OI] + k[632]*y[IDX_HCNI] +
                        k[645]*y[IDX_HCOI] + k[651]*y[IDX_HNCI] +
                        k[733]*y[IDX_N2I] + k[790]*y[IDX_NH2I] +
                        k[805]*y[IDX_NHI] + k[807]*y[IDX_NOI] +
                        k[821]*y[IDX_CO2I] + k[829]*y[IDX_OI] +
                        k[858]*y[IDX_OHI];
    IJth(jmatrix, 104, 62) = 0.0 + k[230]*y[IDX_O2II] - k[1106]*y[IDX_O2I];
    IJth(jmatrix, 104, 63) = 0.0 + k[651]*y[IDX_O2HII];
    IJth(jmatrix, 104, 65) = 0.0 - k[173]*y[IDX_O2I];
    IJth(jmatrix, 104, 67) = 0.0 - k[921]*y[IDX_O2I];
    IJth(jmatrix, 104, 68) = 0.0 - k[73]*y[IDX_O2I];
    IJth(jmatrix, 104, 69) = 0.0 - k[113]*y[IDX_O2I] - k[525]*y[IDX_O2I];
    IJth(jmatrix, 104, 70) = 0.0 + k[198]*y[IDX_O2II];
    IJth(jmatrix, 104, 71) = 0.0 - k[911]*y[IDX_O2I] - k[912]*y[IDX_O2I] -
                        k[913]*y[IDX_O2I] + k[914]*y[IDX_O2HI];
    IJth(jmatrix, 104, 72) = 0.0 + k[668]*y[IDX_HeII] + k[821]*y[IDX_O2HII] +
                        k[1059]*y[IDX_OI];
    IJth(jmatrix, 104, 73) = 0.0 - k[168]*y[IDX_O2I] - k[728]*y[IDX_O2I] -
                        k[729]*y[IDX_O2I];
    IJth(jmatrix, 104, 74) = 0.0 - k[214]*y[IDX_O2I] + k[818]*y[IDX_NO2I];
    IJth(jmatrix, 104, 75) = 0.0 - k[133]*y[IDX_O2I];
    IJth(jmatrix, 104, 76) = 0.0 - k[777]*y[IDX_O2I] - k[778]*y[IDX_O2I];
    IJth(jmatrix, 104, 77) = 0.0 - k[179]*y[IDX_O2I] - k[765]*y[IDX_O2I] -
                        k[766]*y[IDX_O2I];
    IJth(jmatrix, 104, 78) = 0.0 + k[30]*y[IDX_CI] + k[44]*y[IDX_CH2I] +
                        k[61]*y[IDX_CHI] + k[116]*y[IDX_H2COI] +
                        k[137]*y[IDX_HCOI] + k[152]*y[IDX_MgI] +
                        k[187]*y[IDX_NH2I] + k[198]*y[IDX_NH3I] +
                        k[205]*y[IDX_NOI] + k[230]*y[IDX_SiI] +
                        k[551]*y[IDX_H2COI] + k[820]*y[IDX_CH3OHI];
    IJth(jmatrix, 104, 79) = 0.0 - k[442]*y[IDX_O2I];
    IJth(jmatrix, 104, 80) = 0.0 + k[187]*y[IDX_O2II] + k[790]*y[IDX_O2HII];
    IJth(jmatrix, 104, 81) = 0.0 - k[420]*y[IDX_O2I];
    IJth(jmatrix, 104, 82) = 0.0 - k[121]*y[IDX_O2I];
    IJth(jmatrix, 104, 86) = 0.0 + k[733]*y[IDX_O2HII];
    IJth(jmatrix, 104, 87) = 0.0 - k[376]*y[IDX_O2I] - k[377]*y[IDX_O2I];
    IJth(jmatrix, 104, 88) = 0.0 + k[632]*y[IDX_O2HII];
    IJth(jmatrix, 104, 89) = 0.0 - k[411]*y[IDX_O2I] - k[412]*y[IDX_O2I] -
                        k[413]*y[IDX_O2I];
    IJth(jmatrix, 104, 90) = 0.0 + k[44]*y[IDX_O2II] + k[436]*y[IDX_O2HII] -
                        k[889]*y[IDX_O2I] - k[890]*y[IDX_O2I] -
                        k[891]*y[IDX_O2I] - k[892]*y[IDX_O2I] -
                        k[893]*y[IDX_O2I];
    IJth(jmatrix, 104, 91) = 0.0 - k[549]*y[IDX_O2I];
    IJth(jmatrix, 104, 92) = 0.0 + k[805]*y[IDX_O2HII] - k[1044]*y[IDX_O2I] -
                        k[1045]*y[IDX_O2I];
    IJth(jmatrix, 104, 93) = 0.0 - k[224]*y[IDX_O2I];
    IJth(jmatrix, 104, 94) = 0.0 + k[483]*y[IDX_O2HII] - k[948]*y[IDX_O2I] -
                        k[949]*y[IDX_O2I];
    IJth(jmatrix, 104, 95) = 0.0 + k[116]*y[IDX_O2II] + k[551]*y[IDX_O2II] +
                        k[552]*y[IDX_O2HII];
    IJth(jmatrix, 104, 96) = 0.0 + k[137]*y[IDX_O2II] + k[645]*y[IDX_O2HII] -
                        k[1001]*y[IDX_O2I] - k[1002]*y[IDX_O2I] +
                        k[1003]*y[IDX_O2HI];
    IJth(jmatrix, 104, 97) = 0.0 - k[146]*y[IDX_O2I] + k[668]*y[IDX_CO2I] -
                        k[696]*y[IDX_O2I];
    IJth(jmatrix, 104, 98) = 0.0 + k[61]*y[IDX_O2II] + k[474]*y[IDX_O2HII] -
                        k[933]*y[IDX_O2I] - k[934]*y[IDX_O2I] -
                        k[935]*y[IDX_O2I] - k[936]*y[IDX_O2I] +
                        k[938]*y[IDX_O2HI];
    IJth(jmatrix, 104, 99) = 0.0 - k[597]*y[IDX_O2I];
    IJth(jmatrix, 104, 101) = 0.0 + k[205]*y[IDX_O2II] + k[807]*y[IDX_O2HII] +
                        k[1051]*y[IDX_NOI] + k[1051]*y[IDX_NOI] -
                        k[1052]*y[IDX_O2I] + k[1076]*y[IDX_OI];
    IJth(jmatrix, 104, 102) = 0.0 + k[1021]*y[IDX_NO2I] - k[1023]*y[IDX_O2I] +
                        k[1024]*y[IDX_O2HI];
    IJth(jmatrix, 104, 103) = 0.0 + k[858]*y[IDX_O2HII] + k[1080]*y[IDX_OI] +
                        k[1100]*y[IDX_O2HI];
    IJth(jmatrix, 104, 104) = 0.0 - k[6]*y[IDX_H2I] - k[12]*y[IDX_HI] -
                        k[51]*y[IDX_CH4II] - k[68]*y[IDX_CNII] -
                        k[73]*y[IDX_COII] - k[88]*y[IDX_HII] -
                        k[113]*y[IDX_H2II] - k[121]*y[IDX_H2OII] -
                        k[133]*y[IDX_HCNII] - k[146]*y[IDX_HeII] -
                        k[168]*y[IDX_NII] - k[173]*y[IDX_N2II] -
                        k[179]*y[IDX_NHII] - k[214]*y[IDX_OII] -
                        k[224]*y[IDX_OHII] - k[279] - k[280] - k[376]*y[IDX_CII]
                        - k[377]*y[IDX_CII] - k[411]*y[IDX_CHII] -
                        k[412]*y[IDX_CHII] - k[413]*y[IDX_CHII] -
                        k[420]*y[IDX_CH2II] - k[442]*y[IDX_CH3II] -
                        k[481]*y[IDX_CNII] - k[525]*y[IDX_H2II] -
                        k[549]*y[IDX_H2COII] - k[597]*y[IDX_H3II] -
                        k[696]*y[IDX_HeII] - k[728]*y[IDX_NII] -
                        k[729]*y[IDX_NII] - k[765]*y[IDX_NHII] -
                        k[766]*y[IDX_NHII] - k[777]*y[IDX_NH2II] -
                        k[778]*y[IDX_NH2II] - k[862]*y[IDX_SiH2II] -
                        k[873]*y[IDX_CI] - k[889]*y[IDX_CH2I] -
                        k[890]*y[IDX_CH2I] - k[891]*y[IDX_CH2I] -
                        k[892]*y[IDX_CH2I] - k[893]*y[IDX_CH2I] -
                        k[911]*y[IDX_CH3I] - k[912]*y[IDX_CH3I] -
                        k[913]*y[IDX_CH3I] - k[921]*y[IDX_CH4I] -
                        k[933]*y[IDX_CHI] - k[934]*y[IDX_CHI] -
                        k[935]*y[IDX_CHI] - k[936]*y[IDX_CHI] -
                        k[948]*y[IDX_CNI] - k[949]*y[IDX_CNI] -
                        k[953]*y[IDX_COI] - k[963]*y[IDX_H2I] -
                        k[964]*y[IDX_H2I] - k[989]*y[IDX_HI] -
                        k[1001]*y[IDX_HCOI] - k[1002]*y[IDX_HCOI] -
                        k[1023]*y[IDX_NI] - k[1044]*y[IDX_NHI] -
                        k[1045]*y[IDX_NHI] - k[1052]*y[IDX_NOI] -
                        k[1054]*y[IDX_OCNI] - k[1055]*y[IDX_OCNI] -
                        k[1106]*y[IDX_SiI] - k[1168] - k[1169] - k[1292];
    IJth(jmatrix, 104, 105) = 0.0 + k[30]*y[IDX_O2II] + k[392]*y[IDX_O2HII] -
                        k[873]*y[IDX_O2I];
    IJth(jmatrix, 104, 106) = 0.0 - k[88]*y[IDX_O2I];
    IJth(jmatrix, 104, 108) = 0.0 + k[571]*y[IDX_O2HII];
    IJth(jmatrix, 104, 109) = 0.0 + k[824]*y[IDX_HCO2II] + k[829]*y[IDX_O2HII] +
                        k[835]*y[IDX_SiOII] + k[1059]*y[IDX_CO2I] +
                        k[1070]*y[IDX_HNOI] + k[1075]*y[IDX_NO2I] +
                        k[1076]*y[IDX_NOI] + k[1077]*y[IDX_O2HI] +
                        k[1079]*y[IDX_OCNI] + k[1080]*y[IDX_OHI] +
                        k[1211]*y[IDX_OI] + k[1211]*y[IDX_OI];
    IJth(jmatrix, 104, 110) = 0.0 + k[347]*y[IDX_O2HII];
    IJth(jmatrix, 104, 111) = 0.0 + k[488]*y[IDX_O2HII] - k[953]*y[IDX_O2I];
    IJth(jmatrix, 104, 112) = 0.0 - k[6]*y[IDX_O2I] + k[544]*y[IDX_O2HII] -
                        k[963]*y[IDX_O2I] - k[964]*y[IDX_O2I];
    IJth(jmatrix, 104, 113) = 0.0 - k[12]*y[IDX_O2I] - k[989]*y[IDX_O2I] +
                        k[991]*y[IDX_O2HI];
    IJth(jmatrix, 105, 22) = 0.0 + k[351]*y[IDX_EM];
    IJth(jmatrix, 105, 28) = 0.0 - k[1004]*y[IDX_CI];
    IJth(jmatrix, 105, 30) = 0.0 + k[350]*y[IDX_EM];
    IJth(jmatrix, 105, 32) = 0.0 + k[22]*y[IDX_CII] + k[286];
    IJth(jmatrix, 105, 33) = 0.0 + k[23]*y[IDX_CII] + k[287] + k[700]*y[IDX_HeII] +
                        k[1177];
    IJth(jmatrix, 105, 36) = 0.0 + k[349]*y[IDX_EM] + k[831]*y[IDX_OI];
    IJth(jmatrix, 105, 38) = 0.0 + k[24]*y[IDX_CII] + k[288] + k[701]*y[IDX_HeII] +
                        k[1084]*y[IDX_OI] + k[1178];
    IJth(jmatrix, 105, 42) = 0.0 - k[874]*y[IDX_CI];
    IJth(jmatrix, 105, 43) = 0.0 + k[25]*y[IDX_CII];
    IJth(jmatrix, 105, 45) = 0.0 - k[394]*y[IDX_CI];
    IJth(jmatrix, 105, 47) = 0.0 - k[877]*y[IDX_CI];
    IJth(jmatrix, 105, 48) = 0.0 + k[26]*y[IDX_CII];
    IJth(jmatrix, 105, 49) = 0.0 - k[395]*y[IDX_CI];
    IJth(jmatrix, 105, 50) = 0.0 - k[387]*y[IDX_CI];
    IJth(jmatrix, 105, 53) = 0.0 + k[18]*y[IDX_CII];
    IJth(jmatrix, 105, 57) = 0.0 - k[27]*y[IDX_CI] + k[303]*y[IDX_EM] +
                        k[737]*y[IDX_NI];
    IJth(jmatrix, 105, 59) = 0.0 - k[389]*y[IDX_CI];
    IJth(jmatrix, 105, 60) = 0.0 - k[392]*y[IDX_CI];
    IJth(jmatrix, 105, 62) = 0.0 + k[21]*y[IDX_CII] + k[1104]*y[IDX_COI];
    IJth(jmatrix, 105, 63) = 0.0 + k[407]*y[IDX_CHII] + k[685]*y[IDX_HeII];
    IJth(jmatrix, 105, 64) = 0.0 - k[388]*y[IDX_CI];
    IJth(jmatrix, 105, 65) = 0.0 - k[29]*y[IDX_CI];
    IJth(jmatrix, 105, 68) = 0.0 - k[28]*y[IDX_CI] + k[304]*y[IDX_EM] +
                        k[458]*y[IDX_CHI];
    IJth(jmatrix, 105, 69) = 0.0 - k[509]*y[IDX_CI];
    IJth(jmatrix, 105, 70) = 0.0 + k[19]*y[IDX_CII];
    IJth(jmatrix, 105, 72) = 0.0 + k[667]*y[IDX_HeII];
    IJth(jmatrix, 105, 73) = 0.0 + k[720]*y[IDX_COI];
    IJth(jmatrix, 105, 74) = 0.0 + k[811]*y[IDX_CNI] - k[1195]*y[IDX_CI];
    IJth(jmatrix, 105, 75) = 0.0 - k[385]*y[IDX_CI];
    IJth(jmatrix, 105, 77) = 0.0 - k[390]*y[IDX_CI];
    IJth(jmatrix, 105, 78) = 0.0 - k[30]*y[IDX_CI] - k[391]*y[IDX_CI];
    IJth(jmatrix, 105, 80) = 0.0 - k[866]*y[IDX_CI] - k[867]*y[IDX_CI] -
                        k[868]*y[IDX_CI];
    IJth(jmatrix, 105, 81) = 0.0 + k[295]*y[IDX_EM] + k[296]*y[IDX_EM];
    IJth(jmatrix, 105, 82) = 0.0 - k[383]*y[IDX_CI];
    IJth(jmatrix, 105, 85) = 0.0 - k[384]*y[IDX_CI];
    IJth(jmatrix, 105, 86) = 0.0 - k[865]*y[IDX_CI];
    IJth(jmatrix, 105, 87) = 0.0 + k[14]*y[IDX_CH2I] + k[15]*y[IDX_CHI] +
                        k[16]*y[IDX_H2COI] + k[17]*y[IDX_HCOI] +
                        k[18]*y[IDX_MgI] + k[19]*y[IDX_NH3I] + k[20]*y[IDX_NOI]
                        + k[21]*y[IDX_SiI] + k[22]*y[IDX_SiC2I] +
                        k[23]*y[IDX_SiC3I] + k[24]*y[IDX_SiCI] +
                        k[25]*y[IDX_SiH2I] + k[26]*y[IDX_SiH3I] +
                        k[1214]*y[IDX_EM];
    IJth(jmatrix, 105, 88) = 0.0 + k[405]*y[IDX_CHII];
    IJth(jmatrix, 105, 89) = 0.0 + k[294]*y[IDX_EM] + k[400]*y[IDX_H2COI] +
                        k[403]*y[IDX_H2OI] + k[405]*y[IDX_HCNI] +
                        k[407]*y[IDX_HNCI] + k[1108];
    IJth(jmatrix, 105, 90) = 0.0 + k[14]*y[IDX_CII] - k[863]*y[IDX_CI];
    IJth(jmatrix, 105, 92) = 0.0 - k[869]*y[IDX_CI] - k[870]*y[IDX_CI];
    IJth(jmatrix, 105, 93) = 0.0 - k[393]*y[IDX_CI];
    IJth(jmatrix, 105, 94) = 0.0 + k[251] + k[663]*y[IDX_HeII] + k[811]*y[IDX_OII] +
                        k[1011]*y[IDX_NI] + k[1058]*y[IDX_OI] + k[1130];
    IJth(jmatrix, 105, 95) = 0.0 + k[16]*y[IDX_CII] + k[400]*y[IDX_CHII];
    IJth(jmatrix, 105, 96) = 0.0 + k[17]*y[IDX_CII] - k[864]*y[IDX_CI];
    IJth(jmatrix, 105, 97) = 0.0 - k[139]*y[IDX_CI] + k[663]*y[IDX_CNI] +
                        k[667]*y[IDX_CO2I] + k[685]*y[IDX_HNCI] +
                        k[700]*y[IDX_SiC3I] + k[701]*y[IDX_SiCI];
    IJth(jmatrix, 105, 98) = 0.0 + k[2]*y[IDX_H2I] + k[9]*y[IDX_HI] +
                        k[15]*y[IDX_CII] + k[250] + k[458]*y[IDX_COII] +
                        k[929]*y[IDX_NI] + k[940]*y[IDX_OI] + k[970]*y[IDX_HI] +
                        k[1128];
    IJth(jmatrix, 105, 99) = 0.0 - k[576]*y[IDX_CI];
    IJth(jmatrix, 105, 101) = 0.0 + k[20]*y[IDX_CII] - k[871]*y[IDX_CI] -
                        k[872]*y[IDX_CI];
    IJth(jmatrix, 105, 102) = 0.0 + k[737]*y[IDX_CNII] + k[929]*y[IDX_CHI] +
                        k[1011]*y[IDX_CNI] - k[1194]*y[IDX_CI];
    IJth(jmatrix, 105, 103) = 0.0 - k[875]*y[IDX_CI] - k[876]*y[IDX_CI];
    IJth(jmatrix, 105, 104) = 0.0 - k[873]*y[IDX_CI];
    IJth(jmatrix, 105, 105) = 0.0 - k[27]*y[IDX_CNII] - k[28]*y[IDX_COII] -
                        k[29]*y[IDX_N2II] - k[30]*y[IDX_O2II] -
                        k[139]*y[IDX_HeII] - k[231] - k[240] -
                        k[383]*y[IDX_H2OII] - k[384]*y[IDX_H3OII] -
                        k[385]*y[IDX_HCNII] - k[386]*y[IDX_HCOII] -
                        k[387]*y[IDX_HCO2II] - k[388]*y[IDX_HNOII] -
                        k[389]*y[IDX_N2HII] - k[390]*y[IDX_NHII] -
                        k[391]*y[IDX_O2II] - k[392]*y[IDX_O2HII] -
                        k[393]*y[IDX_OHII] - k[394]*y[IDX_SiHII] -
                        k[395]*y[IDX_SiOII] - k[509]*y[IDX_H2II] -
                        k[576]*y[IDX_H3II] - k[863]*y[IDX_CH2I] -
                        k[864]*y[IDX_HCOI] - k[865]*y[IDX_N2I] -
                        k[866]*y[IDX_NH2I] - k[867]*y[IDX_NH2I] -
                        k[868]*y[IDX_NH2I] - k[869]*y[IDX_NHI] -
                        k[870]*y[IDX_NHI] - k[871]*y[IDX_NOI] -
                        k[872]*y[IDX_NOI] - k[873]*y[IDX_O2I] -
                        k[874]*y[IDX_OCNI] - k[875]*y[IDX_OHI] -
                        k[876]*y[IDX_OHI] - k[877]*y[IDX_SiHI] -
                        k[955]*y[IDX_H2I] - k[1004]*y[IDX_HNCOI] - k[1107] -
                        k[1194]*y[IDX_NI] - k[1195]*y[IDX_OII] -
                        k[1196]*y[IDX_OI] - k[1200]*y[IDX_H2I] -
                        k[1206]*y[IDX_HI] - k[1250];
    IJth(jmatrix, 105, 107) = 0.0 - k[386]*y[IDX_CI];
    IJth(jmatrix, 105, 108) = 0.0 + k[403]*y[IDX_CHII];
    IJth(jmatrix, 105, 109) = 0.0 + k[831]*y[IDX_SiCII] + k[940]*y[IDX_CHI] +
                        k[1058]*y[IDX_CNI] + k[1084]*y[IDX_SiCI] -
                        k[1196]*y[IDX_CI];
    IJth(jmatrix, 105, 110) = 0.0 + k[294]*y[IDX_CHII] + k[295]*y[IDX_CH2II] +
                        k[296]*y[IDX_CH2II] + k[303]*y[IDX_CNII] +
                        k[304]*y[IDX_COII] + k[349]*y[IDX_SiCII] +
                        k[350]*y[IDX_SiC2II] + k[351]*y[IDX_SiC3II] +
                        k[1214]*y[IDX_CII];
    IJth(jmatrix, 105, 111) = 0.0 + k[253] + k[720]*y[IDX_NII] + k[972]*y[IDX_HI] +
                        k[1104]*y[IDX_SiI] + k[1133];
    IJth(jmatrix, 105, 112) = 0.0 + k[2]*y[IDX_CHI] - k[955]*y[IDX_CI] -
                        k[1200]*y[IDX_CI];
    IJth(jmatrix, 105, 113) = 0.0 + k[9]*y[IDX_CHI] + k[970]*y[IDX_CHI] +
                        k[972]*y[IDX_COI] - k[1206]*y[IDX_CI];
    IJth(jmatrix, 106, 26) = 0.0 - k[499]*y[IDX_HII];
    IJth(jmatrix, 106, 28) = 0.0 - k[502]*y[IDX_HII];
    IJth(jmatrix, 106, 32) = 0.0 - k[92]*y[IDX_HII];
    IJth(jmatrix, 106, 33) = 0.0 - k[93]*y[IDX_HII];
    IJth(jmatrix, 106, 38) = 0.0 - k[94]*y[IDX_HII];
    IJth(jmatrix, 106, 39) = 0.0 - k[504]*y[IDX_HII];
    IJth(jmatrix, 106, 43) = 0.0 - k[95]*y[IDX_HII] - k[505]*y[IDX_HII];
    IJth(jmatrix, 106, 46) = 0.0 - k[97]*y[IDX_HII] - k[507]*y[IDX_HII];
    IJth(jmatrix, 106, 47) = 0.0 - k[98]*y[IDX_HII] - k[508]*y[IDX_HII];
    IJth(jmatrix, 106, 48) = 0.0 - k[96]*y[IDX_HII] - k[506]*y[IDX_HII];
    IJth(jmatrix, 106, 51) = 0.0 - k[503]*y[IDX_HII] + k[687]*y[IDX_HeII];
    IJth(jmatrix, 106, 52) = 0.0 - k[492]*y[IDX_HII] - k[493]*y[IDX_HII] -
                        k[494]*y[IDX_HII];
    IJth(jmatrix, 106, 53) = 0.0 - k[83]*y[IDX_HII];
    IJth(jmatrix, 106, 56) = 0.0 - k[99]*y[IDX_HII];
    IJth(jmatrix, 106, 57) = 0.0 + k[126]*y[IDX_HI];
    IJth(jmatrix, 106, 62) = 0.0 - k[91]*y[IDX_HII];
    IJth(jmatrix, 106, 63) = 0.0 - k[1]*y[IDX_HII] + k[1]*y[IDX_HII];
    IJth(jmatrix, 106, 67) = 0.0 - k[77]*y[IDX_HII] - k[495]*y[IDX_HII] +
                        k[661]*y[IDX_HeII];
    IJth(jmatrix, 106, 68) = 0.0 + k[127]*y[IDX_HI];
    IJth(jmatrix, 106, 69) = 0.0 + k[128]*y[IDX_HI] + k[1134];
    IJth(jmatrix, 106, 70) = 0.0 - k[85]*y[IDX_HII];
    IJth(jmatrix, 106, 71) = 0.0 - k[76]*y[IDX_HII];
    IJth(jmatrix, 106, 72) = 0.0 - k[496]*y[IDX_HII];
    IJth(jmatrix, 106, 74) = 0.0 + k[131]*y[IDX_HI];
    IJth(jmatrix, 106, 75) = 0.0 + k[129]*y[IDX_HI];
    IJth(jmatrix, 106, 77) = 0.0 + k[1156];
    IJth(jmatrix, 106, 80) = 0.0 - k[84]*y[IDX_HII];
    IJth(jmatrix, 106, 81) = 0.0 + k[1111];
    IJth(jmatrix, 106, 88) = 0.0 - k[81]*y[IDX_HII];
    IJth(jmatrix, 106, 89) = 0.0 + k[1108];
    IJth(jmatrix, 106, 90) = 0.0 - k[75]*y[IDX_HII] - k[491]*y[IDX_HII];
    IJth(jmatrix, 106, 92) = 0.0 - k[86]*y[IDX_HII];
    IJth(jmatrix, 106, 95) = 0.0 - k[79]*y[IDX_HII] - k[497]*y[IDX_HII] -
                        k[498]*y[IDX_HII];
    IJth(jmatrix, 106, 96) = 0.0 - k[82]*y[IDX_HII] - k[500]*y[IDX_HII] -
                        k[501]*y[IDX_HII];
    IJth(jmatrix, 106, 97) = 0.0 + k[130]*y[IDX_HI] + k[536]*y[IDX_H2I] +
                        k[661]*y[IDX_CH4I] + k[674]*y[IDX_H2OI] +
                        k[687]*y[IDX_HNOI];
    IJth(jmatrix, 106, 98) = 0.0 - k[78]*y[IDX_HII];
    IJth(jmatrix, 106, 99) = 0.0 + k[1146];
    IJth(jmatrix, 106, 100) = 0.0 - k[1198]*y[IDX_HII];
    IJth(jmatrix, 106, 101) = 0.0 - k[87]*y[IDX_HII];
    IJth(jmatrix, 106, 103) = 0.0 - k[90]*y[IDX_HII];
    IJth(jmatrix, 106, 104) = 0.0 - k[88]*y[IDX_HII];
    IJth(jmatrix, 106, 106) = 0.0 - k[1]*y[IDX_HNCI] + k[1]*y[IDX_HNCI] -
                        k[75]*y[IDX_CH2I] - k[76]*y[IDX_CH3I] -
                        k[77]*y[IDX_CH4I] - k[78]*y[IDX_CHI] -
                        k[79]*y[IDX_H2COI] - k[80]*y[IDX_H2OI] -
                        k[81]*y[IDX_HCNI] - k[82]*y[IDX_HCOI] - k[83]*y[IDX_MgI]
                        - k[84]*y[IDX_NH2I] - k[85]*y[IDX_NH3I] -
                        k[86]*y[IDX_NHI] - k[87]*y[IDX_NOI] - k[88]*y[IDX_O2I] -
                        k[89]*y[IDX_OI] - k[90]*y[IDX_OHI] - k[91]*y[IDX_SiI] -
                        k[92]*y[IDX_SiC2I] - k[93]*y[IDX_SiC3I] -
                        k[94]*y[IDX_SiCI] - k[95]*y[IDX_SiH2I] -
                        k[96]*y[IDX_SiH3I] - k[97]*y[IDX_SiH4I] -
                        k[98]*y[IDX_SiHI] - k[99]*y[IDX_SiOI] -
                        k[491]*y[IDX_CH2I] - k[492]*y[IDX_CH3OHI] -
                        k[493]*y[IDX_CH3OHI] - k[494]*y[IDX_CH3OHI] -
                        k[495]*y[IDX_CH4I] - k[496]*y[IDX_CO2I] -
                        k[497]*y[IDX_H2COI] - k[498]*y[IDX_H2COI] -
                        k[499]*y[IDX_H2SiOI] - k[500]*y[IDX_HCOI] -
                        k[501]*y[IDX_HCOI] - k[502]*y[IDX_HNCOI] -
                        k[503]*y[IDX_HNOI] - k[504]*y[IDX_NO2I] -
                        k[505]*y[IDX_SiH2I] - k[506]*y[IDX_SiH3I] -
                        k[507]*y[IDX_SiH4I] - k[508]*y[IDX_SiHI] -
                        k[1197]*y[IDX_HI] - k[1198]*y[IDX_HeI] -
                        k[1216]*y[IDX_EM];
    IJth(jmatrix, 106, 108) = 0.0 - k[80]*y[IDX_HII] + k[674]*y[IDX_HeII];
    IJth(jmatrix, 106, 109) = 0.0 - k[89]*y[IDX_HII];
    IJth(jmatrix, 106, 110) = 0.0 - k[1216]*y[IDX_HII];
    IJth(jmatrix, 106, 112) = 0.0 + k[233] + k[536]*y[IDX_HeII];
    IJth(jmatrix, 106, 113) = 0.0 + k[126]*y[IDX_CNII] + k[127]*y[IDX_COII] +
                        k[128]*y[IDX_H2II] + k[129]*y[IDX_HCNII] +
                        k[130]*y[IDX_HeII] + k[131]*y[IDX_OII] + k[236] + k[258]
                        - k[1197]*y[IDX_HII];
    IJth(jmatrix, 107, 29) = 0.0 + k[5]*y[IDX_H2I];
    IJth(jmatrix, 107, 35) = 0.0 + k[489]*y[IDX_COI];
    IJth(jmatrix, 107, 43) = 0.0 - k[637]*y[IDX_HCOII];
    IJth(jmatrix, 107, 46) = 0.0 - k[638]*y[IDX_HCOII];
    IJth(jmatrix, 107, 47) = 0.0 - k[639]*y[IDX_HCOII];
    IJth(jmatrix, 107, 49) = 0.0 + k[138]*y[IDX_HCOI] + k[478]*y[IDX_CHI];
    IJth(jmatrix, 107, 50) = 0.0 + k[485]*y[IDX_COI] + k[824]*y[IDX_OI];
    IJth(jmatrix, 107, 52) = 0.0 + k[494]*y[IDX_HII];
    IJth(jmatrix, 107, 53) = 0.0 - k[149]*y[IDX_HCOII];
    IJth(jmatrix, 107, 55) = 0.0 + k[448]*y[IDX_COI];
    IJth(jmatrix, 107, 56) = 0.0 - k[640]*y[IDX_HCOII];
    IJth(jmatrix, 107, 57) = 0.0 + k[66]*y[IDX_HCOI] + k[479]*y[IDX_H2COI] +
                        k[561]*y[IDX_H2OI];
    IJth(jmatrix, 107, 59) = 0.0 + k[487]*y[IDX_COI];
    IJth(jmatrix, 107, 60) = 0.0 + k[488]*y[IDX_COI];
    IJth(jmatrix, 107, 62) = 0.0 - k[861]*y[IDX_HCOII];
    IJth(jmatrix, 107, 63) = 0.0 - k[648]*y[IDX_HCOII];
    IJth(jmatrix, 107, 64) = 0.0 + k[486]*y[IDX_COI];
    IJth(jmatrix, 107, 65) = 0.0 + k[171]*y[IDX_HCOI] + k[730]*y[IDX_H2COI];
    IJth(jmatrix, 107, 67) = 0.0 + k[451]*y[IDX_COII];
    IJth(jmatrix, 107, 68) = 0.0 + k[71]*y[IDX_HCOI] + k[422]*y[IDX_CH2I] +
                        k[451]*y[IDX_CH4I] + k[458]*y[IDX_CHI] +
                        k[484]*y[IDX_H2COI] + k[532]*y[IDX_H2I] +
                        k[562]*y[IDX_H2OI] + k[779]*y[IDX_NH2I] +
                        k[792]*y[IDX_NH3I] + k[795]*y[IDX_NHI] +
                        k[851]*y[IDX_OHI];
    IJth(jmatrix, 107, 69) = 0.0 + k[108]*y[IDX_HCOI] + k[515]*y[IDX_COI] +
                        k[517]*y[IDX_H2COI];
    IJth(jmatrix, 107, 70) = 0.0 + k[792]*y[IDX_COII];
    IJth(jmatrix, 107, 72) = 0.0 + k[398]*y[IDX_CHII] + k[496]*y[IDX_HII];
    IJth(jmatrix, 107, 73) = 0.0 + k[162]*y[IDX_HCOI] + k[721]*y[IDX_H2COI];
    IJth(jmatrix, 107, 74) = 0.0 + k[211]*y[IDX_HCOI] + k[813]*y[IDX_H2COI] +
                        k[814]*y[IDX_HCNI];
    IJth(jmatrix, 107, 75) = 0.0 + k[621]*y[IDX_COI];
    IJth(jmatrix, 107, 76) = 0.0 + k[180]*y[IDX_HCOI];
    IJth(jmatrix, 107, 77) = 0.0 + k[751]*y[IDX_COI] + k[753]*y[IDX_H2COI];
    IJth(jmatrix, 107, 78) = 0.0 + k[137]*y[IDX_HCOI] + k[473]*y[IDX_CHI] +
                        k[551]*y[IDX_H2COI];
    IJth(jmatrix, 107, 79) = 0.0 + k[46]*y[IDX_HCOI] + k[440]*y[IDX_H2COI] +
                        k[444]*y[IDX_OI];
    IJth(jmatrix, 107, 80) = 0.0 + k[779]*y[IDX_COII] - k[787]*y[IDX_HCOII];
    IJth(jmatrix, 107, 81) = 0.0 + k[417]*y[IDX_H2COI] + k[420]*y[IDX_O2I] +
                        k[421]*y[IDX_OI];
    IJth(jmatrix, 107, 82) = 0.0 + k[118]*y[IDX_HCOI] + k[553]*y[IDX_COI];
    IJth(jmatrix, 107, 83) = 0.0 + k[189]*y[IDX_HCOI];
    IJth(jmatrix, 107, 85) = 0.0 + k[384]*y[IDX_CI];
    IJth(jmatrix, 107, 87) = 0.0 + k[17]*y[IDX_HCOI] + k[369]*y[IDX_H2COI] +
                        k[370]*y[IDX_H2OI];
    IJth(jmatrix, 107, 88) = 0.0 - k[629]*y[IDX_HCOII] + k[814]*y[IDX_OII];
    IJth(jmatrix, 107, 89) = 0.0 + k[31]*y[IDX_HCOI] + k[398]*y[IDX_CO2I] +
                        k[401]*y[IDX_H2COI] + k[404]*y[IDX_H2OI] +
                        k[412]*y[IDX_O2I];
    IJth(jmatrix, 107, 90) = 0.0 + k[422]*y[IDX_COII] - k[429]*y[IDX_HCOII];
    IJth(jmatrix, 107, 91) = 0.0 + k[136]*y[IDX_HCOI] + k[549]*y[IDX_O2I];
    IJth(jmatrix, 107, 92) = 0.0 + k[795]*y[IDX_COII] - k[799]*y[IDX_HCOII];
    IJth(jmatrix, 107, 93) = 0.0 + k[221]*y[IDX_HCOI] + k[838]*y[IDX_COI];
    IJth(jmatrix, 107, 95) = 0.0 + k[369]*y[IDX_CII] + k[401]*y[IDX_CHII] +
                        k[417]*y[IDX_CH2II] + k[440]*y[IDX_CH3II] +
                        k[479]*y[IDX_CNII] + k[484]*y[IDX_COII] +
                        k[498]*y[IDX_HII] + k[517]*y[IDX_H2II] +
                        k[551]*y[IDX_O2II] - k[635]*y[IDX_HCOII] +
                        k[671]*y[IDX_HeII] + k[721]*y[IDX_NII] +
                        k[730]*y[IDX_N2II] + k[753]*y[IDX_NHII] +
                        k[813]*y[IDX_OII] + k[1139];
    IJth(jmatrix, 107, 96) = 0.0 + k[17]*y[IDX_CII] + k[31]*y[IDX_CHII] +
                        k[46]*y[IDX_CH3II] + k[66]*y[IDX_CNII] +
                        k[71]*y[IDX_COII] + k[82]*y[IDX_HII] +
                        k[108]*y[IDX_H2II] + k[118]*y[IDX_H2OII] +
                        k[136]*y[IDX_H2COII] + k[137]*y[IDX_O2II] +
                        k[138]*y[IDX_SiOII] + k[162]*y[IDX_NII] +
                        k[171]*y[IDX_N2II] + k[180]*y[IDX_NH2II] +
                        k[189]*y[IDX_NH3II] + k[211]*y[IDX_OII] +
                        k[221]*y[IDX_OHII] + k[261] - k[636]*y[IDX_HCOII] +
                        k[1150];
    IJth(jmatrix, 107, 97) = 0.0 + k[671]*y[IDX_H2COI];
    IJth(jmatrix, 107, 98) = 0.0 + k[0]*y[IDX_OI] + k[458]*y[IDX_COII] -
                        k[466]*y[IDX_HCOII] + k[473]*y[IDX_O2II] +
                        k[478]*y[IDX_SiOII];
    IJth(jmatrix, 107, 99) = 0.0 + k[583]*y[IDX_COI];
    IJth(jmatrix, 107, 103) = 0.0 + k[851]*y[IDX_COII] - k[854]*y[IDX_HCOII] -
                        k[855]*y[IDX_HCOII];
    IJth(jmatrix, 107, 104) = 0.0 + k[412]*y[IDX_CHII] + k[420]*y[IDX_CH2II] +
                        k[549]*y[IDX_H2COII];
    IJth(jmatrix, 107, 105) = 0.0 + k[384]*y[IDX_H3OII] - k[386]*y[IDX_HCOII];
    IJth(jmatrix, 107, 106) = 0.0 + k[82]*y[IDX_HCOI] + k[494]*y[IDX_CH3OHI] +
                        k[496]*y[IDX_CO2I] + k[498]*y[IDX_H2COI];
    IJth(jmatrix, 107, 107) = 0.0 - k[149]*y[IDX_MgI] - k[330]*y[IDX_EM] -
                        k[386]*y[IDX_CI] - k[429]*y[IDX_CH2I] -
                        k[466]*y[IDX_CHI] - k[566]*y[IDX_H2OI] -
                        k[629]*y[IDX_HCNI] - k[635]*y[IDX_H2COI] -
                        k[636]*y[IDX_HCOI] - k[637]*y[IDX_SiH2I] -
                        k[638]*y[IDX_SiH4I] - k[639]*y[IDX_SiHI] -
                        k[640]*y[IDX_SiOI] - k[648]*y[IDX_HNCI] -
                        k[787]*y[IDX_NH2I] - k[799]*y[IDX_NHI] -
                        k[854]*y[IDX_OHI] - k[855]*y[IDX_OHI] -
                        k[861]*y[IDX_SiI] - k[1148] - k[1281];
    IJth(jmatrix, 107, 108) = 0.0 + k[370]*y[IDX_CII] + k[404]*y[IDX_CHII] +
                        k[561]*y[IDX_CNII] + k[562]*y[IDX_COII] -
                        k[566]*y[IDX_HCOII];
    IJth(jmatrix, 107, 109) = 0.0 + k[0]*y[IDX_CHI] + k[421]*y[IDX_CH2II] +
                        k[444]*y[IDX_CH3II] + k[824]*y[IDX_HCO2II];
    IJth(jmatrix, 107, 110) = 0.0 - k[330]*y[IDX_HCOII];
    IJth(jmatrix, 107, 111) = 0.0 + k[448]*y[IDX_CH4II] + k[485]*y[IDX_HCO2II] +
                        k[486]*y[IDX_HNOII] + k[487]*y[IDX_N2HII] +
                        k[488]*y[IDX_O2HII] + k[489]*y[IDX_SiH4II] +
                        k[515]*y[IDX_H2II] + k[553]*y[IDX_H2OII] +
                        k[583]*y[IDX_H3II] + k[621]*y[IDX_HCNII] +
                        k[751]*y[IDX_NHII] + k[838]*y[IDX_OHII];
    IJth(jmatrix, 107, 112) = 0.0 + k[5]*y[IDX_HOCII] + k[532]*y[IDX_COII];
    IJth(jmatrix, 108, 20) = 0.0 + k[1315] + k[1316] + k[1317] + k[1318];
    IJth(jmatrix, 108, 34) = 0.0 - k[575]*y[IDX_H2OI];
    IJth(jmatrix, 108, 35) = 0.0 - k[574]*y[IDX_H2OI];
    IJth(jmatrix, 108, 37) = 0.0 + k[990]*y[IDX_HI] + k[1100]*y[IDX_OHI];
    IJth(jmatrix, 108, 43) = 0.0 + k[611]*y[IDX_H3OII];
    IJth(jmatrix, 108, 45) = 0.0 - k[573]*y[IDX_H2OI];
    IJth(jmatrix, 108, 47) = 0.0 + k[612]*y[IDX_H3OII];
    IJth(jmatrix, 108, 50) = 0.0 - k[567]*y[IDX_H2OI];
    IJth(jmatrix, 108, 51) = 0.0 + k[1097]*y[IDX_OHI];
    IJth(jmatrix, 108, 52) = 0.0 + k[492]*y[IDX_HII] + k[579]*y[IDX_H3II] +
                        k[808]*y[IDX_OII];
    IJth(jmatrix, 108, 53) = 0.0 + k[119]*y[IDX_H2OII];
    IJth(jmatrix, 108, 55) = 0.0 - k[450]*y[IDX_H2OI];
    IJth(jmatrix, 108, 56) = 0.0 + k[613]*y[IDX_H3OII];
    IJth(jmatrix, 108, 57) = 0.0 - k[560]*y[IDX_H2OI] - k[561]*y[IDX_H2OI];
    IJth(jmatrix, 108, 59) = 0.0 - k[570]*y[IDX_H2OI];
    IJth(jmatrix, 108, 60) = 0.0 - k[571]*y[IDX_H2OI];
    IJth(jmatrix, 108, 61) = 0.0 - k[572]*y[IDX_H2OI];
    IJth(jmatrix, 108, 62) = 0.0 + k[122]*y[IDX_H2OII] + k[610]*y[IDX_H3OII];
    IJth(jmatrix, 108, 63) = 0.0 + k[609]*y[IDX_H3OII];
    IJth(jmatrix, 108, 64) = 0.0 - k[568]*y[IDX_H2OI];
    IJth(jmatrix, 108, 65) = 0.0 - k[125]*y[IDX_H2OI] - k[569]*y[IDX_H2OI];
    IJth(jmatrix, 108, 66) = 0.0 + k[318]*y[IDX_EM] - k[564]*y[IDX_H2OI];
    IJth(jmatrix, 108, 67) = 0.0 + k[922]*y[IDX_OHI];
    IJth(jmatrix, 108, 68) = 0.0 - k[123]*y[IDX_H2OI] - k[562]*y[IDX_H2OI];
    IJth(jmatrix, 108, 69) = 0.0 - k[106]*y[IDX_H2OI] - k[518]*y[IDX_H2OI];
    IJth(jmatrix, 108, 70) = 0.0 + k[195]*y[IDX_H2OII] + k[1098]*y[IDX_OHI];
    IJth(jmatrix, 108, 71) = 0.0 - k[904]*y[IDX_H2OI] + k[910]*y[IDX_NOI] +
                        k[912]*y[IDX_O2I] + k[919]*y[IDX_OHI];
    IJth(jmatrix, 108, 73) = 0.0 - k[160]*y[IDX_H2OI];
    IJth(jmatrix, 108, 74) = 0.0 - k[210]*y[IDX_H2OI] + k[808]*y[IDX_CH3OHI];
    IJth(jmatrix, 108, 75) = 0.0 - k[124]*y[IDX_H2OI] - k[565]*y[IDX_H2OI];
    IJth(jmatrix, 108, 76) = 0.0 - k[771]*y[IDX_H2OI] - k[772]*y[IDX_H2OI];
    IJth(jmatrix, 108, 77) = 0.0 - k[176]*y[IDX_H2OI] - k[754]*y[IDX_H2OI] -
                        k[755]*y[IDX_H2OI] - k[756]*y[IDX_H2OI] -
                        k[757]*y[IDX_H2OI];
    IJth(jmatrix, 108, 80) = 0.0 + k[185]*y[IDX_H2OII] + k[783]*y[IDX_H3OII] +
                        k[1029]*y[IDX_NOI] + k[1031]*y[IDX_OHI];
    IJth(jmatrix, 108, 81) = 0.0 - k[418]*y[IDX_H2OI];
    IJth(jmatrix, 108, 82) = 0.0 + k[40]*y[IDX_CH2I] + k[56]*y[IDX_CHI] +
                        k[117]*y[IDX_H2COI] + k[118]*y[IDX_HCOI] +
                        k[119]*y[IDX_MgI] + k[120]*y[IDX_NOI] +
                        k[121]*y[IDX_O2I] + k[122]*y[IDX_SiI] +
                        k[185]*y[IDX_NH2I] + k[195]*y[IDX_NH3I] -
                        k[555]*y[IDX_H2OI];
    IJth(jmatrix, 108, 85) = 0.0 + k[322]*y[IDX_EM] + k[425]*y[IDX_CH2I] +
                        k[462]*y[IDX_CHI] + k[607]*y[IDX_H2COI] +
                        k[608]*y[IDX_HCNI] + k[609]*y[IDX_HNCI] +
                        k[610]*y[IDX_SiI] + k[611]*y[IDX_SiH2I] +
                        k[612]*y[IDX_SiHI] + k[613]*y[IDX_SiOI] +
                        k[783]*y[IDX_NH2I];
    IJth(jmatrix, 108, 87) = 0.0 - k[370]*y[IDX_H2OI] - k[371]*y[IDX_H2OI];
    IJth(jmatrix, 108, 88) = 0.0 + k[608]*y[IDX_H3OII] + k[1094]*y[IDX_OHI];
    IJth(jmatrix, 108, 89) = 0.0 - k[402]*y[IDX_H2OI] - k[403]*y[IDX_H2OI] -
                        k[404]*y[IDX_H2OI];
    IJth(jmatrix, 108, 90) = 0.0 + k[40]*y[IDX_H2OII] + k[425]*y[IDX_H3OII] +
                        k[891]*y[IDX_O2I] + k[899]*y[IDX_OHI];
    IJth(jmatrix, 108, 91) = 0.0 - k[563]*y[IDX_H2OI];
    IJth(jmatrix, 108, 92) = 0.0 - k[1036]*y[IDX_H2OI] + k[1048]*y[IDX_OHI];
    IJth(jmatrix, 108, 93) = 0.0 - k[220]*y[IDX_H2OI] - k[840]*y[IDX_H2OI];
    IJth(jmatrix, 108, 95) = 0.0 + k[117]*y[IDX_H2OII] + k[607]*y[IDX_H3OII] +
                        k[1093]*y[IDX_OHI];
    IJth(jmatrix, 108, 96) = 0.0 + k[118]*y[IDX_H2OII] + k[1096]*y[IDX_OHI];
    IJth(jmatrix, 108, 97) = 0.0 - k[143]*y[IDX_H2OI] - k[673]*y[IDX_H2OI] -
                        k[674]*y[IDX_H2OI];
    IJth(jmatrix, 108, 98) = 0.0 + k[56]*y[IDX_H2OII] + k[462]*y[IDX_H3OII];
    IJth(jmatrix, 108, 99) = 0.0 + k[579]*y[IDX_CH3OHI] - k[586]*y[IDX_H2OI];
    IJth(jmatrix, 108, 101) = 0.0 + k[120]*y[IDX_H2OII] + k[910]*y[IDX_CH3I] +
                        k[1029]*y[IDX_NH2I];
    IJth(jmatrix, 108, 103) = 0.0 + k[899]*y[IDX_CH2I] + k[919]*y[IDX_CH3I] +
                        k[922]*y[IDX_CH4I] + k[966]*y[IDX_H2I] +
                        k[1031]*y[IDX_NH2I] + k[1048]*y[IDX_NHI] +
                        k[1093]*y[IDX_H2COI] + k[1094]*y[IDX_HCNI] +
                        k[1096]*y[IDX_HCOI] + k[1097]*y[IDX_HNOI] +
                        k[1098]*y[IDX_NH3I] + k[1100]*y[IDX_O2HI] +
                        k[1101]*y[IDX_OHI] + k[1101]*y[IDX_OHI] +
                        k[1208]*y[IDX_HI];
    IJth(jmatrix, 108, 104) = 0.0 + k[121]*y[IDX_H2OII] + k[891]*y[IDX_CH2I] +
                        k[912]*y[IDX_CH3I];
    IJth(jmatrix, 108, 106) = 0.0 - k[80]*y[IDX_H2OI] + k[492]*y[IDX_CH3OHI];
    IJth(jmatrix, 108, 107) = 0.0 - k[566]*y[IDX_H2OI];
    IJth(jmatrix, 108, 108) = 0.0 - k[4]*y[IDX_H2I] - k[11]*y[IDX_HI] -
                        k[80]*y[IDX_HII] - k[106]*y[IDX_H2II] -
                        k[123]*y[IDX_COII] - k[124]*y[IDX_HCNII] -
                        k[125]*y[IDX_N2II] - k[143]*y[IDX_HeII] -
                        k[160]*y[IDX_NII] - k[176]*y[IDX_NHII] -
                        k[210]*y[IDX_OII] - k[220]*y[IDX_OHII] - k[256] -
                        k[370]*y[IDX_CII] - k[371]*y[IDX_CII] -
                        k[402]*y[IDX_CHII] - k[403]*y[IDX_CHII] -
                        k[404]*y[IDX_CHII] - k[418]*y[IDX_CH2II] -
                        k[450]*y[IDX_CH4II] - k[518]*y[IDX_H2II] -
                        k[555]*y[IDX_H2OII] - k[560]*y[IDX_CNII] -
                        k[561]*y[IDX_CNII] - k[562]*y[IDX_COII] -
                        k[563]*y[IDX_H2COII] - k[564]*y[IDX_H3COII] -
                        k[565]*y[IDX_HCNII] - k[566]*y[IDX_HCOII] -
                        k[567]*y[IDX_HCO2II] - k[568]*y[IDX_HNOII] -
                        k[569]*y[IDX_N2II] - k[570]*y[IDX_N2HII] -
                        k[571]*y[IDX_O2HII] - k[572]*y[IDX_SiII] -
                        k[573]*y[IDX_SiHII] - k[574]*y[IDX_SiH4II] -
                        k[575]*y[IDX_SiH5II] - k[586]*y[IDX_H3II] -
                        k[673]*y[IDX_HeII] - k[674]*y[IDX_HeII] -
                        k[754]*y[IDX_NHII] - k[755]*y[IDX_NHII] -
                        k[756]*y[IDX_NHII] - k[757]*y[IDX_NHII] -
                        k[771]*y[IDX_NH2II] - k[772]*y[IDX_NH2II] -
                        k[840]*y[IDX_OHII] - k[904]*y[IDX_CH3I] -
                        k[975]*y[IDX_HI] - k[1036]*y[IDX_NHI] -
                        k[1062]*y[IDX_OI] - k[1141] - k[1142] - k[1257];
    IJth(jmatrix, 108, 109) = 0.0 - k[1062]*y[IDX_H2OI];
    IJth(jmatrix, 108, 110) = 0.0 + k[318]*y[IDX_H3COII] + k[322]*y[IDX_H3OII];
    IJth(jmatrix, 108, 112) = 0.0 - k[4]*y[IDX_H2OI] + k[966]*y[IDX_OHI];
    IJth(jmatrix, 108, 113) = 0.0 - k[11]*y[IDX_H2OI] - k[975]*y[IDX_H2OI] +
                        k[990]*y[IDX_O2HI] + k[1208]*y[IDX_OHI];
    IJth(jmatrix, 109, 23) = 0.0 - k[1060]*y[IDX_OI];
    IJth(jmatrix, 109, 32) = 0.0 - k[1081]*y[IDX_OI];
    IJth(jmatrix, 109, 33) = 0.0 - k[1082]*y[IDX_OI];
    IJth(jmatrix, 109, 36) = 0.0 - k[831]*y[IDX_OI];
    IJth(jmatrix, 109, 37) = 0.0 + k[990]*y[IDX_HI] - k[1077]*y[IDX_OI] + k[1171];
    IJth(jmatrix, 109, 38) = 0.0 - k[1083]*y[IDX_OI] - k[1084]*y[IDX_OI];
    IJth(jmatrix, 109, 39) = 0.0 + k[276] + k[1019]*y[IDX_NI] + k[1019]*y[IDX_NI] -
                        k[1075]*y[IDX_OI] + k[1164];
    IJth(jmatrix, 109, 40) = 0.0 - k[834]*y[IDX_OI];
    IJth(jmatrix, 109, 41) = 0.0 - k[833]*y[IDX_OI];
    IJth(jmatrix, 109, 42) = 0.0 + k[283] + k[697]*y[IDX_HeII] + k[993]*y[IDX_HI] -
                        k[1078]*y[IDX_OI] - k[1079]*y[IDX_OI] + k[1172];
    IJth(jmatrix, 109, 43) = 0.0 - k[1085]*y[IDX_OI] - k[1086]*y[IDX_OI];
    IJth(jmatrix, 109, 45) = 0.0 - k[832]*y[IDX_OI];
    IJth(jmatrix, 109, 46) = 0.0 - k[1088]*y[IDX_OI];
    IJth(jmatrix, 109, 47) = 0.0 + k[849]*y[IDX_OHII] - k[1089]*y[IDX_OI];
    IJth(jmatrix, 109, 48) = 0.0 - k[1087]*y[IDX_OI];
    IJth(jmatrix, 109, 49) = 0.0 + k[362]*y[IDX_EM] - k[835]*y[IDX_OI] + k[1189];
    IJth(jmatrix, 109, 50) = 0.0 + k[332]*y[IDX_EM] - k[824]*y[IDX_OI];
    IJth(jmatrix, 109, 51) = 0.0 + k[980]*y[IDX_HI] - k[1068]*y[IDX_OI] -
                        k[1069]*y[IDX_OI] - k[1070]*y[IDX_OI];
    IJth(jmatrix, 109, 55) = 0.0 - k[822]*y[IDX_OI];
    IJth(jmatrix, 109, 56) = 0.0 + k[293] + k[710]*y[IDX_HeII] + k[850]*y[IDX_OHII] +
                        k[1190];
    IJth(jmatrix, 109, 57) = 0.0 - k[216]*y[IDX_OI];
    IJth(jmatrix, 109, 59) = 0.0 - k[826]*y[IDX_OI];
    IJth(jmatrix, 109, 60) = 0.0 - k[829]*y[IDX_OI];
    IJth(jmatrix, 109, 61) = 0.0 - k[1212]*y[IDX_OI];
    IJth(jmatrix, 109, 62) = 0.0 + k[848]*y[IDX_OHII] + k[1106]*y[IDX_O2I] -
                        k[1213]*y[IDX_OI];
    IJth(jmatrix, 109, 63) = 0.0 + k[844]*y[IDX_OHII];
    IJth(jmatrix, 109, 65) = 0.0 - k[218]*y[IDX_OI] - k[825]*y[IDX_OI];
    IJth(jmatrix, 109, 67) = 0.0 + k[207]*y[IDX_OII] - k[1056]*y[IDX_OI];
    IJth(jmatrix, 109, 68) = 0.0 - k[217]*y[IDX_OI] + k[304]*y[IDX_EM] +
                        k[851]*y[IDX_OHI] + k[1131];
    IJth(jmatrix, 109, 69) = 0.0 - k[526]*y[IDX_OI];
    IJth(jmatrix, 109, 70) = 0.0 + k[213]*y[IDX_OII] - k[1074]*y[IDX_OI];
    IJth(jmatrix, 109, 71) = 0.0 - k[915]*y[IDX_OI] - k[916]*y[IDX_OI] +
                        k[917]*y[IDX_OHI];
    IJth(jmatrix, 109, 72) = 0.0 + k[252] + k[496]*y[IDX_HII] + k[665]*y[IDX_HeII] +
                        k[837]*y[IDX_OHII] - k[1059]*y[IDX_OI] + k[1132];
    IJth(jmatrix, 109, 73) = 0.0 + k[727]*y[IDX_NOI] + k[728]*y[IDX_O2I];
    IJth(jmatrix, 109, 74) = 0.0 + k[43]*y[IDX_CH2I] + k[60]*y[IDX_CHI] +
                        k[131]*y[IDX_HI] + k[202]*y[IDX_NHI] +
                        k[207]*y[IDX_CH4I] + k[208]*y[IDX_COI] +
                        k[209]*y[IDX_H2COI] + k[210]*y[IDX_H2OI] +
                        k[211]*y[IDX_HCOI] + k[212]*y[IDX_NH2I] +
                        k[213]*y[IDX_NH3I] + k[214]*y[IDX_O2I] +
                        k[215]*y[IDX_OHI] + k[1221]*y[IDX_EM];
    IJth(jmatrix, 109, 76) = 0.0 + k[777]*y[IDX_O2I] - k[827]*y[IDX_OI];
    IJth(jmatrix, 109, 77) = 0.0 + k[756]*y[IDX_H2OI] + k[764]*y[IDX_NOI] -
                        k[767]*y[IDX_OI];
    IJth(jmatrix, 109, 78) = 0.0 + k[346]*y[IDX_EM] + k[346]*y[IDX_EM] +
                        k[391]*y[IDX_CI] + k[435]*y[IDX_CH2I] +
                        k[473]*y[IDX_CHI] + k[742]*y[IDX_NI] + k[804]*y[IDX_NHI]
                        + k[1167];
    IJth(jmatrix, 109, 79) = 0.0 + k[442]*y[IDX_O2I] - k[443]*y[IDX_OI] -
                        k[444]*y[IDX_OI];
    IJth(jmatrix, 109, 80) = 0.0 + k[212]*y[IDX_OII] + k[791]*y[IDX_OHII] +
                        k[1032]*y[IDX_OHI] - k[1072]*y[IDX_OI] -
                        k[1073]*y[IDX_OI];
    IJth(jmatrix, 109, 81) = 0.0 - k[421]*y[IDX_OI];
    IJth(jmatrix, 109, 82) = 0.0 + k[312]*y[IDX_EM] + k[313]*y[IDX_EM] -
                        k[823]*y[IDX_OI] + k[852]*y[IDX_OHI];
    IJth(jmatrix, 109, 83) = 0.0 - k[828]*y[IDX_OI];
    IJth(jmatrix, 109, 84) = 0.0 + k[345]*y[IDX_EM];
    IJth(jmatrix, 109, 85) = 0.0 + k[323]*y[IDX_EM];
    IJth(jmatrix, 109, 86) = 0.0 + k[845]*y[IDX_OHII] - k[1071]*y[IDX_OI];
    IJth(jmatrix, 109, 87) = 0.0 + k[376]*y[IDX_O2I] - k[1193]*y[IDX_OI];
    IJth(jmatrix, 109, 88) = 0.0 + k[841]*y[IDX_OHII] - k[1063]*y[IDX_OI] -
                        k[1064]*y[IDX_OI] - k[1065]*y[IDX_OI];
    IJth(jmatrix, 109, 89) = 0.0 + k[412]*y[IDX_O2I] - k[414]*y[IDX_OI];
    IJth(jmatrix, 109, 90) = 0.0 + k[43]*y[IDX_OII] + k[435]*y[IDX_O2II] +
                        k[437]*y[IDX_OHII] + k[892]*y[IDX_O2I] -
                        k[894]*y[IDX_OI] - k[895]*y[IDX_OI] - k[896]*y[IDX_OI] -
                        k[897]*y[IDX_OI] + k[900]*y[IDX_OHI];
    IJth(jmatrix, 109, 91) = 0.0 + k[306]*y[IDX_EM];
    IJth(jmatrix, 109, 92) = 0.0 + k[202]*y[IDX_OII] + k[804]*y[IDX_O2II] +
                        k[806]*y[IDX_OHII] + k[1042]*y[IDX_NOI] +
                        k[1044]*y[IDX_O2I] - k[1046]*y[IDX_OI] -
                        k[1047]*y[IDX_OI] + k[1050]*y[IDX_OHI];
    IJth(jmatrix, 109, 93) = 0.0 + k[348]*y[IDX_EM] + k[393]*y[IDX_CI] +
                        k[437]*y[IDX_CH2I] + k[475]*y[IDX_CHI] +
                        k[791]*y[IDX_NH2I] + k[806]*y[IDX_NHI] -
                        k[830]*y[IDX_OI] + k[836]*y[IDX_CNI] +
                        k[837]*y[IDX_CO2I] + k[838]*y[IDX_COI] +
                        k[839]*y[IDX_H2COI] + k[840]*y[IDX_H2OI] +
                        k[841]*y[IDX_HCNI] + k[843]*y[IDX_HCOI] +
                        k[844]*y[IDX_HNCI] + k[845]*y[IDX_N2I] +
                        k[846]*y[IDX_NOI] + k[847]*y[IDX_OHI] +
                        k[848]*y[IDX_SiI] + k[849]*y[IDX_SiHI] +
                        k[850]*y[IDX_SiOI];
    IJth(jmatrix, 109, 94) = 0.0 + k[836]*y[IDX_OHII] + k[949]*y[IDX_O2I] -
                        k[1057]*y[IDX_OI] - k[1058]*y[IDX_OI] +
                        k[1090]*y[IDX_OHI];
    IJth(jmatrix, 109, 95) = 0.0 + k[209]*y[IDX_OII] + k[672]*y[IDX_HeII] +
                        k[839]*y[IDX_OHII] - k[1061]*y[IDX_OI];
    IJth(jmatrix, 109, 96) = 0.0 + k[211]*y[IDX_OII] + k[682]*y[IDX_HeII] +
                        k[843]*y[IDX_OHII] + k[978]*y[IDX_HI] +
                        k[1015]*y[IDX_NI] - k[1066]*y[IDX_OI] -
                        k[1067]*y[IDX_OI];
    IJth(jmatrix, 109, 97) = 0.0 + k[665]*y[IDX_CO2I] + k[669]*y[IDX_COI] +
                        k[672]*y[IDX_H2COI] + k[682]*y[IDX_HCOI] +
                        k[695]*y[IDX_NOI] + k[696]*y[IDX_O2I] +
                        k[697]*y[IDX_OCNI] + k[710]*y[IDX_SiOI];
    IJth(jmatrix, 109, 98) = 0.0 - k[0]*y[IDX_OI] + k[60]*y[IDX_OII] +
                        k[473]*y[IDX_O2II] + k[475]*y[IDX_OHII] +
                        k[930]*y[IDX_NOI] + k[934]*y[IDX_O2I] +
                        k[936]*y[IDX_O2I] - k[939]*y[IDX_OI] - k[940]*y[IDX_OI];
    IJth(jmatrix, 109, 99) = 0.0 - k[598]*y[IDX_OI] - k[599]*y[IDX_OI];
    IJth(jmatrix, 109, 101) = 0.0 + k[278] + k[695]*y[IDX_HeII] + k[727]*y[IDX_NII] +
                        k[764]*y[IDX_NHII] + k[846]*y[IDX_OHII] +
                        k[871]*y[IDX_CI] + k[930]*y[IDX_CHI] + k[987]*y[IDX_HI]
                        + k[1022]*y[IDX_NI] + k[1042]*y[IDX_NHI] +
                        k[1052]*y[IDX_O2I] - k[1076]*y[IDX_OI] + k[1166];
    IJth(jmatrix, 109, 102) = 0.0 + k[742]*y[IDX_O2II] + k[1015]*y[IDX_HCOI] +
                        k[1019]*y[IDX_NO2I] + k[1019]*y[IDX_NO2I] +
                        k[1022]*y[IDX_NOI] + k[1023]*y[IDX_O2I] +
                        k[1026]*y[IDX_OHI];
    IJth(jmatrix, 109, 103) = 0.0 + k[7]*y[IDX_H2I] + k[13]*y[IDX_HI] +
                        k[215]*y[IDX_OII] + k[284] + k[847]*y[IDX_OHII] +
                        k[851]*y[IDX_COII] + k[852]*y[IDX_H2OII] +
                        k[876]*y[IDX_CI] + k[900]*y[IDX_CH2I] +
                        k[917]*y[IDX_CH3I] + k[996]*y[IDX_HI] +
                        k[1026]*y[IDX_NI] + k[1032]*y[IDX_NH2I] +
                        k[1050]*y[IDX_NHI] - k[1080]*y[IDX_OI] +
                        k[1090]*y[IDX_CNI] + k[1101]*y[IDX_OHI] +
                        k[1101]*y[IDX_OHI] + k[1174];
    IJth(jmatrix, 109, 104) = 0.0 + k[6]*y[IDX_H2I] + k[6]*y[IDX_H2I] +
                        k[12]*y[IDX_HI] + k[12]*y[IDX_HI] + k[214]*y[IDX_OII] +
                        k[280] + k[280] + k[376]*y[IDX_CII] + k[412]*y[IDX_CHII]
                        + k[442]*y[IDX_CH3II] + k[696]*y[IDX_HeII] +
                        k[728]*y[IDX_NII] + k[777]*y[IDX_NH2II] +
                        k[873]*y[IDX_CI] + k[892]*y[IDX_CH2I] +
                        k[934]*y[IDX_CHI] + k[936]*y[IDX_CHI] +
                        k[949]*y[IDX_CNI] + k[953]*y[IDX_COI] + k[989]*y[IDX_HI]
                        + k[1023]*y[IDX_NI] + k[1044]*y[IDX_NHI] +
                        k[1052]*y[IDX_NOI] + k[1106]*y[IDX_SiI] + k[1169] +
                        k[1169];
    IJth(jmatrix, 109, 105) = 0.0 + k[391]*y[IDX_O2II] + k[393]*y[IDX_OHII] +
                        k[871]*y[IDX_NOI] + k[873]*y[IDX_O2I] +
                        k[876]*y[IDX_OHI] - k[1196]*y[IDX_OI];
    IJth(jmatrix, 109, 106) = 0.0 - k[89]*y[IDX_OI] + k[496]*y[IDX_CO2I];
    IJth(jmatrix, 109, 108) = 0.0 + k[210]*y[IDX_OII] + k[756]*y[IDX_NHII] +
                        k[840]*y[IDX_OHII] - k[1062]*y[IDX_OI];
    IJth(jmatrix, 109, 109) = 0.0 - k[0]*y[IDX_CHI] - k[89]*y[IDX_HII] -
                        k[216]*y[IDX_CNII] - k[217]*y[IDX_COII] -
                        k[218]*y[IDX_N2II] - k[239] - k[282] -
                        k[414]*y[IDX_CHII] - k[421]*y[IDX_CH2II] -
                        k[443]*y[IDX_CH3II] - k[444]*y[IDX_CH3II] -
                        k[526]*y[IDX_H2II] - k[598]*y[IDX_H3II] -
                        k[599]*y[IDX_H3II] - k[767]*y[IDX_NHII] -
                        k[822]*y[IDX_CH4II] - k[823]*y[IDX_H2OII] -
                        k[824]*y[IDX_HCO2II] - k[825]*y[IDX_N2II] -
                        k[826]*y[IDX_N2HII] - k[827]*y[IDX_NH2II] -
                        k[828]*y[IDX_NH3II] - k[829]*y[IDX_O2HII] -
                        k[830]*y[IDX_OHII] - k[831]*y[IDX_SiCII] -
                        k[832]*y[IDX_SiHII] - k[833]*y[IDX_SiH2II] -
                        k[834]*y[IDX_SiH3II] - k[835]*y[IDX_SiOII] -
                        k[894]*y[IDX_CH2I] - k[895]*y[IDX_CH2I] -
                        k[896]*y[IDX_CH2I] - k[897]*y[IDX_CH2I] -
                        k[915]*y[IDX_CH3I] - k[916]*y[IDX_CH3I] -
                        k[939]*y[IDX_CHI] - k[940]*y[IDX_CHI] -
                        k[965]*y[IDX_H2I] - k[1046]*y[IDX_NHI] -
                        k[1047]*y[IDX_NHI] - k[1056]*y[IDX_CH4I] -
                        k[1057]*y[IDX_CNI] - k[1058]*y[IDX_CNI] -
                        k[1059]*y[IDX_CO2I] - k[1060]*y[IDX_H2CNI] -
                        k[1061]*y[IDX_H2COI] - k[1062]*y[IDX_H2OI] -
                        k[1063]*y[IDX_HCNI] - k[1064]*y[IDX_HCNI] -
                        k[1065]*y[IDX_HCNI] - k[1066]*y[IDX_HCOI] -
                        k[1067]*y[IDX_HCOI] - k[1068]*y[IDX_HNOI] -
                        k[1069]*y[IDX_HNOI] - k[1070]*y[IDX_HNOI] -
                        k[1071]*y[IDX_N2I] - k[1072]*y[IDX_NH2I] -
                        k[1073]*y[IDX_NH2I] - k[1074]*y[IDX_NH3I] -
                        k[1075]*y[IDX_NO2I] - k[1076]*y[IDX_NOI] -
                        k[1077]*y[IDX_O2HI] - k[1078]*y[IDX_OCNI] -
                        k[1079]*y[IDX_OCNI] - k[1080]*y[IDX_OHI] -
                        k[1081]*y[IDX_SiC2I] - k[1082]*y[IDX_SiC3I] -
                        k[1083]*y[IDX_SiCI] - k[1084]*y[IDX_SiCI] -
                        k[1085]*y[IDX_SiH2I] - k[1086]*y[IDX_SiH2I] -
                        k[1087]*y[IDX_SiH3I] - k[1088]*y[IDX_SiH4I] -
                        k[1089]*y[IDX_SiHI] - k[1193]*y[IDX_CII] -
                        k[1196]*y[IDX_CI] - k[1207]*y[IDX_HI] -
                        k[1211]*y[IDX_OI] - k[1211]*y[IDX_OI] -
                        k[1211]*y[IDX_OI] - k[1211]*y[IDX_OI] -
                        k[1212]*y[IDX_SiII] - k[1213]*y[IDX_SiI] - k[1291];
    IJth(jmatrix, 109, 110) = 0.0 + k[304]*y[IDX_COII] + k[306]*y[IDX_H2COII] +
                        k[312]*y[IDX_H2OII] + k[313]*y[IDX_H2OII] +
                        k[323]*y[IDX_H3OII] + k[332]*y[IDX_HCO2II] +
                        k[345]*y[IDX_NOII] + k[346]*y[IDX_O2II] +
                        k[346]*y[IDX_O2II] + k[348]*y[IDX_OHII] +
                        k[362]*y[IDX_SiOII] + k[1221]*y[IDX_OII];
    IJth(jmatrix, 109, 111) = 0.0 + k[208]*y[IDX_OII] + k[253] + k[669]*y[IDX_HeII] +
                        k[838]*y[IDX_OHII] + k[953]*y[IDX_O2I] + k[1133];
    IJth(jmatrix, 109, 112) = 0.0 + k[6]*y[IDX_O2I] + k[6]*y[IDX_O2I] +
                        k[7]*y[IDX_OHI] - k[965]*y[IDX_OI];
    IJth(jmatrix, 109, 113) = 0.0 + k[12]*y[IDX_O2I] + k[12]*y[IDX_O2I] +
                        k[13]*y[IDX_OHI] + k[131]*y[IDX_OII] +
                        k[978]*y[IDX_HCOI] + k[980]*y[IDX_HNOI] +
                        k[987]*y[IDX_NOI] + k[989]*y[IDX_O2I] +
                        k[990]*y[IDX_O2HI] + k[993]*y[IDX_OCNI] +
                        k[996]*y[IDX_OHI] - k[1207]*y[IDX_OI];
    IJth(jmatrix, 110, 22) = 0.0 - k[351]*y[IDX_EM];
    IJth(jmatrix, 110, 25) = 0.0 - k[310]*y[IDX_EM] - k[311]*y[IDX_EM];
    IJth(jmatrix, 110, 27) = 0.0 - k[336]*y[IDX_EM];
    IJth(jmatrix, 110, 29) = 0.0 - k[335]*y[IDX_EM];
    IJth(jmatrix, 110, 30) = 0.0 - k[350]*y[IDX_EM];
    IJth(jmatrix, 110, 34) = 0.0 - k[360]*y[IDX_EM] - k[361]*y[IDX_EM];
    IJth(jmatrix, 110, 35) = 0.0 - k[358]*y[IDX_EM] - k[359]*y[IDX_EM];
    IJth(jmatrix, 110, 36) = 0.0 - k[349]*y[IDX_EM];
    IJth(jmatrix, 110, 40) = 0.0 - k[356]*y[IDX_EM] - k[357]*y[IDX_EM];
    IJth(jmatrix, 110, 41) = 0.0 - k[353]*y[IDX_EM] - k[354]*y[IDX_EM] -
                        k[355]*y[IDX_EM];
    IJth(jmatrix, 110, 43) = 0.0 + k[1180];
    IJth(jmatrix, 110, 44) = 0.0 - k[363]*y[IDX_EM] - k[364]*y[IDX_EM];
    IJth(jmatrix, 110, 45) = 0.0 - k[352]*y[IDX_EM];
    IJth(jmatrix, 110, 48) = 0.0 + k[1183];
    IJth(jmatrix, 110, 49) = 0.0 - k[362]*y[IDX_EM];
    IJth(jmatrix, 110, 50) = 0.0 - k[331]*y[IDX_EM] - k[332]*y[IDX_EM] -
                        k[333]*y[IDX_EM];
    IJth(jmatrix, 110, 52) = 0.0 + k[1120];
    IJth(jmatrix, 110, 53) = 0.0 + k[266] + k[1154];
    IJth(jmatrix, 110, 54) = 0.0 - k[1219]*y[IDX_EM];
    IJth(jmatrix, 110, 55) = 0.0 - k[301]*y[IDX_EM] - k[302]*y[IDX_EM];
    IJth(jmatrix, 110, 56) = 0.0 + k[1191];
    IJth(jmatrix, 110, 57) = 0.0 - k[303]*y[IDX_EM];
    IJth(jmatrix, 110, 58) = 0.0 - k[327]*y[IDX_EM] - k[328]*y[IDX_EM] -
                        k[329]*y[IDX_EM];
    IJth(jmatrix, 110, 59) = 0.0 - k[338]*y[IDX_EM] - k[339]*y[IDX_EM];
    IJth(jmatrix, 110, 60) = 0.0 - k[347]*y[IDX_EM];
    IJth(jmatrix, 110, 61) = 0.0 - k[1222]*y[IDX_EM];
    IJth(jmatrix, 110, 62) = 0.0 + k[285] + k[1176];
    IJth(jmatrix, 110, 64) = 0.0 - k[334]*y[IDX_EM];
    IJth(jmatrix, 110, 65) = 0.0 - k[337]*y[IDX_EM];
    IJth(jmatrix, 110, 66) = 0.0 - k[317]*y[IDX_EM] - k[318]*y[IDX_EM] -
                        k[319]*y[IDX_EM] - k[320]*y[IDX_EM] - k[321]*y[IDX_EM];
    IJth(jmatrix, 110, 67) = 0.0 + k[1126];
    IJth(jmatrix, 110, 68) = 0.0 - k[304]*y[IDX_EM];
    IJth(jmatrix, 110, 69) = 0.0 - k[305]*y[IDX_EM];
    IJth(jmatrix, 110, 70) = 0.0 + k[272] + k[1160];
    IJth(jmatrix, 110, 71) = 0.0 + k[245] + k[1117];
    IJth(jmatrix, 110, 73) = 0.0 - k[1220]*y[IDX_EM];
    IJth(jmatrix, 110, 74) = 0.0 - k[1221]*y[IDX_EM];
    IJth(jmatrix, 110, 75) = 0.0 - k[326]*y[IDX_EM];
    IJth(jmatrix, 110, 76) = 0.0 - k[341]*y[IDX_EM] - k[342]*y[IDX_EM];
    IJth(jmatrix, 110, 77) = 0.0 - k[340]*y[IDX_EM];
    IJth(jmatrix, 110, 78) = 0.0 - k[346]*y[IDX_EM];
    IJth(jmatrix, 110, 79) = 0.0 - k[298]*y[IDX_EM] - k[299]*y[IDX_EM] -
                        k[300]*y[IDX_EM] - k[1215]*y[IDX_EM];
    IJth(jmatrix, 110, 80) = 0.0 + k[269] + k[1157];
    IJth(jmatrix, 110, 81) = 0.0 - k[295]*y[IDX_EM] - k[296]*y[IDX_EM] -
                        k[297]*y[IDX_EM];
    IJth(jmatrix, 110, 82) = 0.0 - k[312]*y[IDX_EM] - k[313]*y[IDX_EM] -
                        k[314]*y[IDX_EM];
    IJth(jmatrix, 110, 83) = 0.0 - k[343]*y[IDX_EM] - k[344]*y[IDX_EM];
    IJth(jmatrix, 110, 84) = 0.0 - k[345]*y[IDX_EM];
    IJth(jmatrix, 110, 85) = 0.0 - k[322]*y[IDX_EM] - k[323]*y[IDX_EM] -
                        k[324]*y[IDX_EM] - k[325]*y[IDX_EM];
    IJth(jmatrix, 110, 87) = 0.0 - k[1214]*y[IDX_EM];
    IJth(jmatrix, 110, 89) = 0.0 - k[294]*y[IDX_EM];
    IJth(jmatrix, 110, 90) = 0.0 + k[242] + k[1112];
    IJth(jmatrix, 110, 91) = 0.0 - k[306]*y[IDX_EM] - k[307]*y[IDX_EM] -
                        k[308]*y[IDX_EM] - k[309]*y[IDX_EM] - k[1217]*y[IDX_EM];
    IJth(jmatrix, 110, 92) = 0.0 + k[275] + k[1163];
    IJth(jmatrix, 110, 93) = 0.0 - k[348]*y[IDX_EM];
    IJth(jmatrix, 110, 95) = 0.0 + k[1138] + k[1139];
    IJth(jmatrix, 110, 96) = 0.0 + k[261] + k[1150];
    IJth(jmatrix, 110, 97) = 0.0 - k[1218]*y[IDX_EM];
    IJth(jmatrix, 110, 98) = 0.0 + k[0]*y[IDX_OI] + k[1129];
    IJth(jmatrix, 110, 99) = 0.0 - k[315]*y[IDX_EM] - k[316]*y[IDX_EM];
    IJth(jmatrix, 110, 100) = 0.0 + k[237] + k[265];
    IJth(jmatrix, 110, 101) = 0.0 + k[277] + k[1165];
    IJth(jmatrix, 110, 102) = 0.0 + k[238] + k[268];
    IJth(jmatrix, 110, 103) = 0.0 + k[1175];
    IJth(jmatrix, 110, 104) = 0.0 + k[279] + k[1168];
    IJth(jmatrix, 110, 105) = 0.0 + k[231] + k[240] + k[1107];
    IJth(jmatrix, 110, 106) = 0.0 - k[1216]*y[IDX_EM];
    IJth(jmatrix, 110, 107) = 0.0 - k[330]*y[IDX_EM];
    IJth(jmatrix, 110, 108) = 0.0 + k[1141];
    IJth(jmatrix, 110, 109) = 0.0 + k[0]*y[IDX_CHI] + k[239] + k[282];
    IJth(jmatrix, 110, 110) = 0.0 - k[8]*y[IDX_H2I] + k[8]*y[IDX_H2I] -
                        k[294]*y[IDX_CHII] - k[295]*y[IDX_CH2II] -
                        k[296]*y[IDX_CH2II] - k[297]*y[IDX_CH2II] -
                        k[298]*y[IDX_CH3II] - k[299]*y[IDX_CH3II] -
                        k[300]*y[IDX_CH3II] - k[301]*y[IDX_CH4II] -
                        k[302]*y[IDX_CH4II] - k[303]*y[IDX_CNII] -
                        k[304]*y[IDX_COII] - k[305]*y[IDX_H2II] -
                        k[306]*y[IDX_H2COII] - k[307]*y[IDX_H2COII] -
                        k[308]*y[IDX_H2COII] - k[309]*y[IDX_H2COII] -
                        k[310]*y[IDX_H2NOII] - k[311]*y[IDX_H2NOII] -
                        k[312]*y[IDX_H2OII] - k[313]*y[IDX_H2OII] -
                        k[314]*y[IDX_H2OII] - k[315]*y[IDX_H3II] -
                        k[316]*y[IDX_H3II] - k[317]*y[IDX_H3COII] -
                        k[318]*y[IDX_H3COII] - k[319]*y[IDX_H3COII] -
                        k[320]*y[IDX_H3COII] - k[321]*y[IDX_H3COII] -
                        k[322]*y[IDX_H3OII] - k[323]*y[IDX_H3OII] -
                        k[324]*y[IDX_H3OII] - k[325]*y[IDX_H3OII] -
                        k[326]*y[IDX_HCNII] - k[327]*y[IDX_HCNHII] -
                        k[328]*y[IDX_HCNHII] - k[329]*y[IDX_HCNHII] -
                        k[330]*y[IDX_HCOII] - k[331]*y[IDX_HCO2II] -
                        k[332]*y[IDX_HCO2II] - k[333]*y[IDX_HCO2II] -
                        k[334]*y[IDX_HNOII] - k[335]*y[IDX_HOCII] -
                        k[336]*y[IDX_HeHII] - k[337]*y[IDX_N2II] -
                        k[338]*y[IDX_N2HII] - k[339]*y[IDX_N2HII] -
                        k[340]*y[IDX_NHII] - k[341]*y[IDX_NH2II] -
                        k[342]*y[IDX_NH2II] - k[343]*y[IDX_NH3II] -
                        k[344]*y[IDX_NH3II] - k[345]*y[IDX_NOII] -
                        k[346]*y[IDX_O2II] - k[347]*y[IDX_O2HII] -
                        k[348]*y[IDX_OHII] - k[349]*y[IDX_SiCII] -
                        k[350]*y[IDX_SiC2II] - k[351]*y[IDX_SiC3II] -
                        k[352]*y[IDX_SiHII] - k[353]*y[IDX_SiH2II] -
                        k[354]*y[IDX_SiH2II] - k[355]*y[IDX_SiH2II] -
                        k[356]*y[IDX_SiH3II] - k[357]*y[IDX_SiH3II] -
                        k[358]*y[IDX_SiH4II] - k[359]*y[IDX_SiH4II] -
                        k[360]*y[IDX_SiH5II] - k[361]*y[IDX_SiH5II] -
                        k[362]*y[IDX_SiOII] - k[363]*y[IDX_SiOHII] -
                        k[364]*y[IDX_SiOHII] - k[1214]*y[IDX_CII] -
                        k[1215]*y[IDX_CH3II] - k[1216]*y[IDX_HII] -
                        k[1217]*y[IDX_H2COII] - k[1218]*y[IDX_HeII] -
                        k[1219]*y[IDX_MgII] - k[1220]*y[IDX_NII] -
                        k[1221]*y[IDX_OII] - k[1222]*y[IDX_SiII] - k[1306];
    IJth(jmatrix, 110, 111) = 0.0 + k[232];
    IJth(jmatrix, 110, 112) = 0.0 - k[8]*y[IDX_EM] + k[8]*y[IDX_EM] + k[233] + k[234];
    IJth(jmatrix, 110, 113) = 0.0 + k[236] + k[258];
    IJth(jmatrix, 111, 4) = 0.0 + k[1331] + k[1332] + k[1333] + k[1334];
    IJth(jmatrix, 111, 28) = 0.0 + k[263] + k[502]*y[IDX_HII] + k[1004]*y[IDX_CI] +
                        k[1152];
    IJth(jmatrix, 111, 29) = 0.0 + k[335]*y[IDX_EM];
    IJth(jmatrix, 111, 32) = 0.0 + k[1081]*y[IDX_OI];
    IJth(jmatrix, 111, 33) = 0.0 + k[1082]*y[IDX_OI];
    IJth(jmatrix, 111, 35) = 0.0 - k[489]*y[IDX_COI];
    IJth(jmatrix, 111, 37) = 0.0 - k[954]*y[IDX_COI];
    IJth(jmatrix, 111, 38) = 0.0 + k[1083]*y[IDX_OI];
    IJth(jmatrix, 111, 39) = 0.0 - k[952]*y[IDX_COI];
    IJth(jmatrix, 111, 42) = 0.0 + k[874]*y[IDX_CI] + k[994]*y[IDX_HI] +
                        k[1055]*y[IDX_O2I] + k[1078]*y[IDX_OI];
    IJth(jmatrix, 111, 43) = 0.0 + k[637]*y[IDX_HCOII];
    IJth(jmatrix, 111, 46) = 0.0 + k[638]*y[IDX_HCOII];
    IJth(jmatrix, 111, 47) = 0.0 + k[639]*y[IDX_HCOII];
    IJth(jmatrix, 111, 49) = 0.0 + k[395]*y[IDX_CI] - k[490]*y[IDX_COI];
    IJth(jmatrix, 111, 50) = 0.0 + k[332]*y[IDX_EM] + k[333]*y[IDX_EM] -
                        k[485]*y[IDX_COI];
    IJth(jmatrix, 111, 51) = 0.0 - k[951]*y[IDX_COI];
    IJth(jmatrix, 111, 55) = 0.0 - k[448]*y[IDX_COI];
    IJth(jmatrix, 111, 56) = 0.0 + k[382]*y[IDX_CII] + k[640]*y[IDX_HCOII];
    IJth(jmatrix, 111, 57) = 0.0 - k[63]*y[IDX_COI] + k[480]*y[IDX_HCOI] +
                        k[481]*y[IDX_O2I];
    IJth(jmatrix, 111, 59) = 0.0 - k[487]*y[IDX_COI];
    IJth(jmatrix, 111, 60) = 0.0 - k[488]*y[IDX_COI];
    IJth(jmatrix, 111, 62) = 0.0 + k[861]*y[IDX_HCOII] + k[1103]*y[IDX_CO2I] -
                        k[1104]*y[IDX_COI];
    IJth(jmatrix, 111, 63) = 0.0 + k[648]*y[IDX_HCOII];
    IJth(jmatrix, 111, 64) = 0.0 - k[486]*y[IDX_COI];
    IJth(jmatrix, 111, 65) = 0.0 - k[74]*y[IDX_COI] + k[731]*y[IDX_HCOI];
    IJth(jmatrix, 111, 66) = 0.0 + k[319]*y[IDX_EM];
    IJth(jmatrix, 111, 67) = 0.0 + k[52]*y[IDX_COII];
    IJth(jmatrix, 111, 68) = 0.0 + k[28]*y[IDX_CI] + k[38]*y[IDX_CH2I] +
                        k[52]*y[IDX_CH4I] + k[54]*y[IDX_CHI] +
                        k[70]*y[IDX_H2COI] + k[71]*y[IDX_HCOI] +
                        k[72]*y[IDX_NOI] + k[73]*y[IDX_O2I] + k[123]*y[IDX_H2OI]
                        + k[127]*y[IDX_HI] + k[134]*y[IDX_HCNI] +
                        k[184]*y[IDX_NH2I] + k[193]*y[IDX_NH3I] +
                        k[200]*y[IDX_NHI] + k[217]*y[IDX_OI] + k[226]*y[IDX_OHI];
    IJth(jmatrix, 111, 69) = 0.0 - k[104]*y[IDX_COI] - k[515]*y[IDX_COI] +
                        k[519]*y[IDX_HCOI];
    IJth(jmatrix, 111, 70) = 0.0 + k[193]*y[IDX_COII];
    IJth(jmatrix, 111, 71) = 0.0 + k[905]*y[IDX_HCOI] + k[915]*y[IDX_OI];
    IJth(jmatrix, 111, 72) = 0.0 + k[252] + k[367]*y[IDX_CII] + k[398]*y[IDX_CHII] +
                        k[416]*y[IDX_CH2II] + k[666]*y[IDX_HeII] +
                        k[749]*y[IDX_NHII] + k[812]*y[IDX_OII] +
                        k[923]*y[IDX_CHI] + k[971]*y[IDX_HI] + k[1012]*y[IDX_NI]
                        + k[1059]*y[IDX_OI] + k[1103]*y[IDX_SiI] + k[1132];
    IJth(jmatrix, 111, 73) = 0.0 - k[158]*y[IDX_COI] - k[720]*y[IDX_COI] +
                        k[723]*y[IDX_HCOI];
    IJth(jmatrix, 111, 74) = 0.0 - k[208]*y[IDX_COI] + k[812]*y[IDX_CO2I] +
                        k[816]*y[IDX_HCOI];
    IJth(jmatrix, 111, 75) = 0.0 - k[621]*y[IDX_COI] + k[625]*y[IDX_HCOI];
    IJth(jmatrix, 111, 77) = 0.0 + k[749]*y[IDX_CO2I] - k[751]*y[IDX_COI];
    IJth(jmatrix, 111, 78) = 0.0 + k[644]*y[IDX_HCOI];
    IJth(jmatrix, 111, 79) = 0.0 + k[441]*y[IDX_HCOI];
    IJth(jmatrix, 111, 80) = 0.0 + k[184]*y[IDX_COII] + k[787]*y[IDX_HCOII];
    IJth(jmatrix, 111, 81) = 0.0 + k[416]*y[IDX_CO2I] + k[419]*y[IDX_HCOI];
    IJth(jmatrix, 111, 82) = 0.0 - k[553]*y[IDX_COI] + k[557]*y[IDX_HCOI];
    IJth(jmatrix, 111, 87) = 0.0 + k[367]*y[IDX_CO2I] + k[368]*y[IDX_H2COI] +
                        k[372]*y[IDX_HCOI] + k[377]*y[IDX_O2I] +
                        k[382]*y[IDX_SiOI];
    IJth(jmatrix, 111, 88) = 0.0 + k[134]*y[IDX_COII] + k[629]*y[IDX_HCOII] +
                        k[1064]*y[IDX_OI] + k[1095]*y[IDX_OHI];
    IJth(jmatrix, 111, 89) = 0.0 + k[398]*y[IDX_CO2I] + k[399]*y[IDX_H2COI] +
                        k[406]*y[IDX_HCOI];
    IJth(jmatrix, 111, 90) = 0.0 + k[38]*y[IDX_COII] + k[429]*y[IDX_HCOII] +
                        k[882]*y[IDX_HCOI] + k[891]*y[IDX_O2I] +
                        k[894]*y[IDX_OI] + k[895]*y[IDX_OI];
    IJth(jmatrix, 111, 91) = 0.0 + k[307]*y[IDX_EM] + k[308]*y[IDX_EM] +
                        k[641]*y[IDX_HCOI];
    IJth(jmatrix, 111, 92) = 0.0 + k[200]*y[IDX_COII] + k[799]*y[IDX_HCOII];
    IJth(jmatrix, 111, 93) = 0.0 - k[838]*y[IDX_COI] + k[842]*y[IDX_HCOI];
    IJth(jmatrix, 111, 94) = 0.0 + k[943]*y[IDX_HCOI] + k[946]*y[IDX_NOI] +
                        k[948]*y[IDX_O2I] + k[1057]*y[IDX_OI];
    IJth(jmatrix, 111, 95) = 0.0 + k[70]*y[IDX_COII] + k[255] + k[368]*y[IDX_CII] +
                        k[399]*y[IDX_CHII] + k[635]*y[IDX_HCOII] + k[1136] +
                        k[1137];
    IJth(jmatrix, 111, 96) = 0.0 + k[71]*y[IDX_COII] + k[260] + k[372]*y[IDX_CII] +
                        k[406]*y[IDX_CHII] + k[419]*y[IDX_CH2II] +
                        k[441]*y[IDX_CH3II] + k[480]*y[IDX_CNII] +
                        k[501]*y[IDX_HII] + k[519]*y[IDX_H2II] +
                        k[557]*y[IDX_H2OII] + k[625]*y[IDX_HCNII] +
                        k[636]*y[IDX_HCOII] + k[641]*y[IDX_H2COII] +
                        k[644]*y[IDX_O2II] + k[681]*y[IDX_HeII] +
                        k[723]*y[IDX_NII] + k[731]*y[IDX_N2II] +
                        k[816]*y[IDX_OII] + k[842]*y[IDX_OHII] +
                        k[864]*y[IDX_CI] + k[882]*y[IDX_CH2I] +
                        k[905]*y[IDX_CH3I] + k[925]*y[IDX_CHI] +
                        k[943]*y[IDX_CNI] + k[977]*y[IDX_HI] +
                        k[997]*y[IDX_HCOI] + k[997]*y[IDX_HCOI] +
                        k[997]*y[IDX_HCOI] + k[997]*y[IDX_HCOI] +
                        k[998]*y[IDX_HCOI] + k[998]*y[IDX_HCOI] +
                        k[1000]*y[IDX_NOI] + k[1002]*y[IDX_O2I] +
                        k[1014]*y[IDX_NI] + k[1067]*y[IDX_OI] +
                        k[1096]*y[IDX_OHI] + k[1149];
    IJth(jmatrix, 111, 97) = 0.0 + k[666]*y[IDX_CO2I] - k[669]*y[IDX_COI] +
                        k[681]*y[IDX_HCOI];
    IJth(jmatrix, 111, 98) = 0.0 + k[54]*y[IDX_COII] + k[466]*y[IDX_HCOII] +
                        k[923]*y[IDX_CO2I] + k[925]*y[IDX_HCOI] +
                        k[934]*y[IDX_O2I] + k[935]*y[IDX_O2I] + k[939]*y[IDX_OI];
    IJth(jmatrix, 111, 99) = 0.0 - k[583]*y[IDX_COI] - k[584]*y[IDX_COI];
    IJth(jmatrix, 111, 101) = 0.0 + k[72]*y[IDX_COII] + k[872]*y[IDX_CI] +
                        k[946]*y[IDX_CNI] + k[1000]*y[IDX_HCOI];
    IJth(jmatrix, 111, 102) = 0.0 + k[1012]*y[IDX_CO2I] + k[1014]*y[IDX_HCOI];
    IJth(jmatrix, 111, 103) = 0.0 + k[226]*y[IDX_COII] + k[854]*y[IDX_HCOII] +
                        k[875]*y[IDX_CI] - k[1092]*y[IDX_COI] +
                        k[1095]*y[IDX_HCNI] + k[1096]*y[IDX_HCOI];
    IJth(jmatrix, 111, 104) = 0.0 + k[73]*y[IDX_COII] + k[377]*y[IDX_CII] +
                        k[481]*y[IDX_CNII] + k[873]*y[IDX_CI] +
                        k[891]*y[IDX_CH2I] + k[934]*y[IDX_CHI] +
                        k[935]*y[IDX_CHI] + k[948]*y[IDX_CNI] -
                        k[953]*y[IDX_COI] + k[1002]*y[IDX_HCOI] +
                        k[1055]*y[IDX_OCNI];
    IJth(jmatrix, 111, 105) = 0.0 + k[28]*y[IDX_COII] + k[386]*y[IDX_HCOII] +
                        k[395]*y[IDX_SiOII] + k[864]*y[IDX_HCOI] +
                        k[872]*y[IDX_NOI] + k[873]*y[IDX_O2I] +
                        k[874]*y[IDX_OCNI] + k[875]*y[IDX_OHI] +
                        k[1004]*y[IDX_HNCOI] + k[1196]*y[IDX_OI];
    IJth(jmatrix, 111, 106) = 0.0 + k[501]*y[IDX_HCOI] + k[502]*y[IDX_HNCOI];
    IJth(jmatrix, 111, 107) = 0.0 + k[330]*y[IDX_EM] + k[386]*y[IDX_CI] +
                        k[429]*y[IDX_CH2I] + k[466]*y[IDX_CHI] +
                        k[566]*y[IDX_H2OI] + k[629]*y[IDX_HCNI] +
                        k[635]*y[IDX_H2COI] + k[636]*y[IDX_HCOI] +
                        k[637]*y[IDX_SiH2I] + k[638]*y[IDX_SiH4I] +
                        k[639]*y[IDX_SiHI] + k[640]*y[IDX_SiOI] +
                        k[648]*y[IDX_HNCI] + k[787]*y[IDX_NH2I] +
                        k[799]*y[IDX_NHI] + k[854]*y[IDX_OHI] +
                        k[861]*y[IDX_SiI];
    IJth(jmatrix, 111, 108) = 0.0 + k[123]*y[IDX_COII] + k[566]*y[IDX_HCOII];
    IJth(jmatrix, 111, 109) = 0.0 + k[217]*y[IDX_COII] + k[894]*y[IDX_CH2I] +
                        k[895]*y[IDX_CH2I] + k[915]*y[IDX_CH3I] +
                        k[939]*y[IDX_CHI] + k[1057]*y[IDX_CNI] +
                        k[1059]*y[IDX_CO2I] + k[1064]*y[IDX_HCNI] +
                        k[1067]*y[IDX_HCOI] + k[1078]*y[IDX_OCNI] +
                        k[1081]*y[IDX_SiC2I] + k[1082]*y[IDX_SiC3I] +
                        k[1083]*y[IDX_SiCI] + k[1196]*y[IDX_CI];
    IJth(jmatrix, 111, 110) = 0.0 + k[307]*y[IDX_H2COII] + k[308]*y[IDX_H2COII] +
                        k[319]*y[IDX_H3COII] + k[330]*y[IDX_HCOII] +
                        k[332]*y[IDX_HCO2II] + k[333]*y[IDX_HCO2II] +
                        k[335]*y[IDX_HOCII];
    IJth(jmatrix, 111, 111) = 0.0 - k[63]*y[IDX_CNII] - k[74]*y[IDX_N2II] -
                        k[104]*y[IDX_H2II] - k[158]*y[IDX_NII] -
                        k[208]*y[IDX_OII] - k[232] - k[253] -
                        k[448]*y[IDX_CH4II] - k[485]*y[IDX_HCO2II] -
                        k[486]*y[IDX_HNOII] - k[487]*y[IDX_N2HII] -
                        k[488]*y[IDX_O2HII] - k[489]*y[IDX_SiH4II] -
                        k[490]*y[IDX_SiOII] - k[515]*y[IDX_H2II] -
                        k[553]*y[IDX_H2OII] - k[583]*y[IDX_H3II] -
                        k[584]*y[IDX_H3II] - k[621]*y[IDX_HCNII] -
                        k[669]*y[IDX_HeII] - k[720]*y[IDX_NII] -
                        k[751]*y[IDX_NHII] - k[838]*y[IDX_OHII] -
                        k[951]*y[IDX_HNOI] - k[952]*y[IDX_NO2I] -
                        k[953]*y[IDX_O2I] - k[954]*y[IDX_O2HI] -
                        k[972]*y[IDX_HI] - k[1092]*y[IDX_OHI] -
                        k[1104]*y[IDX_SiI] - k[1133] - k[1227] - k[1251] -
                        k[1304];
    IJth(jmatrix, 111, 113) = 0.0 + k[127]*y[IDX_COII] + k[971]*y[IDX_CO2I] -
                        k[972]*y[IDX_COI] + k[977]*y[IDX_HCOI] +
                        k[994]*y[IDX_OCNI];
    IJth(jmatrix, 112, 23) = 0.0 + k[973]*y[IDX_HI] + k[1060]*y[IDX_OI];
    IJth(jmatrix, 112, 25) = 0.0 + k[311]*y[IDX_EM];
    IJth(jmatrix, 112, 26) = 0.0 + k[257] + k[499]*y[IDX_HII] + k[1143];
    IJth(jmatrix, 112, 27) = 0.0 - k[537]*y[IDX_H2I];
    IJth(jmatrix, 112, 29) = 0.0 - k[5]*y[IDX_H2I] + k[5]*y[IDX_H2I];
    IJth(jmatrix, 112, 34) = 0.0 + k[360]*y[IDX_EM];
    IJth(jmatrix, 112, 35) = 0.0 + k[358]*y[IDX_EM] - k[546]*y[IDX_H2I];
    IJth(jmatrix, 112, 37) = 0.0 + k[991]*y[IDX_HI];
    IJth(jmatrix, 112, 39) = 0.0 + k[595]*y[IDX_H3II];
    IJth(jmatrix, 112, 40) = 0.0 + k[357]*y[IDX_EM] + k[834]*y[IDX_OI] -
                        k[1204]*y[IDX_H2I];
    IJth(jmatrix, 112, 41) = 0.0 + k[353]*y[IDX_EM];
    IJth(jmatrix, 112, 43) = 0.0 + k[380]*y[IDX_CII] + k[505]*y[IDX_HII] +
                        k[602]*y[IDX_H3II] + k[703]*y[IDX_HeII] +
                        k[1085]*y[IDX_OI];
    IJth(jmatrix, 112, 45) = 0.0 + k[619]*y[IDX_HI] - k[1203]*y[IDX_H2I];
    IJth(jmatrix, 112, 46) = 0.0 + k[291] + k[507]*y[IDX_HII] + k[604]*y[IDX_H3II] +
                        k[707]*y[IDX_HeII] + k[707]*y[IDX_HeII] +
                        k[708]*y[IDX_HeII] + k[1185] + k[1187];
    IJth(jmatrix, 112, 47) = 0.0 + k[508]*y[IDX_HII] + k[605]*y[IDX_H3II];
    IJth(jmatrix, 112, 48) = 0.0 + k[506]*y[IDX_HII] + k[603]*y[IDX_H3II] +
                        k[705]*y[IDX_HeII] + k[1184];
    IJth(jmatrix, 112, 49) = 0.0 - k[547]*y[IDX_H2I];
    IJth(jmatrix, 112, 51) = 0.0 + k[503]*y[IDX_HII] + k[590]*y[IDX_H3II] +
                        k[981]*y[IDX_HI];
    IJth(jmatrix, 112, 52) = 0.0 + k[247] + k[493]*y[IDX_HII] + k[494]*y[IDX_HII] +
                        k[494]*y[IDX_HII] + k[579]*y[IDX_H3II] + k[1119];
    IJth(jmatrix, 112, 53) = 0.0 + k[591]*y[IDX_H3II];
    IJth(jmatrix, 112, 55) = 0.0 + k[617]*y[IDX_HI] + k[1122];
    IJth(jmatrix, 112, 56) = 0.0 + k[606]*y[IDX_H3II];
    IJth(jmatrix, 112, 57) = 0.0 - k[531]*y[IDX_H2I];
    IJth(jmatrix, 112, 60) = 0.0 - k[544]*y[IDX_H2I];
    IJth(jmatrix, 112, 61) = 0.0 - k[1202]*y[IDX_H2I];
    IJth(jmatrix, 112, 62) = 0.0 + k[601]*y[IDX_H3II];
    IJth(jmatrix, 112, 63) = 0.0 + k[589]*y[IDX_H3II];
    IJth(jmatrix, 112, 65) = 0.0 + k[455]*y[IDX_CH4I] - k[539]*y[IDX_H2I];
    IJth(jmatrix, 112, 66) = 0.0 + k[319]*y[IDX_EM];
    IJth(jmatrix, 112, 67) = 0.0 + k[101]*y[IDX_H2II] + k[249] + k[455]*y[IDX_N2II] +
                        k[495]*y[IDX_HII] + k[511]*y[IDX_H2II] +
                        k[658]*y[IDX_HeII] + k[659]*y[IDX_HeII] +
                        k[717]*y[IDX_NII] + k[969]*y[IDX_HI] + k[1124] + k[1127];
    IJth(jmatrix, 112, 68) = 0.0 - k[532]*y[IDX_H2I] - k[533]*y[IDX_H2I];
    IJth(jmatrix, 112, 69) = 0.0 + k[100]*y[IDX_CH2I] + k[101]*y[IDX_CH4I] +
                        k[102]*y[IDX_CHI] + k[103]*y[IDX_CNI] +
                        k[104]*y[IDX_COI] + k[105]*y[IDX_H2COI] +
                        k[106]*y[IDX_H2OI] + k[107]*y[IDX_HCNI] +
                        k[108]*y[IDX_HCOI] + k[109]*y[IDX_NH2I] +
                        k[110]*y[IDX_NH3I] + k[111]*y[IDX_NHI] +
                        k[112]*y[IDX_NOI] + k[113]*y[IDX_O2I] +
                        k[114]*y[IDX_OHI] + k[128]*y[IDX_HI] +
                        k[511]*y[IDX_CH4I] - k[516]*y[IDX_H2I] +
                        k[517]*y[IDX_H2COI];
    IJth(jmatrix, 112, 70) = 0.0 + k[110]*y[IDX_H2II] + k[273] + k[374]*y[IDX_CII] +
                        k[691]*y[IDX_HeII] + k[724]*y[IDX_NII] +
                        k[984]*y[IDX_HI] + k[1161];
    IJth(jmatrix, 112, 71) = 0.0 + k[246] + k[578]*y[IDX_H3II] + k[655]*y[IDX_HeII] +
                        k[915]*y[IDX_OI] + k[918]*y[IDX_OHI] - k[957]*y[IDX_H2I]
                        + k[968]*y[IDX_HI] + k[1009]*y[IDX_NI] + k[1118];
    IJth(jmatrix, 112, 72) = 0.0 + k[582]*y[IDX_H3II];
    IJth(jmatrix, 112, 73) = 0.0 - k[538]*y[IDX_H2I] + k[717]*y[IDX_CH4I] +
                        k[724]*y[IDX_NH3I];
    IJth(jmatrix, 112, 74) = 0.0 - k[543]*y[IDX_H2I];
    IJth(jmatrix, 112, 75) = 0.0 - k[535]*y[IDX_H2I];
    IJth(jmatrix, 112, 76) = 0.0 - k[542]*y[IDX_H2I];
    IJth(jmatrix, 112, 77) = 0.0 - k[540]*y[IDX_H2I] - k[541]*y[IDX_H2I] +
                        k[755]*y[IDX_H2OI];
    IJth(jmatrix, 112, 79) = 0.0 + k[299]*y[IDX_EM] + k[444]*y[IDX_OI] +
                        k[445]*y[IDX_OHI] + k[616]*y[IDX_HI] + k[794]*y[IDX_NHI]
                        + k[1114];
    IJth(jmatrix, 112, 80) = 0.0 + k[109]*y[IDX_H2II] + k[409]*y[IDX_CHII] +
                        k[593]*y[IDX_H3II] + k[689]*y[IDX_HeII] -
                        k[961]*y[IDX_H2I] + k[983]*y[IDX_HI];
    IJth(jmatrix, 112, 81) = 0.0 + k[295]*y[IDX_EM] - k[530]*y[IDX_H2I] +
                        k[615]*y[IDX_HI] + k[1109];
    IJth(jmatrix, 112, 82) = 0.0 + k[312]*y[IDX_EM] - k[534]*y[IDX_H2I] +
                        k[739]*y[IDX_NI] + k[823]*y[IDX_OI];
    IJth(jmatrix, 112, 83) = 0.0 + k[828]*y[IDX_OI];
    IJth(jmatrix, 112, 85) = 0.0 + k[323]*y[IDX_EM] + k[324]*y[IDX_EM] +
                        k[384]*y[IDX_CI];
    IJth(jmatrix, 112, 86) = 0.0 + k[592]*y[IDX_H3II];
    IJth(jmatrix, 112, 87) = 0.0 + k[374]*y[IDX_NH3I] + k[380]*y[IDX_SiH2I] -
                        k[528]*y[IDX_H2I] - k[1199]*y[IDX_H2I];
    IJth(jmatrix, 112, 88) = 0.0 + k[107]*y[IDX_H2II] + k[587]*y[IDX_H3II] +
                        k[976]*y[IDX_HI];
    IJth(jmatrix, 112, 89) = 0.0 + k[404]*y[IDX_H2OI] + k[409]*y[IDX_NH2I] +
                        k[410]*y[IDX_NHI] + k[415]*y[IDX_OHI] -
                        k[529]*y[IDX_H2I] + k[614]*y[IDX_HI];
    IJth(jmatrix, 112, 90) = 0.0 + k[100]*y[IDX_H2II] + k[491]*y[IDX_HII] +
                        k[577]*y[IDX_H3II] + k[653]*y[IDX_HeII] +
                        k[889]*y[IDX_O2I] + k[894]*y[IDX_OI] - k[956]*y[IDX_H2I]
                        + k[967]*y[IDX_HI];
    IJth(jmatrix, 112, 91) = 0.0 + k[307]*y[IDX_EM];
    IJth(jmatrix, 112, 92) = 0.0 + k[111]*y[IDX_H2II] + k[410]*y[IDX_CHII] +
                        k[594]*y[IDX_H3II] + k[794]*y[IDX_CH3II] -
                        k[962]*y[IDX_H2I] + k[985]*y[IDX_HI] +
                        k[1038]*y[IDX_NHI] + k[1038]*y[IDX_NHI];
    IJth(jmatrix, 112, 93) = 0.0 - k[545]*y[IDX_H2I];
    IJth(jmatrix, 112, 94) = 0.0 + k[103]*y[IDX_H2II] + k[581]*y[IDX_H3II] -
                        k[959]*y[IDX_H2I];
    IJth(jmatrix, 112, 95) = 0.0 + k[105]*y[IDX_H2II] + k[255] + k[497]*y[IDX_HII] +
                        k[498]*y[IDX_HII] + k[517]*y[IDX_H2II] +
                        k[585]*y[IDX_H3II] + k[670]*y[IDX_HeII] +
                        k[974]*y[IDX_HI] + k[1136];
    IJth(jmatrix, 112, 96) = 0.0 + k[108]*y[IDX_H2II] + k[500]*y[IDX_HII] +
                        k[588]*y[IDX_H3II] + k[977]*y[IDX_HI] +
                        k[997]*y[IDX_HCOI] + k[997]*y[IDX_HCOI];
    IJth(jmatrix, 112, 97) = 0.0 - k[115]*y[IDX_H2I] - k[536]*y[IDX_H2I] +
                        k[653]*y[IDX_CH2I] + k[655]*y[IDX_CH3I] +
                        k[658]*y[IDX_CH4I] + k[659]*y[IDX_CH4I] +
                        k[670]*y[IDX_H2COI] + k[689]*y[IDX_NH2I] +
                        k[691]*y[IDX_NH3I] + k[703]*y[IDX_SiH2I] +
                        k[705]*y[IDX_SiH3I] + k[707]*y[IDX_SiH4I] +
                        k[707]*y[IDX_SiH4I] + k[708]*y[IDX_SiH4I];
    IJth(jmatrix, 112, 98) = 0.0 - k[2]*y[IDX_H2I] + k[2]*y[IDX_H2I] +
                        k[102]*y[IDX_H2II] + k[580]*y[IDX_H3II] -
                        k[958]*y[IDX_H2I] + k[970]*y[IDX_HI] -
                        k[1201]*y[IDX_H2I];
    IJth(jmatrix, 112, 99) = 0.0 + k[315]*y[IDX_EM] + k[576]*y[IDX_CI] +
                        k[577]*y[IDX_CH2I] + k[578]*y[IDX_CH3I] +
                        k[579]*y[IDX_CH3OHI] + k[580]*y[IDX_CHI] +
                        k[581]*y[IDX_CNI] + k[582]*y[IDX_CO2I] +
                        k[583]*y[IDX_COI] + k[584]*y[IDX_COI] +
                        k[585]*y[IDX_H2COI] + k[586]*y[IDX_H2OI] +
                        k[587]*y[IDX_HCNI] + k[588]*y[IDX_HCOI] +
                        k[589]*y[IDX_HNCI] + k[590]*y[IDX_HNOI] +
                        k[591]*y[IDX_MgI] + k[592]*y[IDX_N2I] +
                        k[593]*y[IDX_NH2I] + k[594]*y[IDX_NHI] +
                        k[595]*y[IDX_NO2I] + k[596]*y[IDX_NOI] +
                        k[597]*y[IDX_O2I] + k[599]*y[IDX_OI] + k[600]*y[IDX_OHI]
                        + k[601]*y[IDX_SiI] + k[602]*y[IDX_SiH2I] +
                        k[603]*y[IDX_SiH3I] + k[604]*y[IDX_SiH4I] +
                        k[605]*y[IDX_SiHI] + k[606]*y[IDX_SiOI] + k[1146];
    IJth(jmatrix, 112, 101) = 0.0 + k[112]*y[IDX_H2II] + k[596]*y[IDX_H3II];
    IJth(jmatrix, 112, 102) = 0.0 + k[739]*y[IDX_H2OII] - k[960]*y[IDX_H2I] +
                        k[1009]*y[IDX_CH3I];
    IJth(jmatrix, 112, 103) = 0.0 - k[7]*y[IDX_H2I] + k[7]*y[IDX_H2I] +
                        k[114]*y[IDX_H2II] + k[415]*y[IDX_CHII] +
                        k[445]*y[IDX_CH3II] + k[600]*y[IDX_H3II] +
                        k[918]*y[IDX_CH3I] - k[966]*y[IDX_H2I] +
                        k[996]*y[IDX_HI];
    IJth(jmatrix, 112, 104) = 0.0 - k[6]*y[IDX_H2I] + k[6]*y[IDX_H2I] +
                        k[113]*y[IDX_H2II] + k[597]*y[IDX_H3II] +
                        k[889]*y[IDX_CH2I] - k[963]*y[IDX_H2I] -
                        k[964]*y[IDX_H2I];
    IJth(jmatrix, 112, 105) = 0.0 + k[384]*y[IDX_H3OII] + k[576]*y[IDX_H3II] -
                        k[955]*y[IDX_H2I] - k[1200]*y[IDX_H2I];
    IJth(jmatrix, 112, 106) = 0.0 + k[491]*y[IDX_CH2I] + k[493]*y[IDX_CH3OHI] +
                        k[494]*y[IDX_CH3OHI] + k[494]*y[IDX_CH3OHI] +
                        k[495]*y[IDX_CH4I] + k[497]*y[IDX_H2COI] +
                        k[498]*y[IDX_H2COI] + k[499]*y[IDX_H2SiOI] +
                        k[500]*y[IDX_HCOI] + k[503]*y[IDX_HNOI] +
                        k[505]*y[IDX_SiH2I] + k[506]*y[IDX_SiH3I] +
                        k[507]*y[IDX_SiH4I] + k[508]*y[IDX_SiHI];
    IJth(jmatrix, 112, 108) = 0.0 - k[4]*y[IDX_H2I] + k[4]*y[IDX_H2I] +
                        k[106]*y[IDX_H2II] + k[404]*y[IDX_CHII] +
                        k[586]*y[IDX_H3II] + k[755]*y[IDX_NHII] +
                        k[975]*y[IDX_HI];
    IJth(jmatrix, 112, 109) = 0.0 + k[444]*y[IDX_CH3II] + k[599]*y[IDX_H3II] +
                        k[823]*y[IDX_H2OII] + k[828]*y[IDX_NH3II] +
                        k[834]*y[IDX_SiH3II] + k[894]*y[IDX_CH2I] +
                        k[915]*y[IDX_CH3I] - k[965]*y[IDX_H2I] +
                        k[1060]*y[IDX_H2CNI] + k[1085]*y[IDX_SiH2I];
    IJth(jmatrix, 112, 110) = 0.0 - k[8]*y[IDX_H2I] + k[295]*y[IDX_CH2II] +
                        k[299]*y[IDX_CH3II] + k[307]*y[IDX_H2COII] +
                        k[311]*y[IDX_H2NOII] + k[312]*y[IDX_H2OII] +
                        k[315]*y[IDX_H3II] + k[319]*y[IDX_H3COII] +
                        k[323]*y[IDX_H3OII] + k[324]*y[IDX_H3OII] +
                        k[353]*y[IDX_SiH2II] + k[357]*y[IDX_SiH3II] +
                        k[358]*y[IDX_SiH4II] + k[360]*y[IDX_SiH5II];
    IJth(jmatrix, 112, 111) = 0.0 + k[104]*y[IDX_H2II] + k[583]*y[IDX_H3II] +
                        k[584]*y[IDX_H3II];
    IJth(jmatrix, 112, 112) = 0.0 - k[2]*y[IDX_CHI] + k[2]*y[IDX_CHI] -
                        k[3]*y[IDX_H2I] - k[3]*y[IDX_H2I] - k[3]*y[IDX_H2I] -
                        k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] -
                        k[4]*y[IDX_H2OI] + k[4]*y[IDX_H2OI] - k[5]*y[IDX_HOCII]
                        + k[5]*y[IDX_HOCII] - k[6]*y[IDX_O2I] + k[6]*y[IDX_O2I]
                        - k[7]*y[IDX_OHI] + k[7]*y[IDX_OHI] - k[8]*y[IDX_EM] -
                        k[10]*y[IDX_HI] - k[115]*y[IDX_HeII] - k[233] - k[234] -
                        k[235] - k[516]*y[IDX_H2II] - k[528]*y[IDX_CII] -
                        k[529]*y[IDX_CHII] - k[530]*y[IDX_CH2II] -
                        k[531]*y[IDX_CNII] - k[532]*y[IDX_COII] -
                        k[533]*y[IDX_COII] - k[534]*y[IDX_H2OII] -
                        k[535]*y[IDX_HCNII] - k[536]*y[IDX_HeII] -
                        k[537]*y[IDX_HeHII] - k[538]*y[IDX_NII] -
                        k[539]*y[IDX_N2II] - k[540]*y[IDX_NHII] -
                        k[541]*y[IDX_NHII] - k[542]*y[IDX_NH2II] -
                        k[543]*y[IDX_OII] - k[544]*y[IDX_O2HII] -
                        k[545]*y[IDX_OHII] - k[546]*y[IDX_SiH4II] -
                        k[547]*y[IDX_SiOII] - k[955]*y[IDX_CI] -
                        k[956]*y[IDX_CH2I] - k[957]*y[IDX_CH3I] -
                        k[958]*y[IDX_CHI] - k[959]*y[IDX_CNI] - k[960]*y[IDX_NI]
                        - k[961]*y[IDX_NH2I] - k[962]*y[IDX_NHI] -
                        k[963]*y[IDX_O2I] - k[964]*y[IDX_O2I] - k[965]*y[IDX_OI]
                        - k[966]*y[IDX_OHI] - k[1199]*y[IDX_CII] -
                        k[1200]*y[IDX_CI] - k[1201]*y[IDX_CHI] -
                        k[1202]*y[IDX_SiII] - k[1203]*y[IDX_SiHII] -
                        k[1204]*y[IDX_SiH3II];
    IJth(jmatrix, 112, 113) = 0.0 - k[10]*y[IDX_H2I] + k[128]*y[IDX_H2II] +
                        k[614]*y[IDX_CHII] + k[615]*y[IDX_CH2II] +
                        k[616]*y[IDX_CH3II] + k[617]*y[IDX_CH4II] +
                        k[619]*y[IDX_SiHII] + k[967]*y[IDX_CH2I] +
                        k[968]*y[IDX_CH3I] + k[969]*y[IDX_CH4I] +
                        k[970]*y[IDX_CHI] + k[973]*y[IDX_H2CNI] +
                        k[974]*y[IDX_H2COI] + k[975]*y[IDX_H2OI] +
                        k[976]*y[IDX_HCNI] + k[977]*y[IDX_HCOI] +
                        k[981]*y[IDX_HNOI] + k[983]*y[IDX_NH2I] +
                        k[984]*y[IDX_NH3I] + k[985]*y[IDX_NHI] +
                        k[991]*y[IDX_O2HI] + k[996]*y[IDX_OHI];
    IJth(jmatrix, 113, 23) = 0.0 + k[254] - k[973]*y[IDX_HI] + k[1135];
    IJth(jmatrix, 113, 25) = 0.0 + k[310]*y[IDX_EM] + k[1296];
    IJth(jmatrix, 113, 26) = 0.0 + k[675]*y[IDX_HeII] + k[1144] + k[1144];
    IJth(jmatrix, 113, 27) = 0.0 + k[336]*y[IDX_EM] - k[618]*y[IDX_HI];
    IJth(jmatrix, 113, 29) = 0.0 + k[335]*y[IDX_EM];
    IJth(jmatrix, 113, 32) = 0.0 + k[92]*y[IDX_HII];
    IJth(jmatrix, 113, 33) = 0.0 + k[93]*y[IDX_HII];
    IJth(jmatrix, 113, 34) = 0.0 + k[361]*y[IDX_EM] + k[1248];
    IJth(jmatrix, 113, 35) = 0.0 + k[359]*y[IDX_EM] + k[546]*y[IDX_H2I];
    IJth(jmatrix, 113, 37) = 0.0 + k[281] - k[990]*y[IDX_HI] - k[991]*y[IDX_HI] -
                        k[992]*y[IDX_HI] + k[1170];
    IJth(jmatrix, 113, 38) = 0.0 + k[94]*y[IDX_HII];
    IJth(jmatrix, 113, 39) = 0.0 - k[986]*y[IDX_HI];
    IJth(jmatrix, 113, 40) = 0.0 + k[356]*y[IDX_EM];
    IJth(jmatrix, 113, 41) = 0.0 + k[354]*y[IDX_EM] + k[354]*y[IDX_EM] +
                        k[355]*y[IDX_EM] + k[833]*y[IDX_OI];
    IJth(jmatrix, 113, 42) = 0.0 - k[993]*y[IDX_HI] - k[994]*y[IDX_HI] -
                        k[995]*y[IDX_HI];
    IJth(jmatrix, 113, 43) = 0.0 + k[95]*y[IDX_HII] + k[289] + k[704]*y[IDX_HeII] +
                        k[1086]*y[IDX_OI] + k[1086]*y[IDX_OI] + k[1181];
    IJth(jmatrix, 113, 44) = 0.0 + k[364]*y[IDX_EM];
    IJth(jmatrix, 113, 45) = 0.0 + k[352]*y[IDX_EM] + k[394]*y[IDX_CI] -
                        k[619]*y[IDX_HI] + k[832]*y[IDX_OI] + k[1179];
    IJth(jmatrix, 113, 46) = 0.0 + k[97]*y[IDX_HII] + k[708]*y[IDX_HeII] + k[1186] +
                        k[1187];
    IJth(jmatrix, 113, 47) = 0.0 + k[98]*y[IDX_HII] + k[292] + k[381]*y[IDX_CII] +
                        k[709]*y[IDX_HeII] + k[877]*y[IDX_CI] +
                        k[1089]*y[IDX_OI] + k[1188];
    IJth(jmatrix, 113, 48) = 0.0 + k[96]*y[IDX_HII] + k[290] + k[706]*y[IDX_HeII] +
                        k[1087]*y[IDX_OI] + k[1182];
    IJth(jmatrix, 113, 49) = 0.0 + k[547]*y[IDX_H2I];
    IJth(jmatrix, 113, 50) = 0.0 + k[331]*y[IDX_EM] + k[332]*y[IDX_EM] + k[1287];
    IJth(jmatrix, 113, 51) = 0.0 + k[264] + k[686]*y[IDX_HeII] - k[980]*y[IDX_HI] -
                        k[981]*y[IDX_HI] - k[982]*y[IDX_HI] + k[1068]*y[IDX_OI]
                        + k[1153];
    IJth(jmatrix, 113, 52) = 0.0 + k[712]*y[IDX_NII] + k[714]*y[IDX_NII] +
                        k[715]*y[IDX_NII] + k[820]*y[IDX_O2II] + k[1120];
    IJth(jmatrix, 113, 53) = 0.0 + k[83]*y[IDX_HII] + k[591]*y[IDX_H3II];
    IJth(jmatrix, 113, 55) = 0.0 + k[301]*y[IDX_EM] + k[301]*y[IDX_EM] +
                        k[302]*y[IDX_EM] - k[617]*y[IDX_HI] + k[1123];
    IJth(jmatrix, 113, 56) = 0.0 + k[99]*y[IDX_HII];
    IJth(jmatrix, 113, 57) = 0.0 - k[126]*y[IDX_HI] + k[531]*y[IDX_H2I];
    IJth(jmatrix, 113, 58) = 0.0 + k[327]*y[IDX_EM] + k[327]*y[IDX_EM] +
                        k[328]*y[IDX_EM] + k[329]*y[IDX_EM] + k[1301];
    IJth(jmatrix, 113, 59) = 0.0 + k[338]*y[IDX_EM] + k[1302];
    IJth(jmatrix, 113, 60) = 0.0 + k[347]*y[IDX_EM];
    IJth(jmatrix, 113, 61) = 0.0 + k[476]*y[IDX_CHI] + k[572]*y[IDX_H2OI] +
                        k[859]*y[IDX_OHI] - k[1209]*y[IDX_HI];
    IJth(jmatrix, 113, 62) = 0.0 + k[91]*y[IDX_HII] + k[1102]*y[IDX_OHI];
    IJth(jmatrix, 113, 63) = 0.0 + k[262] + k[683]*y[IDX_HeII] + k[684]*y[IDX_HeII] -
                        k[979]*y[IDX_HI] + k[979]*y[IDX_HI] + k[1151];
    IJth(jmatrix, 113, 64) = 0.0 + k[334]*y[IDX_EM];
    IJth(jmatrix, 113, 65) = 0.0 + k[456]*y[IDX_CH4I] + k[539]*y[IDX_H2I] +
                        k[730]*y[IDX_H2COI];
    IJth(jmatrix, 113, 66) = 0.0 + k[319]*y[IDX_EM] + k[320]*y[IDX_EM] +
                        k[321]*y[IDX_EM] + k[321]*y[IDX_EM];
    IJth(jmatrix, 113, 67) = 0.0 + k[77]*y[IDX_HII] + k[456]*y[IDX_N2II] +
                        k[511]*y[IDX_H2II] + k[658]*y[IDX_HeII] +
                        k[660]*y[IDX_HeII] + k[716]*y[IDX_NII] +
                        k[717]*y[IDX_NII] + k[718]*y[IDX_NII] +
                        k[718]*y[IDX_NII] - k[969]*y[IDX_HI] + k[1125] + k[1127];
    IJth(jmatrix, 113, 68) = 0.0 - k[127]*y[IDX_HI] + k[532]*y[IDX_H2I] +
                        k[533]*y[IDX_H2I];
    IJth(jmatrix, 113, 69) = 0.0 - k[128]*y[IDX_HI] + k[305]*y[IDX_EM] +
                        k[305]*y[IDX_EM] + k[509]*y[IDX_CI] + k[510]*y[IDX_CH2I]
                        + k[511]*y[IDX_CH4I] + k[512]*y[IDX_CHI] +
                        k[513]*y[IDX_CNI] + k[514]*y[IDX_CO2I] +
                        k[515]*y[IDX_COI] + k[516]*y[IDX_H2I] +
                        k[517]*y[IDX_H2COI] + k[518]*y[IDX_H2OI] +
                        k[520]*y[IDX_HeI] + k[521]*y[IDX_N2I] + k[522]*y[IDX_NI]
                        + k[523]*y[IDX_NHI] + k[524]*y[IDX_NOI] +
                        k[525]*y[IDX_O2I] + k[526]*y[IDX_OI] + k[527]*y[IDX_OHI]
                        + k[1134];
    IJth(jmatrix, 113, 70) = 0.0 + k[85]*y[IDX_HII] + k[271] + k[692]*y[IDX_HeII] -
                        k[984]*y[IDX_HI] + k[1159];
    IJth(jmatrix, 113, 71) = 0.0 + k[76]*y[IDX_HII] + k[244] + k[915]*y[IDX_OI] +
                        k[916]*y[IDX_OI] + k[957]*y[IDX_H2I] - k[968]*y[IDX_HI]
                        + k[1008]*y[IDX_NI] + k[1010]*y[IDX_NI] +
                        k[1010]*y[IDX_NI] + k[1116];
    IJth(jmatrix, 113, 72) = 0.0 + k[514]*y[IDX_H2II] - k[971]*y[IDX_HI];
    IJth(jmatrix, 113, 73) = 0.0 + k[468]*y[IDX_CHI] + k[538]*y[IDX_H2I] +
                        k[712]*y[IDX_CH3OHI] + k[714]*y[IDX_CH3OHI] +
                        k[715]*y[IDX_CH3OHI] + k[716]*y[IDX_CH4I] +
                        k[717]*y[IDX_CH4I] + k[718]*y[IDX_CH4I] +
                        k[718]*y[IDX_CH4I] + k[726]*y[IDX_NHI];
    IJth(jmatrix, 113, 74) = 0.0 - k[131]*y[IDX_HI] + k[472]*y[IDX_CHI] +
                        k[543]*y[IDX_H2I] + k[803]*y[IDX_NHI] +
                        k[819]*y[IDX_OHI];
    IJth(jmatrix, 113, 75) = 0.0 - k[129]*y[IDX_HI] + k[326]*y[IDX_EM] +
                        k[535]*y[IDX_H2I];
    IJth(jmatrix, 113, 76) = 0.0 + k[341]*y[IDX_EM] + k[341]*y[IDX_EM] +
                        k[342]*y[IDX_EM] + k[542]*y[IDX_H2I] + k[741]*y[IDX_NI]
                        + k[827]*y[IDX_OI];
    IJth(jmatrix, 113, 77) = 0.0 + k[340]*y[IDX_EM] + k[541]*y[IDX_H2I] +
                        k[740]*y[IDX_NI];
    IJth(jmatrix, 113, 78) = 0.0 + k[551]*y[IDX_H2COI] + k[820]*y[IDX_CH3OHI];
    IJth(jmatrix, 113, 79) = 0.0 + k[298]*y[IDX_EM] + k[300]*y[IDX_EM] +
                        k[300]*y[IDX_EM] + k[443]*y[IDX_OI] - k[616]*y[IDX_HI] +
                        k[1115];
    IJth(jmatrix, 113, 80) = 0.0 + k[84]*y[IDX_HII] + k[270] + k[373]*y[IDX_CII] +
                        k[690]*y[IDX_HeII] + k[866]*y[IDX_CI] + k[867]*y[IDX_CI]
                        + k[961]*y[IDX_H2I] - k[983]*y[IDX_HI] +
                        k[1030]*y[IDX_NOI] + k[1072]*y[IDX_OI] + k[1158];
    IJth(jmatrix, 113, 81) = 0.0 + k[296]*y[IDX_EM] + k[296]*y[IDX_EM] +
                        k[297]*y[IDX_EM] + k[418]*y[IDX_H2OI] + k[421]*y[IDX_OI]
                        + k[530]*y[IDX_H2I] - k[615]*y[IDX_HI] +
                        k[736]*y[IDX_NI] + k[1110];
    IJth(jmatrix, 113, 82) = 0.0 + k[313]*y[IDX_EM] + k[313]*y[IDX_EM] +
                        k[314]*y[IDX_EM] + k[534]*y[IDX_H2I] + k[738]*y[IDX_NI]
                        + k[1140];
    IJth(jmatrix, 113, 83) = 0.0 + k[343]*y[IDX_EM] + k[344]*y[IDX_EM] +
                        k[344]*y[IDX_EM];
    IJth(jmatrix, 113, 85) = 0.0 + k[322]*y[IDX_EM] + k[323]*y[IDX_EM] +
                        k[325]*y[IDX_EM] + k[325]*y[IDX_EM] + k[1286];
    IJth(jmatrix, 113, 86) = 0.0 + k[521]*y[IDX_H2II];
    IJth(jmatrix, 113, 87) = 0.0 + k[370]*y[IDX_H2OI] + k[371]*y[IDX_H2OI] +
                        k[373]*y[IDX_NH2I] + k[375]*y[IDX_NHI] +
                        k[379]*y[IDX_OHI] + k[381]*y[IDX_SiHI] +
                        k[528]*y[IDX_H2I] - k[1205]*y[IDX_HI];
    IJth(jmatrix, 113, 88) = 0.0 + k[81]*y[IDX_HII] + k[259] + k[676]*y[IDX_HeII] +
                        k[678]*y[IDX_HeII] - k[976]*y[IDX_HI] +
                        k[1065]*y[IDX_OI] + k[1147];
    IJth(jmatrix, 113, 89) = 0.0 + k[241] + k[294]*y[IDX_EM] + k[402]*y[IDX_H2OI] +
                        k[408]*y[IDX_NI] + k[414]*y[IDX_OI] + k[529]*y[IDX_H2I]
                        - k[614]*y[IDX_HI];
    IJth(jmatrix, 113, 90) = 0.0 + k[75]*y[IDX_HII] + k[243] + k[510]*y[IDX_H2II] +
                        k[654]*y[IDX_HeII] + k[888]*y[IDX_NOI] +
                        k[890]*y[IDX_O2I] + k[890]*y[IDX_O2I] + k[895]*y[IDX_OI]
                        + k[895]*y[IDX_OI] + k[896]*y[IDX_OI] +
                        k[898]*y[IDX_OHI] + k[956]*y[IDX_H2I] - k[967]*y[IDX_HI]
                        + k[1005]*y[IDX_NI] + k[1006]*y[IDX_NI] + k[1113];
    IJth(jmatrix, 113, 91) = 0.0 + k[308]*y[IDX_EM] + k[308]*y[IDX_EM] +
                        k[309]*y[IDX_EM];
    IJth(jmatrix, 113, 92) = 0.0 + k[86]*y[IDX_HII] + k[274] + k[375]*y[IDX_CII] +
                        k[523]*y[IDX_H2II] + k[693]*y[IDX_HeII] +
                        k[726]*y[IDX_NII] + k[803]*y[IDX_OII] + k[869]*y[IDX_CI]
                        + k[962]*y[IDX_H2I] - k[985]*y[IDX_HI] +
                        k[1018]*y[IDX_NI] + k[1039]*y[IDX_NHI] +
                        k[1039]*y[IDX_NHI] + k[1039]*y[IDX_NHI] +
                        k[1039]*y[IDX_NHI] + k[1042]*y[IDX_NOI] +
                        k[1046]*y[IDX_OI] + k[1049]*y[IDX_OHI] + k[1162];
    IJth(jmatrix, 113, 93) = 0.0 + k[348]*y[IDX_EM] + k[545]*y[IDX_H2I] +
                        k[743]*y[IDX_NI] + k[830]*y[IDX_OI] + k[1173];
    IJth(jmatrix, 113, 94) = 0.0 + k[513]*y[IDX_H2II] + k[959]*y[IDX_H2I] +
                        k[1091]*y[IDX_OHI];
    IJth(jmatrix, 113, 95) = 0.0 + k[79]*y[IDX_HII] + k[497]*y[IDX_HII] +
                        k[517]*y[IDX_H2II] + k[551]*y[IDX_O2II] +
                        k[671]*y[IDX_HeII] + k[730]*y[IDX_N2II] -
                        k[974]*y[IDX_HI] + k[1137] + k[1137] + k[1139];
    IJth(jmatrix, 113, 96) = 0.0 + k[82]*y[IDX_HII] + k[260] + k[680]*y[IDX_HeII] -
                        k[977]*y[IDX_HI] - k[978]*y[IDX_HI] + k[1016]*y[IDX_NI]
                        + k[1066]*y[IDX_OI] + k[1149];
    IJth(jmatrix, 113, 97) = 0.0 - k[130]*y[IDX_HI] + k[536]*y[IDX_H2I] +
                        k[654]*y[IDX_CH2I] + k[658]*y[IDX_CH4I] +
                        k[660]*y[IDX_CH4I] + k[662]*y[IDX_CHI] +
                        k[671]*y[IDX_H2COI] + k[673]*y[IDX_H2OI] +
                        k[675]*y[IDX_H2SiOI] + k[676]*y[IDX_HCNI] +
                        k[678]*y[IDX_HCNI] + k[680]*y[IDX_HCOI] +
                        k[683]*y[IDX_HNCI] + k[684]*y[IDX_HNCI] +
                        k[686]*y[IDX_HNOI] + k[690]*y[IDX_NH2I] +
                        k[692]*y[IDX_NH3I] + k[693]*y[IDX_NHI] +
                        k[699]*y[IDX_OHI] + k[704]*y[IDX_SiH2I] +
                        k[706]*y[IDX_SiH3I] + k[708]*y[IDX_SiH4I] +
                        k[709]*y[IDX_SiHI];
    IJth(jmatrix, 113, 98) = 0.0 + k[2]*y[IDX_H2I] - k[9]*y[IDX_HI] + k[9]*y[IDX_HI]
                        + k[9]*y[IDX_HI] + k[78]*y[IDX_HII] + k[250] +
                        k[468]*y[IDX_NII] + k[472]*y[IDX_OII] +
                        k[476]*y[IDX_SiII] + k[512]*y[IDX_H2II] +
                        k[662]*y[IDX_HeII] + k[928]*y[IDX_NI] +
                        k[932]*y[IDX_NOI] + k[933]*y[IDX_O2I] +
                        k[934]*y[IDX_O2I] + k[939]*y[IDX_OI] + k[941]*y[IDX_OHI]
                        + k[958]*y[IDX_H2I] - k[970]*y[IDX_HI] + k[1128];
    IJth(jmatrix, 113, 99) = 0.0 + k[315]*y[IDX_EM] + k[316]*y[IDX_EM] +
                        k[316]*y[IDX_EM] + k[316]*y[IDX_EM] + k[591]*y[IDX_MgI]
                        + k[598]*y[IDX_OI] + k[1145];
    IJth(jmatrix, 113, 100) = 0.0 + k[520]*y[IDX_H2II];
    IJth(jmatrix, 113, 101) = 0.0 + k[87]*y[IDX_HII] + k[524]*y[IDX_H2II] +
                        k[888]*y[IDX_CH2I] + k[932]*y[IDX_CHI] -
                        k[987]*y[IDX_HI] - k[988]*y[IDX_HI] +
                        k[1030]*y[IDX_NH2I] + k[1042]*y[IDX_NHI] +
                        k[1099]*y[IDX_OHI];
    IJth(jmatrix, 113, 102) = 0.0 + k[408]*y[IDX_CHII] + k[522]*y[IDX_H2II] +
                        k[736]*y[IDX_CH2II] + k[738]*y[IDX_H2OII] +
                        k[740]*y[IDX_NHII] + k[741]*y[IDX_NH2II] +
                        k[743]*y[IDX_OHII] + k[928]*y[IDX_CHI] +
                        k[960]*y[IDX_H2I] + k[1005]*y[IDX_CH2I] +
                        k[1006]*y[IDX_CH2I] + k[1008]*y[IDX_CH3I] +
                        k[1010]*y[IDX_CH3I] + k[1010]*y[IDX_CH3I] +
                        k[1016]*y[IDX_HCOI] + k[1018]*y[IDX_NHI] +
                        k[1025]*y[IDX_OHI];
    IJth(jmatrix, 113, 103) = 0.0 + k[7]*y[IDX_H2I] - k[13]*y[IDX_HI] +
                        k[13]*y[IDX_HI] + k[13]*y[IDX_HI] + k[90]*y[IDX_HII] +
                        k[284] + k[379]*y[IDX_CII] + k[527]*y[IDX_H2II] +
                        k[699]*y[IDX_HeII] + k[819]*y[IDX_OII] +
                        k[855]*y[IDX_HCOII] + k[859]*y[IDX_SiII] +
                        k[875]*y[IDX_CI] + k[898]*y[IDX_CH2I] +
                        k[941]*y[IDX_CHI] + k[966]*y[IDX_H2I] - k[996]*y[IDX_HI]
                        + k[1025]*y[IDX_NI] + k[1049]*y[IDX_NHI] +
                        k[1080]*y[IDX_OI] + k[1091]*y[IDX_CNI] +
                        k[1092]*y[IDX_COI] + k[1099]*y[IDX_NOI] +
                        k[1102]*y[IDX_SiI] + k[1174] - k[1208]*y[IDX_HI];
    IJth(jmatrix, 113, 104) = 0.0 - k[12]*y[IDX_HI] + k[12]*y[IDX_HI] +
                        k[88]*y[IDX_HII] + k[525]*y[IDX_H2II] +
                        k[890]*y[IDX_CH2I] + k[890]*y[IDX_CH2I] +
                        k[933]*y[IDX_CHI] + k[934]*y[IDX_CHI] +
                        k[963]*y[IDX_H2I] - k[989]*y[IDX_HI];
    IJth(jmatrix, 113, 105) = 0.0 + k[394]*y[IDX_SiHII] + k[509]*y[IDX_H2II] +
                        k[866]*y[IDX_NH2I] + k[867]*y[IDX_NH2I] +
                        k[869]*y[IDX_NHI] + k[875]*y[IDX_OHI] +
                        k[877]*y[IDX_SiHI] + k[955]*y[IDX_H2I] -
                        k[1206]*y[IDX_HI];
    IJth(jmatrix, 113, 106) = 0.0 + k[75]*y[IDX_CH2I] + k[76]*y[IDX_CH3I] +
                        k[77]*y[IDX_CH4I] + k[78]*y[IDX_CHI] +
                        k[79]*y[IDX_H2COI] + k[80]*y[IDX_H2OI] +
                        k[81]*y[IDX_HCNI] + k[82]*y[IDX_HCOI] + k[83]*y[IDX_MgI]
                        + k[84]*y[IDX_NH2I] + k[85]*y[IDX_NH3I] +
                        k[86]*y[IDX_NHI] + k[87]*y[IDX_NOI] + k[88]*y[IDX_O2I] +
                        k[89]*y[IDX_OI] + k[90]*y[IDX_OHI] + k[91]*y[IDX_SiI] +
                        k[92]*y[IDX_SiC2I] + k[93]*y[IDX_SiC3I] +
                        k[94]*y[IDX_SiCI] + k[95]*y[IDX_SiH2I] +
                        k[96]*y[IDX_SiH3I] + k[97]*y[IDX_SiH4I] +
                        k[98]*y[IDX_SiHI] + k[99]*y[IDX_SiOI] +
                        k[497]*y[IDX_H2COI] - k[1197]*y[IDX_HI] +
                        k[1216]*y[IDX_EM];
    IJth(jmatrix, 113, 107) = 0.0 + k[330]*y[IDX_EM] + k[855]*y[IDX_OHI] + k[1148];
    IJth(jmatrix, 113, 108) = 0.0 + k[4]*y[IDX_H2I] - k[11]*y[IDX_HI] +
                        k[11]*y[IDX_HI] + k[11]*y[IDX_HI] + k[80]*y[IDX_HII] +
                        k[256] + k[370]*y[IDX_CII] + k[371]*y[IDX_CII] +
                        k[402]*y[IDX_CHII] + k[418]*y[IDX_CH2II] +
                        k[518]*y[IDX_H2II] + k[572]*y[IDX_SiII] +
                        k[673]*y[IDX_HeII] - k[975]*y[IDX_HI] + k[1142];
    IJth(jmatrix, 113, 109) = 0.0 + k[89]*y[IDX_HII] + k[414]*y[IDX_CHII] +
                        k[421]*y[IDX_CH2II] + k[443]*y[IDX_CH3II] +
                        k[526]*y[IDX_H2II] + k[598]*y[IDX_H3II] +
                        k[827]*y[IDX_NH2II] + k[830]*y[IDX_OHII] +
                        k[832]*y[IDX_SiHII] + k[833]*y[IDX_SiH2II] +
                        k[895]*y[IDX_CH2I] + k[895]*y[IDX_CH2I] +
                        k[896]*y[IDX_CH2I] + k[915]*y[IDX_CH3I] +
                        k[916]*y[IDX_CH3I] + k[939]*y[IDX_CHI] +
                        k[965]*y[IDX_H2I] + k[1046]*y[IDX_NHI] +
                        k[1065]*y[IDX_HCNI] + k[1066]*y[IDX_HCOI] +
                        k[1068]*y[IDX_HNOI] + k[1072]*y[IDX_NH2I] +
                        k[1080]*y[IDX_OHI] + k[1086]*y[IDX_SiH2I] +
                        k[1086]*y[IDX_SiH2I] + k[1087]*y[IDX_SiH3I] +
                        k[1089]*y[IDX_SiHI] - k[1207]*y[IDX_HI];
    IJth(jmatrix, 113, 110) = 0.0 + k[8]*y[IDX_H2I] + k[8]*y[IDX_H2I] +
                        k[294]*y[IDX_CHII] + k[296]*y[IDX_CH2II] +
                        k[296]*y[IDX_CH2II] + k[297]*y[IDX_CH2II] +
                        k[298]*y[IDX_CH3II] + k[300]*y[IDX_CH3II] +
                        k[300]*y[IDX_CH3II] + k[301]*y[IDX_CH4II] +
                        k[301]*y[IDX_CH4II] + k[302]*y[IDX_CH4II] +
                        k[305]*y[IDX_H2II] + k[305]*y[IDX_H2II] +
                        k[308]*y[IDX_H2COII] + k[308]*y[IDX_H2COII] +
                        k[309]*y[IDX_H2COII] + k[310]*y[IDX_H2NOII] +
                        k[313]*y[IDX_H2OII] + k[313]*y[IDX_H2OII] +
                        k[314]*y[IDX_H2OII] + k[315]*y[IDX_H3II] +
                        k[316]*y[IDX_H3II] + k[316]*y[IDX_H3II] +
                        k[316]*y[IDX_H3II] + k[319]*y[IDX_H3COII] +
                        k[320]*y[IDX_H3COII] + k[321]*y[IDX_H3COII] +
                        k[321]*y[IDX_H3COII] + k[322]*y[IDX_H3OII] +
                        k[323]*y[IDX_H3OII] + k[325]*y[IDX_H3OII] +
                        k[325]*y[IDX_H3OII] + k[326]*y[IDX_HCNII] +
                        k[327]*y[IDX_HCNHII] + k[327]*y[IDX_HCNHII] +
                        k[328]*y[IDX_HCNHII] + k[329]*y[IDX_HCNHII] +
                        k[330]*y[IDX_HCOII] + k[331]*y[IDX_HCO2II] +
                        k[332]*y[IDX_HCO2II] + k[334]*y[IDX_HNOII] +
                        k[335]*y[IDX_HOCII] + k[336]*y[IDX_HeHII] +
                        k[338]*y[IDX_N2HII] + k[340]*y[IDX_NHII] +
                        k[341]*y[IDX_NH2II] + k[341]*y[IDX_NH2II] +
                        k[342]*y[IDX_NH2II] + k[343]*y[IDX_NH3II] +
                        k[344]*y[IDX_NH3II] + k[344]*y[IDX_NH3II] +
                        k[347]*y[IDX_O2HII] + k[348]*y[IDX_OHII] +
                        k[352]*y[IDX_SiHII] + k[354]*y[IDX_SiH2II] +
                        k[354]*y[IDX_SiH2II] + k[355]*y[IDX_SiH2II] +
                        k[356]*y[IDX_SiH3II] + k[359]*y[IDX_SiH4II] +
                        k[361]*y[IDX_SiH5II] + k[364]*y[IDX_SiOHII] +
                        k[1216]*y[IDX_HII];
    IJth(jmatrix, 113, 111) = 0.0 + k[515]*y[IDX_H2II] - k[972]*y[IDX_HI] +
                        k[1092]*y[IDX_OHI];
    IJth(jmatrix, 113, 112) = 0.0 + k[2]*y[IDX_CHI] + k[3]*y[IDX_H2I] +
                        k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] +
                        k[4]*y[IDX_H2OI] + k[7]*y[IDX_OHI] + k[8]*y[IDX_EM] +
                        k[8]*y[IDX_EM] - k[10]*y[IDX_HI] + k[10]*y[IDX_HI] +
                        k[10]*y[IDX_HI] + k[10]*y[IDX_HI] + k[233] + k[235] +
                        k[235] + k[516]*y[IDX_H2II] + k[528]*y[IDX_CII] +
                        k[529]*y[IDX_CHII] + k[530]*y[IDX_CH2II] +
                        k[531]*y[IDX_CNII] + k[532]*y[IDX_COII] +
                        k[533]*y[IDX_COII] + k[534]*y[IDX_H2OII] +
                        k[535]*y[IDX_HCNII] + k[536]*y[IDX_HeII] +
                        k[538]*y[IDX_NII] + k[539]*y[IDX_N2II] +
                        k[541]*y[IDX_NHII] + k[542]*y[IDX_NH2II] +
                        k[543]*y[IDX_OII] + k[545]*y[IDX_OHII] +
                        k[546]*y[IDX_SiH4II] + k[547]*y[IDX_SiOII] +
                        k[955]*y[IDX_CI] + k[956]*y[IDX_CH2I] +
                        k[957]*y[IDX_CH3I] + k[958]*y[IDX_CHI] +
                        k[959]*y[IDX_CNI] + k[960]*y[IDX_NI] +
                        k[961]*y[IDX_NH2I] + k[962]*y[IDX_NHI] +
                        k[963]*y[IDX_O2I] + k[965]*y[IDX_OI] + k[966]*y[IDX_OHI];
    IJth(jmatrix, 113, 113) = 0.0 - k[9]*y[IDX_CHI] + k[9]*y[IDX_CHI] +
                        k[9]*y[IDX_CHI] - k[10]*y[IDX_H2I] + k[10]*y[IDX_H2I] +
                        k[10]*y[IDX_H2I] + k[10]*y[IDX_H2I] - k[11]*y[IDX_H2OI]
                        + k[11]*y[IDX_H2OI] + k[11]*y[IDX_H2OI] -
                        k[12]*y[IDX_O2I] + k[12]*y[IDX_O2I] - k[13]*y[IDX_OHI] +
                        k[13]*y[IDX_OHI] + k[13]*y[IDX_OHI] - k[126]*y[IDX_CNII]
                        - k[127]*y[IDX_COII] - k[128]*y[IDX_H2II] -
                        k[129]*y[IDX_HCNII] - k[130]*y[IDX_HeII] -
                        k[131]*y[IDX_OII] - k[236] - k[258] - k[614]*y[IDX_CHII]
                        - k[615]*y[IDX_CH2II] - k[616]*y[IDX_CH3II] -
                        k[617]*y[IDX_CH4II] - k[618]*y[IDX_HeHII] -
                        k[619]*y[IDX_SiHII] - k[967]*y[IDX_CH2I] -
                        k[968]*y[IDX_CH3I] - k[969]*y[IDX_CH4I] -
                        k[970]*y[IDX_CHI] - k[971]*y[IDX_CO2I] -
                        k[972]*y[IDX_COI] - k[973]*y[IDX_H2CNI] -
                        k[974]*y[IDX_H2COI] - k[975]*y[IDX_H2OI] -
                        k[976]*y[IDX_HCNI] - k[977]*y[IDX_HCOI] -
                        k[978]*y[IDX_HCOI] - k[979]*y[IDX_HNCI] +
                        k[979]*y[IDX_HNCI] - k[980]*y[IDX_HNOI] -
                        k[981]*y[IDX_HNOI] - k[982]*y[IDX_HNOI] -
                        k[983]*y[IDX_NH2I] - k[984]*y[IDX_NH3I] -
                        k[985]*y[IDX_NHI] - k[986]*y[IDX_NO2I] -
                        k[987]*y[IDX_NOI] - k[988]*y[IDX_NOI] -
                        k[989]*y[IDX_O2I] - k[990]*y[IDX_O2HI] -
                        k[991]*y[IDX_O2HI] - k[992]*y[IDX_O2HI] -
                        k[993]*y[IDX_OCNI] - k[994]*y[IDX_OCNI] -
                        k[995]*y[IDX_OCNI] - k[996]*y[IDX_OHI] -
                        k[1197]*y[IDX_HII] - k[1205]*y[IDX_CII] -
                        k[1206]*y[IDX_CI] - k[1207]*y[IDX_OI] -
                        k[1208]*y[IDX_OHI] - k[1209]*y[IDX_SiII];
    
    // clang-format on

    /* */

    return NAUNET_SUCCESS;
}