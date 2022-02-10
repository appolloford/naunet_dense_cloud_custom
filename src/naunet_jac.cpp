#include <math.h>
/* */
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>  // access to dense SUNMatrix
/* */
/*  */
#include "naunet_ode.h"
/*  */
#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_physics.h"

#define IJth(A, i, j)            SM_ELEMENT_D(A, i, j)
#define NVEC_CUDA_CONTENT(x)     ((N_VectorContent_Cuda)(x->content))
#define NVEC_CUDA_STREAM(x)      (NVEC_CUDA_CONTENT(x)->stream_exec_policy->stream())
#define NVEC_CUDA_BLOCKSIZE(x)   (NVEC_CUDA_CONTENT(x)->stream_exec_policy->blockSize())
#define NVEC_CUDA_GRIDSIZE(x, n) (NVEC_CUDA_CONTENT(x)->stream_exec_policy->gridSize(n))

/* */

int Jac(realtype t, N_Vector u, N_Vector fu, SUNMatrix jmatrix, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    /* */
    realtype *y            = N_VGetArrayPointer(u);
    NaunetData *u_data     = (NaunetData *)user_data;
    // clang-format off
    realtype rG = u_data->rG;
    realtype gdens = u_data->gdens;
    realtype sites = u_data->sites;
    realtype fr = u_data->fr;
    realtype opt_thd = u_data->opt_thd;
    realtype opt_crd = u_data->opt_crd;
    realtype opt_h2d = u_data->opt_h2d;
    realtype opt_uvd = u_data->opt_uvd;
    realtype crdeseff = u_data->crdeseff;
    realtype h2deseff = u_data->h2deseff;
    realtype nH = u_data->nH;
    realtype zeta = u_data->zeta;
    realtype Tgas = u_data->Tgas;
    realtype Av = u_data->Av;
    realtype omega = u_data->omega;
    realtype G0 = u_data->G0;
    realtype uvcreff = u_data->uvcreff;
    
        
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
    IJth(jmatrix, 0, 0) = 0.0 - k[1359] - k[1360] - k[1361] - k[1362];
    IJth(jmatrix, 0, 32) = 0.0 + k[1223];
    IJth(jmatrix, 0, 37) = 0.0 + k[1304];
    IJth(jmatrix, 0, 53) = 0.0 + k[1249];
    IJth(jmatrix, 1, 1) = 0.0 - k[1307] - k[1308] - k[1309] - k[1310];
    IJth(jmatrix, 1, 24) = 0.0 + k[1250];
    IJth(jmatrix, 1, 25) = 0.0 + k[1264];
    IJth(jmatrix, 1, 26) = 0.0 + k[1253];
    IJth(jmatrix, 1, 27) = 0.0 + k[1272];
    IJth(jmatrix, 1, 28) = 0.0 + k[1256];
    IJth(jmatrix, 1, 29) = 0.0 + k[1278];
    IJth(jmatrix, 1, 30) = 0.0 + k[1259];
    IJth(jmatrix, 1, 31) = 0.0 + k[1285];
    IJth(jmatrix, 1, 33) = 0.0 + k[1260];
    IJth(jmatrix, 1, 34) = 0.0 + k[1288];
    IJth(jmatrix, 2, 2) = 0.0 - k[1331] - k[1332] - k[1333] - k[1334];
    IJth(jmatrix, 2, 37) = 0.0 + k[1251];
    IJth(jmatrix, 2, 38) = 0.0 + k[1275];
    IJth(jmatrix, 3, 3) = 0.0 - k[1383] - k[1384] - k[1385] - k[1386];
    IJth(jmatrix, 3, 39) = 0.0 + k[1258];
    IJth(jmatrix, 3, 60) = 0.0 + k[1287];
    IJth(jmatrix, 4, 4) = 0.0 - k[1339] - k[1340] - k[1341] - k[1342];
    IJth(jmatrix, 4, 45) = 0.0 + k[1298];
    IJth(jmatrix, 5, 5) = 0.0 - k[1347] - k[1348] - k[1349] - k[1350];
    IJth(jmatrix, 5, 37) = 0.0 + k[1227];
    IJth(jmatrix, 5, 46) = 0.0 + k[1252];
    IJth(jmatrix, 5, 47) = 0.0 + k[1284];
    IJth(jmatrix, 5, 58) = 0.0 + k[1261];
    IJth(jmatrix, 5, 59) = 0.0 + k[1281];
    IJth(jmatrix, 5, 68) = 0.0 + k[1226];
    IJth(jmatrix, 6, 6) = 0.0 - k[1315] - k[1316] - k[1317] - k[1318];
    IJth(jmatrix, 6, 49) = 0.0 + k[1257];
    IJth(jmatrix, 6, 50) = 0.0 + k[1280];
    IJth(jmatrix, 6, 54) = 0.0 + k[1286];
    IJth(jmatrix, 6, 85) = 0.0 + k[1291];
    IJth(jmatrix, 6, 86) = 0.0 + k[1269];
    IJth(jmatrix, 6, 92) = 0.0 + k[1254];
    IJth(jmatrix, 6, 93) = 0.0 + k[1274];
    IJth(jmatrix, 7, 7) = 0.0 - k[1391] - k[1392] - k[1393] - k[1394];
    IJth(jmatrix, 7, 51) = 0.0 + k[1247];
    IJth(jmatrix, 7, 111) = 0.0 + k[1229];
    IJth(jmatrix, 7, 112) = 0.0 + k[1245];
    IJth(jmatrix, 7, 113) = 0.0 + k[1246];
    IJth(jmatrix, 8, 8) = 0.0 - k[1323] - k[1324] - k[1325] - k[1326];
    IJth(jmatrix, 8, 35) = 0.0 + k[1263];
    IJth(jmatrix, 8, 36) = 0.0 + k[1276];
    IJth(jmatrix, 8, 55) = 0.0 + k[1266];
    IJth(jmatrix, 8, 56) = 0.0 + k[1282];
    IJth(jmatrix, 8, 57) = 0.0 + k[1301];
    IJth(jmatrix, 9, 9) = 0.0 - k[1327] - k[1328] - k[1329] - k[1330];
    IJth(jmatrix, 9, 64) = 0.0 + k[1305];
    IJth(jmatrix, 10, 10) = 0.0 - k[1375] - k[1376] - k[1377] - k[1378];
    IJth(jmatrix, 10, 65) = 0.0 + k[1224];
    IJth(jmatrix, 10, 91) = 0.0 + k[1225];
    IJth(jmatrix, 11, 11) = 0.0 - k[1351] - k[1352] - k[1353] - k[1354];
    IJth(jmatrix, 11, 48) = 0.0 + k[1296];
    IJth(jmatrix, 11, 66) = 0.0 + k[1294];
    IJth(jmatrix, 11, 67) = 0.0 + k[1295];
    IJth(jmatrix, 12, 12) = 0.0 - k[1319] - k[1320] - k[1321] - k[1322];
    IJth(jmatrix, 12, 69) = 0.0 + k[1300];
    IJth(jmatrix, 12, 70) = 0.0 + k[1299];
    IJth(jmatrix, 13, 13) = 0.0 - k[1335] - k[1336] - k[1337] - k[1338];
    IJth(jmatrix, 13, 73) = 0.0 + k[1262];
    IJth(jmatrix, 13, 74) = 0.0 + k[1271];
    IJth(jmatrix, 13, 75) = 0.0 + k[1302];
    IJth(jmatrix, 14, 14) = 0.0 - k[1311] - k[1312] - k[1313] - k[1314];
    IJth(jmatrix, 14, 71) = 0.0 + k[1290];
    IJth(jmatrix, 14, 72) = 0.0 + k[1268];
    IJth(jmatrix, 14, 76) = 0.0 + k[1265];
    IJth(jmatrix, 14, 77) = 0.0 + k[1273];
    IJth(jmatrix, 14, 78) = 0.0 + k[1289];
    IJth(jmatrix, 14, 79) = 0.0 + k[1279];
    IJth(jmatrix, 14, 80) = 0.0 + k[1267];
    IJth(jmatrix, 14, 81) = 0.0 + k[1283];
    IJth(jmatrix, 15, 15) = 0.0 - k[1343] - k[1344] - k[1345] - k[1346];
    IJth(jmatrix, 15, 82) = 0.0 + k[1255];
    IJth(jmatrix, 15, 83) = 0.0 + k[1277];
    IJth(jmatrix, 16, 16) = 0.0 - k[1387] - k[1388] - k[1389] - k[1390];
    IJth(jmatrix, 16, 84) = 0.0 + k[1293];
    IJth(jmatrix, 17, 17) = 0.0 - k[1355] - k[1356] - k[1357] - k[1358];
    IJth(jmatrix, 17, 87) = 0.0 + k[1292];
    IJth(jmatrix, 17, 88) = 0.0 + k[1270];
    IJth(jmatrix, 18, 18) = 0.0 - k[1367] - k[1368] - k[1369] - k[1370];
    IJth(jmatrix, 18, 89) = 0.0 + k[1303];
    IJth(jmatrix, 18, 90) = 0.0 + k[1297];
    IJth(jmatrix, 19, 19) = 0.0 - k[1371] - k[1372] - k[1373] - k[1374];
    IJth(jmatrix, 19, 96) = 0.0 + k[1239];
    IJth(jmatrix, 19, 97) = 0.0 + k[1241];
    IJth(jmatrix, 20, 20) = 0.0 - k[1395] - k[1396] - k[1397] - k[1398];
    IJth(jmatrix, 20, 98) = 0.0 + k[1240];
    IJth(jmatrix, 20, 99) = 0.0 + k[1242];
    IJth(jmatrix, 21, 21) = 0.0 - k[1399] - k[1400] - k[1401] - k[1402];
    IJth(jmatrix, 21, 100) = 0.0 + k[1243];
    IJth(jmatrix, 21, 101) = 0.0 + k[1244];
    IJth(jmatrix, 22, 22) = 0.0 - k[1363] - k[1364] - k[1365] - k[1366];
    IJth(jmatrix, 22, 94) = 0.0 + k[1228];
    IJth(jmatrix, 22, 95) = 0.0 + k[1231];
    IJth(jmatrix, 22, 102) = 0.0 + k[1230];
    IJth(jmatrix, 22, 103) = 0.0 + k[1232];
    IJth(jmatrix, 22, 104) = 0.0 + k[1233];
    IJth(jmatrix, 22, 105) = 0.0 + k[1234];
    IJth(jmatrix, 22, 106) = 0.0 + k[1235];
    IJth(jmatrix, 22, 107) = 0.0 + k[1236];
    IJth(jmatrix, 22, 108) = 0.0 + k[1237];
    IJth(jmatrix, 22, 109) = 0.0 + k[1238];
    IJth(jmatrix, 22, 110) = 0.0 + k[1248];
    IJth(jmatrix, 23, 23) = 0.0 - k[1379] - k[1380] - k[1381] - k[1382];
    IJth(jmatrix, 24, 24) = 0.0 - k[27]*y[IDX_CNII] - k[28]*y[IDX_COII] -
        k[29]*y[IDX_N2II] - k[30]*y[IDX_O2II] - k[139]*y[IDX_HeII] - k[231] -
        k[240] - k[383]*y[IDX_H2OII] - k[384]*y[IDX_H3OII] - k[385]*y[IDX_HCNII]
        - k[386]*y[IDX_HCOII] - k[387]*y[IDX_HCO2II] - k[388]*y[IDX_HNOII] -
        k[389]*y[IDX_N2HII] - k[390]*y[IDX_NHII] - k[391]*y[IDX_O2II] -
        k[392]*y[IDX_O2HII] - k[393]*y[IDX_OHII] - k[394]*y[IDX_SiHII] -
        k[395]*y[IDX_SiOII] - k[509]*y[IDX_H2II] - k[576]*y[IDX_H3II] -
        k[863]*y[IDX_CH2I] - k[864]*y[IDX_HCOI] - k[865]*y[IDX_N2I] -
        k[866]*y[IDX_NH2I] - k[867]*y[IDX_NH2I] - k[868]*y[IDX_NH2I] -
        k[869]*y[IDX_NHI] - k[870]*y[IDX_NHI] - k[871]*y[IDX_NOI] -
        k[872]*y[IDX_NOI] - k[873]*y[IDX_O2I] - k[874]*y[IDX_OCNI] -
        k[875]*y[IDX_OHI] - k[876]*y[IDX_OHI] - k[877]*y[IDX_SiHI] -
        k[955]*y[IDX_H2I] - k[1004]*y[IDX_HNCOI] - k[1107] - k[1194]*y[IDX_NI] -
        k[1195]*y[IDX_OII] - k[1196]*y[IDX_OI] - k[1200]*y[IDX_H2I] -
        k[1206]*y[IDX_HI] - k[1250];
    IJth(jmatrix, 24, 25) = 0.0 + k[14]*y[IDX_CH2I] + k[15]*y[IDX_CHI] +
        k[16]*y[IDX_H2COI] + k[17]*y[IDX_HCOI] + k[18]*y[IDX_MgI] +
        k[19]*y[IDX_NH3I] + k[20]*y[IDX_NOI] + k[21]*y[IDX_SiI] +
        k[22]*y[IDX_SiC2I] + k[23]*y[IDX_SiC3I] + k[24]*y[IDX_SiCI] +
        k[25]*y[IDX_SiH2I] + k[26]*y[IDX_SiH3I] + k[1214]*y[IDX_EM];
    IJth(jmatrix, 24, 26) = 0.0 + k[2]*y[IDX_H2I] + k[9]*y[IDX_HI] +
        k[15]*y[IDX_CII] + k[250] + k[458]*y[IDX_COII] + k[929]*y[IDX_NI] +
        k[940]*y[IDX_OI] + k[970]*y[IDX_HI] + k[1128];
    IJth(jmatrix, 24, 27) = 0.0 + k[294]*y[IDX_EM] + k[400]*y[IDX_H2COI] +
        k[403]*y[IDX_H2OI] + k[405]*y[IDX_HCNI] + k[407]*y[IDX_HNCI] + k[1108];
    IJth(jmatrix, 24, 28) = 0.0 + k[14]*y[IDX_CII] - k[863]*y[IDX_CI];
    IJth(jmatrix, 24, 29) = 0.0 + k[295]*y[IDX_EM] + k[296]*y[IDX_EM];
    IJth(jmatrix, 24, 35) = 0.0 + k[251] + k[663]*y[IDX_HeII] +
        k[811]*y[IDX_OII] + k[1011]*y[IDX_NI] + k[1058]*y[IDX_OI] + k[1130];
    IJth(jmatrix, 24, 36) = 0.0 - k[27]*y[IDX_CI] + k[303]*y[IDX_EM] +
        k[737]*y[IDX_NI];
    IJth(jmatrix, 24, 37) = 0.0 + k[253] + k[720]*y[IDX_NII] +
        k[972]*y[IDX_HI] + k[1104]*y[IDX_SiI] + k[1133];
    IJth(jmatrix, 24, 38) = 0.0 - k[28]*y[IDX_CI] + k[304]*y[IDX_EM] +
        k[458]*y[IDX_CHI];
    IJth(jmatrix, 24, 39) = 0.0 + k[667]*y[IDX_HeII];
    IJth(jmatrix, 24, 40) = 0.0 + k[294]*y[IDX_CHII] + k[295]*y[IDX_CH2II] +
        k[296]*y[IDX_CH2II] + k[303]*y[IDX_CNII] + k[304]*y[IDX_COII] +
        k[349]*y[IDX_SiCII] + k[350]*y[IDX_SiC2II] + k[351]*y[IDX_SiC3II] +
        k[1214]*y[IDX_CII];
    IJth(jmatrix, 24, 41) = 0.0 + k[9]*y[IDX_CHI] + k[970]*y[IDX_CHI] +
        k[972]*y[IDX_COI] - k[1206]*y[IDX_CI];
    IJth(jmatrix, 24, 43) = 0.0 + k[2]*y[IDX_CHI] - k[955]*y[IDX_CI] -
        k[1200]*y[IDX_CI];
    IJth(jmatrix, 24, 44) = 0.0 - k[509]*y[IDX_CI];
    IJth(jmatrix, 24, 46) = 0.0 + k[16]*y[IDX_CII] + k[400]*y[IDX_CHII];
    IJth(jmatrix, 24, 49) = 0.0 + k[403]*y[IDX_CHII];
    IJth(jmatrix, 24, 50) = 0.0 - k[383]*y[IDX_CI];
    IJth(jmatrix, 24, 52) = 0.0 - k[576]*y[IDX_CI];
    IJth(jmatrix, 24, 54) = 0.0 - k[384]*y[IDX_CI];
    IJth(jmatrix, 24, 55) = 0.0 + k[405]*y[IDX_CHII];
    IJth(jmatrix, 24, 56) = 0.0 - k[385]*y[IDX_CI];
    IJth(jmatrix, 24, 58) = 0.0 + k[17]*y[IDX_CII] - k[864]*y[IDX_CI];
    IJth(jmatrix, 24, 59) = 0.0 - k[386]*y[IDX_CI];
    IJth(jmatrix, 24, 60) = 0.0 - k[387]*y[IDX_CI];
    IJth(jmatrix, 24, 62) = 0.0 - k[139]*y[IDX_CI] + k[663]*y[IDX_CNI] +
        k[667]*y[IDX_CO2I] + k[685]*y[IDX_HNCI] + k[700]*y[IDX_SiC3I] +
        k[701]*y[IDX_SiCI];
    IJth(jmatrix, 24, 64) = 0.0 + k[407]*y[IDX_CHII] + k[685]*y[IDX_HeII];
    IJth(jmatrix, 24, 65) = 0.0 - k[1004]*y[IDX_CI];
    IJth(jmatrix, 24, 67) = 0.0 - k[388]*y[IDX_CI];
    IJth(jmatrix, 24, 69) = 0.0 + k[18]*y[IDX_CII];
    IJth(jmatrix, 24, 71) = 0.0 + k[737]*y[IDX_CNII] + k[929]*y[IDX_CHI] +
        k[1011]*y[IDX_CNI] - k[1194]*y[IDX_CI];
    IJth(jmatrix, 24, 72) = 0.0 + k[720]*y[IDX_COI];
    IJth(jmatrix, 24, 73) = 0.0 - k[865]*y[IDX_CI];
    IJth(jmatrix, 24, 74) = 0.0 - k[29]*y[IDX_CI];
    IJth(jmatrix, 24, 75) = 0.0 - k[389]*y[IDX_CI];
    IJth(jmatrix, 24, 76) = 0.0 - k[869]*y[IDX_CI] - k[870]*y[IDX_CI];
    IJth(jmatrix, 24, 77) = 0.0 - k[390]*y[IDX_CI];
    IJth(jmatrix, 24, 78) = 0.0 - k[866]*y[IDX_CI] - k[867]*y[IDX_CI] -
        k[868]*y[IDX_CI];
    IJth(jmatrix, 24, 80) = 0.0 + k[19]*y[IDX_CII];
    IJth(jmatrix, 24, 82) = 0.0 + k[20]*y[IDX_CII] - k[871]*y[IDX_CI] -
        k[872]*y[IDX_CI];
    IJth(jmatrix, 24, 85) = 0.0 + k[831]*y[IDX_SiCII] + k[940]*y[IDX_CHI] +
        k[1058]*y[IDX_CNI] + k[1084]*y[IDX_SiCI] - k[1196]*y[IDX_CI];
    IJth(jmatrix, 24, 86) = 0.0 + k[811]*y[IDX_CNI] - k[1195]*y[IDX_CI];
    IJth(jmatrix, 24, 87) = 0.0 - k[873]*y[IDX_CI];
    IJth(jmatrix, 24, 88) = 0.0 - k[30]*y[IDX_CI] - k[391]*y[IDX_CI];
    IJth(jmatrix, 24, 90) = 0.0 - k[392]*y[IDX_CI];
    IJth(jmatrix, 24, 91) = 0.0 - k[874]*y[IDX_CI];
    IJth(jmatrix, 24, 92) = 0.0 - k[875]*y[IDX_CI] - k[876]*y[IDX_CI];
    IJth(jmatrix, 24, 93) = 0.0 - k[393]*y[IDX_CI];
    IJth(jmatrix, 24, 94) = 0.0 + k[21]*y[IDX_CII] + k[1104]*y[IDX_COI];
    IJth(jmatrix, 24, 96) = 0.0 + k[24]*y[IDX_CII] + k[288] +
        k[701]*y[IDX_HeII] + k[1084]*y[IDX_OI] + k[1178];
    IJth(jmatrix, 24, 97) = 0.0 + k[349]*y[IDX_EM] + k[831]*y[IDX_OI];
    IJth(jmatrix, 24, 98) = 0.0 + k[22]*y[IDX_CII] + k[286];
    IJth(jmatrix, 24, 99) = 0.0 + k[350]*y[IDX_EM];
    IJth(jmatrix, 24, 100) = 0.0 + k[23]*y[IDX_CII] + k[287] +
        k[700]*y[IDX_HeII] + k[1177];
    IJth(jmatrix, 24, 101) = 0.0 + k[351]*y[IDX_EM];
    IJth(jmatrix, 24, 102) = 0.0 - k[877]*y[IDX_CI];
    IJth(jmatrix, 24, 103) = 0.0 - k[394]*y[IDX_CI];
    IJth(jmatrix, 24, 104) = 0.0 + k[25]*y[IDX_CII];
    IJth(jmatrix, 24, 106) = 0.0 + k[26]*y[IDX_CII];
    IJth(jmatrix, 24, 112) = 0.0 - k[395]*y[IDX_CI];
    IJth(jmatrix, 25, 24) = 0.0 + k[27]*y[IDX_CNII] + k[28]*y[IDX_COII] +
        k[29]*y[IDX_N2II] + k[30]*y[IDX_O2II] + k[139]*y[IDX_HeII] + k[231] +
        k[240] + k[1107];
    IJth(jmatrix, 25, 25) = 0.0 - k[14]*y[IDX_CH2I] - k[15]*y[IDX_CHI] -
        k[16]*y[IDX_H2COI] - k[17]*y[IDX_HCOI] - k[18]*y[IDX_MgI] -
        k[19]*y[IDX_NH3I] - k[20]*y[IDX_NOI] - k[21]*y[IDX_SiI] -
        k[22]*y[IDX_SiC2I] - k[23]*y[IDX_SiC3I] - k[24]*y[IDX_SiCI] -
        k[25]*y[IDX_SiH2I] - k[26]*y[IDX_SiH3I] - k[365]*y[IDX_CH3OHI] -
        k[366]*y[IDX_CH3OHI] - k[367]*y[IDX_CO2I] - k[368]*y[IDX_H2COI] -
        k[369]*y[IDX_H2COI] - k[370]*y[IDX_H2OI] - k[371]*y[IDX_H2OI] -
        k[372]*y[IDX_HCOI] - k[373]*y[IDX_NH2I] - k[374]*y[IDX_NH3I] -
        k[375]*y[IDX_NHI] - k[376]*y[IDX_O2I] - k[377]*y[IDX_O2I] -
        k[378]*y[IDX_OCNI] - k[379]*y[IDX_OHI] - k[380]*y[IDX_SiH2I] -
        k[381]*y[IDX_SiHI] - k[382]*y[IDX_SiOI] - k[528]*y[IDX_H2I] -
        k[1192]*y[IDX_NI] - k[1193]*y[IDX_OI] - k[1199]*y[IDX_H2I] -
        k[1205]*y[IDX_HI] - k[1214]*y[IDX_EM] - k[1264];
    IJth(jmatrix, 25, 26) = 0.0 - k[15]*y[IDX_CII] + k[662]*y[IDX_HeII];
    IJth(jmatrix, 25, 27) = 0.0 + k[241] + k[614]*y[IDX_HI];
    IJth(jmatrix, 25, 28) = 0.0 - k[14]*y[IDX_CII] + k[653]*y[IDX_HeII];
    IJth(jmatrix, 25, 29) = 0.0 + k[1109];
    IJth(jmatrix, 25, 32) = 0.0 - k[365]*y[IDX_CII] - k[366]*y[IDX_CII];
    IJth(jmatrix, 25, 35) = 0.0 + k[664]*y[IDX_HeII];
    IJth(jmatrix, 25, 36) = 0.0 + k[27]*y[IDX_CI];
    IJth(jmatrix, 25, 37) = 0.0 + k[669]*y[IDX_HeII];
    IJth(jmatrix, 25, 38) = 0.0 + k[28]*y[IDX_CI] + k[1131];
    IJth(jmatrix, 25, 39) = 0.0 - k[367]*y[IDX_CII] + k[668]*y[IDX_HeII];
    IJth(jmatrix, 25, 40) = 0.0 - k[1214]*y[IDX_CII];
    IJth(jmatrix, 25, 41) = 0.0 + k[614]*y[IDX_CHII] - k[1205]*y[IDX_CII];
    IJth(jmatrix, 25, 43) = 0.0 - k[528]*y[IDX_CII] - k[1199]*y[IDX_CII];
    IJth(jmatrix, 25, 46) = 0.0 - k[16]*y[IDX_CII] - k[368]*y[IDX_CII] -
        k[369]*y[IDX_CII];
    IJth(jmatrix, 25, 49) = 0.0 - k[370]*y[IDX_CII] - k[371]*y[IDX_CII];
    IJth(jmatrix, 25, 55) = 0.0 + k[678]*y[IDX_HeII];
    IJth(jmatrix, 25, 58) = 0.0 - k[17]*y[IDX_CII] - k[372]*y[IDX_CII];
    IJth(jmatrix, 25, 62) = 0.0 + k[139]*y[IDX_CI] + k[653]*y[IDX_CH2I] +
        k[662]*y[IDX_CHI] + k[664]*y[IDX_CNI] + k[668]*y[IDX_CO2I] +
        k[669]*y[IDX_COI] + k[678]*y[IDX_HCNI] + k[684]*y[IDX_HNCI] +
        k[702]*y[IDX_SiCI];
    IJth(jmatrix, 25, 64) = 0.0 + k[684]*y[IDX_HeII];
    IJth(jmatrix, 25, 69) = 0.0 - k[18]*y[IDX_CII];
    IJth(jmatrix, 25, 71) = 0.0 - k[1192]*y[IDX_CII];
    IJth(jmatrix, 25, 74) = 0.0 + k[29]*y[IDX_CI];
    IJth(jmatrix, 25, 76) = 0.0 - k[375]*y[IDX_CII];
    IJth(jmatrix, 25, 78) = 0.0 - k[373]*y[IDX_CII];
    IJth(jmatrix, 25, 80) = 0.0 - k[19]*y[IDX_CII] - k[374]*y[IDX_CII];
    IJth(jmatrix, 25, 82) = 0.0 - k[20]*y[IDX_CII];
    IJth(jmatrix, 25, 85) = 0.0 - k[1193]*y[IDX_CII];
    IJth(jmatrix, 25, 87) = 0.0 - k[376]*y[IDX_CII] - k[377]*y[IDX_CII];
    IJth(jmatrix, 25, 88) = 0.0 + k[30]*y[IDX_CI];
    IJth(jmatrix, 25, 91) = 0.0 - k[378]*y[IDX_CII];
    IJth(jmatrix, 25, 92) = 0.0 - k[379]*y[IDX_CII];
    IJth(jmatrix, 25, 94) = 0.0 - k[21]*y[IDX_CII];
    IJth(jmatrix, 25, 96) = 0.0 - k[24]*y[IDX_CII] + k[702]*y[IDX_HeII];
    IJth(jmatrix, 25, 98) = 0.0 - k[22]*y[IDX_CII];
    IJth(jmatrix, 25, 100) = 0.0 - k[23]*y[IDX_CII];
    IJth(jmatrix, 25, 102) = 0.0 - k[381]*y[IDX_CII];
    IJth(jmatrix, 25, 104) = 0.0 - k[25]*y[IDX_CII] - k[380]*y[IDX_CII];
    IJth(jmatrix, 25, 106) = 0.0 - k[26]*y[IDX_CII];
    IJth(jmatrix, 25, 111) = 0.0 - k[382]*y[IDX_CII];
    IJth(jmatrix, 26, 24) = 0.0 + k[863]*y[IDX_CH2I] + k[863]*y[IDX_CH2I] +
        k[864]*y[IDX_HCOI] + k[868]*y[IDX_NH2I] + k[870]*y[IDX_NHI] +
        k[876]*y[IDX_OHI] + k[955]*y[IDX_H2I] + k[1206]*y[IDX_HI];
    IJth(jmatrix, 26, 25) = 0.0 - k[15]*y[IDX_CHI] + k[365]*y[IDX_CH3OHI] +
        k[369]*y[IDX_H2COI];
    IJth(jmatrix, 26, 26) = 0.0 - k[0]*y[IDX_OI] - k[2]*y[IDX_H2I] -
        k[9]*y[IDX_HI] - k[15]*y[IDX_CII] - k[53]*y[IDX_CNII] -
        k[54]*y[IDX_COII] - k[55]*y[IDX_H2COII] - k[56]*y[IDX_H2OII] -
        k[57]*y[IDX_NII] - k[58]*y[IDX_N2II] - k[59]*y[IDX_NH2II] -
        k[60]*y[IDX_OII] - k[61]*y[IDX_O2II] - k[62]*y[IDX_OHII] -
        k[78]*y[IDX_HII] - k[102]*y[IDX_H2II] - k[141]*y[IDX_HeII] - k[250] -
        k[458]*y[IDX_COII] - k[459]*y[IDX_H2COII] - k[460]*y[IDX_H2OII] -
        k[461]*y[IDX_H3COII] - k[462]*y[IDX_H3OII] - k[463]*y[IDX_HCNII] -
        k[464]*y[IDX_HCNHII] - k[465]*y[IDX_HCNHII] - k[466]*y[IDX_HCOII] -
        k[467]*y[IDX_HNOII] - k[468]*y[IDX_NII] - k[469]*y[IDX_N2HII] -
        k[470]*y[IDX_NHII] - k[471]*y[IDX_NH2II] - k[472]*y[IDX_OII] -
        k[473]*y[IDX_O2II] - k[474]*y[IDX_O2HII] - k[475]*y[IDX_OHII] -
        k[476]*y[IDX_SiII] - k[477]*y[IDX_SiHII] - k[478]*y[IDX_SiOII] -
        k[512]*y[IDX_H2II] - k[580]*y[IDX_H3II] - k[662]*y[IDX_HeII] -
        k[923]*y[IDX_CO2I] - k[924]*y[IDX_H2COI] - k[925]*y[IDX_HCOI] -
        k[926]*y[IDX_HNOI] - k[927]*y[IDX_N2I] - k[928]*y[IDX_NI] -
        k[929]*y[IDX_NI] - k[930]*y[IDX_NOI] - k[931]*y[IDX_NOI] -
        k[932]*y[IDX_NOI] - k[933]*y[IDX_O2I] - k[934]*y[IDX_O2I] -
        k[935]*y[IDX_O2I] - k[936]*y[IDX_O2I] - k[937]*y[IDX_O2HI] -
        k[938]*y[IDX_O2HI] - k[939]*y[IDX_OI] - k[940]*y[IDX_OI] -
        k[941]*y[IDX_OHI] - k[958]*y[IDX_H2I] - k[970]*y[IDX_HI] - k[1128] -
        k[1129] - k[1201]*y[IDX_H2I] - k[1253];
    IJth(jmatrix, 26, 27) = 0.0 + k[31]*y[IDX_HCOI] + k[32]*y[IDX_MgI] +
        k[33]*y[IDX_NH3I] + k[34]*y[IDX_NOI] + k[35]*y[IDX_SiI];
    IJth(jmatrix, 26, 28) = 0.0 + k[243] + k[422]*y[IDX_COII] +
        k[863]*y[IDX_CI] + k[863]*y[IDX_CI] + k[878]*y[IDX_CH2I] +
        k[878]*y[IDX_CH2I] + k[880]*y[IDX_CNI] + k[897]*y[IDX_OI] +
        k[899]*y[IDX_OHI] + k[967]*y[IDX_HI] + k[1007]*y[IDX_NI] + k[1113];
    IJth(jmatrix, 26, 29) = 0.0 + k[297]*y[IDX_EM] + k[1111];
    IJth(jmatrix, 26, 30) = 0.0 + k[246] + k[1118];
    IJth(jmatrix, 26, 31) = 0.0 + k[299]*y[IDX_EM] + k[300]*y[IDX_EM];
    IJth(jmatrix, 26, 32) = 0.0 + k[365]*y[IDX_CII];
    IJth(jmatrix, 26, 33) = 0.0 + k[1127];
    IJth(jmatrix, 26, 35) = 0.0 + k[880]*y[IDX_CH2I];
    IJth(jmatrix, 26, 36) = 0.0 - k[53]*y[IDX_CHI];
    IJth(jmatrix, 26, 38) = 0.0 - k[54]*y[IDX_CHI] + k[422]*y[IDX_CH2I] -
        k[458]*y[IDX_CHI];
    IJth(jmatrix, 26, 39) = 0.0 - k[923]*y[IDX_CHI];
    IJth(jmatrix, 26, 40) = 0.0 + k[297]*y[IDX_CH2II] + k[299]*y[IDX_CH3II]
        + k[300]*y[IDX_CH3II] + k[318]*y[IDX_H3COII];
    IJth(jmatrix, 26, 41) = 0.0 - k[9]*y[IDX_CHI] + k[967]*y[IDX_CH2I] -
        k[970]*y[IDX_CHI] + k[1206]*y[IDX_CI];
    IJth(jmatrix, 26, 42) = 0.0 - k[78]*y[IDX_CHI];
    IJth(jmatrix, 26, 43) = 0.0 - k[2]*y[IDX_CHI] + k[955]*y[IDX_CI] -
        k[958]*y[IDX_CHI] - k[1201]*y[IDX_CHI];
    IJth(jmatrix, 26, 44) = 0.0 - k[102]*y[IDX_CHI] - k[512]*y[IDX_CHI];
    IJth(jmatrix, 26, 46) = 0.0 + k[369]*y[IDX_CII] - k[924]*y[IDX_CHI];
    IJth(jmatrix, 26, 47) = 0.0 - k[55]*y[IDX_CHI] - k[459]*y[IDX_CHI];
    IJth(jmatrix, 26, 50) = 0.0 - k[56]*y[IDX_CHI] - k[460]*y[IDX_CHI];
    IJth(jmatrix, 26, 52) = 0.0 - k[580]*y[IDX_CHI];
    IJth(jmatrix, 26, 53) = 0.0 + k[318]*y[IDX_EM] - k[461]*y[IDX_CHI];
    IJth(jmatrix, 26, 54) = 0.0 - k[462]*y[IDX_CHI];
    IJth(jmatrix, 26, 55) = 0.0 + k[677]*y[IDX_HeII] + k[815]*y[IDX_OII];
    IJth(jmatrix, 26, 56) = 0.0 - k[463]*y[IDX_CHI];
    IJth(jmatrix, 26, 57) = 0.0 - k[464]*y[IDX_CHI] - k[465]*y[IDX_CHI];
    IJth(jmatrix, 26, 58) = 0.0 + k[31]*y[IDX_CHII] + k[864]*y[IDX_CI] -
        k[925]*y[IDX_CHI];
    IJth(jmatrix, 26, 59) = 0.0 - k[466]*y[IDX_CHI];
    IJth(jmatrix, 26, 62) = 0.0 - k[141]*y[IDX_CHI] - k[662]*y[IDX_CHI] +
        k[677]*y[IDX_HCNI];
    IJth(jmatrix, 26, 66) = 0.0 - k[926]*y[IDX_CHI];
    IJth(jmatrix, 26, 67) = 0.0 - k[467]*y[IDX_CHI];
    IJth(jmatrix, 26, 69) = 0.0 + k[32]*y[IDX_CHII];
    IJth(jmatrix, 26, 71) = 0.0 - k[928]*y[IDX_CHI] - k[929]*y[IDX_CHI] +
        k[1007]*y[IDX_CH2I];
    IJth(jmatrix, 26, 72) = 0.0 - k[57]*y[IDX_CHI] - k[468]*y[IDX_CHI];
    IJth(jmatrix, 26, 73) = 0.0 - k[927]*y[IDX_CHI];
    IJth(jmatrix, 26, 74) = 0.0 - k[58]*y[IDX_CHI];
    IJth(jmatrix, 26, 75) = 0.0 - k[469]*y[IDX_CHI];
    IJth(jmatrix, 26, 76) = 0.0 + k[870]*y[IDX_CI];
    IJth(jmatrix, 26, 77) = 0.0 - k[470]*y[IDX_CHI];
    IJth(jmatrix, 26, 78) = 0.0 + k[868]*y[IDX_CI];
    IJth(jmatrix, 26, 79) = 0.0 - k[59]*y[IDX_CHI] - k[471]*y[IDX_CHI];
    IJth(jmatrix, 26, 80) = 0.0 + k[33]*y[IDX_CHII];
    IJth(jmatrix, 26, 82) = 0.0 + k[34]*y[IDX_CHII] - k[930]*y[IDX_CHI] -
        k[931]*y[IDX_CHI] - k[932]*y[IDX_CHI];
    IJth(jmatrix, 26, 85) = 0.0 - k[0]*y[IDX_CHI] + k[897]*y[IDX_CH2I] -
        k[939]*y[IDX_CHI] - k[940]*y[IDX_CHI];
    IJth(jmatrix, 26, 86) = 0.0 - k[60]*y[IDX_CHI] - k[472]*y[IDX_CHI] +
        k[815]*y[IDX_HCNI];
    IJth(jmatrix, 26, 87) = 0.0 - k[933]*y[IDX_CHI] - k[934]*y[IDX_CHI] -
        k[935]*y[IDX_CHI] - k[936]*y[IDX_CHI];
    IJth(jmatrix, 26, 88) = 0.0 - k[61]*y[IDX_CHI] - k[473]*y[IDX_CHI];
    IJth(jmatrix, 26, 89) = 0.0 - k[937]*y[IDX_CHI] - k[938]*y[IDX_CHI];
    IJth(jmatrix, 26, 90) = 0.0 - k[474]*y[IDX_CHI];
    IJth(jmatrix, 26, 92) = 0.0 + k[876]*y[IDX_CI] + k[899]*y[IDX_CH2I] -
        k[941]*y[IDX_CHI];
    IJth(jmatrix, 26, 93) = 0.0 - k[62]*y[IDX_CHI] - k[475]*y[IDX_CHI];
    IJth(jmatrix, 26, 94) = 0.0 + k[35]*y[IDX_CHII];
    IJth(jmatrix, 26, 95) = 0.0 - k[476]*y[IDX_CHI];
    IJth(jmatrix, 26, 103) = 0.0 - k[477]*y[IDX_CHI];
    IJth(jmatrix, 26, 112) = 0.0 - k[478]*y[IDX_CHI];
    IJth(jmatrix, 27, 24) = 0.0 + k[383]*y[IDX_H2OII] + k[385]*y[IDX_HCNII]
        + k[386]*y[IDX_HCOII] + k[387]*y[IDX_HCO2II] + k[388]*y[IDX_HNOII] +
        k[389]*y[IDX_N2HII] + k[390]*y[IDX_NHII] + k[392]*y[IDX_O2HII] +
        k[393]*y[IDX_OHII] + k[509]*y[IDX_H2II] + k[576]*y[IDX_H3II];
    IJth(jmatrix, 27, 25) = 0.0 + k[15]*y[IDX_CHI] + k[372]*y[IDX_HCOI] +
        k[528]*y[IDX_H2I] + k[1205]*y[IDX_HI];
    IJth(jmatrix, 27, 26) = 0.0 + k[15]*y[IDX_CII] + k[53]*y[IDX_CNII] +
        k[54]*y[IDX_COII] + k[55]*y[IDX_H2COII] + k[56]*y[IDX_H2OII] +
        k[57]*y[IDX_NII] + k[58]*y[IDX_N2II] + k[59]*y[IDX_NH2II] +
        k[60]*y[IDX_OII] + k[61]*y[IDX_O2II] + k[62]*y[IDX_OHII] +
        k[78]*y[IDX_HII] + k[102]*y[IDX_H2II] + k[141]*y[IDX_HeII] + k[1129];
    IJth(jmatrix, 27, 27) = 0.0 - k[31]*y[IDX_HCOI] - k[32]*y[IDX_MgI] -
        k[33]*y[IDX_NH3I] - k[34]*y[IDX_NOI] - k[35]*y[IDX_SiI] - k[241] -
        k[294]*y[IDX_EM] - k[396]*y[IDX_CH3OHI] - k[397]*y[IDX_CH3OHI] -
        k[398]*y[IDX_CO2I] - k[399]*y[IDX_H2COI] - k[400]*y[IDX_H2COI] -
        k[401]*y[IDX_H2COI] - k[402]*y[IDX_H2OI] - k[403]*y[IDX_H2OI] -
        k[404]*y[IDX_H2OI] - k[405]*y[IDX_HCNI] - k[406]*y[IDX_HCOI] -
        k[407]*y[IDX_HNCI] - k[408]*y[IDX_NI] - k[409]*y[IDX_NH2I] -
        k[410]*y[IDX_NHI] - k[411]*y[IDX_O2I] - k[412]*y[IDX_O2I] -
        k[413]*y[IDX_O2I] - k[414]*y[IDX_OI] - k[415]*y[IDX_OHI] -
        k[529]*y[IDX_H2I] - k[614]*y[IDX_HI] - k[1108] - k[1272];
    IJth(jmatrix, 27, 28) = 0.0 + k[491]*y[IDX_HII] + k[654]*y[IDX_HeII];
    IJth(jmatrix, 27, 29) = 0.0 + k[615]*y[IDX_HI] + k[1110];
    IJth(jmatrix, 27, 30) = 0.0 + k[655]*y[IDX_HeII];
    IJth(jmatrix, 27, 31) = 0.0 + k[1114];
    IJth(jmatrix, 27, 32) = 0.0 - k[396]*y[IDX_CHII] - k[397]*y[IDX_CHII];
    IJth(jmatrix, 27, 33) = 0.0 + k[658]*y[IDX_HeII];
    IJth(jmatrix, 27, 36) = 0.0 + k[53]*y[IDX_CHI];
    IJth(jmatrix, 27, 38) = 0.0 + k[54]*y[IDX_CHI];
    IJth(jmatrix, 27, 39) = 0.0 - k[398]*y[IDX_CHII];
    IJth(jmatrix, 27, 40) = 0.0 - k[294]*y[IDX_CHII];
    IJth(jmatrix, 27, 41) = 0.0 - k[614]*y[IDX_CHII] + k[615]*y[IDX_CH2II] +
        k[1205]*y[IDX_CII];
    IJth(jmatrix, 27, 42) = 0.0 + k[78]*y[IDX_CHI] + k[491]*y[IDX_CH2I];
    IJth(jmatrix, 27, 43) = 0.0 + k[528]*y[IDX_CII] - k[529]*y[IDX_CHII];
    IJth(jmatrix, 27, 44) = 0.0 + k[102]*y[IDX_CHI] + k[509]*y[IDX_CI];
    IJth(jmatrix, 27, 46) = 0.0 - k[399]*y[IDX_CHII] - k[400]*y[IDX_CHII] -
        k[401]*y[IDX_CHII];
    IJth(jmatrix, 27, 47) = 0.0 + k[55]*y[IDX_CHI];
    IJth(jmatrix, 27, 49) = 0.0 - k[402]*y[IDX_CHII] - k[403]*y[IDX_CHII] -
        k[404]*y[IDX_CHII];
    IJth(jmatrix, 27, 50) = 0.0 + k[56]*y[IDX_CHI] + k[383]*y[IDX_CI];
    IJth(jmatrix, 27, 52) = 0.0 + k[576]*y[IDX_CI];
    IJth(jmatrix, 27, 55) = 0.0 - k[405]*y[IDX_CHII] + k[679]*y[IDX_HeII];
    IJth(jmatrix, 27, 56) = 0.0 + k[385]*y[IDX_CI];
    IJth(jmatrix, 27, 58) = 0.0 - k[31]*y[IDX_CHII] + k[372]*y[IDX_CII] -
        k[406]*y[IDX_CHII] + k[682]*y[IDX_HeII];
    IJth(jmatrix, 27, 59) = 0.0 + k[386]*y[IDX_CI];
    IJth(jmatrix, 27, 60) = 0.0 + k[387]*y[IDX_CI];
    IJth(jmatrix, 27, 62) = 0.0 + k[141]*y[IDX_CHI] + k[654]*y[IDX_CH2I] +
        k[655]*y[IDX_CH3I] + k[658]*y[IDX_CH4I] + k[679]*y[IDX_HCNI] +
        k[682]*y[IDX_HCOI];
    IJth(jmatrix, 27, 64) = 0.0 - k[407]*y[IDX_CHII];
    IJth(jmatrix, 27, 67) = 0.0 + k[388]*y[IDX_CI];
    IJth(jmatrix, 27, 69) = 0.0 - k[32]*y[IDX_CHII];
    IJth(jmatrix, 27, 71) = 0.0 - k[408]*y[IDX_CHII];
    IJth(jmatrix, 27, 72) = 0.0 + k[57]*y[IDX_CHI];
    IJth(jmatrix, 27, 74) = 0.0 + k[58]*y[IDX_CHI];
    IJth(jmatrix, 27, 75) = 0.0 + k[389]*y[IDX_CI];
    IJth(jmatrix, 27, 76) = 0.0 - k[410]*y[IDX_CHII];
    IJth(jmatrix, 27, 77) = 0.0 + k[390]*y[IDX_CI];
    IJth(jmatrix, 27, 78) = 0.0 - k[409]*y[IDX_CHII];
    IJth(jmatrix, 27, 79) = 0.0 + k[59]*y[IDX_CHI];
    IJth(jmatrix, 27, 80) = 0.0 - k[33]*y[IDX_CHII];
    IJth(jmatrix, 27, 82) = 0.0 - k[34]*y[IDX_CHII];
    IJth(jmatrix, 27, 85) = 0.0 - k[414]*y[IDX_CHII];
    IJth(jmatrix, 27, 86) = 0.0 + k[60]*y[IDX_CHI];
    IJth(jmatrix, 27, 87) = 0.0 - k[411]*y[IDX_CHII] - k[412]*y[IDX_CHII] -
        k[413]*y[IDX_CHII];
    IJth(jmatrix, 27, 88) = 0.0 + k[61]*y[IDX_CHI];
    IJth(jmatrix, 27, 90) = 0.0 + k[392]*y[IDX_CI];
    IJth(jmatrix, 27, 92) = 0.0 - k[415]*y[IDX_CHII];
    IJth(jmatrix, 27, 93) = 0.0 + k[62]*y[IDX_CHI] + k[393]*y[IDX_CI];
    IJth(jmatrix, 27, 94) = 0.0 - k[35]*y[IDX_CHII];
    IJth(jmatrix, 28, 24) = 0.0 - k[863]*y[IDX_CH2I] + k[1200]*y[IDX_H2I];
    IJth(jmatrix, 28, 25) = 0.0 - k[14]*y[IDX_CH2I];
    IJth(jmatrix, 28, 26) = 0.0 + k[924]*y[IDX_H2COI] + k[925]*y[IDX_HCOI] +
        k[926]*y[IDX_HNOI] + k[938]*y[IDX_O2HI] + k[958]*y[IDX_H2I];
    IJth(jmatrix, 28, 27) = 0.0 + k[397]*y[IDX_CH3OHI] +
        k[401]*y[IDX_H2COI];
    IJth(jmatrix, 28, 28) = 0.0 - k[14]*y[IDX_CII] - k[37]*y[IDX_CNII] -
        k[38]*y[IDX_COII] - k[39]*y[IDX_H2COII] - k[40]*y[IDX_H2OII] -
        k[41]*y[IDX_N2II] - k[42]*y[IDX_NH2II] - k[43]*y[IDX_OII] -
        k[44]*y[IDX_O2II] - k[45]*y[IDX_OHII] - k[75]*y[IDX_HII] -
        k[100]*y[IDX_H2II] - k[155]*y[IDX_NII] - k[242] - k[243] -
        k[422]*y[IDX_COII] - k[423]*y[IDX_H2COII] - k[424]*y[IDX_H2OII] -
        k[425]*y[IDX_H3OII] - k[426]*y[IDX_HCNII] - k[427]*y[IDX_HCNHII] -
        k[428]*y[IDX_HCNHII] - k[429]*y[IDX_HCOII] - k[430]*y[IDX_HNOII] -
        k[431]*y[IDX_N2HII] - k[432]*y[IDX_NHII] - k[433]*y[IDX_NH2II] -
        k[434]*y[IDX_NH3II] - k[435]*y[IDX_O2II] - k[436]*y[IDX_O2HII] -
        k[437]*y[IDX_OHII] - k[438]*y[IDX_SiOII] - k[491]*y[IDX_HII] -
        k[510]*y[IDX_H2II] - k[577]*y[IDX_H3II] - k[653]*y[IDX_HeII] -
        k[654]*y[IDX_HeII] - k[863]*y[IDX_CI] - k[878]*y[IDX_CH2I] -
        k[878]*y[IDX_CH2I] - k[878]*y[IDX_CH2I] - k[878]*y[IDX_CH2I] -
        k[879]*y[IDX_CH4I] - k[880]*y[IDX_CNI] - k[881]*y[IDX_H2COI] -
        k[882]*y[IDX_HCOI] - k[883]*y[IDX_HNOI] - k[884]*y[IDX_N2I] -
        k[885]*y[IDX_NO2I] - k[886]*y[IDX_NOI] - k[887]*y[IDX_NOI] -
        k[888]*y[IDX_NOI] - k[889]*y[IDX_O2I] - k[890]*y[IDX_O2I] -
        k[891]*y[IDX_O2I] - k[892]*y[IDX_O2I] - k[893]*y[IDX_O2I] -
        k[894]*y[IDX_OI] - k[895]*y[IDX_OI] - k[896]*y[IDX_OI] -
        k[897]*y[IDX_OI] - k[898]*y[IDX_OHI] - k[899]*y[IDX_OHI] -
        k[900]*y[IDX_OHI] - k[956]*y[IDX_H2I] - k[967]*y[IDX_HI] -
        k[1005]*y[IDX_NI] - k[1006]*y[IDX_NI] - k[1007]*y[IDX_NI] - k[1112] -
        k[1113] - k[1256];
    IJth(jmatrix, 28, 29) = 0.0 + k[36]*y[IDX_NOI];
    IJth(jmatrix, 28, 30) = 0.0 + k[244] + k[901]*y[IDX_CH3I] +
        k[901]*y[IDX_CH3I] + k[902]*y[IDX_CNI] + k[913]*y[IDX_O2I] +
        k[919]*y[IDX_OHI] + k[968]*y[IDX_HI] + k[1116];
    IJth(jmatrix, 28, 31) = 0.0 + k[298]*y[IDX_EM];
    IJth(jmatrix, 28, 32) = 0.0 + k[397]*y[IDX_CHII];
    IJth(jmatrix, 28, 33) = 0.0 + k[249] + k[457]*y[IDX_OHII] -
        k[879]*y[IDX_CH2I] + k[1124];
    IJth(jmatrix, 28, 34) = 0.0 + k[301]*y[IDX_EM];
    IJth(jmatrix, 28, 35) = 0.0 - k[880]*y[IDX_CH2I] + k[902]*y[IDX_CH3I];
    IJth(jmatrix, 28, 36) = 0.0 - k[37]*y[IDX_CH2I];
    IJth(jmatrix, 28, 38) = 0.0 - k[38]*y[IDX_CH2I] - k[422]*y[IDX_CH2I];
    IJth(jmatrix, 28, 40) = 0.0 + k[298]*y[IDX_CH3II] + k[301]*y[IDX_CH4II]
        + k[306]*y[IDX_H2COII] + k[317]*y[IDX_H3COII];
    IJth(jmatrix, 28, 41) = 0.0 - k[967]*y[IDX_CH2I] + k[968]*y[IDX_CH3I] +
        k[978]*y[IDX_HCOI];
    IJth(jmatrix, 28, 42) = 0.0 - k[75]*y[IDX_CH2I] - k[491]*y[IDX_CH2I];
    IJth(jmatrix, 28, 43) = 0.0 - k[956]*y[IDX_CH2I] + k[958]*y[IDX_CHI] +
        k[1200]*y[IDX_CI];
    IJth(jmatrix, 28, 44) = 0.0 - k[100]*y[IDX_CH2I] - k[510]*y[IDX_CH2I];
    IJth(jmatrix, 28, 46) = 0.0 + k[401]*y[IDX_CHII] + k[722]*y[IDX_NII] -
        k[881]*y[IDX_CH2I] + k[924]*y[IDX_CHI];
    IJth(jmatrix, 28, 47) = 0.0 - k[39]*y[IDX_CH2I] + k[306]*y[IDX_EM] -
        k[423]*y[IDX_CH2I];
    IJth(jmatrix, 28, 50) = 0.0 - k[40]*y[IDX_CH2I] - k[424]*y[IDX_CH2I];
    IJth(jmatrix, 28, 52) = 0.0 - k[577]*y[IDX_CH2I];
    IJth(jmatrix, 28, 53) = 0.0 + k[317]*y[IDX_EM];
    IJth(jmatrix, 28, 54) = 0.0 - k[425]*y[IDX_CH2I];
    IJth(jmatrix, 28, 56) = 0.0 - k[426]*y[IDX_CH2I];
    IJth(jmatrix, 28, 57) = 0.0 - k[427]*y[IDX_CH2I] - k[428]*y[IDX_CH2I];
    IJth(jmatrix, 28, 58) = 0.0 - k[882]*y[IDX_CH2I] + k[925]*y[IDX_CHI] +
        k[978]*y[IDX_HI];
    IJth(jmatrix, 28, 59) = 0.0 - k[429]*y[IDX_CH2I];
    IJth(jmatrix, 28, 62) = 0.0 - k[653]*y[IDX_CH2I] - k[654]*y[IDX_CH2I];
    IJth(jmatrix, 28, 66) = 0.0 - k[883]*y[IDX_CH2I] + k[926]*y[IDX_CHI];
    IJth(jmatrix, 28, 67) = 0.0 - k[430]*y[IDX_CH2I];
    IJth(jmatrix, 28, 71) = 0.0 - k[1005]*y[IDX_CH2I] - k[1006]*y[IDX_CH2I]
        - k[1007]*y[IDX_CH2I];
    IJth(jmatrix, 28, 72) = 0.0 - k[155]*y[IDX_CH2I] + k[722]*y[IDX_H2COI];
    IJth(jmatrix, 28, 73) = 0.0 - k[884]*y[IDX_CH2I];
    IJth(jmatrix, 28, 74) = 0.0 - k[41]*y[IDX_CH2I];
    IJth(jmatrix, 28, 75) = 0.0 - k[431]*y[IDX_CH2I];
    IJth(jmatrix, 28, 77) = 0.0 - k[432]*y[IDX_CH2I];
    IJth(jmatrix, 28, 79) = 0.0 - k[42]*y[IDX_CH2I] - k[433]*y[IDX_CH2I];
    IJth(jmatrix, 28, 81) = 0.0 - k[434]*y[IDX_CH2I];
    IJth(jmatrix, 28, 82) = 0.0 + k[36]*y[IDX_CH2II] - k[886]*y[IDX_CH2I] -
        k[887]*y[IDX_CH2I] - k[888]*y[IDX_CH2I];
    IJth(jmatrix, 28, 84) = 0.0 - k[885]*y[IDX_CH2I];
    IJth(jmatrix, 28, 85) = 0.0 - k[894]*y[IDX_CH2I] - k[895]*y[IDX_CH2I] -
        k[896]*y[IDX_CH2I] - k[897]*y[IDX_CH2I];
    IJth(jmatrix, 28, 86) = 0.0 - k[43]*y[IDX_CH2I];
    IJth(jmatrix, 28, 87) = 0.0 - k[889]*y[IDX_CH2I] - k[890]*y[IDX_CH2I] -
        k[891]*y[IDX_CH2I] - k[892]*y[IDX_CH2I] - k[893]*y[IDX_CH2I] +
        k[913]*y[IDX_CH3I];
    IJth(jmatrix, 28, 88) = 0.0 - k[44]*y[IDX_CH2I] - k[435]*y[IDX_CH2I];
    IJth(jmatrix, 28, 89) = 0.0 + k[938]*y[IDX_CHI];
    IJth(jmatrix, 28, 90) = 0.0 - k[436]*y[IDX_CH2I];
    IJth(jmatrix, 28, 92) = 0.0 - k[898]*y[IDX_CH2I] - k[899]*y[IDX_CH2I] -
        k[900]*y[IDX_CH2I] + k[919]*y[IDX_CH3I];
    IJth(jmatrix, 28, 93) = 0.0 - k[45]*y[IDX_CH2I] - k[437]*y[IDX_CH2I] +
        k[457]*y[IDX_CH4I];
    IJth(jmatrix, 28, 112) = 0.0 - k[438]*y[IDX_CH2I];
    IJth(jmatrix, 29, 25) = 0.0 + k[14]*y[IDX_CH2I] + k[368]*y[IDX_H2COI] +
        k[1199]*y[IDX_H2I];
    IJth(jmatrix, 29, 26) = 0.0 + k[459]*y[IDX_H2COII] + k[460]*y[IDX_H2OII]
        + k[461]*y[IDX_H3COII] + k[462]*y[IDX_H3OII] + k[463]*y[IDX_HCNII] +
        k[464]*y[IDX_HCNHII] + k[465]*y[IDX_HCNHII] + k[466]*y[IDX_HCOII] +
        k[467]*y[IDX_HNOII] + k[469]*y[IDX_N2HII] + k[470]*y[IDX_NHII] +
        k[471]*y[IDX_NH2II] + k[474]*y[IDX_O2HII] + k[475]*y[IDX_OHII] +
        k[477]*y[IDX_SiHII] + k[512]*y[IDX_H2II] + k[580]*y[IDX_H3II];
    IJth(jmatrix, 29, 27) = 0.0 + k[406]*y[IDX_HCOI] + k[529]*y[IDX_H2I];
    IJth(jmatrix, 29, 28) = 0.0 + k[14]*y[IDX_CII] + k[37]*y[IDX_CNII] +
        k[38]*y[IDX_COII] + k[39]*y[IDX_H2COII] + k[40]*y[IDX_H2OII] +
        k[41]*y[IDX_N2II] + k[42]*y[IDX_NH2II] + k[43]*y[IDX_OII] +
        k[44]*y[IDX_O2II] + k[45]*y[IDX_OHII] + k[75]*y[IDX_HII] +
        k[100]*y[IDX_H2II] + k[155]*y[IDX_NII] + k[242] + k[1112];
    IJth(jmatrix, 29, 29) = 0.0 - k[36]*y[IDX_NOI] - k[295]*y[IDX_EM] -
        k[296]*y[IDX_EM] - k[297]*y[IDX_EM] - k[416]*y[IDX_CO2I] -
        k[417]*y[IDX_H2COI] - k[418]*y[IDX_H2OI] - k[419]*y[IDX_HCOI] -
        k[420]*y[IDX_O2I] - k[421]*y[IDX_OI] - k[530]*y[IDX_H2I] -
        k[615]*y[IDX_HI] - k[736]*y[IDX_NI] - k[1109] - k[1110] - k[1111] -
        k[1278];
    IJth(jmatrix, 29, 31) = 0.0 + k[616]*y[IDX_HI] + k[1115];
    IJth(jmatrix, 29, 33) = 0.0 + k[455]*y[IDX_N2II] + k[659]*y[IDX_HeII];
    IJth(jmatrix, 29, 34) = 0.0 + k[1122];
    IJth(jmatrix, 29, 36) = 0.0 + k[37]*y[IDX_CH2I];
    IJth(jmatrix, 29, 38) = 0.0 + k[38]*y[IDX_CH2I];
    IJth(jmatrix, 29, 39) = 0.0 - k[416]*y[IDX_CH2II];
    IJth(jmatrix, 29, 40) = 0.0 - k[295]*y[IDX_CH2II] - k[296]*y[IDX_CH2II]
        - k[297]*y[IDX_CH2II];
    IJth(jmatrix, 29, 41) = 0.0 - k[615]*y[IDX_CH2II] + k[616]*y[IDX_CH3II];
    IJth(jmatrix, 29, 42) = 0.0 + k[75]*y[IDX_CH2I];
    IJth(jmatrix, 29, 43) = 0.0 + k[529]*y[IDX_CHII] - k[530]*y[IDX_CH2II] +
        k[1199]*y[IDX_CII];
    IJth(jmatrix, 29, 44) = 0.0 + k[100]*y[IDX_CH2I] + k[512]*y[IDX_CHI];
    IJth(jmatrix, 29, 46) = 0.0 + k[368]*y[IDX_CII] - k[417]*y[IDX_CH2II] +
        k[672]*y[IDX_HeII];
    IJth(jmatrix, 29, 47) = 0.0 + k[39]*y[IDX_CH2I] + k[459]*y[IDX_CHI];
    IJth(jmatrix, 29, 49) = 0.0 - k[418]*y[IDX_CH2II];
    IJth(jmatrix, 29, 50) = 0.0 + k[40]*y[IDX_CH2I] + k[460]*y[IDX_CHI];
    IJth(jmatrix, 29, 52) = 0.0 + k[580]*y[IDX_CHI];
    IJth(jmatrix, 29, 53) = 0.0 + k[461]*y[IDX_CHI];
    IJth(jmatrix, 29, 54) = 0.0 + k[462]*y[IDX_CHI];
    IJth(jmatrix, 29, 56) = 0.0 + k[463]*y[IDX_CHI];
    IJth(jmatrix, 29, 57) = 0.0 + k[464]*y[IDX_CHI] + k[465]*y[IDX_CHI];
    IJth(jmatrix, 29, 58) = 0.0 + k[406]*y[IDX_CHII] - k[419]*y[IDX_CH2II];
    IJth(jmatrix, 29, 59) = 0.0 + k[466]*y[IDX_CHI];
    IJth(jmatrix, 29, 62) = 0.0 + k[659]*y[IDX_CH4I] + k[672]*y[IDX_H2COI];
    IJth(jmatrix, 29, 67) = 0.0 + k[467]*y[IDX_CHI];
    IJth(jmatrix, 29, 71) = 0.0 - k[736]*y[IDX_CH2II];
    IJth(jmatrix, 29, 72) = 0.0 + k[155]*y[IDX_CH2I];
    IJth(jmatrix, 29, 74) = 0.0 + k[41]*y[IDX_CH2I] + k[455]*y[IDX_CH4I];
    IJth(jmatrix, 29, 75) = 0.0 + k[469]*y[IDX_CHI];
    IJth(jmatrix, 29, 77) = 0.0 + k[470]*y[IDX_CHI];
    IJth(jmatrix, 29, 79) = 0.0 + k[42]*y[IDX_CH2I] + k[471]*y[IDX_CHI];
    IJth(jmatrix, 29, 82) = 0.0 - k[36]*y[IDX_CH2II];
    IJth(jmatrix, 29, 85) = 0.0 - k[421]*y[IDX_CH2II];
    IJth(jmatrix, 29, 86) = 0.0 + k[43]*y[IDX_CH2I];
    IJth(jmatrix, 29, 87) = 0.0 - k[420]*y[IDX_CH2II];
    IJth(jmatrix, 29, 88) = 0.0 + k[44]*y[IDX_CH2I];
    IJth(jmatrix, 29, 90) = 0.0 + k[474]*y[IDX_CHI];
    IJth(jmatrix, 29, 93) = 0.0 + k[45]*y[IDX_CH2I] + k[475]*y[IDX_CHI];
    IJth(jmatrix, 29, 103) = 0.0 + k[477]*y[IDX_CHI];
    IJth(jmatrix, 30, 26) = 0.0 + k[1201]*y[IDX_H2I];
    IJth(jmatrix, 30, 28) = 0.0 + k[878]*y[IDX_CH2I] + k[878]*y[IDX_CH2I] +
        k[879]*y[IDX_CH4I] + k[879]*y[IDX_CH4I] + k[881]*y[IDX_H2COI] +
        k[882]*y[IDX_HCOI] + k[883]*y[IDX_HNOI] + k[900]*y[IDX_OHI] +
        k[956]*y[IDX_H2I];
    IJth(jmatrix, 30, 29) = 0.0 + k[417]*y[IDX_H2COI];
    IJth(jmatrix, 30, 30) = 0.0 - k[76]*y[IDX_HII] - k[244] - k[245] -
        k[246] - k[578]*y[IDX_H3II] - k[655]*y[IDX_HeII] - k[901]*y[IDX_CH3I] -
        k[901]*y[IDX_CH3I] - k[901]*y[IDX_CH3I] - k[901]*y[IDX_CH3I] -
        k[902]*y[IDX_CNI] - k[903]*y[IDX_H2COI] - k[904]*y[IDX_H2OI] -
        k[905]*y[IDX_HCOI] - k[906]*y[IDX_HNOI] - k[907]*y[IDX_NH2I] -
        k[908]*y[IDX_NH3I] - k[909]*y[IDX_NO2I] - k[910]*y[IDX_NOI] -
        k[911]*y[IDX_O2I] - k[912]*y[IDX_O2I] - k[913]*y[IDX_O2I] -
        k[914]*y[IDX_O2HI] - k[915]*y[IDX_OI] - k[916]*y[IDX_OI] -
        k[917]*y[IDX_OHI] - k[918]*y[IDX_OHI] - k[919]*y[IDX_OHI] -
        k[957]*y[IDX_H2I] - k[968]*y[IDX_HI] - k[1008]*y[IDX_NI] -
        k[1009]*y[IDX_NI] - k[1010]*y[IDX_NI] - k[1116] - k[1117] - k[1118] -
        k[1259];
    IJth(jmatrix, 30, 31) = 0.0 + k[46]*y[IDX_HCOI] + k[47]*y[IDX_MgI] +
        k[48]*y[IDX_NOI] + k[1215]*y[IDX_EM];
    IJth(jmatrix, 30, 32) = 0.0 + k[248] + k[656]*y[IDX_HeII] +
        k[714]*y[IDX_NII] + k[860]*y[IDX_SiII] + k[1121];
    IJth(jmatrix, 30, 33) = 0.0 + k[451]*y[IDX_COII] + k[452]*y[IDX_H2COII]
        + k[453]*y[IDX_H2OII] + k[454]*y[IDX_HCNII] + k[661]*y[IDX_HeII] +
        k[879]*y[IDX_CH2I] + k[879]*y[IDX_CH2I] + k[920]*y[IDX_CNI] +
        k[921]*y[IDX_O2I] + k[922]*y[IDX_OHI] + k[969]*y[IDX_HI] +
        k[1028]*y[IDX_NH2I] + k[1034]*y[IDX_NHI] + k[1056]*y[IDX_OI] + k[1125];
    IJth(jmatrix, 30, 34) = 0.0 + k[302]*y[IDX_EM] + k[447]*y[IDX_CO2I] +
        k[448]*y[IDX_COI] + k[449]*y[IDX_H2COI] + k[450]*y[IDX_H2OI];
    IJth(jmatrix, 30, 35) = 0.0 - k[902]*y[IDX_CH3I] + k[920]*y[IDX_CH4I];
    IJth(jmatrix, 30, 37) = 0.0 + k[448]*y[IDX_CH4II];
    IJth(jmatrix, 30, 38) = 0.0 + k[451]*y[IDX_CH4I];
    IJth(jmatrix, 30, 39) = 0.0 + k[447]*y[IDX_CH4II];
    IJth(jmatrix, 30, 40) = 0.0 + k[302]*y[IDX_CH4II] +
        k[1215]*y[IDX_CH3II];
    IJth(jmatrix, 30, 41) = 0.0 - k[968]*y[IDX_CH3I] + k[969]*y[IDX_CH4I];
    IJth(jmatrix, 30, 42) = 0.0 - k[76]*y[IDX_CH3I];
    IJth(jmatrix, 30, 43) = 0.0 + k[956]*y[IDX_CH2I] - k[957]*y[IDX_CH3I] +
        k[1201]*y[IDX_CHI];
    IJth(jmatrix, 30, 46) = 0.0 + k[417]*y[IDX_CH2II] + k[449]*y[IDX_CH4II]
        + k[881]*y[IDX_CH2I] - k[903]*y[IDX_CH3I];
    IJth(jmatrix, 30, 47) = 0.0 + k[452]*y[IDX_CH4I];
    IJth(jmatrix, 30, 49) = 0.0 + k[450]*y[IDX_CH4II] - k[904]*y[IDX_CH3I];
    IJth(jmatrix, 30, 50) = 0.0 + k[453]*y[IDX_CH4I];
    IJth(jmatrix, 30, 52) = 0.0 - k[578]*y[IDX_CH3I];
    IJth(jmatrix, 30, 56) = 0.0 + k[454]*y[IDX_CH4I];
    IJth(jmatrix, 30, 58) = 0.0 + k[46]*y[IDX_CH3II] + k[882]*y[IDX_CH2I] -
        k[905]*y[IDX_CH3I];
    IJth(jmatrix, 30, 62) = 0.0 - k[655]*y[IDX_CH3I] + k[656]*y[IDX_CH3OHI]
        + k[661]*y[IDX_CH4I];
    IJth(jmatrix, 30, 66) = 0.0 + k[883]*y[IDX_CH2I] - k[906]*y[IDX_CH3I];
    IJth(jmatrix, 30, 69) = 0.0 + k[47]*y[IDX_CH3II];
    IJth(jmatrix, 30, 71) = 0.0 - k[1008]*y[IDX_CH3I] - k[1009]*y[IDX_CH3I]
        - k[1010]*y[IDX_CH3I];
    IJth(jmatrix, 30, 72) = 0.0 + k[714]*y[IDX_CH3OHI];
    IJth(jmatrix, 30, 76) = 0.0 + k[1034]*y[IDX_CH4I];
    IJth(jmatrix, 30, 78) = 0.0 - k[907]*y[IDX_CH3I] + k[1028]*y[IDX_CH4I];
    IJth(jmatrix, 30, 80) = 0.0 - k[908]*y[IDX_CH3I];
    IJth(jmatrix, 30, 82) = 0.0 + k[48]*y[IDX_CH3II] - k[910]*y[IDX_CH3I];
    IJth(jmatrix, 30, 84) = 0.0 - k[909]*y[IDX_CH3I];
    IJth(jmatrix, 30, 85) = 0.0 - k[915]*y[IDX_CH3I] - k[916]*y[IDX_CH3I] +
        k[1056]*y[IDX_CH4I];
    IJth(jmatrix, 30, 87) = 0.0 - k[911]*y[IDX_CH3I] - k[912]*y[IDX_CH3I] -
        k[913]*y[IDX_CH3I] + k[921]*y[IDX_CH4I];
    IJth(jmatrix, 30, 89) = 0.0 - k[914]*y[IDX_CH3I];
    IJth(jmatrix, 30, 92) = 0.0 + k[900]*y[IDX_CH2I] - k[917]*y[IDX_CH3I] -
        k[918]*y[IDX_CH3I] - k[919]*y[IDX_CH3I] + k[922]*y[IDX_CH4I];
    IJth(jmatrix, 30, 95) = 0.0 + k[860]*y[IDX_CH3OHI];
    IJth(jmatrix, 31, 25) = 0.0 + k[366]*y[IDX_CH3OHI];
    IJth(jmatrix, 31, 27) = 0.0 + k[396]*y[IDX_CH3OHI] +
        k[399]*y[IDX_H2COI];
    IJth(jmatrix, 31, 28) = 0.0 + k[423]*y[IDX_H2COII] + k[424]*y[IDX_H2OII]
        + k[425]*y[IDX_H3OII] + k[426]*y[IDX_HCNII] + k[427]*y[IDX_HCNHII] +
        k[428]*y[IDX_HCNHII] + k[429]*y[IDX_HCOII] + k[430]*y[IDX_HNOII] +
        k[431]*y[IDX_N2HII] + k[432]*y[IDX_NHII] + k[433]*y[IDX_NH2II] +
        k[434]*y[IDX_NH3II] + k[436]*y[IDX_O2HII] + k[437]*y[IDX_OHII] +
        k[510]*y[IDX_H2II] + k[577]*y[IDX_H3II];
    IJth(jmatrix, 31, 29) = 0.0 + k[419]*y[IDX_HCOI] + k[530]*y[IDX_H2I];
    IJth(jmatrix, 31, 30) = 0.0 + k[76]*y[IDX_HII] + k[245] + k[1117];
    IJth(jmatrix, 31, 31) = 0.0 - k[46]*y[IDX_HCOI] - k[47]*y[IDX_MgI] -
        k[48]*y[IDX_NOI] - k[298]*y[IDX_EM] - k[299]*y[IDX_EM] -
        k[300]*y[IDX_EM] - k[439]*y[IDX_CH3OHI] - k[440]*y[IDX_H2COI] -
        k[441]*y[IDX_HCOI] - k[442]*y[IDX_O2I] - k[443]*y[IDX_OI] -
        k[444]*y[IDX_OI] - k[445]*y[IDX_OHI] - k[446]*y[IDX_SiH4I] -
        k[616]*y[IDX_HI] - k[794]*y[IDX_NHI] - k[1114] - k[1115] -
        k[1215]*y[IDX_EM] - k[1285];
    IJth(jmatrix, 31, 32) = 0.0 + k[366]*y[IDX_CII] + k[396]*y[IDX_CHII] -
        k[439]*y[IDX_CH3II] + k[492]*y[IDX_HII] + k[579]*y[IDX_H3II] +
        k[657]*y[IDX_HeII] + k[715]*y[IDX_NII];
    IJth(jmatrix, 31, 33) = 0.0 + k[456]*y[IDX_N2II] + k[495]*y[IDX_HII] +
        k[511]*y[IDX_H2II] + k[660]*y[IDX_HeII] + k[716]*y[IDX_NII] +
        k[810]*y[IDX_OII];
    IJth(jmatrix, 31, 34) = 0.0 + k[617]*y[IDX_HI] + k[822]*y[IDX_OI] +
        k[1123];
    IJth(jmatrix, 31, 40) = 0.0 - k[298]*y[IDX_CH3II] - k[299]*y[IDX_CH3II]
        - k[300]*y[IDX_CH3II] - k[1215]*y[IDX_CH3II];
    IJth(jmatrix, 31, 41) = 0.0 - k[616]*y[IDX_CH3II] + k[617]*y[IDX_CH4II];
    IJth(jmatrix, 31, 42) = 0.0 + k[76]*y[IDX_CH3I] + k[492]*y[IDX_CH3OHI] +
        k[495]*y[IDX_CH4I];
    IJth(jmatrix, 31, 43) = 0.0 + k[530]*y[IDX_CH2II];
    IJth(jmatrix, 31, 44) = 0.0 + k[510]*y[IDX_CH2I] + k[511]*y[IDX_CH4I];
    IJth(jmatrix, 31, 46) = 0.0 + k[399]*y[IDX_CHII] - k[440]*y[IDX_CH3II];
    IJth(jmatrix, 31, 47) = 0.0 + k[423]*y[IDX_CH2I];
    IJth(jmatrix, 31, 50) = 0.0 + k[424]*y[IDX_CH2I];
    IJth(jmatrix, 31, 52) = 0.0 + k[577]*y[IDX_CH2I] + k[579]*y[IDX_CH3OHI];
    IJth(jmatrix, 31, 54) = 0.0 + k[425]*y[IDX_CH2I];
    IJth(jmatrix, 31, 56) = 0.0 + k[426]*y[IDX_CH2I];
    IJth(jmatrix, 31, 57) = 0.0 + k[427]*y[IDX_CH2I] + k[428]*y[IDX_CH2I];
    IJth(jmatrix, 31, 58) = 0.0 - k[46]*y[IDX_CH3II] + k[419]*y[IDX_CH2II] -
        k[441]*y[IDX_CH3II];
    IJth(jmatrix, 31, 59) = 0.0 + k[429]*y[IDX_CH2I];
    IJth(jmatrix, 31, 62) = 0.0 + k[657]*y[IDX_CH3OHI] + k[660]*y[IDX_CH4I];
    IJth(jmatrix, 31, 67) = 0.0 + k[430]*y[IDX_CH2I];
    IJth(jmatrix, 31, 69) = 0.0 - k[47]*y[IDX_CH3II];
    IJth(jmatrix, 31, 72) = 0.0 + k[715]*y[IDX_CH3OHI] + k[716]*y[IDX_CH4I];
    IJth(jmatrix, 31, 74) = 0.0 + k[456]*y[IDX_CH4I];
    IJth(jmatrix, 31, 75) = 0.0 + k[431]*y[IDX_CH2I];
    IJth(jmatrix, 31, 76) = 0.0 - k[794]*y[IDX_CH3II];
    IJth(jmatrix, 31, 77) = 0.0 + k[432]*y[IDX_CH2I];
    IJth(jmatrix, 31, 79) = 0.0 + k[433]*y[IDX_CH2I];
    IJth(jmatrix, 31, 81) = 0.0 + k[434]*y[IDX_CH2I];
    IJth(jmatrix, 31, 82) = 0.0 - k[48]*y[IDX_CH3II];
    IJth(jmatrix, 31, 85) = 0.0 - k[443]*y[IDX_CH3II] - k[444]*y[IDX_CH3II]
        + k[822]*y[IDX_CH4II];
    IJth(jmatrix, 31, 86) = 0.0 + k[810]*y[IDX_CH4I];
    IJth(jmatrix, 31, 87) = 0.0 - k[442]*y[IDX_CH3II];
    IJth(jmatrix, 31, 90) = 0.0 + k[436]*y[IDX_CH2I];
    IJth(jmatrix, 31, 92) = 0.0 - k[445]*y[IDX_CH3II];
    IJth(jmatrix, 31, 93) = 0.0 + k[437]*y[IDX_CH2I];
    IJth(jmatrix, 31, 108) = 0.0 - k[446]*y[IDX_CH3II];
    IJth(jmatrix, 32, 0) = 0.0 + k[1359] + k[1360] + k[1361] + k[1362];
    IJth(jmatrix, 32, 25) = 0.0 - k[365]*y[IDX_CH3OHI] -
        k[366]*y[IDX_CH3OHI];
    IJth(jmatrix, 32, 27) = 0.0 - k[396]*y[IDX_CH3OHI] -
        k[397]*y[IDX_CH3OHI];
    IJth(jmatrix, 32, 31) = 0.0 - k[439]*y[IDX_CH3OHI];
    IJth(jmatrix, 32, 32) = 0.0 - k[247] - k[248] - k[365]*y[IDX_CII] -
        k[366]*y[IDX_CII] - k[396]*y[IDX_CHII] - k[397]*y[IDX_CHII] -
        k[439]*y[IDX_CH3II] - k[492]*y[IDX_HII] - k[493]*y[IDX_HII] -
        k[494]*y[IDX_HII] - k[579]*y[IDX_H3II] - k[656]*y[IDX_HeII] -
        k[657]*y[IDX_HeII] - k[712]*y[IDX_NII] - k[713]*y[IDX_NII] -
        k[714]*y[IDX_NII] - k[715]*y[IDX_NII] - k[808]*y[IDX_OII] -
        k[809]*y[IDX_OII] - k[820]*y[IDX_O2II] - k[860]*y[IDX_SiII] - k[1119] -
        k[1120] - k[1121] - k[1223];
    IJth(jmatrix, 32, 42) = 0.0 - k[492]*y[IDX_CH3OHI] -
        k[493]*y[IDX_CH3OHI] - k[494]*y[IDX_CH3OHI];
    IJth(jmatrix, 32, 52) = 0.0 - k[579]*y[IDX_CH3OHI];
    IJth(jmatrix, 32, 62) = 0.0 - k[656]*y[IDX_CH3OHI] -
        k[657]*y[IDX_CH3OHI];
    IJth(jmatrix, 32, 72) = 0.0 - k[712]*y[IDX_CH3OHI] -
        k[713]*y[IDX_CH3OHI] - k[714]*y[IDX_CH3OHI] - k[715]*y[IDX_CH3OHI];
    IJth(jmatrix, 32, 86) = 0.0 - k[808]*y[IDX_CH3OHI] -
        k[809]*y[IDX_CH3OHI];
    IJth(jmatrix, 32, 88) = 0.0 - k[820]*y[IDX_CH3OHI];
    IJth(jmatrix, 32, 95) = 0.0 - k[860]*y[IDX_CH3OHI];
    IJth(jmatrix, 33, 1) = 0.0 + k[1307] + k[1308] + k[1309] + k[1310];
    IJth(jmatrix, 33, 28) = 0.0 - k[879]*y[IDX_CH4I];
    IJth(jmatrix, 33, 30) = 0.0 + k[901]*y[IDX_CH3I] + k[901]*y[IDX_CH3I] +
        k[903]*y[IDX_H2COI] + k[904]*y[IDX_H2OI] + k[905]*y[IDX_HCOI] +
        k[906]*y[IDX_HNOI] + k[907]*y[IDX_NH2I] + k[908]*y[IDX_NH3I] +
        k[914]*y[IDX_O2HI] + k[917]*y[IDX_OHI] + k[957]*y[IDX_H2I];
    IJth(jmatrix, 33, 31) = 0.0 + k[439]*y[IDX_CH3OHI] + k[440]*y[IDX_H2COI]
        + k[446]*y[IDX_SiH4I];
    IJth(jmatrix, 33, 32) = 0.0 + k[439]*y[IDX_CH3II];
    IJth(jmatrix, 33, 33) = 0.0 - k[52]*y[IDX_COII] - k[77]*y[IDX_HII] -
        k[101]*y[IDX_H2II] - k[140]*y[IDX_HeII] - k[156]*y[IDX_NII] -
        k[207]*y[IDX_OII] - k[249] - k[451]*y[IDX_COII] - k[452]*y[IDX_H2COII] -
        k[453]*y[IDX_H2OII] - k[454]*y[IDX_HCNII] - k[455]*y[IDX_N2II] -
        k[456]*y[IDX_N2II] - k[457]*y[IDX_OHII] - k[495]*y[IDX_HII] -
        k[511]*y[IDX_H2II] - k[658]*y[IDX_HeII] - k[659]*y[IDX_HeII] -
        k[660]*y[IDX_HeII] - k[661]*y[IDX_HeII] - k[716]*y[IDX_NII] -
        k[717]*y[IDX_NII] - k[718]*y[IDX_NII] - k[810]*y[IDX_OII] -
        k[879]*y[IDX_CH2I] - k[920]*y[IDX_CNI] - k[921]*y[IDX_O2I] -
        k[922]*y[IDX_OHI] - k[969]*y[IDX_HI] - k[1028]*y[IDX_NH2I] -
        k[1034]*y[IDX_NHI] - k[1056]*y[IDX_OI] - k[1124] - k[1125] - k[1126] -
        k[1127] - k[1260];
    IJth(jmatrix, 33, 34) = 0.0 + k[49]*y[IDX_H2COI] + k[50]*y[IDX_NH3I] +
        k[51]*y[IDX_O2I];
    IJth(jmatrix, 33, 35) = 0.0 - k[920]*y[IDX_CH4I];
    IJth(jmatrix, 33, 38) = 0.0 - k[52]*y[IDX_CH4I] - k[451]*y[IDX_CH4I];
    IJth(jmatrix, 33, 41) = 0.0 - k[969]*y[IDX_CH4I];
    IJth(jmatrix, 33, 42) = 0.0 - k[77]*y[IDX_CH4I] - k[495]*y[IDX_CH4I];
    IJth(jmatrix, 33, 43) = 0.0 + k[957]*y[IDX_CH3I];
    IJth(jmatrix, 33, 44) = 0.0 - k[101]*y[IDX_CH4I] - k[511]*y[IDX_CH4I];
    IJth(jmatrix, 33, 46) = 0.0 + k[49]*y[IDX_CH4II] + k[440]*y[IDX_CH3II] +
        k[903]*y[IDX_CH3I];
    IJth(jmatrix, 33, 47) = 0.0 - k[452]*y[IDX_CH4I];
    IJth(jmatrix, 33, 49) = 0.0 + k[904]*y[IDX_CH3I];
    IJth(jmatrix, 33, 50) = 0.0 - k[453]*y[IDX_CH4I];
    IJth(jmatrix, 33, 56) = 0.0 - k[454]*y[IDX_CH4I];
    IJth(jmatrix, 33, 58) = 0.0 + k[905]*y[IDX_CH3I];
    IJth(jmatrix, 33, 62) = 0.0 - k[140]*y[IDX_CH4I] - k[658]*y[IDX_CH4I] -
        k[659]*y[IDX_CH4I] - k[660]*y[IDX_CH4I] - k[661]*y[IDX_CH4I];
    IJth(jmatrix, 33, 66) = 0.0 + k[906]*y[IDX_CH3I];
    IJth(jmatrix, 33, 72) = 0.0 - k[156]*y[IDX_CH4I] - k[716]*y[IDX_CH4I] -
        k[717]*y[IDX_CH4I] - k[718]*y[IDX_CH4I];
    IJth(jmatrix, 33, 74) = 0.0 - k[455]*y[IDX_CH4I] - k[456]*y[IDX_CH4I];
    IJth(jmatrix, 33, 76) = 0.0 - k[1034]*y[IDX_CH4I];
    IJth(jmatrix, 33, 78) = 0.0 + k[907]*y[IDX_CH3I] - k[1028]*y[IDX_CH4I];
    IJth(jmatrix, 33, 80) = 0.0 + k[50]*y[IDX_CH4II] + k[908]*y[IDX_CH3I];
    IJth(jmatrix, 33, 85) = 0.0 - k[1056]*y[IDX_CH4I];
    IJth(jmatrix, 33, 86) = 0.0 - k[207]*y[IDX_CH4I] - k[810]*y[IDX_CH4I];
    IJth(jmatrix, 33, 87) = 0.0 + k[51]*y[IDX_CH4II] - k[921]*y[IDX_CH4I];
    IJth(jmatrix, 33, 89) = 0.0 + k[914]*y[IDX_CH3I];
    IJth(jmatrix, 33, 92) = 0.0 + k[917]*y[IDX_CH3I] - k[922]*y[IDX_CH4I];
    IJth(jmatrix, 33, 93) = 0.0 - k[457]*y[IDX_CH4I];
    IJth(jmatrix, 33, 108) = 0.0 + k[446]*y[IDX_CH3II];
    IJth(jmatrix, 34, 30) = 0.0 + k[578]*y[IDX_H3II];
    IJth(jmatrix, 34, 31) = 0.0 + k[441]*y[IDX_HCOI];
    IJth(jmatrix, 34, 33) = 0.0 + k[52]*y[IDX_COII] + k[77]*y[IDX_HII] +
        k[101]*y[IDX_H2II] + k[140]*y[IDX_HeII] + k[156]*y[IDX_NII] +
        k[207]*y[IDX_OII] + k[1126];
    IJth(jmatrix, 34, 34) = 0.0 - k[49]*y[IDX_H2COI] - k[50]*y[IDX_NH3I] -
        k[51]*y[IDX_O2I] - k[301]*y[IDX_EM] - k[302]*y[IDX_EM] -
        k[447]*y[IDX_CO2I] - k[448]*y[IDX_COI] - k[449]*y[IDX_H2COI] -
        k[450]*y[IDX_H2OI] - k[617]*y[IDX_HI] - k[822]*y[IDX_OI] - k[1122] -
        k[1123] - k[1288];
    IJth(jmatrix, 34, 37) = 0.0 - k[448]*y[IDX_CH4II];
    IJth(jmatrix, 34, 38) = 0.0 + k[52]*y[IDX_CH4I];
    IJth(jmatrix, 34, 39) = 0.0 - k[447]*y[IDX_CH4II];
    IJth(jmatrix, 34, 40) = 0.0 - k[301]*y[IDX_CH4II] - k[302]*y[IDX_CH4II];
    IJth(jmatrix, 34, 41) = 0.0 - k[617]*y[IDX_CH4II];
    IJth(jmatrix, 34, 42) = 0.0 + k[77]*y[IDX_CH4I];
    IJth(jmatrix, 34, 44) = 0.0 + k[101]*y[IDX_CH4I];
    IJth(jmatrix, 34, 46) = 0.0 - k[49]*y[IDX_CH4II] - k[449]*y[IDX_CH4II];
    IJth(jmatrix, 34, 49) = 0.0 - k[450]*y[IDX_CH4II];
    IJth(jmatrix, 34, 52) = 0.0 + k[578]*y[IDX_CH3I];
    IJth(jmatrix, 34, 58) = 0.0 + k[441]*y[IDX_CH3II];
    IJth(jmatrix, 34, 62) = 0.0 + k[140]*y[IDX_CH4I];
    IJth(jmatrix, 34, 72) = 0.0 + k[156]*y[IDX_CH4I];
    IJth(jmatrix, 34, 80) = 0.0 - k[50]*y[IDX_CH4II];
    IJth(jmatrix, 34, 85) = 0.0 - k[822]*y[IDX_CH4II];
    IJth(jmatrix, 34, 86) = 0.0 + k[207]*y[IDX_CH4I];
    IJth(jmatrix, 34, 87) = 0.0 - k[51]*y[IDX_CH4II];
    IJth(jmatrix, 35, 24) = 0.0 + k[27]*y[IDX_CNII] + k[385]*y[IDX_HCNII] +
        k[865]*y[IDX_N2I] + k[869]*y[IDX_NHI] + k[871]*y[IDX_NOI] +
        k[874]*y[IDX_OCNI] + k[1194]*y[IDX_NI];
    IJth(jmatrix, 35, 25) = 0.0 + k[378]*y[IDX_OCNI];
    IJth(jmatrix, 35, 26) = 0.0 + k[53]*y[IDX_CNII] + k[463]*y[IDX_HCNII] +
        k[928]*y[IDX_NI];
    IJth(jmatrix, 35, 28) = 0.0 + k[37]*y[IDX_CNII] + k[426]*y[IDX_HCNII] -
        k[880]*y[IDX_CNI];
    IJth(jmatrix, 35, 30) = 0.0 - k[902]*y[IDX_CNI];
    IJth(jmatrix, 35, 33) = 0.0 - k[920]*y[IDX_CNI];
    IJth(jmatrix, 35, 35) = 0.0 - k[69]*y[IDX_N2II] - k[103]*y[IDX_H2II] -
        k[157]*y[IDX_NII] - k[251] - k[482]*y[IDX_HNOII] - k[483]*y[IDX_O2HII] -
        k[513]*y[IDX_H2II] - k[581]*y[IDX_H3II] - k[663]*y[IDX_HeII] -
        k[664]*y[IDX_HeII] - k[747]*y[IDX_NHII] - k[811]*y[IDX_OII] -
        k[836]*y[IDX_OHII] - k[880]*y[IDX_CH2I] - k[902]*y[IDX_CH3I] -
        k[920]*y[IDX_CH4I] - k[942]*y[IDX_H2COI] - k[943]*y[IDX_HCOI] -
        k[944]*y[IDX_HNOI] - k[945]*y[IDX_NO2I] - k[946]*y[IDX_NOI] -
        k[947]*y[IDX_NOI] - k[948]*y[IDX_O2I] - k[949]*y[IDX_O2I] -
        k[950]*y[IDX_SiH4I] - k[959]*y[IDX_H2I] - k[1011]*y[IDX_NI] -
        k[1033]*y[IDX_NH3I] - k[1035]*y[IDX_NHI] - k[1057]*y[IDX_OI] -
        k[1058]*y[IDX_OI] - k[1090]*y[IDX_OHI] - k[1091]*y[IDX_OHI] - k[1130] -
        k[1263];
    IJth(jmatrix, 35, 36) = 0.0 + k[27]*y[IDX_CI] + k[37]*y[IDX_CH2I] +
        k[53]*y[IDX_CHI] + k[63]*y[IDX_COI] + k[64]*y[IDX_H2COI] +
        k[65]*y[IDX_HCNI] + k[66]*y[IDX_HCOI] + k[67]*y[IDX_NOI] +
        k[68]*y[IDX_O2I] + k[126]*y[IDX_HI] + k[183]*y[IDX_NH2I] +
        k[199]*y[IDX_NHI] + k[216]*y[IDX_OI] + k[225]*y[IDX_OHI];
    IJth(jmatrix, 35, 37) = 0.0 + k[63]*y[IDX_CNII] + k[621]*y[IDX_HCNII];
    IJth(jmatrix, 35, 39) = 0.0 + k[620]*y[IDX_HCNII];
    IJth(jmatrix, 35, 40) = 0.0 + k[326]*y[IDX_HCNII] +
        k[327]*y[IDX_HCNHII];
    IJth(jmatrix, 35, 41) = 0.0 + k[126]*y[IDX_CNII] + k[976]*y[IDX_HCNI] +
        k[995]*y[IDX_OCNI];
    IJth(jmatrix, 35, 43) = 0.0 - k[959]*y[IDX_CNI];
    IJth(jmatrix, 35, 44) = 0.0 - k[103]*y[IDX_CNI] - k[513]*y[IDX_CNI];
    IJth(jmatrix, 35, 46) = 0.0 + k[64]*y[IDX_CNII] + k[622]*y[IDX_HCNII] -
        k[942]*y[IDX_CNI];
    IJth(jmatrix, 35, 49) = 0.0 + k[565]*y[IDX_HCNII];
    IJth(jmatrix, 35, 52) = 0.0 - k[581]*y[IDX_CNI];
    IJth(jmatrix, 35, 55) = 0.0 + k[65]*y[IDX_CNII] + k[259] +
        k[623]*y[IDX_HCNII] + k[976]*y[IDX_HI] + k[1063]*y[IDX_OI] +
        k[1094]*y[IDX_OHI] + k[1147];
    IJth(jmatrix, 35, 56) = 0.0 + k[326]*y[IDX_EM] + k[385]*y[IDX_CI] +
        k[426]*y[IDX_CH2I] + k[463]*y[IDX_CHI] + k[565]*y[IDX_H2OI] +
        k[620]*y[IDX_CO2I] + k[621]*y[IDX_COI] + k[622]*y[IDX_H2COI] +
        k[623]*y[IDX_HCNI] + k[624]*y[IDX_HCOI] + k[626]*y[IDX_HNCI] +
        k[784]*y[IDX_NH2I] + k[798]*y[IDX_NHI] + k[853]*y[IDX_OHI];
    IJth(jmatrix, 35, 57) = 0.0 + k[327]*y[IDX_EM];
    IJth(jmatrix, 35, 58) = 0.0 + k[66]*y[IDX_CNII] + k[624]*y[IDX_HCNII] -
        k[943]*y[IDX_CNI];
    IJth(jmatrix, 35, 62) = 0.0 - k[663]*y[IDX_CNI] - k[664]*y[IDX_CNI] +
        k[698]*y[IDX_OCNI];
    IJth(jmatrix, 35, 64) = 0.0 + k[262] + k[626]*y[IDX_HCNII] + k[1151];
    IJth(jmatrix, 35, 66) = 0.0 - k[944]*y[IDX_CNI];
    IJth(jmatrix, 35, 67) = 0.0 - k[482]*y[IDX_CNI];
    IJth(jmatrix, 35, 71) = 0.0 + k[744]*y[IDX_SiCII] + k[928]*y[IDX_CHI] -
        k[1011]*y[IDX_CNI] + k[1027]*y[IDX_SiCI] + k[1194]*y[IDX_CI];
    IJth(jmatrix, 35, 72) = 0.0 - k[157]*y[IDX_CNI];
    IJth(jmatrix, 35, 73) = 0.0 + k[865]*y[IDX_CI];
    IJth(jmatrix, 35, 74) = 0.0 - k[69]*y[IDX_CNI];
    IJth(jmatrix, 35, 76) = 0.0 + k[199]*y[IDX_CNII] + k[798]*y[IDX_HCNII] +
        k[869]*y[IDX_CI] - k[1035]*y[IDX_CNI];
    IJth(jmatrix, 35, 77) = 0.0 - k[747]*y[IDX_CNI];
    IJth(jmatrix, 35, 78) = 0.0 + k[183]*y[IDX_CNII] + k[784]*y[IDX_HCNII];
    IJth(jmatrix, 35, 80) = 0.0 - k[1033]*y[IDX_CNI];
    IJth(jmatrix, 35, 82) = 0.0 + k[67]*y[IDX_CNII] + k[871]*y[IDX_CI] -
        k[946]*y[IDX_CNI] - k[947]*y[IDX_CNI];
    IJth(jmatrix, 35, 84) = 0.0 - k[945]*y[IDX_CNI];
    IJth(jmatrix, 35, 85) = 0.0 + k[216]*y[IDX_CNII] - k[1057]*y[IDX_CNI] -
        k[1058]*y[IDX_CNI] + k[1063]*y[IDX_HCNI] + k[1079]*y[IDX_OCNI];
    IJth(jmatrix, 35, 86) = 0.0 - k[811]*y[IDX_CNI];
    IJth(jmatrix, 35, 87) = 0.0 + k[68]*y[IDX_CNII] - k[948]*y[IDX_CNI] -
        k[949]*y[IDX_CNI];
    IJth(jmatrix, 35, 90) = 0.0 - k[483]*y[IDX_CNI];
    IJth(jmatrix, 35, 91) = 0.0 + k[283] + k[378]*y[IDX_CII] +
        k[698]*y[IDX_HeII] + k[874]*y[IDX_CI] + k[995]*y[IDX_HI] +
        k[1079]*y[IDX_OI] + k[1172];
    IJth(jmatrix, 35, 92) = 0.0 + k[225]*y[IDX_CNII] + k[853]*y[IDX_HCNII] -
        k[1090]*y[IDX_CNI] - k[1091]*y[IDX_CNI] + k[1094]*y[IDX_HCNI];
    IJth(jmatrix, 35, 93) = 0.0 - k[836]*y[IDX_CNI];
    IJth(jmatrix, 35, 96) = 0.0 + k[1027]*y[IDX_NI];
    IJth(jmatrix, 35, 97) = 0.0 + k[744]*y[IDX_NI];
    IJth(jmatrix, 35, 108) = 0.0 - k[950]*y[IDX_CNI];
    IJth(jmatrix, 36, 24) = 0.0 - k[27]*y[IDX_CNII];
    IJth(jmatrix, 36, 25) = 0.0 + k[375]*y[IDX_NHI] + k[1192]*y[IDX_NI];
    IJth(jmatrix, 36, 26) = 0.0 - k[53]*y[IDX_CNII] + k[468]*y[IDX_NII];
    IJth(jmatrix, 36, 27) = 0.0 + k[408]*y[IDX_NI] + k[410]*y[IDX_NHI];
    IJth(jmatrix, 36, 28) = 0.0 - k[37]*y[IDX_CNII];
    IJth(jmatrix, 36, 35) = 0.0 + k[69]*y[IDX_N2II] + k[103]*y[IDX_H2II] +
        k[157]*y[IDX_NII];
    IJth(jmatrix, 36, 36) = 0.0 - k[27]*y[IDX_CI] - k[37]*y[IDX_CH2I] -
        k[53]*y[IDX_CHI] - k[63]*y[IDX_COI] - k[64]*y[IDX_H2COI] -
        k[65]*y[IDX_HCNI] - k[66]*y[IDX_HCOI] - k[67]*y[IDX_NOI] -
        k[68]*y[IDX_O2I] - k[126]*y[IDX_HI] - k[183]*y[IDX_NH2I] -
        k[199]*y[IDX_NHI] - k[216]*y[IDX_OI] - k[225]*y[IDX_OHI] -
        k[303]*y[IDX_EM] - k[479]*y[IDX_H2COI] - k[480]*y[IDX_HCOI] -
        k[481]*y[IDX_O2I] - k[531]*y[IDX_H2I] - k[560]*y[IDX_H2OI] -
        k[561]*y[IDX_H2OI] - k[737]*y[IDX_NI] - k[1276];
    IJth(jmatrix, 36, 37) = 0.0 - k[63]*y[IDX_CNII];
    IJth(jmatrix, 36, 40) = 0.0 - k[303]*y[IDX_CNII];
    IJth(jmatrix, 36, 41) = 0.0 - k[126]*y[IDX_CNII];
    IJth(jmatrix, 36, 43) = 0.0 - k[531]*y[IDX_CNII];
    IJth(jmatrix, 36, 44) = 0.0 + k[103]*y[IDX_CNI];
    IJth(jmatrix, 36, 46) = 0.0 - k[64]*y[IDX_CNII] - k[479]*y[IDX_CNII];
    IJth(jmatrix, 36, 49) = 0.0 - k[560]*y[IDX_CNII] - k[561]*y[IDX_CNII];
    IJth(jmatrix, 36, 55) = 0.0 - k[65]*y[IDX_CNII] + k[676]*y[IDX_HeII];
    IJth(jmatrix, 36, 58) = 0.0 - k[66]*y[IDX_CNII] - k[480]*y[IDX_CNII];
    IJth(jmatrix, 36, 62) = 0.0 + k[676]*y[IDX_HCNI] + k[683]*y[IDX_HNCI] +
        k[697]*y[IDX_OCNI];
    IJth(jmatrix, 36, 64) = 0.0 + k[683]*y[IDX_HeII];
    IJth(jmatrix, 36, 71) = 0.0 + k[408]*y[IDX_CHII] - k[737]*y[IDX_CNII] +
        k[1192]*y[IDX_CII];
    IJth(jmatrix, 36, 72) = 0.0 + k[157]*y[IDX_CNI] + k[468]*y[IDX_CHI];
    IJth(jmatrix, 36, 74) = 0.0 + k[69]*y[IDX_CNI];
    IJth(jmatrix, 36, 76) = 0.0 - k[199]*y[IDX_CNII] + k[375]*y[IDX_CII] +
        k[410]*y[IDX_CHII];
    IJth(jmatrix, 36, 78) = 0.0 - k[183]*y[IDX_CNII];
    IJth(jmatrix, 36, 82) = 0.0 - k[67]*y[IDX_CNII];
    IJth(jmatrix, 36, 85) = 0.0 - k[216]*y[IDX_CNII];
    IJth(jmatrix, 36, 87) = 0.0 - k[68]*y[IDX_CNII] - k[481]*y[IDX_CNII];
    IJth(jmatrix, 36, 91) = 0.0 + k[697]*y[IDX_HeII];
    IJth(jmatrix, 36, 92) = 0.0 - k[225]*y[IDX_CNII];
    IJth(jmatrix, 37, 2) = 0.0 + k[1331] + k[1332] + k[1333] + k[1334];
    IJth(jmatrix, 37, 24) = 0.0 + k[28]*y[IDX_COII] + k[386]*y[IDX_HCOII] +
        k[395]*y[IDX_SiOII] + k[864]*y[IDX_HCOI] + k[872]*y[IDX_NOI] +
        k[873]*y[IDX_O2I] + k[874]*y[IDX_OCNI] + k[875]*y[IDX_OHI] +
        k[1004]*y[IDX_HNCOI] + k[1196]*y[IDX_OI];
    IJth(jmatrix, 37, 25) = 0.0 + k[367]*y[IDX_CO2I] + k[368]*y[IDX_H2COI] +
        k[372]*y[IDX_HCOI] + k[377]*y[IDX_O2I] + k[382]*y[IDX_SiOI];
    IJth(jmatrix, 37, 26) = 0.0 + k[54]*y[IDX_COII] + k[466]*y[IDX_HCOII] +
        k[923]*y[IDX_CO2I] + k[925]*y[IDX_HCOI] + k[934]*y[IDX_O2I] +
        k[935]*y[IDX_O2I] + k[939]*y[IDX_OI];
    IJth(jmatrix, 37, 27) = 0.0 + k[398]*y[IDX_CO2I] + k[399]*y[IDX_H2COI] +
        k[406]*y[IDX_HCOI];
    IJth(jmatrix, 37, 28) = 0.0 + k[38]*y[IDX_COII] + k[429]*y[IDX_HCOII] +
        k[882]*y[IDX_HCOI] + k[891]*y[IDX_O2I] + k[894]*y[IDX_OI] +
        k[895]*y[IDX_OI];
    IJth(jmatrix, 37, 29) = 0.0 + k[416]*y[IDX_CO2I] + k[419]*y[IDX_HCOI];
    IJth(jmatrix, 37, 30) = 0.0 + k[905]*y[IDX_HCOI] + k[915]*y[IDX_OI];
    IJth(jmatrix, 37, 31) = 0.0 + k[441]*y[IDX_HCOI];
    IJth(jmatrix, 37, 33) = 0.0 + k[52]*y[IDX_COII];
    IJth(jmatrix, 37, 34) = 0.0 - k[448]*y[IDX_COI];
    IJth(jmatrix, 37, 35) = 0.0 + k[943]*y[IDX_HCOI] + k[946]*y[IDX_NOI] +
        k[948]*y[IDX_O2I] + k[1057]*y[IDX_OI];
    IJth(jmatrix, 37, 36) = 0.0 - k[63]*y[IDX_COI] + k[480]*y[IDX_HCOI] +
        k[481]*y[IDX_O2I];
    IJth(jmatrix, 37, 37) = 0.0 - k[63]*y[IDX_CNII] - k[74]*y[IDX_N2II] -
        k[104]*y[IDX_H2II] - k[158]*y[IDX_NII] - k[208]*y[IDX_OII] - k[232] -
        k[253] - k[448]*y[IDX_CH4II] - k[485]*y[IDX_HCO2II] -
        k[486]*y[IDX_HNOII] - k[487]*y[IDX_N2HII] - k[488]*y[IDX_O2HII] -
        k[489]*y[IDX_SiH4II] - k[490]*y[IDX_SiOII] - k[515]*y[IDX_H2II] -
        k[553]*y[IDX_H2OII] - k[583]*y[IDX_H3II] - k[584]*y[IDX_H3II] -
        k[621]*y[IDX_HCNII] - k[669]*y[IDX_HeII] - k[720]*y[IDX_NII] -
        k[751]*y[IDX_NHII] - k[838]*y[IDX_OHII] - k[951]*y[IDX_HNOI] -
        k[952]*y[IDX_NO2I] - k[953]*y[IDX_O2I] - k[954]*y[IDX_O2HI] -
        k[972]*y[IDX_HI] - k[1092]*y[IDX_OHI] - k[1104]*y[IDX_SiI] - k[1133] -
        k[1227] - k[1251] - k[1304];
    IJth(jmatrix, 37, 38) = 0.0 + k[28]*y[IDX_CI] + k[38]*y[IDX_CH2I] +
        k[52]*y[IDX_CH4I] + k[54]*y[IDX_CHI] + k[70]*y[IDX_H2COI] +
        k[71]*y[IDX_HCOI] + k[72]*y[IDX_NOI] + k[73]*y[IDX_O2I] +
        k[123]*y[IDX_H2OI] + k[127]*y[IDX_HI] + k[134]*y[IDX_HCNI] +
        k[184]*y[IDX_NH2I] + k[193]*y[IDX_NH3I] + k[200]*y[IDX_NHI] +
        k[217]*y[IDX_OI] + k[226]*y[IDX_OHI];
    IJth(jmatrix, 37, 39) = 0.0 + k[252] + k[367]*y[IDX_CII] +
        k[398]*y[IDX_CHII] + k[416]*y[IDX_CH2II] + k[666]*y[IDX_HeII] +
        k[749]*y[IDX_NHII] + k[812]*y[IDX_OII] + k[923]*y[IDX_CHI] +
        k[971]*y[IDX_HI] + k[1012]*y[IDX_NI] + k[1059]*y[IDX_OI] +
        k[1103]*y[IDX_SiI] + k[1132];
    IJth(jmatrix, 37, 40) = 0.0 + k[307]*y[IDX_H2COII] +
        k[308]*y[IDX_H2COII] + k[319]*y[IDX_H3COII] + k[330]*y[IDX_HCOII] +
        k[332]*y[IDX_HCO2II] + k[333]*y[IDX_HCO2II] + k[335]*y[IDX_HOCII];
    IJth(jmatrix, 37, 41) = 0.0 + k[127]*y[IDX_COII] + k[971]*y[IDX_CO2I] -
        k[972]*y[IDX_COI] + k[977]*y[IDX_HCOI] + k[994]*y[IDX_OCNI];
    IJth(jmatrix, 37, 42) = 0.0 + k[501]*y[IDX_HCOI] + k[502]*y[IDX_HNCOI];
    IJth(jmatrix, 37, 44) = 0.0 - k[104]*y[IDX_COI] - k[515]*y[IDX_COI] +
        k[519]*y[IDX_HCOI];
    IJth(jmatrix, 37, 46) = 0.0 + k[70]*y[IDX_COII] + k[255] +
        k[368]*y[IDX_CII] + k[399]*y[IDX_CHII] + k[635]*y[IDX_HCOII] + k[1136] +
        k[1137];
    IJth(jmatrix, 37, 47) = 0.0 + k[307]*y[IDX_EM] + k[308]*y[IDX_EM] +
        k[641]*y[IDX_HCOI];
    IJth(jmatrix, 37, 49) = 0.0 + k[123]*y[IDX_COII] + k[566]*y[IDX_HCOII];
    IJth(jmatrix, 37, 50) = 0.0 - k[553]*y[IDX_COI] + k[557]*y[IDX_HCOI];
    IJth(jmatrix, 37, 52) = 0.0 - k[583]*y[IDX_COI] - k[584]*y[IDX_COI];
    IJth(jmatrix, 37, 53) = 0.0 + k[319]*y[IDX_EM];
    IJth(jmatrix, 37, 55) = 0.0 + k[134]*y[IDX_COII] + k[629]*y[IDX_HCOII] +
        k[1064]*y[IDX_OI] + k[1095]*y[IDX_OHI];
    IJth(jmatrix, 37, 56) = 0.0 - k[621]*y[IDX_COI] + k[625]*y[IDX_HCOI];
    IJth(jmatrix, 37, 58) = 0.0 + k[71]*y[IDX_COII] + k[260] +
        k[372]*y[IDX_CII] + k[406]*y[IDX_CHII] + k[419]*y[IDX_CH2II] +
        k[441]*y[IDX_CH3II] + k[480]*y[IDX_CNII] + k[501]*y[IDX_HII] +
        k[519]*y[IDX_H2II] + k[557]*y[IDX_H2OII] + k[625]*y[IDX_HCNII] +
        k[636]*y[IDX_HCOII] + k[641]*y[IDX_H2COII] + k[644]*y[IDX_O2II] +
        k[681]*y[IDX_HeII] + k[723]*y[IDX_NII] + k[731]*y[IDX_N2II] +
        k[816]*y[IDX_OII] + k[842]*y[IDX_OHII] + k[864]*y[IDX_CI] +
        k[882]*y[IDX_CH2I] + k[905]*y[IDX_CH3I] + k[925]*y[IDX_CHI] +
        k[943]*y[IDX_CNI] + k[977]*y[IDX_HI] + k[997]*y[IDX_HCOI] +
        k[997]*y[IDX_HCOI] + k[997]*y[IDX_HCOI] + k[997]*y[IDX_HCOI] +
        k[998]*y[IDX_HCOI] + k[998]*y[IDX_HCOI] + k[1000]*y[IDX_NOI] +
        k[1002]*y[IDX_O2I] + k[1014]*y[IDX_NI] + k[1067]*y[IDX_OI] +
        k[1096]*y[IDX_OHI] + k[1149];
    IJth(jmatrix, 37, 59) = 0.0 + k[330]*y[IDX_EM] + k[386]*y[IDX_CI] +
        k[429]*y[IDX_CH2I] + k[466]*y[IDX_CHI] + k[566]*y[IDX_H2OI] +
        k[629]*y[IDX_HCNI] + k[635]*y[IDX_H2COI] + k[636]*y[IDX_HCOI] +
        k[637]*y[IDX_SiH2I] + k[638]*y[IDX_SiH4I] + k[639]*y[IDX_SiHI] +
        k[640]*y[IDX_SiOI] + k[648]*y[IDX_HNCI] + k[787]*y[IDX_NH2I] +
        k[799]*y[IDX_NHI] + k[854]*y[IDX_OHI] + k[861]*y[IDX_SiI];
    IJth(jmatrix, 37, 60) = 0.0 + k[332]*y[IDX_EM] + k[333]*y[IDX_EM] -
        k[485]*y[IDX_COI];
    IJth(jmatrix, 37, 62) = 0.0 + k[666]*y[IDX_CO2I] - k[669]*y[IDX_COI] +
        k[681]*y[IDX_HCOI];
    IJth(jmatrix, 37, 64) = 0.0 + k[648]*y[IDX_HCOII];
    IJth(jmatrix, 37, 65) = 0.0 + k[263] + k[502]*y[IDX_HII] +
        k[1004]*y[IDX_CI] + k[1152];
    IJth(jmatrix, 37, 66) = 0.0 - k[951]*y[IDX_COI];
    IJth(jmatrix, 37, 67) = 0.0 - k[486]*y[IDX_COI];
    IJth(jmatrix, 37, 68) = 0.0 + k[335]*y[IDX_EM];
    IJth(jmatrix, 37, 71) = 0.0 + k[1012]*y[IDX_CO2I] + k[1014]*y[IDX_HCOI];
    IJth(jmatrix, 37, 72) = 0.0 - k[158]*y[IDX_COI] - k[720]*y[IDX_COI] +
        k[723]*y[IDX_HCOI];
    IJth(jmatrix, 37, 74) = 0.0 - k[74]*y[IDX_COI] + k[731]*y[IDX_HCOI];
    IJth(jmatrix, 37, 75) = 0.0 - k[487]*y[IDX_COI];
    IJth(jmatrix, 37, 76) = 0.0 + k[200]*y[IDX_COII] + k[799]*y[IDX_HCOII];
    IJth(jmatrix, 37, 77) = 0.0 + k[749]*y[IDX_CO2I] - k[751]*y[IDX_COI];
    IJth(jmatrix, 37, 78) = 0.0 + k[184]*y[IDX_COII] + k[787]*y[IDX_HCOII];
    IJth(jmatrix, 37, 80) = 0.0 + k[193]*y[IDX_COII];
    IJth(jmatrix, 37, 82) = 0.0 + k[72]*y[IDX_COII] + k[872]*y[IDX_CI] +
        k[946]*y[IDX_CNI] + k[1000]*y[IDX_HCOI];
    IJth(jmatrix, 37, 84) = 0.0 - k[952]*y[IDX_COI];
    IJth(jmatrix, 37, 85) = 0.0 + k[217]*y[IDX_COII] + k[894]*y[IDX_CH2I] +
        k[895]*y[IDX_CH2I] + k[915]*y[IDX_CH3I] + k[939]*y[IDX_CHI] +
        k[1057]*y[IDX_CNI] + k[1059]*y[IDX_CO2I] + k[1064]*y[IDX_HCNI] +
        k[1067]*y[IDX_HCOI] + k[1078]*y[IDX_OCNI] + k[1081]*y[IDX_SiC2I] +
        k[1082]*y[IDX_SiC3I] + k[1083]*y[IDX_SiCI] + k[1196]*y[IDX_CI];
    IJth(jmatrix, 37, 86) = 0.0 - k[208]*y[IDX_COI] + k[812]*y[IDX_CO2I] +
        k[816]*y[IDX_HCOI];
    IJth(jmatrix, 37, 87) = 0.0 + k[73]*y[IDX_COII] + k[377]*y[IDX_CII] +
        k[481]*y[IDX_CNII] + k[873]*y[IDX_CI] + k[891]*y[IDX_CH2I] +
        k[934]*y[IDX_CHI] + k[935]*y[IDX_CHI] + k[948]*y[IDX_CNI] -
        k[953]*y[IDX_COI] + k[1002]*y[IDX_HCOI] + k[1055]*y[IDX_OCNI];
    IJth(jmatrix, 37, 88) = 0.0 + k[644]*y[IDX_HCOI];
    IJth(jmatrix, 37, 89) = 0.0 - k[954]*y[IDX_COI];
    IJth(jmatrix, 37, 90) = 0.0 - k[488]*y[IDX_COI];
    IJth(jmatrix, 37, 91) = 0.0 + k[874]*y[IDX_CI] + k[994]*y[IDX_HI] +
        k[1055]*y[IDX_O2I] + k[1078]*y[IDX_OI];
    IJth(jmatrix, 37, 92) = 0.0 + k[226]*y[IDX_COII] + k[854]*y[IDX_HCOII] +
        k[875]*y[IDX_CI] - k[1092]*y[IDX_COI] + k[1095]*y[IDX_HCNI] +
        k[1096]*y[IDX_HCOI];
    IJth(jmatrix, 37, 93) = 0.0 - k[838]*y[IDX_COI] + k[842]*y[IDX_HCOI];
    IJth(jmatrix, 37, 94) = 0.0 + k[861]*y[IDX_HCOII] + k[1103]*y[IDX_CO2I]
        - k[1104]*y[IDX_COI];
    IJth(jmatrix, 37, 96) = 0.0 + k[1083]*y[IDX_OI];
    IJth(jmatrix, 37, 98) = 0.0 + k[1081]*y[IDX_OI];
    IJth(jmatrix, 37, 100) = 0.0 + k[1082]*y[IDX_OI];
    IJth(jmatrix, 37, 102) = 0.0 + k[639]*y[IDX_HCOII];
    IJth(jmatrix, 37, 104) = 0.0 + k[637]*y[IDX_HCOII];
    IJth(jmatrix, 37, 108) = 0.0 + k[638]*y[IDX_HCOII];
    IJth(jmatrix, 37, 109) = 0.0 - k[489]*y[IDX_COI];
    IJth(jmatrix, 37, 111) = 0.0 + k[382]*y[IDX_CII] + k[640]*y[IDX_HCOII];
    IJth(jmatrix, 37, 112) = 0.0 + k[395]*y[IDX_CI] - k[490]*y[IDX_COI];
    IJth(jmatrix, 38, 24) = 0.0 - k[28]*y[IDX_COII] + k[391]*y[IDX_O2II] +
        k[1195]*y[IDX_OII];
    IJth(jmatrix, 38, 25) = 0.0 + k[367]*y[IDX_CO2I] + k[376]*y[IDX_O2I] +
        k[378]*y[IDX_OCNI] + k[379]*y[IDX_OHI] + k[1193]*y[IDX_OI];
    IJth(jmatrix, 38, 26) = 0.0 - k[54]*y[IDX_COII] - k[458]*y[IDX_COII] +
        k[472]*y[IDX_OII];
    IJth(jmatrix, 38, 27) = 0.0 + k[411]*y[IDX_O2I] + k[414]*y[IDX_OI] +
        k[415]*y[IDX_OHI];
    IJth(jmatrix, 38, 28) = 0.0 - k[38]*y[IDX_COII] - k[422]*y[IDX_COII];
    IJth(jmatrix, 38, 33) = 0.0 - k[52]*y[IDX_COII] - k[451]*y[IDX_COII];
    IJth(jmatrix, 38, 36) = 0.0 + k[63]*y[IDX_COI];
    IJth(jmatrix, 38, 37) = 0.0 + k[63]*y[IDX_CNII] + k[74]*y[IDX_N2II] +
        k[104]*y[IDX_H2II] + k[158]*y[IDX_NII] + k[208]*y[IDX_OII] + k[232];
    IJth(jmatrix, 38, 38) = 0.0 - k[28]*y[IDX_CI] - k[38]*y[IDX_CH2I] -
        k[52]*y[IDX_CH4I] - k[54]*y[IDX_CHI] - k[70]*y[IDX_H2COI] -
        k[71]*y[IDX_HCOI] - k[72]*y[IDX_NOI] - k[73]*y[IDX_O2I] -
        k[123]*y[IDX_H2OI] - k[127]*y[IDX_HI] - k[134]*y[IDX_HCNI] -
        k[184]*y[IDX_NH2I] - k[193]*y[IDX_NH3I] - k[200]*y[IDX_NHI] -
        k[217]*y[IDX_OI] - k[226]*y[IDX_OHI] - k[304]*y[IDX_EM] -
        k[422]*y[IDX_CH2I] - k[451]*y[IDX_CH4I] - k[458]*y[IDX_CHI] -
        k[484]*y[IDX_H2COI] - k[532]*y[IDX_H2I] - k[533]*y[IDX_H2I] -
        k[562]*y[IDX_H2OI] - k[779]*y[IDX_NH2I] - k[792]*y[IDX_NH3I] -
        k[795]*y[IDX_NHI] - k[851]*y[IDX_OHI] - k[1131] - k[1275];
    IJth(jmatrix, 38, 39) = 0.0 + k[367]*y[IDX_CII] + k[665]*y[IDX_HeII] +
        k[719]*y[IDX_NII];
    IJth(jmatrix, 38, 40) = 0.0 - k[304]*y[IDX_COII];
    IJth(jmatrix, 38, 41) = 0.0 - k[127]*y[IDX_COII];
    IJth(jmatrix, 38, 42) = 0.0 + k[497]*y[IDX_H2COI] + k[500]*y[IDX_HCOI];
    IJth(jmatrix, 38, 43) = 0.0 - k[532]*y[IDX_COII] - k[533]*y[IDX_COII];
    IJth(jmatrix, 38, 44) = 0.0 + k[104]*y[IDX_COI];
    IJth(jmatrix, 38, 46) = 0.0 - k[70]*y[IDX_COII] - k[484]*y[IDX_COII] +
        k[497]*y[IDX_HII] + k[670]*y[IDX_HeII];
    IJth(jmatrix, 38, 49) = 0.0 - k[123]*y[IDX_COII] - k[562]*y[IDX_COII];
    IJth(jmatrix, 38, 55) = 0.0 - k[134]*y[IDX_COII];
    IJth(jmatrix, 38, 58) = 0.0 - k[71]*y[IDX_COII] + k[500]*y[IDX_HII] +
        k[680]*y[IDX_HeII];
    IJth(jmatrix, 38, 59) = 0.0 + k[1148];
    IJth(jmatrix, 38, 62) = 0.0 + k[665]*y[IDX_CO2I] + k[670]*y[IDX_H2COI] +
        k[680]*y[IDX_HCOI];
    IJth(jmatrix, 38, 72) = 0.0 + k[158]*y[IDX_COI] + k[719]*y[IDX_CO2I];
    IJth(jmatrix, 38, 74) = 0.0 + k[74]*y[IDX_COI];
    IJth(jmatrix, 38, 76) = 0.0 - k[200]*y[IDX_COII] - k[795]*y[IDX_COII];
    IJth(jmatrix, 38, 78) = 0.0 - k[184]*y[IDX_COII] - k[779]*y[IDX_COII];
    IJth(jmatrix, 38, 80) = 0.0 - k[193]*y[IDX_COII] - k[792]*y[IDX_COII];
    IJth(jmatrix, 38, 82) = 0.0 - k[72]*y[IDX_COII];
    IJth(jmatrix, 38, 85) = 0.0 - k[217]*y[IDX_COII] + k[414]*y[IDX_CHII] +
        k[1193]*y[IDX_CII];
    IJth(jmatrix, 38, 86) = 0.0 + k[208]*y[IDX_COI] + k[472]*y[IDX_CHI] +
        k[1195]*y[IDX_CI];
    IJth(jmatrix, 38, 87) = 0.0 - k[73]*y[IDX_COII] + k[376]*y[IDX_CII] +
        k[411]*y[IDX_CHII];
    IJth(jmatrix, 38, 88) = 0.0 + k[391]*y[IDX_CI];
    IJth(jmatrix, 38, 91) = 0.0 + k[378]*y[IDX_CII];
    IJth(jmatrix, 38, 92) = 0.0 - k[226]*y[IDX_COII] + k[379]*y[IDX_CII] +
        k[415]*y[IDX_CHII] - k[851]*y[IDX_COII];
    IJth(jmatrix, 39, 3) = 0.0 + k[1383] + k[1384] + k[1385] + k[1386];
    IJth(jmatrix, 39, 24) = 0.0 + k[387]*y[IDX_HCO2II];
    IJth(jmatrix, 39, 25) = 0.0 - k[367]*y[IDX_CO2I];
    IJth(jmatrix, 39, 26) = 0.0 - k[923]*y[IDX_CO2I] + k[933]*y[IDX_O2I];
    IJth(jmatrix, 39, 27) = 0.0 - k[398]*y[IDX_CO2I];
    IJth(jmatrix, 39, 28) = 0.0 + k[889]*y[IDX_O2I] + k[890]*y[IDX_O2I];
    IJth(jmatrix, 39, 29) = 0.0 - k[416]*y[IDX_CO2I];
    IJth(jmatrix, 39, 34) = 0.0 - k[447]*y[IDX_CO2I];
    IJth(jmatrix, 39, 37) = 0.0 + k[485]*y[IDX_HCO2II] + k[490]*y[IDX_SiOII]
        + k[951]*y[IDX_HNOI] + k[952]*y[IDX_NO2I] + k[953]*y[IDX_O2I] +
        k[954]*y[IDX_O2HI] + k[1092]*y[IDX_OHI];
    IJth(jmatrix, 39, 39) = 0.0 - k[252] - k[367]*y[IDX_CII] -
        k[398]*y[IDX_CHII] - k[416]*y[IDX_CH2II] - k[447]*y[IDX_CH4II] -
        k[496]*y[IDX_HII] - k[514]*y[IDX_H2II] - k[582]*y[IDX_H3II] -
        k[620]*y[IDX_HCNII] - k[652]*y[IDX_HNOII] - k[665]*y[IDX_HeII] -
        k[666]*y[IDX_HeII] - k[667]*y[IDX_HeII] - k[668]*y[IDX_HeII] -
        k[719]*y[IDX_NII] - k[734]*y[IDX_N2HII] - k[748]*y[IDX_NHII] -
        k[749]*y[IDX_NHII] - k[750]*y[IDX_NHII] - k[812]*y[IDX_OII] -
        k[821]*y[IDX_O2HII] - k[837]*y[IDX_OHII] - k[923]*y[IDX_CHI] -
        k[971]*y[IDX_HI] - k[1012]*y[IDX_NI] - k[1059]*y[IDX_OI] -
        k[1103]*y[IDX_SiI] - k[1132] - k[1258];
    IJth(jmatrix, 39, 40) = 0.0 + k[331]*y[IDX_HCO2II];
    IJth(jmatrix, 39, 41) = 0.0 - k[971]*y[IDX_CO2I];
    IJth(jmatrix, 39, 42) = 0.0 - k[496]*y[IDX_CO2I];
    IJth(jmatrix, 39, 44) = 0.0 - k[514]*y[IDX_CO2I];
    IJth(jmatrix, 39, 49) = 0.0 + k[567]*y[IDX_HCO2II];
    IJth(jmatrix, 39, 52) = 0.0 - k[582]*y[IDX_CO2I];
    IJth(jmatrix, 39, 56) = 0.0 - k[620]*y[IDX_CO2I];
    IJth(jmatrix, 39, 58) = 0.0 + k[1001]*y[IDX_O2I] + k[1066]*y[IDX_OI];
    IJth(jmatrix, 39, 60) = 0.0 + k[331]*y[IDX_EM] + k[387]*y[IDX_CI] +
        k[485]*y[IDX_COI] + k[567]*y[IDX_H2OI];
    IJth(jmatrix, 39, 62) = 0.0 - k[665]*y[IDX_CO2I] - k[666]*y[IDX_CO2I] -
        k[667]*y[IDX_CO2I] - k[668]*y[IDX_CO2I];
    IJth(jmatrix, 39, 66) = 0.0 + k[951]*y[IDX_COI];
    IJth(jmatrix, 39, 67) = 0.0 - k[652]*y[IDX_CO2I];
    IJth(jmatrix, 39, 71) = 0.0 - k[1012]*y[IDX_CO2I];
    IJth(jmatrix, 39, 72) = 0.0 - k[719]*y[IDX_CO2I];
    IJth(jmatrix, 39, 75) = 0.0 - k[734]*y[IDX_CO2I];
    IJth(jmatrix, 39, 77) = 0.0 - k[748]*y[IDX_CO2I] - k[749]*y[IDX_CO2I] -
        k[750]*y[IDX_CO2I];
    IJth(jmatrix, 39, 82) = 0.0 + k[1053]*y[IDX_OCNI];
    IJth(jmatrix, 39, 84) = 0.0 + k[952]*y[IDX_COI];
    IJth(jmatrix, 39, 85) = 0.0 - k[1059]*y[IDX_CO2I] + k[1066]*y[IDX_HCOI];
    IJth(jmatrix, 39, 86) = 0.0 - k[812]*y[IDX_CO2I];
    IJth(jmatrix, 39, 87) = 0.0 + k[889]*y[IDX_CH2I] + k[890]*y[IDX_CH2I] +
        k[933]*y[IDX_CHI] + k[953]*y[IDX_COI] + k[1001]*y[IDX_HCOI] +
        k[1054]*y[IDX_OCNI];
    IJth(jmatrix, 39, 89) = 0.0 + k[954]*y[IDX_COI];
    IJth(jmatrix, 39, 90) = 0.0 - k[821]*y[IDX_CO2I];
    IJth(jmatrix, 39, 91) = 0.0 + k[1053]*y[IDX_NOI] + k[1054]*y[IDX_O2I];
    IJth(jmatrix, 39, 92) = 0.0 + k[1092]*y[IDX_COI];
    IJth(jmatrix, 39, 93) = 0.0 - k[837]*y[IDX_CO2I];
    IJth(jmatrix, 39, 94) = 0.0 - k[1103]*y[IDX_CO2I];
    IJth(jmatrix, 39, 112) = 0.0 + k[490]*y[IDX_COI];
    IJth(jmatrix, 40, 24) = 0.0 + k[231] + k[240] + k[1107];
    IJth(jmatrix, 40, 25) = 0.0 - k[1214]*y[IDX_EM];
    IJth(jmatrix, 40, 26) = 0.0 + k[0]*y[IDX_OI] + k[1129];
    IJth(jmatrix, 40, 27) = 0.0 - k[294]*y[IDX_EM];
    IJth(jmatrix, 40, 28) = 0.0 + k[242] + k[1112];
    IJth(jmatrix, 40, 29) = 0.0 - k[295]*y[IDX_EM] - k[296]*y[IDX_EM] -
        k[297]*y[IDX_EM];
    IJth(jmatrix, 40, 30) = 0.0 + k[245] + k[1117];
    IJth(jmatrix, 40, 31) = 0.0 - k[298]*y[IDX_EM] - k[299]*y[IDX_EM] -
        k[300]*y[IDX_EM] - k[1215]*y[IDX_EM];
    IJth(jmatrix, 40, 32) = 0.0 + k[1120];
    IJth(jmatrix, 40, 33) = 0.0 + k[1126];
    IJth(jmatrix, 40, 34) = 0.0 - k[301]*y[IDX_EM] - k[302]*y[IDX_EM];
    IJth(jmatrix, 40, 36) = 0.0 - k[303]*y[IDX_EM];
    IJth(jmatrix, 40, 37) = 0.0 + k[232];
    IJth(jmatrix, 40, 38) = 0.0 - k[304]*y[IDX_EM];
    IJth(jmatrix, 40, 40) = 0.0 - k[8]*y[IDX_H2I] + k[8]*y[IDX_H2I] -
        k[294]*y[IDX_CHII] - k[295]*y[IDX_CH2II] - k[296]*y[IDX_CH2II] -
        k[297]*y[IDX_CH2II] - k[298]*y[IDX_CH3II] - k[299]*y[IDX_CH3II] -
        k[300]*y[IDX_CH3II] - k[301]*y[IDX_CH4II] - k[302]*y[IDX_CH4II] -
        k[303]*y[IDX_CNII] - k[304]*y[IDX_COII] - k[305]*y[IDX_H2II] -
        k[306]*y[IDX_H2COII] - k[307]*y[IDX_H2COII] - k[308]*y[IDX_H2COII] -
        k[309]*y[IDX_H2COII] - k[310]*y[IDX_H2NOII] - k[311]*y[IDX_H2NOII] -
        k[312]*y[IDX_H2OII] - k[313]*y[IDX_H2OII] - k[314]*y[IDX_H2OII] -
        k[315]*y[IDX_H3II] - k[316]*y[IDX_H3II] - k[317]*y[IDX_H3COII] -
        k[318]*y[IDX_H3COII] - k[319]*y[IDX_H3COII] - k[320]*y[IDX_H3COII] -
        k[321]*y[IDX_H3COII] - k[322]*y[IDX_H3OII] - k[323]*y[IDX_H3OII] -
        k[324]*y[IDX_H3OII] - k[325]*y[IDX_H3OII] - k[326]*y[IDX_HCNII] -
        k[327]*y[IDX_HCNHII] - k[328]*y[IDX_HCNHII] - k[329]*y[IDX_HCNHII] -
        k[330]*y[IDX_HCOII] - k[331]*y[IDX_HCO2II] - k[332]*y[IDX_HCO2II] -
        k[333]*y[IDX_HCO2II] - k[334]*y[IDX_HNOII] - k[335]*y[IDX_HOCII] -
        k[336]*y[IDX_HeHII] - k[337]*y[IDX_N2II] - k[338]*y[IDX_N2HII] -
        k[339]*y[IDX_N2HII] - k[340]*y[IDX_NHII] - k[341]*y[IDX_NH2II] -
        k[342]*y[IDX_NH2II] - k[343]*y[IDX_NH3II] - k[344]*y[IDX_NH3II] -
        k[345]*y[IDX_NOII] - k[346]*y[IDX_O2II] - k[347]*y[IDX_O2HII] -
        k[348]*y[IDX_OHII] - k[349]*y[IDX_SiCII] - k[350]*y[IDX_SiC2II] -
        k[351]*y[IDX_SiC3II] - k[352]*y[IDX_SiHII] - k[353]*y[IDX_SiH2II] -
        k[354]*y[IDX_SiH2II] - k[355]*y[IDX_SiH2II] - k[356]*y[IDX_SiH3II] -
        k[357]*y[IDX_SiH3II] - k[358]*y[IDX_SiH4II] - k[359]*y[IDX_SiH4II] -
        k[360]*y[IDX_SiH5II] - k[361]*y[IDX_SiH5II] - k[362]*y[IDX_SiOII] -
        k[363]*y[IDX_SiOHII] - k[364]*y[IDX_SiOHII] - k[1214]*y[IDX_CII] -
        k[1215]*y[IDX_CH3II] - k[1216]*y[IDX_HII] - k[1217]*y[IDX_H2COII] -
        k[1218]*y[IDX_HeII] - k[1219]*y[IDX_MgII] - k[1220]*y[IDX_NII] -
        k[1221]*y[IDX_OII] - k[1222]*y[IDX_SiII] - k[1306];
    IJth(jmatrix, 40, 41) = 0.0 + k[236] + k[258];
    IJth(jmatrix, 40, 42) = 0.0 - k[1216]*y[IDX_EM];
    IJth(jmatrix, 40, 43) = 0.0 - k[8]*y[IDX_EM] + k[8]*y[IDX_EM] + k[233] +
        k[234];
    IJth(jmatrix, 40, 44) = 0.0 - k[305]*y[IDX_EM];
    IJth(jmatrix, 40, 46) = 0.0 + k[1138] + k[1139];
    IJth(jmatrix, 40, 47) = 0.0 - k[306]*y[IDX_EM] - k[307]*y[IDX_EM] -
        k[308]*y[IDX_EM] - k[309]*y[IDX_EM] - k[1217]*y[IDX_EM];
    IJth(jmatrix, 40, 48) = 0.0 - k[310]*y[IDX_EM] - k[311]*y[IDX_EM];
    IJth(jmatrix, 40, 49) = 0.0 + k[1141];
    IJth(jmatrix, 40, 50) = 0.0 - k[312]*y[IDX_EM] - k[313]*y[IDX_EM] -
        k[314]*y[IDX_EM];
    IJth(jmatrix, 40, 52) = 0.0 - k[315]*y[IDX_EM] - k[316]*y[IDX_EM];
    IJth(jmatrix, 40, 53) = 0.0 - k[317]*y[IDX_EM] - k[318]*y[IDX_EM] -
        k[319]*y[IDX_EM] - k[320]*y[IDX_EM] - k[321]*y[IDX_EM];
    IJth(jmatrix, 40, 54) = 0.0 - k[322]*y[IDX_EM] - k[323]*y[IDX_EM] -
        k[324]*y[IDX_EM] - k[325]*y[IDX_EM];
    IJth(jmatrix, 40, 56) = 0.0 - k[326]*y[IDX_EM];
    IJth(jmatrix, 40, 57) = 0.0 - k[327]*y[IDX_EM] - k[328]*y[IDX_EM] -
        k[329]*y[IDX_EM];
    IJth(jmatrix, 40, 58) = 0.0 + k[261] + k[1150];
    IJth(jmatrix, 40, 59) = 0.0 - k[330]*y[IDX_EM];
    IJth(jmatrix, 40, 60) = 0.0 - k[331]*y[IDX_EM] - k[332]*y[IDX_EM] -
        k[333]*y[IDX_EM];
    IJth(jmatrix, 40, 61) = 0.0 + k[237] + k[265];
    IJth(jmatrix, 40, 62) = 0.0 - k[1218]*y[IDX_EM];
    IJth(jmatrix, 40, 63) = 0.0 - k[336]*y[IDX_EM];
    IJth(jmatrix, 40, 67) = 0.0 - k[334]*y[IDX_EM];
    IJth(jmatrix, 40, 68) = 0.0 - k[335]*y[IDX_EM];
    IJth(jmatrix, 40, 69) = 0.0 + k[266] + k[1154];
    IJth(jmatrix, 40, 70) = 0.0 - k[1219]*y[IDX_EM];
    IJth(jmatrix, 40, 71) = 0.0 + k[238] + k[268];
    IJth(jmatrix, 40, 72) = 0.0 - k[1220]*y[IDX_EM];
    IJth(jmatrix, 40, 74) = 0.0 - k[337]*y[IDX_EM];
    IJth(jmatrix, 40, 75) = 0.0 - k[338]*y[IDX_EM] - k[339]*y[IDX_EM];
    IJth(jmatrix, 40, 76) = 0.0 + k[275] + k[1163];
    IJth(jmatrix, 40, 77) = 0.0 - k[340]*y[IDX_EM];
    IJth(jmatrix, 40, 78) = 0.0 + k[269] + k[1157];
    IJth(jmatrix, 40, 79) = 0.0 - k[341]*y[IDX_EM] - k[342]*y[IDX_EM];
    IJth(jmatrix, 40, 80) = 0.0 + k[272] + k[1160];
    IJth(jmatrix, 40, 81) = 0.0 - k[343]*y[IDX_EM] - k[344]*y[IDX_EM];
    IJth(jmatrix, 40, 82) = 0.0 + k[277] + k[1165];
    IJth(jmatrix, 40, 83) = 0.0 - k[345]*y[IDX_EM];
    IJth(jmatrix, 40, 85) = 0.0 + k[0]*y[IDX_CHI] + k[239] + k[282];
    IJth(jmatrix, 40, 86) = 0.0 - k[1221]*y[IDX_EM];
    IJth(jmatrix, 40, 87) = 0.0 + k[279] + k[1168];
    IJth(jmatrix, 40, 88) = 0.0 - k[346]*y[IDX_EM];
    IJth(jmatrix, 40, 90) = 0.0 - k[347]*y[IDX_EM];
    IJth(jmatrix, 40, 92) = 0.0 + k[1175];
    IJth(jmatrix, 40, 93) = 0.0 - k[348]*y[IDX_EM];
    IJth(jmatrix, 40, 94) = 0.0 + k[285] + k[1176];
    IJth(jmatrix, 40, 95) = 0.0 - k[1222]*y[IDX_EM];
    IJth(jmatrix, 40, 97) = 0.0 - k[349]*y[IDX_EM];
    IJth(jmatrix, 40, 99) = 0.0 - k[350]*y[IDX_EM];
    IJth(jmatrix, 40, 101) = 0.0 - k[351]*y[IDX_EM];
    IJth(jmatrix, 40, 103) = 0.0 - k[352]*y[IDX_EM];
    IJth(jmatrix, 40, 104) = 0.0 + k[1180];
    IJth(jmatrix, 40, 105) = 0.0 - k[353]*y[IDX_EM] - k[354]*y[IDX_EM] -
        k[355]*y[IDX_EM];
    IJth(jmatrix, 40, 106) = 0.0 + k[1183];
    IJth(jmatrix, 40, 107) = 0.0 - k[356]*y[IDX_EM] - k[357]*y[IDX_EM];
    IJth(jmatrix, 40, 109) = 0.0 - k[358]*y[IDX_EM] - k[359]*y[IDX_EM];
    IJth(jmatrix, 40, 110) = 0.0 - k[360]*y[IDX_EM] - k[361]*y[IDX_EM];
    IJth(jmatrix, 40, 111) = 0.0 + k[1191];
    IJth(jmatrix, 40, 112) = 0.0 - k[362]*y[IDX_EM];
    IJth(jmatrix, 40, 113) = 0.0 - k[363]*y[IDX_EM] - k[364]*y[IDX_EM];
    IJth(jmatrix, 41, 24) = 0.0 + k[394]*y[IDX_SiHII] + k[509]*y[IDX_H2II] +
        k[866]*y[IDX_NH2I] + k[867]*y[IDX_NH2I] + k[869]*y[IDX_NHI] +
        k[875]*y[IDX_OHI] + k[877]*y[IDX_SiHI] + k[955]*y[IDX_H2I] -
        k[1206]*y[IDX_HI];
    IJth(jmatrix, 41, 25) = 0.0 + k[370]*y[IDX_H2OI] + k[371]*y[IDX_H2OI] +
        k[373]*y[IDX_NH2I] + k[375]*y[IDX_NHI] + k[379]*y[IDX_OHI] +
        k[381]*y[IDX_SiHI] + k[528]*y[IDX_H2I] - k[1205]*y[IDX_HI];
    IJth(jmatrix, 41, 26) = 0.0 + k[2]*y[IDX_H2I] - k[9]*y[IDX_HI] +
        k[9]*y[IDX_HI] + k[9]*y[IDX_HI] + k[78]*y[IDX_HII] + k[250] +
        k[468]*y[IDX_NII] + k[472]*y[IDX_OII] + k[476]*y[IDX_SiII] +
        k[512]*y[IDX_H2II] + k[662]*y[IDX_HeII] + k[928]*y[IDX_NI] +
        k[932]*y[IDX_NOI] + k[933]*y[IDX_O2I] + k[934]*y[IDX_O2I] +
        k[939]*y[IDX_OI] + k[941]*y[IDX_OHI] + k[958]*y[IDX_H2I] -
        k[970]*y[IDX_HI] + k[1128];
    IJth(jmatrix, 41, 27) = 0.0 + k[241] + k[294]*y[IDX_EM] +
        k[402]*y[IDX_H2OI] + k[408]*y[IDX_NI] + k[414]*y[IDX_OI] +
        k[529]*y[IDX_H2I] - k[614]*y[IDX_HI];
    IJth(jmatrix, 41, 28) = 0.0 + k[75]*y[IDX_HII] + k[243] +
        k[510]*y[IDX_H2II] + k[654]*y[IDX_HeII] + k[888]*y[IDX_NOI] +
        k[890]*y[IDX_O2I] + k[890]*y[IDX_O2I] + k[895]*y[IDX_OI] +
        k[895]*y[IDX_OI] + k[896]*y[IDX_OI] + k[898]*y[IDX_OHI] +
        k[956]*y[IDX_H2I] - k[967]*y[IDX_HI] + k[1005]*y[IDX_NI] +
        k[1006]*y[IDX_NI] + k[1113];
    IJth(jmatrix, 41, 29) = 0.0 + k[296]*y[IDX_EM] + k[296]*y[IDX_EM] +
        k[297]*y[IDX_EM] + k[418]*y[IDX_H2OI] + k[421]*y[IDX_OI] +
        k[530]*y[IDX_H2I] - k[615]*y[IDX_HI] + k[736]*y[IDX_NI] + k[1110];
    IJth(jmatrix, 41, 30) = 0.0 + k[76]*y[IDX_HII] + k[244] +
        k[915]*y[IDX_OI] + k[916]*y[IDX_OI] + k[957]*y[IDX_H2I] -
        k[968]*y[IDX_HI] + k[1008]*y[IDX_NI] + k[1010]*y[IDX_NI] +
        k[1010]*y[IDX_NI] + k[1116];
    IJth(jmatrix, 41, 31) = 0.0 + k[298]*y[IDX_EM] + k[300]*y[IDX_EM] +
        k[300]*y[IDX_EM] + k[443]*y[IDX_OI] - k[616]*y[IDX_HI] + k[1115];
    IJth(jmatrix, 41, 32) = 0.0 + k[712]*y[IDX_NII] + k[714]*y[IDX_NII] +
        k[715]*y[IDX_NII] + k[820]*y[IDX_O2II] + k[1120];
    IJth(jmatrix, 41, 33) = 0.0 + k[77]*y[IDX_HII] + k[456]*y[IDX_N2II] +
        k[511]*y[IDX_H2II] + k[658]*y[IDX_HeII] + k[660]*y[IDX_HeII] +
        k[716]*y[IDX_NII] + k[717]*y[IDX_NII] + k[718]*y[IDX_NII] +
        k[718]*y[IDX_NII] - k[969]*y[IDX_HI] + k[1125] + k[1127];
    IJth(jmatrix, 41, 34) = 0.0 + k[301]*y[IDX_EM] + k[301]*y[IDX_EM] +
        k[302]*y[IDX_EM] - k[617]*y[IDX_HI] + k[1123];
    IJth(jmatrix, 41, 35) = 0.0 + k[513]*y[IDX_H2II] + k[959]*y[IDX_H2I] +
        k[1091]*y[IDX_OHI];
    IJth(jmatrix, 41, 36) = 0.0 - k[126]*y[IDX_HI] + k[531]*y[IDX_H2I];
    IJth(jmatrix, 41, 37) = 0.0 + k[515]*y[IDX_H2II] - k[972]*y[IDX_HI] +
        k[1092]*y[IDX_OHI];
    IJth(jmatrix, 41, 38) = 0.0 - k[127]*y[IDX_HI] + k[532]*y[IDX_H2I] +
        k[533]*y[IDX_H2I];
    IJth(jmatrix, 41, 39) = 0.0 + k[514]*y[IDX_H2II] - k[971]*y[IDX_HI];
    IJth(jmatrix, 41, 40) = 0.0 + k[8]*y[IDX_H2I] + k[8]*y[IDX_H2I] +
        k[294]*y[IDX_CHII] + k[296]*y[IDX_CH2II] + k[296]*y[IDX_CH2II] +
        k[297]*y[IDX_CH2II] + k[298]*y[IDX_CH3II] + k[300]*y[IDX_CH3II] +
        k[300]*y[IDX_CH3II] + k[301]*y[IDX_CH4II] + k[301]*y[IDX_CH4II] +
        k[302]*y[IDX_CH4II] + k[305]*y[IDX_H2II] + k[305]*y[IDX_H2II] +
        k[308]*y[IDX_H2COII] + k[308]*y[IDX_H2COII] + k[309]*y[IDX_H2COII] +
        k[310]*y[IDX_H2NOII] + k[313]*y[IDX_H2OII] + k[313]*y[IDX_H2OII] +
        k[314]*y[IDX_H2OII] + k[315]*y[IDX_H3II] + k[316]*y[IDX_H3II] +
        k[316]*y[IDX_H3II] + k[316]*y[IDX_H3II] + k[319]*y[IDX_H3COII] +
        k[320]*y[IDX_H3COII] + k[321]*y[IDX_H3COII] + k[321]*y[IDX_H3COII] +
        k[322]*y[IDX_H3OII] + k[323]*y[IDX_H3OII] + k[325]*y[IDX_H3OII] +
        k[325]*y[IDX_H3OII] + k[326]*y[IDX_HCNII] + k[327]*y[IDX_HCNHII] +
        k[327]*y[IDX_HCNHII] + k[328]*y[IDX_HCNHII] + k[329]*y[IDX_HCNHII] +
        k[330]*y[IDX_HCOII] + k[331]*y[IDX_HCO2II] + k[332]*y[IDX_HCO2II] +
        k[334]*y[IDX_HNOII] + k[335]*y[IDX_HOCII] + k[336]*y[IDX_HeHII] +
        k[338]*y[IDX_N2HII] + k[340]*y[IDX_NHII] + k[341]*y[IDX_NH2II] +
        k[341]*y[IDX_NH2II] + k[342]*y[IDX_NH2II] + k[343]*y[IDX_NH3II] +
        k[344]*y[IDX_NH3II] + k[344]*y[IDX_NH3II] + k[347]*y[IDX_O2HII] +
        k[348]*y[IDX_OHII] + k[352]*y[IDX_SiHII] + k[354]*y[IDX_SiH2II] +
        k[354]*y[IDX_SiH2II] + k[355]*y[IDX_SiH2II] + k[356]*y[IDX_SiH3II] +
        k[359]*y[IDX_SiH4II] + k[361]*y[IDX_SiH5II] + k[364]*y[IDX_SiOHII] +
        k[1216]*y[IDX_HII];
    IJth(jmatrix, 41, 41) = 0.0 - k[9]*y[IDX_CHI] + k[9]*y[IDX_CHI] +
        k[9]*y[IDX_CHI] - k[10]*y[IDX_H2I] + k[10]*y[IDX_H2I] + k[10]*y[IDX_H2I]
        + k[10]*y[IDX_H2I] - k[11]*y[IDX_H2OI] + k[11]*y[IDX_H2OI] +
        k[11]*y[IDX_H2OI] - k[12]*y[IDX_O2I] + k[12]*y[IDX_O2I] -
        k[13]*y[IDX_OHI] + k[13]*y[IDX_OHI] + k[13]*y[IDX_OHI] -
        k[126]*y[IDX_CNII] - k[127]*y[IDX_COII] - k[128]*y[IDX_H2II] -
        k[129]*y[IDX_HCNII] - k[130]*y[IDX_HeII] - k[131]*y[IDX_OII] - k[236] -
        k[258] - k[614]*y[IDX_CHII] - k[615]*y[IDX_CH2II] - k[616]*y[IDX_CH3II]
        - k[617]*y[IDX_CH4II] - k[618]*y[IDX_HeHII] - k[619]*y[IDX_SiHII] -
        k[967]*y[IDX_CH2I] - k[968]*y[IDX_CH3I] - k[969]*y[IDX_CH4I] -
        k[970]*y[IDX_CHI] - k[971]*y[IDX_CO2I] - k[972]*y[IDX_COI] -
        k[973]*y[IDX_H2CNI] - k[974]*y[IDX_H2COI] - k[975]*y[IDX_H2OI] -
        k[976]*y[IDX_HCNI] - k[977]*y[IDX_HCOI] - k[978]*y[IDX_HCOI] -
        k[979]*y[IDX_HNCI] + k[979]*y[IDX_HNCI] - k[980]*y[IDX_HNOI] -
        k[981]*y[IDX_HNOI] - k[982]*y[IDX_HNOI] - k[983]*y[IDX_NH2I] -
        k[984]*y[IDX_NH3I] - k[985]*y[IDX_NHI] - k[986]*y[IDX_NO2I] -
        k[987]*y[IDX_NOI] - k[988]*y[IDX_NOI] - k[989]*y[IDX_O2I] -
        k[990]*y[IDX_O2HI] - k[991]*y[IDX_O2HI] - k[992]*y[IDX_O2HI] -
        k[993]*y[IDX_OCNI] - k[994]*y[IDX_OCNI] - k[995]*y[IDX_OCNI] -
        k[996]*y[IDX_OHI] - k[1197]*y[IDX_HII] - k[1205]*y[IDX_CII] -
        k[1206]*y[IDX_CI] - k[1207]*y[IDX_OI] - k[1208]*y[IDX_OHI] -
        k[1209]*y[IDX_SiII];
    IJth(jmatrix, 41, 42) = 0.0 + k[75]*y[IDX_CH2I] + k[76]*y[IDX_CH3I] +
        k[77]*y[IDX_CH4I] + k[78]*y[IDX_CHI] + k[79]*y[IDX_H2COI] +
        k[80]*y[IDX_H2OI] + k[81]*y[IDX_HCNI] + k[82]*y[IDX_HCOI] +
        k[83]*y[IDX_MgI] + k[84]*y[IDX_NH2I] + k[85]*y[IDX_NH3I] +
        k[86]*y[IDX_NHI] + k[87]*y[IDX_NOI] + k[88]*y[IDX_O2I] + k[89]*y[IDX_OI]
        + k[90]*y[IDX_OHI] + k[91]*y[IDX_SiI] + k[92]*y[IDX_SiC2I] +
        k[93]*y[IDX_SiC3I] + k[94]*y[IDX_SiCI] + k[95]*y[IDX_SiH2I] +
        k[96]*y[IDX_SiH3I] + k[97]*y[IDX_SiH4I] + k[98]*y[IDX_SiHI] +
        k[99]*y[IDX_SiOI] + k[497]*y[IDX_H2COI] - k[1197]*y[IDX_HI] +
        k[1216]*y[IDX_EM];
    IJth(jmatrix, 41, 43) = 0.0 + k[2]*y[IDX_CHI] + k[3]*y[IDX_H2I] +
        k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] + k[4]*y[IDX_H2OI] +
        k[7]*y[IDX_OHI] + k[8]*y[IDX_EM] + k[8]*y[IDX_EM] - k[10]*y[IDX_HI] +
        k[10]*y[IDX_HI] + k[10]*y[IDX_HI] + k[10]*y[IDX_HI] + k[233] + k[235] +
        k[235] + k[516]*y[IDX_H2II] + k[528]*y[IDX_CII] + k[529]*y[IDX_CHII] +
        k[530]*y[IDX_CH2II] + k[531]*y[IDX_CNII] + k[532]*y[IDX_COII] +
        k[533]*y[IDX_COII] + k[534]*y[IDX_H2OII] + k[535]*y[IDX_HCNII] +
        k[536]*y[IDX_HeII] + k[538]*y[IDX_NII] + k[539]*y[IDX_N2II] +
        k[541]*y[IDX_NHII] + k[542]*y[IDX_NH2II] + k[543]*y[IDX_OII] +
        k[545]*y[IDX_OHII] + k[546]*y[IDX_SiH4II] + k[547]*y[IDX_SiOII] +
        k[955]*y[IDX_CI] + k[956]*y[IDX_CH2I] + k[957]*y[IDX_CH3I] +
        k[958]*y[IDX_CHI] + k[959]*y[IDX_CNI] + k[960]*y[IDX_NI] +
        k[961]*y[IDX_NH2I] + k[962]*y[IDX_NHI] + k[963]*y[IDX_O2I] +
        k[965]*y[IDX_OI] + k[966]*y[IDX_OHI];
    IJth(jmatrix, 41, 44) = 0.0 - k[128]*y[IDX_HI] + k[305]*y[IDX_EM] +
        k[305]*y[IDX_EM] + k[509]*y[IDX_CI] + k[510]*y[IDX_CH2I] +
        k[511]*y[IDX_CH4I] + k[512]*y[IDX_CHI] + k[513]*y[IDX_CNI] +
        k[514]*y[IDX_CO2I] + k[515]*y[IDX_COI] + k[516]*y[IDX_H2I] +
        k[517]*y[IDX_H2COI] + k[518]*y[IDX_H2OI] + k[520]*y[IDX_HeI] +
        k[521]*y[IDX_N2I] + k[522]*y[IDX_NI] + k[523]*y[IDX_NHI] +
        k[524]*y[IDX_NOI] + k[525]*y[IDX_O2I] + k[526]*y[IDX_OI] +
        k[527]*y[IDX_OHI] + k[1134];
    IJth(jmatrix, 41, 45) = 0.0 + k[254] - k[973]*y[IDX_HI] + k[1135];
    IJth(jmatrix, 41, 46) = 0.0 + k[79]*y[IDX_HII] + k[497]*y[IDX_HII] +
        k[517]*y[IDX_H2II] + k[551]*y[IDX_O2II] + k[671]*y[IDX_HeII] +
        k[730]*y[IDX_N2II] - k[974]*y[IDX_HI] + k[1137] + k[1137] + k[1139];
    IJth(jmatrix, 41, 47) = 0.0 + k[308]*y[IDX_EM] + k[308]*y[IDX_EM] +
        k[309]*y[IDX_EM];
    IJth(jmatrix, 41, 48) = 0.0 + k[310]*y[IDX_EM] + k[1296];
    IJth(jmatrix, 41, 49) = 0.0 + k[4]*y[IDX_H2I] - k[11]*y[IDX_HI] +
        k[11]*y[IDX_HI] + k[11]*y[IDX_HI] + k[80]*y[IDX_HII] + k[256] +
        k[370]*y[IDX_CII] + k[371]*y[IDX_CII] + k[402]*y[IDX_CHII] +
        k[418]*y[IDX_CH2II] + k[518]*y[IDX_H2II] + k[572]*y[IDX_SiII] +
        k[673]*y[IDX_HeII] - k[975]*y[IDX_HI] + k[1142];
    IJth(jmatrix, 41, 50) = 0.0 + k[313]*y[IDX_EM] + k[313]*y[IDX_EM] +
        k[314]*y[IDX_EM] + k[534]*y[IDX_H2I] + k[738]*y[IDX_NI] + k[1140];
    IJth(jmatrix, 41, 51) = 0.0 + k[675]*y[IDX_HeII] + k[1144] + k[1144];
    IJth(jmatrix, 41, 52) = 0.0 + k[315]*y[IDX_EM] + k[316]*y[IDX_EM] +
        k[316]*y[IDX_EM] + k[316]*y[IDX_EM] + k[591]*y[IDX_MgI] +
        k[598]*y[IDX_OI] + k[1145];
    IJth(jmatrix, 41, 53) = 0.0 + k[319]*y[IDX_EM] + k[320]*y[IDX_EM] +
        k[321]*y[IDX_EM] + k[321]*y[IDX_EM];
    IJth(jmatrix, 41, 54) = 0.0 + k[322]*y[IDX_EM] + k[323]*y[IDX_EM] +
        k[325]*y[IDX_EM] + k[325]*y[IDX_EM] + k[1286];
    IJth(jmatrix, 41, 55) = 0.0 + k[81]*y[IDX_HII] + k[259] +
        k[676]*y[IDX_HeII] + k[678]*y[IDX_HeII] - k[976]*y[IDX_HI] +
        k[1065]*y[IDX_OI] + k[1147];
    IJth(jmatrix, 41, 56) = 0.0 - k[129]*y[IDX_HI] + k[326]*y[IDX_EM] +
        k[535]*y[IDX_H2I];
    IJth(jmatrix, 41, 57) = 0.0 + k[327]*y[IDX_EM] + k[327]*y[IDX_EM] +
        k[328]*y[IDX_EM] + k[329]*y[IDX_EM] + k[1301];
    IJth(jmatrix, 41, 58) = 0.0 + k[82]*y[IDX_HII] + k[260] +
        k[680]*y[IDX_HeII] - k[977]*y[IDX_HI] - k[978]*y[IDX_HI] +
        k[1016]*y[IDX_NI] + k[1066]*y[IDX_OI] + k[1149];
    IJth(jmatrix, 41, 59) = 0.0 + k[330]*y[IDX_EM] + k[855]*y[IDX_OHI] +
        k[1148];
    IJth(jmatrix, 41, 60) = 0.0 + k[331]*y[IDX_EM] + k[332]*y[IDX_EM] +
        k[1287];
    IJth(jmatrix, 41, 61) = 0.0 + k[520]*y[IDX_H2II];
    IJth(jmatrix, 41, 62) = 0.0 - k[130]*y[IDX_HI] + k[536]*y[IDX_H2I] +
        k[654]*y[IDX_CH2I] + k[658]*y[IDX_CH4I] + k[660]*y[IDX_CH4I] +
        k[662]*y[IDX_CHI] + k[671]*y[IDX_H2COI] + k[673]*y[IDX_H2OI] +
        k[675]*y[IDX_H2SiOI] + k[676]*y[IDX_HCNI] + k[678]*y[IDX_HCNI] +
        k[680]*y[IDX_HCOI] + k[683]*y[IDX_HNCI] + k[684]*y[IDX_HNCI] +
        k[686]*y[IDX_HNOI] + k[690]*y[IDX_NH2I] + k[692]*y[IDX_NH3I] +
        k[693]*y[IDX_NHI] + k[699]*y[IDX_OHI] + k[704]*y[IDX_SiH2I] +
        k[706]*y[IDX_SiH3I] + k[708]*y[IDX_SiH4I] + k[709]*y[IDX_SiHI];
    IJth(jmatrix, 41, 63) = 0.0 + k[336]*y[IDX_EM] - k[618]*y[IDX_HI];
    IJth(jmatrix, 41, 64) = 0.0 + k[262] + k[683]*y[IDX_HeII] +
        k[684]*y[IDX_HeII] - k[979]*y[IDX_HI] + k[979]*y[IDX_HI] + k[1151];
    IJth(jmatrix, 41, 66) = 0.0 + k[264] + k[686]*y[IDX_HeII] -
        k[980]*y[IDX_HI] - k[981]*y[IDX_HI] - k[982]*y[IDX_HI] +
        k[1068]*y[IDX_OI] + k[1153];
    IJth(jmatrix, 41, 67) = 0.0 + k[334]*y[IDX_EM];
    IJth(jmatrix, 41, 68) = 0.0 + k[335]*y[IDX_EM];
    IJth(jmatrix, 41, 69) = 0.0 + k[83]*y[IDX_HII] + k[591]*y[IDX_H3II];
    IJth(jmatrix, 41, 71) = 0.0 + k[408]*y[IDX_CHII] + k[522]*y[IDX_H2II] +
        k[736]*y[IDX_CH2II] + k[738]*y[IDX_H2OII] + k[740]*y[IDX_NHII] +
        k[741]*y[IDX_NH2II] + k[743]*y[IDX_OHII] + k[928]*y[IDX_CHI] +
        k[960]*y[IDX_H2I] + k[1005]*y[IDX_CH2I] + k[1006]*y[IDX_CH2I] +
        k[1008]*y[IDX_CH3I] + k[1010]*y[IDX_CH3I] + k[1010]*y[IDX_CH3I] +
        k[1016]*y[IDX_HCOI] + k[1018]*y[IDX_NHI] + k[1025]*y[IDX_OHI];
    IJth(jmatrix, 41, 72) = 0.0 + k[468]*y[IDX_CHI] + k[538]*y[IDX_H2I] +
        k[712]*y[IDX_CH3OHI] + k[714]*y[IDX_CH3OHI] + k[715]*y[IDX_CH3OHI] +
        k[716]*y[IDX_CH4I] + k[717]*y[IDX_CH4I] + k[718]*y[IDX_CH4I] +
        k[718]*y[IDX_CH4I] + k[726]*y[IDX_NHI];
    IJth(jmatrix, 41, 73) = 0.0 + k[521]*y[IDX_H2II];
    IJth(jmatrix, 41, 74) = 0.0 + k[456]*y[IDX_CH4I] + k[539]*y[IDX_H2I] +
        k[730]*y[IDX_H2COI];
    IJth(jmatrix, 41, 75) = 0.0 + k[338]*y[IDX_EM] + k[1302];
    IJth(jmatrix, 41, 76) = 0.0 + k[86]*y[IDX_HII] + k[274] +
        k[375]*y[IDX_CII] + k[523]*y[IDX_H2II] + k[693]*y[IDX_HeII] +
        k[726]*y[IDX_NII] + k[803]*y[IDX_OII] + k[869]*y[IDX_CI] +
        k[962]*y[IDX_H2I] - k[985]*y[IDX_HI] + k[1018]*y[IDX_NI] +
        k[1039]*y[IDX_NHI] + k[1039]*y[IDX_NHI] + k[1039]*y[IDX_NHI] +
        k[1039]*y[IDX_NHI] + k[1042]*y[IDX_NOI] + k[1046]*y[IDX_OI] +
        k[1049]*y[IDX_OHI] + k[1162];
    IJth(jmatrix, 41, 77) = 0.0 + k[340]*y[IDX_EM] + k[541]*y[IDX_H2I] +
        k[740]*y[IDX_NI];
    IJth(jmatrix, 41, 78) = 0.0 + k[84]*y[IDX_HII] + k[270] +
        k[373]*y[IDX_CII] + k[690]*y[IDX_HeII] + k[866]*y[IDX_CI] +
        k[867]*y[IDX_CI] + k[961]*y[IDX_H2I] - k[983]*y[IDX_HI] +
        k[1030]*y[IDX_NOI] + k[1072]*y[IDX_OI] + k[1158];
    IJth(jmatrix, 41, 79) = 0.0 + k[341]*y[IDX_EM] + k[341]*y[IDX_EM] +
        k[342]*y[IDX_EM] + k[542]*y[IDX_H2I] + k[741]*y[IDX_NI] +
        k[827]*y[IDX_OI];
    IJth(jmatrix, 41, 80) = 0.0 + k[85]*y[IDX_HII] + k[271] +
        k[692]*y[IDX_HeII] - k[984]*y[IDX_HI] + k[1159];
    IJth(jmatrix, 41, 81) = 0.0 + k[343]*y[IDX_EM] + k[344]*y[IDX_EM] +
        k[344]*y[IDX_EM];
    IJth(jmatrix, 41, 82) = 0.0 + k[87]*y[IDX_HII] + k[524]*y[IDX_H2II] +
        k[888]*y[IDX_CH2I] + k[932]*y[IDX_CHI] - k[987]*y[IDX_HI] -
        k[988]*y[IDX_HI] + k[1030]*y[IDX_NH2I] + k[1042]*y[IDX_NHI] +
        k[1099]*y[IDX_OHI];
    IJth(jmatrix, 41, 84) = 0.0 - k[986]*y[IDX_HI];
    IJth(jmatrix, 41, 85) = 0.0 + k[89]*y[IDX_HII] + k[414]*y[IDX_CHII] +
        k[421]*y[IDX_CH2II] + k[443]*y[IDX_CH3II] + k[526]*y[IDX_H2II] +
        k[598]*y[IDX_H3II] + k[827]*y[IDX_NH2II] + k[830]*y[IDX_OHII] +
        k[832]*y[IDX_SiHII] + k[833]*y[IDX_SiH2II] + k[895]*y[IDX_CH2I] +
        k[895]*y[IDX_CH2I] + k[896]*y[IDX_CH2I] + k[915]*y[IDX_CH3I] +
        k[916]*y[IDX_CH3I] + k[939]*y[IDX_CHI] + k[965]*y[IDX_H2I] +
        k[1046]*y[IDX_NHI] + k[1065]*y[IDX_HCNI] + k[1066]*y[IDX_HCOI] +
        k[1068]*y[IDX_HNOI] + k[1072]*y[IDX_NH2I] + k[1080]*y[IDX_OHI] +
        k[1086]*y[IDX_SiH2I] + k[1086]*y[IDX_SiH2I] + k[1087]*y[IDX_SiH3I] +
        k[1089]*y[IDX_SiHI] - k[1207]*y[IDX_HI];
    IJth(jmatrix, 41, 86) = 0.0 - k[131]*y[IDX_HI] + k[472]*y[IDX_CHI] +
        k[543]*y[IDX_H2I] + k[803]*y[IDX_NHI] + k[819]*y[IDX_OHI];
    IJth(jmatrix, 41, 87) = 0.0 - k[12]*y[IDX_HI] + k[12]*y[IDX_HI] +
        k[88]*y[IDX_HII] + k[525]*y[IDX_H2II] + k[890]*y[IDX_CH2I] +
        k[890]*y[IDX_CH2I] + k[933]*y[IDX_CHI] + k[934]*y[IDX_CHI] +
        k[963]*y[IDX_H2I] - k[989]*y[IDX_HI];
    IJth(jmatrix, 41, 88) = 0.0 + k[551]*y[IDX_H2COI] +
        k[820]*y[IDX_CH3OHI];
    IJth(jmatrix, 41, 89) = 0.0 + k[281] - k[990]*y[IDX_HI] -
        k[991]*y[IDX_HI] - k[992]*y[IDX_HI] + k[1170];
    IJth(jmatrix, 41, 90) = 0.0 + k[347]*y[IDX_EM];
    IJth(jmatrix, 41, 91) = 0.0 - k[993]*y[IDX_HI] - k[994]*y[IDX_HI] -
        k[995]*y[IDX_HI];
    IJth(jmatrix, 41, 92) = 0.0 + k[7]*y[IDX_H2I] - k[13]*y[IDX_HI] +
        k[13]*y[IDX_HI] + k[13]*y[IDX_HI] + k[90]*y[IDX_HII] + k[284] +
        k[379]*y[IDX_CII] + k[527]*y[IDX_H2II] + k[699]*y[IDX_HeII] +
        k[819]*y[IDX_OII] + k[855]*y[IDX_HCOII] + k[859]*y[IDX_SiII] +
        k[875]*y[IDX_CI] + k[898]*y[IDX_CH2I] + k[941]*y[IDX_CHI] +
        k[966]*y[IDX_H2I] - k[996]*y[IDX_HI] + k[1025]*y[IDX_NI] +
        k[1049]*y[IDX_NHI] + k[1080]*y[IDX_OI] + k[1091]*y[IDX_CNI] +
        k[1092]*y[IDX_COI] + k[1099]*y[IDX_NOI] + k[1102]*y[IDX_SiI] + k[1174] -
        k[1208]*y[IDX_HI];
    IJth(jmatrix, 41, 93) = 0.0 + k[348]*y[IDX_EM] + k[545]*y[IDX_H2I] +
        k[743]*y[IDX_NI] + k[830]*y[IDX_OI] + k[1173];
    IJth(jmatrix, 41, 94) = 0.0 + k[91]*y[IDX_HII] + k[1102]*y[IDX_OHI];
    IJth(jmatrix, 41, 95) = 0.0 + k[476]*y[IDX_CHI] + k[572]*y[IDX_H2OI] +
        k[859]*y[IDX_OHI] - k[1209]*y[IDX_HI];
    IJth(jmatrix, 41, 96) = 0.0 + k[94]*y[IDX_HII];
    IJth(jmatrix, 41, 98) = 0.0 + k[92]*y[IDX_HII];
    IJth(jmatrix, 41, 100) = 0.0 + k[93]*y[IDX_HII];
    IJth(jmatrix, 41, 102) = 0.0 + k[98]*y[IDX_HII] + k[292] +
        k[381]*y[IDX_CII] + k[709]*y[IDX_HeII] + k[877]*y[IDX_CI] +
        k[1089]*y[IDX_OI] + k[1188];
    IJth(jmatrix, 41, 103) = 0.0 + k[352]*y[IDX_EM] + k[394]*y[IDX_CI] -
        k[619]*y[IDX_HI] + k[832]*y[IDX_OI] + k[1179];
    IJth(jmatrix, 41, 104) = 0.0 + k[95]*y[IDX_HII] + k[289] +
        k[704]*y[IDX_HeII] + k[1086]*y[IDX_OI] + k[1086]*y[IDX_OI] + k[1181];
    IJth(jmatrix, 41, 105) = 0.0 + k[354]*y[IDX_EM] + k[354]*y[IDX_EM] +
        k[355]*y[IDX_EM] + k[833]*y[IDX_OI];
    IJth(jmatrix, 41, 106) = 0.0 + k[96]*y[IDX_HII] + k[290] +
        k[706]*y[IDX_HeII] + k[1087]*y[IDX_OI] + k[1182];
    IJth(jmatrix, 41, 107) = 0.0 + k[356]*y[IDX_EM];
    IJth(jmatrix, 41, 108) = 0.0 + k[97]*y[IDX_HII] + k[708]*y[IDX_HeII] +
        k[1186] + k[1187];
    IJth(jmatrix, 41, 109) = 0.0 + k[359]*y[IDX_EM] + k[546]*y[IDX_H2I];
    IJth(jmatrix, 41, 110) = 0.0 + k[361]*y[IDX_EM] + k[1248];
    IJth(jmatrix, 41, 111) = 0.0 + k[99]*y[IDX_HII];
    IJth(jmatrix, 41, 112) = 0.0 + k[547]*y[IDX_H2I];
    IJth(jmatrix, 41, 113) = 0.0 + k[364]*y[IDX_EM];
    IJth(jmatrix, 42, 26) = 0.0 - k[78]*y[IDX_HII];
    IJth(jmatrix, 42, 27) = 0.0 + k[1108];
    IJth(jmatrix, 42, 28) = 0.0 - k[75]*y[IDX_HII] - k[491]*y[IDX_HII];
    IJth(jmatrix, 42, 29) = 0.0 + k[1111];
    IJth(jmatrix, 42, 30) = 0.0 - k[76]*y[IDX_HII];
    IJth(jmatrix, 42, 32) = 0.0 - k[492]*y[IDX_HII] - k[493]*y[IDX_HII] -
        k[494]*y[IDX_HII];
    IJth(jmatrix, 42, 33) = 0.0 - k[77]*y[IDX_HII] - k[495]*y[IDX_HII] +
        k[661]*y[IDX_HeII];
    IJth(jmatrix, 42, 36) = 0.0 + k[126]*y[IDX_HI];
    IJth(jmatrix, 42, 38) = 0.0 + k[127]*y[IDX_HI];
    IJth(jmatrix, 42, 39) = 0.0 - k[496]*y[IDX_HII];
    IJth(jmatrix, 42, 40) = 0.0 - k[1216]*y[IDX_HII];
    IJth(jmatrix, 42, 41) = 0.0 + k[126]*y[IDX_CNII] + k[127]*y[IDX_COII] +
        k[128]*y[IDX_H2II] + k[129]*y[IDX_HCNII] + k[130]*y[IDX_HeII] +
        k[131]*y[IDX_OII] + k[236] + k[258] - k[1197]*y[IDX_HII];
    IJth(jmatrix, 42, 42) = 0.0 - k[1]*y[IDX_HNCI] + k[1]*y[IDX_HNCI] -
        k[75]*y[IDX_CH2I] - k[76]*y[IDX_CH3I] - k[77]*y[IDX_CH4I] -
        k[78]*y[IDX_CHI] - k[79]*y[IDX_H2COI] - k[80]*y[IDX_H2OI] -
        k[81]*y[IDX_HCNI] - k[82]*y[IDX_HCOI] - k[83]*y[IDX_MgI] -
        k[84]*y[IDX_NH2I] - k[85]*y[IDX_NH3I] - k[86]*y[IDX_NHI] -
        k[87]*y[IDX_NOI] - k[88]*y[IDX_O2I] - k[89]*y[IDX_OI] - k[90]*y[IDX_OHI]
        - k[91]*y[IDX_SiI] - k[92]*y[IDX_SiC2I] - k[93]*y[IDX_SiC3I] -
        k[94]*y[IDX_SiCI] - k[95]*y[IDX_SiH2I] - k[96]*y[IDX_SiH3I] -
        k[97]*y[IDX_SiH4I] - k[98]*y[IDX_SiHI] - k[99]*y[IDX_SiOI] -
        k[491]*y[IDX_CH2I] - k[492]*y[IDX_CH3OHI] - k[493]*y[IDX_CH3OHI] -
        k[494]*y[IDX_CH3OHI] - k[495]*y[IDX_CH4I] - k[496]*y[IDX_CO2I] -
        k[497]*y[IDX_H2COI] - k[498]*y[IDX_H2COI] - k[499]*y[IDX_H2SiOI] -
        k[500]*y[IDX_HCOI] - k[501]*y[IDX_HCOI] - k[502]*y[IDX_HNCOI] -
        k[503]*y[IDX_HNOI] - k[504]*y[IDX_NO2I] - k[505]*y[IDX_SiH2I] -
        k[506]*y[IDX_SiH3I] - k[507]*y[IDX_SiH4I] - k[508]*y[IDX_SiHI] -
        k[1197]*y[IDX_HI] - k[1198]*y[IDX_HeI] - k[1216]*y[IDX_EM];
    IJth(jmatrix, 42, 43) = 0.0 + k[233] + k[536]*y[IDX_HeII];
    IJth(jmatrix, 42, 44) = 0.0 + k[128]*y[IDX_HI] + k[1134];
    IJth(jmatrix, 42, 46) = 0.0 - k[79]*y[IDX_HII] - k[497]*y[IDX_HII] -
        k[498]*y[IDX_HII];
    IJth(jmatrix, 42, 49) = 0.0 - k[80]*y[IDX_HII] + k[674]*y[IDX_HeII];
    IJth(jmatrix, 42, 51) = 0.0 - k[499]*y[IDX_HII];
    IJth(jmatrix, 42, 52) = 0.0 + k[1146];
    IJth(jmatrix, 42, 55) = 0.0 - k[81]*y[IDX_HII];
    IJth(jmatrix, 42, 56) = 0.0 + k[129]*y[IDX_HI];
    IJth(jmatrix, 42, 58) = 0.0 - k[82]*y[IDX_HII] - k[500]*y[IDX_HII] -
        k[501]*y[IDX_HII];
    IJth(jmatrix, 42, 61) = 0.0 - k[1198]*y[IDX_HII];
    IJth(jmatrix, 42, 62) = 0.0 + k[130]*y[IDX_HI] + k[536]*y[IDX_H2I] +
        k[661]*y[IDX_CH4I] + k[674]*y[IDX_H2OI] + k[687]*y[IDX_HNOI];
    IJth(jmatrix, 42, 64) = 0.0 - k[1]*y[IDX_HII] + k[1]*y[IDX_HII];
    IJth(jmatrix, 42, 65) = 0.0 - k[502]*y[IDX_HII];
    IJth(jmatrix, 42, 66) = 0.0 - k[503]*y[IDX_HII] + k[687]*y[IDX_HeII];
    IJth(jmatrix, 42, 69) = 0.0 - k[83]*y[IDX_HII];
    IJth(jmatrix, 42, 76) = 0.0 - k[86]*y[IDX_HII];
    IJth(jmatrix, 42, 77) = 0.0 + k[1156];
    IJth(jmatrix, 42, 78) = 0.0 - k[84]*y[IDX_HII];
    IJth(jmatrix, 42, 80) = 0.0 - k[85]*y[IDX_HII];
    IJth(jmatrix, 42, 82) = 0.0 - k[87]*y[IDX_HII];
    IJth(jmatrix, 42, 84) = 0.0 - k[504]*y[IDX_HII];
    IJth(jmatrix, 42, 85) = 0.0 - k[89]*y[IDX_HII];
    IJth(jmatrix, 42, 86) = 0.0 + k[131]*y[IDX_HI];
    IJth(jmatrix, 42, 87) = 0.0 - k[88]*y[IDX_HII];
    IJth(jmatrix, 42, 92) = 0.0 - k[90]*y[IDX_HII];
    IJth(jmatrix, 42, 94) = 0.0 - k[91]*y[IDX_HII];
    IJth(jmatrix, 42, 96) = 0.0 - k[94]*y[IDX_HII];
    IJth(jmatrix, 42, 98) = 0.0 - k[92]*y[IDX_HII];
    IJth(jmatrix, 42, 100) = 0.0 - k[93]*y[IDX_HII];
    IJth(jmatrix, 42, 102) = 0.0 - k[98]*y[IDX_HII] - k[508]*y[IDX_HII];
    IJth(jmatrix, 42, 104) = 0.0 - k[95]*y[IDX_HII] - k[505]*y[IDX_HII];
    IJth(jmatrix, 42, 106) = 0.0 - k[96]*y[IDX_HII] - k[506]*y[IDX_HII];
    IJth(jmatrix, 42, 108) = 0.0 - k[97]*y[IDX_HII] - k[507]*y[IDX_HII];
    IJth(jmatrix, 42, 111) = 0.0 - k[99]*y[IDX_HII];
    IJth(jmatrix, 43, 24) = 0.0 + k[384]*y[IDX_H3OII] + k[576]*y[IDX_H3II] -
        k[955]*y[IDX_H2I] - k[1200]*y[IDX_H2I];
    IJth(jmatrix, 43, 25) = 0.0 + k[374]*y[IDX_NH3I] + k[380]*y[IDX_SiH2I] -
        k[528]*y[IDX_H2I] - k[1199]*y[IDX_H2I];
    IJth(jmatrix, 43, 26) = 0.0 - k[2]*y[IDX_H2I] + k[2]*y[IDX_H2I] +
        k[102]*y[IDX_H2II] + k[580]*y[IDX_H3II] - k[958]*y[IDX_H2I] +
        k[970]*y[IDX_HI] - k[1201]*y[IDX_H2I];
    IJth(jmatrix, 43, 27) = 0.0 + k[404]*y[IDX_H2OI] + k[409]*y[IDX_NH2I] +
        k[410]*y[IDX_NHI] + k[415]*y[IDX_OHI] - k[529]*y[IDX_H2I] +
        k[614]*y[IDX_HI];
    IJth(jmatrix, 43, 28) = 0.0 + k[100]*y[IDX_H2II] + k[491]*y[IDX_HII] +
        k[577]*y[IDX_H3II] + k[653]*y[IDX_HeII] + k[889]*y[IDX_O2I] +
        k[894]*y[IDX_OI] - k[956]*y[IDX_H2I] + k[967]*y[IDX_HI];
    IJth(jmatrix, 43, 29) = 0.0 + k[295]*y[IDX_EM] - k[530]*y[IDX_H2I] +
        k[615]*y[IDX_HI] + k[1109];
    IJth(jmatrix, 43, 30) = 0.0 + k[246] + k[578]*y[IDX_H3II] +
        k[655]*y[IDX_HeII] + k[915]*y[IDX_OI] + k[918]*y[IDX_OHI] -
        k[957]*y[IDX_H2I] + k[968]*y[IDX_HI] + k[1009]*y[IDX_NI] + k[1118];
    IJth(jmatrix, 43, 31) = 0.0 + k[299]*y[IDX_EM] + k[444]*y[IDX_OI] +
        k[445]*y[IDX_OHI] + k[616]*y[IDX_HI] + k[794]*y[IDX_NHI] + k[1114];
    IJth(jmatrix, 43, 32) = 0.0 + k[247] + k[493]*y[IDX_HII] +
        k[494]*y[IDX_HII] + k[494]*y[IDX_HII] + k[579]*y[IDX_H3II] + k[1119];
    IJth(jmatrix, 43, 33) = 0.0 + k[101]*y[IDX_H2II] + k[249] +
        k[455]*y[IDX_N2II] + k[495]*y[IDX_HII] + k[511]*y[IDX_H2II] +
        k[658]*y[IDX_HeII] + k[659]*y[IDX_HeII] + k[717]*y[IDX_NII] +
        k[969]*y[IDX_HI] + k[1124] + k[1127];
    IJth(jmatrix, 43, 34) = 0.0 + k[617]*y[IDX_HI] + k[1122];
    IJth(jmatrix, 43, 35) = 0.0 + k[103]*y[IDX_H2II] + k[581]*y[IDX_H3II] -
        k[959]*y[IDX_H2I];
    IJth(jmatrix, 43, 36) = 0.0 - k[531]*y[IDX_H2I];
    IJth(jmatrix, 43, 37) = 0.0 + k[104]*y[IDX_H2II] + k[583]*y[IDX_H3II] +
        k[584]*y[IDX_H3II];
    IJth(jmatrix, 43, 38) = 0.0 - k[532]*y[IDX_H2I] - k[533]*y[IDX_H2I];
    IJth(jmatrix, 43, 39) = 0.0 + k[582]*y[IDX_H3II];
    IJth(jmatrix, 43, 40) = 0.0 - k[8]*y[IDX_H2I] + k[295]*y[IDX_CH2II] +
        k[299]*y[IDX_CH3II] + k[307]*y[IDX_H2COII] + k[311]*y[IDX_H2NOII] +
        k[312]*y[IDX_H2OII] + k[315]*y[IDX_H3II] + k[319]*y[IDX_H3COII] +
        k[323]*y[IDX_H3OII] + k[324]*y[IDX_H3OII] + k[353]*y[IDX_SiH2II] +
        k[357]*y[IDX_SiH3II] + k[358]*y[IDX_SiH4II] + k[360]*y[IDX_SiH5II];
    IJth(jmatrix, 43, 41) = 0.0 - k[10]*y[IDX_H2I] + k[128]*y[IDX_H2II] +
        k[614]*y[IDX_CHII] + k[615]*y[IDX_CH2II] + k[616]*y[IDX_CH3II] +
        k[617]*y[IDX_CH4II] + k[619]*y[IDX_SiHII] + k[967]*y[IDX_CH2I] +
        k[968]*y[IDX_CH3I] + k[969]*y[IDX_CH4I] + k[970]*y[IDX_CHI] +
        k[973]*y[IDX_H2CNI] + k[974]*y[IDX_H2COI] + k[975]*y[IDX_H2OI] +
        k[976]*y[IDX_HCNI] + k[977]*y[IDX_HCOI] + k[981]*y[IDX_HNOI] +
        k[983]*y[IDX_NH2I] + k[984]*y[IDX_NH3I] + k[985]*y[IDX_NHI] +
        k[991]*y[IDX_O2HI] + k[996]*y[IDX_OHI];
    IJth(jmatrix, 43, 42) = 0.0 + k[491]*y[IDX_CH2I] + k[493]*y[IDX_CH3OHI]
        + k[494]*y[IDX_CH3OHI] + k[494]*y[IDX_CH3OHI] + k[495]*y[IDX_CH4I] +
        k[497]*y[IDX_H2COI] + k[498]*y[IDX_H2COI] + k[499]*y[IDX_H2SiOI] +
        k[500]*y[IDX_HCOI] + k[503]*y[IDX_HNOI] + k[505]*y[IDX_SiH2I] +
        k[506]*y[IDX_SiH3I] + k[507]*y[IDX_SiH4I] + k[508]*y[IDX_SiHI];
    IJth(jmatrix, 43, 43) = 0.0 - k[2]*y[IDX_CHI] + k[2]*y[IDX_CHI] -
        k[3]*y[IDX_H2I] - k[3]*y[IDX_H2I] - k[3]*y[IDX_H2I] - k[3]*y[IDX_H2I] +
        k[3]*y[IDX_H2I] + k[3]*y[IDX_H2I] - k[4]*y[IDX_H2OI] + k[4]*y[IDX_H2OI]
        - k[5]*y[IDX_HOCII] + k[5]*y[IDX_HOCII] - k[6]*y[IDX_O2I] +
        k[6]*y[IDX_O2I] - k[7]*y[IDX_OHI] + k[7]*y[IDX_OHI] - k[8]*y[IDX_EM] -
        k[10]*y[IDX_HI] - k[115]*y[IDX_HeII] - k[233] - k[234] - k[235] -
        k[516]*y[IDX_H2II] - k[528]*y[IDX_CII] - k[529]*y[IDX_CHII] -
        k[530]*y[IDX_CH2II] - k[531]*y[IDX_CNII] - k[532]*y[IDX_COII] -
        k[533]*y[IDX_COII] - k[534]*y[IDX_H2OII] - k[535]*y[IDX_HCNII] -
        k[536]*y[IDX_HeII] - k[537]*y[IDX_HeHII] - k[538]*y[IDX_NII] -
        k[539]*y[IDX_N2II] - k[540]*y[IDX_NHII] - k[541]*y[IDX_NHII] -
        k[542]*y[IDX_NH2II] - k[543]*y[IDX_OII] - k[544]*y[IDX_O2HII] -
        k[545]*y[IDX_OHII] - k[546]*y[IDX_SiH4II] - k[547]*y[IDX_SiOII] -
        k[955]*y[IDX_CI] - k[956]*y[IDX_CH2I] - k[957]*y[IDX_CH3I] -
        k[958]*y[IDX_CHI] - k[959]*y[IDX_CNI] - k[960]*y[IDX_NI] -
        k[961]*y[IDX_NH2I] - k[962]*y[IDX_NHI] - k[963]*y[IDX_O2I] -
        k[964]*y[IDX_O2I] - k[965]*y[IDX_OI] - k[966]*y[IDX_OHI] -
        k[1199]*y[IDX_CII] - k[1200]*y[IDX_CI] - k[1201]*y[IDX_CHI] -
        k[1202]*y[IDX_SiII] - k[1203]*y[IDX_SiHII] - k[1204]*y[IDX_SiH3II];
    IJth(jmatrix, 43, 44) = 0.0 + k[100]*y[IDX_CH2I] + k[101]*y[IDX_CH4I] +
        k[102]*y[IDX_CHI] + k[103]*y[IDX_CNI] + k[104]*y[IDX_COI] +
        k[105]*y[IDX_H2COI] + k[106]*y[IDX_H2OI] + k[107]*y[IDX_HCNI] +
        k[108]*y[IDX_HCOI] + k[109]*y[IDX_NH2I] + k[110]*y[IDX_NH3I] +
        k[111]*y[IDX_NHI] + k[112]*y[IDX_NOI] + k[113]*y[IDX_O2I] +
        k[114]*y[IDX_OHI] + k[128]*y[IDX_HI] + k[511]*y[IDX_CH4I] -
        k[516]*y[IDX_H2I] + k[517]*y[IDX_H2COI];
    IJth(jmatrix, 43, 45) = 0.0 + k[973]*y[IDX_HI] + k[1060]*y[IDX_OI];
    IJth(jmatrix, 43, 46) = 0.0 + k[105]*y[IDX_H2II] + k[255] +
        k[497]*y[IDX_HII] + k[498]*y[IDX_HII] + k[517]*y[IDX_H2II] +
        k[585]*y[IDX_H3II] + k[670]*y[IDX_HeII] + k[974]*y[IDX_HI] + k[1136];
    IJth(jmatrix, 43, 47) = 0.0 + k[307]*y[IDX_EM];
    IJth(jmatrix, 43, 48) = 0.0 + k[311]*y[IDX_EM];
    IJth(jmatrix, 43, 49) = 0.0 - k[4]*y[IDX_H2I] + k[4]*y[IDX_H2I] +
        k[106]*y[IDX_H2II] + k[404]*y[IDX_CHII] + k[586]*y[IDX_H3II] +
        k[755]*y[IDX_NHII] + k[975]*y[IDX_HI];
    IJth(jmatrix, 43, 50) = 0.0 + k[312]*y[IDX_EM] - k[534]*y[IDX_H2I] +
        k[739]*y[IDX_NI] + k[823]*y[IDX_OI];
    IJth(jmatrix, 43, 51) = 0.0 + k[257] + k[499]*y[IDX_HII] + k[1143];
    IJth(jmatrix, 43, 52) = 0.0 + k[315]*y[IDX_EM] + k[576]*y[IDX_CI] +
        k[577]*y[IDX_CH2I] + k[578]*y[IDX_CH3I] + k[579]*y[IDX_CH3OHI] +
        k[580]*y[IDX_CHI] + k[581]*y[IDX_CNI] + k[582]*y[IDX_CO2I] +
        k[583]*y[IDX_COI] + k[584]*y[IDX_COI] + k[585]*y[IDX_H2COI] +
        k[586]*y[IDX_H2OI] + k[587]*y[IDX_HCNI] + k[588]*y[IDX_HCOI] +
        k[589]*y[IDX_HNCI] + k[590]*y[IDX_HNOI] + k[591]*y[IDX_MgI] +
        k[592]*y[IDX_N2I] + k[593]*y[IDX_NH2I] + k[594]*y[IDX_NHI] +
        k[595]*y[IDX_NO2I] + k[596]*y[IDX_NOI] + k[597]*y[IDX_O2I] +
        k[599]*y[IDX_OI] + k[600]*y[IDX_OHI] + k[601]*y[IDX_SiI] +
        k[602]*y[IDX_SiH2I] + k[603]*y[IDX_SiH3I] + k[604]*y[IDX_SiH4I] +
        k[605]*y[IDX_SiHI] + k[606]*y[IDX_SiOI] + k[1146];
    IJth(jmatrix, 43, 53) = 0.0 + k[319]*y[IDX_EM];
    IJth(jmatrix, 43, 54) = 0.0 + k[323]*y[IDX_EM] + k[324]*y[IDX_EM] +
        k[384]*y[IDX_CI];
    IJth(jmatrix, 43, 55) = 0.0 + k[107]*y[IDX_H2II] + k[587]*y[IDX_H3II] +
        k[976]*y[IDX_HI];
    IJth(jmatrix, 43, 56) = 0.0 - k[535]*y[IDX_H2I];
    IJth(jmatrix, 43, 58) = 0.0 + k[108]*y[IDX_H2II] + k[500]*y[IDX_HII] +
        k[588]*y[IDX_H3II] + k[977]*y[IDX_HI] + k[997]*y[IDX_HCOI] +
        k[997]*y[IDX_HCOI];
    IJth(jmatrix, 43, 62) = 0.0 - k[115]*y[IDX_H2I] - k[536]*y[IDX_H2I] +
        k[653]*y[IDX_CH2I] + k[655]*y[IDX_CH3I] + k[658]*y[IDX_CH4I] +
        k[659]*y[IDX_CH4I] + k[670]*y[IDX_H2COI] + k[689]*y[IDX_NH2I] +
        k[691]*y[IDX_NH3I] + k[703]*y[IDX_SiH2I] + k[705]*y[IDX_SiH3I] +
        k[707]*y[IDX_SiH4I] + k[707]*y[IDX_SiH4I] + k[708]*y[IDX_SiH4I];
    IJth(jmatrix, 43, 63) = 0.0 - k[537]*y[IDX_H2I];
    IJth(jmatrix, 43, 64) = 0.0 + k[589]*y[IDX_H3II];
    IJth(jmatrix, 43, 66) = 0.0 + k[503]*y[IDX_HII] + k[590]*y[IDX_H3II] +
        k[981]*y[IDX_HI];
    IJth(jmatrix, 43, 68) = 0.0 - k[5]*y[IDX_H2I] + k[5]*y[IDX_H2I];
    IJth(jmatrix, 43, 69) = 0.0 + k[591]*y[IDX_H3II];
    IJth(jmatrix, 43, 71) = 0.0 + k[739]*y[IDX_H2OII] - k[960]*y[IDX_H2I] +
        k[1009]*y[IDX_CH3I];
    IJth(jmatrix, 43, 72) = 0.0 - k[538]*y[IDX_H2I] + k[717]*y[IDX_CH4I] +
        k[724]*y[IDX_NH3I];
    IJth(jmatrix, 43, 73) = 0.0 + k[592]*y[IDX_H3II];
    IJth(jmatrix, 43, 74) = 0.0 + k[455]*y[IDX_CH4I] - k[539]*y[IDX_H2I];
    IJth(jmatrix, 43, 76) = 0.0 + k[111]*y[IDX_H2II] + k[410]*y[IDX_CHII] +
        k[594]*y[IDX_H3II] + k[794]*y[IDX_CH3II] - k[962]*y[IDX_H2I] +
        k[985]*y[IDX_HI] + k[1038]*y[IDX_NHI] + k[1038]*y[IDX_NHI];
    IJth(jmatrix, 43, 77) = 0.0 - k[540]*y[IDX_H2I] - k[541]*y[IDX_H2I] +
        k[755]*y[IDX_H2OI];
    IJth(jmatrix, 43, 78) = 0.0 + k[109]*y[IDX_H2II] + k[409]*y[IDX_CHII] +
        k[593]*y[IDX_H3II] + k[689]*y[IDX_HeII] - k[961]*y[IDX_H2I] +
        k[983]*y[IDX_HI];
    IJth(jmatrix, 43, 79) = 0.0 - k[542]*y[IDX_H2I];
    IJth(jmatrix, 43, 80) = 0.0 + k[110]*y[IDX_H2II] + k[273] +
        k[374]*y[IDX_CII] + k[691]*y[IDX_HeII] + k[724]*y[IDX_NII] +
        k[984]*y[IDX_HI] + k[1161];
    IJth(jmatrix, 43, 81) = 0.0 + k[828]*y[IDX_OI];
    IJth(jmatrix, 43, 82) = 0.0 + k[112]*y[IDX_H2II] + k[596]*y[IDX_H3II];
    IJth(jmatrix, 43, 84) = 0.0 + k[595]*y[IDX_H3II];
    IJth(jmatrix, 43, 85) = 0.0 + k[444]*y[IDX_CH3II] + k[599]*y[IDX_H3II] +
        k[823]*y[IDX_H2OII] + k[828]*y[IDX_NH3II] + k[834]*y[IDX_SiH3II] +
        k[894]*y[IDX_CH2I] + k[915]*y[IDX_CH3I] - k[965]*y[IDX_H2I] +
        k[1060]*y[IDX_H2CNI] + k[1085]*y[IDX_SiH2I];
    IJth(jmatrix, 43, 86) = 0.0 - k[543]*y[IDX_H2I];
    IJth(jmatrix, 43, 87) = 0.0 - k[6]*y[IDX_H2I] + k[6]*y[IDX_H2I] +
        k[113]*y[IDX_H2II] + k[597]*y[IDX_H3II] + k[889]*y[IDX_CH2I] -
        k[963]*y[IDX_H2I] - k[964]*y[IDX_H2I];
    IJth(jmatrix, 43, 89) = 0.0 + k[991]*y[IDX_HI];
    IJth(jmatrix, 43, 90) = 0.0 - k[544]*y[IDX_H2I];
    IJth(jmatrix, 43, 92) = 0.0 - k[7]*y[IDX_H2I] + k[7]*y[IDX_H2I] +
        k[114]*y[IDX_H2II] + k[415]*y[IDX_CHII] + k[445]*y[IDX_CH3II] +
        k[600]*y[IDX_H3II] + k[918]*y[IDX_CH3I] - k[966]*y[IDX_H2I] +
        k[996]*y[IDX_HI];
    IJth(jmatrix, 43, 93) = 0.0 - k[545]*y[IDX_H2I];
    IJth(jmatrix, 43, 94) = 0.0 + k[601]*y[IDX_H3II];
    IJth(jmatrix, 43, 95) = 0.0 - k[1202]*y[IDX_H2I];
    IJth(jmatrix, 43, 102) = 0.0 + k[508]*y[IDX_HII] + k[605]*y[IDX_H3II];
    IJth(jmatrix, 43, 103) = 0.0 + k[619]*y[IDX_HI] - k[1203]*y[IDX_H2I];
    IJth(jmatrix, 43, 104) = 0.0 + k[380]*y[IDX_CII] + k[505]*y[IDX_HII] +
        k[602]*y[IDX_H3II] + k[703]*y[IDX_HeII] + k[1085]*y[IDX_OI];
    IJth(jmatrix, 43, 105) = 0.0 + k[353]*y[IDX_EM];
    IJth(jmatrix, 43, 106) = 0.0 + k[506]*y[IDX_HII] + k[603]*y[IDX_H3II] +
        k[705]*y[IDX_HeII] + k[1184];
    IJth(jmatrix, 43, 107) = 0.0 + k[357]*y[IDX_EM] + k[834]*y[IDX_OI] -
        k[1204]*y[IDX_H2I];
    IJth(jmatrix, 43, 108) = 0.0 + k[291] + k[507]*y[IDX_HII] +
        k[604]*y[IDX_H3II] + k[707]*y[IDX_HeII] + k[707]*y[IDX_HeII] +
        k[708]*y[IDX_HeII] + k[1185] + k[1187];
    IJth(jmatrix, 43, 109) = 0.0 + k[358]*y[IDX_EM] - k[546]*y[IDX_H2I];
    IJth(jmatrix, 43, 110) = 0.0 + k[360]*y[IDX_EM];
    IJth(jmatrix, 43, 111) = 0.0 + k[606]*y[IDX_H3II];
    IJth(jmatrix, 43, 112) = 0.0 - k[547]*y[IDX_H2I];
    IJth(jmatrix, 44, 24) = 0.0 - k[509]*y[IDX_H2II];
    IJth(jmatrix, 44, 26) = 0.0 - k[102]*y[IDX_H2II] - k[512]*y[IDX_H2II];
    IJth(jmatrix, 44, 28) = 0.0 - k[100]*y[IDX_H2II] - k[510]*y[IDX_H2II];
    IJth(jmatrix, 44, 33) = 0.0 - k[101]*y[IDX_H2II] - k[511]*y[IDX_H2II];
    IJth(jmatrix, 44, 35) = 0.0 - k[103]*y[IDX_H2II] - k[513]*y[IDX_H2II];
    IJth(jmatrix, 44, 37) = 0.0 - k[104]*y[IDX_H2II] - k[515]*y[IDX_H2II];
    IJth(jmatrix, 44, 39) = 0.0 - k[514]*y[IDX_H2II];
    IJth(jmatrix, 44, 40) = 0.0 - k[305]*y[IDX_H2II];
    IJth(jmatrix, 44, 41) = 0.0 - k[128]*y[IDX_H2II] + k[618]*y[IDX_HeHII] +
        k[1197]*y[IDX_HII];
    IJth(jmatrix, 44, 42) = 0.0 + k[501]*y[IDX_HCOI] + k[1197]*y[IDX_HI];
    IJth(jmatrix, 44, 43) = 0.0 + k[115]*y[IDX_HeII] + k[234] -
        k[516]*y[IDX_H2II];
    IJth(jmatrix, 44, 44) = 0.0 - k[100]*y[IDX_CH2I] - k[101]*y[IDX_CH4I] -
        k[102]*y[IDX_CHI] - k[103]*y[IDX_CNI] - k[104]*y[IDX_COI] -
        k[105]*y[IDX_H2COI] - k[106]*y[IDX_H2OI] - k[107]*y[IDX_HCNI] -
        k[108]*y[IDX_HCOI] - k[109]*y[IDX_NH2I] - k[110]*y[IDX_NH3I] -
        k[111]*y[IDX_NHI] - k[112]*y[IDX_NOI] - k[113]*y[IDX_O2I] -
        k[114]*y[IDX_OHI] - k[128]*y[IDX_HI] - k[305]*y[IDX_EM] -
        k[509]*y[IDX_CI] - k[510]*y[IDX_CH2I] - k[511]*y[IDX_CH4I] -
        k[512]*y[IDX_CHI] - k[513]*y[IDX_CNI] - k[514]*y[IDX_CO2I] -
        k[515]*y[IDX_COI] - k[516]*y[IDX_H2I] - k[517]*y[IDX_H2COI] -
        k[518]*y[IDX_H2OI] - k[519]*y[IDX_HCOI] - k[520]*y[IDX_HeI] -
        k[521]*y[IDX_N2I] - k[522]*y[IDX_NI] - k[523]*y[IDX_NHI] -
        k[524]*y[IDX_NOI] - k[525]*y[IDX_O2I] - k[526]*y[IDX_OI] -
        k[527]*y[IDX_OHI] - k[1134];
    IJth(jmatrix, 44, 46) = 0.0 - k[105]*y[IDX_H2II] - k[517]*y[IDX_H2II];
    IJth(jmatrix, 44, 49) = 0.0 - k[106]*y[IDX_H2II] - k[518]*y[IDX_H2II];
    IJth(jmatrix, 44, 52) = 0.0 + k[1145];
    IJth(jmatrix, 44, 55) = 0.0 - k[107]*y[IDX_H2II];
    IJth(jmatrix, 44, 58) = 0.0 - k[108]*y[IDX_H2II] + k[501]*y[IDX_HII] -
        k[519]*y[IDX_H2II];
    IJth(jmatrix, 44, 61) = 0.0 - k[520]*y[IDX_H2II];
    IJth(jmatrix, 44, 62) = 0.0 + k[115]*y[IDX_H2I];
    IJth(jmatrix, 44, 63) = 0.0 + k[618]*y[IDX_HI];
    IJth(jmatrix, 44, 71) = 0.0 - k[522]*y[IDX_H2II];
    IJth(jmatrix, 44, 73) = 0.0 - k[521]*y[IDX_H2II];
    IJth(jmatrix, 44, 76) = 0.0 - k[111]*y[IDX_H2II] - k[523]*y[IDX_H2II];
    IJth(jmatrix, 44, 78) = 0.0 - k[109]*y[IDX_H2II];
    IJth(jmatrix, 44, 80) = 0.0 - k[110]*y[IDX_H2II];
    IJth(jmatrix, 44, 82) = 0.0 - k[112]*y[IDX_H2II] - k[524]*y[IDX_H2II];
    IJth(jmatrix, 44, 85) = 0.0 - k[526]*y[IDX_H2II];
    IJth(jmatrix, 44, 87) = 0.0 - k[113]*y[IDX_H2II] - k[525]*y[IDX_H2II];
    IJth(jmatrix, 44, 92) = 0.0 - k[114]*y[IDX_H2II] - k[527]*y[IDX_H2II];
    IJth(jmatrix, 45, 4) = 0.0 + k[1339] + k[1340] + k[1341] + k[1342];
    IJth(jmatrix, 45, 30) = 0.0 + k[1008]*y[IDX_NI];
    IJth(jmatrix, 45, 41) = 0.0 - k[973]*y[IDX_H2CNI];
    IJth(jmatrix, 45, 45) = 0.0 - k[254] - k[973]*y[IDX_HI] -
        k[1013]*y[IDX_NI] - k[1060]*y[IDX_OI] - k[1135] - k[1298];
    IJth(jmatrix, 45, 71) = 0.0 + k[1008]*y[IDX_CH3I] -
        k[1013]*y[IDX_H2CNI];
    IJth(jmatrix, 45, 85) = 0.0 - k[1060]*y[IDX_H2CNI];
    IJth(jmatrix, 46, 5) = 0.0 + k[1347] + k[1348] + k[1349] + k[1350];
    IJth(jmatrix, 46, 25) = 0.0 - k[16]*y[IDX_H2COI] - k[368]*y[IDX_H2COI] -
        k[369]*y[IDX_H2COI];
    IJth(jmatrix, 46, 26) = 0.0 + k[55]*y[IDX_H2COII] + k[461]*y[IDX_H3COII]
        - k[924]*y[IDX_H2COI];
    IJth(jmatrix, 46, 27) = 0.0 + k[396]*y[IDX_CH3OHI] - k[399]*y[IDX_H2COI]
        - k[400]*y[IDX_H2COI] - k[401]*y[IDX_H2COI];
    IJth(jmatrix, 46, 28) = 0.0 + k[39]*y[IDX_H2COII] + k[438]*y[IDX_SiOII]
        - k[881]*y[IDX_H2COI] + k[885]*y[IDX_NO2I] + k[886]*y[IDX_NOI] +
        k[892]*y[IDX_O2I] + k[898]*y[IDX_OHI];
    IJth(jmatrix, 46, 29) = 0.0 - k[417]*y[IDX_H2COI];
    IJth(jmatrix, 46, 30) = 0.0 - k[903]*y[IDX_H2COI] + k[909]*y[IDX_NO2I] +
        k[911]*y[IDX_O2I] + k[916]*y[IDX_OI] + k[918]*y[IDX_OHI];
    IJth(jmatrix, 46, 31) = 0.0 - k[440]*y[IDX_H2COI];
    IJth(jmatrix, 46, 32) = 0.0 + k[247] + k[396]*y[IDX_CHII] + k[1119];
    IJth(jmatrix, 46, 34) = 0.0 - k[49]*y[IDX_H2COI] - k[449]*y[IDX_H2COI];
    IJth(jmatrix, 46, 35) = 0.0 - k[942]*y[IDX_H2COI];
    IJth(jmatrix, 46, 36) = 0.0 - k[64]*y[IDX_H2COI] - k[479]*y[IDX_H2COI];
    IJth(jmatrix, 46, 38) = 0.0 - k[70]*y[IDX_H2COI] - k[484]*y[IDX_H2COI];
    IJth(jmatrix, 46, 40) = 0.0 + k[320]*y[IDX_H3COII] +
        k[1217]*y[IDX_H2COII];
    IJth(jmatrix, 46, 41) = 0.0 - k[974]*y[IDX_H2COI];
    IJth(jmatrix, 46, 42) = 0.0 - k[79]*y[IDX_H2COI] - k[497]*y[IDX_H2COI] -
        k[498]*y[IDX_H2COI];
    IJth(jmatrix, 46, 44) = 0.0 - k[105]*y[IDX_H2COI] - k[517]*y[IDX_H2COI];
    IJth(jmatrix, 46, 46) = 0.0 - k[16]*y[IDX_CII] - k[49]*y[IDX_CH4II] -
        k[64]*y[IDX_CNII] - k[70]*y[IDX_COII] - k[79]*y[IDX_HII] -
        k[105]*y[IDX_H2II] - k[116]*y[IDX_O2II] - k[117]*y[IDX_H2OII] -
        k[142]*y[IDX_HeII] - k[159]*y[IDX_NII] - k[170]*y[IDX_N2II] -
        k[175]*y[IDX_NHII] - k[209]*y[IDX_OII] - k[219]*y[IDX_OHII] - k[255] -
        k[368]*y[IDX_CII] - k[369]*y[IDX_CII] - k[399]*y[IDX_CHII] -
        k[400]*y[IDX_CHII] - k[401]*y[IDX_CHII] - k[417]*y[IDX_CH2II] -
        k[440]*y[IDX_CH3II] - k[449]*y[IDX_CH4II] - k[479]*y[IDX_CNII] -
        k[484]*y[IDX_COII] - k[497]*y[IDX_HII] - k[498]*y[IDX_HII] -
        k[517]*y[IDX_H2II] - k[548]*y[IDX_H2COII] - k[550]*y[IDX_HNOII] -
        k[551]*y[IDX_O2II] - k[552]*y[IDX_O2HII] - k[554]*y[IDX_H2OII] -
        k[585]*y[IDX_H3II] - k[607]*y[IDX_H3OII] - k[622]*y[IDX_HCNII] -
        k[633]*y[IDX_HCNHII] - k[634]*y[IDX_HCNHII] - k[635]*y[IDX_HCOII] -
        k[670]*y[IDX_HeII] - k[671]*y[IDX_HeII] - k[672]*y[IDX_HeII] -
        k[721]*y[IDX_NII] - k[722]*y[IDX_NII] - k[730]*y[IDX_N2II] -
        k[735]*y[IDX_N2HII] - k[752]*y[IDX_NHII] - k[753]*y[IDX_NHII] -
        k[769]*y[IDX_NH2II] - k[770]*y[IDX_NH2II] - k[813]*y[IDX_OII] -
        k[839]*y[IDX_OHII] - k[881]*y[IDX_CH2I] - k[903]*y[IDX_CH3I] -
        k[924]*y[IDX_CHI] - k[942]*y[IDX_CNI] - k[974]*y[IDX_HI] -
        k[1061]*y[IDX_OI] - k[1093]*y[IDX_OHI] - k[1136] - k[1137] - k[1138] -
        k[1139] - k[1252];
    IJth(jmatrix, 46, 47) = 0.0 + k[39]*y[IDX_CH2I] + k[55]*y[IDX_CHI] +
        k[136]*y[IDX_HCOI] + k[148]*y[IDX_MgI] + k[194]*y[IDX_NH3I] +
        k[203]*y[IDX_NOI] + k[228]*y[IDX_SiI] - k[548]*y[IDX_H2COI] +
        k[1217]*y[IDX_EM];
    IJth(jmatrix, 46, 49) = 0.0 + k[564]*y[IDX_H3COII];
    IJth(jmatrix, 46, 50) = 0.0 - k[117]*y[IDX_H2COI] - k[554]*y[IDX_H2COI];
    IJth(jmatrix, 46, 52) = 0.0 - k[585]*y[IDX_H2COI];
    IJth(jmatrix, 46, 53) = 0.0 + k[320]*y[IDX_EM] + k[461]*y[IDX_CHI] +
        k[564]*y[IDX_H2OI] + k[628]*y[IDX_HCNI] + k[647]*y[IDX_HNCI] +
        k[782]*y[IDX_NH2I];
    IJth(jmatrix, 46, 54) = 0.0 - k[607]*y[IDX_H2COI];
    IJth(jmatrix, 46, 55) = 0.0 + k[628]*y[IDX_H3COII];
    IJth(jmatrix, 46, 56) = 0.0 - k[622]*y[IDX_H2COI];
    IJth(jmatrix, 46, 57) = 0.0 - k[633]*y[IDX_H2COI] - k[634]*y[IDX_H2COI];
    IJth(jmatrix, 46, 58) = 0.0 + k[136]*y[IDX_H2COII] + k[998]*y[IDX_HCOI]
        + k[998]*y[IDX_HCOI] + k[999]*y[IDX_HNOI] + k[1003]*y[IDX_O2HI];
    IJth(jmatrix, 46, 59) = 0.0 - k[635]*y[IDX_H2COI];
    IJth(jmatrix, 46, 62) = 0.0 - k[142]*y[IDX_H2COI] - k[670]*y[IDX_H2COI]
        - k[671]*y[IDX_H2COI] - k[672]*y[IDX_H2COI];
    IJth(jmatrix, 46, 64) = 0.0 + k[647]*y[IDX_H3COII];
    IJth(jmatrix, 46, 66) = 0.0 + k[999]*y[IDX_HCOI];
    IJth(jmatrix, 46, 67) = 0.0 - k[550]*y[IDX_H2COI];
    IJth(jmatrix, 46, 69) = 0.0 + k[148]*y[IDX_H2COII];
    IJth(jmatrix, 46, 72) = 0.0 - k[159]*y[IDX_H2COI] - k[721]*y[IDX_H2COI]
        - k[722]*y[IDX_H2COI];
    IJth(jmatrix, 46, 74) = 0.0 - k[170]*y[IDX_H2COI] - k[730]*y[IDX_H2COI];
    IJth(jmatrix, 46, 75) = 0.0 - k[735]*y[IDX_H2COI];
    IJth(jmatrix, 46, 77) = 0.0 - k[175]*y[IDX_H2COI] - k[752]*y[IDX_H2COI]
        - k[753]*y[IDX_H2COI];
    IJth(jmatrix, 46, 78) = 0.0 + k[782]*y[IDX_H3COII];
    IJth(jmatrix, 46, 79) = 0.0 - k[769]*y[IDX_H2COI] - k[770]*y[IDX_H2COI];
    IJth(jmatrix, 46, 80) = 0.0 + k[194]*y[IDX_H2COII];
    IJth(jmatrix, 46, 82) = 0.0 + k[203]*y[IDX_H2COII] + k[886]*y[IDX_CH2I];
    IJth(jmatrix, 46, 84) = 0.0 + k[885]*y[IDX_CH2I] + k[909]*y[IDX_CH3I];
    IJth(jmatrix, 46, 85) = 0.0 + k[916]*y[IDX_CH3I] - k[1061]*y[IDX_H2COI];
    IJth(jmatrix, 46, 86) = 0.0 - k[209]*y[IDX_H2COI] - k[813]*y[IDX_H2COI];
    IJth(jmatrix, 46, 87) = 0.0 + k[892]*y[IDX_CH2I] + k[911]*y[IDX_CH3I];
    IJth(jmatrix, 46, 88) = 0.0 - k[116]*y[IDX_H2COI] - k[551]*y[IDX_H2COI];
    IJth(jmatrix, 46, 89) = 0.0 + k[1003]*y[IDX_HCOI];
    IJth(jmatrix, 46, 90) = 0.0 - k[552]*y[IDX_H2COI];
    IJth(jmatrix, 46, 92) = 0.0 + k[898]*y[IDX_CH2I] + k[918]*y[IDX_CH3I] -
        k[1093]*y[IDX_H2COI];
    IJth(jmatrix, 46, 93) = 0.0 - k[219]*y[IDX_H2COI] - k[839]*y[IDX_H2COI];
    IJth(jmatrix, 46, 94) = 0.0 + k[228]*y[IDX_H2COII];
    IJth(jmatrix, 46, 112) = 0.0 + k[438]*y[IDX_CH2I];
    IJth(jmatrix, 47, 25) = 0.0 + k[16]*y[IDX_H2COI];
    IJth(jmatrix, 47, 26) = 0.0 - k[55]*y[IDX_H2COII] -
        k[459]*y[IDX_H2COII];
    IJth(jmatrix, 47, 27) = 0.0 + k[402]*y[IDX_H2OI];
    IJth(jmatrix, 47, 28) = 0.0 - k[39]*y[IDX_H2COII] - k[423]*y[IDX_H2COII]
        + k[435]*y[IDX_O2II];
    IJth(jmatrix, 47, 29) = 0.0 + k[416]*y[IDX_CO2I];
    IJth(jmatrix, 47, 31) = 0.0 + k[443]*y[IDX_OI] + k[445]*y[IDX_OHI];
    IJth(jmatrix, 47, 32) = 0.0 + k[712]*y[IDX_NII] + k[808]*y[IDX_OII];
    IJth(jmatrix, 47, 33) = 0.0 - k[452]*y[IDX_H2COII];
    IJth(jmatrix, 47, 34) = 0.0 + k[49]*y[IDX_H2COI];
    IJth(jmatrix, 47, 36) = 0.0 + k[64]*y[IDX_H2COI];
    IJth(jmatrix, 47, 38) = 0.0 + k[70]*y[IDX_H2COI];
    IJth(jmatrix, 47, 39) = 0.0 + k[416]*y[IDX_CH2II];
    IJth(jmatrix, 47, 40) = 0.0 - k[306]*y[IDX_H2COII] -
        k[307]*y[IDX_H2COII] - k[308]*y[IDX_H2COII] - k[309]*y[IDX_H2COII] -
        k[1217]*y[IDX_H2COII];
    IJth(jmatrix, 47, 42) = 0.0 + k[79]*y[IDX_H2COI];
    IJth(jmatrix, 47, 44) = 0.0 + k[105]*y[IDX_H2COI];
    IJth(jmatrix, 47, 46) = 0.0 + k[16]*y[IDX_CII] + k[49]*y[IDX_CH4II] +
        k[64]*y[IDX_CNII] + k[70]*y[IDX_COII] + k[79]*y[IDX_HII] +
        k[105]*y[IDX_H2II] + k[116]*y[IDX_O2II] + k[117]*y[IDX_H2OII] +
        k[142]*y[IDX_HeII] + k[159]*y[IDX_NII] + k[170]*y[IDX_N2II] +
        k[175]*y[IDX_NHII] + k[209]*y[IDX_OII] + k[219]*y[IDX_OHII] -
        k[548]*y[IDX_H2COII] + k[1138];
    IJth(jmatrix, 47, 47) = 0.0 - k[39]*y[IDX_CH2I] - k[55]*y[IDX_CHI] -
        k[136]*y[IDX_HCOI] - k[148]*y[IDX_MgI] - k[194]*y[IDX_NH3I] -
        k[203]*y[IDX_NOI] - k[228]*y[IDX_SiI] - k[306]*y[IDX_EM] -
        k[307]*y[IDX_EM] - k[308]*y[IDX_EM] - k[309]*y[IDX_EM] -
        k[423]*y[IDX_CH2I] - k[452]*y[IDX_CH4I] - k[459]*y[IDX_CHI] -
        k[548]*y[IDX_H2COI] - k[549]*y[IDX_O2I] - k[563]*y[IDX_H2OI] -
        k[627]*y[IDX_HCNI] - k[641]*y[IDX_HCOI] - k[646]*y[IDX_HNCI] -
        k[780]*y[IDX_NH2I] - k[796]*y[IDX_NHI] - k[1217]*y[IDX_EM] - k[1284];
    IJth(jmatrix, 47, 49) = 0.0 + k[402]*y[IDX_CHII] - k[563]*y[IDX_H2COII];
    IJth(jmatrix, 47, 50) = 0.0 + k[117]*y[IDX_H2COI] + k[558]*y[IDX_HCOI];
    IJth(jmatrix, 47, 52) = 0.0 + k[588]*y[IDX_HCOI];
    IJth(jmatrix, 47, 55) = 0.0 - k[627]*y[IDX_H2COII];
    IJth(jmatrix, 47, 56) = 0.0 + k[624]*y[IDX_HCOI];
    IJth(jmatrix, 47, 58) = 0.0 - k[136]*y[IDX_H2COII] + k[558]*y[IDX_H2OII]
        + k[588]*y[IDX_H3II] + k[624]*y[IDX_HCNII] + k[636]*y[IDX_HCOII] -
        k[641]*y[IDX_H2COII] + k[642]*y[IDX_HNOII] + k[643]*y[IDX_N2HII] +
        k[645]*y[IDX_O2HII] + k[759]*y[IDX_NHII] + k[774]*y[IDX_NH2II] +
        k[843]*y[IDX_OHII];
    IJth(jmatrix, 47, 59) = 0.0 + k[636]*y[IDX_HCOI];
    IJth(jmatrix, 47, 62) = 0.0 + k[142]*y[IDX_H2COI];
    IJth(jmatrix, 47, 64) = 0.0 - k[646]*y[IDX_H2COII];
    IJth(jmatrix, 47, 67) = 0.0 + k[642]*y[IDX_HCOI];
    IJth(jmatrix, 47, 69) = 0.0 - k[148]*y[IDX_H2COII];
    IJth(jmatrix, 47, 72) = 0.0 + k[159]*y[IDX_H2COI] +
        k[712]*y[IDX_CH3OHI];
    IJth(jmatrix, 47, 74) = 0.0 + k[170]*y[IDX_H2COI];
    IJth(jmatrix, 47, 75) = 0.0 + k[643]*y[IDX_HCOI];
    IJth(jmatrix, 47, 76) = 0.0 - k[796]*y[IDX_H2COII];
    IJth(jmatrix, 47, 77) = 0.0 + k[175]*y[IDX_H2COI] + k[759]*y[IDX_HCOI];
    IJth(jmatrix, 47, 78) = 0.0 - k[780]*y[IDX_H2COII];
    IJth(jmatrix, 47, 79) = 0.0 + k[774]*y[IDX_HCOI];
    IJth(jmatrix, 47, 80) = 0.0 - k[194]*y[IDX_H2COII];
    IJth(jmatrix, 47, 82) = 0.0 - k[203]*y[IDX_H2COII];
    IJth(jmatrix, 47, 85) = 0.0 + k[443]*y[IDX_CH3II];
    IJth(jmatrix, 47, 86) = 0.0 + k[209]*y[IDX_H2COI] +
        k[808]*y[IDX_CH3OHI];
    IJth(jmatrix, 47, 87) = 0.0 - k[549]*y[IDX_H2COII];
    IJth(jmatrix, 47, 88) = 0.0 + k[116]*y[IDX_H2COI] + k[435]*y[IDX_CH2I];
    IJth(jmatrix, 47, 90) = 0.0 + k[645]*y[IDX_HCOI];
    IJth(jmatrix, 47, 92) = 0.0 + k[445]*y[IDX_CH3II];
    IJth(jmatrix, 47, 93) = 0.0 + k[219]*y[IDX_H2COI] + k[843]*y[IDX_HCOI];
    IJth(jmatrix, 47, 94) = 0.0 - k[228]*y[IDX_H2COII];
    IJth(jmatrix, 48, 40) = 0.0 - k[310]*y[IDX_H2NOII] -
        k[311]*y[IDX_H2NOII];
    IJth(jmatrix, 48, 48) = 0.0 - k[310]*y[IDX_EM] - k[311]*y[IDX_EM] -
        k[1296];
    IJth(jmatrix, 48, 52) = 0.0 + k[590]*y[IDX_HNOI];
    IJth(jmatrix, 48, 66) = 0.0 + k[590]*y[IDX_H3II];
    IJth(jmatrix, 48, 79) = 0.0 + k[777]*y[IDX_O2I];
    IJth(jmatrix, 48, 87) = 0.0 + k[777]*y[IDX_NH2II];
    IJth(jmatrix, 49, 6) = 0.0 + k[1315] + k[1316] + k[1317] + k[1318];
    IJth(jmatrix, 49, 25) = 0.0 - k[370]*y[IDX_H2OI] - k[371]*y[IDX_H2OI];
    IJth(jmatrix, 49, 26) = 0.0 + k[56]*y[IDX_H2OII] + k[462]*y[IDX_H3OII];
    IJth(jmatrix, 49, 27) = 0.0 - k[402]*y[IDX_H2OI] - k[403]*y[IDX_H2OI] -
        k[404]*y[IDX_H2OI];
    IJth(jmatrix, 49, 28) = 0.0 + k[40]*y[IDX_H2OII] + k[425]*y[IDX_H3OII] +
        k[891]*y[IDX_O2I] + k[899]*y[IDX_OHI];
    IJth(jmatrix, 49, 29) = 0.0 - k[418]*y[IDX_H2OI];
    IJth(jmatrix, 49, 30) = 0.0 - k[904]*y[IDX_H2OI] + k[910]*y[IDX_NOI] +
        k[912]*y[IDX_O2I] + k[919]*y[IDX_OHI];
    IJth(jmatrix, 49, 32) = 0.0 + k[492]*y[IDX_HII] + k[579]*y[IDX_H3II] +
        k[808]*y[IDX_OII];
    IJth(jmatrix, 49, 33) = 0.0 + k[922]*y[IDX_OHI];
    IJth(jmatrix, 49, 34) = 0.0 - k[450]*y[IDX_H2OI];
    IJth(jmatrix, 49, 36) = 0.0 - k[560]*y[IDX_H2OI] - k[561]*y[IDX_H2OI];
    IJth(jmatrix, 49, 38) = 0.0 - k[123]*y[IDX_H2OI] - k[562]*y[IDX_H2OI];
    IJth(jmatrix, 49, 40) = 0.0 + k[318]*y[IDX_H3COII] +
        k[322]*y[IDX_H3OII];
    IJth(jmatrix, 49, 41) = 0.0 - k[11]*y[IDX_H2OI] - k[975]*y[IDX_H2OI] +
        k[990]*y[IDX_O2HI] + k[1208]*y[IDX_OHI];
    IJth(jmatrix, 49, 42) = 0.0 - k[80]*y[IDX_H2OI] + k[492]*y[IDX_CH3OHI];
    IJth(jmatrix, 49, 43) = 0.0 - k[4]*y[IDX_H2OI] + k[966]*y[IDX_OHI];
    IJth(jmatrix, 49, 44) = 0.0 - k[106]*y[IDX_H2OI] - k[518]*y[IDX_H2OI];
    IJth(jmatrix, 49, 46) = 0.0 + k[117]*y[IDX_H2OII] + k[607]*y[IDX_H3OII]
        + k[1093]*y[IDX_OHI];
    IJth(jmatrix, 49, 47) = 0.0 - k[563]*y[IDX_H2OI];
    IJth(jmatrix, 49, 49) = 0.0 - k[4]*y[IDX_H2I] - k[11]*y[IDX_HI] -
        k[80]*y[IDX_HII] - k[106]*y[IDX_H2II] - k[123]*y[IDX_COII] -
        k[124]*y[IDX_HCNII] - k[125]*y[IDX_N2II] - k[143]*y[IDX_HeII] -
        k[160]*y[IDX_NII] - k[176]*y[IDX_NHII] - k[210]*y[IDX_OII] -
        k[220]*y[IDX_OHII] - k[256] - k[370]*y[IDX_CII] - k[371]*y[IDX_CII] -
        k[402]*y[IDX_CHII] - k[403]*y[IDX_CHII] - k[404]*y[IDX_CHII] -
        k[418]*y[IDX_CH2II] - k[450]*y[IDX_CH4II] - k[518]*y[IDX_H2II] -
        k[555]*y[IDX_H2OII] - k[560]*y[IDX_CNII] - k[561]*y[IDX_CNII] -
        k[562]*y[IDX_COII] - k[563]*y[IDX_H2COII] - k[564]*y[IDX_H3COII] -
        k[565]*y[IDX_HCNII] - k[566]*y[IDX_HCOII] - k[567]*y[IDX_HCO2II] -
        k[568]*y[IDX_HNOII] - k[569]*y[IDX_N2II] - k[570]*y[IDX_N2HII] -
        k[571]*y[IDX_O2HII] - k[572]*y[IDX_SiII] - k[573]*y[IDX_SiHII] -
        k[574]*y[IDX_SiH4II] - k[575]*y[IDX_SiH5II] - k[586]*y[IDX_H3II] -
        k[673]*y[IDX_HeII] - k[674]*y[IDX_HeII] - k[754]*y[IDX_NHII] -
        k[755]*y[IDX_NHII] - k[756]*y[IDX_NHII] - k[757]*y[IDX_NHII] -
        k[771]*y[IDX_NH2II] - k[772]*y[IDX_NH2II] - k[840]*y[IDX_OHII] -
        k[904]*y[IDX_CH3I] - k[975]*y[IDX_HI] - k[1036]*y[IDX_NHI] -
        k[1062]*y[IDX_OI] - k[1141] - k[1142] - k[1257];
    IJth(jmatrix, 49, 50) = 0.0 + k[40]*y[IDX_CH2I] + k[56]*y[IDX_CHI] +
        k[117]*y[IDX_H2COI] + k[118]*y[IDX_HCOI] + k[119]*y[IDX_MgI] +
        k[120]*y[IDX_NOI] + k[121]*y[IDX_O2I] + k[122]*y[IDX_SiI] +
        k[185]*y[IDX_NH2I] + k[195]*y[IDX_NH3I] - k[555]*y[IDX_H2OI];
    IJth(jmatrix, 49, 52) = 0.0 + k[579]*y[IDX_CH3OHI] - k[586]*y[IDX_H2OI];
    IJth(jmatrix, 49, 53) = 0.0 + k[318]*y[IDX_EM] - k[564]*y[IDX_H2OI];
    IJth(jmatrix, 49, 54) = 0.0 + k[322]*y[IDX_EM] + k[425]*y[IDX_CH2I] +
        k[462]*y[IDX_CHI] + k[607]*y[IDX_H2COI] + k[608]*y[IDX_HCNI] +
        k[609]*y[IDX_HNCI] + k[610]*y[IDX_SiI] + k[611]*y[IDX_SiH2I] +
        k[612]*y[IDX_SiHI] + k[613]*y[IDX_SiOI] + k[783]*y[IDX_NH2I];
    IJth(jmatrix, 49, 55) = 0.0 + k[608]*y[IDX_H3OII] + k[1094]*y[IDX_OHI];
    IJth(jmatrix, 49, 56) = 0.0 - k[124]*y[IDX_H2OI] - k[565]*y[IDX_H2OI];
    IJth(jmatrix, 49, 58) = 0.0 + k[118]*y[IDX_H2OII] + k[1096]*y[IDX_OHI];
    IJth(jmatrix, 49, 59) = 0.0 - k[566]*y[IDX_H2OI];
    IJth(jmatrix, 49, 60) = 0.0 - k[567]*y[IDX_H2OI];
    IJth(jmatrix, 49, 62) = 0.0 - k[143]*y[IDX_H2OI] - k[673]*y[IDX_H2OI] -
        k[674]*y[IDX_H2OI];
    IJth(jmatrix, 49, 64) = 0.0 + k[609]*y[IDX_H3OII];
    IJth(jmatrix, 49, 66) = 0.0 + k[1097]*y[IDX_OHI];
    IJth(jmatrix, 49, 67) = 0.0 - k[568]*y[IDX_H2OI];
    IJth(jmatrix, 49, 69) = 0.0 + k[119]*y[IDX_H2OII];
    IJth(jmatrix, 49, 72) = 0.0 - k[160]*y[IDX_H2OI];
    IJth(jmatrix, 49, 74) = 0.0 - k[125]*y[IDX_H2OI] - k[569]*y[IDX_H2OI];
    IJth(jmatrix, 49, 75) = 0.0 - k[570]*y[IDX_H2OI];
    IJth(jmatrix, 49, 76) = 0.0 - k[1036]*y[IDX_H2OI] + k[1048]*y[IDX_OHI];
    IJth(jmatrix, 49, 77) = 0.0 - k[176]*y[IDX_H2OI] - k[754]*y[IDX_H2OI] -
        k[755]*y[IDX_H2OI] - k[756]*y[IDX_H2OI] - k[757]*y[IDX_H2OI];
    IJth(jmatrix, 49, 78) = 0.0 + k[185]*y[IDX_H2OII] + k[783]*y[IDX_H3OII]
        + k[1029]*y[IDX_NOI] + k[1031]*y[IDX_OHI];
    IJth(jmatrix, 49, 79) = 0.0 - k[771]*y[IDX_H2OI] - k[772]*y[IDX_H2OI];
    IJth(jmatrix, 49, 80) = 0.0 + k[195]*y[IDX_H2OII] + k[1098]*y[IDX_OHI];
    IJth(jmatrix, 49, 82) = 0.0 + k[120]*y[IDX_H2OII] + k[910]*y[IDX_CH3I] +
        k[1029]*y[IDX_NH2I];
    IJth(jmatrix, 49, 85) = 0.0 - k[1062]*y[IDX_H2OI];
    IJth(jmatrix, 49, 86) = 0.0 - k[210]*y[IDX_H2OI] + k[808]*y[IDX_CH3OHI];
    IJth(jmatrix, 49, 87) = 0.0 + k[121]*y[IDX_H2OII] + k[891]*y[IDX_CH2I] +
        k[912]*y[IDX_CH3I];
    IJth(jmatrix, 49, 89) = 0.0 + k[990]*y[IDX_HI] + k[1100]*y[IDX_OHI];
    IJth(jmatrix, 49, 90) = 0.0 - k[571]*y[IDX_H2OI];
    IJth(jmatrix, 49, 92) = 0.0 + k[899]*y[IDX_CH2I] + k[919]*y[IDX_CH3I] +
        k[922]*y[IDX_CH4I] + k[966]*y[IDX_H2I] + k[1031]*y[IDX_NH2I] +
        k[1048]*y[IDX_NHI] + k[1093]*y[IDX_H2COI] + k[1094]*y[IDX_HCNI] +
        k[1096]*y[IDX_HCOI] + k[1097]*y[IDX_HNOI] + k[1098]*y[IDX_NH3I] +
        k[1100]*y[IDX_O2HI] + k[1101]*y[IDX_OHI] + k[1101]*y[IDX_OHI] +
        k[1208]*y[IDX_HI];
    IJth(jmatrix, 49, 93) = 0.0 - k[220]*y[IDX_H2OI] - k[840]*y[IDX_H2OI];
    IJth(jmatrix, 49, 94) = 0.0 + k[122]*y[IDX_H2OII] + k[610]*y[IDX_H3OII];
    IJth(jmatrix, 49, 95) = 0.0 - k[572]*y[IDX_H2OI];
    IJth(jmatrix, 49, 102) = 0.0 + k[612]*y[IDX_H3OII];
    IJth(jmatrix, 49, 103) = 0.0 - k[573]*y[IDX_H2OI];
    IJth(jmatrix, 49, 104) = 0.0 + k[611]*y[IDX_H3OII];
    IJth(jmatrix, 49, 109) = 0.0 - k[574]*y[IDX_H2OI];
    IJth(jmatrix, 49, 110) = 0.0 - k[575]*y[IDX_H2OI];
    IJth(jmatrix, 49, 111) = 0.0 + k[613]*y[IDX_H3OII];
    IJth(jmatrix, 50, 24) = 0.0 - k[383]*y[IDX_H2OII];
    IJth(jmatrix, 50, 26) = 0.0 - k[56]*y[IDX_H2OII] - k[460]*y[IDX_H2OII];
    IJth(jmatrix, 50, 28) = 0.0 - k[40]*y[IDX_H2OII] - k[424]*y[IDX_H2OII];
    IJth(jmatrix, 50, 33) = 0.0 - k[453]*y[IDX_H2OII];
    IJth(jmatrix, 50, 37) = 0.0 - k[553]*y[IDX_H2OII];
    IJth(jmatrix, 50, 38) = 0.0 + k[123]*y[IDX_H2OI];
    IJth(jmatrix, 50, 40) = 0.0 - k[312]*y[IDX_H2OII] - k[313]*y[IDX_H2OII]
        - k[314]*y[IDX_H2OII];
    IJth(jmatrix, 50, 42) = 0.0 + k[80]*y[IDX_H2OI];
    IJth(jmatrix, 50, 43) = 0.0 - k[534]*y[IDX_H2OII] + k[545]*y[IDX_OHII];
    IJth(jmatrix, 50, 44) = 0.0 + k[106]*y[IDX_H2OI] + k[527]*y[IDX_OHI];
    IJth(jmatrix, 50, 46) = 0.0 - k[117]*y[IDX_H2OII] - k[554]*y[IDX_H2OII];
    IJth(jmatrix, 50, 49) = 0.0 + k[80]*y[IDX_HII] + k[106]*y[IDX_H2II] +
        k[123]*y[IDX_COII] + k[124]*y[IDX_HCNII] + k[125]*y[IDX_N2II] +
        k[143]*y[IDX_HeII] + k[160]*y[IDX_NII] + k[176]*y[IDX_NHII] +
        k[210]*y[IDX_OII] + k[220]*y[IDX_OHII] - k[555]*y[IDX_H2OII] + k[1141];
    IJth(jmatrix, 50, 50) = 0.0 - k[40]*y[IDX_CH2I] - k[56]*y[IDX_CHI] -
        k[117]*y[IDX_H2COI] - k[118]*y[IDX_HCOI] - k[119]*y[IDX_MgI] -
        k[120]*y[IDX_NOI] - k[121]*y[IDX_O2I] - k[122]*y[IDX_SiI] -
        k[185]*y[IDX_NH2I] - k[195]*y[IDX_NH3I] - k[312]*y[IDX_EM] -
        k[313]*y[IDX_EM] - k[314]*y[IDX_EM] - k[383]*y[IDX_CI] -
        k[424]*y[IDX_CH2I] - k[453]*y[IDX_CH4I] - k[460]*y[IDX_CHI] -
        k[534]*y[IDX_H2I] - k[553]*y[IDX_COI] - k[554]*y[IDX_H2COI] -
        k[555]*y[IDX_H2OI] - k[556]*y[IDX_HCNI] - k[557]*y[IDX_HCOI] -
        k[558]*y[IDX_HCOI] - k[559]*y[IDX_HNCI] - k[738]*y[IDX_NI] -
        k[739]*y[IDX_NI] - k[781]*y[IDX_NH2I] - k[797]*y[IDX_NHI] -
        k[823]*y[IDX_OI] - k[852]*y[IDX_OHI] - k[1140] - k[1280];
    IJth(jmatrix, 50, 52) = 0.0 + k[598]*y[IDX_OI] + k[600]*y[IDX_OHI];
    IJth(jmatrix, 50, 55) = 0.0 - k[556]*y[IDX_H2OII];
    IJth(jmatrix, 50, 56) = 0.0 + k[124]*y[IDX_H2OI] + k[853]*y[IDX_OHI];
    IJth(jmatrix, 50, 58) = 0.0 - k[118]*y[IDX_H2OII] - k[557]*y[IDX_H2OII]
        - k[558]*y[IDX_H2OII] + k[842]*y[IDX_OHII];
    IJth(jmatrix, 50, 59) = 0.0 + k[854]*y[IDX_OHI];
    IJth(jmatrix, 50, 62) = 0.0 + k[143]*y[IDX_H2OI];
    IJth(jmatrix, 50, 64) = 0.0 - k[559]*y[IDX_H2OII];
    IJth(jmatrix, 50, 67) = 0.0 + k[856]*y[IDX_OHI];
    IJth(jmatrix, 50, 69) = 0.0 - k[119]*y[IDX_H2OII];
    IJth(jmatrix, 50, 71) = 0.0 - k[738]*y[IDX_H2OII] - k[739]*y[IDX_H2OII];
    IJth(jmatrix, 50, 72) = 0.0 + k[160]*y[IDX_H2OI];
    IJth(jmatrix, 50, 74) = 0.0 + k[125]*y[IDX_H2OI];
    IJth(jmatrix, 50, 75) = 0.0 + k[857]*y[IDX_OHI];
    IJth(jmatrix, 50, 76) = 0.0 - k[797]*y[IDX_H2OII];
    IJth(jmatrix, 50, 77) = 0.0 + k[176]*y[IDX_H2OI] + k[768]*y[IDX_OHI];
    IJth(jmatrix, 50, 78) = 0.0 - k[185]*y[IDX_H2OII] - k[781]*y[IDX_H2OII];
    IJth(jmatrix, 50, 80) = 0.0 - k[195]*y[IDX_H2OII];
    IJth(jmatrix, 50, 82) = 0.0 - k[120]*y[IDX_H2OII];
    IJth(jmatrix, 50, 85) = 0.0 + k[598]*y[IDX_H3II] - k[823]*y[IDX_H2OII];
    IJth(jmatrix, 50, 86) = 0.0 + k[210]*y[IDX_H2OI];
    IJth(jmatrix, 50, 87) = 0.0 - k[121]*y[IDX_H2OII];
    IJth(jmatrix, 50, 90) = 0.0 + k[858]*y[IDX_OHI];
    IJth(jmatrix, 50, 92) = 0.0 + k[527]*y[IDX_H2II] + k[600]*y[IDX_H3II] +
        k[768]*y[IDX_NHII] + k[847]*y[IDX_OHII] - k[852]*y[IDX_H2OII] +
        k[853]*y[IDX_HCNII] + k[854]*y[IDX_HCOII] + k[856]*y[IDX_HNOII] +
        k[857]*y[IDX_N2HII] + k[858]*y[IDX_O2HII];
    IJth(jmatrix, 50, 93) = 0.0 + k[220]*y[IDX_H2OI] + k[545]*y[IDX_H2I] +
        k[842]*y[IDX_HCOI] + k[847]*y[IDX_OHI];
    IJth(jmatrix, 50, 94) = 0.0 - k[122]*y[IDX_H2OII];
    IJth(jmatrix, 51, 7) = 0.0 + k[1391] + k[1392] + k[1393] + k[1394];
    IJth(jmatrix, 51, 42) = 0.0 - k[499]*y[IDX_H2SiOI];
    IJth(jmatrix, 51, 51) = 0.0 - k[257] - k[499]*y[IDX_HII] -
        k[675]*y[IDX_HeII] - k[1143] - k[1144] - k[1247];
    IJth(jmatrix, 51, 62) = 0.0 - k[675]*y[IDX_H2SiOI];
    IJth(jmatrix, 51, 85) = 0.0 + k[1087]*y[IDX_SiH3I];
    IJth(jmatrix, 51, 106) = 0.0 + k[1087]*y[IDX_OI];
    IJth(jmatrix, 52, 24) = 0.0 - k[576]*y[IDX_H3II];
    IJth(jmatrix, 52, 26) = 0.0 - k[580]*y[IDX_H3II];
    IJth(jmatrix, 52, 28) = 0.0 - k[577]*y[IDX_H3II];
    IJth(jmatrix, 52, 30) = 0.0 - k[578]*y[IDX_H3II];
    IJth(jmatrix, 52, 32) = 0.0 - k[579]*y[IDX_H3II];
    IJth(jmatrix, 52, 35) = 0.0 - k[581]*y[IDX_H3II];
    IJth(jmatrix, 52, 37) = 0.0 - k[583]*y[IDX_H3II] - k[584]*y[IDX_H3II];
    IJth(jmatrix, 52, 39) = 0.0 - k[582]*y[IDX_H3II];
    IJth(jmatrix, 52, 40) = 0.0 - k[315]*y[IDX_H3II] - k[316]*y[IDX_H3II];
    IJth(jmatrix, 52, 43) = 0.0 + k[516]*y[IDX_H2II] + k[537]*y[IDX_HeHII] +
        k[540]*y[IDX_NHII] + k[544]*y[IDX_O2HII];
    IJth(jmatrix, 52, 44) = 0.0 + k[516]*y[IDX_H2I] + k[519]*y[IDX_HCOI];
    IJth(jmatrix, 52, 46) = 0.0 - k[585]*y[IDX_H3II];
    IJth(jmatrix, 52, 49) = 0.0 - k[586]*y[IDX_H3II];
    IJth(jmatrix, 52, 52) = 0.0 - k[315]*y[IDX_EM] - k[316]*y[IDX_EM] -
        k[576]*y[IDX_CI] - k[577]*y[IDX_CH2I] - k[578]*y[IDX_CH3I] -
        k[579]*y[IDX_CH3OHI] - k[580]*y[IDX_CHI] - k[581]*y[IDX_CNI] -
        k[582]*y[IDX_CO2I] - k[583]*y[IDX_COI] - k[584]*y[IDX_COI] -
        k[585]*y[IDX_H2COI] - k[586]*y[IDX_H2OI] - k[587]*y[IDX_HCNI] -
        k[588]*y[IDX_HCOI] - k[589]*y[IDX_HNCI] - k[590]*y[IDX_HNOI] -
        k[591]*y[IDX_MgI] - k[592]*y[IDX_N2I] - k[593]*y[IDX_NH2I] -
        k[594]*y[IDX_NHI] - k[595]*y[IDX_NO2I] - k[596]*y[IDX_NOI] -
        k[597]*y[IDX_O2I] - k[598]*y[IDX_OI] - k[599]*y[IDX_OI] -
        k[600]*y[IDX_OHI] - k[601]*y[IDX_SiI] - k[602]*y[IDX_SiH2I] -
        k[603]*y[IDX_SiH3I] - k[604]*y[IDX_SiH4I] - k[605]*y[IDX_SiHI] -
        k[606]*y[IDX_SiOI] - k[1145] - k[1146];
    IJth(jmatrix, 52, 55) = 0.0 - k[587]*y[IDX_H3II];
    IJth(jmatrix, 52, 58) = 0.0 + k[519]*y[IDX_H2II] - k[588]*y[IDX_H3II];
    IJth(jmatrix, 52, 63) = 0.0 + k[537]*y[IDX_H2I];
    IJth(jmatrix, 52, 64) = 0.0 - k[589]*y[IDX_H3II];
    IJth(jmatrix, 52, 66) = 0.0 - k[590]*y[IDX_H3II];
    IJth(jmatrix, 52, 69) = 0.0 - k[591]*y[IDX_H3II];
    IJth(jmatrix, 52, 73) = 0.0 - k[592]*y[IDX_H3II];
    IJth(jmatrix, 52, 76) = 0.0 - k[594]*y[IDX_H3II];
    IJth(jmatrix, 52, 77) = 0.0 + k[540]*y[IDX_H2I];
    IJth(jmatrix, 52, 78) = 0.0 - k[593]*y[IDX_H3II];
    IJth(jmatrix, 52, 82) = 0.0 - k[596]*y[IDX_H3II];
    IJth(jmatrix, 52, 84) = 0.0 - k[595]*y[IDX_H3II];
    IJth(jmatrix, 52, 85) = 0.0 - k[598]*y[IDX_H3II] - k[599]*y[IDX_H3II];
    IJth(jmatrix, 52, 87) = 0.0 - k[597]*y[IDX_H3II];
    IJth(jmatrix, 52, 90) = 0.0 + k[544]*y[IDX_H2I];
    IJth(jmatrix, 52, 92) = 0.0 - k[600]*y[IDX_H3II];
    IJth(jmatrix, 52, 94) = 0.0 - k[601]*y[IDX_H3II];
    IJth(jmatrix, 52, 102) = 0.0 - k[605]*y[IDX_H3II];
    IJth(jmatrix, 52, 104) = 0.0 - k[602]*y[IDX_H3II];
    IJth(jmatrix, 52, 106) = 0.0 - k[603]*y[IDX_H3II];
    IJth(jmatrix, 52, 108) = 0.0 - k[604]*y[IDX_H3II];
    IJth(jmatrix, 52, 111) = 0.0 - k[606]*y[IDX_H3II];
    IJth(jmatrix, 53, 25) = 0.0 + k[365]*y[IDX_CH3OHI];
    IJth(jmatrix, 53, 26) = 0.0 - k[461]*y[IDX_H3COII];
    IJth(jmatrix, 53, 27) = 0.0 + k[397]*y[IDX_CH3OHI] +
        k[400]*y[IDX_H2COI];
    IJth(jmatrix, 53, 29) = 0.0 + k[418]*y[IDX_H2OI];
    IJth(jmatrix, 53, 31) = 0.0 + k[439]*y[IDX_CH3OHI] + k[442]*y[IDX_O2I];
    IJth(jmatrix, 53, 32) = 0.0 + k[365]*y[IDX_CII] + k[397]*y[IDX_CHII] +
        k[439]*y[IDX_CH3II] + k[493]*y[IDX_HII] + k[713]*y[IDX_NII] +
        k[809]*y[IDX_OII] + k[820]*y[IDX_O2II] + k[1120];
    IJth(jmatrix, 53, 33) = 0.0 + k[452]*y[IDX_H2COII];
    IJth(jmatrix, 53, 34) = 0.0 + k[449]*y[IDX_H2COI];
    IJth(jmatrix, 53, 40) = 0.0 - k[317]*y[IDX_H3COII] -
        k[318]*y[IDX_H3COII] - k[319]*y[IDX_H3COII] - k[320]*y[IDX_H3COII] -
        k[321]*y[IDX_H3COII];
    IJth(jmatrix, 53, 42) = 0.0 + k[493]*y[IDX_CH3OHI];
    IJth(jmatrix, 53, 46) = 0.0 + k[400]*y[IDX_CHII] + k[449]*y[IDX_CH4II] +
        k[548]*y[IDX_H2COII] + k[550]*y[IDX_HNOII] + k[552]*y[IDX_O2HII] +
        k[554]*y[IDX_H2OII] + k[585]*y[IDX_H3II] + k[607]*y[IDX_H3OII] +
        k[622]*y[IDX_HCNII] + k[633]*y[IDX_HCNHII] + k[634]*y[IDX_HCNHII] +
        k[635]*y[IDX_HCOII] + k[735]*y[IDX_N2HII] + k[752]*y[IDX_NHII] +
        k[769]*y[IDX_NH2II] + k[839]*y[IDX_OHII];
    IJth(jmatrix, 53, 47) = 0.0 + k[452]*y[IDX_CH4I] + k[548]*y[IDX_H2COI] +
        k[641]*y[IDX_HCOI] + k[796]*y[IDX_NHI];
    IJth(jmatrix, 53, 49) = 0.0 + k[418]*y[IDX_CH2II] -
        k[564]*y[IDX_H3COII];
    IJth(jmatrix, 53, 50) = 0.0 + k[554]*y[IDX_H2COI];
    IJth(jmatrix, 53, 52) = 0.0 + k[585]*y[IDX_H2COI];
    IJth(jmatrix, 53, 53) = 0.0 - k[317]*y[IDX_EM] - k[318]*y[IDX_EM] -
        k[319]*y[IDX_EM] - k[320]*y[IDX_EM] - k[321]*y[IDX_EM] -
        k[461]*y[IDX_CHI] - k[564]*y[IDX_H2OI] - k[628]*y[IDX_HCNI] -
        k[647]*y[IDX_HNCI] - k[782]*y[IDX_NH2I] - k[1249];
    IJth(jmatrix, 53, 54) = 0.0 + k[607]*y[IDX_H2COI];
    IJth(jmatrix, 53, 55) = 0.0 - k[628]*y[IDX_H3COII];
    IJth(jmatrix, 53, 56) = 0.0 + k[622]*y[IDX_H2COI];
    IJth(jmatrix, 53, 57) = 0.0 + k[633]*y[IDX_H2COI] + k[634]*y[IDX_H2COI];
    IJth(jmatrix, 53, 58) = 0.0 + k[641]*y[IDX_H2COII];
    IJth(jmatrix, 53, 59) = 0.0 + k[635]*y[IDX_H2COI];
    IJth(jmatrix, 53, 64) = 0.0 - k[647]*y[IDX_H3COII];
    IJth(jmatrix, 53, 67) = 0.0 + k[550]*y[IDX_H2COI];
    IJth(jmatrix, 53, 72) = 0.0 + k[713]*y[IDX_CH3OHI];
    IJth(jmatrix, 53, 75) = 0.0 + k[735]*y[IDX_H2COI];
    IJth(jmatrix, 53, 76) = 0.0 + k[796]*y[IDX_H2COII];
    IJth(jmatrix, 53, 77) = 0.0 + k[752]*y[IDX_H2COI];
    IJth(jmatrix, 53, 78) = 0.0 - k[782]*y[IDX_H3COII];
    IJth(jmatrix, 53, 79) = 0.0 + k[769]*y[IDX_H2COI];
    IJth(jmatrix, 53, 86) = 0.0 + k[809]*y[IDX_CH3OHI];
    IJth(jmatrix, 53, 87) = 0.0 + k[442]*y[IDX_CH3II];
    IJth(jmatrix, 53, 88) = 0.0 + k[820]*y[IDX_CH3OHI];
    IJth(jmatrix, 53, 90) = 0.0 + k[552]*y[IDX_H2COI];
    IJth(jmatrix, 53, 93) = 0.0 + k[839]*y[IDX_H2COI];
    IJth(jmatrix, 54, 24) = 0.0 - k[384]*y[IDX_H3OII];
    IJth(jmatrix, 54, 26) = 0.0 - k[462]*y[IDX_H3OII];
    IJth(jmatrix, 54, 27) = 0.0 + k[403]*y[IDX_H2OI];
    IJth(jmatrix, 54, 28) = 0.0 - k[425]*y[IDX_H3OII];
    IJth(jmatrix, 54, 33) = 0.0 + k[453]*y[IDX_H2OII] + k[457]*y[IDX_OHII];
    IJth(jmatrix, 54, 34) = 0.0 + k[450]*y[IDX_H2OI];
    IJth(jmatrix, 54, 40) = 0.0 - k[322]*y[IDX_H3OII] - k[323]*y[IDX_H3OII]
        - k[324]*y[IDX_H3OII] - k[325]*y[IDX_H3OII];
    IJth(jmatrix, 54, 43) = 0.0 + k[534]*y[IDX_H2OII];
    IJth(jmatrix, 54, 44) = 0.0 + k[518]*y[IDX_H2OI];
    IJth(jmatrix, 54, 46) = 0.0 - k[607]*y[IDX_H3OII];
    IJth(jmatrix, 54, 47) = 0.0 + k[563]*y[IDX_H2OI];
    IJth(jmatrix, 54, 49) = 0.0 + k[403]*y[IDX_CHII] + k[450]*y[IDX_CH4II] +
        k[518]*y[IDX_H2II] + k[555]*y[IDX_H2OII] + k[563]*y[IDX_H2COII] +
        k[564]*y[IDX_H3COII] + k[565]*y[IDX_HCNII] + k[566]*y[IDX_HCOII] +
        k[567]*y[IDX_HCO2II] + k[568]*y[IDX_HNOII] + k[570]*y[IDX_N2HII] +
        k[571]*y[IDX_O2HII] + k[573]*y[IDX_SiHII] + k[574]*y[IDX_SiH4II] +
        k[575]*y[IDX_SiH5II] + k[586]*y[IDX_H3II] + k[754]*y[IDX_NHII] +
        k[771]*y[IDX_NH2II] + k[840]*y[IDX_OHII];
    IJth(jmatrix, 54, 50) = 0.0 + k[453]*y[IDX_CH4I] + k[534]*y[IDX_H2I] +
        k[555]*y[IDX_H2OI] + k[557]*y[IDX_HCOI] + k[797]*y[IDX_NHI] +
        k[852]*y[IDX_OHI];
    IJth(jmatrix, 54, 52) = 0.0 + k[586]*y[IDX_H2OI];
    IJth(jmatrix, 54, 53) = 0.0 + k[564]*y[IDX_H2OI];
    IJth(jmatrix, 54, 54) = 0.0 - k[322]*y[IDX_EM] - k[323]*y[IDX_EM] -
        k[324]*y[IDX_EM] - k[325]*y[IDX_EM] - k[384]*y[IDX_CI] -
        k[425]*y[IDX_CH2I] - k[462]*y[IDX_CHI] - k[607]*y[IDX_H2COI] -
        k[608]*y[IDX_HCNI] - k[609]*y[IDX_HNCI] - k[610]*y[IDX_SiI] -
        k[611]*y[IDX_SiH2I] - k[612]*y[IDX_SiHI] - k[613]*y[IDX_SiOI] -
        k[783]*y[IDX_NH2I] - k[1286];
    IJth(jmatrix, 54, 55) = 0.0 - k[608]*y[IDX_H3OII];
    IJth(jmatrix, 54, 56) = 0.0 + k[565]*y[IDX_H2OI];
    IJth(jmatrix, 54, 58) = 0.0 + k[557]*y[IDX_H2OII];
    IJth(jmatrix, 54, 59) = 0.0 + k[566]*y[IDX_H2OI];
    IJth(jmatrix, 54, 60) = 0.0 + k[567]*y[IDX_H2OI];
    IJth(jmatrix, 54, 64) = 0.0 - k[609]*y[IDX_H3OII];
    IJth(jmatrix, 54, 67) = 0.0 + k[568]*y[IDX_H2OI];
    IJth(jmatrix, 54, 75) = 0.0 + k[570]*y[IDX_H2OI];
    IJth(jmatrix, 54, 76) = 0.0 + k[797]*y[IDX_H2OII];
    IJth(jmatrix, 54, 77) = 0.0 + k[754]*y[IDX_H2OI];
    IJth(jmatrix, 54, 78) = 0.0 - k[783]*y[IDX_H3OII];
    IJth(jmatrix, 54, 79) = 0.0 + k[771]*y[IDX_H2OI];
    IJth(jmatrix, 54, 90) = 0.0 + k[571]*y[IDX_H2OI];
    IJth(jmatrix, 54, 92) = 0.0 + k[852]*y[IDX_H2OII];
    IJth(jmatrix, 54, 93) = 0.0 + k[457]*y[IDX_CH4I] + k[840]*y[IDX_H2OI];
    IJth(jmatrix, 54, 94) = 0.0 - k[610]*y[IDX_H3OII];
    IJth(jmatrix, 54, 102) = 0.0 - k[612]*y[IDX_H3OII];
    IJth(jmatrix, 54, 103) = 0.0 + k[573]*y[IDX_H2OI];
    IJth(jmatrix, 54, 104) = 0.0 - k[611]*y[IDX_H3OII];
    IJth(jmatrix, 54, 109) = 0.0 + k[574]*y[IDX_H2OI];
    IJth(jmatrix, 54, 110) = 0.0 + k[575]*y[IDX_H2OI];
    IJth(jmatrix, 54, 111) = 0.0 - k[613]*y[IDX_H3OII];
    IJth(jmatrix, 55, 8) = 0.0 + k[1323] + k[1324] + k[1325] + k[1326];
    IJth(jmatrix, 55, 24) = 0.0 + k[866]*y[IDX_NH2I];
    IJth(jmatrix, 55, 26) = 0.0 + k[464]*y[IDX_HCNHII] + k[927]*y[IDX_N2I] +
        k[930]*y[IDX_NOI];
    IJth(jmatrix, 55, 27) = 0.0 - k[405]*y[IDX_HCNI];
    IJth(jmatrix, 55, 28) = 0.0 + k[427]*y[IDX_HCNHII] + k[880]*y[IDX_CNI] +
        k[884]*y[IDX_N2I] + k[887]*y[IDX_NOI] + k[1005]*y[IDX_NI];
    IJth(jmatrix, 55, 30) = 0.0 + k[902]*y[IDX_CNI] + k[910]*y[IDX_NOI] +
        k[1009]*y[IDX_NI] + k[1010]*y[IDX_NI];
    IJth(jmatrix, 55, 33) = 0.0 + k[920]*y[IDX_CNI];
    IJth(jmatrix, 55, 35) = 0.0 + k[880]*y[IDX_CH2I] + k[902]*y[IDX_CH3I] +
        k[920]*y[IDX_CH4I] + k[942]*y[IDX_H2COI] + k[943]*y[IDX_HCOI] +
        k[944]*y[IDX_HNOI] + k[950]*y[IDX_SiH4I] + k[959]*y[IDX_H2I] +
        k[1033]*y[IDX_NH3I] + k[1035]*y[IDX_NHI] + k[1090]*y[IDX_OHI];
    IJth(jmatrix, 55, 36) = 0.0 - k[65]*y[IDX_HCNI] + k[479]*y[IDX_H2COI];
    IJth(jmatrix, 55, 38) = 0.0 - k[134]*y[IDX_HCNI];
    IJth(jmatrix, 55, 40) = 0.0 + k[328]*y[IDX_HCNHII];
    IJth(jmatrix, 55, 41) = 0.0 + k[129]*y[IDX_HCNII] + k[973]*y[IDX_H2CNI]
        - k[976]*y[IDX_HCNI] + k[979]*y[IDX_HNCI] + k[993]*y[IDX_OCNI];
    IJth(jmatrix, 55, 42) = 0.0 + k[1]*y[IDX_HNCI] - k[81]*y[IDX_HCNI];
    IJth(jmatrix, 55, 43) = 0.0 + k[959]*y[IDX_CNI];
    IJth(jmatrix, 55, 44) = 0.0 - k[107]*y[IDX_HCNI];
    IJth(jmatrix, 55, 45) = 0.0 + k[254] + k[973]*y[IDX_HI] +
        k[1013]*y[IDX_NI] + k[1135];
    IJth(jmatrix, 55, 46) = 0.0 + k[479]*y[IDX_CNII] + k[633]*y[IDX_HCNHII]
        + k[942]*y[IDX_CNI];
    IJth(jmatrix, 55, 47) = 0.0 - k[627]*y[IDX_HCNI];
    IJth(jmatrix, 55, 49) = 0.0 + k[124]*y[IDX_HCNII];
    IJth(jmatrix, 55, 50) = 0.0 - k[556]*y[IDX_HCNI];
    IJth(jmatrix, 55, 52) = 0.0 - k[587]*y[IDX_HCNI];
    IJth(jmatrix, 55, 53) = 0.0 - k[628]*y[IDX_HCNI];
    IJth(jmatrix, 55, 54) = 0.0 - k[608]*y[IDX_HCNI];
    IJth(jmatrix, 55, 55) = 0.0 - k[65]*y[IDX_CNII] - k[81]*y[IDX_HII] -
        k[107]*y[IDX_H2II] - k[134]*y[IDX_COII] - k[135]*y[IDX_N2II] -
        k[161]*y[IDX_NII] - k[259] - k[405]*y[IDX_CHII] - k[556]*y[IDX_H2OII] -
        k[587]*y[IDX_H3II] - k[608]*y[IDX_H3OII] - k[623]*y[IDX_HCNII] -
        k[627]*y[IDX_H2COII] - k[628]*y[IDX_H3COII] - k[629]*y[IDX_HCOII] -
        k[630]*y[IDX_HNOII] - k[631]*y[IDX_N2HII] - k[632]*y[IDX_O2HII] -
        k[676]*y[IDX_HeII] - k[677]*y[IDX_HeII] - k[678]*y[IDX_HeII] -
        k[679]*y[IDX_HeII] - k[758]*y[IDX_NHII] - k[773]*y[IDX_NH2II] -
        k[814]*y[IDX_OII] - k[815]*y[IDX_OII] - k[841]*y[IDX_OHII] -
        k[976]*y[IDX_HI] - k[1063]*y[IDX_OI] - k[1064]*y[IDX_OI] -
        k[1065]*y[IDX_OI] - k[1094]*y[IDX_OHI] - k[1095]*y[IDX_OHI] - k[1147] -
        k[1266];
    IJth(jmatrix, 55, 56) = 0.0 + k[124]*y[IDX_H2OI] + k[129]*y[IDX_HI] +
        k[132]*y[IDX_NOI] + k[133]*y[IDX_O2I] + k[196]*y[IDX_NH3I] -
        k[623]*y[IDX_HCNI];
    IJth(jmatrix, 55, 57) = 0.0 + k[328]*y[IDX_EM] + k[427]*y[IDX_CH2I] +
        k[464]*y[IDX_CHI] + k[633]*y[IDX_H2COI] + k[785]*y[IDX_NH2I];
    IJth(jmatrix, 55, 58) = 0.0 + k[943]*y[IDX_CNI] + k[1015]*y[IDX_NI];
    IJth(jmatrix, 55, 59) = 0.0 - k[629]*y[IDX_HCNI];
    IJth(jmatrix, 55, 62) = 0.0 - k[676]*y[IDX_HCNI] - k[677]*y[IDX_HCNI] -
        k[678]*y[IDX_HCNI] - k[679]*y[IDX_HCNI];
    IJth(jmatrix, 55, 64) = 0.0 + k[1]*y[IDX_HII] + k[979]*y[IDX_HI];
    IJth(jmatrix, 55, 66) = 0.0 + k[944]*y[IDX_CNI];
    IJth(jmatrix, 55, 67) = 0.0 - k[630]*y[IDX_HCNI];
    IJth(jmatrix, 55, 71) = 0.0 + k[1005]*y[IDX_CH2I] + k[1009]*y[IDX_CH3I]
        + k[1010]*y[IDX_CH3I] + k[1013]*y[IDX_H2CNI] + k[1015]*y[IDX_HCOI];
    IJth(jmatrix, 55, 72) = 0.0 - k[161]*y[IDX_HCNI];
    IJth(jmatrix, 55, 73) = 0.0 + k[884]*y[IDX_CH2I] + k[927]*y[IDX_CHI];
    IJth(jmatrix, 55, 74) = 0.0 - k[135]*y[IDX_HCNI];
    IJth(jmatrix, 55, 75) = 0.0 - k[631]*y[IDX_HCNI];
    IJth(jmatrix, 55, 76) = 0.0 + k[1035]*y[IDX_CNI];
    IJth(jmatrix, 55, 77) = 0.0 - k[758]*y[IDX_HCNI];
    IJth(jmatrix, 55, 78) = 0.0 + k[785]*y[IDX_HCNHII] + k[866]*y[IDX_CI];
    IJth(jmatrix, 55, 79) = 0.0 - k[773]*y[IDX_HCNI];
    IJth(jmatrix, 55, 80) = 0.0 + k[196]*y[IDX_HCNII] + k[1033]*y[IDX_CNI];
    IJth(jmatrix, 55, 82) = 0.0 + k[132]*y[IDX_HCNII] + k[887]*y[IDX_CH2I] +
        k[910]*y[IDX_CH3I] + k[930]*y[IDX_CHI];
    IJth(jmatrix, 55, 85) = 0.0 - k[1063]*y[IDX_HCNI] - k[1064]*y[IDX_HCNI]
        - k[1065]*y[IDX_HCNI];
    IJth(jmatrix, 55, 86) = 0.0 - k[814]*y[IDX_HCNI] - k[815]*y[IDX_HCNI];
    IJth(jmatrix, 55, 87) = 0.0 + k[133]*y[IDX_HCNII];
    IJth(jmatrix, 55, 90) = 0.0 - k[632]*y[IDX_HCNI];
    IJth(jmatrix, 55, 91) = 0.0 + k[993]*y[IDX_HI];
    IJth(jmatrix, 55, 92) = 0.0 + k[1090]*y[IDX_CNI] - k[1094]*y[IDX_HCNI] -
        k[1095]*y[IDX_HCNI];
    IJth(jmatrix, 55, 93) = 0.0 - k[841]*y[IDX_HCNI];
    IJth(jmatrix, 55, 108) = 0.0 + k[950]*y[IDX_CNI];
    IJth(jmatrix, 56, 24) = 0.0 - k[385]*y[IDX_HCNII];
    IJth(jmatrix, 56, 25) = 0.0 + k[373]*y[IDX_NH2I] + k[374]*y[IDX_NH3I];
    IJth(jmatrix, 56, 26) = 0.0 - k[463]*y[IDX_HCNII];
    IJth(jmatrix, 56, 27) = 0.0 + k[409]*y[IDX_NH2I];
    IJth(jmatrix, 56, 28) = 0.0 - k[426]*y[IDX_HCNII];
    IJth(jmatrix, 56, 29) = 0.0 + k[736]*y[IDX_NI];
    IJth(jmatrix, 56, 33) = 0.0 - k[454]*y[IDX_HCNII] + k[717]*y[IDX_NII];
    IJth(jmatrix, 56, 35) = 0.0 + k[482]*y[IDX_HNOII] + k[483]*y[IDX_O2HII]
        + k[513]*y[IDX_H2II] + k[581]*y[IDX_H3II] + k[747]*y[IDX_NHII] +
        k[836]*y[IDX_OHII];
    IJth(jmatrix, 56, 36) = 0.0 + k[65]*y[IDX_HCNI] + k[480]*y[IDX_HCOI] +
        k[531]*y[IDX_H2I] + k[560]*y[IDX_H2OI];
    IJth(jmatrix, 56, 37) = 0.0 - k[621]*y[IDX_HCNII];
    IJth(jmatrix, 56, 38) = 0.0 + k[134]*y[IDX_HCNI];
    IJth(jmatrix, 56, 39) = 0.0 - k[620]*y[IDX_HCNII];
    IJth(jmatrix, 56, 40) = 0.0 - k[326]*y[IDX_HCNII];
    IJth(jmatrix, 56, 41) = 0.0 - k[129]*y[IDX_HCNII];
    IJth(jmatrix, 56, 42) = 0.0 + k[81]*y[IDX_HCNI];
    IJth(jmatrix, 56, 43) = 0.0 + k[531]*y[IDX_CNII] - k[535]*y[IDX_HCNII];
    IJth(jmatrix, 56, 44) = 0.0 + k[107]*y[IDX_HCNI] + k[513]*y[IDX_CNI];
    IJth(jmatrix, 56, 46) = 0.0 - k[622]*y[IDX_HCNII];
    IJth(jmatrix, 56, 49) = 0.0 - k[124]*y[IDX_HCNII] + k[560]*y[IDX_CNII] -
        k[565]*y[IDX_HCNII];
    IJth(jmatrix, 56, 52) = 0.0 + k[581]*y[IDX_CNI];
    IJth(jmatrix, 56, 55) = 0.0 + k[65]*y[IDX_CNII] + k[81]*y[IDX_HII] +
        k[107]*y[IDX_H2II] + k[134]*y[IDX_COII] + k[135]*y[IDX_N2II] +
        k[161]*y[IDX_NII] - k[623]*y[IDX_HCNII];
    IJth(jmatrix, 56, 56) = 0.0 - k[124]*y[IDX_H2OI] - k[129]*y[IDX_HI] -
        k[132]*y[IDX_NOI] - k[133]*y[IDX_O2I] - k[196]*y[IDX_NH3I] -
        k[326]*y[IDX_EM] - k[385]*y[IDX_CI] - k[426]*y[IDX_CH2I] -
        k[454]*y[IDX_CH4I] - k[463]*y[IDX_CHI] - k[535]*y[IDX_H2I] -
        k[565]*y[IDX_H2OI] - k[620]*y[IDX_CO2I] - k[621]*y[IDX_COI] -
        k[622]*y[IDX_H2COI] - k[623]*y[IDX_HCNI] - k[624]*y[IDX_HCOI] -
        k[625]*y[IDX_HCOI] - k[626]*y[IDX_HNCI] - k[784]*y[IDX_NH2I] -
        k[793]*y[IDX_NH3I] - k[798]*y[IDX_NHI] - k[853]*y[IDX_OHI] - k[1282];
    IJth(jmatrix, 56, 58) = 0.0 + k[480]*y[IDX_CNII] - k[624]*y[IDX_HCNII] -
        k[625]*y[IDX_HCNII];
    IJth(jmatrix, 56, 64) = 0.0 - k[626]*y[IDX_HCNII];
    IJth(jmatrix, 56, 67) = 0.0 + k[482]*y[IDX_CNI];
    IJth(jmatrix, 56, 71) = 0.0 + k[736]*y[IDX_CH2II];
    IJth(jmatrix, 56, 72) = 0.0 + k[161]*y[IDX_HCNI] + k[717]*y[IDX_CH4I];
    IJth(jmatrix, 56, 74) = 0.0 + k[135]*y[IDX_HCNI];
    IJth(jmatrix, 56, 76) = 0.0 - k[798]*y[IDX_HCNII];
    IJth(jmatrix, 56, 77) = 0.0 + k[747]*y[IDX_CNI];
    IJth(jmatrix, 56, 78) = 0.0 + k[373]*y[IDX_CII] + k[409]*y[IDX_CHII] -
        k[784]*y[IDX_HCNII];
    IJth(jmatrix, 56, 80) = 0.0 - k[196]*y[IDX_HCNII] + k[374]*y[IDX_CII] -
        k[793]*y[IDX_HCNII];
    IJth(jmatrix, 56, 82) = 0.0 - k[132]*y[IDX_HCNII];
    IJth(jmatrix, 56, 87) = 0.0 - k[133]*y[IDX_HCNII];
    IJth(jmatrix, 56, 90) = 0.0 + k[483]*y[IDX_CNI];
    IJth(jmatrix, 56, 92) = 0.0 - k[853]*y[IDX_HCNII];
    IJth(jmatrix, 56, 93) = 0.0 + k[836]*y[IDX_CNI];
    IJth(jmatrix, 57, 26) = 0.0 - k[464]*y[IDX_HCNHII] -
        k[465]*y[IDX_HCNHII];
    IJth(jmatrix, 57, 27) = 0.0 + k[405]*y[IDX_HCNI] + k[407]*y[IDX_HNCI];
    IJth(jmatrix, 57, 28) = 0.0 - k[427]*y[IDX_HCNHII] -
        k[428]*y[IDX_HCNHII];
    IJth(jmatrix, 57, 31) = 0.0 + k[794]*y[IDX_NHI];
    IJth(jmatrix, 57, 33) = 0.0 + k[454]*y[IDX_HCNII] + k[718]*y[IDX_NII];
    IJth(jmatrix, 57, 40) = 0.0 - k[327]*y[IDX_HCNHII] -
        k[328]*y[IDX_HCNHII] - k[329]*y[IDX_HCNHII];
    IJth(jmatrix, 57, 43) = 0.0 + k[535]*y[IDX_HCNII];
    IJth(jmatrix, 57, 46) = 0.0 - k[633]*y[IDX_HCNHII] -
        k[634]*y[IDX_HCNHII];
    IJth(jmatrix, 57, 47) = 0.0 + k[627]*y[IDX_HCNI] + k[646]*y[IDX_HNCI];
    IJth(jmatrix, 57, 50) = 0.0 + k[556]*y[IDX_HCNI] + k[559]*y[IDX_HNCI];
    IJth(jmatrix, 57, 52) = 0.0 + k[587]*y[IDX_HCNI] + k[589]*y[IDX_HNCI];
    IJth(jmatrix, 57, 53) = 0.0 + k[628]*y[IDX_HCNI] + k[647]*y[IDX_HNCI];
    IJth(jmatrix, 57, 54) = 0.0 + k[608]*y[IDX_HCNI] + k[609]*y[IDX_HNCI];
    IJth(jmatrix, 57, 55) = 0.0 + k[405]*y[IDX_CHII] + k[556]*y[IDX_H2OII] +
        k[587]*y[IDX_H3II] + k[608]*y[IDX_H3OII] + k[623]*y[IDX_HCNII] +
        k[627]*y[IDX_H2COII] + k[628]*y[IDX_H3COII] + k[629]*y[IDX_HCOII] +
        k[630]*y[IDX_HNOII] + k[631]*y[IDX_N2HII] + k[632]*y[IDX_O2HII] +
        k[758]*y[IDX_NHII] + k[773]*y[IDX_NH2II] + k[841]*y[IDX_OHII];
    IJth(jmatrix, 57, 56) = 0.0 + k[454]*y[IDX_CH4I] + k[535]*y[IDX_H2I] +
        k[623]*y[IDX_HCNI] + k[625]*y[IDX_HCOI] + k[626]*y[IDX_HNCI] +
        k[793]*y[IDX_NH3I];
    IJth(jmatrix, 57, 57) = 0.0 - k[327]*y[IDX_EM] - k[328]*y[IDX_EM] -
        k[329]*y[IDX_EM] - k[427]*y[IDX_CH2I] - k[428]*y[IDX_CH2I] -
        k[464]*y[IDX_CHI] - k[465]*y[IDX_CHI] - k[633]*y[IDX_H2COI] -
        k[634]*y[IDX_H2COI] - k[785]*y[IDX_NH2I] - k[786]*y[IDX_NH2I] - k[1301];
    IJth(jmatrix, 57, 58) = 0.0 + k[625]*y[IDX_HCNII];
    IJth(jmatrix, 57, 59) = 0.0 + k[629]*y[IDX_HCNI] + k[648]*y[IDX_HNCI];
    IJth(jmatrix, 57, 64) = 0.0 + k[407]*y[IDX_CHII] + k[559]*y[IDX_H2OII] +
        k[589]*y[IDX_H3II] + k[609]*y[IDX_H3OII] + k[626]*y[IDX_HCNII] +
        k[646]*y[IDX_H2COII] + k[647]*y[IDX_H3COII] + k[648]*y[IDX_HCOII] +
        k[649]*y[IDX_HNOII] + k[650]*y[IDX_N2HII] + k[651]*y[IDX_O2HII] +
        k[760]*y[IDX_NHII] + k[775]*y[IDX_NH2II] + k[844]*y[IDX_OHII];
    IJth(jmatrix, 57, 67) = 0.0 + k[630]*y[IDX_HCNI] + k[649]*y[IDX_HNCI];
    IJth(jmatrix, 57, 72) = 0.0 + k[718]*y[IDX_CH4I];
    IJth(jmatrix, 57, 75) = 0.0 + k[631]*y[IDX_HCNI] + k[650]*y[IDX_HNCI];
    IJth(jmatrix, 57, 76) = 0.0 + k[794]*y[IDX_CH3II];
    IJth(jmatrix, 57, 77) = 0.0 + k[758]*y[IDX_HCNI] + k[760]*y[IDX_HNCI];
    IJth(jmatrix, 57, 78) = 0.0 - k[785]*y[IDX_HCNHII] -
        k[786]*y[IDX_HCNHII];
    IJth(jmatrix, 57, 79) = 0.0 + k[773]*y[IDX_HCNI] + k[775]*y[IDX_HNCI];
    IJth(jmatrix, 57, 80) = 0.0 + k[793]*y[IDX_HCNII];
    IJth(jmatrix, 57, 90) = 0.0 + k[632]*y[IDX_HCNI] + k[651]*y[IDX_HNCI];
    IJth(jmatrix, 57, 93) = 0.0 + k[841]*y[IDX_HCNI] + k[844]*y[IDX_HNCI];
    IJth(jmatrix, 58, 24) = 0.0 - k[864]*y[IDX_HCOI];
    IJth(jmatrix, 58, 25) = 0.0 - k[17]*y[IDX_HCOI] + k[366]*y[IDX_CH3OHI] -
        k[372]*y[IDX_HCOI];
    IJth(jmatrix, 58, 26) = 0.0 + k[459]*y[IDX_H2COII] + k[923]*y[IDX_CO2I]
        + k[924]*y[IDX_H2COI] - k[925]*y[IDX_HCOI] + k[931]*y[IDX_NOI] +
        k[936]*y[IDX_O2I] + k[937]*y[IDX_O2HI] + k[941]*y[IDX_OHI];
    IJth(jmatrix, 58, 27) = 0.0 - k[31]*y[IDX_HCOI] - k[406]*y[IDX_HCOI] +
        k[413]*y[IDX_O2I];
    IJth(jmatrix, 58, 28) = 0.0 + k[423]*y[IDX_H2COII] + k[881]*y[IDX_H2COI]
        - k[882]*y[IDX_HCOI] + k[893]*y[IDX_O2I] + k[896]*y[IDX_OI];
    IJth(jmatrix, 58, 29) = 0.0 - k[419]*y[IDX_HCOI];
    IJth(jmatrix, 58, 30) = 0.0 + k[903]*y[IDX_H2COI] - k[905]*y[IDX_HCOI] +
        k[912]*y[IDX_O2I];
    IJth(jmatrix, 58, 31) = 0.0 - k[46]*y[IDX_HCOI] - k[441]*y[IDX_HCOI];
    IJth(jmatrix, 58, 32) = 0.0 + k[366]*y[IDX_CII];
    IJth(jmatrix, 58, 35) = 0.0 + k[942]*y[IDX_H2COI] - k[943]*y[IDX_HCOI];
    IJth(jmatrix, 58, 36) = 0.0 - k[66]*y[IDX_HCOI] - k[480]*y[IDX_HCOI];
    IJth(jmatrix, 58, 38) = 0.0 - k[71]*y[IDX_HCOI] + k[484]*y[IDX_H2COI];
    IJth(jmatrix, 58, 39) = 0.0 + k[750]*y[IDX_NHII] + k[923]*y[IDX_CHI];
    IJth(jmatrix, 58, 40) = 0.0 + k[309]*y[IDX_H2COII] +
        k[321]*y[IDX_H3COII];
    IJth(jmatrix, 58, 41) = 0.0 + k[974]*y[IDX_H2COI] - k[977]*y[IDX_HCOI] -
        k[978]*y[IDX_HCOI];
    IJth(jmatrix, 58, 42) = 0.0 - k[82]*y[IDX_HCOI] - k[500]*y[IDX_HCOI] -
        k[501]*y[IDX_HCOI];
    IJth(jmatrix, 58, 44) = 0.0 - k[108]*y[IDX_HCOI] - k[519]*y[IDX_HCOI];
    IJth(jmatrix, 58, 46) = 0.0 + k[484]*y[IDX_COII] + k[548]*y[IDX_H2COII]
        + k[770]*y[IDX_NH2II] + k[881]*y[IDX_CH2I] + k[903]*y[IDX_CH3I] +
        k[924]*y[IDX_CHI] + k[942]*y[IDX_CNI] + k[974]*y[IDX_HI] +
        k[1061]*y[IDX_OI] + k[1093]*y[IDX_OHI];
    IJth(jmatrix, 58, 47) = 0.0 - k[136]*y[IDX_HCOI] + k[309]*y[IDX_EM] +
        k[423]*y[IDX_CH2I] + k[459]*y[IDX_CHI] + k[548]*y[IDX_H2COI] +
        k[563]*y[IDX_H2OI] + k[627]*y[IDX_HCNI] - k[641]*y[IDX_HCOI] +
        k[646]*y[IDX_HNCI] + k[780]*y[IDX_NH2I];
    IJth(jmatrix, 58, 49) = 0.0 + k[563]*y[IDX_H2COII];
    IJth(jmatrix, 58, 50) = 0.0 - k[118]*y[IDX_HCOI] - k[557]*y[IDX_HCOI] -
        k[558]*y[IDX_HCOI];
    IJth(jmatrix, 58, 52) = 0.0 - k[588]*y[IDX_HCOI];
    IJth(jmatrix, 58, 53) = 0.0 + k[321]*y[IDX_EM];
    IJth(jmatrix, 58, 55) = 0.0 + k[627]*y[IDX_H2COII];
    IJth(jmatrix, 58, 56) = 0.0 - k[624]*y[IDX_HCOI] - k[625]*y[IDX_HCOI];
    IJth(jmatrix, 58, 58) = 0.0 - k[17]*y[IDX_CII] - k[31]*y[IDX_CHII] -
        k[46]*y[IDX_CH3II] - k[66]*y[IDX_CNII] - k[71]*y[IDX_COII] -
        k[82]*y[IDX_HII] - k[108]*y[IDX_H2II] - k[118]*y[IDX_H2OII] -
        k[136]*y[IDX_H2COII] - k[137]*y[IDX_O2II] - k[138]*y[IDX_SiOII] -
        k[162]*y[IDX_NII] - k[171]*y[IDX_N2II] - k[180]*y[IDX_NH2II] -
        k[189]*y[IDX_NH3II] - k[211]*y[IDX_OII] - k[221]*y[IDX_OHII] - k[260] -
        k[261] - k[372]*y[IDX_CII] - k[406]*y[IDX_CHII] - k[419]*y[IDX_CH2II] -
        k[441]*y[IDX_CH3II] - k[480]*y[IDX_CNII] - k[500]*y[IDX_HII] -
        k[501]*y[IDX_HII] - k[519]*y[IDX_H2II] - k[557]*y[IDX_H2OII] -
        k[558]*y[IDX_H2OII] - k[588]*y[IDX_H3II] - k[624]*y[IDX_HCNII] -
        k[625]*y[IDX_HCNII] - k[636]*y[IDX_HCOII] - k[641]*y[IDX_H2COII] -
        k[642]*y[IDX_HNOII] - k[643]*y[IDX_N2HII] - k[644]*y[IDX_O2II] -
        k[645]*y[IDX_O2HII] - k[680]*y[IDX_HeII] - k[681]*y[IDX_HeII] -
        k[682]*y[IDX_HeII] - k[723]*y[IDX_NII] - k[731]*y[IDX_N2II] -
        k[759]*y[IDX_NHII] - k[774]*y[IDX_NH2II] - k[816]*y[IDX_OII] -
        k[842]*y[IDX_OHII] - k[843]*y[IDX_OHII] - k[864]*y[IDX_CI] -
        k[882]*y[IDX_CH2I] - k[905]*y[IDX_CH3I] - k[925]*y[IDX_CHI] -
        k[943]*y[IDX_CNI] - k[977]*y[IDX_HI] - k[978]*y[IDX_HI] -
        k[997]*y[IDX_HCOI] - k[997]*y[IDX_HCOI] - k[997]*y[IDX_HCOI] -
        k[997]*y[IDX_HCOI] - k[998]*y[IDX_HCOI] - k[998]*y[IDX_HCOI] -
        k[998]*y[IDX_HCOI] - k[998]*y[IDX_HCOI] - k[999]*y[IDX_HNOI] -
        k[1000]*y[IDX_NOI] - k[1001]*y[IDX_O2I] - k[1002]*y[IDX_O2I] -
        k[1003]*y[IDX_O2HI] - k[1014]*y[IDX_NI] - k[1015]*y[IDX_NI] -
        k[1016]*y[IDX_NI] - k[1066]*y[IDX_OI] - k[1067]*y[IDX_OI] -
        k[1096]*y[IDX_OHI] - k[1149] - k[1150] - k[1261];
    IJth(jmatrix, 58, 59) = 0.0 + k[149]*y[IDX_MgI] - k[636]*y[IDX_HCOI];
    IJth(jmatrix, 58, 62) = 0.0 - k[680]*y[IDX_HCOI] - k[681]*y[IDX_HCOI] -
        k[682]*y[IDX_HCOI];
    IJth(jmatrix, 58, 64) = 0.0 + k[646]*y[IDX_H2COII];
    IJth(jmatrix, 58, 66) = 0.0 - k[999]*y[IDX_HCOI];
    IJth(jmatrix, 58, 67) = 0.0 - k[642]*y[IDX_HCOI];
    IJth(jmatrix, 58, 69) = 0.0 + k[149]*y[IDX_HCOII];
    IJth(jmatrix, 58, 71) = 0.0 - k[1014]*y[IDX_HCOI] - k[1015]*y[IDX_HCOI]
        - k[1016]*y[IDX_HCOI];
    IJth(jmatrix, 58, 72) = 0.0 - k[162]*y[IDX_HCOI] - k[723]*y[IDX_HCOI];
    IJth(jmatrix, 58, 74) = 0.0 - k[171]*y[IDX_HCOI] - k[731]*y[IDX_HCOI];
    IJth(jmatrix, 58, 75) = 0.0 - k[643]*y[IDX_HCOI];
    IJth(jmatrix, 58, 77) = 0.0 + k[750]*y[IDX_CO2I] - k[759]*y[IDX_HCOI];
    IJth(jmatrix, 58, 78) = 0.0 + k[780]*y[IDX_H2COII];
    IJth(jmatrix, 58, 79) = 0.0 - k[180]*y[IDX_HCOI] + k[770]*y[IDX_H2COI] -
        k[774]*y[IDX_HCOI];
    IJth(jmatrix, 58, 81) = 0.0 - k[189]*y[IDX_HCOI];
    IJth(jmatrix, 58, 82) = 0.0 + k[931]*y[IDX_CHI] - k[1000]*y[IDX_HCOI];
    IJth(jmatrix, 58, 85) = 0.0 + k[896]*y[IDX_CH2I] + k[1061]*y[IDX_H2COI]
        - k[1066]*y[IDX_HCOI] - k[1067]*y[IDX_HCOI];
    IJth(jmatrix, 58, 86) = 0.0 - k[211]*y[IDX_HCOI] - k[816]*y[IDX_HCOI];
    IJth(jmatrix, 58, 87) = 0.0 + k[413]*y[IDX_CHII] + k[893]*y[IDX_CH2I] +
        k[912]*y[IDX_CH3I] + k[936]*y[IDX_CHI] - k[1001]*y[IDX_HCOI] -
        k[1002]*y[IDX_HCOI];
    IJth(jmatrix, 58, 88) = 0.0 - k[137]*y[IDX_HCOI] - k[644]*y[IDX_HCOI];
    IJth(jmatrix, 58, 89) = 0.0 + k[937]*y[IDX_CHI] - k[1003]*y[IDX_HCOI];
    IJth(jmatrix, 58, 90) = 0.0 - k[645]*y[IDX_HCOI];
    IJth(jmatrix, 58, 92) = 0.0 + k[941]*y[IDX_CHI] + k[1093]*y[IDX_H2COI] -
        k[1096]*y[IDX_HCOI];
    IJth(jmatrix, 58, 93) = 0.0 - k[221]*y[IDX_HCOI] - k[842]*y[IDX_HCOI] -
        k[843]*y[IDX_HCOI];
    IJth(jmatrix, 58, 112) = 0.0 - k[138]*y[IDX_HCOI];
    IJth(jmatrix, 59, 24) = 0.0 + k[384]*y[IDX_H3OII] - k[386]*y[IDX_HCOII];
    IJth(jmatrix, 59, 25) = 0.0 + k[17]*y[IDX_HCOI] + k[369]*y[IDX_H2COI] +
        k[370]*y[IDX_H2OI];
    IJth(jmatrix, 59, 26) = 0.0 + k[0]*y[IDX_OI] + k[458]*y[IDX_COII] -
        k[466]*y[IDX_HCOII] + k[473]*y[IDX_O2II] + k[478]*y[IDX_SiOII];
    IJth(jmatrix, 59, 27) = 0.0 + k[31]*y[IDX_HCOI] + k[398]*y[IDX_CO2I] +
        k[401]*y[IDX_H2COI] + k[404]*y[IDX_H2OI] + k[412]*y[IDX_O2I];
    IJth(jmatrix, 59, 28) = 0.0 + k[422]*y[IDX_COII] - k[429]*y[IDX_HCOII];
    IJth(jmatrix, 59, 29) = 0.0 + k[417]*y[IDX_H2COI] + k[420]*y[IDX_O2I] +
        k[421]*y[IDX_OI];
    IJth(jmatrix, 59, 31) = 0.0 + k[46]*y[IDX_HCOI] + k[440]*y[IDX_H2COI] +
        k[444]*y[IDX_OI];
    IJth(jmatrix, 59, 32) = 0.0 + k[494]*y[IDX_HII];
    IJth(jmatrix, 59, 33) = 0.0 + k[451]*y[IDX_COII];
    IJth(jmatrix, 59, 34) = 0.0 + k[448]*y[IDX_COI];
    IJth(jmatrix, 59, 36) = 0.0 + k[66]*y[IDX_HCOI] + k[479]*y[IDX_H2COI] +
        k[561]*y[IDX_H2OI];
    IJth(jmatrix, 59, 37) = 0.0 + k[448]*y[IDX_CH4II] + k[485]*y[IDX_HCO2II]
        + k[486]*y[IDX_HNOII] + k[487]*y[IDX_N2HII] + k[488]*y[IDX_O2HII] +
        k[489]*y[IDX_SiH4II] + k[515]*y[IDX_H2II] + k[553]*y[IDX_H2OII] +
        k[583]*y[IDX_H3II] + k[621]*y[IDX_HCNII] + k[751]*y[IDX_NHII] +
        k[838]*y[IDX_OHII];
    IJth(jmatrix, 59, 38) = 0.0 + k[71]*y[IDX_HCOI] + k[422]*y[IDX_CH2I] +
        k[451]*y[IDX_CH4I] + k[458]*y[IDX_CHI] + k[484]*y[IDX_H2COI] +
        k[532]*y[IDX_H2I] + k[562]*y[IDX_H2OI] + k[779]*y[IDX_NH2I] +
        k[792]*y[IDX_NH3I] + k[795]*y[IDX_NHI] + k[851]*y[IDX_OHI];
    IJth(jmatrix, 59, 39) = 0.0 + k[398]*y[IDX_CHII] + k[496]*y[IDX_HII];
    IJth(jmatrix, 59, 40) = 0.0 - k[330]*y[IDX_HCOII];
    IJth(jmatrix, 59, 42) = 0.0 + k[82]*y[IDX_HCOI] + k[494]*y[IDX_CH3OHI] +
        k[496]*y[IDX_CO2I] + k[498]*y[IDX_H2COI];
    IJth(jmatrix, 59, 43) = 0.0 + k[5]*y[IDX_HOCII] + k[532]*y[IDX_COII];
    IJth(jmatrix, 59, 44) = 0.0 + k[108]*y[IDX_HCOI] + k[515]*y[IDX_COI] +
        k[517]*y[IDX_H2COI];
    IJth(jmatrix, 59, 46) = 0.0 + k[369]*y[IDX_CII] + k[401]*y[IDX_CHII] +
        k[417]*y[IDX_CH2II] + k[440]*y[IDX_CH3II] + k[479]*y[IDX_CNII] +
        k[484]*y[IDX_COII] + k[498]*y[IDX_HII] + k[517]*y[IDX_H2II] +
        k[551]*y[IDX_O2II] - k[635]*y[IDX_HCOII] + k[671]*y[IDX_HeII] +
        k[721]*y[IDX_NII] + k[730]*y[IDX_N2II] + k[753]*y[IDX_NHII] +
        k[813]*y[IDX_OII] + k[1139];
    IJth(jmatrix, 59, 47) = 0.0 + k[136]*y[IDX_HCOI] + k[549]*y[IDX_O2I];
    IJth(jmatrix, 59, 49) = 0.0 + k[370]*y[IDX_CII] + k[404]*y[IDX_CHII] +
        k[561]*y[IDX_CNII] + k[562]*y[IDX_COII] - k[566]*y[IDX_HCOII];
    IJth(jmatrix, 59, 50) = 0.0 + k[118]*y[IDX_HCOI] + k[553]*y[IDX_COI];
    IJth(jmatrix, 59, 52) = 0.0 + k[583]*y[IDX_COI];
    IJth(jmatrix, 59, 54) = 0.0 + k[384]*y[IDX_CI];
    IJth(jmatrix, 59, 55) = 0.0 - k[629]*y[IDX_HCOII] + k[814]*y[IDX_OII];
    IJth(jmatrix, 59, 56) = 0.0 + k[621]*y[IDX_COI];
    IJth(jmatrix, 59, 58) = 0.0 + k[17]*y[IDX_CII] + k[31]*y[IDX_CHII] +
        k[46]*y[IDX_CH3II] + k[66]*y[IDX_CNII] + k[71]*y[IDX_COII] +
        k[82]*y[IDX_HII] + k[108]*y[IDX_H2II] + k[118]*y[IDX_H2OII] +
        k[136]*y[IDX_H2COII] + k[137]*y[IDX_O2II] + k[138]*y[IDX_SiOII] +
        k[162]*y[IDX_NII] + k[171]*y[IDX_N2II] + k[180]*y[IDX_NH2II] +
        k[189]*y[IDX_NH3II] + k[211]*y[IDX_OII] + k[221]*y[IDX_OHII] + k[261] -
        k[636]*y[IDX_HCOII] + k[1150];
    IJth(jmatrix, 59, 59) = 0.0 - k[149]*y[IDX_MgI] - k[330]*y[IDX_EM] -
        k[386]*y[IDX_CI] - k[429]*y[IDX_CH2I] - k[466]*y[IDX_CHI] -
        k[566]*y[IDX_H2OI] - k[629]*y[IDX_HCNI] - k[635]*y[IDX_H2COI] -
        k[636]*y[IDX_HCOI] - k[637]*y[IDX_SiH2I] - k[638]*y[IDX_SiH4I] -
        k[639]*y[IDX_SiHI] - k[640]*y[IDX_SiOI] - k[648]*y[IDX_HNCI] -
        k[787]*y[IDX_NH2I] - k[799]*y[IDX_NHI] - k[854]*y[IDX_OHI] -
        k[855]*y[IDX_OHI] - k[861]*y[IDX_SiI] - k[1148] - k[1281];
    IJth(jmatrix, 59, 60) = 0.0 + k[485]*y[IDX_COI] + k[824]*y[IDX_OI];
    IJth(jmatrix, 59, 62) = 0.0 + k[671]*y[IDX_H2COI];
    IJth(jmatrix, 59, 64) = 0.0 - k[648]*y[IDX_HCOII];
    IJth(jmatrix, 59, 67) = 0.0 + k[486]*y[IDX_COI];
    IJth(jmatrix, 59, 68) = 0.0 + k[5]*y[IDX_H2I];
    IJth(jmatrix, 59, 69) = 0.0 - k[149]*y[IDX_HCOII];
    IJth(jmatrix, 59, 72) = 0.0 + k[162]*y[IDX_HCOI] + k[721]*y[IDX_H2COI];
    IJth(jmatrix, 59, 74) = 0.0 + k[171]*y[IDX_HCOI] + k[730]*y[IDX_H2COI];
    IJth(jmatrix, 59, 75) = 0.0 + k[487]*y[IDX_COI];
    IJth(jmatrix, 59, 76) = 0.0 + k[795]*y[IDX_COII] - k[799]*y[IDX_HCOII];
    IJth(jmatrix, 59, 77) = 0.0 + k[751]*y[IDX_COI] + k[753]*y[IDX_H2COI];
    IJth(jmatrix, 59, 78) = 0.0 + k[779]*y[IDX_COII] - k[787]*y[IDX_HCOII];
    IJth(jmatrix, 59, 79) = 0.0 + k[180]*y[IDX_HCOI];
    IJth(jmatrix, 59, 80) = 0.0 + k[792]*y[IDX_COII];
    IJth(jmatrix, 59, 81) = 0.0 + k[189]*y[IDX_HCOI];
    IJth(jmatrix, 59, 85) = 0.0 + k[0]*y[IDX_CHI] + k[421]*y[IDX_CH2II] +
        k[444]*y[IDX_CH3II] + k[824]*y[IDX_HCO2II];
    IJth(jmatrix, 59, 86) = 0.0 + k[211]*y[IDX_HCOI] + k[813]*y[IDX_H2COI] +
        k[814]*y[IDX_HCNI];
    IJth(jmatrix, 59, 87) = 0.0 + k[412]*y[IDX_CHII] + k[420]*y[IDX_CH2II] +
        k[549]*y[IDX_H2COII];
    IJth(jmatrix, 59, 88) = 0.0 + k[137]*y[IDX_HCOI] + k[473]*y[IDX_CHI] +
        k[551]*y[IDX_H2COI];
    IJth(jmatrix, 59, 90) = 0.0 + k[488]*y[IDX_COI];
    IJth(jmatrix, 59, 92) = 0.0 + k[851]*y[IDX_COII] - k[854]*y[IDX_HCOII] -
        k[855]*y[IDX_HCOII];
    IJth(jmatrix, 59, 93) = 0.0 + k[221]*y[IDX_HCOI] + k[838]*y[IDX_COI];
    IJth(jmatrix, 59, 94) = 0.0 - k[861]*y[IDX_HCOII];
    IJth(jmatrix, 59, 102) = 0.0 - k[639]*y[IDX_HCOII];
    IJth(jmatrix, 59, 104) = 0.0 - k[637]*y[IDX_HCOII];
    IJth(jmatrix, 59, 108) = 0.0 - k[638]*y[IDX_HCOII];
    IJth(jmatrix, 59, 109) = 0.0 + k[489]*y[IDX_COI];
    IJth(jmatrix, 59, 111) = 0.0 - k[640]*y[IDX_HCOII];
    IJth(jmatrix, 59, 112) = 0.0 + k[138]*y[IDX_HCOI] + k[478]*y[IDX_CHI];
    IJth(jmatrix, 60, 24) = 0.0 - k[387]*y[IDX_HCO2II];
    IJth(jmatrix, 60, 34) = 0.0 + k[447]*y[IDX_CO2I];
    IJth(jmatrix, 60, 37) = 0.0 - k[485]*y[IDX_HCO2II];
    IJth(jmatrix, 60, 39) = 0.0 + k[447]*y[IDX_CH4II] + k[514]*y[IDX_H2II] +
        k[582]*y[IDX_H3II] + k[620]*y[IDX_HCNII] + k[652]*y[IDX_HNOII] +
        k[734]*y[IDX_N2HII] + k[748]*y[IDX_NHII] + k[821]*y[IDX_O2HII] +
        k[837]*y[IDX_OHII];
    IJth(jmatrix, 60, 40) = 0.0 - k[331]*y[IDX_HCO2II] -
        k[332]*y[IDX_HCO2II] - k[333]*y[IDX_HCO2II];
    IJth(jmatrix, 60, 44) = 0.0 + k[514]*y[IDX_CO2I];
    IJth(jmatrix, 60, 49) = 0.0 - k[567]*y[IDX_HCO2II];
    IJth(jmatrix, 60, 52) = 0.0 + k[582]*y[IDX_CO2I];
    IJth(jmatrix, 60, 56) = 0.0 + k[620]*y[IDX_CO2I];
    IJth(jmatrix, 60, 59) = 0.0 + k[855]*y[IDX_OHI];
    IJth(jmatrix, 60, 60) = 0.0 - k[331]*y[IDX_EM] - k[332]*y[IDX_EM] -
        k[333]*y[IDX_EM] - k[387]*y[IDX_CI] - k[485]*y[IDX_COI] -
        k[567]*y[IDX_H2OI] - k[824]*y[IDX_OI] - k[1287];
    IJth(jmatrix, 60, 67) = 0.0 + k[652]*y[IDX_CO2I];
    IJth(jmatrix, 60, 75) = 0.0 + k[734]*y[IDX_CO2I];
    IJth(jmatrix, 60, 77) = 0.0 + k[748]*y[IDX_CO2I];
    IJth(jmatrix, 60, 85) = 0.0 - k[824]*y[IDX_HCO2II];
    IJth(jmatrix, 60, 90) = 0.0 + k[821]*y[IDX_CO2I];
    IJth(jmatrix, 60, 92) = 0.0 + k[855]*y[IDX_HCOII];
    IJth(jmatrix, 60, 93) = 0.0 + k[837]*y[IDX_CO2I];
    IJth(jmatrix, 61, 24) = 0.0 + k[139]*y[IDX_HeII];
    IJth(jmatrix, 61, 26) = 0.0 + k[141]*y[IDX_HeII] + k[662]*y[IDX_HeII];
    IJth(jmatrix, 61, 28) = 0.0 + k[653]*y[IDX_HeII] + k[654]*y[IDX_HeII];
    IJth(jmatrix, 61, 30) = 0.0 + k[655]*y[IDX_HeII];
    IJth(jmatrix, 61, 32) = 0.0 + k[656]*y[IDX_HeII] + k[657]*y[IDX_HeII];
    IJth(jmatrix, 61, 33) = 0.0 + k[140]*y[IDX_HeII] + k[658]*y[IDX_HeII] +
        k[659]*y[IDX_HeII] + k[660]*y[IDX_HeII] + k[661]*y[IDX_HeII];
    IJth(jmatrix, 61, 35) = 0.0 + k[663]*y[IDX_HeII] + k[664]*y[IDX_HeII];
    IJth(jmatrix, 61, 37) = 0.0 + k[669]*y[IDX_HeII];
    IJth(jmatrix, 61, 39) = 0.0 + k[665]*y[IDX_HeII] + k[666]*y[IDX_HeII] +
        k[667]*y[IDX_HeII] + k[668]*y[IDX_HeII];
    IJth(jmatrix, 61, 40) = 0.0 + k[336]*y[IDX_HeHII] + k[1218]*y[IDX_HeII];
    IJth(jmatrix, 61, 41) = 0.0 + k[130]*y[IDX_HeII] + k[618]*y[IDX_HeHII];
    IJth(jmatrix, 61, 42) = 0.0 - k[1198]*y[IDX_HeI];
    IJth(jmatrix, 61, 43) = 0.0 + k[115]*y[IDX_HeII] + k[536]*y[IDX_HeII] +
        k[537]*y[IDX_HeHII];
    IJth(jmatrix, 61, 44) = 0.0 - k[520]*y[IDX_HeI];
    IJth(jmatrix, 61, 46) = 0.0 + k[142]*y[IDX_HeII] + k[670]*y[IDX_HeII] +
        k[671]*y[IDX_HeII] + k[672]*y[IDX_HeII];
    IJth(jmatrix, 61, 49) = 0.0 + k[143]*y[IDX_HeII] + k[673]*y[IDX_HeII] +
        k[674]*y[IDX_HeII];
    IJth(jmatrix, 61, 51) = 0.0 + k[675]*y[IDX_HeII];
    IJth(jmatrix, 61, 55) = 0.0 + k[676]*y[IDX_HeII] + k[677]*y[IDX_HeII] +
        k[678]*y[IDX_HeII] + k[679]*y[IDX_HeII];
    IJth(jmatrix, 61, 58) = 0.0 + k[680]*y[IDX_HeII] + k[682]*y[IDX_HeII];
    IJth(jmatrix, 61, 61) = 0.0 - k[237] - k[265] - k[520]*y[IDX_H2II] -
        k[1198]*y[IDX_HII];
    IJth(jmatrix, 61, 62) = 0.0 + k[115]*y[IDX_H2I] + k[130]*y[IDX_HI] +
        k[139]*y[IDX_CI] + k[140]*y[IDX_CH4I] + k[141]*y[IDX_CHI] +
        k[142]*y[IDX_H2COI] + k[143]*y[IDX_H2OI] + k[144]*y[IDX_N2I] +
        k[145]*y[IDX_NH3I] + k[146]*y[IDX_O2I] + k[147]*y[IDX_SiI] +
        k[536]*y[IDX_H2I] + k[653]*y[IDX_CH2I] + k[654]*y[IDX_CH2I] +
        k[655]*y[IDX_CH3I] + k[656]*y[IDX_CH3OHI] + k[657]*y[IDX_CH3OHI] +
        k[658]*y[IDX_CH4I] + k[659]*y[IDX_CH4I] + k[660]*y[IDX_CH4I] +
        k[661]*y[IDX_CH4I] + k[662]*y[IDX_CHI] + k[663]*y[IDX_CNI] +
        k[664]*y[IDX_CNI] + k[665]*y[IDX_CO2I] + k[666]*y[IDX_CO2I] +
        k[667]*y[IDX_CO2I] + k[668]*y[IDX_CO2I] + k[669]*y[IDX_COI] +
        k[670]*y[IDX_H2COI] + k[671]*y[IDX_H2COI] + k[672]*y[IDX_H2COI] +
        k[673]*y[IDX_H2OI] + k[674]*y[IDX_H2OI] + k[675]*y[IDX_H2SiOI] +
        k[676]*y[IDX_HCNI] + k[677]*y[IDX_HCNI] + k[678]*y[IDX_HCNI] +
        k[679]*y[IDX_HCNI] + k[680]*y[IDX_HCOI] + k[682]*y[IDX_HCOI] +
        k[683]*y[IDX_HNCI] + k[684]*y[IDX_HNCI] + k[685]*y[IDX_HNCI] +
        k[686]*y[IDX_HNOI] + k[687]*y[IDX_HNOI] + k[688]*y[IDX_N2I] +
        k[689]*y[IDX_NH2I] + k[690]*y[IDX_NH2I] + k[691]*y[IDX_NH3I] +
        k[692]*y[IDX_NH3I] + k[693]*y[IDX_NHI] + k[694]*y[IDX_NOI] +
        k[695]*y[IDX_NOI] + k[696]*y[IDX_O2I] + k[697]*y[IDX_OCNI] +
        k[698]*y[IDX_OCNI] + k[699]*y[IDX_OHI] + k[700]*y[IDX_SiC3I] +
        k[701]*y[IDX_SiCI] + k[702]*y[IDX_SiCI] + k[703]*y[IDX_SiH2I] +
        k[704]*y[IDX_SiH2I] + k[705]*y[IDX_SiH3I] + k[706]*y[IDX_SiH3I] +
        k[707]*y[IDX_SiH4I] + k[708]*y[IDX_SiH4I] + k[709]*y[IDX_SiHI] +
        k[710]*y[IDX_SiOI] + k[711]*y[IDX_SiOI] + k[1218]*y[IDX_EM];
    IJth(jmatrix, 61, 63) = 0.0 + k[336]*y[IDX_EM] + k[537]*y[IDX_H2I] +
        k[618]*y[IDX_HI];
    IJth(jmatrix, 61, 64) = 0.0 + k[683]*y[IDX_HeII] + k[684]*y[IDX_HeII] +
        k[685]*y[IDX_HeII];
    IJth(jmatrix, 61, 66) = 0.0 + k[686]*y[IDX_HeII] + k[687]*y[IDX_HeII];
    IJth(jmatrix, 61, 73) = 0.0 + k[144]*y[IDX_HeII] + k[688]*y[IDX_HeII];
    IJth(jmatrix, 61, 76) = 0.0 + k[693]*y[IDX_HeII];
    IJth(jmatrix, 61, 78) = 0.0 + k[689]*y[IDX_HeII] + k[690]*y[IDX_HeII];
    IJth(jmatrix, 61, 80) = 0.0 + k[145]*y[IDX_HeII] + k[691]*y[IDX_HeII] +
        k[692]*y[IDX_HeII];
    IJth(jmatrix, 61, 82) = 0.0 + k[694]*y[IDX_HeII] + k[695]*y[IDX_HeII];
    IJth(jmatrix, 61, 87) = 0.0 + k[146]*y[IDX_HeII] + k[696]*y[IDX_HeII];
    IJth(jmatrix, 61, 91) = 0.0 + k[697]*y[IDX_HeII] + k[698]*y[IDX_HeII];
    IJth(jmatrix, 61, 92) = 0.0 + k[699]*y[IDX_HeII];
    IJth(jmatrix, 61, 94) = 0.0 + k[147]*y[IDX_HeII];
    IJth(jmatrix, 61, 96) = 0.0 + k[701]*y[IDX_HeII] + k[702]*y[IDX_HeII];
    IJth(jmatrix, 61, 100) = 0.0 + k[700]*y[IDX_HeII];
    IJth(jmatrix, 61, 102) = 0.0 + k[709]*y[IDX_HeII];
    IJth(jmatrix, 61, 104) = 0.0 + k[703]*y[IDX_HeII] + k[704]*y[IDX_HeII];
    IJth(jmatrix, 61, 106) = 0.0 + k[705]*y[IDX_HeII] + k[706]*y[IDX_HeII];
    IJth(jmatrix, 61, 108) = 0.0 + k[707]*y[IDX_HeII] + k[708]*y[IDX_HeII];
    IJth(jmatrix, 61, 111) = 0.0 + k[710]*y[IDX_HeII] + k[711]*y[IDX_HeII];
    IJth(jmatrix, 62, 24) = 0.0 - k[139]*y[IDX_HeII];
    IJth(jmatrix, 62, 26) = 0.0 - k[141]*y[IDX_HeII] - k[662]*y[IDX_HeII];
    IJth(jmatrix, 62, 28) = 0.0 - k[653]*y[IDX_HeII] - k[654]*y[IDX_HeII];
    IJth(jmatrix, 62, 30) = 0.0 - k[655]*y[IDX_HeII];
    IJth(jmatrix, 62, 32) = 0.0 - k[656]*y[IDX_HeII] - k[657]*y[IDX_HeII];
    IJth(jmatrix, 62, 33) = 0.0 - k[140]*y[IDX_HeII] - k[658]*y[IDX_HeII] -
        k[659]*y[IDX_HeII] - k[660]*y[IDX_HeII] - k[661]*y[IDX_HeII];
    IJth(jmatrix, 62, 35) = 0.0 - k[663]*y[IDX_HeII] - k[664]*y[IDX_HeII];
    IJth(jmatrix, 62, 37) = 0.0 - k[669]*y[IDX_HeII];
    IJth(jmatrix, 62, 39) = 0.0 - k[665]*y[IDX_HeII] - k[666]*y[IDX_HeII] -
        k[667]*y[IDX_HeII] - k[668]*y[IDX_HeII];
    IJth(jmatrix, 62, 40) = 0.0 - k[1218]*y[IDX_HeII];
    IJth(jmatrix, 62, 41) = 0.0 - k[130]*y[IDX_HeII];
    IJth(jmatrix, 62, 43) = 0.0 - k[115]*y[IDX_HeII] - k[536]*y[IDX_HeII];
    IJth(jmatrix, 62, 46) = 0.0 - k[142]*y[IDX_HeII] - k[670]*y[IDX_HeII] -
        k[671]*y[IDX_HeII] - k[672]*y[IDX_HeII];
    IJth(jmatrix, 62, 49) = 0.0 - k[143]*y[IDX_HeII] - k[673]*y[IDX_HeII] -
        k[674]*y[IDX_HeII];
    IJth(jmatrix, 62, 51) = 0.0 - k[675]*y[IDX_HeII];
    IJth(jmatrix, 62, 55) = 0.0 - k[676]*y[IDX_HeII] - k[677]*y[IDX_HeII] -
        k[678]*y[IDX_HeII] - k[679]*y[IDX_HeII];
    IJth(jmatrix, 62, 58) = 0.0 - k[680]*y[IDX_HeII] - k[681]*y[IDX_HeII] -
        k[682]*y[IDX_HeII];
    IJth(jmatrix, 62, 61) = 0.0 + k[237] + k[265];
    IJth(jmatrix, 62, 62) = 0.0 - k[115]*y[IDX_H2I] - k[130]*y[IDX_HI] -
        k[139]*y[IDX_CI] - k[140]*y[IDX_CH4I] - k[141]*y[IDX_CHI] -
        k[142]*y[IDX_H2COI] - k[143]*y[IDX_H2OI] - k[144]*y[IDX_N2I] -
        k[145]*y[IDX_NH3I] - k[146]*y[IDX_O2I] - k[147]*y[IDX_SiI] -
        k[536]*y[IDX_H2I] - k[653]*y[IDX_CH2I] - k[654]*y[IDX_CH2I] -
        k[655]*y[IDX_CH3I] - k[656]*y[IDX_CH3OHI] - k[657]*y[IDX_CH3OHI] -
        k[658]*y[IDX_CH4I] - k[659]*y[IDX_CH4I] - k[660]*y[IDX_CH4I] -
        k[661]*y[IDX_CH4I] - k[662]*y[IDX_CHI] - k[663]*y[IDX_CNI] -
        k[664]*y[IDX_CNI] - k[665]*y[IDX_CO2I] - k[666]*y[IDX_CO2I] -
        k[667]*y[IDX_CO2I] - k[668]*y[IDX_CO2I] - k[669]*y[IDX_COI] -
        k[670]*y[IDX_H2COI] - k[671]*y[IDX_H2COI] - k[672]*y[IDX_H2COI] -
        k[673]*y[IDX_H2OI] - k[674]*y[IDX_H2OI] - k[675]*y[IDX_H2SiOI] -
        k[676]*y[IDX_HCNI] - k[677]*y[IDX_HCNI] - k[678]*y[IDX_HCNI] -
        k[679]*y[IDX_HCNI] - k[680]*y[IDX_HCOI] - k[681]*y[IDX_HCOI] -
        k[682]*y[IDX_HCOI] - k[683]*y[IDX_HNCI] - k[684]*y[IDX_HNCI] -
        k[685]*y[IDX_HNCI] - k[686]*y[IDX_HNOI] - k[687]*y[IDX_HNOI] -
        k[688]*y[IDX_N2I] - k[689]*y[IDX_NH2I] - k[690]*y[IDX_NH2I] -
        k[691]*y[IDX_NH3I] - k[692]*y[IDX_NH3I] - k[693]*y[IDX_NHI] -
        k[694]*y[IDX_NOI] - k[695]*y[IDX_NOI] - k[696]*y[IDX_O2I] -
        k[697]*y[IDX_OCNI] - k[698]*y[IDX_OCNI] - k[699]*y[IDX_OHI] -
        k[700]*y[IDX_SiC3I] - k[701]*y[IDX_SiCI] - k[702]*y[IDX_SiCI] -
        k[703]*y[IDX_SiH2I] - k[704]*y[IDX_SiH2I] - k[705]*y[IDX_SiH3I] -
        k[706]*y[IDX_SiH3I] - k[707]*y[IDX_SiH4I] - k[708]*y[IDX_SiH4I] -
        k[709]*y[IDX_SiHI] - k[710]*y[IDX_SiOI] - k[711]*y[IDX_SiOI] -
        k[1218]*y[IDX_EM];
    IJth(jmatrix, 62, 64) = 0.0 - k[683]*y[IDX_HeII] - k[684]*y[IDX_HeII] -
        k[685]*y[IDX_HeII];
    IJth(jmatrix, 62, 66) = 0.0 - k[686]*y[IDX_HeII] - k[687]*y[IDX_HeII];
    IJth(jmatrix, 62, 73) = 0.0 - k[144]*y[IDX_HeII] - k[688]*y[IDX_HeII];
    IJth(jmatrix, 62, 76) = 0.0 - k[693]*y[IDX_HeII];
    IJth(jmatrix, 62, 78) = 0.0 - k[689]*y[IDX_HeII] - k[690]*y[IDX_HeII];
    IJth(jmatrix, 62, 80) = 0.0 - k[145]*y[IDX_HeII] - k[691]*y[IDX_HeII] -
        k[692]*y[IDX_HeII];
    IJth(jmatrix, 62, 82) = 0.0 - k[694]*y[IDX_HeII] - k[695]*y[IDX_HeII];
    IJth(jmatrix, 62, 87) = 0.0 - k[146]*y[IDX_HeII] - k[696]*y[IDX_HeII];
    IJth(jmatrix, 62, 91) = 0.0 - k[697]*y[IDX_HeII] - k[698]*y[IDX_HeII];
    IJth(jmatrix, 62, 92) = 0.0 - k[699]*y[IDX_HeII];
    IJth(jmatrix, 62, 94) = 0.0 - k[147]*y[IDX_HeII];
    IJth(jmatrix, 62, 96) = 0.0 - k[701]*y[IDX_HeII] - k[702]*y[IDX_HeII];
    IJth(jmatrix, 62, 100) = 0.0 - k[700]*y[IDX_HeII];
    IJth(jmatrix, 62, 102) = 0.0 - k[709]*y[IDX_HeII];
    IJth(jmatrix, 62, 104) = 0.0 - k[703]*y[IDX_HeII] - k[704]*y[IDX_HeII];
    IJth(jmatrix, 62, 106) = 0.0 - k[705]*y[IDX_HeII] - k[706]*y[IDX_HeII];
    IJth(jmatrix, 62, 108) = 0.0 - k[707]*y[IDX_HeII] - k[708]*y[IDX_HeII];
    IJth(jmatrix, 62, 111) = 0.0 - k[710]*y[IDX_HeII] - k[711]*y[IDX_HeII];
    IJth(jmatrix, 63, 40) = 0.0 - k[336]*y[IDX_HeHII];
    IJth(jmatrix, 63, 41) = 0.0 - k[618]*y[IDX_HeHII];
    IJth(jmatrix, 63, 42) = 0.0 + k[1198]*y[IDX_HeI];
    IJth(jmatrix, 63, 43) = 0.0 - k[537]*y[IDX_HeHII];
    IJth(jmatrix, 63, 44) = 0.0 + k[520]*y[IDX_HeI];
    IJth(jmatrix, 63, 58) = 0.0 + k[681]*y[IDX_HeII];
    IJth(jmatrix, 63, 61) = 0.0 + k[520]*y[IDX_H2II] + k[1198]*y[IDX_HII];
    IJth(jmatrix, 63, 62) = 0.0 + k[681]*y[IDX_HCOI];
    IJth(jmatrix, 63, 63) = 0.0 - k[336]*y[IDX_EM] - k[537]*y[IDX_H2I] -
        k[618]*y[IDX_HI];
    IJth(jmatrix, 64, 9) = 0.0 + k[1327] + k[1328] + k[1329] + k[1330];
    IJth(jmatrix, 64, 24) = 0.0 + k[867]*y[IDX_NH2I] + k[1004]*y[IDX_HNCOI];
    IJth(jmatrix, 64, 26) = 0.0 + k[465]*y[IDX_HCNHII];
    IJth(jmatrix, 64, 27) = 0.0 - k[407]*y[IDX_HNCI];
    IJth(jmatrix, 64, 28) = 0.0 + k[428]*y[IDX_HCNHII] + k[1006]*y[IDX_NI];
    IJth(jmatrix, 64, 40) = 0.0 + k[329]*y[IDX_HCNHII];
    IJth(jmatrix, 64, 41) = 0.0 - k[979]*y[IDX_HNCI];
    IJth(jmatrix, 64, 42) = 0.0 - k[1]*y[IDX_HNCI];
    IJth(jmatrix, 64, 46) = 0.0 + k[634]*y[IDX_HCNHII];
    IJth(jmatrix, 64, 47) = 0.0 - k[646]*y[IDX_HNCI];
    IJth(jmatrix, 64, 50) = 0.0 - k[559]*y[IDX_HNCI];
    IJth(jmatrix, 64, 52) = 0.0 - k[589]*y[IDX_HNCI];
    IJth(jmatrix, 64, 53) = 0.0 - k[647]*y[IDX_HNCI];
    IJth(jmatrix, 64, 54) = 0.0 - k[609]*y[IDX_HNCI];
    IJth(jmatrix, 64, 56) = 0.0 - k[626]*y[IDX_HNCI];
    IJth(jmatrix, 64, 57) = 0.0 + k[329]*y[IDX_EM] + k[428]*y[IDX_CH2I] +
        k[465]*y[IDX_CHI] + k[634]*y[IDX_H2COI] + k[786]*y[IDX_NH2I];
    IJth(jmatrix, 64, 59) = 0.0 - k[648]*y[IDX_HNCI];
    IJth(jmatrix, 64, 62) = 0.0 - k[683]*y[IDX_HNCI] - k[684]*y[IDX_HNCI] -
        k[685]*y[IDX_HNCI];
    IJth(jmatrix, 64, 64) = 0.0 - k[1]*y[IDX_HII] - k[262] -
        k[407]*y[IDX_CHII] - k[559]*y[IDX_H2OII] - k[589]*y[IDX_H3II] -
        k[609]*y[IDX_H3OII] - k[626]*y[IDX_HCNII] - k[646]*y[IDX_H2COII] -
        k[647]*y[IDX_H3COII] - k[648]*y[IDX_HCOII] - k[649]*y[IDX_HNOII] -
        k[650]*y[IDX_N2HII] - k[651]*y[IDX_O2HII] - k[683]*y[IDX_HeII] -
        k[684]*y[IDX_HeII] - k[685]*y[IDX_HeII] - k[760]*y[IDX_NHII] -
        k[775]*y[IDX_NH2II] - k[844]*y[IDX_OHII] - k[979]*y[IDX_HI] - k[1151] -
        k[1305];
    IJth(jmatrix, 64, 65) = 0.0 + k[1004]*y[IDX_CI];
    IJth(jmatrix, 64, 67) = 0.0 - k[649]*y[IDX_HNCI];
    IJth(jmatrix, 64, 71) = 0.0 + k[1006]*y[IDX_CH2I];
    IJth(jmatrix, 64, 75) = 0.0 - k[650]*y[IDX_HNCI];
    IJth(jmatrix, 64, 77) = 0.0 - k[760]*y[IDX_HNCI];
    IJth(jmatrix, 64, 78) = 0.0 + k[786]*y[IDX_HCNHII] + k[867]*y[IDX_CI];
    IJth(jmatrix, 64, 79) = 0.0 - k[775]*y[IDX_HNCI];
    IJth(jmatrix, 64, 90) = 0.0 - k[651]*y[IDX_HNCI];
    IJth(jmatrix, 64, 93) = 0.0 - k[844]*y[IDX_HNCI];
    IJth(jmatrix, 65, 10) = 0.0 + k[1375] + k[1376] + k[1377] + k[1378];
    IJth(jmatrix, 65, 24) = 0.0 - k[1004]*y[IDX_HNCOI];
    IJth(jmatrix, 65, 28) = 0.0 + k[888]*y[IDX_NOI];
    IJth(jmatrix, 65, 42) = 0.0 - k[502]*y[IDX_HNCOI];
    IJth(jmatrix, 65, 65) = 0.0 - k[263] - k[502]*y[IDX_HII] -
        k[1004]*y[IDX_CI] - k[1152] - k[1224];
    IJth(jmatrix, 65, 82) = 0.0 + k[888]*y[IDX_CH2I];
    IJth(jmatrix, 66, 11) = 0.0 + k[1351] + k[1352] + k[1353] + k[1354];
    IJth(jmatrix, 66, 26) = 0.0 - k[926]*y[IDX_HNOI];
    IJth(jmatrix, 66, 28) = 0.0 - k[883]*y[IDX_HNOI];
    IJth(jmatrix, 66, 30) = 0.0 - k[906]*y[IDX_HNOI] + k[909]*y[IDX_NO2I];
    IJth(jmatrix, 66, 35) = 0.0 - k[944]*y[IDX_HNOI];
    IJth(jmatrix, 66, 37) = 0.0 - k[951]*y[IDX_HNOI];
    IJth(jmatrix, 66, 40) = 0.0 + k[310]*y[IDX_H2NOII];
    IJth(jmatrix, 66, 41) = 0.0 - k[980]*y[IDX_HNOI] - k[981]*y[IDX_HNOI] -
        k[982]*y[IDX_HNOI];
    IJth(jmatrix, 66, 42) = 0.0 - k[503]*y[IDX_HNOI];
    IJth(jmatrix, 66, 48) = 0.0 + k[310]*y[IDX_EM];
    IJth(jmatrix, 66, 52) = 0.0 - k[590]*y[IDX_HNOI];
    IJth(jmatrix, 66, 58) = 0.0 - k[999]*y[IDX_HNOI] + k[1000]*y[IDX_NOI];
    IJth(jmatrix, 66, 62) = 0.0 - k[686]*y[IDX_HNOI] - k[687]*y[IDX_HNOI];
    IJth(jmatrix, 66, 66) = 0.0 - k[264] - k[503]*y[IDX_HII] -
        k[590]*y[IDX_H3II] - k[686]*y[IDX_HeII] - k[687]*y[IDX_HeII] -
        k[883]*y[IDX_CH2I] - k[906]*y[IDX_CH3I] - k[926]*y[IDX_CHI] -
        k[944]*y[IDX_CNI] - k[951]*y[IDX_COI] - k[980]*y[IDX_HI] -
        k[981]*y[IDX_HI] - k[982]*y[IDX_HI] - k[999]*y[IDX_HCOI] -
        k[1017]*y[IDX_NI] - k[1068]*y[IDX_OI] - k[1069]*y[IDX_OI] -
        k[1070]*y[IDX_OI] - k[1097]*y[IDX_OHI] - k[1153] - k[1294];
    IJth(jmatrix, 66, 67) = 0.0 + k[204]*y[IDX_NOI];
    IJth(jmatrix, 66, 71) = 0.0 - k[1017]*y[IDX_HNOI];
    IJth(jmatrix, 66, 76) = 0.0 + k[1041]*y[IDX_NO2I] + k[1044]*y[IDX_O2I] +
        k[1049]*y[IDX_OHI];
    IJth(jmatrix, 66, 78) = 0.0 + k[1072]*y[IDX_OI];
    IJth(jmatrix, 66, 82) = 0.0 + k[204]*y[IDX_HNOII] + k[1000]*y[IDX_HCOI];
    IJth(jmatrix, 66, 84) = 0.0 + k[909]*y[IDX_CH3I] + k[1041]*y[IDX_NHI];
    IJth(jmatrix, 66, 85) = 0.0 - k[1068]*y[IDX_HNOI] - k[1069]*y[IDX_HNOI]
        - k[1070]*y[IDX_HNOI] + k[1072]*y[IDX_NH2I];
    IJth(jmatrix, 66, 87) = 0.0 + k[1044]*y[IDX_NHI];
    IJth(jmatrix, 66, 92) = 0.0 + k[1049]*y[IDX_NHI] - k[1097]*y[IDX_HNOI];
    IJth(jmatrix, 67, 24) = 0.0 - k[388]*y[IDX_HNOII];
    IJth(jmatrix, 67, 26) = 0.0 - k[467]*y[IDX_HNOII];
    IJth(jmatrix, 67, 28) = 0.0 - k[430]*y[IDX_HNOII];
    IJth(jmatrix, 67, 35) = 0.0 - k[482]*y[IDX_HNOII];
    IJth(jmatrix, 67, 37) = 0.0 - k[486]*y[IDX_HNOII];
    IJth(jmatrix, 67, 39) = 0.0 - k[652]*y[IDX_HNOII] + k[749]*y[IDX_NHII];
    IJth(jmatrix, 67, 40) = 0.0 - k[334]*y[IDX_HNOII];
    IJth(jmatrix, 67, 44) = 0.0 + k[524]*y[IDX_NOI];
    IJth(jmatrix, 67, 46) = 0.0 - k[550]*y[IDX_HNOII];
    IJth(jmatrix, 67, 49) = 0.0 - k[568]*y[IDX_HNOII] + k[755]*y[IDX_NHII];
    IJth(jmatrix, 67, 50) = 0.0 + k[738]*y[IDX_NI];
    IJth(jmatrix, 67, 52) = 0.0 + k[596]*y[IDX_NOI];
    IJth(jmatrix, 67, 55) = 0.0 - k[630]*y[IDX_HNOII];
    IJth(jmatrix, 67, 58) = 0.0 - k[642]*y[IDX_HNOII];
    IJth(jmatrix, 67, 64) = 0.0 - k[649]*y[IDX_HNOII];
    IJth(jmatrix, 67, 67) = 0.0 - k[204]*y[IDX_NOI] - k[334]*y[IDX_EM] -
        k[388]*y[IDX_CI] - k[430]*y[IDX_CH2I] - k[467]*y[IDX_CHI] -
        k[482]*y[IDX_CNI] - k[486]*y[IDX_COI] - k[550]*y[IDX_H2COI] -
        k[568]*y[IDX_H2OI] - k[630]*y[IDX_HCNI] - k[642]*y[IDX_HCOI] -
        k[649]*y[IDX_HNCI] - k[652]*y[IDX_CO2I] - k[732]*y[IDX_N2I] -
        k[788]*y[IDX_NH2I] - k[800]*y[IDX_NHI] - k[856]*y[IDX_OHI] - k[1295];
    IJth(jmatrix, 67, 71) = 0.0 + k[738]*y[IDX_H2OII];
    IJth(jmatrix, 67, 73) = 0.0 - k[732]*y[IDX_HNOII];
    IJth(jmatrix, 67, 76) = 0.0 - k[800]*y[IDX_HNOII] + k[804]*y[IDX_O2II];
    IJth(jmatrix, 67, 77) = 0.0 + k[749]*y[IDX_CO2I] + k[755]*y[IDX_H2OI];
    IJth(jmatrix, 67, 78) = 0.0 - k[788]*y[IDX_HNOII];
    IJth(jmatrix, 67, 79) = 0.0 + k[778]*y[IDX_O2I] + k[827]*y[IDX_OI];
    IJth(jmatrix, 67, 81) = 0.0 + k[828]*y[IDX_OI];
    IJth(jmatrix, 67, 82) = 0.0 - k[204]*y[IDX_HNOII] + k[524]*y[IDX_H2II] +
        k[596]*y[IDX_H3II] + k[807]*y[IDX_O2HII] + k[846]*y[IDX_OHII];
    IJth(jmatrix, 67, 85) = 0.0 + k[827]*y[IDX_NH2II] + k[828]*y[IDX_NH3II];
    IJth(jmatrix, 67, 87) = 0.0 + k[778]*y[IDX_NH2II];
    IJth(jmatrix, 67, 88) = 0.0 + k[804]*y[IDX_NHI];
    IJth(jmatrix, 67, 90) = 0.0 + k[807]*y[IDX_NOI];
    IJth(jmatrix, 67, 92) = 0.0 - k[856]*y[IDX_HNOII];
    IJth(jmatrix, 67, 93) = 0.0 + k[846]*y[IDX_NOI];
    IJth(jmatrix, 68, 25) = 0.0 + k[371]*y[IDX_H2OI];
    IJth(jmatrix, 68, 37) = 0.0 + k[584]*y[IDX_H3II];
    IJth(jmatrix, 68, 38) = 0.0 + k[533]*y[IDX_H2I];
    IJth(jmatrix, 68, 40) = 0.0 - k[335]*y[IDX_HOCII];
    IJth(jmatrix, 68, 43) = 0.0 - k[5]*y[IDX_HOCII] + k[533]*y[IDX_COII];
    IJth(jmatrix, 68, 49) = 0.0 + k[371]*y[IDX_CII];
    IJth(jmatrix, 68, 52) = 0.0 + k[584]*y[IDX_COI];
    IJth(jmatrix, 68, 68) = 0.0 - k[5]*y[IDX_H2I] - k[335]*y[IDX_EM] -
        k[1226];
    IJth(jmatrix, 69, 12) = 0.0 + k[1319] + k[1320] + k[1321] + k[1322];
    IJth(jmatrix, 69, 25) = 0.0 - k[18]*y[IDX_MgI];
    IJth(jmatrix, 69, 27) = 0.0 - k[32]*y[IDX_MgI];
    IJth(jmatrix, 69, 31) = 0.0 - k[47]*y[IDX_MgI];
    IJth(jmatrix, 69, 40) = 0.0 + k[1219]*y[IDX_MgII];
    IJth(jmatrix, 69, 42) = 0.0 - k[83]*y[IDX_MgI];
    IJth(jmatrix, 69, 47) = 0.0 - k[148]*y[IDX_MgI];
    IJth(jmatrix, 69, 50) = 0.0 - k[119]*y[IDX_MgI];
    IJth(jmatrix, 69, 52) = 0.0 - k[591]*y[IDX_MgI];
    IJth(jmatrix, 69, 59) = 0.0 - k[149]*y[IDX_MgI];
    IJth(jmatrix, 69, 69) = 0.0 - k[18]*y[IDX_CII] - k[32]*y[IDX_CHII] -
        k[47]*y[IDX_CH3II] - k[83]*y[IDX_HII] - k[119]*y[IDX_H2OII] -
        k[148]*y[IDX_H2COII] - k[149]*y[IDX_HCOII] - k[150]*y[IDX_N2II] -
        k[151]*y[IDX_NOII] - k[152]*y[IDX_O2II] - k[153]*y[IDX_SiII] -
        k[154]*y[IDX_SiOII] - k[163]*y[IDX_NII] - k[190]*y[IDX_NH3II] - k[266] -
        k[591]*y[IDX_H3II] - k[1154] - k[1300];
    IJth(jmatrix, 69, 70) = 0.0 + k[1219]*y[IDX_EM];
    IJth(jmatrix, 69, 72) = 0.0 - k[163]*y[IDX_MgI];
    IJth(jmatrix, 69, 74) = 0.0 - k[150]*y[IDX_MgI];
    IJth(jmatrix, 69, 81) = 0.0 - k[190]*y[IDX_MgI];
    IJth(jmatrix, 69, 83) = 0.0 - k[151]*y[IDX_MgI];
    IJth(jmatrix, 69, 88) = 0.0 - k[152]*y[IDX_MgI];
    IJth(jmatrix, 69, 95) = 0.0 - k[153]*y[IDX_MgI];
    IJth(jmatrix, 69, 112) = 0.0 - k[154]*y[IDX_MgI];
    IJth(jmatrix, 70, 25) = 0.0 + k[18]*y[IDX_MgI];
    IJth(jmatrix, 70, 27) = 0.0 + k[32]*y[IDX_MgI];
    IJth(jmatrix, 70, 31) = 0.0 + k[47]*y[IDX_MgI];
    IJth(jmatrix, 70, 40) = 0.0 - k[1219]*y[IDX_MgII];
    IJth(jmatrix, 70, 42) = 0.0 + k[83]*y[IDX_MgI];
    IJth(jmatrix, 70, 47) = 0.0 + k[148]*y[IDX_MgI];
    IJth(jmatrix, 70, 50) = 0.0 + k[119]*y[IDX_MgI];
    IJth(jmatrix, 70, 52) = 0.0 + k[591]*y[IDX_MgI];
    IJth(jmatrix, 70, 59) = 0.0 + k[149]*y[IDX_MgI];
    IJth(jmatrix, 70, 69) = 0.0 + k[18]*y[IDX_CII] + k[32]*y[IDX_CHII] +
        k[47]*y[IDX_CH3II] + k[83]*y[IDX_HII] + k[119]*y[IDX_H2OII] +
        k[148]*y[IDX_H2COII] + k[149]*y[IDX_HCOII] + k[150]*y[IDX_N2II] +
        k[151]*y[IDX_NOII] + k[152]*y[IDX_O2II] + k[153]*y[IDX_SiII] +
        k[154]*y[IDX_SiOII] + k[163]*y[IDX_NII] + k[190]*y[IDX_NH3II] + k[266] +
        k[591]*y[IDX_H3II] + k[1154];
    IJth(jmatrix, 70, 70) = 0.0 - k[1219]*y[IDX_EM] - k[1299];
    IJth(jmatrix, 70, 72) = 0.0 + k[163]*y[IDX_MgI];
    IJth(jmatrix, 70, 74) = 0.0 + k[150]*y[IDX_MgI];
    IJth(jmatrix, 70, 81) = 0.0 + k[190]*y[IDX_MgI];
    IJth(jmatrix, 70, 83) = 0.0 + k[151]*y[IDX_MgI];
    IJth(jmatrix, 70, 88) = 0.0 + k[152]*y[IDX_MgI];
    IJth(jmatrix, 70, 95) = 0.0 + k[153]*y[IDX_MgI];
    IJth(jmatrix, 70, 112) = 0.0 + k[154]*y[IDX_MgI];
    IJth(jmatrix, 71, 24) = 0.0 + k[390]*y[IDX_NHII] + k[865]*y[IDX_N2I] +
        k[870]*y[IDX_NHI] + k[872]*y[IDX_NOI] - k[1194]*y[IDX_NI];
    IJth(jmatrix, 71, 25) = 0.0 - k[1192]*y[IDX_NI];
    IJth(jmatrix, 71, 26) = 0.0 + k[57]*y[IDX_NII] + k[470]*y[IDX_NHII] +
        k[927]*y[IDX_N2I] - k[928]*y[IDX_NI] - k[929]*y[IDX_NI] +
        k[931]*y[IDX_NOI];
    IJth(jmatrix, 71, 27) = 0.0 - k[408]*y[IDX_NI];
    IJth(jmatrix, 71, 28) = 0.0 + k[155]*y[IDX_NII] + k[432]*y[IDX_NHII] +
        k[886]*y[IDX_NOI] - k[1005]*y[IDX_NI] - k[1006]*y[IDX_NI] -
        k[1007]*y[IDX_NI];
    IJth(jmatrix, 71, 29) = 0.0 - k[736]*y[IDX_NI];
    IJth(jmatrix, 71, 30) = 0.0 - k[1008]*y[IDX_NI] - k[1009]*y[IDX_NI] -
        k[1010]*y[IDX_NI];
    IJth(jmatrix, 71, 33) = 0.0 + k[156]*y[IDX_NII] + k[716]*y[IDX_NII];
    IJth(jmatrix, 71, 35) = 0.0 + k[157]*y[IDX_NII] + k[251] +
        k[664]*y[IDX_HeII] + k[747]*y[IDX_NHII] + k[947]*y[IDX_NOI] -
        k[1011]*y[IDX_NI] + k[1035]*y[IDX_NHI] + k[1057]*y[IDX_OI] + k[1130];
    IJth(jmatrix, 71, 36) = 0.0 + k[303]*y[IDX_EM] - k[737]*y[IDX_NI];
    IJth(jmatrix, 71, 37) = 0.0 + k[158]*y[IDX_NII] + k[751]*y[IDX_NHII];
    IJth(jmatrix, 71, 38) = 0.0 + k[795]*y[IDX_NHI];
    IJth(jmatrix, 71, 39) = 0.0 + k[748]*y[IDX_NHII] - k[1012]*y[IDX_NI];
    IJth(jmatrix, 71, 40) = 0.0 + k[303]*y[IDX_CNII] + k[337]*y[IDX_N2II] +
        k[337]*y[IDX_N2II] + k[339]*y[IDX_N2HII] + k[340]*y[IDX_NHII] +
        k[341]*y[IDX_NH2II] + k[345]*y[IDX_NOII] + k[1220]*y[IDX_NII];
    IJth(jmatrix, 71, 41) = 0.0 + k[985]*y[IDX_NHI] + k[988]*y[IDX_NOI];
    IJth(jmatrix, 71, 43) = 0.0 + k[540]*y[IDX_NHII] - k[960]*y[IDX_NI];
    IJth(jmatrix, 71, 44) = 0.0 - k[522]*y[IDX_NI];
    IJth(jmatrix, 71, 45) = 0.0 - k[1013]*y[IDX_NI];
    IJth(jmatrix, 71, 46) = 0.0 + k[159]*y[IDX_NII] + k[752]*y[IDX_NHII];
    IJth(jmatrix, 71, 47) = 0.0 + k[796]*y[IDX_NHI];
    IJth(jmatrix, 71, 49) = 0.0 + k[160]*y[IDX_NII] + k[754]*y[IDX_NHII];
    IJth(jmatrix, 71, 50) = 0.0 - k[738]*y[IDX_NI] - k[739]*y[IDX_NI] +
        k[797]*y[IDX_NHI];
    IJth(jmatrix, 71, 55) = 0.0 + k[161]*y[IDX_NII] + k[678]*y[IDX_HeII] +
        k[679]*y[IDX_HeII] + k[758]*y[IDX_NHII] + k[814]*y[IDX_OII];
    IJth(jmatrix, 71, 58) = 0.0 + k[162]*y[IDX_NII] + k[759]*y[IDX_NHII] -
        k[1014]*y[IDX_NI] - k[1015]*y[IDX_NI] - k[1016]*y[IDX_NI];
    IJth(jmatrix, 71, 62) = 0.0 + k[664]*y[IDX_CNI] + k[678]*y[IDX_HCNI] +
        k[679]*y[IDX_HCNI] + k[684]*y[IDX_HNCI] + k[688]*y[IDX_N2I] +
        k[694]*y[IDX_NOI];
    IJth(jmatrix, 71, 64) = 0.0 + k[684]*y[IDX_HeII] + k[760]*y[IDX_NHII];
    IJth(jmatrix, 71, 66) = 0.0 - k[1017]*y[IDX_NI];
    IJth(jmatrix, 71, 69) = 0.0 + k[163]*y[IDX_NII];
    IJth(jmatrix, 71, 71) = 0.0 - k[174]*y[IDX_N2II] - k[238] - k[268] -
        k[408]*y[IDX_CHII] - k[522]*y[IDX_H2II] - k[736]*y[IDX_CH2II] -
        k[737]*y[IDX_CNII] - k[738]*y[IDX_H2OII] - k[739]*y[IDX_H2OII] -
        k[740]*y[IDX_NHII] - k[741]*y[IDX_NH2II] - k[742]*y[IDX_O2II] -
        k[743]*y[IDX_OHII] - k[744]*y[IDX_SiCII] - k[745]*y[IDX_SiOII] -
        k[746]*y[IDX_SiOII] - k[928]*y[IDX_CHI] - k[929]*y[IDX_CHI] -
        k[960]*y[IDX_H2I] - k[1005]*y[IDX_CH2I] - k[1006]*y[IDX_CH2I] -
        k[1007]*y[IDX_CH2I] - k[1008]*y[IDX_CH3I] - k[1009]*y[IDX_CH3I] -
        k[1010]*y[IDX_CH3I] - k[1011]*y[IDX_CNI] - k[1012]*y[IDX_CO2I] -
        k[1013]*y[IDX_H2CNI] - k[1014]*y[IDX_HCOI] - k[1015]*y[IDX_HCOI] -
        k[1016]*y[IDX_HCOI] - k[1017]*y[IDX_HNOI] - k[1018]*y[IDX_NHI] -
        k[1019]*y[IDX_NO2I] - k[1020]*y[IDX_NO2I] - k[1021]*y[IDX_NO2I] -
        k[1022]*y[IDX_NOI] - k[1023]*y[IDX_O2I] - k[1024]*y[IDX_O2HI] -
        k[1025]*y[IDX_OHI] - k[1026]*y[IDX_OHI] - k[1027]*y[IDX_SiCI] -
        k[1192]*y[IDX_CII] - k[1194]*y[IDX_CI] - k[1210]*y[IDX_NII] - k[1290];
    IJth(jmatrix, 71, 72) = 0.0 + k[57]*y[IDX_CHI] + k[155]*y[IDX_CH2I] +
        k[156]*y[IDX_CH4I] + k[157]*y[IDX_CNI] + k[158]*y[IDX_COI] +
        k[159]*y[IDX_H2COI] + k[160]*y[IDX_H2OI] + k[161]*y[IDX_HCNI] +
        k[162]*y[IDX_HCOI] + k[163]*y[IDX_MgI] + k[164]*y[IDX_NH2I] +
        k[165]*y[IDX_NH3I] + k[166]*y[IDX_NHI] + k[167]*y[IDX_NOI] +
        k[168]*y[IDX_O2I] + k[169]*y[IDX_OHI] + k[716]*y[IDX_CH4I] -
        k[1210]*y[IDX_NI] + k[1220]*y[IDX_EM];
    IJth(jmatrix, 71, 73) = 0.0 + k[267] + k[267] + k[688]*y[IDX_HeII] +
        k[761]*y[IDX_NHII] + k[817]*y[IDX_OII] + k[865]*y[IDX_CI] +
        k[927]*y[IDX_CHI] + k[1071]*y[IDX_OI] + k[1155] + k[1155];
    IJth(jmatrix, 71, 74) = 0.0 - k[174]*y[IDX_NI] + k[337]*y[IDX_EM] +
        k[337]*y[IDX_EM] + k[825]*y[IDX_OI];
    IJth(jmatrix, 71, 75) = 0.0 + k[339]*y[IDX_EM];
    IJth(jmatrix, 71, 76) = 0.0 + k[166]*y[IDX_NII] + k[274] +
        k[763]*y[IDX_NHII] + k[795]*y[IDX_COII] + k[796]*y[IDX_H2COII] +
        k[797]*y[IDX_H2OII] + k[802]*y[IDX_NH2II] + k[870]*y[IDX_CI] +
        k[985]*y[IDX_HI] - k[1018]*y[IDX_NI] + k[1035]*y[IDX_CNI] +
        k[1040]*y[IDX_NHI] + k[1040]*y[IDX_NHI] + k[1047]*y[IDX_OI] +
        k[1048]*y[IDX_OHI] + k[1162];
    IJth(jmatrix, 71, 77) = 0.0 + k[340]*y[IDX_EM] + k[390]*y[IDX_CI] +
        k[432]*y[IDX_CH2I] + k[470]*y[IDX_CHI] + k[540]*y[IDX_H2I] -
        k[740]*y[IDX_NI] + k[747]*y[IDX_CNI] + k[748]*y[IDX_CO2I] +
        k[751]*y[IDX_COI] + k[752]*y[IDX_H2COI] + k[754]*y[IDX_H2OI] +
        k[758]*y[IDX_HCNI] + k[759]*y[IDX_HCOI] + k[760]*y[IDX_HNCI] +
        k[761]*y[IDX_N2I] + k[762]*y[IDX_NH2I] + k[763]*y[IDX_NHI] +
        k[766]*y[IDX_O2I] + k[767]*y[IDX_OI] + k[768]*y[IDX_OHI] + k[1156];
    IJth(jmatrix, 71, 78) = 0.0 + k[164]*y[IDX_NII] + k[762]*y[IDX_NHII];
    IJth(jmatrix, 71, 79) = 0.0 + k[341]*y[IDX_EM] - k[741]*y[IDX_NI] +
        k[802]*y[IDX_NHI];
    IJth(jmatrix, 71, 80) = 0.0 + k[165]*y[IDX_NII];
    IJth(jmatrix, 71, 82) = 0.0 + k[167]*y[IDX_NII] + k[278] +
        k[694]*y[IDX_HeII] + k[872]*y[IDX_CI] + k[886]*y[IDX_CH2I] +
        k[931]*y[IDX_CHI] + k[947]*y[IDX_CNI] + k[988]*y[IDX_HI] -
        k[1022]*y[IDX_NI] + k[1076]*y[IDX_OI] + k[1105]*y[IDX_SiI] + k[1166];
    IJth(jmatrix, 71, 83) = 0.0 + k[345]*y[IDX_EM];
    IJth(jmatrix, 71, 84) = 0.0 - k[1019]*y[IDX_NI] - k[1020]*y[IDX_NI] -
        k[1021]*y[IDX_NI];
    IJth(jmatrix, 71, 85) = 0.0 + k[767]*y[IDX_NHII] + k[825]*y[IDX_N2II] +
        k[1047]*y[IDX_NHI] + k[1057]*y[IDX_CNI] + k[1071]*y[IDX_N2I] +
        k[1076]*y[IDX_NOI];
    IJth(jmatrix, 71, 86) = 0.0 + k[814]*y[IDX_HCNI] + k[817]*y[IDX_N2I];
    IJth(jmatrix, 71, 87) = 0.0 + k[168]*y[IDX_NII] + k[766]*y[IDX_NHII] -
        k[1023]*y[IDX_NI];
    IJth(jmatrix, 71, 88) = 0.0 - k[742]*y[IDX_NI];
    IJth(jmatrix, 71, 89) = 0.0 - k[1024]*y[IDX_NI];
    IJth(jmatrix, 71, 92) = 0.0 + k[169]*y[IDX_NII] + k[768]*y[IDX_NHII] -
        k[1025]*y[IDX_NI] - k[1026]*y[IDX_NI] + k[1048]*y[IDX_NHI];
    IJth(jmatrix, 71, 93) = 0.0 - k[743]*y[IDX_NI];
    IJth(jmatrix, 71, 94) = 0.0 + k[1105]*y[IDX_NOI];
    IJth(jmatrix, 71, 96) = 0.0 - k[1027]*y[IDX_NI];
    IJth(jmatrix, 71, 97) = 0.0 - k[744]*y[IDX_NI];
    IJth(jmatrix, 71, 112) = 0.0 - k[745]*y[IDX_NI] - k[746]*y[IDX_NI];
    IJth(jmatrix, 72, 26) = 0.0 - k[57]*y[IDX_NII] - k[468]*y[IDX_NII];
    IJth(jmatrix, 72, 28) = 0.0 - k[155]*y[IDX_NII];
    IJth(jmatrix, 72, 32) = 0.0 - k[712]*y[IDX_NII] - k[713]*y[IDX_NII] -
        k[714]*y[IDX_NII] - k[715]*y[IDX_NII];
    IJth(jmatrix, 72, 33) = 0.0 - k[156]*y[IDX_NII] - k[716]*y[IDX_NII] -
        k[717]*y[IDX_NII] - k[718]*y[IDX_NII];
    IJth(jmatrix, 72, 35) = 0.0 - k[157]*y[IDX_NII] + k[663]*y[IDX_HeII];
    IJth(jmatrix, 72, 37) = 0.0 - k[158]*y[IDX_NII] - k[720]*y[IDX_NII];
    IJth(jmatrix, 72, 39) = 0.0 - k[719]*y[IDX_NII];
    IJth(jmatrix, 72, 40) = 0.0 - k[1220]*y[IDX_NII];
    IJth(jmatrix, 72, 43) = 0.0 - k[538]*y[IDX_NII];
    IJth(jmatrix, 72, 46) = 0.0 - k[159]*y[IDX_NII] - k[721]*y[IDX_NII] -
        k[722]*y[IDX_NII];
    IJth(jmatrix, 72, 49) = 0.0 - k[160]*y[IDX_NII];
    IJth(jmatrix, 72, 55) = 0.0 - k[161]*y[IDX_NII] + k[677]*y[IDX_HeII];
    IJth(jmatrix, 72, 58) = 0.0 - k[162]*y[IDX_NII] - k[723]*y[IDX_NII];
    IJth(jmatrix, 72, 62) = 0.0 + k[663]*y[IDX_CNI] + k[677]*y[IDX_HCNI] +
        k[688]*y[IDX_N2I] + k[689]*y[IDX_NH2I] + k[693]*y[IDX_NHI] +
        k[695]*y[IDX_NOI];
    IJth(jmatrix, 72, 69) = 0.0 - k[163]*y[IDX_NII];
    IJth(jmatrix, 72, 71) = 0.0 + k[174]*y[IDX_N2II] + k[238] + k[268] -
        k[1210]*y[IDX_NII];
    IJth(jmatrix, 72, 72) = 0.0 - k[57]*y[IDX_CHI] - k[155]*y[IDX_CH2I] -
        k[156]*y[IDX_CH4I] - k[157]*y[IDX_CNI] - k[158]*y[IDX_COI] -
        k[159]*y[IDX_H2COI] - k[160]*y[IDX_H2OI] - k[161]*y[IDX_HCNI] -
        k[162]*y[IDX_HCOI] - k[163]*y[IDX_MgI] - k[164]*y[IDX_NH2I] -
        k[165]*y[IDX_NH3I] - k[166]*y[IDX_NHI] - k[167]*y[IDX_NOI] -
        k[168]*y[IDX_O2I] - k[169]*y[IDX_OHI] - k[468]*y[IDX_CHI] -
        k[538]*y[IDX_H2I] - k[712]*y[IDX_CH3OHI] - k[713]*y[IDX_CH3OHI] -
        k[714]*y[IDX_CH3OHI] - k[715]*y[IDX_CH3OHI] - k[716]*y[IDX_CH4I] -
        k[717]*y[IDX_CH4I] - k[718]*y[IDX_CH4I] - k[719]*y[IDX_CO2I] -
        k[720]*y[IDX_COI] - k[721]*y[IDX_H2COI] - k[722]*y[IDX_H2COI] -
        k[723]*y[IDX_HCOI] - k[724]*y[IDX_NH3I] - k[725]*y[IDX_NH3I] -
        k[726]*y[IDX_NHI] - k[727]*y[IDX_NOI] - k[728]*y[IDX_O2I] -
        k[729]*y[IDX_O2I] - k[1210]*y[IDX_NI] - k[1220]*y[IDX_EM] - k[1268];
    IJth(jmatrix, 72, 73) = 0.0 + k[688]*y[IDX_HeII];
    IJth(jmatrix, 72, 74) = 0.0 + k[174]*y[IDX_NI];
    IJth(jmatrix, 72, 76) = 0.0 - k[166]*y[IDX_NII] + k[693]*y[IDX_HeII] -
        k[726]*y[IDX_NII];
    IJth(jmatrix, 72, 78) = 0.0 - k[164]*y[IDX_NII] + k[689]*y[IDX_HeII];
    IJth(jmatrix, 72, 80) = 0.0 - k[165]*y[IDX_NII] - k[724]*y[IDX_NII] -
        k[725]*y[IDX_NII];
    IJth(jmatrix, 72, 82) = 0.0 - k[167]*y[IDX_NII] + k[695]*y[IDX_HeII] -
        k[727]*y[IDX_NII];
    IJth(jmatrix, 72, 87) = 0.0 - k[168]*y[IDX_NII] - k[728]*y[IDX_NII] -
        k[729]*y[IDX_NII];
    IJth(jmatrix, 72, 92) = 0.0 - k[169]*y[IDX_NII];
    IJth(jmatrix, 73, 13) = 0.0 + k[1335] + k[1336] + k[1337] + k[1338];
    IJth(jmatrix, 73, 24) = 0.0 + k[29]*y[IDX_N2II] + k[389]*y[IDX_N2HII] -
        k[865]*y[IDX_N2I];
    IJth(jmatrix, 73, 26) = 0.0 + k[58]*y[IDX_N2II] + k[469]*y[IDX_N2HII] -
        k[927]*y[IDX_N2I];
    IJth(jmatrix, 73, 28) = 0.0 + k[41]*y[IDX_N2II] + k[431]*y[IDX_N2HII] -
        k[884]*y[IDX_N2I];
    IJth(jmatrix, 73, 33) = 0.0 + k[455]*y[IDX_N2II] + k[456]*y[IDX_N2II];
    IJth(jmatrix, 73, 35) = 0.0 + k[69]*y[IDX_N2II] + k[946]*y[IDX_NOI] +
        k[1011]*y[IDX_NI];
    IJth(jmatrix, 73, 37) = 0.0 + k[74]*y[IDX_N2II] + k[487]*y[IDX_N2HII];
    IJth(jmatrix, 73, 39) = 0.0 + k[734]*y[IDX_N2HII];
    IJth(jmatrix, 73, 40) = 0.0 + k[338]*y[IDX_N2HII];
    IJth(jmatrix, 73, 44) = 0.0 - k[521]*y[IDX_N2I];
    IJth(jmatrix, 73, 46) = 0.0 + k[170]*y[IDX_N2II] + k[730]*y[IDX_N2II] +
        k[735]*y[IDX_N2HII];
    IJth(jmatrix, 73, 49) = 0.0 + k[125]*y[IDX_N2II] + k[570]*y[IDX_N2HII];
    IJth(jmatrix, 73, 52) = 0.0 - k[592]*y[IDX_N2I];
    IJth(jmatrix, 73, 55) = 0.0 + k[135]*y[IDX_N2II] + k[631]*y[IDX_N2HII];
    IJth(jmatrix, 73, 58) = 0.0 + k[171]*y[IDX_N2II] + k[643]*y[IDX_N2HII];
    IJth(jmatrix, 73, 62) = 0.0 - k[144]*y[IDX_N2I] - k[688]*y[IDX_N2I];
    IJth(jmatrix, 73, 64) = 0.0 + k[650]*y[IDX_N2HII];
    IJth(jmatrix, 73, 67) = 0.0 - k[732]*y[IDX_N2I];
    IJth(jmatrix, 73, 69) = 0.0 + k[150]*y[IDX_N2II];
    IJth(jmatrix, 73, 71) = 0.0 + k[174]*y[IDX_N2II] + k[1011]*y[IDX_CNI] +
        k[1018]*y[IDX_NHI] + k[1019]*y[IDX_NO2I] + k[1021]*y[IDX_NO2I] +
        k[1022]*y[IDX_NOI];
    IJth(jmatrix, 73, 73) = 0.0 - k[144]*y[IDX_HeII] - k[267] -
        k[521]*y[IDX_H2II] - k[592]*y[IDX_H3II] - k[688]*y[IDX_HeII] -
        k[732]*y[IDX_HNOII] - k[733]*y[IDX_O2HII] - k[761]*y[IDX_NHII] -
        k[817]*y[IDX_OII] - k[845]*y[IDX_OHII] - k[865]*y[IDX_CI] -
        k[884]*y[IDX_CH2I] - k[927]*y[IDX_CHI] - k[1071]*y[IDX_OI] - k[1155] -
        k[1262];
    IJth(jmatrix, 73, 74) = 0.0 + k[29]*y[IDX_CI] + k[41]*y[IDX_CH2I] +
        k[58]*y[IDX_CHI] + k[69]*y[IDX_CNI] + k[74]*y[IDX_COI] +
        k[125]*y[IDX_H2OI] + k[135]*y[IDX_HCNI] + k[150]*y[IDX_MgI] +
        k[170]*y[IDX_H2COI] + k[171]*y[IDX_HCOI] + k[172]*y[IDX_NOI] +
        k[173]*y[IDX_O2I] + k[174]*y[IDX_NI] + k[186]*y[IDX_NH2I] +
        k[197]*y[IDX_NH3I] + k[201]*y[IDX_NHI] + k[218]*y[IDX_OI] +
        k[227]*y[IDX_OHI] + k[455]*y[IDX_CH4I] + k[456]*y[IDX_CH4I] +
        k[730]*y[IDX_H2COI];
    IJth(jmatrix, 73, 75) = 0.0 + k[338]*y[IDX_EM] + k[389]*y[IDX_CI] +
        k[431]*y[IDX_CH2I] + k[469]*y[IDX_CHI] + k[487]*y[IDX_COI] +
        k[570]*y[IDX_H2OI] + k[631]*y[IDX_HCNI] + k[643]*y[IDX_HCOI] +
        k[650]*y[IDX_HNCI] + k[734]*y[IDX_CO2I] + k[735]*y[IDX_H2COI] +
        k[789]*y[IDX_NH2I] + k[801]*y[IDX_NHI] + k[826]*y[IDX_OI] +
        k[857]*y[IDX_OHI];
    IJth(jmatrix, 73, 76) = 0.0 + k[201]*y[IDX_N2II] + k[801]*y[IDX_N2HII] +
        k[1018]*y[IDX_NI] + k[1038]*y[IDX_NHI] + k[1038]*y[IDX_NHI] +
        k[1039]*y[IDX_NHI] + k[1039]*y[IDX_NHI] + k[1042]*y[IDX_NOI] +
        k[1043]*y[IDX_NOI];
    IJth(jmatrix, 73, 77) = 0.0 - k[761]*y[IDX_N2I];
    IJth(jmatrix, 73, 78) = 0.0 + k[186]*y[IDX_N2II] + k[789]*y[IDX_N2HII] +
        k[1029]*y[IDX_NOI] + k[1030]*y[IDX_NOI];
    IJth(jmatrix, 73, 80) = 0.0 + k[197]*y[IDX_N2II];
    IJth(jmatrix, 73, 82) = 0.0 + k[172]*y[IDX_N2II] + k[946]*y[IDX_CNI] +
        k[1022]*y[IDX_NI] + k[1029]*y[IDX_NH2I] + k[1030]*y[IDX_NH2I] +
        k[1042]*y[IDX_NHI] + k[1043]*y[IDX_NHI] + k[1051]*y[IDX_NOI] +
        k[1051]*y[IDX_NOI] + k[1053]*y[IDX_OCNI];
    IJth(jmatrix, 73, 84) = 0.0 + k[1019]*y[IDX_NI] + k[1021]*y[IDX_NI];
    IJth(jmatrix, 73, 85) = 0.0 + k[218]*y[IDX_N2II] + k[826]*y[IDX_N2HII] -
        k[1071]*y[IDX_N2I];
    IJth(jmatrix, 73, 86) = 0.0 - k[817]*y[IDX_N2I];
    IJth(jmatrix, 73, 87) = 0.0 + k[173]*y[IDX_N2II];
    IJth(jmatrix, 73, 90) = 0.0 - k[733]*y[IDX_N2I];
    IJth(jmatrix, 73, 91) = 0.0 + k[1053]*y[IDX_NOI];
    IJth(jmatrix, 73, 92) = 0.0 + k[227]*y[IDX_N2II] + k[857]*y[IDX_N2HII];
    IJth(jmatrix, 73, 93) = 0.0 - k[845]*y[IDX_N2I];
    IJth(jmatrix, 74, 24) = 0.0 - k[29]*y[IDX_N2II];
    IJth(jmatrix, 74, 26) = 0.0 - k[58]*y[IDX_N2II];
    IJth(jmatrix, 74, 28) = 0.0 - k[41]*y[IDX_N2II];
    IJth(jmatrix, 74, 33) = 0.0 - k[455]*y[IDX_N2II] - k[456]*y[IDX_N2II];
    IJth(jmatrix, 74, 35) = 0.0 - k[69]*y[IDX_N2II];
    IJth(jmatrix, 74, 36) = 0.0 + k[737]*y[IDX_NI];
    IJth(jmatrix, 74, 37) = 0.0 - k[74]*y[IDX_N2II];
    IJth(jmatrix, 74, 40) = 0.0 - k[337]*y[IDX_N2II];
    IJth(jmatrix, 74, 43) = 0.0 - k[539]*y[IDX_N2II];
    IJth(jmatrix, 74, 46) = 0.0 - k[170]*y[IDX_N2II] - k[730]*y[IDX_N2II];
    IJth(jmatrix, 74, 49) = 0.0 - k[125]*y[IDX_N2II] - k[569]*y[IDX_N2II];
    IJth(jmatrix, 74, 55) = 0.0 - k[135]*y[IDX_N2II];
    IJth(jmatrix, 74, 58) = 0.0 - k[171]*y[IDX_N2II] - k[731]*y[IDX_N2II];
    IJth(jmatrix, 74, 62) = 0.0 + k[144]*y[IDX_N2I];
    IJth(jmatrix, 74, 69) = 0.0 - k[150]*y[IDX_N2II];
    IJth(jmatrix, 74, 71) = 0.0 - k[174]*y[IDX_N2II] + k[737]*y[IDX_CNII] +
        k[740]*y[IDX_NHII] + k[1210]*y[IDX_NII];
    IJth(jmatrix, 74, 72) = 0.0 + k[726]*y[IDX_NHI] + k[727]*y[IDX_NOI] +
        k[1210]*y[IDX_NI];
    IJth(jmatrix, 74, 73) = 0.0 + k[144]*y[IDX_HeII];
    IJth(jmatrix, 74, 74) = 0.0 - k[29]*y[IDX_CI] - k[41]*y[IDX_CH2I] -
        k[58]*y[IDX_CHI] - k[69]*y[IDX_CNI] - k[74]*y[IDX_COI] -
        k[125]*y[IDX_H2OI] - k[135]*y[IDX_HCNI] - k[150]*y[IDX_MgI] -
        k[170]*y[IDX_H2COI] - k[171]*y[IDX_HCOI] - k[172]*y[IDX_NOI] -
        k[173]*y[IDX_O2I] - k[174]*y[IDX_NI] - k[186]*y[IDX_NH2I] -
        k[197]*y[IDX_NH3I] - k[201]*y[IDX_NHI] - k[218]*y[IDX_OI] -
        k[227]*y[IDX_OHI] - k[337]*y[IDX_EM] - k[455]*y[IDX_CH4I] -
        k[456]*y[IDX_CH4I] - k[539]*y[IDX_H2I] - k[569]*y[IDX_H2OI] -
        k[730]*y[IDX_H2COI] - k[731]*y[IDX_HCOI] - k[825]*y[IDX_OI] - k[1271];
    IJth(jmatrix, 74, 76) = 0.0 - k[201]*y[IDX_N2II] + k[726]*y[IDX_NII];
    IJth(jmatrix, 74, 77) = 0.0 + k[740]*y[IDX_NI];
    IJth(jmatrix, 74, 78) = 0.0 - k[186]*y[IDX_N2II];
    IJth(jmatrix, 74, 80) = 0.0 - k[197]*y[IDX_N2II];
    IJth(jmatrix, 74, 82) = 0.0 - k[172]*y[IDX_N2II] + k[727]*y[IDX_NII];
    IJth(jmatrix, 74, 85) = 0.0 - k[218]*y[IDX_N2II] - k[825]*y[IDX_N2II];
    IJth(jmatrix, 74, 87) = 0.0 - k[173]*y[IDX_N2II];
    IJth(jmatrix, 74, 92) = 0.0 - k[227]*y[IDX_N2II];
    IJth(jmatrix, 75, 24) = 0.0 - k[389]*y[IDX_N2HII];
    IJth(jmatrix, 75, 26) = 0.0 - k[469]*y[IDX_N2HII];
    IJth(jmatrix, 75, 28) = 0.0 - k[431]*y[IDX_N2HII];
    IJth(jmatrix, 75, 37) = 0.0 - k[487]*y[IDX_N2HII];
    IJth(jmatrix, 75, 39) = 0.0 - k[734]*y[IDX_N2HII];
    IJth(jmatrix, 75, 40) = 0.0 - k[338]*y[IDX_N2HII] - k[339]*y[IDX_N2HII];
    IJth(jmatrix, 75, 43) = 0.0 + k[539]*y[IDX_N2II];
    IJth(jmatrix, 75, 44) = 0.0 + k[521]*y[IDX_N2I];
    IJth(jmatrix, 75, 46) = 0.0 - k[735]*y[IDX_N2HII];
    IJth(jmatrix, 75, 49) = 0.0 + k[569]*y[IDX_N2II] - k[570]*y[IDX_N2HII];
    IJth(jmatrix, 75, 52) = 0.0 + k[592]*y[IDX_N2I];
    IJth(jmatrix, 75, 55) = 0.0 - k[631]*y[IDX_N2HII];
    IJth(jmatrix, 75, 58) = 0.0 - k[643]*y[IDX_N2HII] + k[731]*y[IDX_N2II];
    IJth(jmatrix, 75, 64) = 0.0 - k[650]*y[IDX_N2HII];
    IJth(jmatrix, 75, 67) = 0.0 + k[732]*y[IDX_N2I];
    IJth(jmatrix, 75, 71) = 0.0 + k[741]*y[IDX_NH2II];
    IJth(jmatrix, 75, 72) = 0.0 + k[724]*y[IDX_NH3I];
    IJth(jmatrix, 75, 73) = 0.0 + k[521]*y[IDX_H2II] + k[592]*y[IDX_H3II] +
        k[732]*y[IDX_HNOII] + k[733]*y[IDX_O2HII] + k[761]*y[IDX_NHII] +
        k[845]*y[IDX_OHII];
    IJth(jmatrix, 75, 74) = 0.0 + k[539]*y[IDX_H2I] + k[569]*y[IDX_H2OI] +
        k[731]*y[IDX_HCOI];
    IJth(jmatrix, 75, 75) = 0.0 - k[338]*y[IDX_EM] - k[339]*y[IDX_EM] -
        k[389]*y[IDX_CI] - k[431]*y[IDX_CH2I] - k[469]*y[IDX_CHI] -
        k[487]*y[IDX_COI] - k[570]*y[IDX_H2OI] - k[631]*y[IDX_HCNI] -
        k[643]*y[IDX_HCOI] - k[650]*y[IDX_HNCI] - k[734]*y[IDX_CO2I] -
        k[735]*y[IDX_H2COI] - k[789]*y[IDX_NH2I] - k[801]*y[IDX_NHI] -
        k[826]*y[IDX_OI] - k[857]*y[IDX_OHI] - k[1302];
    IJth(jmatrix, 75, 76) = 0.0 - k[801]*y[IDX_N2HII];
    IJth(jmatrix, 75, 77) = 0.0 + k[761]*y[IDX_N2I] + k[764]*y[IDX_NOI];
    IJth(jmatrix, 75, 78) = 0.0 - k[789]*y[IDX_N2HII];
    IJth(jmatrix, 75, 79) = 0.0 + k[741]*y[IDX_NI];
    IJth(jmatrix, 75, 80) = 0.0 + k[724]*y[IDX_NII];
    IJth(jmatrix, 75, 82) = 0.0 + k[764]*y[IDX_NHII];
    IJth(jmatrix, 75, 85) = 0.0 - k[826]*y[IDX_N2HII];
    IJth(jmatrix, 75, 90) = 0.0 + k[733]*y[IDX_N2I];
    IJth(jmatrix, 75, 92) = 0.0 - k[857]*y[IDX_N2HII];
    IJth(jmatrix, 75, 93) = 0.0 + k[845]*y[IDX_N2I];
    IJth(jmatrix, 76, 24) = 0.0 + k[868]*y[IDX_NH2I] - k[869]*y[IDX_NHI] -
        k[870]*y[IDX_NHI];
    IJth(jmatrix, 76, 25) = 0.0 - k[375]*y[IDX_NHI];
    IJth(jmatrix, 76, 26) = 0.0 + k[471]*y[IDX_NH2II] + k[929]*y[IDX_NI];
    IJth(jmatrix, 76, 27) = 0.0 - k[410]*y[IDX_NHI];
    IJth(jmatrix, 76, 28) = 0.0 + k[433]*y[IDX_NH2II] + k[884]*y[IDX_N2I] +
        k[1007]*y[IDX_NI];
    IJth(jmatrix, 76, 30) = 0.0 + k[907]*y[IDX_NH2I];
    IJth(jmatrix, 76, 31) = 0.0 - k[794]*y[IDX_NHI];
    IJth(jmatrix, 76, 32) = 0.0 + k[712]*y[IDX_NII] + k[713]*y[IDX_NII];
    IJth(jmatrix, 76, 33) = 0.0 - k[1034]*y[IDX_NHI];
    IJth(jmatrix, 76, 35) = 0.0 - k[1035]*y[IDX_NHI];
    IJth(jmatrix, 76, 36) = 0.0 - k[199]*y[IDX_NHI] + k[561]*y[IDX_H2OI];
    IJth(jmatrix, 76, 37) = 0.0 + k[951]*y[IDX_HNOI];
    IJth(jmatrix, 76, 38) = 0.0 - k[200]*y[IDX_NHI] + k[779]*y[IDX_NH2I] -
        k[795]*y[IDX_NHI];
    IJth(jmatrix, 76, 40) = 0.0 + k[339]*y[IDX_N2HII] + k[342]*y[IDX_NH2II]
        + k[344]*y[IDX_NH3II];
    IJth(jmatrix, 76, 41) = 0.0 + k[982]*y[IDX_HNOI] + k[983]*y[IDX_NH2I] -
        k[985]*y[IDX_NHI] + k[987]*y[IDX_NOI] + k[994]*y[IDX_OCNI];
    IJth(jmatrix, 76, 42) = 0.0 - k[86]*y[IDX_NHI];
    IJth(jmatrix, 76, 43) = 0.0 + k[960]*y[IDX_NI] - k[962]*y[IDX_NHI];
    IJth(jmatrix, 76, 44) = 0.0 - k[111]*y[IDX_NHI] - k[523]*y[IDX_NHI];
    IJth(jmatrix, 76, 45) = 0.0 + k[1013]*y[IDX_NI];
    IJth(jmatrix, 76, 46) = 0.0 + k[175]*y[IDX_NHII] + k[721]*y[IDX_NII] +
        k[769]*y[IDX_NH2II];
    IJth(jmatrix, 76, 47) = 0.0 - k[796]*y[IDX_NHI];
    IJth(jmatrix, 76, 49) = 0.0 + k[176]*y[IDX_NHII] + k[561]*y[IDX_CNII] +
        k[771]*y[IDX_NH2II] - k[1036]*y[IDX_NHI];
    IJth(jmatrix, 76, 50) = 0.0 - k[797]*y[IDX_NHI];
    IJth(jmatrix, 76, 52) = 0.0 - k[594]*y[IDX_NHI];
    IJth(jmatrix, 76, 55) = 0.0 + k[773]*y[IDX_NH2II] + k[1064]*y[IDX_OI];
    IJth(jmatrix, 76, 56) = 0.0 - k[798]*y[IDX_NHI];
    IJth(jmatrix, 76, 58) = 0.0 + k[774]*y[IDX_NH2II] + k[1014]*y[IDX_NI];
    IJth(jmatrix, 76, 59) = 0.0 - k[799]*y[IDX_NHI];
    IJth(jmatrix, 76, 62) = 0.0 - k[693]*y[IDX_NHI];
    IJth(jmatrix, 76, 64) = 0.0 + k[775]*y[IDX_NH2II];
    IJth(jmatrix, 76, 65) = 0.0 + k[263] + k[1152];
    IJth(jmatrix, 76, 66) = 0.0 + k[951]*y[IDX_COI] + k[982]*y[IDX_HI] +
        k[1017]*y[IDX_NI] + k[1070]*y[IDX_OI];
    IJth(jmatrix, 76, 67) = 0.0 - k[800]*y[IDX_NHI];
    IJth(jmatrix, 76, 71) = 0.0 + k[929]*y[IDX_CHI] + k[960]*y[IDX_H2I] +
        k[1007]*y[IDX_CH2I] + k[1013]*y[IDX_H2CNI] + k[1014]*y[IDX_HCOI] +
        k[1017]*y[IDX_HNOI] - k[1018]*y[IDX_NHI] + k[1024]*y[IDX_O2HI] +
        k[1026]*y[IDX_OHI];
    IJth(jmatrix, 76, 72) = 0.0 - k[166]*y[IDX_NHI] + k[712]*y[IDX_CH3OHI] +
        k[713]*y[IDX_CH3OHI] + k[721]*y[IDX_H2COI] + k[725]*y[IDX_NH3I] -
        k[726]*y[IDX_NHI];
    IJth(jmatrix, 76, 73) = 0.0 + k[884]*y[IDX_CH2I];
    IJth(jmatrix, 76, 74) = 0.0 - k[201]*y[IDX_NHI];
    IJth(jmatrix, 76, 75) = 0.0 + k[339]*y[IDX_EM] - k[801]*y[IDX_NHI];
    IJth(jmatrix, 76, 76) = 0.0 - k[86]*y[IDX_HII] - k[111]*y[IDX_H2II] -
        k[166]*y[IDX_NII] - k[199]*y[IDX_CNII] - k[200]*y[IDX_COII] -
        k[201]*y[IDX_N2II] - k[202]*y[IDX_OII] - k[274] - k[275] -
        k[375]*y[IDX_CII] - k[410]*y[IDX_CHII] - k[523]*y[IDX_H2II] -
        k[594]*y[IDX_H3II] - k[693]*y[IDX_HeII] - k[726]*y[IDX_NII] -
        k[763]*y[IDX_NHII] - k[794]*y[IDX_CH3II] - k[795]*y[IDX_COII] -
        k[796]*y[IDX_H2COII] - k[797]*y[IDX_H2OII] - k[798]*y[IDX_HCNII] -
        k[799]*y[IDX_HCOII] - k[800]*y[IDX_HNOII] - k[801]*y[IDX_N2HII] -
        k[802]*y[IDX_NH2II] - k[803]*y[IDX_OII] - k[804]*y[IDX_O2II] -
        k[805]*y[IDX_O2HII] - k[806]*y[IDX_OHII] - k[869]*y[IDX_CI] -
        k[870]*y[IDX_CI] - k[962]*y[IDX_H2I] - k[985]*y[IDX_HI] -
        k[1018]*y[IDX_NI] - k[1034]*y[IDX_CH4I] - k[1035]*y[IDX_CNI] -
        k[1036]*y[IDX_H2OI] - k[1037]*y[IDX_NH3I] - k[1038]*y[IDX_NHI] -
        k[1038]*y[IDX_NHI] - k[1038]*y[IDX_NHI] - k[1038]*y[IDX_NHI] -
        k[1039]*y[IDX_NHI] - k[1039]*y[IDX_NHI] - k[1039]*y[IDX_NHI] -
        k[1039]*y[IDX_NHI] - k[1040]*y[IDX_NHI] - k[1040]*y[IDX_NHI] -
        k[1040]*y[IDX_NHI] - k[1040]*y[IDX_NHI] - k[1041]*y[IDX_NO2I] -
        k[1042]*y[IDX_NOI] - k[1043]*y[IDX_NOI] - k[1044]*y[IDX_O2I] -
        k[1045]*y[IDX_O2I] - k[1046]*y[IDX_OI] - k[1047]*y[IDX_OI] -
        k[1048]*y[IDX_OHI] - k[1049]*y[IDX_OHI] - k[1050]*y[IDX_OHI] - k[1162] -
        k[1163] - k[1265];
    IJth(jmatrix, 76, 77) = 0.0 + k[175]*y[IDX_H2COI] + k[176]*y[IDX_H2OI] +
        k[177]*y[IDX_NH3I] + k[178]*y[IDX_NOI] + k[179]*y[IDX_O2I] -
        k[763]*y[IDX_NHI];
    IJth(jmatrix, 76, 78) = 0.0 + k[270] + k[776]*y[IDX_NH2II] +
        k[779]*y[IDX_COII] + k[868]*y[IDX_CI] + k[907]*y[IDX_CH3I] +
        k[983]*y[IDX_HI] + k[1031]*y[IDX_OHI] + k[1073]*y[IDX_OI] + k[1158];
    IJth(jmatrix, 76, 79) = 0.0 + k[342]*y[IDX_EM] + k[433]*y[IDX_CH2I] +
        k[471]*y[IDX_CHI] + k[769]*y[IDX_H2COI] + k[771]*y[IDX_H2OI] +
        k[773]*y[IDX_HCNI] + k[774]*y[IDX_HCOI] + k[775]*y[IDX_HNCI] +
        k[776]*y[IDX_NH2I] - k[802]*y[IDX_NHI];
    IJth(jmatrix, 76, 80) = 0.0 + k[177]*y[IDX_NHII] + k[273] +
        k[725]*y[IDX_NII] - k[1037]*y[IDX_NHI] + k[1161];
    IJth(jmatrix, 76, 81) = 0.0 + k[344]*y[IDX_EM];
    IJth(jmatrix, 76, 82) = 0.0 + k[178]*y[IDX_NHII] + k[987]*y[IDX_HI] -
        k[1042]*y[IDX_NHI] - k[1043]*y[IDX_NHI];
    IJth(jmatrix, 76, 84) = 0.0 - k[1041]*y[IDX_NHI];
    IJth(jmatrix, 76, 85) = 0.0 - k[1046]*y[IDX_NHI] - k[1047]*y[IDX_NHI] +
        k[1064]*y[IDX_HCNI] + k[1070]*y[IDX_HNOI] + k[1073]*y[IDX_NH2I];
    IJth(jmatrix, 76, 86) = 0.0 - k[202]*y[IDX_NHI] - k[803]*y[IDX_NHI];
    IJth(jmatrix, 76, 87) = 0.0 + k[179]*y[IDX_NHII] - k[1044]*y[IDX_NHI] -
        k[1045]*y[IDX_NHI];
    IJth(jmatrix, 76, 88) = 0.0 - k[804]*y[IDX_NHI];
    IJth(jmatrix, 76, 89) = 0.0 + k[1024]*y[IDX_NI];
    IJth(jmatrix, 76, 90) = 0.0 - k[805]*y[IDX_NHI];
    IJth(jmatrix, 76, 91) = 0.0 + k[994]*y[IDX_HI];
    IJth(jmatrix, 76, 92) = 0.0 + k[1026]*y[IDX_NI] + k[1031]*y[IDX_NH2I] -
        k[1048]*y[IDX_NHI] - k[1049]*y[IDX_NHI] - k[1050]*y[IDX_NHI];
    IJth(jmatrix, 76, 93) = 0.0 - k[806]*y[IDX_NHI];
    IJth(jmatrix, 77, 24) = 0.0 - k[390]*y[IDX_NHII];
    IJth(jmatrix, 77, 26) = 0.0 - k[470]*y[IDX_NHII];
    IJth(jmatrix, 77, 28) = 0.0 - k[432]*y[IDX_NHII];
    IJth(jmatrix, 77, 35) = 0.0 - k[747]*y[IDX_NHII];
    IJth(jmatrix, 77, 36) = 0.0 + k[199]*y[IDX_NHI];
    IJth(jmatrix, 77, 37) = 0.0 - k[751]*y[IDX_NHII];
    IJth(jmatrix, 77, 38) = 0.0 + k[200]*y[IDX_NHI];
    IJth(jmatrix, 77, 39) = 0.0 - k[748]*y[IDX_NHII] - k[749]*y[IDX_NHII] -
        k[750]*y[IDX_NHII];
    IJth(jmatrix, 77, 40) = 0.0 - k[340]*y[IDX_NHII];
    IJth(jmatrix, 77, 42) = 0.0 + k[86]*y[IDX_NHI];
    IJth(jmatrix, 77, 43) = 0.0 + k[538]*y[IDX_NII] - k[540]*y[IDX_NHII] -
        k[541]*y[IDX_NHII];
    IJth(jmatrix, 77, 44) = 0.0 + k[111]*y[IDX_NHI] + k[522]*y[IDX_NI];
    IJth(jmatrix, 77, 46) = 0.0 - k[175]*y[IDX_NHII] - k[752]*y[IDX_NHII] -
        k[753]*y[IDX_NHII];
    IJth(jmatrix, 77, 49) = 0.0 - k[176]*y[IDX_NHII] - k[754]*y[IDX_NHII] -
        k[755]*y[IDX_NHII] - k[756]*y[IDX_NHII] - k[757]*y[IDX_NHII];
    IJth(jmatrix, 77, 55) = 0.0 - k[758]*y[IDX_NHII];
    IJth(jmatrix, 77, 58) = 0.0 + k[723]*y[IDX_NII] - k[759]*y[IDX_NHII];
    IJth(jmatrix, 77, 62) = 0.0 + k[685]*y[IDX_HNCI] + k[690]*y[IDX_NH2I] +
        k[691]*y[IDX_NH3I];
    IJth(jmatrix, 77, 64) = 0.0 + k[685]*y[IDX_HeII] - k[760]*y[IDX_NHII];
    IJth(jmatrix, 77, 71) = 0.0 + k[522]*y[IDX_H2II] - k[740]*y[IDX_NHII];
    IJth(jmatrix, 77, 72) = 0.0 + k[166]*y[IDX_NHI] + k[538]*y[IDX_H2I] +
        k[723]*y[IDX_HCOI];
    IJth(jmatrix, 77, 73) = 0.0 - k[761]*y[IDX_NHII];
    IJth(jmatrix, 77, 74) = 0.0 + k[201]*y[IDX_NHI];
    IJth(jmatrix, 77, 76) = 0.0 + k[86]*y[IDX_HII] + k[111]*y[IDX_H2II] +
        k[166]*y[IDX_NII] + k[199]*y[IDX_CNII] + k[200]*y[IDX_COII] +
        k[201]*y[IDX_N2II] + k[202]*y[IDX_OII] + k[275] - k[763]*y[IDX_NHII] +
        k[1163];
    IJth(jmatrix, 77, 77) = 0.0 - k[175]*y[IDX_H2COI] - k[176]*y[IDX_H2OI] -
        k[177]*y[IDX_NH3I] - k[178]*y[IDX_NOI] - k[179]*y[IDX_O2I] -
        k[340]*y[IDX_EM] - k[390]*y[IDX_CI] - k[432]*y[IDX_CH2I] -
        k[470]*y[IDX_CHI] - k[540]*y[IDX_H2I] - k[541]*y[IDX_H2I] -
        k[740]*y[IDX_NI] - k[747]*y[IDX_CNI] - k[748]*y[IDX_CO2I] -
        k[749]*y[IDX_CO2I] - k[750]*y[IDX_CO2I] - k[751]*y[IDX_COI] -
        k[752]*y[IDX_H2COI] - k[753]*y[IDX_H2COI] - k[754]*y[IDX_H2OI] -
        k[755]*y[IDX_H2OI] - k[756]*y[IDX_H2OI] - k[757]*y[IDX_H2OI] -
        k[758]*y[IDX_HCNI] - k[759]*y[IDX_HCOI] - k[760]*y[IDX_HNCI] -
        k[761]*y[IDX_N2I] - k[762]*y[IDX_NH2I] - k[763]*y[IDX_NHI] -
        k[764]*y[IDX_NOI] - k[765]*y[IDX_O2I] - k[766]*y[IDX_O2I] -
        k[767]*y[IDX_OI] - k[768]*y[IDX_OHI] - k[1156] - k[1273];
    IJth(jmatrix, 77, 78) = 0.0 + k[690]*y[IDX_HeII] - k[762]*y[IDX_NHII];
    IJth(jmatrix, 77, 80) = 0.0 - k[177]*y[IDX_NHII] + k[691]*y[IDX_HeII];
    IJth(jmatrix, 77, 82) = 0.0 - k[178]*y[IDX_NHII] - k[764]*y[IDX_NHII];
    IJth(jmatrix, 77, 85) = 0.0 - k[767]*y[IDX_NHII];
    IJth(jmatrix, 77, 86) = 0.0 + k[202]*y[IDX_NHI];
    IJth(jmatrix, 77, 87) = 0.0 - k[179]*y[IDX_NHII] - k[765]*y[IDX_NHII] -
        k[766]*y[IDX_NHII];
    IJth(jmatrix, 77, 92) = 0.0 - k[768]*y[IDX_NHII];
    IJth(jmatrix, 78, 24) = 0.0 - k[866]*y[IDX_NH2I] - k[867]*y[IDX_NH2I] -
        k[868]*y[IDX_NH2I];
    IJth(jmatrix, 78, 25) = 0.0 - k[373]*y[IDX_NH2I];
    IJth(jmatrix, 78, 26) = 0.0 + k[59]*y[IDX_NH2II];
    IJth(jmatrix, 78, 27) = 0.0 - k[409]*y[IDX_NH2I];
    IJth(jmatrix, 78, 28) = 0.0 + k[42]*y[IDX_NH2II] + k[434]*y[IDX_NH3II];
    IJth(jmatrix, 78, 30) = 0.0 - k[907]*y[IDX_NH2I] + k[908]*y[IDX_NH3I];
    IJth(jmatrix, 78, 33) = 0.0 - k[1028]*y[IDX_NH2I] + k[1034]*y[IDX_NHI];
    IJth(jmatrix, 78, 35) = 0.0 + k[1033]*y[IDX_NH3I];
    IJth(jmatrix, 78, 36) = 0.0 - k[183]*y[IDX_NH2I];
    IJth(jmatrix, 78, 38) = 0.0 - k[184]*y[IDX_NH2I] - k[779]*y[IDX_NH2I] +
        k[792]*y[IDX_NH3I];
    IJth(jmatrix, 78, 40) = 0.0 + k[343]*y[IDX_NH3II];
    IJth(jmatrix, 78, 41) = 0.0 + k[980]*y[IDX_HNOI] - k[983]*y[IDX_NH2I] +
        k[984]*y[IDX_NH3I];
    IJth(jmatrix, 78, 42) = 0.0 - k[84]*y[IDX_NH2I];
    IJth(jmatrix, 78, 43) = 0.0 - k[961]*y[IDX_NH2I] + k[962]*y[IDX_NHI];
    IJth(jmatrix, 78, 44) = 0.0 - k[109]*y[IDX_NH2I];
    IJth(jmatrix, 78, 46) = 0.0 + k[753]*y[IDX_NHII];
    IJth(jmatrix, 78, 47) = 0.0 - k[780]*y[IDX_NH2I];
    IJth(jmatrix, 78, 49) = 0.0 + k[1036]*y[IDX_NHI];
    IJth(jmatrix, 78, 50) = 0.0 - k[185]*y[IDX_NH2I] - k[781]*y[IDX_NH2I];
    IJth(jmatrix, 78, 52) = 0.0 - k[593]*y[IDX_NH2I];
    IJth(jmatrix, 78, 53) = 0.0 - k[782]*y[IDX_NH2I];
    IJth(jmatrix, 78, 54) = 0.0 - k[783]*y[IDX_NH2I];
    IJth(jmatrix, 78, 55) = 0.0 + k[1095]*y[IDX_OHI];
    IJth(jmatrix, 78, 56) = 0.0 - k[784]*y[IDX_NH2I] + k[793]*y[IDX_NH3I];
    IJth(jmatrix, 78, 57) = 0.0 - k[785]*y[IDX_NH2I] - k[786]*y[IDX_NH2I];
    IJth(jmatrix, 78, 58) = 0.0 + k[180]*y[IDX_NH2II];
    IJth(jmatrix, 78, 59) = 0.0 - k[787]*y[IDX_NH2I];
    IJth(jmatrix, 78, 62) = 0.0 - k[689]*y[IDX_NH2I] - k[690]*y[IDX_NH2I];
    IJth(jmatrix, 78, 66) = 0.0 + k[980]*y[IDX_HI];
    IJth(jmatrix, 78, 67) = 0.0 - k[788]*y[IDX_NH2I];
    IJth(jmatrix, 78, 72) = 0.0 - k[164]*y[IDX_NH2I];
    IJth(jmatrix, 78, 74) = 0.0 - k[186]*y[IDX_NH2I];
    IJth(jmatrix, 78, 75) = 0.0 - k[789]*y[IDX_NH2I];
    IJth(jmatrix, 78, 76) = 0.0 + k[962]*y[IDX_H2I] + k[1034]*y[IDX_CH4I] +
        k[1036]*y[IDX_H2OI] + k[1037]*y[IDX_NH3I] + k[1037]*y[IDX_NH3I] +
        k[1040]*y[IDX_NHI] + k[1040]*y[IDX_NHI] + k[1050]*y[IDX_OHI];
    IJth(jmatrix, 78, 77) = 0.0 + k[753]*y[IDX_H2COI] - k[762]*y[IDX_NH2I];
    IJth(jmatrix, 78, 78) = 0.0 - k[84]*y[IDX_HII] - k[109]*y[IDX_H2II] -
        k[164]*y[IDX_NII] - k[183]*y[IDX_CNII] - k[184]*y[IDX_COII] -
        k[185]*y[IDX_H2OII] - k[186]*y[IDX_N2II] - k[187]*y[IDX_O2II] -
        k[188]*y[IDX_OHII] - k[212]*y[IDX_OII] - k[269] - k[270] -
        k[373]*y[IDX_CII] - k[409]*y[IDX_CHII] - k[593]*y[IDX_H3II] -
        k[689]*y[IDX_HeII] - k[690]*y[IDX_HeII] - k[762]*y[IDX_NHII] -
        k[776]*y[IDX_NH2II] - k[779]*y[IDX_COII] - k[780]*y[IDX_H2COII] -
        k[781]*y[IDX_H2OII] - k[782]*y[IDX_H3COII] - k[783]*y[IDX_H3OII] -
        k[784]*y[IDX_HCNII] - k[785]*y[IDX_HCNHII] - k[786]*y[IDX_HCNHII] -
        k[787]*y[IDX_HCOII] - k[788]*y[IDX_HNOII] - k[789]*y[IDX_N2HII] -
        k[790]*y[IDX_O2HII] - k[791]*y[IDX_OHII] - k[866]*y[IDX_CI] -
        k[867]*y[IDX_CI] - k[868]*y[IDX_CI] - k[907]*y[IDX_CH3I] -
        k[961]*y[IDX_H2I] - k[983]*y[IDX_HI] - k[1028]*y[IDX_CH4I] -
        k[1029]*y[IDX_NOI] - k[1030]*y[IDX_NOI] - k[1031]*y[IDX_OHI] -
        k[1032]*y[IDX_OHI] - k[1072]*y[IDX_OI] - k[1073]*y[IDX_OI] - k[1157] -
        k[1158] - k[1289];
    IJth(jmatrix, 78, 79) = 0.0 + k[42]*y[IDX_CH2I] + k[59]*y[IDX_CHI] +
        k[180]*y[IDX_HCOI] + k[181]*y[IDX_NH3I] + k[182]*y[IDX_NOI] -
        k[776]*y[IDX_NH2I];
    IJth(jmatrix, 78, 80) = 0.0 + k[181]*y[IDX_NH2II] + k[271] +
        k[792]*y[IDX_COII] + k[793]*y[IDX_HCNII] + k[908]*y[IDX_CH3I] +
        k[984]*y[IDX_HI] + k[1033]*y[IDX_CNI] + k[1037]*y[IDX_NHI] +
        k[1037]*y[IDX_NHI] + k[1074]*y[IDX_OI] + k[1098]*y[IDX_OHI] + k[1159];
    IJth(jmatrix, 78, 81) = 0.0 + k[343]*y[IDX_EM] + k[434]*y[IDX_CH2I];
    IJth(jmatrix, 78, 82) = 0.0 + k[182]*y[IDX_NH2II] - k[1029]*y[IDX_NH2I]
        - k[1030]*y[IDX_NH2I];
    IJth(jmatrix, 78, 85) = 0.0 - k[1072]*y[IDX_NH2I] - k[1073]*y[IDX_NH2I]
        + k[1074]*y[IDX_NH3I];
    IJth(jmatrix, 78, 86) = 0.0 - k[212]*y[IDX_NH2I];
    IJth(jmatrix, 78, 88) = 0.0 - k[187]*y[IDX_NH2I];
    IJth(jmatrix, 78, 90) = 0.0 - k[790]*y[IDX_NH2I];
    IJth(jmatrix, 78, 92) = 0.0 - k[1031]*y[IDX_NH2I] - k[1032]*y[IDX_NH2I]
        + k[1050]*y[IDX_NHI] + k[1095]*y[IDX_HCNI] + k[1098]*y[IDX_NH3I];
    IJth(jmatrix, 78, 93) = 0.0 - k[188]*y[IDX_NH2I] - k[791]*y[IDX_NH2I];
    IJth(jmatrix, 79, 26) = 0.0 - k[59]*y[IDX_NH2II] - k[471]*y[IDX_NH2II];
    IJth(jmatrix, 79, 28) = 0.0 - k[42]*y[IDX_NH2II] - k[433]*y[IDX_NH2II];
    IJth(jmatrix, 79, 36) = 0.0 + k[183]*y[IDX_NH2I];
    IJth(jmatrix, 79, 38) = 0.0 + k[184]*y[IDX_NH2I];
    IJth(jmatrix, 79, 40) = 0.0 - k[341]*y[IDX_NH2II] - k[342]*y[IDX_NH2II];
    IJth(jmatrix, 79, 42) = 0.0 + k[84]*y[IDX_NH2I] + k[502]*y[IDX_HNCOI];
    IJth(jmatrix, 79, 43) = 0.0 + k[541]*y[IDX_NHII] - k[542]*y[IDX_NH2II];
    IJth(jmatrix, 79, 44) = 0.0 + k[109]*y[IDX_NH2I] + k[523]*y[IDX_NHI];
    IJth(jmatrix, 79, 46) = 0.0 - k[769]*y[IDX_NH2II] - k[770]*y[IDX_NH2II];
    IJth(jmatrix, 79, 49) = 0.0 + k[757]*y[IDX_NHII] - k[771]*y[IDX_NH2II] -
        k[772]*y[IDX_NH2II];
    IJth(jmatrix, 79, 50) = 0.0 + k[185]*y[IDX_NH2I];
    IJth(jmatrix, 79, 52) = 0.0 + k[594]*y[IDX_NHI];
    IJth(jmatrix, 79, 55) = 0.0 - k[773]*y[IDX_NH2II];
    IJth(jmatrix, 79, 56) = 0.0 + k[798]*y[IDX_NHI];
    IJth(jmatrix, 79, 58) = 0.0 - k[180]*y[IDX_NH2II] - k[774]*y[IDX_NH2II];
    IJth(jmatrix, 79, 59) = 0.0 + k[799]*y[IDX_NHI];
    IJth(jmatrix, 79, 62) = 0.0 + k[692]*y[IDX_NH3I];
    IJth(jmatrix, 79, 64) = 0.0 - k[775]*y[IDX_NH2II];
    IJth(jmatrix, 79, 65) = 0.0 + k[502]*y[IDX_HII];
    IJth(jmatrix, 79, 67) = 0.0 + k[800]*y[IDX_NHI];
    IJth(jmatrix, 79, 71) = 0.0 - k[741]*y[IDX_NH2II];
    IJth(jmatrix, 79, 72) = 0.0 + k[164]*y[IDX_NH2I] + k[725]*y[IDX_NH3I];
    IJth(jmatrix, 79, 74) = 0.0 + k[186]*y[IDX_NH2I];
    IJth(jmatrix, 79, 75) = 0.0 + k[801]*y[IDX_NHI];
    IJth(jmatrix, 79, 76) = 0.0 + k[523]*y[IDX_H2II] + k[594]*y[IDX_H3II] +
        k[763]*y[IDX_NHII] + k[798]*y[IDX_HCNII] + k[799]*y[IDX_HCOII] +
        k[800]*y[IDX_HNOII] + k[801]*y[IDX_N2HII] - k[802]*y[IDX_NH2II] +
        k[805]*y[IDX_O2HII] + k[806]*y[IDX_OHII];
    IJth(jmatrix, 79, 77) = 0.0 + k[541]*y[IDX_H2I] + k[757]*y[IDX_H2OI] +
        k[763]*y[IDX_NHI];
    IJth(jmatrix, 79, 78) = 0.0 + k[84]*y[IDX_HII] + k[109]*y[IDX_H2II] +
        k[164]*y[IDX_NII] + k[183]*y[IDX_CNII] + k[184]*y[IDX_COII] +
        k[185]*y[IDX_H2OII] + k[186]*y[IDX_N2II] + k[187]*y[IDX_O2II] +
        k[188]*y[IDX_OHII] + k[212]*y[IDX_OII] + k[269] - k[776]*y[IDX_NH2II] +
        k[1157];
    IJth(jmatrix, 79, 79) = 0.0 - k[42]*y[IDX_CH2I] - k[59]*y[IDX_CHI] -
        k[180]*y[IDX_HCOI] - k[181]*y[IDX_NH3I] - k[182]*y[IDX_NOI] -
        k[341]*y[IDX_EM] - k[342]*y[IDX_EM] - k[433]*y[IDX_CH2I] -
        k[471]*y[IDX_CHI] - k[542]*y[IDX_H2I] - k[741]*y[IDX_NI] -
        k[769]*y[IDX_H2COI] - k[770]*y[IDX_H2COI] - k[771]*y[IDX_H2OI] -
        k[772]*y[IDX_H2OI] - k[773]*y[IDX_HCNI] - k[774]*y[IDX_HCOI] -
        k[775]*y[IDX_HNCI] - k[776]*y[IDX_NH2I] - k[777]*y[IDX_O2I] -
        k[778]*y[IDX_O2I] - k[802]*y[IDX_NHI] - k[827]*y[IDX_OI] - k[1279];
    IJth(jmatrix, 79, 80) = 0.0 - k[181]*y[IDX_NH2II] + k[692]*y[IDX_HeII] +
        k[725]*y[IDX_NII];
    IJth(jmatrix, 79, 82) = 0.0 - k[182]*y[IDX_NH2II];
    IJth(jmatrix, 79, 85) = 0.0 - k[827]*y[IDX_NH2II];
    IJth(jmatrix, 79, 86) = 0.0 + k[212]*y[IDX_NH2I];
    IJth(jmatrix, 79, 87) = 0.0 - k[777]*y[IDX_NH2II] - k[778]*y[IDX_NH2II];
    IJth(jmatrix, 79, 88) = 0.0 + k[187]*y[IDX_NH2I];
    IJth(jmatrix, 79, 90) = 0.0 + k[805]*y[IDX_NHI];
    IJth(jmatrix, 79, 93) = 0.0 + k[188]*y[IDX_NH2I] + k[806]*y[IDX_NHI];
    IJth(jmatrix, 80, 14) = 0.0 + k[1311] + k[1312] + k[1313] + k[1314];
    IJth(jmatrix, 80, 25) = 0.0 - k[19]*y[IDX_NH3I] - k[374]*y[IDX_NH3I];
    IJth(jmatrix, 80, 27) = 0.0 - k[33]*y[IDX_NH3I];
    IJth(jmatrix, 80, 30) = 0.0 - k[908]*y[IDX_NH3I];
    IJth(jmatrix, 80, 33) = 0.0 + k[1028]*y[IDX_NH2I];
    IJth(jmatrix, 80, 34) = 0.0 - k[50]*y[IDX_NH3I];
    IJth(jmatrix, 80, 35) = 0.0 - k[1033]*y[IDX_NH3I];
    IJth(jmatrix, 80, 38) = 0.0 - k[193]*y[IDX_NH3I] - k[792]*y[IDX_NH3I];
    IJth(jmatrix, 80, 41) = 0.0 - k[984]*y[IDX_NH3I];
    IJth(jmatrix, 80, 42) = 0.0 - k[85]*y[IDX_NH3I];
    IJth(jmatrix, 80, 43) = 0.0 + k[961]*y[IDX_NH2I];
    IJth(jmatrix, 80, 44) = 0.0 - k[110]*y[IDX_NH3I];
    IJth(jmatrix, 80, 47) = 0.0 - k[194]*y[IDX_NH3I];
    IJth(jmatrix, 80, 50) = 0.0 - k[195]*y[IDX_NH3I];
    IJth(jmatrix, 80, 56) = 0.0 - k[196]*y[IDX_NH3I] - k[793]*y[IDX_NH3I];
    IJth(jmatrix, 80, 58) = 0.0 + k[189]*y[IDX_NH3II];
    IJth(jmatrix, 80, 62) = 0.0 - k[145]*y[IDX_NH3I] - k[691]*y[IDX_NH3I] -
        k[692]*y[IDX_NH3I];
    IJth(jmatrix, 80, 69) = 0.0 + k[190]*y[IDX_NH3II];
    IJth(jmatrix, 80, 72) = 0.0 - k[165]*y[IDX_NH3I] - k[724]*y[IDX_NH3I] -
        k[725]*y[IDX_NH3I];
    IJth(jmatrix, 80, 74) = 0.0 - k[197]*y[IDX_NH3I];
    IJth(jmatrix, 80, 76) = 0.0 - k[1037]*y[IDX_NH3I];
    IJth(jmatrix, 80, 77) = 0.0 - k[177]*y[IDX_NH3I];
    IJth(jmatrix, 80, 78) = 0.0 + k[961]*y[IDX_H2I] + k[1028]*y[IDX_CH4I] +
        k[1032]*y[IDX_OHI];
    IJth(jmatrix, 80, 79) = 0.0 - k[181]*y[IDX_NH3I];
    IJth(jmatrix, 80, 80) = 0.0 - k[19]*y[IDX_CII] - k[33]*y[IDX_CHII] -
        k[50]*y[IDX_CH4II] - k[85]*y[IDX_HII] - k[110]*y[IDX_H2II] -
        k[145]*y[IDX_HeII] - k[165]*y[IDX_NII] - k[177]*y[IDX_NHII] -
        k[181]*y[IDX_NH2II] - k[193]*y[IDX_COII] - k[194]*y[IDX_H2COII] -
        k[195]*y[IDX_H2OII] - k[196]*y[IDX_HCNII] - k[197]*y[IDX_N2II] -
        k[198]*y[IDX_O2II] - k[213]*y[IDX_OII] - k[222]*y[IDX_OHII] - k[271] -
        k[272] - k[273] - k[374]*y[IDX_CII] - k[691]*y[IDX_HeII] -
        k[692]*y[IDX_HeII] - k[724]*y[IDX_NII] - k[725]*y[IDX_NII] -
        k[792]*y[IDX_COII] - k[793]*y[IDX_HCNII] - k[908]*y[IDX_CH3I] -
        k[984]*y[IDX_HI] - k[1033]*y[IDX_CNI] - k[1037]*y[IDX_NHI] -
        k[1074]*y[IDX_OI] - k[1098]*y[IDX_OHI] - k[1159] - k[1160] - k[1161] -
        k[1267];
    IJth(jmatrix, 80, 81) = 0.0 + k[189]*y[IDX_HCOI] + k[190]*y[IDX_MgI] +
        k[191]*y[IDX_NOI] + k[192]*y[IDX_SiI];
    IJth(jmatrix, 80, 82) = 0.0 + k[191]*y[IDX_NH3II];
    IJth(jmatrix, 80, 85) = 0.0 - k[1074]*y[IDX_NH3I];
    IJth(jmatrix, 80, 86) = 0.0 - k[213]*y[IDX_NH3I];
    IJth(jmatrix, 80, 88) = 0.0 - k[198]*y[IDX_NH3I];
    IJth(jmatrix, 80, 92) = 0.0 + k[1032]*y[IDX_NH2I] - k[1098]*y[IDX_NH3I];
    IJth(jmatrix, 80, 93) = 0.0 - k[222]*y[IDX_NH3I];
    IJth(jmatrix, 80, 94) = 0.0 + k[192]*y[IDX_NH3II];
    IJth(jmatrix, 81, 25) = 0.0 + k[19]*y[IDX_NH3I];
    IJth(jmatrix, 81, 27) = 0.0 + k[33]*y[IDX_NH3I];
    IJth(jmatrix, 81, 28) = 0.0 - k[434]*y[IDX_NH3II];
    IJth(jmatrix, 81, 34) = 0.0 + k[50]*y[IDX_NH3I];
    IJth(jmatrix, 81, 38) = 0.0 + k[193]*y[IDX_NH3I];
    IJth(jmatrix, 81, 40) = 0.0 - k[343]*y[IDX_NH3II] - k[344]*y[IDX_NH3II];
    IJth(jmatrix, 81, 42) = 0.0 + k[85]*y[IDX_NH3I];
    IJth(jmatrix, 81, 43) = 0.0 + k[542]*y[IDX_NH2II];
    IJth(jmatrix, 81, 44) = 0.0 + k[110]*y[IDX_NH3I];
    IJth(jmatrix, 81, 46) = 0.0 + k[770]*y[IDX_NH2II];
    IJth(jmatrix, 81, 47) = 0.0 + k[194]*y[IDX_NH3I] + k[780]*y[IDX_NH2I];
    IJth(jmatrix, 81, 49) = 0.0 + k[756]*y[IDX_NHII] + k[772]*y[IDX_NH2II];
    IJth(jmatrix, 81, 50) = 0.0 + k[195]*y[IDX_NH3I] + k[781]*y[IDX_NH2I];
    IJth(jmatrix, 81, 52) = 0.0 + k[593]*y[IDX_NH2I];
    IJth(jmatrix, 81, 53) = 0.0 + k[782]*y[IDX_NH2I];
    IJth(jmatrix, 81, 54) = 0.0 + k[783]*y[IDX_NH2I];
    IJth(jmatrix, 81, 56) = 0.0 + k[196]*y[IDX_NH3I] + k[784]*y[IDX_NH2I];
    IJth(jmatrix, 81, 57) = 0.0 + k[785]*y[IDX_NH2I] + k[786]*y[IDX_NH2I];
    IJth(jmatrix, 81, 58) = 0.0 - k[189]*y[IDX_NH3II];
    IJth(jmatrix, 81, 59) = 0.0 + k[787]*y[IDX_NH2I];
    IJth(jmatrix, 81, 62) = 0.0 + k[145]*y[IDX_NH3I];
    IJth(jmatrix, 81, 67) = 0.0 + k[788]*y[IDX_NH2I];
    IJth(jmatrix, 81, 69) = 0.0 - k[190]*y[IDX_NH3II];
    IJth(jmatrix, 81, 72) = 0.0 + k[165]*y[IDX_NH3I];
    IJth(jmatrix, 81, 74) = 0.0 + k[197]*y[IDX_NH3I];
    IJth(jmatrix, 81, 75) = 0.0 + k[789]*y[IDX_NH2I];
    IJth(jmatrix, 81, 76) = 0.0 + k[802]*y[IDX_NH2II];
    IJth(jmatrix, 81, 77) = 0.0 + k[177]*y[IDX_NH3I] + k[756]*y[IDX_H2OI] +
        k[762]*y[IDX_NH2I];
    IJth(jmatrix, 81, 78) = 0.0 + k[593]*y[IDX_H3II] + k[762]*y[IDX_NHII] +
        k[776]*y[IDX_NH2II] + k[780]*y[IDX_H2COII] + k[781]*y[IDX_H2OII] +
        k[782]*y[IDX_H3COII] + k[783]*y[IDX_H3OII] + k[784]*y[IDX_HCNII] +
        k[785]*y[IDX_HCNHII] + k[786]*y[IDX_HCNHII] + k[787]*y[IDX_HCOII] +
        k[788]*y[IDX_HNOII] + k[789]*y[IDX_N2HII] + k[790]*y[IDX_O2HII] +
        k[791]*y[IDX_OHII];
    IJth(jmatrix, 81, 79) = 0.0 + k[181]*y[IDX_NH3I] + k[542]*y[IDX_H2I] +
        k[770]*y[IDX_H2COI] + k[772]*y[IDX_H2OI] + k[776]*y[IDX_NH2I] +
        k[802]*y[IDX_NHI];
    IJth(jmatrix, 81, 80) = 0.0 + k[19]*y[IDX_CII] + k[33]*y[IDX_CHII] +
        k[50]*y[IDX_CH4II] + k[85]*y[IDX_HII] + k[110]*y[IDX_H2II] +
        k[145]*y[IDX_HeII] + k[165]*y[IDX_NII] + k[177]*y[IDX_NHII] +
        k[181]*y[IDX_NH2II] + k[193]*y[IDX_COII] + k[194]*y[IDX_H2COII] +
        k[195]*y[IDX_H2OII] + k[196]*y[IDX_HCNII] + k[197]*y[IDX_N2II] +
        k[198]*y[IDX_O2II] + k[213]*y[IDX_OII] + k[222]*y[IDX_OHII] + k[272] +
        k[1160];
    IJth(jmatrix, 81, 81) = 0.0 - k[189]*y[IDX_HCOI] - k[190]*y[IDX_MgI] -
        k[191]*y[IDX_NOI] - k[192]*y[IDX_SiI] - k[343]*y[IDX_EM] -
        k[344]*y[IDX_EM] - k[434]*y[IDX_CH2I] - k[828]*y[IDX_OI] - k[1283];
    IJth(jmatrix, 81, 82) = 0.0 - k[191]*y[IDX_NH3II];
    IJth(jmatrix, 81, 85) = 0.0 - k[828]*y[IDX_NH3II];
    IJth(jmatrix, 81, 86) = 0.0 + k[213]*y[IDX_NH3I];
    IJth(jmatrix, 81, 88) = 0.0 + k[198]*y[IDX_NH3I];
    IJth(jmatrix, 81, 90) = 0.0 + k[790]*y[IDX_NH2I];
    IJth(jmatrix, 81, 93) = 0.0 + k[222]*y[IDX_NH3I] + k[791]*y[IDX_NH2I];
    IJth(jmatrix, 81, 94) = 0.0 - k[192]*y[IDX_NH3II];
    IJth(jmatrix, 82, 15) = 0.0 + k[1343] + k[1344] + k[1345] + k[1346];
    IJth(jmatrix, 82, 24) = 0.0 + k[388]*y[IDX_HNOII] - k[871]*y[IDX_NOI] -
        k[872]*y[IDX_NOI];
    IJth(jmatrix, 82, 25) = 0.0 - k[20]*y[IDX_NOI];
    IJth(jmatrix, 82, 26) = 0.0 + k[467]*y[IDX_HNOII] + k[926]*y[IDX_HNOI] -
        k[930]*y[IDX_NOI] - k[931]*y[IDX_NOI] - k[932]*y[IDX_NOI];
    IJth(jmatrix, 82, 27) = 0.0 - k[34]*y[IDX_NOI];
    IJth(jmatrix, 82, 28) = 0.0 + k[430]*y[IDX_HNOII] + k[883]*y[IDX_HNOI] +
        k[885]*y[IDX_NO2I] - k[886]*y[IDX_NOI] - k[887]*y[IDX_NOI] -
        k[888]*y[IDX_NOI];
    IJth(jmatrix, 82, 29) = 0.0 - k[36]*y[IDX_NOI];
    IJth(jmatrix, 82, 30) = 0.0 + k[906]*y[IDX_HNOI] - k[910]*y[IDX_NOI];
    IJth(jmatrix, 82, 31) = 0.0 - k[48]*y[IDX_NOI];
    IJth(jmatrix, 82, 32) = 0.0 + k[715]*y[IDX_NII];
    IJth(jmatrix, 82, 35) = 0.0 + k[482]*y[IDX_HNOII] + k[944]*y[IDX_HNOI] +
        k[945]*y[IDX_NO2I] - k[946]*y[IDX_NOI] - k[947]*y[IDX_NOI] +
        k[948]*y[IDX_O2I] + k[1058]*y[IDX_OI];
    IJth(jmatrix, 82, 36) = 0.0 - k[67]*y[IDX_NOI];
    IJth(jmatrix, 82, 37) = 0.0 + k[486]*y[IDX_HNOII] + k[952]*y[IDX_NO2I];
    IJth(jmatrix, 82, 38) = 0.0 - k[72]*y[IDX_NOI];
    IJth(jmatrix, 82, 39) = 0.0 + k[652]*y[IDX_HNOII] + k[719]*y[IDX_NII] +
        k[1012]*y[IDX_NI];
    IJth(jmatrix, 82, 40) = 0.0 + k[311]*y[IDX_H2NOII] +
        k[334]*y[IDX_HNOII];
    IJth(jmatrix, 82, 41) = 0.0 + k[981]*y[IDX_HNOI] + k[986]*y[IDX_NO2I] -
        k[987]*y[IDX_NOI] - k[988]*y[IDX_NOI];
    IJth(jmatrix, 82, 42) = 0.0 - k[87]*y[IDX_NOI];
    IJth(jmatrix, 82, 44) = 0.0 - k[112]*y[IDX_NOI] - k[524]*y[IDX_NOI];
    IJth(jmatrix, 82, 46) = 0.0 + k[550]*y[IDX_HNOII];
    IJth(jmatrix, 82, 47) = 0.0 - k[203]*y[IDX_NOI];
    IJth(jmatrix, 82, 48) = 0.0 + k[311]*y[IDX_EM];
    IJth(jmatrix, 82, 49) = 0.0 + k[568]*y[IDX_HNOII];
    IJth(jmatrix, 82, 50) = 0.0 - k[120]*y[IDX_NOI];
    IJth(jmatrix, 82, 52) = 0.0 - k[596]*y[IDX_NOI];
    IJth(jmatrix, 82, 55) = 0.0 + k[630]*y[IDX_HNOII];
    IJth(jmatrix, 82, 56) = 0.0 - k[132]*y[IDX_NOI];
    IJth(jmatrix, 82, 58) = 0.0 + k[642]*y[IDX_HNOII] + k[999]*y[IDX_HNOI] -
        k[1000]*y[IDX_NOI];
    IJth(jmatrix, 82, 62) = 0.0 + k[687]*y[IDX_HNOI] - k[694]*y[IDX_NOI] -
        k[695]*y[IDX_NOI];
    IJth(jmatrix, 82, 64) = 0.0 + k[649]*y[IDX_HNOII];
    IJth(jmatrix, 82, 66) = 0.0 + k[264] + k[687]*y[IDX_HeII] +
        k[883]*y[IDX_CH2I] + k[906]*y[IDX_CH3I] + k[926]*y[IDX_CHI] +
        k[944]*y[IDX_CNI] + k[981]*y[IDX_HI] + k[999]*y[IDX_HCOI] +
        k[1017]*y[IDX_NI] + k[1069]*y[IDX_OI] + k[1097]*y[IDX_OHI] + k[1153];
    IJth(jmatrix, 82, 67) = 0.0 - k[204]*y[IDX_NOI] + k[334]*y[IDX_EM] +
        k[388]*y[IDX_CI] + k[430]*y[IDX_CH2I] + k[467]*y[IDX_CHI] +
        k[482]*y[IDX_CNI] + k[486]*y[IDX_COI] + k[550]*y[IDX_H2COI] +
        k[568]*y[IDX_H2OI] + k[630]*y[IDX_HCNI] + k[642]*y[IDX_HCOI] +
        k[649]*y[IDX_HNCI] + k[652]*y[IDX_CO2I] + k[732]*y[IDX_N2I] +
        k[788]*y[IDX_NH2I] + k[800]*y[IDX_NHI] + k[856]*y[IDX_OHI];
    IJth(jmatrix, 82, 69) = 0.0 + k[151]*y[IDX_NOII];
    IJth(jmatrix, 82, 71) = 0.0 + k[746]*y[IDX_SiOII] + k[1012]*y[IDX_CO2I]
        + k[1017]*y[IDX_HNOI] + k[1020]*y[IDX_NO2I] + k[1020]*y[IDX_NO2I] -
        k[1022]*y[IDX_NOI] + k[1023]*y[IDX_O2I] + k[1025]*y[IDX_OHI];
    IJth(jmatrix, 82, 72) = 0.0 - k[167]*y[IDX_NOI] + k[715]*y[IDX_CH3OHI] +
        k[719]*y[IDX_CO2I] - k[727]*y[IDX_NOI] + k[729]*y[IDX_O2I];
    IJth(jmatrix, 82, 73) = 0.0 + k[732]*y[IDX_HNOII] + k[1071]*y[IDX_OI];
    IJth(jmatrix, 82, 74) = 0.0 - k[172]*y[IDX_NOI];
    IJth(jmatrix, 82, 76) = 0.0 + k[800]*y[IDX_HNOII] + k[1041]*y[IDX_NO2I]
        - k[1042]*y[IDX_NOI] - k[1043]*y[IDX_NOI] + k[1045]*y[IDX_O2I] +
        k[1046]*y[IDX_OI];
    IJth(jmatrix, 82, 77) = 0.0 - k[178]*y[IDX_NOI] - k[764]*y[IDX_NOI];
    IJth(jmatrix, 82, 78) = 0.0 + k[788]*y[IDX_HNOII] - k[1029]*y[IDX_NOI] -
        k[1030]*y[IDX_NOI];
    IJth(jmatrix, 82, 79) = 0.0 - k[182]*y[IDX_NOI];
    IJth(jmatrix, 82, 81) = 0.0 - k[191]*y[IDX_NOI];
    IJth(jmatrix, 82, 82) = 0.0 - k[20]*y[IDX_CII] - k[34]*y[IDX_CHII] -
        k[36]*y[IDX_CH2II] - k[48]*y[IDX_CH3II] - k[67]*y[IDX_CNII] -
        k[72]*y[IDX_COII] - k[87]*y[IDX_HII] - k[112]*y[IDX_H2II] -
        k[120]*y[IDX_H2OII] - k[132]*y[IDX_HCNII] - k[167]*y[IDX_NII] -
        k[172]*y[IDX_N2II] - k[178]*y[IDX_NHII] - k[182]*y[IDX_NH2II] -
        k[191]*y[IDX_NH3II] - k[203]*y[IDX_H2COII] - k[204]*y[IDX_HNOII] -
        k[205]*y[IDX_O2II] - k[206]*y[IDX_SiOII] - k[223]*y[IDX_OHII] - k[277] -
        k[278] - k[524]*y[IDX_H2II] - k[596]*y[IDX_H3II] - k[694]*y[IDX_HeII] -
        k[695]*y[IDX_HeII] - k[727]*y[IDX_NII] - k[764]*y[IDX_NHII] -
        k[807]*y[IDX_O2HII] - k[846]*y[IDX_OHII] - k[871]*y[IDX_CI] -
        k[872]*y[IDX_CI] - k[886]*y[IDX_CH2I] - k[887]*y[IDX_CH2I] -
        k[888]*y[IDX_CH2I] - k[910]*y[IDX_CH3I] - k[930]*y[IDX_CHI] -
        k[931]*y[IDX_CHI] - k[932]*y[IDX_CHI] - k[946]*y[IDX_CNI] -
        k[947]*y[IDX_CNI] - k[987]*y[IDX_HI] - k[988]*y[IDX_HI] -
        k[1000]*y[IDX_HCOI] - k[1022]*y[IDX_NI] - k[1029]*y[IDX_NH2I] -
        k[1030]*y[IDX_NH2I] - k[1042]*y[IDX_NHI] - k[1043]*y[IDX_NHI] -
        k[1051]*y[IDX_NOI] - k[1051]*y[IDX_NOI] - k[1051]*y[IDX_NOI] -
        k[1051]*y[IDX_NOI] - k[1052]*y[IDX_O2I] - k[1053]*y[IDX_OCNI] -
        k[1076]*y[IDX_OI] - k[1099]*y[IDX_OHI] - k[1105]*y[IDX_SiI] - k[1165] -
        k[1166] - k[1255];
    IJth(jmatrix, 82, 83) = 0.0 + k[151]*y[IDX_MgI] + k[229]*y[IDX_SiI];
    IJth(jmatrix, 82, 84) = 0.0 + k[276] + k[885]*y[IDX_CH2I] +
        k[945]*y[IDX_CNI] + k[952]*y[IDX_COI] + k[986]*y[IDX_HI] +
        k[1020]*y[IDX_NI] + k[1020]*y[IDX_NI] + k[1041]*y[IDX_NHI] +
        k[1075]*y[IDX_OI] + k[1164];
    IJth(jmatrix, 82, 85) = 0.0 + k[1046]*y[IDX_NHI] + k[1058]*y[IDX_CNI] +
        k[1069]*y[IDX_HNOI] + k[1071]*y[IDX_N2I] + k[1075]*y[IDX_NO2I] -
        k[1076]*y[IDX_NOI] + k[1078]*y[IDX_OCNI];
    IJth(jmatrix, 82, 87) = 0.0 + k[729]*y[IDX_NII] + k[948]*y[IDX_CNI] +
        k[1023]*y[IDX_NI] + k[1045]*y[IDX_NHI] - k[1052]*y[IDX_NOI] +
        k[1054]*y[IDX_OCNI];
    IJth(jmatrix, 82, 88) = 0.0 - k[205]*y[IDX_NOI];
    IJth(jmatrix, 82, 90) = 0.0 - k[807]*y[IDX_NOI];
    IJth(jmatrix, 82, 91) = 0.0 - k[1053]*y[IDX_NOI] + k[1054]*y[IDX_O2I] +
        k[1078]*y[IDX_OI];
    IJth(jmatrix, 82, 92) = 0.0 + k[856]*y[IDX_HNOII] + k[1025]*y[IDX_NI] +
        k[1097]*y[IDX_HNOI] - k[1099]*y[IDX_NOI];
    IJth(jmatrix, 82, 93) = 0.0 - k[223]*y[IDX_NOI] - k[846]*y[IDX_NOI];
    IJth(jmatrix, 82, 94) = 0.0 + k[229]*y[IDX_NOII] - k[1105]*y[IDX_NOI];
    IJth(jmatrix, 82, 112) = 0.0 - k[206]*y[IDX_NOI] + k[746]*y[IDX_NI];
    IJth(jmatrix, 83, 25) = 0.0 + k[20]*y[IDX_NOI];
    IJth(jmatrix, 83, 27) = 0.0 + k[34]*y[IDX_NOI];
    IJth(jmatrix, 83, 29) = 0.0 + k[36]*y[IDX_NOI];
    IJth(jmatrix, 83, 31) = 0.0 + k[48]*y[IDX_NOI];
    IJth(jmatrix, 83, 32) = 0.0 + k[714]*y[IDX_NII];
    IJth(jmatrix, 83, 35) = 0.0 + k[811]*y[IDX_OII];
    IJth(jmatrix, 83, 36) = 0.0 + k[67]*y[IDX_NOI] + k[481]*y[IDX_O2I];
    IJth(jmatrix, 83, 37) = 0.0 + k[720]*y[IDX_NII];
    IJth(jmatrix, 83, 38) = 0.0 + k[72]*y[IDX_NOI];
    IJth(jmatrix, 83, 39) = 0.0 + k[750]*y[IDX_NHII];
    IJth(jmatrix, 83, 40) = 0.0 - k[345]*y[IDX_NOII];
    IJth(jmatrix, 83, 42) = 0.0 + k[87]*y[IDX_NOI] + k[503]*y[IDX_HNOI] +
        k[504]*y[IDX_NO2I];
    IJth(jmatrix, 83, 44) = 0.0 + k[112]*y[IDX_NOI];
    IJth(jmatrix, 83, 46) = 0.0 + k[722]*y[IDX_NII];
    IJth(jmatrix, 83, 47) = 0.0 + k[203]*y[IDX_NOI];
    IJth(jmatrix, 83, 50) = 0.0 + k[120]*y[IDX_NOI] + k[739]*y[IDX_NI];
    IJth(jmatrix, 83, 52) = 0.0 + k[595]*y[IDX_NO2I];
    IJth(jmatrix, 83, 55) = 0.0 + k[815]*y[IDX_OII];
    IJth(jmatrix, 83, 56) = 0.0 + k[132]*y[IDX_NOI];
    IJth(jmatrix, 83, 62) = 0.0 + k[686]*y[IDX_HNOI];
    IJth(jmatrix, 83, 66) = 0.0 + k[503]*y[IDX_HII] + k[686]*y[IDX_HeII];
    IJth(jmatrix, 83, 67) = 0.0 + k[204]*y[IDX_NOI];
    IJth(jmatrix, 83, 69) = 0.0 - k[151]*y[IDX_NOII];
    IJth(jmatrix, 83, 71) = 0.0 + k[739]*y[IDX_H2OII] + k[742]*y[IDX_O2II] +
        k[743]*y[IDX_OHII] + k[745]*y[IDX_SiOII];
    IJth(jmatrix, 83, 72) = 0.0 + k[167]*y[IDX_NOI] + k[714]*y[IDX_CH3OHI] +
        k[720]*y[IDX_COI] + k[722]*y[IDX_H2COI] + k[728]*y[IDX_O2I];
    IJth(jmatrix, 83, 73) = 0.0 + k[817]*y[IDX_OII];
    IJth(jmatrix, 83, 74) = 0.0 + k[172]*y[IDX_NOI] + k[825]*y[IDX_OI];
    IJth(jmatrix, 83, 76) = 0.0 + k[803]*y[IDX_OII];
    IJth(jmatrix, 83, 77) = 0.0 + k[178]*y[IDX_NOI] + k[750]*y[IDX_CO2I] +
        k[765]*y[IDX_O2I];
    IJth(jmatrix, 83, 79) = 0.0 + k[182]*y[IDX_NOI];
    IJth(jmatrix, 83, 81) = 0.0 + k[191]*y[IDX_NOI];
    IJth(jmatrix, 83, 82) = 0.0 + k[20]*y[IDX_CII] + k[34]*y[IDX_CHII] +
        k[36]*y[IDX_CH2II] + k[48]*y[IDX_CH3II] + k[67]*y[IDX_CNII] +
        k[72]*y[IDX_COII] + k[87]*y[IDX_HII] + k[112]*y[IDX_H2II] +
        k[120]*y[IDX_H2OII] + k[132]*y[IDX_HCNII] + k[167]*y[IDX_NII] +
        k[172]*y[IDX_N2II] + k[178]*y[IDX_NHII] + k[182]*y[IDX_NH2II] +
        k[191]*y[IDX_NH3II] + k[203]*y[IDX_H2COII] + k[204]*y[IDX_HNOII] +
        k[205]*y[IDX_O2II] + k[206]*y[IDX_SiOII] + k[223]*y[IDX_OHII] + k[277] +
        k[1165];
    IJth(jmatrix, 83, 83) = 0.0 - k[151]*y[IDX_MgI] - k[229]*y[IDX_SiI] -
        k[345]*y[IDX_EM] - k[1277];
    IJth(jmatrix, 83, 84) = 0.0 + k[504]*y[IDX_HII] + k[595]*y[IDX_H3II] +
        k[818]*y[IDX_OII];
    IJth(jmatrix, 83, 85) = 0.0 + k[825]*y[IDX_N2II];
    IJth(jmatrix, 83, 86) = 0.0 + k[803]*y[IDX_NHI] + k[811]*y[IDX_CNI] +
        k[815]*y[IDX_HCNI] + k[817]*y[IDX_N2I] + k[818]*y[IDX_NO2I];
    IJth(jmatrix, 83, 87) = 0.0 + k[481]*y[IDX_CNII] + k[728]*y[IDX_NII] +
        k[765]*y[IDX_NHII];
    IJth(jmatrix, 83, 88) = 0.0 + k[205]*y[IDX_NOI] + k[742]*y[IDX_NI];
    IJth(jmatrix, 83, 93) = 0.0 + k[223]*y[IDX_NOI] + k[743]*y[IDX_NI];
    IJth(jmatrix, 83, 94) = 0.0 - k[229]*y[IDX_NOII];
    IJth(jmatrix, 83, 112) = 0.0 + k[206]*y[IDX_NOI] + k[745]*y[IDX_NI];
    IJth(jmatrix, 84, 16) = 0.0 + k[1387] + k[1388] + k[1389] + k[1390];
    IJth(jmatrix, 84, 28) = 0.0 - k[885]*y[IDX_NO2I];
    IJth(jmatrix, 84, 30) = 0.0 - k[909]*y[IDX_NO2I];
    IJth(jmatrix, 84, 35) = 0.0 - k[945]*y[IDX_NO2I];
    IJth(jmatrix, 84, 37) = 0.0 - k[952]*y[IDX_NO2I];
    IJth(jmatrix, 84, 41) = 0.0 - k[986]*y[IDX_NO2I];
    IJth(jmatrix, 84, 42) = 0.0 - k[504]*y[IDX_NO2I];
    IJth(jmatrix, 84, 52) = 0.0 - k[595]*y[IDX_NO2I];
    IJth(jmatrix, 84, 66) = 0.0 + k[1068]*y[IDX_OI];
    IJth(jmatrix, 84, 71) = 0.0 - k[1019]*y[IDX_NO2I] - k[1020]*y[IDX_NO2I]
        - k[1021]*y[IDX_NO2I];
    IJth(jmatrix, 84, 76) = 0.0 - k[1041]*y[IDX_NO2I];
    IJth(jmatrix, 84, 82) = 0.0 + k[1052]*y[IDX_O2I] + k[1099]*y[IDX_OHI];
    IJth(jmatrix, 84, 84) = 0.0 - k[276] - k[504]*y[IDX_HII] -
        k[595]*y[IDX_H3II] - k[818]*y[IDX_OII] - k[885]*y[IDX_CH2I] -
        k[909]*y[IDX_CH3I] - k[945]*y[IDX_CNI] - k[952]*y[IDX_COI] -
        k[986]*y[IDX_HI] - k[1019]*y[IDX_NI] - k[1020]*y[IDX_NI] -
        k[1021]*y[IDX_NI] - k[1041]*y[IDX_NHI] - k[1075]*y[IDX_OI] - k[1164] -
        k[1293];
    IJth(jmatrix, 84, 85) = 0.0 + k[1068]*y[IDX_HNOI] - k[1075]*y[IDX_NO2I];
    IJth(jmatrix, 84, 86) = 0.0 - k[818]*y[IDX_NO2I];
    IJth(jmatrix, 84, 87) = 0.0 + k[1052]*y[IDX_NOI] + k[1055]*y[IDX_OCNI];
    IJth(jmatrix, 84, 91) = 0.0 + k[1055]*y[IDX_O2I];
    IJth(jmatrix, 84, 92) = 0.0 + k[1099]*y[IDX_NOI];
    IJth(jmatrix, 85, 24) = 0.0 + k[391]*y[IDX_O2II] + k[393]*y[IDX_OHII] +
        k[871]*y[IDX_NOI] + k[873]*y[IDX_O2I] + k[876]*y[IDX_OHI] -
        k[1196]*y[IDX_OI];
    IJth(jmatrix, 85, 25) = 0.0 + k[376]*y[IDX_O2I] - k[1193]*y[IDX_OI];
    IJth(jmatrix, 85, 26) = 0.0 - k[0]*y[IDX_OI] + k[60]*y[IDX_OII] +
        k[473]*y[IDX_O2II] + k[475]*y[IDX_OHII] + k[930]*y[IDX_NOI] +
        k[934]*y[IDX_O2I] + k[936]*y[IDX_O2I] - k[939]*y[IDX_OI] -
        k[940]*y[IDX_OI];
    IJth(jmatrix, 85, 27) = 0.0 + k[412]*y[IDX_O2I] - k[414]*y[IDX_OI];
    IJth(jmatrix, 85, 28) = 0.0 + k[43]*y[IDX_OII] + k[435]*y[IDX_O2II] +
        k[437]*y[IDX_OHII] + k[892]*y[IDX_O2I] - k[894]*y[IDX_OI] -
        k[895]*y[IDX_OI] - k[896]*y[IDX_OI] - k[897]*y[IDX_OI] +
        k[900]*y[IDX_OHI];
    IJth(jmatrix, 85, 29) = 0.0 - k[421]*y[IDX_OI];
    IJth(jmatrix, 85, 30) = 0.0 - k[915]*y[IDX_OI] - k[916]*y[IDX_OI] +
        k[917]*y[IDX_OHI];
    IJth(jmatrix, 85, 31) = 0.0 + k[442]*y[IDX_O2I] - k[443]*y[IDX_OI] -
        k[444]*y[IDX_OI];
    IJth(jmatrix, 85, 33) = 0.0 + k[207]*y[IDX_OII] - k[1056]*y[IDX_OI];
    IJth(jmatrix, 85, 34) = 0.0 - k[822]*y[IDX_OI];
    IJth(jmatrix, 85, 35) = 0.0 + k[836]*y[IDX_OHII] + k[949]*y[IDX_O2I] -
        k[1057]*y[IDX_OI] - k[1058]*y[IDX_OI] + k[1090]*y[IDX_OHI];
    IJth(jmatrix, 85, 36) = 0.0 - k[216]*y[IDX_OI];
    IJth(jmatrix, 85, 37) = 0.0 + k[208]*y[IDX_OII] + k[253] +
        k[669]*y[IDX_HeII] + k[838]*y[IDX_OHII] + k[953]*y[IDX_O2I] + k[1133];
    IJth(jmatrix, 85, 38) = 0.0 - k[217]*y[IDX_OI] + k[304]*y[IDX_EM] +
        k[851]*y[IDX_OHI] + k[1131];
    IJth(jmatrix, 85, 39) = 0.0 + k[252] + k[496]*y[IDX_HII] +
        k[665]*y[IDX_HeII] + k[837]*y[IDX_OHII] - k[1059]*y[IDX_OI] + k[1132];
    IJth(jmatrix, 85, 40) = 0.0 + k[304]*y[IDX_COII] + k[306]*y[IDX_H2COII]
        + k[312]*y[IDX_H2OII] + k[313]*y[IDX_H2OII] + k[323]*y[IDX_H3OII] +
        k[332]*y[IDX_HCO2II] + k[345]*y[IDX_NOII] + k[346]*y[IDX_O2II] +
        k[346]*y[IDX_O2II] + k[348]*y[IDX_OHII] + k[362]*y[IDX_SiOII] +
        k[1221]*y[IDX_OII];
    IJth(jmatrix, 85, 41) = 0.0 + k[12]*y[IDX_O2I] + k[12]*y[IDX_O2I] +
        k[13]*y[IDX_OHI] + k[131]*y[IDX_OII] + k[978]*y[IDX_HCOI] +
        k[980]*y[IDX_HNOI] + k[987]*y[IDX_NOI] + k[989]*y[IDX_O2I] +
        k[990]*y[IDX_O2HI] + k[993]*y[IDX_OCNI] + k[996]*y[IDX_OHI] -
        k[1207]*y[IDX_OI];
    IJth(jmatrix, 85, 42) = 0.0 - k[89]*y[IDX_OI] + k[496]*y[IDX_CO2I];
    IJth(jmatrix, 85, 43) = 0.0 + k[6]*y[IDX_O2I] + k[6]*y[IDX_O2I] +
        k[7]*y[IDX_OHI] - k[965]*y[IDX_OI];
    IJth(jmatrix, 85, 44) = 0.0 - k[526]*y[IDX_OI];
    IJth(jmatrix, 85, 45) = 0.0 - k[1060]*y[IDX_OI];
    IJth(jmatrix, 85, 46) = 0.0 + k[209]*y[IDX_OII] + k[672]*y[IDX_HeII] +
        k[839]*y[IDX_OHII] - k[1061]*y[IDX_OI];
    IJth(jmatrix, 85, 47) = 0.0 + k[306]*y[IDX_EM];
    IJth(jmatrix, 85, 49) = 0.0 + k[210]*y[IDX_OII] + k[756]*y[IDX_NHII] +
        k[840]*y[IDX_OHII] - k[1062]*y[IDX_OI];
    IJth(jmatrix, 85, 50) = 0.0 + k[312]*y[IDX_EM] + k[313]*y[IDX_EM] -
        k[823]*y[IDX_OI] + k[852]*y[IDX_OHI];
    IJth(jmatrix, 85, 52) = 0.0 - k[598]*y[IDX_OI] - k[599]*y[IDX_OI];
    IJth(jmatrix, 85, 54) = 0.0 + k[323]*y[IDX_EM];
    IJth(jmatrix, 85, 55) = 0.0 + k[841]*y[IDX_OHII] - k[1063]*y[IDX_OI] -
        k[1064]*y[IDX_OI] - k[1065]*y[IDX_OI];
    IJth(jmatrix, 85, 58) = 0.0 + k[211]*y[IDX_OII] + k[682]*y[IDX_HeII] +
        k[843]*y[IDX_OHII] + k[978]*y[IDX_HI] + k[1015]*y[IDX_NI] -
        k[1066]*y[IDX_OI] - k[1067]*y[IDX_OI];
    IJth(jmatrix, 85, 60) = 0.0 + k[332]*y[IDX_EM] - k[824]*y[IDX_OI];
    IJth(jmatrix, 85, 62) = 0.0 + k[665]*y[IDX_CO2I] + k[669]*y[IDX_COI] +
        k[672]*y[IDX_H2COI] + k[682]*y[IDX_HCOI] + k[695]*y[IDX_NOI] +
        k[696]*y[IDX_O2I] + k[697]*y[IDX_OCNI] + k[710]*y[IDX_SiOI];
    IJth(jmatrix, 85, 64) = 0.0 + k[844]*y[IDX_OHII];
    IJth(jmatrix, 85, 66) = 0.0 + k[980]*y[IDX_HI] - k[1068]*y[IDX_OI] -
        k[1069]*y[IDX_OI] - k[1070]*y[IDX_OI];
    IJth(jmatrix, 85, 71) = 0.0 + k[742]*y[IDX_O2II] + k[1015]*y[IDX_HCOI] +
        k[1019]*y[IDX_NO2I] + k[1019]*y[IDX_NO2I] + k[1022]*y[IDX_NOI] +
        k[1023]*y[IDX_O2I] + k[1026]*y[IDX_OHI];
    IJth(jmatrix, 85, 72) = 0.0 + k[727]*y[IDX_NOI] + k[728]*y[IDX_O2I];
    IJth(jmatrix, 85, 73) = 0.0 + k[845]*y[IDX_OHII] - k[1071]*y[IDX_OI];
    IJth(jmatrix, 85, 74) = 0.0 - k[218]*y[IDX_OI] - k[825]*y[IDX_OI];
    IJth(jmatrix, 85, 75) = 0.0 - k[826]*y[IDX_OI];
    IJth(jmatrix, 85, 76) = 0.0 + k[202]*y[IDX_OII] + k[804]*y[IDX_O2II] +
        k[806]*y[IDX_OHII] + k[1042]*y[IDX_NOI] + k[1044]*y[IDX_O2I] -
        k[1046]*y[IDX_OI] - k[1047]*y[IDX_OI] + k[1050]*y[IDX_OHI];
    IJth(jmatrix, 85, 77) = 0.0 + k[756]*y[IDX_H2OI] + k[764]*y[IDX_NOI] -
        k[767]*y[IDX_OI];
    IJth(jmatrix, 85, 78) = 0.0 + k[212]*y[IDX_OII] + k[791]*y[IDX_OHII] +
        k[1032]*y[IDX_OHI] - k[1072]*y[IDX_OI] - k[1073]*y[IDX_OI];
    IJth(jmatrix, 85, 79) = 0.0 + k[777]*y[IDX_O2I] - k[827]*y[IDX_OI];
    IJth(jmatrix, 85, 80) = 0.0 + k[213]*y[IDX_OII] - k[1074]*y[IDX_OI];
    IJth(jmatrix, 85, 81) = 0.0 - k[828]*y[IDX_OI];
    IJth(jmatrix, 85, 82) = 0.0 + k[278] + k[695]*y[IDX_HeII] +
        k[727]*y[IDX_NII] + k[764]*y[IDX_NHII] + k[846]*y[IDX_OHII] +
        k[871]*y[IDX_CI] + k[930]*y[IDX_CHI] + k[987]*y[IDX_HI] +
        k[1022]*y[IDX_NI] + k[1042]*y[IDX_NHI] + k[1052]*y[IDX_O2I] -
        k[1076]*y[IDX_OI] + k[1166];
    IJth(jmatrix, 85, 83) = 0.0 + k[345]*y[IDX_EM];
    IJth(jmatrix, 85, 84) = 0.0 + k[276] + k[1019]*y[IDX_NI] +
        k[1019]*y[IDX_NI] - k[1075]*y[IDX_OI] + k[1164];
    IJth(jmatrix, 85, 85) = 0.0 - k[0]*y[IDX_CHI] - k[89]*y[IDX_HII] -
        k[216]*y[IDX_CNII] - k[217]*y[IDX_COII] - k[218]*y[IDX_N2II] - k[239] -
        k[282] - k[414]*y[IDX_CHII] - k[421]*y[IDX_CH2II] - k[443]*y[IDX_CH3II]
        - k[444]*y[IDX_CH3II] - k[526]*y[IDX_H2II] - k[598]*y[IDX_H3II] -
        k[599]*y[IDX_H3II] - k[767]*y[IDX_NHII] - k[822]*y[IDX_CH4II] -
        k[823]*y[IDX_H2OII] - k[824]*y[IDX_HCO2II] - k[825]*y[IDX_N2II] -
        k[826]*y[IDX_N2HII] - k[827]*y[IDX_NH2II] - k[828]*y[IDX_NH3II] -
        k[829]*y[IDX_O2HII] - k[830]*y[IDX_OHII] - k[831]*y[IDX_SiCII] -
        k[832]*y[IDX_SiHII] - k[833]*y[IDX_SiH2II] - k[834]*y[IDX_SiH3II] -
        k[835]*y[IDX_SiOII] - k[894]*y[IDX_CH2I] - k[895]*y[IDX_CH2I] -
        k[896]*y[IDX_CH2I] - k[897]*y[IDX_CH2I] - k[915]*y[IDX_CH3I] -
        k[916]*y[IDX_CH3I] - k[939]*y[IDX_CHI] - k[940]*y[IDX_CHI] -
        k[965]*y[IDX_H2I] - k[1046]*y[IDX_NHI] - k[1047]*y[IDX_NHI] -
        k[1056]*y[IDX_CH4I] - k[1057]*y[IDX_CNI] - k[1058]*y[IDX_CNI] -
        k[1059]*y[IDX_CO2I] - k[1060]*y[IDX_H2CNI] - k[1061]*y[IDX_H2COI] -
        k[1062]*y[IDX_H2OI] - k[1063]*y[IDX_HCNI] - k[1064]*y[IDX_HCNI] -
        k[1065]*y[IDX_HCNI] - k[1066]*y[IDX_HCOI] - k[1067]*y[IDX_HCOI] -
        k[1068]*y[IDX_HNOI] - k[1069]*y[IDX_HNOI] - k[1070]*y[IDX_HNOI] -
        k[1071]*y[IDX_N2I] - k[1072]*y[IDX_NH2I] - k[1073]*y[IDX_NH2I] -
        k[1074]*y[IDX_NH3I] - k[1075]*y[IDX_NO2I] - k[1076]*y[IDX_NOI] -
        k[1077]*y[IDX_O2HI] - k[1078]*y[IDX_OCNI] - k[1079]*y[IDX_OCNI] -
        k[1080]*y[IDX_OHI] - k[1081]*y[IDX_SiC2I] - k[1082]*y[IDX_SiC3I] -
        k[1083]*y[IDX_SiCI] - k[1084]*y[IDX_SiCI] - k[1085]*y[IDX_SiH2I] -
        k[1086]*y[IDX_SiH2I] - k[1087]*y[IDX_SiH3I] - k[1088]*y[IDX_SiH4I] -
        k[1089]*y[IDX_SiHI] - k[1193]*y[IDX_CII] - k[1196]*y[IDX_CI] -
        k[1207]*y[IDX_HI] - k[1211]*y[IDX_OI] - k[1211]*y[IDX_OI] -
        k[1211]*y[IDX_OI] - k[1211]*y[IDX_OI] - k[1212]*y[IDX_SiII] -
        k[1213]*y[IDX_SiI] - k[1291];
    IJth(jmatrix, 85, 86) = 0.0 + k[43]*y[IDX_CH2I] + k[60]*y[IDX_CHI] +
        k[131]*y[IDX_HI] + k[202]*y[IDX_NHI] + k[207]*y[IDX_CH4I] +
        k[208]*y[IDX_COI] + k[209]*y[IDX_H2COI] + k[210]*y[IDX_H2OI] +
        k[211]*y[IDX_HCOI] + k[212]*y[IDX_NH2I] + k[213]*y[IDX_NH3I] +
        k[214]*y[IDX_O2I] + k[215]*y[IDX_OHI] + k[1221]*y[IDX_EM];
    IJth(jmatrix, 85, 87) = 0.0 + k[6]*y[IDX_H2I] + k[6]*y[IDX_H2I] +
        k[12]*y[IDX_HI] + k[12]*y[IDX_HI] + k[214]*y[IDX_OII] + k[280] + k[280]
        + k[376]*y[IDX_CII] + k[412]*y[IDX_CHII] + k[442]*y[IDX_CH3II] +
        k[696]*y[IDX_HeII] + k[728]*y[IDX_NII] + k[777]*y[IDX_NH2II] +
        k[873]*y[IDX_CI] + k[892]*y[IDX_CH2I] + k[934]*y[IDX_CHI] +
        k[936]*y[IDX_CHI] + k[949]*y[IDX_CNI] + k[953]*y[IDX_COI] +
        k[989]*y[IDX_HI] + k[1023]*y[IDX_NI] + k[1044]*y[IDX_NHI] +
        k[1052]*y[IDX_NOI] + k[1106]*y[IDX_SiI] + k[1169] + k[1169];
    IJth(jmatrix, 85, 88) = 0.0 + k[346]*y[IDX_EM] + k[346]*y[IDX_EM] +
        k[391]*y[IDX_CI] + k[435]*y[IDX_CH2I] + k[473]*y[IDX_CHI] +
        k[742]*y[IDX_NI] + k[804]*y[IDX_NHI] + k[1167];
    IJth(jmatrix, 85, 89) = 0.0 + k[990]*y[IDX_HI] - k[1077]*y[IDX_OI] +
        k[1171];
    IJth(jmatrix, 85, 90) = 0.0 - k[829]*y[IDX_OI];
    IJth(jmatrix, 85, 91) = 0.0 + k[283] + k[697]*y[IDX_HeII] +
        k[993]*y[IDX_HI] - k[1078]*y[IDX_OI] - k[1079]*y[IDX_OI] + k[1172];
    IJth(jmatrix, 85, 92) = 0.0 + k[7]*y[IDX_H2I] + k[13]*y[IDX_HI] +
        k[215]*y[IDX_OII] + k[284] + k[847]*y[IDX_OHII] + k[851]*y[IDX_COII] +
        k[852]*y[IDX_H2OII] + k[876]*y[IDX_CI] + k[900]*y[IDX_CH2I] +
        k[917]*y[IDX_CH3I] + k[996]*y[IDX_HI] + k[1026]*y[IDX_NI] +
        k[1032]*y[IDX_NH2I] + k[1050]*y[IDX_NHI] - k[1080]*y[IDX_OI] +
        k[1090]*y[IDX_CNI] + k[1101]*y[IDX_OHI] + k[1101]*y[IDX_OHI] + k[1174];
    IJth(jmatrix, 85, 93) = 0.0 + k[348]*y[IDX_EM] + k[393]*y[IDX_CI] +
        k[437]*y[IDX_CH2I] + k[475]*y[IDX_CHI] + k[791]*y[IDX_NH2I] +
        k[806]*y[IDX_NHI] - k[830]*y[IDX_OI] + k[836]*y[IDX_CNI] +
        k[837]*y[IDX_CO2I] + k[838]*y[IDX_COI] + k[839]*y[IDX_H2COI] +
        k[840]*y[IDX_H2OI] + k[841]*y[IDX_HCNI] + k[843]*y[IDX_HCOI] +
        k[844]*y[IDX_HNCI] + k[845]*y[IDX_N2I] + k[846]*y[IDX_NOI] +
        k[847]*y[IDX_OHI] + k[848]*y[IDX_SiI] + k[849]*y[IDX_SiHI] +
        k[850]*y[IDX_SiOI];
    IJth(jmatrix, 85, 94) = 0.0 + k[848]*y[IDX_OHII] + k[1106]*y[IDX_O2I] -
        k[1213]*y[IDX_OI];
    IJth(jmatrix, 85, 95) = 0.0 - k[1212]*y[IDX_OI];
    IJth(jmatrix, 85, 96) = 0.0 - k[1083]*y[IDX_OI] - k[1084]*y[IDX_OI];
    IJth(jmatrix, 85, 97) = 0.0 - k[831]*y[IDX_OI];
    IJth(jmatrix, 85, 98) = 0.0 - k[1081]*y[IDX_OI];
    IJth(jmatrix, 85, 100) = 0.0 - k[1082]*y[IDX_OI];
    IJth(jmatrix, 85, 102) = 0.0 + k[849]*y[IDX_OHII] - k[1089]*y[IDX_OI];
    IJth(jmatrix, 85, 103) = 0.0 - k[832]*y[IDX_OI];
    IJth(jmatrix, 85, 104) = 0.0 - k[1085]*y[IDX_OI] - k[1086]*y[IDX_OI];
    IJth(jmatrix, 85, 105) = 0.0 - k[833]*y[IDX_OI];
    IJth(jmatrix, 85, 106) = 0.0 - k[1087]*y[IDX_OI];
    IJth(jmatrix, 85, 107) = 0.0 - k[834]*y[IDX_OI];
    IJth(jmatrix, 85, 108) = 0.0 - k[1088]*y[IDX_OI];
    IJth(jmatrix, 85, 111) = 0.0 + k[293] + k[710]*y[IDX_HeII] +
        k[850]*y[IDX_OHII] + k[1190];
    IJth(jmatrix, 85, 112) = 0.0 + k[362]*y[IDX_EM] - k[835]*y[IDX_OI] +
        k[1189];
    IJth(jmatrix, 86, 24) = 0.0 - k[1195]*y[IDX_OII];
    IJth(jmatrix, 86, 25) = 0.0 + k[377]*y[IDX_O2I];
    IJth(jmatrix, 86, 26) = 0.0 - k[60]*y[IDX_OII] - k[472]*y[IDX_OII];
    IJth(jmatrix, 86, 27) = 0.0 + k[413]*y[IDX_O2I];
    IJth(jmatrix, 86, 28) = 0.0 - k[43]*y[IDX_OII];
    IJth(jmatrix, 86, 32) = 0.0 - k[808]*y[IDX_OII] - k[809]*y[IDX_OII];
    IJth(jmatrix, 86, 33) = 0.0 - k[207]*y[IDX_OII] - k[810]*y[IDX_OII];
    IJth(jmatrix, 86, 35) = 0.0 - k[811]*y[IDX_OII];
    IJth(jmatrix, 86, 36) = 0.0 + k[216]*y[IDX_OI];
    IJth(jmatrix, 86, 37) = 0.0 - k[208]*y[IDX_OII];
    IJth(jmatrix, 86, 38) = 0.0 + k[217]*y[IDX_OI];
    IJth(jmatrix, 86, 39) = 0.0 + k[666]*y[IDX_HeII] - k[812]*y[IDX_OII];
    IJth(jmatrix, 86, 40) = 0.0 - k[1221]*y[IDX_OII];
    IJth(jmatrix, 86, 41) = 0.0 - k[131]*y[IDX_OII];
    IJth(jmatrix, 86, 42) = 0.0 + k[89]*y[IDX_OI];
    IJth(jmatrix, 86, 43) = 0.0 - k[543]*y[IDX_OII];
    IJth(jmatrix, 86, 46) = 0.0 - k[209]*y[IDX_OII] - k[813]*y[IDX_OII];
    IJth(jmatrix, 86, 49) = 0.0 - k[210]*y[IDX_OII];
    IJth(jmatrix, 86, 55) = 0.0 - k[814]*y[IDX_OII] - k[815]*y[IDX_OII];
    IJth(jmatrix, 86, 58) = 0.0 - k[211]*y[IDX_OII] - k[816]*y[IDX_OII];
    IJth(jmatrix, 86, 62) = 0.0 + k[666]*y[IDX_CO2I] + k[694]*y[IDX_NOI] +
        k[696]*y[IDX_O2I] + k[698]*y[IDX_OCNI] + k[699]*y[IDX_OHI] +
        k[711]*y[IDX_SiOI];
    IJth(jmatrix, 86, 72) = 0.0 + k[729]*y[IDX_O2I];
    IJth(jmatrix, 86, 73) = 0.0 - k[817]*y[IDX_OII];
    IJth(jmatrix, 86, 74) = 0.0 + k[218]*y[IDX_OI];
    IJth(jmatrix, 86, 76) = 0.0 - k[202]*y[IDX_OII] - k[803]*y[IDX_OII];
    IJth(jmatrix, 86, 78) = 0.0 - k[212]*y[IDX_OII];
    IJth(jmatrix, 86, 80) = 0.0 - k[213]*y[IDX_OII];
    IJth(jmatrix, 86, 82) = 0.0 + k[694]*y[IDX_HeII];
    IJth(jmatrix, 86, 84) = 0.0 - k[818]*y[IDX_OII];
    IJth(jmatrix, 86, 85) = 0.0 + k[89]*y[IDX_HII] + k[216]*y[IDX_CNII] +
        k[217]*y[IDX_COII] + k[218]*y[IDX_N2II] + k[239] + k[282];
    IJth(jmatrix, 86, 86) = 0.0 - k[43]*y[IDX_CH2I] - k[60]*y[IDX_CHI] -
        k[131]*y[IDX_HI] - k[202]*y[IDX_NHI] - k[207]*y[IDX_CH4I] -
        k[208]*y[IDX_COI] - k[209]*y[IDX_H2COI] - k[210]*y[IDX_H2OI] -
        k[211]*y[IDX_HCOI] - k[212]*y[IDX_NH2I] - k[213]*y[IDX_NH3I] -
        k[214]*y[IDX_O2I] - k[215]*y[IDX_OHI] - k[472]*y[IDX_CHI] -
        k[543]*y[IDX_H2I] - k[803]*y[IDX_NHI] - k[808]*y[IDX_CH3OHI] -
        k[809]*y[IDX_CH3OHI] - k[810]*y[IDX_CH4I] - k[811]*y[IDX_CNI] -
        k[812]*y[IDX_CO2I] - k[813]*y[IDX_H2COI] - k[814]*y[IDX_HCNI] -
        k[815]*y[IDX_HCNI] - k[816]*y[IDX_HCOI] - k[817]*y[IDX_N2I] -
        k[818]*y[IDX_NO2I] - k[819]*y[IDX_OHI] - k[1195]*y[IDX_CI] -
        k[1221]*y[IDX_EM] - k[1269];
    IJth(jmatrix, 86, 87) = 0.0 - k[214]*y[IDX_OII] + k[377]*y[IDX_CII] +
        k[413]*y[IDX_CHII] + k[696]*y[IDX_HeII] + k[729]*y[IDX_NII];
    IJth(jmatrix, 86, 88) = 0.0 + k[1167];
    IJth(jmatrix, 86, 91) = 0.0 + k[698]*y[IDX_HeII];
    IJth(jmatrix, 86, 92) = 0.0 - k[215]*y[IDX_OII] + k[699]*y[IDX_HeII] -
        k[819]*y[IDX_OII];
    IJth(jmatrix, 86, 93) = 0.0 + k[1173];
    IJth(jmatrix, 86, 111) = 0.0 + k[711]*y[IDX_HeII];
    IJth(jmatrix, 87, 17) = 0.0 + k[1355] + k[1356] + k[1357] + k[1358];
    IJth(jmatrix, 87, 24) = 0.0 + k[30]*y[IDX_O2II] + k[392]*y[IDX_O2HII] -
        k[873]*y[IDX_O2I];
    IJth(jmatrix, 87, 25) = 0.0 - k[376]*y[IDX_O2I] - k[377]*y[IDX_O2I];
    IJth(jmatrix, 87, 26) = 0.0 + k[61]*y[IDX_O2II] + k[474]*y[IDX_O2HII] -
        k[933]*y[IDX_O2I] - k[934]*y[IDX_O2I] - k[935]*y[IDX_O2I] -
        k[936]*y[IDX_O2I] + k[938]*y[IDX_O2HI];
    IJth(jmatrix, 87, 27) = 0.0 - k[411]*y[IDX_O2I] - k[412]*y[IDX_O2I] -
        k[413]*y[IDX_O2I];
    IJth(jmatrix, 87, 28) = 0.0 + k[44]*y[IDX_O2II] + k[436]*y[IDX_O2HII] -
        k[889]*y[IDX_O2I] - k[890]*y[IDX_O2I] - k[891]*y[IDX_O2I] -
        k[892]*y[IDX_O2I] - k[893]*y[IDX_O2I];
    IJth(jmatrix, 87, 29) = 0.0 - k[420]*y[IDX_O2I];
    IJth(jmatrix, 87, 30) = 0.0 - k[911]*y[IDX_O2I] - k[912]*y[IDX_O2I] -
        k[913]*y[IDX_O2I] + k[914]*y[IDX_O2HI];
    IJth(jmatrix, 87, 31) = 0.0 - k[442]*y[IDX_O2I];
    IJth(jmatrix, 87, 32) = 0.0 + k[820]*y[IDX_O2II];
    IJth(jmatrix, 87, 33) = 0.0 - k[921]*y[IDX_O2I];
    IJth(jmatrix, 87, 34) = 0.0 - k[51]*y[IDX_O2I];
    IJth(jmatrix, 87, 35) = 0.0 + k[483]*y[IDX_O2HII] - k[948]*y[IDX_O2I] -
        k[949]*y[IDX_O2I];
    IJth(jmatrix, 87, 36) = 0.0 - k[68]*y[IDX_O2I] - k[481]*y[IDX_O2I];
    IJth(jmatrix, 87, 37) = 0.0 + k[488]*y[IDX_O2HII] - k[953]*y[IDX_O2I];
    IJth(jmatrix, 87, 38) = 0.0 - k[73]*y[IDX_O2I];
    IJth(jmatrix, 87, 39) = 0.0 + k[668]*y[IDX_HeII] + k[821]*y[IDX_O2HII] +
        k[1059]*y[IDX_OI];
    IJth(jmatrix, 87, 40) = 0.0 + k[347]*y[IDX_O2HII];
    IJth(jmatrix, 87, 41) = 0.0 - k[12]*y[IDX_O2I] - k[989]*y[IDX_O2I] +
        k[991]*y[IDX_O2HI];
    IJth(jmatrix, 87, 42) = 0.0 - k[88]*y[IDX_O2I];
    IJth(jmatrix, 87, 43) = 0.0 - k[6]*y[IDX_O2I] + k[544]*y[IDX_O2HII] -
        k[963]*y[IDX_O2I] - k[964]*y[IDX_O2I];
    IJth(jmatrix, 87, 44) = 0.0 - k[113]*y[IDX_O2I] - k[525]*y[IDX_O2I];
    IJth(jmatrix, 87, 46) = 0.0 + k[116]*y[IDX_O2II] + k[551]*y[IDX_O2II] +
        k[552]*y[IDX_O2HII];
    IJth(jmatrix, 87, 47) = 0.0 - k[549]*y[IDX_O2I];
    IJth(jmatrix, 87, 49) = 0.0 + k[571]*y[IDX_O2HII];
    IJth(jmatrix, 87, 50) = 0.0 - k[121]*y[IDX_O2I];
    IJth(jmatrix, 87, 52) = 0.0 - k[597]*y[IDX_O2I];
    IJth(jmatrix, 87, 55) = 0.0 + k[632]*y[IDX_O2HII];
    IJth(jmatrix, 87, 56) = 0.0 - k[133]*y[IDX_O2I];
    IJth(jmatrix, 87, 58) = 0.0 + k[137]*y[IDX_O2II] + k[645]*y[IDX_O2HII] -
        k[1001]*y[IDX_O2I] - k[1002]*y[IDX_O2I] + k[1003]*y[IDX_O2HI];
    IJth(jmatrix, 87, 60) = 0.0 + k[824]*y[IDX_OI];
    IJth(jmatrix, 87, 62) = 0.0 - k[146]*y[IDX_O2I] + k[668]*y[IDX_CO2I] -
        k[696]*y[IDX_O2I];
    IJth(jmatrix, 87, 64) = 0.0 + k[651]*y[IDX_O2HII];
    IJth(jmatrix, 87, 66) = 0.0 + k[1070]*y[IDX_OI];
    IJth(jmatrix, 87, 69) = 0.0 + k[152]*y[IDX_O2II];
    IJth(jmatrix, 87, 71) = 0.0 + k[1021]*y[IDX_NO2I] - k[1023]*y[IDX_O2I] +
        k[1024]*y[IDX_O2HI];
    IJth(jmatrix, 87, 72) = 0.0 - k[168]*y[IDX_O2I] - k[728]*y[IDX_O2I] -
        k[729]*y[IDX_O2I];
    IJth(jmatrix, 87, 73) = 0.0 + k[733]*y[IDX_O2HII];
    IJth(jmatrix, 87, 74) = 0.0 - k[173]*y[IDX_O2I];
    IJth(jmatrix, 87, 76) = 0.0 + k[805]*y[IDX_O2HII] - k[1044]*y[IDX_O2I] -
        k[1045]*y[IDX_O2I];
    IJth(jmatrix, 87, 77) = 0.0 - k[179]*y[IDX_O2I] - k[765]*y[IDX_O2I] -
        k[766]*y[IDX_O2I];
    IJth(jmatrix, 87, 78) = 0.0 + k[187]*y[IDX_O2II] + k[790]*y[IDX_O2HII];
    IJth(jmatrix, 87, 79) = 0.0 - k[777]*y[IDX_O2I] - k[778]*y[IDX_O2I];
    IJth(jmatrix, 87, 80) = 0.0 + k[198]*y[IDX_O2II];
    IJth(jmatrix, 87, 82) = 0.0 + k[205]*y[IDX_O2II] + k[807]*y[IDX_O2HII] +
        k[1051]*y[IDX_NOI] + k[1051]*y[IDX_NOI] - k[1052]*y[IDX_O2I] +
        k[1076]*y[IDX_OI];
    IJth(jmatrix, 87, 84) = 0.0 + k[818]*y[IDX_OII] + k[1021]*y[IDX_NI] +
        k[1075]*y[IDX_OI];
    IJth(jmatrix, 87, 85) = 0.0 + k[824]*y[IDX_HCO2II] + k[829]*y[IDX_O2HII]
        + k[835]*y[IDX_SiOII] + k[1059]*y[IDX_CO2I] + k[1070]*y[IDX_HNOI] +
        k[1075]*y[IDX_NO2I] + k[1076]*y[IDX_NOI] + k[1077]*y[IDX_O2HI] +
        k[1079]*y[IDX_OCNI] + k[1080]*y[IDX_OHI] + k[1211]*y[IDX_OI] +
        k[1211]*y[IDX_OI];
    IJth(jmatrix, 87, 86) = 0.0 - k[214]*y[IDX_O2I] + k[818]*y[IDX_NO2I];
    IJth(jmatrix, 87, 87) = 0.0 - k[6]*y[IDX_H2I] - k[12]*y[IDX_HI] -
        k[51]*y[IDX_CH4II] - k[68]*y[IDX_CNII] - k[73]*y[IDX_COII] -
        k[88]*y[IDX_HII] - k[113]*y[IDX_H2II] - k[121]*y[IDX_H2OII] -
        k[133]*y[IDX_HCNII] - k[146]*y[IDX_HeII] - k[168]*y[IDX_NII] -
        k[173]*y[IDX_N2II] - k[179]*y[IDX_NHII] - k[214]*y[IDX_OII] -
        k[224]*y[IDX_OHII] - k[279] - k[280] - k[376]*y[IDX_CII] -
        k[377]*y[IDX_CII] - k[411]*y[IDX_CHII] - k[412]*y[IDX_CHII] -
        k[413]*y[IDX_CHII] - k[420]*y[IDX_CH2II] - k[442]*y[IDX_CH3II] -
        k[481]*y[IDX_CNII] - k[525]*y[IDX_H2II] - k[549]*y[IDX_H2COII] -
        k[597]*y[IDX_H3II] - k[696]*y[IDX_HeII] - k[728]*y[IDX_NII] -
        k[729]*y[IDX_NII] - k[765]*y[IDX_NHII] - k[766]*y[IDX_NHII] -
        k[777]*y[IDX_NH2II] - k[778]*y[IDX_NH2II] - k[862]*y[IDX_SiH2II] -
        k[873]*y[IDX_CI] - k[889]*y[IDX_CH2I] - k[890]*y[IDX_CH2I] -
        k[891]*y[IDX_CH2I] - k[892]*y[IDX_CH2I] - k[893]*y[IDX_CH2I] -
        k[911]*y[IDX_CH3I] - k[912]*y[IDX_CH3I] - k[913]*y[IDX_CH3I] -
        k[921]*y[IDX_CH4I] - k[933]*y[IDX_CHI] - k[934]*y[IDX_CHI] -
        k[935]*y[IDX_CHI] - k[936]*y[IDX_CHI] - k[948]*y[IDX_CNI] -
        k[949]*y[IDX_CNI] - k[953]*y[IDX_COI] - k[963]*y[IDX_H2I] -
        k[964]*y[IDX_H2I] - k[989]*y[IDX_HI] - k[1001]*y[IDX_HCOI] -
        k[1002]*y[IDX_HCOI] - k[1023]*y[IDX_NI] - k[1044]*y[IDX_NHI] -
        k[1045]*y[IDX_NHI] - k[1052]*y[IDX_NOI] - k[1054]*y[IDX_OCNI] -
        k[1055]*y[IDX_OCNI] - k[1106]*y[IDX_SiI] - k[1168] - k[1169] - k[1292];
    IJth(jmatrix, 87, 88) = 0.0 + k[30]*y[IDX_CI] + k[44]*y[IDX_CH2I] +
        k[61]*y[IDX_CHI] + k[116]*y[IDX_H2COI] + k[137]*y[IDX_HCOI] +
        k[152]*y[IDX_MgI] + k[187]*y[IDX_NH2I] + k[198]*y[IDX_NH3I] +
        k[205]*y[IDX_NOI] + k[230]*y[IDX_SiI] + k[551]*y[IDX_H2COI] +
        k[820]*y[IDX_CH3OHI];
    IJth(jmatrix, 87, 89) = 0.0 + k[281] + k[914]*y[IDX_CH3I] +
        k[938]*y[IDX_CHI] + k[991]*y[IDX_HI] + k[1003]*y[IDX_HCOI] +
        k[1024]*y[IDX_NI] + k[1077]*y[IDX_OI] + k[1100]*y[IDX_OHI] + k[1170];
    IJth(jmatrix, 87, 90) = 0.0 + k[347]*y[IDX_EM] + k[392]*y[IDX_CI] +
        k[436]*y[IDX_CH2I] + k[474]*y[IDX_CHI] + k[483]*y[IDX_CNI] +
        k[488]*y[IDX_COI] + k[544]*y[IDX_H2I] + k[552]*y[IDX_H2COI] +
        k[571]*y[IDX_H2OI] + k[632]*y[IDX_HCNI] + k[645]*y[IDX_HCOI] +
        k[651]*y[IDX_HNCI] + k[733]*y[IDX_N2I] + k[790]*y[IDX_NH2I] +
        k[805]*y[IDX_NHI] + k[807]*y[IDX_NOI] + k[821]*y[IDX_CO2I] +
        k[829]*y[IDX_OI] + k[858]*y[IDX_OHI];
    IJth(jmatrix, 87, 91) = 0.0 - k[1054]*y[IDX_O2I] - k[1055]*y[IDX_O2I] +
        k[1079]*y[IDX_OI];
    IJth(jmatrix, 87, 92) = 0.0 + k[858]*y[IDX_O2HII] + k[1080]*y[IDX_OI] +
        k[1100]*y[IDX_O2HI];
    IJth(jmatrix, 87, 93) = 0.0 - k[224]*y[IDX_O2I];
    IJth(jmatrix, 87, 94) = 0.0 + k[230]*y[IDX_O2II] - k[1106]*y[IDX_O2I];
    IJth(jmatrix, 87, 105) = 0.0 - k[862]*y[IDX_O2I];
    IJth(jmatrix, 87, 112) = 0.0 + k[835]*y[IDX_OI];
    IJth(jmatrix, 88, 24) = 0.0 - k[30]*y[IDX_O2II] - k[391]*y[IDX_O2II];
    IJth(jmatrix, 88, 26) = 0.0 - k[61]*y[IDX_O2II] - k[473]*y[IDX_O2II];
    IJth(jmatrix, 88, 28) = 0.0 - k[44]*y[IDX_O2II] - k[435]*y[IDX_O2II];
    IJth(jmatrix, 88, 32) = 0.0 - k[820]*y[IDX_O2II];
    IJth(jmatrix, 88, 34) = 0.0 + k[51]*y[IDX_O2I];
    IJth(jmatrix, 88, 36) = 0.0 + k[68]*y[IDX_O2I];
    IJth(jmatrix, 88, 38) = 0.0 + k[73]*y[IDX_O2I];
    IJth(jmatrix, 88, 39) = 0.0 + k[667]*y[IDX_HeII] + k[812]*y[IDX_OII];
    IJth(jmatrix, 88, 40) = 0.0 - k[346]*y[IDX_O2II];
    IJth(jmatrix, 88, 42) = 0.0 + k[88]*y[IDX_O2I];
    IJth(jmatrix, 88, 44) = 0.0 + k[113]*y[IDX_O2I];
    IJth(jmatrix, 88, 46) = 0.0 - k[116]*y[IDX_O2II] - k[551]*y[IDX_O2II];
    IJth(jmatrix, 88, 50) = 0.0 + k[121]*y[IDX_O2I] + k[823]*y[IDX_OI];
    IJth(jmatrix, 88, 56) = 0.0 + k[133]*y[IDX_O2I];
    IJth(jmatrix, 88, 58) = 0.0 - k[137]*y[IDX_O2II] - k[644]*y[IDX_O2II];
    IJth(jmatrix, 88, 62) = 0.0 + k[146]*y[IDX_O2I] + k[667]*y[IDX_CO2I];
    IJth(jmatrix, 88, 69) = 0.0 - k[152]*y[IDX_O2II];
    IJth(jmatrix, 88, 71) = 0.0 - k[742]*y[IDX_O2II];
    IJth(jmatrix, 88, 72) = 0.0 + k[168]*y[IDX_O2I];
    IJth(jmatrix, 88, 74) = 0.0 + k[173]*y[IDX_O2I];
    IJth(jmatrix, 88, 76) = 0.0 - k[804]*y[IDX_O2II];
    IJth(jmatrix, 88, 77) = 0.0 + k[179]*y[IDX_O2I];
    IJth(jmatrix, 88, 78) = 0.0 - k[187]*y[IDX_O2II];
    IJth(jmatrix, 88, 80) = 0.0 - k[198]*y[IDX_O2II];
    IJth(jmatrix, 88, 82) = 0.0 - k[205]*y[IDX_O2II];
    IJth(jmatrix, 88, 85) = 0.0 + k[823]*y[IDX_H2OII] + k[830]*y[IDX_OHII];
    IJth(jmatrix, 88, 86) = 0.0 + k[214]*y[IDX_O2I] + k[812]*y[IDX_CO2I] +
        k[819]*y[IDX_OHI];
    IJth(jmatrix, 88, 87) = 0.0 + k[51]*y[IDX_CH4II] + k[68]*y[IDX_CNII] +
        k[73]*y[IDX_COII] + k[88]*y[IDX_HII] + k[113]*y[IDX_H2II] +
        k[121]*y[IDX_H2OII] + k[133]*y[IDX_HCNII] + k[146]*y[IDX_HeII] +
        k[168]*y[IDX_NII] + k[173]*y[IDX_N2II] + k[179]*y[IDX_NHII] +
        k[214]*y[IDX_OII] + k[224]*y[IDX_OHII] + k[279] + k[1168];
    IJth(jmatrix, 88, 88) = 0.0 - k[30]*y[IDX_CI] - k[44]*y[IDX_CH2I] -
        k[61]*y[IDX_CHI] - k[116]*y[IDX_H2COI] - k[137]*y[IDX_HCOI] -
        k[152]*y[IDX_MgI] - k[187]*y[IDX_NH2I] - k[198]*y[IDX_NH3I] -
        k[205]*y[IDX_NOI] - k[230]*y[IDX_SiI] - k[346]*y[IDX_EM] -
        k[391]*y[IDX_CI] - k[435]*y[IDX_CH2I] - k[473]*y[IDX_CHI] -
        k[551]*y[IDX_H2COI] - k[644]*y[IDX_HCOI] - k[742]*y[IDX_NI] -
        k[804]*y[IDX_NHI] - k[820]*y[IDX_CH3OHI] - k[1167] - k[1270];
    IJth(jmatrix, 88, 92) = 0.0 + k[819]*y[IDX_OII];
    IJth(jmatrix, 88, 93) = 0.0 + k[224]*y[IDX_O2I] + k[830]*y[IDX_OI];
    IJth(jmatrix, 88, 94) = 0.0 - k[230]*y[IDX_O2II];
    IJth(jmatrix, 89, 18) = 0.0 + k[1367] + k[1368] + k[1369] + k[1370];
    IJth(jmatrix, 89, 26) = 0.0 - k[937]*y[IDX_O2HI] - k[938]*y[IDX_O2HI];
    IJth(jmatrix, 89, 30) = 0.0 + k[913]*y[IDX_O2I] - k[914]*y[IDX_O2HI];
    IJth(jmatrix, 89, 33) = 0.0 + k[921]*y[IDX_O2I];
    IJth(jmatrix, 89, 37) = 0.0 - k[954]*y[IDX_O2HI];
    IJth(jmatrix, 89, 41) = 0.0 - k[990]*y[IDX_O2HI] - k[991]*y[IDX_O2HI] -
        k[992]*y[IDX_O2HI];
    IJth(jmatrix, 89, 43) = 0.0 + k[963]*y[IDX_O2I];
    IJth(jmatrix, 89, 47) = 0.0 + k[549]*y[IDX_O2I];
    IJth(jmatrix, 89, 58) = 0.0 + k[1002]*y[IDX_O2I] - k[1003]*y[IDX_O2HI];
    IJth(jmatrix, 89, 71) = 0.0 - k[1024]*y[IDX_O2HI];
    IJth(jmatrix, 89, 85) = 0.0 - k[1077]*y[IDX_O2HI];
    IJth(jmatrix, 89, 87) = 0.0 + k[549]*y[IDX_H2COII] + k[913]*y[IDX_CH3I]
        + k[921]*y[IDX_CH4I] + k[963]*y[IDX_H2I] + k[1002]*y[IDX_HCOI];
    IJth(jmatrix, 89, 89) = 0.0 - k[281] - k[914]*y[IDX_CH3I] -
        k[937]*y[IDX_CHI] - k[938]*y[IDX_CHI] - k[954]*y[IDX_COI] -
        k[990]*y[IDX_HI] - k[991]*y[IDX_HI] - k[992]*y[IDX_HI] -
        k[1003]*y[IDX_HCOI] - k[1024]*y[IDX_NI] - k[1077]*y[IDX_OI] -
        k[1100]*y[IDX_OHI] - k[1170] - k[1171] - k[1303];
    IJth(jmatrix, 89, 92) = 0.0 - k[1100]*y[IDX_O2HI];
    IJth(jmatrix, 90, 24) = 0.0 - k[392]*y[IDX_O2HII];
    IJth(jmatrix, 90, 26) = 0.0 - k[474]*y[IDX_O2HII];
    IJth(jmatrix, 90, 28) = 0.0 - k[436]*y[IDX_O2HII];
    IJth(jmatrix, 90, 35) = 0.0 - k[483]*y[IDX_O2HII];
    IJth(jmatrix, 90, 37) = 0.0 - k[488]*y[IDX_O2HII];
    IJth(jmatrix, 90, 39) = 0.0 - k[821]*y[IDX_O2HII];
    IJth(jmatrix, 90, 40) = 0.0 - k[347]*y[IDX_O2HII];
    IJth(jmatrix, 90, 43) = 0.0 - k[544]*y[IDX_O2HII];
    IJth(jmatrix, 90, 44) = 0.0 + k[525]*y[IDX_O2I];
    IJth(jmatrix, 90, 46) = 0.0 - k[552]*y[IDX_O2HII];
    IJth(jmatrix, 90, 49) = 0.0 - k[571]*y[IDX_O2HII];
    IJth(jmatrix, 90, 52) = 0.0 + k[597]*y[IDX_O2I];
    IJth(jmatrix, 90, 55) = 0.0 - k[632]*y[IDX_O2HII];
    IJth(jmatrix, 90, 58) = 0.0 + k[644]*y[IDX_O2II] - k[645]*y[IDX_O2HII];
    IJth(jmatrix, 90, 64) = 0.0 - k[651]*y[IDX_O2HII];
    IJth(jmatrix, 90, 73) = 0.0 - k[733]*y[IDX_O2HII];
    IJth(jmatrix, 90, 76) = 0.0 - k[805]*y[IDX_O2HII];
    IJth(jmatrix, 90, 77) = 0.0 + k[766]*y[IDX_O2I];
    IJth(jmatrix, 90, 78) = 0.0 - k[790]*y[IDX_O2HII];
    IJth(jmatrix, 90, 82) = 0.0 - k[807]*y[IDX_O2HII];
    IJth(jmatrix, 90, 85) = 0.0 - k[829]*y[IDX_O2HII];
    IJth(jmatrix, 90, 87) = 0.0 + k[525]*y[IDX_H2II] + k[597]*y[IDX_H3II] +
        k[766]*y[IDX_NHII];
    IJth(jmatrix, 90, 88) = 0.0 + k[644]*y[IDX_HCOI];
    IJth(jmatrix, 90, 90) = 0.0 - k[347]*y[IDX_EM] - k[392]*y[IDX_CI] -
        k[436]*y[IDX_CH2I] - k[474]*y[IDX_CHI] - k[483]*y[IDX_CNI] -
        k[488]*y[IDX_COI] - k[544]*y[IDX_H2I] - k[552]*y[IDX_H2COI] -
        k[571]*y[IDX_H2OI] - k[632]*y[IDX_HCNI] - k[645]*y[IDX_HCOI] -
        k[651]*y[IDX_HNCI] - k[733]*y[IDX_N2I] - k[790]*y[IDX_NH2I] -
        k[805]*y[IDX_NHI] - k[807]*y[IDX_NOI] - k[821]*y[IDX_CO2I] -
        k[829]*y[IDX_OI] - k[858]*y[IDX_OHI] - k[1297];
    IJth(jmatrix, 90, 92) = 0.0 - k[858]*y[IDX_O2HII];
    IJth(jmatrix, 91, 24) = 0.0 - k[874]*y[IDX_OCNI];
    IJth(jmatrix, 91, 25) = 0.0 - k[378]*y[IDX_OCNI];
    IJth(jmatrix, 91, 26) = 0.0 + k[932]*y[IDX_NOI];
    IJth(jmatrix, 91, 35) = 0.0 + k[945]*y[IDX_NO2I] + k[947]*y[IDX_NOI] +
        k[949]*y[IDX_O2I] + k[1091]*y[IDX_OHI];
    IJth(jmatrix, 91, 41) = 0.0 - k[993]*y[IDX_OCNI] - k[994]*y[IDX_OCNI] -
        k[995]*y[IDX_OCNI];
    IJth(jmatrix, 91, 45) = 0.0 + k[1060]*y[IDX_OI];
    IJth(jmatrix, 91, 55) = 0.0 + k[1065]*y[IDX_OI];
    IJth(jmatrix, 91, 58) = 0.0 + k[1016]*y[IDX_NI];
    IJth(jmatrix, 91, 62) = 0.0 - k[697]*y[IDX_OCNI] - k[698]*y[IDX_OCNI];
    IJth(jmatrix, 91, 71) = 0.0 + k[1016]*y[IDX_HCOI];
    IJth(jmatrix, 91, 82) = 0.0 + k[932]*y[IDX_CHI] + k[947]*y[IDX_CNI] -
        k[1053]*y[IDX_OCNI];
    IJth(jmatrix, 91, 84) = 0.0 + k[945]*y[IDX_CNI];
    IJth(jmatrix, 91, 85) = 0.0 + k[1060]*y[IDX_H2CNI] + k[1065]*y[IDX_HCNI]
        - k[1078]*y[IDX_OCNI] - k[1079]*y[IDX_OCNI];
    IJth(jmatrix, 91, 87) = 0.0 + k[949]*y[IDX_CNI] - k[1054]*y[IDX_OCNI] -
        k[1055]*y[IDX_OCNI];
    IJth(jmatrix, 91, 91) = 0.0 - k[283] - k[378]*y[IDX_CII] -
        k[697]*y[IDX_HeII] - k[698]*y[IDX_HeII] - k[874]*y[IDX_CI] -
        k[993]*y[IDX_HI] - k[994]*y[IDX_HI] - k[995]*y[IDX_HI] -
        k[1053]*y[IDX_NOI] - k[1054]*y[IDX_O2I] - k[1055]*y[IDX_O2I] -
        k[1078]*y[IDX_OI] - k[1079]*y[IDX_OI] - k[1172] - k[1225];
    IJth(jmatrix, 91, 92) = 0.0 + k[1091]*y[IDX_CNI];
    IJth(jmatrix, 92, 24) = 0.0 + k[383]*y[IDX_H2OII] - k[875]*y[IDX_OHI] -
        k[876]*y[IDX_OHI];
    IJth(jmatrix, 92, 25) = 0.0 - k[379]*y[IDX_OHI];
    IJth(jmatrix, 92, 26) = 0.0 + k[62]*y[IDX_OHII] + k[460]*y[IDX_H2OII] +
        k[935]*y[IDX_O2I] + k[937]*y[IDX_O2HI] + k[940]*y[IDX_OI] -
        k[941]*y[IDX_OHI];
    IJth(jmatrix, 92, 27) = 0.0 + k[411]*y[IDX_O2I] - k[415]*y[IDX_OHI];
    IJth(jmatrix, 92, 28) = 0.0 + k[45]*y[IDX_OHII] + k[424]*y[IDX_H2OII] +
        k[887]*y[IDX_NOI] + k[893]*y[IDX_O2I] + k[897]*y[IDX_OI] -
        k[898]*y[IDX_OHI] - k[899]*y[IDX_OHI] - k[900]*y[IDX_OHI];
    IJth(jmatrix, 92, 29) = 0.0 + k[420]*y[IDX_O2I];
    IJth(jmatrix, 92, 30) = 0.0 + k[904]*y[IDX_H2OI] + k[911]*y[IDX_O2I] -
        k[917]*y[IDX_OHI] - k[918]*y[IDX_OHI] - k[919]*y[IDX_OHI];
    IJth(jmatrix, 92, 31) = 0.0 - k[445]*y[IDX_OHI];
    IJth(jmatrix, 92, 32) = 0.0 + k[248] + k[657]*y[IDX_HeII] +
        k[809]*y[IDX_OII] + k[1121];
    IJth(jmatrix, 92, 33) = 0.0 + k[810]*y[IDX_OII] - k[922]*y[IDX_OHI] +
        k[1056]*y[IDX_OI];
    IJth(jmatrix, 92, 34) = 0.0 + k[822]*y[IDX_OI];
    IJth(jmatrix, 92, 35) = 0.0 - k[1090]*y[IDX_OHI] - k[1091]*y[IDX_OHI];
    IJth(jmatrix, 92, 36) = 0.0 - k[225]*y[IDX_OHI] + k[560]*y[IDX_H2OI];
    IJth(jmatrix, 92, 37) = 0.0 + k[553]*y[IDX_H2OII] + k[954]*y[IDX_O2HI] +
        k[972]*y[IDX_HI] - k[1092]*y[IDX_OHI];
    IJth(jmatrix, 92, 38) = 0.0 - k[226]*y[IDX_OHI] + k[562]*y[IDX_H2OI] -
        k[851]*y[IDX_OHI];
    IJth(jmatrix, 92, 39) = 0.0 + k[971]*y[IDX_HI];
    IJth(jmatrix, 92, 40) = 0.0 + k[314]*y[IDX_H2OII] + k[317]*y[IDX_H3COII]
        + k[324]*y[IDX_H3OII] + k[325]*y[IDX_H3OII] + k[333]*y[IDX_HCO2II] +
        k[363]*y[IDX_SiOHII];
    IJth(jmatrix, 92, 41) = 0.0 + k[11]*y[IDX_H2OI] - k[13]*y[IDX_OHI] +
        k[971]*y[IDX_CO2I] + k[972]*y[IDX_COI] + k[975]*y[IDX_H2OI] +
        k[982]*y[IDX_HNOI] + k[986]*y[IDX_NO2I] + k[988]*y[IDX_NOI] +
        k[989]*y[IDX_O2I] + k[992]*y[IDX_O2HI] + k[992]*y[IDX_O2HI] +
        k[995]*y[IDX_OCNI] - k[996]*y[IDX_OHI] + k[1207]*y[IDX_OI] -
        k[1208]*y[IDX_OHI];
    IJth(jmatrix, 92, 42) = 0.0 - k[90]*y[IDX_OHI] + k[504]*y[IDX_NO2I];
    IJth(jmatrix, 92, 43) = 0.0 + k[4]*y[IDX_H2OI] - k[7]*y[IDX_OHI] +
        k[964]*y[IDX_O2I] + k[964]*y[IDX_O2I] + k[965]*y[IDX_OI] -
        k[966]*y[IDX_OHI];
    IJth(jmatrix, 92, 44) = 0.0 - k[114]*y[IDX_OHI] - k[527]*y[IDX_OHI];
    IJth(jmatrix, 92, 46) = 0.0 + k[219]*y[IDX_OHII] + k[554]*y[IDX_H2OII] +
        k[813]*y[IDX_OII] + k[1061]*y[IDX_OI] - k[1093]*y[IDX_OHI];
    IJth(jmatrix, 92, 49) = 0.0 + k[4]*y[IDX_H2I] + k[11]*y[IDX_HI] +
        k[220]*y[IDX_OHII] + k[256] + k[555]*y[IDX_H2OII] + k[560]*y[IDX_CNII] +
        k[562]*y[IDX_COII] + k[569]*y[IDX_N2II] + k[674]*y[IDX_HeII] +
        k[757]*y[IDX_NHII] + k[772]*y[IDX_NH2II] + k[904]*y[IDX_CH3I] +
        k[975]*y[IDX_HI] + k[1036]*y[IDX_NHI] + k[1062]*y[IDX_OI] +
        k[1062]*y[IDX_OI] + k[1142];
    IJth(jmatrix, 92, 50) = 0.0 + k[314]*y[IDX_EM] + k[383]*y[IDX_CI] +
        k[424]*y[IDX_CH2I] + k[460]*y[IDX_CHI] + k[553]*y[IDX_COI] +
        k[554]*y[IDX_H2COI] + k[555]*y[IDX_H2OI] + k[556]*y[IDX_HCNI] +
        k[558]*y[IDX_HCOI] + k[559]*y[IDX_HNCI] + k[781]*y[IDX_NH2I] -
        k[852]*y[IDX_OHI];
    IJth(jmatrix, 92, 52) = 0.0 + k[595]*y[IDX_NO2I] - k[600]*y[IDX_OHI];
    IJth(jmatrix, 92, 53) = 0.0 + k[317]*y[IDX_EM];
    IJth(jmatrix, 92, 54) = 0.0 + k[324]*y[IDX_EM] + k[325]*y[IDX_EM];
    IJth(jmatrix, 92, 55) = 0.0 + k[556]*y[IDX_H2OII] + k[1063]*y[IDX_OI] -
        k[1094]*y[IDX_OHI] - k[1095]*y[IDX_OHI];
    IJth(jmatrix, 92, 56) = 0.0 - k[853]*y[IDX_OHI];
    IJth(jmatrix, 92, 58) = 0.0 + k[221]*y[IDX_OHII] + k[558]*y[IDX_H2OII] +
        k[1001]*y[IDX_O2I] + k[1067]*y[IDX_OI] - k[1096]*y[IDX_OHI];
    IJth(jmatrix, 92, 59) = 0.0 - k[854]*y[IDX_OHI] - k[855]*y[IDX_OHI];
    IJth(jmatrix, 92, 60) = 0.0 + k[333]*y[IDX_EM];
    IJth(jmatrix, 92, 62) = 0.0 + k[657]*y[IDX_CH3OHI] + k[674]*y[IDX_H2OI]
        - k[699]*y[IDX_OHI];
    IJth(jmatrix, 92, 64) = 0.0 + k[559]*y[IDX_H2OII];
    IJth(jmatrix, 92, 66) = 0.0 + k[982]*y[IDX_HI] + k[1069]*y[IDX_OI] -
        k[1097]*y[IDX_OHI];
    IJth(jmatrix, 92, 67) = 0.0 - k[856]*y[IDX_OHI];
    IJth(jmatrix, 92, 71) = 0.0 - k[1025]*y[IDX_OHI] - k[1026]*y[IDX_OHI];
    IJth(jmatrix, 92, 72) = 0.0 - k[169]*y[IDX_OHI];
    IJth(jmatrix, 92, 74) = 0.0 - k[227]*y[IDX_OHI] + k[569]*y[IDX_H2OI];
    IJth(jmatrix, 92, 75) = 0.0 - k[857]*y[IDX_OHI];
    IJth(jmatrix, 92, 76) = 0.0 + k[1036]*y[IDX_H2OI] + k[1043]*y[IDX_NOI] +
        k[1045]*y[IDX_O2I] + k[1047]*y[IDX_OI] - k[1048]*y[IDX_OHI] -
        k[1049]*y[IDX_OHI] - k[1050]*y[IDX_OHI];
    IJth(jmatrix, 92, 77) = 0.0 + k[757]*y[IDX_H2OI] + k[765]*y[IDX_O2I] -
        k[768]*y[IDX_OHI];
    IJth(jmatrix, 92, 78) = 0.0 + k[188]*y[IDX_OHII] + k[781]*y[IDX_H2OII] +
        k[1030]*y[IDX_NOI] - k[1031]*y[IDX_OHI] - k[1032]*y[IDX_OHI] +
        k[1073]*y[IDX_OI];
    IJth(jmatrix, 92, 79) = 0.0 + k[772]*y[IDX_H2OI] + k[778]*y[IDX_O2I];
    IJth(jmatrix, 92, 80) = 0.0 + k[222]*y[IDX_OHII] + k[1074]*y[IDX_OI] -
        k[1098]*y[IDX_OHI];
    IJth(jmatrix, 92, 82) = 0.0 + k[223]*y[IDX_OHII] + k[887]*y[IDX_CH2I] +
        k[988]*y[IDX_HI] + k[1030]*y[IDX_NH2I] + k[1043]*y[IDX_NHI] -
        k[1099]*y[IDX_OHI];
    IJth(jmatrix, 92, 84) = 0.0 + k[504]*y[IDX_HII] + k[595]*y[IDX_H3II] +
        k[986]*y[IDX_HI];
    IJth(jmatrix, 92, 85) = 0.0 + k[822]*y[IDX_CH4II] + k[897]*y[IDX_CH2I] +
        k[940]*y[IDX_CHI] + k[965]*y[IDX_H2I] + k[1047]*y[IDX_NHI] +
        k[1056]*y[IDX_CH4I] + k[1061]*y[IDX_H2COI] + k[1062]*y[IDX_H2OI] +
        k[1062]*y[IDX_H2OI] + k[1063]*y[IDX_HCNI] + k[1067]*y[IDX_HCOI] +
        k[1069]*y[IDX_HNOI] + k[1073]*y[IDX_NH2I] + k[1074]*y[IDX_NH3I] +
        k[1077]*y[IDX_O2HI] - k[1080]*y[IDX_OHI] + k[1088]*y[IDX_SiH4I] +
        k[1207]*y[IDX_HI];
    IJth(jmatrix, 92, 86) = 0.0 - k[215]*y[IDX_OHI] + k[809]*y[IDX_CH3OHI] +
        k[810]*y[IDX_CH4I] + k[813]*y[IDX_H2COI] - k[819]*y[IDX_OHI];
    IJth(jmatrix, 92, 87) = 0.0 + k[224]*y[IDX_OHII] + k[411]*y[IDX_CHII] +
        k[420]*y[IDX_CH2II] + k[765]*y[IDX_NHII] + k[778]*y[IDX_NH2II] +
        k[862]*y[IDX_SiH2II] + k[893]*y[IDX_CH2I] + k[911]*y[IDX_CH3I] +
        k[935]*y[IDX_CHI] + k[964]*y[IDX_H2I] + k[964]*y[IDX_H2I] +
        k[989]*y[IDX_HI] + k[1001]*y[IDX_HCOI] + k[1045]*y[IDX_NHI];
    IJth(jmatrix, 92, 89) = 0.0 + k[937]*y[IDX_CHI] + k[954]*y[IDX_COI] +
        k[992]*y[IDX_HI] + k[992]*y[IDX_HI] + k[1077]*y[IDX_OI] -
        k[1100]*y[IDX_OHI] + k[1171];
    IJth(jmatrix, 92, 90) = 0.0 - k[858]*y[IDX_OHI];
    IJth(jmatrix, 92, 91) = 0.0 + k[995]*y[IDX_HI];
    IJth(jmatrix, 92, 92) = 0.0 - k[7]*y[IDX_H2I] - k[13]*y[IDX_HI] -
        k[90]*y[IDX_HII] - k[114]*y[IDX_H2II] - k[169]*y[IDX_NII] -
        k[215]*y[IDX_OII] - k[225]*y[IDX_CNII] - k[226]*y[IDX_COII] -
        k[227]*y[IDX_N2II] - k[284] - k[379]*y[IDX_CII] - k[415]*y[IDX_CHII] -
        k[445]*y[IDX_CH3II] - k[527]*y[IDX_H2II] - k[600]*y[IDX_H3II] -
        k[699]*y[IDX_HeII] - k[768]*y[IDX_NHII] - k[819]*y[IDX_OII] -
        k[847]*y[IDX_OHII] - k[851]*y[IDX_COII] - k[852]*y[IDX_H2OII] -
        k[853]*y[IDX_HCNII] - k[854]*y[IDX_HCOII] - k[855]*y[IDX_HCOII] -
        k[856]*y[IDX_HNOII] - k[857]*y[IDX_N2HII] - k[858]*y[IDX_O2HII] -
        k[859]*y[IDX_SiII] - k[875]*y[IDX_CI] - k[876]*y[IDX_CI] -
        k[898]*y[IDX_CH2I] - k[899]*y[IDX_CH2I] - k[900]*y[IDX_CH2I] -
        k[917]*y[IDX_CH3I] - k[918]*y[IDX_CH3I] - k[919]*y[IDX_CH3I] -
        k[922]*y[IDX_CH4I] - k[941]*y[IDX_CHI] - k[966]*y[IDX_H2I] -
        k[996]*y[IDX_HI] - k[1025]*y[IDX_NI] - k[1026]*y[IDX_NI] -
        k[1031]*y[IDX_NH2I] - k[1032]*y[IDX_NH2I] - k[1048]*y[IDX_NHI] -
        k[1049]*y[IDX_NHI] - k[1050]*y[IDX_NHI] - k[1080]*y[IDX_OI] -
        k[1090]*y[IDX_CNI] - k[1091]*y[IDX_CNI] - k[1092]*y[IDX_COI] -
        k[1093]*y[IDX_H2COI] - k[1094]*y[IDX_HCNI] - k[1095]*y[IDX_HCNI] -
        k[1096]*y[IDX_HCOI] - k[1097]*y[IDX_HNOI] - k[1098]*y[IDX_NH3I] -
        k[1099]*y[IDX_NOI] - k[1100]*y[IDX_O2HI] - k[1101]*y[IDX_OHI] -
        k[1101]*y[IDX_OHI] - k[1101]*y[IDX_OHI] - k[1101]*y[IDX_OHI] -
        k[1102]*y[IDX_SiI] - k[1174] - k[1175] - k[1208]*y[IDX_HI] - k[1254];
    IJth(jmatrix, 92, 93) = 0.0 + k[45]*y[IDX_CH2I] + k[62]*y[IDX_CHI] +
        k[188]*y[IDX_NH2I] + k[219]*y[IDX_H2COI] + k[220]*y[IDX_H2OI] +
        k[221]*y[IDX_HCOI] + k[222]*y[IDX_NH3I] + k[223]*y[IDX_NOI] +
        k[224]*y[IDX_O2I] - k[847]*y[IDX_OHI];
    IJth(jmatrix, 92, 94) = 0.0 - k[1102]*y[IDX_OHI];
    IJth(jmatrix, 92, 95) = 0.0 - k[859]*y[IDX_OHI];
    IJth(jmatrix, 92, 105) = 0.0 + k[862]*y[IDX_O2I];
    IJth(jmatrix, 92, 108) = 0.0 + k[1088]*y[IDX_OI];
    IJth(jmatrix, 92, 113) = 0.0 + k[363]*y[IDX_EM];
    IJth(jmatrix, 93, 24) = 0.0 - k[393]*y[IDX_OHII];
    IJth(jmatrix, 93, 26) = 0.0 - k[62]*y[IDX_OHII] - k[475]*y[IDX_OHII];
    IJth(jmatrix, 93, 28) = 0.0 - k[45]*y[IDX_OHII] - k[437]*y[IDX_OHII];
    IJth(jmatrix, 93, 32) = 0.0 + k[656]*y[IDX_HeII];
    IJth(jmatrix, 93, 33) = 0.0 - k[457]*y[IDX_OHII];
    IJth(jmatrix, 93, 35) = 0.0 - k[836]*y[IDX_OHII];
    IJth(jmatrix, 93, 36) = 0.0 + k[225]*y[IDX_OHI];
    IJth(jmatrix, 93, 37) = 0.0 - k[838]*y[IDX_OHII];
    IJth(jmatrix, 93, 38) = 0.0 + k[226]*y[IDX_OHI];
    IJth(jmatrix, 93, 39) = 0.0 - k[837]*y[IDX_OHII];
    IJth(jmatrix, 93, 40) = 0.0 - k[348]*y[IDX_OHII];
    IJth(jmatrix, 93, 42) = 0.0 + k[90]*y[IDX_OHI];
    IJth(jmatrix, 93, 43) = 0.0 + k[543]*y[IDX_OII] - k[545]*y[IDX_OHII];
    IJth(jmatrix, 93, 44) = 0.0 + k[114]*y[IDX_OHI] + k[526]*y[IDX_OI];
    IJth(jmatrix, 93, 46) = 0.0 - k[219]*y[IDX_OHII] - k[839]*y[IDX_OHII];
    IJth(jmatrix, 93, 49) = 0.0 - k[220]*y[IDX_OHII] + k[673]*y[IDX_HeII] -
        k[840]*y[IDX_OHII];
    IJth(jmatrix, 93, 50) = 0.0 + k[1140];
    IJth(jmatrix, 93, 52) = 0.0 + k[599]*y[IDX_OI];
    IJth(jmatrix, 93, 55) = 0.0 - k[841]*y[IDX_OHII];
    IJth(jmatrix, 93, 58) = 0.0 - k[221]*y[IDX_OHII] + k[816]*y[IDX_OII] -
        k[842]*y[IDX_OHII] - k[843]*y[IDX_OHII];
    IJth(jmatrix, 93, 62) = 0.0 + k[656]*y[IDX_CH3OHI] + k[673]*y[IDX_H2OI];
    IJth(jmatrix, 93, 64) = 0.0 - k[844]*y[IDX_OHII];
    IJth(jmatrix, 93, 71) = 0.0 - k[743]*y[IDX_OHII];
    IJth(jmatrix, 93, 72) = 0.0 + k[169]*y[IDX_OHI];
    IJth(jmatrix, 93, 73) = 0.0 - k[845]*y[IDX_OHII];
    IJth(jmatrix, 93, 74) = 0.0 + k[227]*y[IDX_OHI];
    IJth(jmatrix, 93, 75) = 0.0 + k[826]*y[IDX_OI];
    IJth(jmatrix, 93, 76) = 0.0 - k[806]*y[IDX_OHII];
    IJth(jmatrix, 93, 77) = 0.0 + k[767]*y[IDX_OI];
    IJth(jmatrix, 93, 78) = 0.0 - k[188]*y[IDX_OHII] - k[791]*y[IDX_OHII];
    IJth(jmatrix, 93, 80) = 0.0 - k[222]*y[IDX_OHII];
    IJth(jmatrix, 93, 82) = 0.0 - k[223]*y[IDX_OHII] - k[846]*y[IDX_OHII];
    IJth(jmatrix, 93, 85) = 0.0 + k[526]*y[IDX_H2II] + k[599]*y[IDX_H3II] +
        k[767]*y[IDX_NHII] + k[826]*y[IDX_N2HII] + k[829]*y[IDX_O2HII] -
        k[830]*y[IDX_OHII];
    IJth(jmatrix, 93, 86) = 0.0 + k[215]*y[IDX_OHI] + k[543]*y[IDX_H2I] +
        k[816]*y[IDX_HCOI];
    IJth(jmatrix, 93, 87) = 0.0 - k[224]*y[IDX_OHII];
    IJth(jmatrix, 93, 90) = 0.0 + k[829]*y[IDX_OI];
    IJth(jmatrix, 93, 92) = 0.0 + k[90]*y[IDX_HII] + k[114]*y[IDX_H2II] +
        k[169]*y[IDX_NII] + k[215]*y[IDX_OII] + k[225]*y[IDX_CNII] +
        k[226]*y[IDX_COII] + k[227]*y[IDX_N2II] - k[847]*y[IDX_OHII] + k[1175];
    IJth(jmatrix, 93, 93) = 0.0 - k[45]*y[IDX_CH2I] - k[62]*y[IDX_CHI] -
        k[188]*y[IDX_NH2I] - k[219]*y[IDX_H2COI] - k[220]*y[IDX_H2OI] -
        k[221]*y[IDX_HCOI] - k[222]*y[IDX_NH3I] - k[223]*y[IDX_NOI] -
        k[224]*y[IDX_O2I] - k[348]*y[IDX_EM] - k[393]*y[IDX_CI] -
        k[437]*y[IDX_CH2I] - k[457]*y[IDX_CH4I] - k[475]*y[IDX_CHI] -
        k[545]*y[IDX_H2I] - k[743]*y[IDX_NI] - k[791]*y[IDX_NH2I] -
        k[806]*y[IDX_NHI] - k[830]*y[IDX_OI] - k[836]*y[IDX_CNI] -
        k[837]*y[IDX_CO2I] - k[838]*y[IDX_COI] - k[839]*y[IDX_H2COI] -
        k[840]*y[IDX_H2OI] - k[841]*y[IDX_HCNI] - k[842]*y[IDX_HCOI] -
        k[843]*y[IDX_HCOI] - k[844]*y[IDX_HNCI] - k[845]*y[IDX_N2I] -
        k[846]*y[IDX_NOI] - k[847]*y[IDX_OHI] - k[848]*y[IDX_SiI] -
        k[849]*y[IDX_SiHI] - k[850]*y[IDX_SiOI] - k[1173] - k[1274];
    IJth(jmatrix, 93, 94) = 0.0 - k[848]*y[IDX_OHII];
    IJth(jmatrix, 93, 102) = 0.0 - k[849]*y[IDX_OHII];
    IJth(jmatrix, 93, 111) = 0.0 - k[850]*y[IDX_OHII];
    IJth(jmatrix, 94, 25) = 0.0 - k[21]*y[IDX_SiI];
    IJth(jmatrix, 94, 26) = 0.0 + k[477]*y[IDX_SiHII] + k[478]*y[IDX_SiOII];
    IJth(jmatrix, 94, 27) = 0.0 - k[35]*y[IDX_SiI];
    IJth(jmatrix, 94, 37) = 0.0 - k[1104]*y[IDX_SiI];
    IJth(jmatrix, 94, 39) = 0.0 - k[1103]*y[IDX_SiI];
    IJth(jmatrix, 94, 40) = 0.0 + k[349]*y[IDX_SiCII] + k[352]*y[IDX_SiHII]
        + k[353]*y[IDX_SiH2II] + k[354]*y[IDX_SiH2II] + k[362]*y[IDX_SiOII] +
        k[363]*y[IDX_SiOHII] + k[1222]*y[IDX_SiII];
    IJth(jmatrix, 94, 42) = 0.0 - k[91]*y[IDX_SiI];
    IJth(jmatrix, 94, 47) = 0.0 - k[228]*y[IDX_SiI];
    IJth(jmatrix, 94, 49) = 0.0 + k[573]*y[IDX_SiHII];
    IJth(jmatrix, 94, 50) = 0.0 - k[122]*y[IDX_SiI];
    IJth(jmatrix, 94, 52) = 0.0 - k[601]*y[IDX_SiI];
    IJth(jmatrix, 94, 54) = 0.0 - k[610]*y[IDX_SiI];
    IJth(jmatrix, 94, 59) = 0.0 - k[861]*y[IDX_SiI];
    IJth(jmatrix, 94, 62) = 0.0 - k[147]*y[IDX_SiI] + k[702]*y[IDX_SiCI] +
        k[711]*y[IDX_SiOI];
    IJth(jmatrix, 94, 69) = 0.0 + k[153]*y[IDX_SiII];
    IJth(jmatrix, 94, 71) = 0.0 + k[745]*y[IDX_SiOII] + k[1027]*y[IDX_SiCI];
    IJth(jmatrix, 94, 81) = 0.0 - k[192]*y[IDX_SiI];
    IJth(jmatrix, 94, 82) = 0.0 - k[1105]*y[IDX_SiI];
    IJth(jmatrix, 94, 83) = 0.0 - k[229]*y[IDX_SiI];
    IJth(jmatrix, 94, 85) = 0.0 + k[1083]*y[IDX_SiCI] - k[1213]*y[IDX_SiI];
    IJth(jmatrix, 94, 87) = 0.0 - k[1106]*y[IDX_SiI];
    IJth(jmatrix, 94, 88) = 0.0 - k[230]*y[IDX_SiI];
    IJth(jmatrix, 94, 92) = 0.0 - k[1102]*y[IDX_SiI];
    IJth(jmatrix, 94, 93) = 0.0 - k[848]*y[IDX_SiI];
    IJth(jmatrix, 94, 94) = 0.0 - k[21]*y[IDX_CII] - k[35]*y[IDX_CHII] -
        k[91]*y[IDX_HII] - k[122]*y[IDX_H2OII] - k[147]*y[IDX_HeII] -
        k[192]*y[IDX_NH3II] - k[228]*y[IDX_H2COII] - k[229]*y[IDX_NOII] -
        k[230]*y[IDX_O2II] - k[285] - k[601]*y[IDX_H3II] - k[610]*y[IDX_H3OII] -
        k[848]*y[IDX_OHII] - k[861]*y[IDX_HCOII] - k[1102]*y[IDX_OHI] -
        k[1103]*y[IDX_CO2I] - k[1104]*y[IDX_COI] - k[1105]*y[IDX_NOI] -
        k[1106]*y[IDX_O2I] - k[1176] - k[1213]*y[IDX_OI] - k[1228];
    IJth(jmatrix, 94, 95) = 0.0 + k[153]*y[IDX_MgI] + k[1222]*y[IDX_EM];
    IJth(jmatrix, 94, 96) = 0.0 + k[288] + k[702]*y[IDX_HeII] +
        k[1027]*y[IDX_NI] + k[1083]*y[IDX_OI] + k[1178];
    IJth(jmatrix, 94, 97) = 0.0 + k[349]*y[IDX_EM];
    IJth(jmatrix, 94, 102) = 0.0 + k[292] + k[1188];
    IJth(jmatrix, 94, 103) = 0.0 + k[352]*y[IDX_EM] + k[477]*y[IDX_CHI] +
        k[573]*y[IDX_H2OI];
    IJth(jmatrix, 94, 105) = 0.0 + k[353]*y[IDX_EM] + k[354]*y[IDX_EM];
    IJth(jmatrix, 94, 111) = 0.0 + k[293] + k[711]*y[IDX_HeII] + k[1190];
    IJth(jmatrix, 94, 112) = 0.0 + k[362]*y[IDX_EM] + k[478]*y[IDX_CHI] +
        k[745]*y[IDX_NI];
    IJth(jmatrix, 94, 113) = 0.0 + k[363]*y[IDX_EM];
    IJth(jmatrix, 95, 24) = 0.0 + k[395]*y[IDX_SiOII];
    IJth(jmatrix, 95, 25) = 0.0 + k[21]*y[IDX_SiI] + k[382]*y[IDX_SiOI];
    IJth(jmatrix, 95, 26) = 0.0 - k[476]*y[IDX_SiII];
    IJth(jmatrix, 95, 27) = 0.0 + k[35]*y[IDX_SiI];
    IJth(jmatrix, 95, 28) = 0.0 + k[438]*y[IDX_SiOII];
    IJth(jmatrix, 95, 32) = 0.0 - k[860]*y[IDX_SiII];
    IJth(jmatrix, 95, 37) = 0.0 + k[490]*y[IDX_SiOII];
    IJth(jmatrix, 95, 40) = 0.0 - k[1222]*y[IDX_SiII];
    IJth(jmatrix, 95, 41) = 0.0 + k[619]*y[IDX_SiHII] - k[1209]*y[IDX_SiII];
    IJth(jmatrix, 95, 42) = 0.0 + k[91]*y[IDX_SiI] + k[508]*y[IDX_SiHI];
    IJth(jmatrix, 95, 43) = 0.0 - k[1202]*y[IDX_SiII];
    IJth(jmatrix, 95, 47) = 0.0 + k[228]*y[IDX_SiI];
    IJth(jmatrix, 95, 49) = 0.0 - k[572]*y[IDX_SiII];
    IJth(jmatrix, 95, 50) = 0.0 + k[122]*y[IDX_SiI];
    IJth(jmatrix, 95, 62) = 0.0 + k[147]*y[IDX_SiI] + k[701]*y[IDX_SiCI] +
        k[703]*y[IDX_SiH2I] + k[707]*y[IDX_SiH4I] + k[709]*y[IDX_SiHI] +
        k[710]*y[IDX_SiOI];
    IJth(jmatrix, 95, 69) = 0.0 - k[153]*y[IDX_SiII];
    IJth(jmatrix, 95, 71) = 0.0 + k[744]*y[IDX_SiCII] + k[746]*y[IDX_SiOII];
    IJth(jmatrix, 95, 81) = 0.0 + k[192]*y[IDX_SiI];
    IJth(jmatrix, 95, 83) = 0.0 + k[229]*y[IDX_SiI];
    IJth(jmatrix, 95, 85) = 0.0 + k[835]*y[IDX_SiOII] - k[1212]*y[IDX_SiII];
    IJth(jmatrix, 95, 88) = 0.0 + k[230]*y[IDX_SiI];
    IJth(jmatrix, 95, 92) = 0.0 - k[859]*y[IDX_SiII];
    IJth(jmatrix, 95, 94) = 0.0 + k[21]*y[IDX_CII] + k[35]*y[IDX_CHII] +
        k[91]*y[IDX_HII] + k[122]*y[IDX_H2OII] + k[147]*y[IDX_HeII] +
        k[192]*y[IDX_NH3II] + k[228]*y[IDX_H2COII] + k[229]*y[IDX_NOII] +
        k[230]*y[IDX_O2II] + k[285] + k[1176];
    IJth(jmatrix, 95, 95) = 0.0 - k[153]*y[IDX_MgI] - k[476]*y[IDX_CHI] -
        k[572]*y[IDX_H2OI] - k[859]*y[IDX_OHI] - k[860]*y[IDX_CH3OHI] -
        k[1202]*y[IDX_H2I] - k[1209]*y[IDX_HI] - k[1212]*y[IDX_OI] -
        k[1222]*y[IDX_EM] - k[1231];
    IJth(jmatrix, 95, 96) = 0.0 + k[701]*y[IDX_HeII];
    IJth(jmatrix, 95, 97) = 0.0 + k[744]*y[IDX_NI];
    IJth(jmatrix, 95, 102) = 0.0 + k[508]*y[IDX_HII] + k[709]*y[IDX_HeII];
    IJth(jmatrix, 95, 103) = 0.0 + k[619]*y[IDX_HI] + k[1179];
    IJth(jmatrix, 95, 104) = 0.0 + k[703]*y[IDX_HeII];
    IJth(jmatrix, 95, 108) = 0.0 + k[707]*y[IDX_HeII];
    IJth(jmatrix, 95, 111) = 0.0 + k[382]*y[IDX_CII] + k[710]*y[IDX_HeII];
    IJth(jmatrix, 95, 112) = 0.0 + k[395]*y[IDX_CI] + k[438]*y[IDX_CH2I] +
        k[490]*y[IDX_COI] + k[746]*y[IDX_NI] + k[835]*y[IDX_OI] + k[1189];
    IJth(jmatrix, 96, 19) = 0.0 + k[1371] + k[1372] + k[1373] + k[1374];
    IJth(jmatrix, 96, 24) = 0.0 + k[877]*y[IDX_SiHI];
    IJth(jmatrix, 96, 25) = 0.0 - k[24]*y[IDX_SiCI];
    IJth(jmatrix, 96, 40) = 0.0 + k[350]*y[IDX_SiC2II];
    IJth(jmatrix, 96, 42) = 0.0 - k[94]*y[IDX_SiCI];
    IJth(jmatrix, 96, 62) = 0.0 - k[701]*y[IDX_SiCI] - k[702]*y[IDX_SiCI];
    IJth(jmatrix, 96, 71) = 0.0 - k[1027]*y[IDX_SiCI];
    IJth(jmatrix, 96, 85) = 0.0 + k[1081]*y[IDX_SiC2I] - k[1083]*y[IDX_SiCI]
        - k[1084]*y[IDX_SiCI];
    IJth(jmatrix, 96, 96) = 0.0 - k[24]*y[IDX_CII] - k[94]*y[IDX_HII] -
        k[288] - k[701]*y[IDX_HeII] - k[702]*y[IDX_HeII] - k[1027]*y[IDX_NI] -
        k[1083]*y[IDX_OI] - k[1084]*y[IDX_OI] - k[1178] - k[1239];
    IJth(jmatrix, 96, 98) = 0.0 + k[286] + k[1081]*y[IDX_OI];
    IJth(jmatrix, 96, 99) = 0.0 + k[350]*y[IDX_EM];
    IJth(jmatrix, 96, 102) = 0.0 + k[877]*y[IDX_CI];
    IJth(jmatrix, 97, 24) = 0.0 + k[394]*y[IDX_SiHII];
    IJth(jmatrix, 97, 25) = 0.0 + k[24]*y[IDX_SiCI] + k[380]*y[IDX_SiH2I] +
        k[381]*y[IDX_SiHI];
    IJth(jmatrix, 97, 26) = 0.0 + k[476]*y[IDX_SiII];
    IJth(jmatrix, 97, 40) = 0.0 - k[349]*y[IDX_SiCII];
    IJth(jmatrix, 97, 42) = 0.0 + k[94]*y[IDX_SiCI];
    IJth(jmatrix, 97, 71) = 0.0 - k[744]*y[IDX_SiCII];
    IJth(jmatrix, 97, 85) = 0.0 - k[831]*y[IDX_SiCII];
    IJth(jmatrix, 97, 95) = 0.0 + k[476]*y[IDX_CHI];
    IJth(jmatrix, 97, 96) = 0.0 + k[24]*y[IDX_CII] + k[94]*y[IDX_HII];
    IJth(jmatrix, 97, 97) = 0.0 - k[349]*y[IDX_EM] - k[744]*y[IDX_NI] -
        k[831]*y[IDX_OI] - k[1241];
    IJth(jmatrix, 97, 102) = 0.0 + k[381]*y[IDX_CII];
    IJth(jmatrix, 97, 103) = 0.0 + k[394]*y[IDX_CI];
    IJth(jmatrix, 97, 104) = 0.0 + k[380]*y[IDX_CII];
    IJth(jmatrix, 98, 20) = 0.0 + k[1395] + k[1396] + k[1397] + k[1398];
    IJth(jmatrix, 98, 25) = 0.0 - k[22]*y[IDX_SiC2I];
    IJth(jmatrix, 98, 40) = 0.0 + k[351]*y[IDX_SiC3II];
    IJth(jmatrix, 98, 42) = 0.0 - k[92]*y[IDX_SiC2I];
    IJth(jmatrix, 98, 85) = 0.0 - k[1081]*y[IDX_SiC2I] +
        k[1082]*y[IDX_SiC3I];
    IJth(jmatrix, 98, 98) = 0.0 - k[22]*y[IDX_CII] - k[92]*y[IDX_HII] -
        k[286] - k[1081]*y[IDX_OI] - k[1240];
    IJth(jmatrix, 98, 100) = 0.0 + k[287] + k[1082]*y[IDX_OI] + k[1177];
    IJth(jmatrix, 98, 101) = 0.0 + k[351]*y[IDX_EM];
    IJth(jmatrix, 99, 25) = 0.0 + k[22]*y[IDX_SiC2I];
    IJth(jmatrix, 99, 40) = 0.0 - k[350]*y[IDX_SiC2II];
    IJth(jmatrix, 99, 42) = 0.0 + k[92]*y[IDX_SiC2I];
    IJth(jmatrix, 99, 62) = 0.0 + k[700]*y[IDX_SiC3I];
    IJth(jmatrix, 99, 98) = 0.0 + k[22]*y[IDX_CII] + k[92]*y[IDX_HII];
    IJth(jmatrix, 99, 99) = 0.0 - k[350]*y[IDX_EM] - k[1242];
    IJth(jmatrix, 99, 100) = 0.0 + k[700]*y[IDX_HeII];
    IJth(jmatrix, 100, 21) = 0.0 + k[1399] + k[1400] + k[1401] + k[1402];
    IJth(jmatrix, 100, 25) = 0.0 - k[23]*y[IDX_SiC3I];
    IJth(jmatrix, 100, 42) = 0.0 - k[93]*y[IDX_SiC3I];
    IJth(jmatrix, 100, 62) = 0.0 - k[700]*y[IDX_SiC3I];
    IJth(jmatrix, 100, 85) = 0.0 - k[1082]*y[IDX_SiC3I];
    IJth(jmatrix, 100, 100) = 0.0 - k[23]*y[IDX_CII] - k[93]*y[IDX_HII] -
        k[287] - k[700]*y[IDX_HeII] - k[1082]*y[IDX_OI] - k[1177] - k[1243];
    IJth(jmatrix, 101, 25) = 0.0 + k[23]*y[IDX_SiC3I];
    IJth(jmatrix, 101, 40) = 0.0 - k[351]*y[IDX_SiC3II];
    IJth(jmatrix, 101, 42) = 0.0 + k[93]*y[IDX_SiC3I];
    IJth(jmatrix, 101, 100) = 0.0 + k[23]*y[IDX_CII] + k[93]*y[IDX_HII];
    IJth(jmatrix, 101, 101) = 0.0 - k[351]*y[IDX_EM] - k[1244];
    IJth(jmatrix, 102, 24) = 0.0 - k[877]*y[IDX_SiHI];
    IJth(jmatrix, 102, 25) = 0.0 - k[381]*y[IDX_SiHI];
    IJth(jmatrix, 102, 40) = 0.0 + k[355]*y[IDX_SiH2II] +
        k[357]*y[IDX_SiH3II];
    IJth(jmatrix, 102, 42) = 0.0 - k[98]*y[IDX_SiHI] - k[508]*y[IDX_SiHI];
    IJth(jmatrix, 102, 52) = 0.0 - k[605]*y[IDX_SiHI];
    IJth(jmatrix, 102, 54) = 0.0 - k[612]*y[IDX_SiHI];
    IJth(jmatrix, 102, 59) = 0.0 - k[639]*y[IDX_SiHI];
    IJth(jmatrix, 102, 62) = 0.0 - k[709]*y[IDX_SiHI];
    IJth(jmatrix, 102, 85) = 0.0 - k[1089]*y[IDX_SiHI];
    IJth(jmatrix, 102, 93) = 0.0 - k[849]*y[IDX_SiHI];
    IJth(jmatrix, 102, 102) = 0.0 - k[98]*y[IDX_HII] - k[292] -
        k[381]*y[IDX_CII] - k[508]*y[IDX_HII] - k[605]*y[IDX_H3II] -
        k[612]*y[IDX_H3OII] - k[639]*y[IDX_HCOII] - k[709]*y[IDX_HeII] -
        k[849]*y[IDX_OHII] - k[877]*y[IDX_CI] - k[1089]*y[IDX_OI] - k[1188] -
        k[1230];
    IJth(jmatrix, 102, 104) = 0.0 + k[289] + k[1181];
    IJth(jmatrix, 102, 105) = 0.0 + k[355]*y[IDX_EM];
    IJth(jmatrix, 102, 106) = 0.0 + k[1184];
    IJth(jmatrix, 102, 107) = 0.0 + k[357]*y[IDX_EM];
    IJth(jmatrix, 102, 108) = 0.0 + k[1187];
    IJth(jmatrix, 103, 24) = 0.0 - k[394]*y[IDX_SiHII];
    IJth(jmatrix, 103, 26) = 0.0 - k[477]*y[IDX_SiHII];
    IJth(jmatrix, 103, 40) = 0.0 - k[352]*y[IDX_SiHII];
    IJth(jmatrix, 103, 41) = 0.0 - k[619]*y[IDX_SiHII] +
        k[1209]*y[IDX_SiII];
    IJth(jmatrix, 103, 42) = 0.0 + k[98]*y[IDX_SiHI] + k[505]*y[IDX_SiH2I];
    IJth(jmatrix, 103, 43) = 0.0 - k[1203]*y[IDX_SiHII];
    IJth(jmatrix, 103, 49) = 0.0 - k[573]*y[IDX_SiHII];
    IJth(jmatrix, 103, 52) = 0.0 + k[601]*y[IDX_SiI];
    IJth(jmatrix, 103, 54) = 0.0 + k[610]*y[IDX_SiI];
    IJth(jmatrix, 103, 59) = 0.0 + k[861]*y[IDX_SiI];
    IJth(jmatrix, 103, 62) = 0.0 + k[704]*y[IDX_SiH2I] + k[705]*y[IDX_SiH3I]
        + k[708]*y[IDX_SiH4I];
    IJth(jmatrix, 103, 85) = 0.0 - k[832]*y[IDX_SiHII];
    IJth(jmatrix, 103, 93) = 0.0 + k[848]*y[IDX_SiI];
    IJth(jmatrix, 103, 94) = 0.0 + k[601]*y[IDX_H3II] + k[610]*y[IDX_H3OII]
        + k[848]*y[IDX_OHII] + k[861]*y[IDX_HCOII];
    IJth(jmatrix, 103, 95) = 0.0 + k[1209]*y[IDX_HI];
    IJth(jmatrix, 103, 102) = 0.0 + k[98]*y[IDX_HII];
    IJth(jmatrix, 103, 103) = 0.0 - k[352]*y[IDX_EM] - k[394]*y[IDX_CI] -
        k[477]*y[IDX_CHI] - k[573]*y[IDX_H2OI] - k[619]*y[IDX_HI] -
        k[832]*y[IDX_OI] - k[1179] - k[1203]*y[IDX_H2I] - k[1232];
    IJth(jmatrix, 103, 104) = 0.0 + k[505]*y[IDX_HII] + k[704]*y[IDX_HeII];
    IJth(jmatrix, 103, 106) = 0.0 + k[705]*y[IDX_HeII];
    IJth(jmatrix, 103, 108) = 0.0 + k[708]*y[IDX_HeII];
    IJth(jmatrix, 104, 25) = 0.0 - k[25]*y[IDX_SiH2I] - k[380]*y[IDX_SiH2I];
    IJth(jmatrix, 104, 40) = 0.0 + k[356]*y[IDX_SiH3II] +
        k[358]*y[IDX_SiH4II];
    IJth(jmatrix, 104, 42) = 0.0 - k[95]*y[IDX_SiH2I] - k[505]*y[IDX_SiH2I];
    IJth(jmatrix, 104, 52) = 0.0 - k[602]*y[IDX_SiH2I];
    IJth(jmatrix, 104, 54) = 0.0 - k[611]*y[IDX_SiH2I];
    IJth(jmatrix, 104, 59) = 0.0 - k[637]*y[IDX_SiH2I];
    IJth(jmatrix, 104, 62) = 0.0 - k[703]*y[IDX_SiH2I] -
        k[704]*y[IDX_SiH2I];
    IJth(jmatrix, 104, 85) = 0.0 - k[1085]*y[IDX_SiH2I] -
        k[1086]*y[IDX_SiH2I];
    IJth(jmatrix, 104, 104) = 0.0 - k[25]*y[IDX_CII] - k[95]*y[IDX_HII] -
        k[289] - k[380]*y[IDX_CII] - k[505]*y[IDX_HII] - k[602]*y[IDX_H3II] -
        k[611]*y[IDX_H3OII] - k[637]*y[IDX_HCOII] - k[703]*y[IDX_HeII] -
        k[704]*y[IDX_HeII] - k[1085]*y[IDX_OI] - k[1086]*y[IDX_OI] - k[1180] -
        k[1181] - k[1233];
    IJth(jmatrix, 104, 106) = 0.0 + k[290] + k[1182];
    IJth(jmatrix, 104, 107) = 0.0 + k[356]*y[IDX_EM];
    IJth(jmatrix, 104, 108) = 0.0 + k[291] + k[1185];
    IJth(jmatrix, 104, 109) = 0.0 + k[358]*y[IDX_EM];
    IJth(jmatrix, 105, 25) = 0.0 + k[25]*y[IDX_SiH2I];
    IJth(jmatrix, 105, 40) = 0.0 - k[353]*y[IDX_SiH2II] -
        k[354]*y[IDX_SiH2II] - k[355]*y[IDX_SiH2II];
    IJth(jmatrix, 105, 42) = 0.0 + k[95]*y[IDX_SiH2I] + k[506]*y[IDX_SiH3I];
    IJth(jmatrix, 105, 43) = 0.0 + k[1202]*y[IDX_SiII];
    IJth(jmatrix, 105, 52) = 0.0 + k[605]*y[IDX_SiHI];
    IJth(jmatrix, 105, 54) = 0.0 + k[612]*y[IDX_SiHI];
    IJth(jmatrix, 105, 59) = 0.0 + k[639]*y[IDX_SiHI];
    IJth(jmatrix, 105, 62) = 0.0 + k[706]*y[IDX_SiH3I];
    IJth(jmatrix, 105, 85) = 0.0 - k[833]*y[IDX_SiH2II];
    IJth(jmatrix, 105, 87) = 0.0 - k[862]*y[IDX_SiH2II];
    IJth(jmatrix, 105, 93) = 0.0 + k[849]*y[IDX_SiHI];
    IJth(jmatrix, 105, 95) = 0.0 + k[1202]*y[IDX_H2I];
    IJth(jmatrix, 105, 102) = 0.0 + k[605]*y[IDX_H3II] + k[612]*y[IDX_H3OII]
        + k[639]*y[IDX_HCOII] + k[849]*y[IDX_OHII];
    IJth(jmatrix, 105, 104) = 0.0 + k[25]*y[IDX_CII] + k[95]*y[IDX_HII] +
        k[1180];
    IJth(jmatrix, 105, 105) = 0.0 - k[353]*y[IDX_EM] - k[354]*y[IDX_EM] -
        k[355]*y[IDX_EM] - k[833]*y[IDX_OI] - k[862]*y[IDX_O2I] - k[1234];
    IJth(jmatrix, 105, 106) = 0.0 + k[506]*y[IDX_HII] + k[706]*y[IDX_HeII];
    IJth(jmatrix, 106, 25) = 0.0 - k[26]*y[IDX_SiH3I];
    IJth(jmatrix, 106, 35) = 0.0 + k[950]*y[IDX_SiH4I];
    IJth(jmatrix, 106, 37) = 0.0 + k[489]*y[IDX_SiH4II];
    IJth(jmatrix, 106, 40) = 0.0 + k[359]*y[IDX_SiH4II] +
        k[360]*y[IDX_SiH5II];
    IJth(jmatrix, 106, 42) = 0.0 - k[96]*y[IDX_SiH3I] - k[506]*y[IDX_SiH3I];
    IJth(jmatrix, 106, 49) = 0.0 + k[574]*y[IDX_SiH4II];
    IJth(jmatrix, 106, 52) = 0.0 - k[603]*y[IDX_SiH3I];
    IJth(jmatrix, 106, 62) = 0.0 - k[705]*y[IDX_SiH3I] -
        k[706]*y[IDX_SiH3I];
    IJth(jmatrix, 106, 85) = 0.0 - k[1087]*y[IDX_SiH3I] +
        k[1088]*y[IDX_SiH4I];
    IJth(jmatrix, 106, 106) = 0.0 - k[26]*y[IDX_CII] - k[96]*y[IDX_HII] -
        k[290] - k[506]*y[IDX_HII] - k[603]*y[IDX_H3II] - k[705]*y[IDX_HeII] -
        k[706]*y[IDX_HeII] - k[1087]*y[IDX_OI] - k[1182] - k[1183] - k[1184] -
        k[1235];
    IJth(jmatrix, 106, 108) = 0.0 + k[950]*y[IDX_CNI] + k[1088]*y[IDX_OI] +
        k[1186];
    IJth(jmatrix, 106, 109) = 0.0 + k[359]*y[IDX_EM] + k[489]*y[IDX_COI] +
        k[574]*y[IDX_H2OI];
    IJth(jmatrix, 106, 110) = 0.0 + k[360]*y[IDX_EM];
    IJth(jmatrix, 107, 25) = 0.0 + k[26]*y[IDX_SiH3I];
    IJth(jmatrix, 107, 31) = 0.0 + k[446]*y[IDX_SiH4I];
    IJth(jmatrix, 107, 40) = 0.0 - k[356]*y[IDX_SiH3II] -
        k[357]*y[IDX_SiH3II];
    IJth(jmatrix, 107, 42) = 0.0 + k[96]*y[IDX_SiH3I] + k[507]*y[IDX_SiH4I];
    IJth(jmatrix, 107, 43) = 0.0 + k[1203]*y[IDX_SiHII] -
        k[1204]*y[IDX_SiH3II];
    IJth(jmatrix, 107, 52) = 0.0 + k[602]*y[IDX_SiH2I];
    IJth(jmatrix, 107, 54) = 0.0 + k[611]*y[IDX_SiH2I];
    IJth(jmatrix, 107, 59) = 0.0 + k[637]*y[IDX_SiH2I];
    IJth(jmatrix, 107, 85) = 0.0 - k[834]*y[IDX_SiH3II];
    IJth(jmatrix, 107, 103) = 0.0 + k[1203]*y[IDX_H2I];
    IJth(jmatrix, 107, 104) = 0.0 + k[602]*y[IDX_H3II] + k[611]*y[IDX_H3OII]
        + k[637]*y[IDX_HCOII];
    IJth(jmatrix, 107, 106) = 0.0 + k[26]*y[IDX_CII] + k[96]*y[IDX_HII] +
        k[1183];
    IJth(jmatrix, 107, 107) = 0.0 - k[356]*y[IDX_EM] - k[357]*y[IDX_EM] -
        k[834]*y[IDX_OI] - k[1204]*y[IDX_H2I] - k[1236];
    IJth(jmatrix, 107, 108) = 0.0 + k[446]*y[IDX_CH3II] + k[507]*y[IDX_HII];
    IJth(jmatrix, 108, 22) = 0.0 + k[1363] + k[1364] + k[1365] + k[1366];
    IJth(jmatrix, 108, 31) = 0.0 - k[446]*y[IDX_SiH4I];
    IJth(jmatrix, 108, 35) = 0.0 - k[950]*y[IDX_SiH4I];
    IJth(jmatrix, 108, 40) = 0.0 + k[361]*y[IDX_SiH5II];
    IJth(jmatrix, 108, 42) = 0.0 - k[97]*y[IDX_SiH4I] - k[507]*y[IDX_SiH4I];
    IJth(jmatrix, 108, 49) = 0.0 + k[575]*y[IDX_SiH5II];
    IJth(jmatrix, 108, 52) = 0.0 - k[604]*y[IDX_SiH4I];
    IJth(jmatrix, 108, 59) = 0.0 - k[638]*y[IDX_SiH4I];
    IJth(jmatrix, 108, 62) = 0.0 - k[707]*y[IDX_SiH4I] -
        k[708]*y[IDX_SiH4I];
    IJth(jmatrix, 108, 85) = 0.0 - k[1088]*y[IDX_SiH4I];
    IJth(jmatrix, 108, 108) = 0.0 - k[97]*y[IDX_HII] - k[291] -
        k[446]*y[IDX_CH3II] - k[507]*y[IDX_HII] - k[604]*y[IDX_H3II] -
        k[638]*y[IDX_HCOII] - k[707]*y[IDX_HeII] - k[708]*y[IDX_HeII] -
        k[950]*y[IDX_CNI] - k[1088]*y[IDX_OI] - k[1185] - k[1186] - k[1187] -
        k[1237];
    IJth(jmatrix, 108, 110) = 0.0 + k[361]*y[IDX_EM] + k[575]*y[IDX_H2OI];
    IJth(jmatrix, 109, 37) = 0.0 - k[489]*y[IDX_SiH4II];
    IJth(jmatrix, 109, 40) = 0.0 - k[358]*y[IDX_SiH4II] -
        k[359]*y[IDX_SiH4II];
    IJth(jmatrix, 109, 42) = 0.0 + k[97]*y[IDX_SiH4I];
    IJth(jmatrix, 109, 43) = 0.0 - k[546]*y[IDX_SiH4II];
    IJth(jmatrix, 109, 49) = 0.0 - k[574]*y[IDX_SiH4II];
    IJth(jmatrix, 109, 52) = 0.0 + k[603]*y[IDX_SiH3I];
    IJth(jmatrix, 109, 106) = 0.0 + k[603]*y[IDX_H3II];
    IJth(jmatrix, 109, 108) = 0.0 + k[97]*y[IDX_HII];
    IJth(jmatrix, 109, 109) = 0.0 - k[358]*y[IDX_EM] - k[359]*y[IDX_EM] -
        k[489]*y[IDX_COI] - k[546]*y[IDX_H2I] - k[574]*y[IDX_H2OI] - k[1238];
    IJth(jmatrix, 110, 40) = 0.0 - k[360]*y[IDX_SiH5II] -
        k[361]*y[IDX_SiH5II];
    IJth(jmatrix, 110, 43) = 0.0 + k[546]*y[IDX_SiH4II] +
        k[1204]*y[IDX_SiH3II];
    IJth(jmatrix, 110, 49) = 0.0 - k[575]*y[IDX_SiH5II];
    IJth(jmatrix, 110, 52) = 0.0 + k[604]*y[IDX_SiH4I];
    IJth(jmatrix, 110, 59) = 0.0 + k[638]*y[IDX_SiH4I];
    IJth(jmatrix, 110, 107) = 0.0 + k[1204]*y[IDX_H2I];
    IJth(jmatrix, 110, 108) = 0.0 + k[604]*y[IDX_H3II] +
        k[638]*y[IDX_HCOII];
    IJth(jmatrix, 110, 109) = 0.0 + k[546]*y[IDX_H2I];
    IJth(jmatrix, 110, 110) = 0.0 - k[360]*y[IDX_EM] - k[361]*y[IDX_EM] -
        k[575]*y[IDX_H2OI] - k[1248];
    IJth(jmatrix, 111, 23) = 0.0 + k[1379] + k[1380] + k[1381] + k[1382];
    IJth(jmatrix, 111, 25) = 0.0 - k[382]*y[IDX_SiOI];
    IJth(jmatrix, 111, 37) = 0.0 + k[1104]*y[IDX_SiI];
    IJth(jmatrix, 111, 39) = 0.0 + k[1103]*y[IDX_SiI];
    IJth(jmatrix, 111, 40) = 0.0 + k[364]*y[IDX_SiOHII];
    IJth(jmatrix, 111, 42) = 0.0 - k[99]*y[IDX_SiOI];
    IJth(jmatrix, 111, 51) = 0.0 + k[257] + k[1143] + k[1144];
    IJth(jmatrix, 111, 52) = 0.0 - k[606]*y[IDX_SiOI];
    IJth(jmatrix, 111, 54) = 0.0 - k[613]*y[IDX_SiOI];
    IJth(jmatrix, 111, 58) = 0.0 + k[138]*y[IDX_SiOII];
    IJth(jmatrix, 111, 59) = 0.0 - k[640]*y[IDX_SiOI];
    IJth(jmatrix, 111, 62) = 0.0 - k[710]*y[IDX_SiOI] - k[711]*y[IDX_SiOI];
    IJth(jmatrix, 111, 69) = 0.0 + k[154]*y[IDX_SiOII];
    IJth(jmatrix, 111, 82) = 0.0 + k[206]*y[IDX_SiOII] + k[1105]*y[IDX_SiI];
    IJth(jmatrix, 111, 85) = 0.0 + k[1084]*y[IDX_SiCI] +
        k[1085]*y[IDX_SiH2I] + k[1086]*y[IDX_SiH2I] + k[1089]*y[IDX_SiHI] +
        k[1213]*y[IDX_SiI];
    IJth(jmatrix, 111, 87) = 0.0 + k[1106]*y[IDX_SiI];
    IJth(jmatrix, 111, 92) = 0.0 + k[1102]*y[IDX_SiI];
    IJth(jmatrix, 111, 93) = 0.0 - k[850]*y[IDX_SiOI];
    IJth(jmatrix, 111, 94) = 0.0 + k[1102]*y[IDX_OHI] + k[1103]*y[IDX_CO2I]
        + k[1104]*y[IDX_COI] + k[1105]*y[IDX_NOI] + k[1106]*y[IDX_O2I] +
        k[1213]*y[IDX_OI];
    IJth(jmatrix, 111, 96) = 0.0 + k[1084]*y[IDX_OI];
    IJth(jmatrix, 111, 102) = 0.0 + k[1089]*y[IDX_OI];
    IJth(jmatrix, 111, 104) = 0.0 + k[1085]*y[IDX_OI] + k[1086]*y[IDX_OI];
    IJth(jmatrix, 111, 111) = 0.0 - k[99]*y[IDX_HII] - k[293] -
        k[382]*y[IDX_CII] - k[606]*y[IDX_H3II] - k[613]*y[IDX_H3OII] -
        k[640]*y[IDX_HCOII] - k[710]*y[IDX_HeII] - k[711]*y[IDX_HeII] -
        k[850]*y[IDX_OHII] - k[1190] - k[1191] - k[1229];
    IJth(jmatrix, 111, 112) = 0.0 + k[138]*y[IDX_HCOI] + k[154]*y[IDX_MgI] +
        k[206]*y[IDX_NOI];
    IJth(jmatrix, 111, 113) = 0.0 + k[364]*y[IDX_EM];
    IJth(jmatrix, 112, 24) = 0.0 - k[395]*y[IDX_SiOII];
    IJth(jmatrix, 112, 26) = 0.0 - k[478]*y[IDX_SiOII];
    IJth(jmatrix, 112, 28) = 0.0 - k[438]*y[IDX_SiOII];
    IJth(jmatrix, 112, 37) = 0.0 - k[490]*y[IDX_SiOII];
    IJth(jmatrix, 112, 40) = 0.0 - k[362]*y[IDX_SiOII];
    IJth(jmatrix, 112, 42) = 0.0 + k[99]*y[IDX_SiOI];
    IJth(jmatrix, 112, 43) = 0.0 - k[547]*y[IDX_SiOII];
    IJth(jmatrix, 112, 58) = 0.0 - k[138]*y[IDX_SiOII];
    IJth(jmatrix, 112, 69) = 0.0 - k[154]*y[IDX_SiOII];
    IJth(jmatrix, 112, 71) = 0.0 - k[745]*y[IDX_SiOII] -
        k[746]*y[IDX_SiOII];
    IJth(jmatrix, 112, 82) = 0.0 - k[206]*y[IDX_SiOII];
    IJth(jmatrix, 112, 85) = 0.0 + k[831]*y[IDX_SiCII] + k[832]*y[IDX_SiHII]
        - k[835]*y[IDX_SiOII] + k[1212]*y[IDX_SiII];
    IJth(jmatrix, 112, 92) = 0.0 + k[859]*y[IDX_SiII];
    IJth(jmatrix, 112, 95) = 0.0 + k[859]*y[IDX_OHI] + k[1212]*y[IDX_OI];
    IJth(jmatrix, 112, 97) = 0.0 + k[831]*y[IDX_OI];
    IJth(jmatrix, 112, 103) = 0.0 + k[832]*y[IDX_OI];
    IJth(jmatrix, 112, 111) = 0.0 + k[99]*y[IDX_HII] + k[1191];
    IJth(jmatrix, 112, 112) = 0.0 - k[138]*y[IDX_HCOI] - k[154]*y[IDX_MgI] -
        k[206]*y[IDX_NOI] - k[362]*y[IDX_EM] - k[395]*y[IDX_CI] -
        k[438]*y[IDX_CH2I] - k[478]*y[IDX_CHI] - k[490]*y[IDX_COI] -
        k[547]*y[IDX_H2I] - k[745]*y[IDX_NI] - k[746]*y[IDX_NI] -
        k[835]*y[IDX_OI] - k[1189] - k[1245];
    IJth(jmatrix, 113, 32) = 0.0 + k[860]*y[IDX_SiII];
    IJth(jmatrix, 113, 40) = 0.0 - k[363]*y[IDX_SiOHII] -
        k[364]*y[IDX_SiOHII];
    IJth(jmatrix, 113, 42) = 0.0 + k[499]*y[IDX_H2SiOI];
    IJth(jmatrix, 113, 43) = 0.0 + k[547]*y[IDX_SiOII];
    IJth(jmatrix, 113, 49) = 0.0 + k[572]*y[IDX_SiII];
    IJth(jmatrix, 113, 51) = 0.0 + k[499]*y[IDX_HII] + k[675]*y[IDX_HeII];
    IJth(jmatrix, 113, 52) = 0.0 + k[606]*y[IDX_SiOI];
    IJth(jmatrix, 113, 54) = 0.0 + k[613]*y[IDX_SiOI];
    IJth(jmatrix, 113, 59) = 0.0 + k[640]*y[IDX_SiOI];
    IJth(jmatrix, 113, 62) = 0.0 + k[675]*y[IDX_H2SiOI];
    IJth(jmatrix, 113, 85) = 0.0 + k[833]*y[IDX_SiH2II] +
        k[834]*y[IDX_SiH3II];
    IJth(jmatrix, 113, 87) = 0.0 + k[862]*y[IDX_SiH2II];
    IJth(jmatrix, 113, 93) = 0.0 + k[850]*y[IDX_SiOI];
    IJth(jmatrix, 113, 95) = 0.0 + k[572]*y[IDX_H2OI] +
        k[860]*y[IDX_CH3OHI];
    IJth(jmatrix, 113, 105) = 0.0 + k[833]*y[IDX_OI] + k[862]*y[IDX_O2I];
    IJth(jmatrix, 113, 107) = 0.0 + k[834]*y[IDX_OI];
    IJth(jmatrix, 113, 111) = 0.0 + k[606]*y[IDX_H3II] + k[613]*y[IDX_H3OII]
        + k[640]*y[IDX_HCOII] + k[850]*y[IDX_OHII];
    IJth(jmatrix, 113, 112) = 0.0 + k[547]*y[IDX_H2I];
    IJth(jmatrix, 113, 113) = 0.0 - k[363]*y[IDX_EM] - k[364]*y[IDX_EM] -
        k[1246];
      // clang-format on

    /* */

    return NAUNET_SUCCESS;
}