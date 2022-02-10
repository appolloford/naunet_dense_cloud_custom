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

int Fex(realtype t, N_Vector u, N_Vector udot, void *user_data) {
    /* */
    realtype *y            = N_VGetArrayPointer(u);
    realtype *ydot         = N_VGetArrayPointer(udot);
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
    
    double mant = GetMantleDens(y);
    double mantabund = mant / nH;
    double garea = (pi*rG*rG) * gdens;
    double garea_per_H = garea / nH;
    double densites = 4.0 * garea * sites;
    double h2col = 0.5*1.59e21*Av;
    double cocol = 1e-5 * h2col;
    double lamdabar = GetCharactWavelength(h2col, cocol);
    double H2shielding = GetShieldingFactor(IDX_H2I, h2col, h2col, Tgas, 1);
    double H2formation = 1.0e-17 * sqrt(Tgas);
    double H2dissociation = 5.1e-11 * G0 * GetGrainScattering(Av, 1000.0) * H2shielding;
    
    
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

    // clang-format off
    ydot[IDX_GCH3OHI] = 0.0 + k[1223]*y[IDX_CH3OHI] + k[1249]*y[IDX_H3COII]
        + k[1304]*y[IDX_COI] - k[1359]*y[IDX_GCH3OHI] - k[1360]*y[IDX_GCH3OHI] -
        k[1361]*y[IDX_GCH3OHI] - k[1362]*y[IDX_GCH3OHI];
    ydot[IDX_GCH4I] = 0.0 + k[1250]*y[IDX_CI] + k[1253]*y[IDX_CHI] +
        k[1256]*y[IDX_CH2I] + k[1259]*y[IDX_CH3I] + k[1260]*y[IDX_CH4I] +
        k[1264]*y[IDX_CII] + k[1272]*y[IDX_CHII] + k[1278]*y[IDX_CH2II] +
        k[1285]*y[IDX_CH3II] + k[1288]*y[IDX_CH4II] - k[1307]*y[IDX_GCH4I] -
        k[1308]*y[IDX_GCH4I] - k[1309]*y[IDX_GCH4I] - k[1310]*y[IDX_GCH4I];
    ydot[IDX_GCOI] = 0.0 + k[1251]*y[IDX_COI] + k[1275]*y[IDX_COII] -
        k[1331]*y[IDX_GCOI] - k[1332]*y[IDX_GCOI] - k[1333]*y[IDX_GCOI] -
        k[1334]*y[IDX_GCOI];
    ydot[IDX_GCO2I] = 0.0 + k[1258]*y[IDX_CO2I] + k[1287]*y[IDX_HCO2II] -
        k[1383]*y[IDX_GCO2I] - k[1384]*y[IDX_GCO2I] - k[1385]*y[IDX_GCO2I] -
        k[1386]*y[IDX_GCO2I];
    ydot[IDX_GH2CNI] = 0.0 + k[1298]*y[IDX_H2CNI] - k[1339]*y[IDX_GH2CNI] -
        k[1340]*y[IDX_GH2CNI] - k[1341]*y[IDX_GH2CNI] - k[1342]*y[IDX_GH2CNI];
    ydot[IDX_GH2COI] = 0.0 + k[1226]*y[IDX_HOCII] + k[1227]*y[IDX_COI] +
        k[1252]*y[IDX_H2COI] + k[1261]*y[IDX_HCOI] + k[1281]*y[IDX_HCOII] +
        k[1284]*y[IDX_H2COII] - k[1347]*y[IDX_GH2COI] - k[1348]*y[IDX_GH2COI] -
        k[1349]*y[IDX_GH2COI] - k[1350]*y[IDX_GH2COI];
    ydot[IDX_GH2OI] = 0.0 + k[1254]*y[IDX_OHI] + k[1257]*y[IDX_H2OI] +
        k[1269]*y[IDX_OII] + k[1274]*y[IDX_OHII] + k[1280]*y[IDX_H2OII] +
        k[1286]*y[IDX_H3OII] + k[1291]*y[IDX_OI] - k[1315]*y[IDX_GH2OI] -
        k[1316]*y[IDX_GH2OI] - k[1317]*y[IDX_GH2OI] - k[1318]*y[IDX_GH2OI];
    ydot[IDX_GH2SiOI] = 0.0 + k[1229]*y[IDX_SiOI] + k[1245]*y[IDX_SiOII] +
        k[1246]*y[IDX_SiOHII] + k[1247]*y[IDX_H2SiOI] - k[1391]*y[IDX_GH2SiOI] -
        k[1392]*y[IDX_GH2SiOI] - k[1393]*y[IDX_GH2SiOI] -
        k[1394]*y[IDX_GH2SiOI];
    ydot[IDX_GHCNI] = 0.0 + k[1263]*y[IDX_CNI] + k[1266]*y[IDX_HCNI] +
        k[1276]*y[IDX_CNII] + k[1282]*y[IDX_HCNII] + k[1301]*y[IDX_HCNHII] -
        k[1323]*y[IDX_GHCNI] - k[1324]*y[IDX_GHCNI] - k[1325]*y[IDX_GHCNI] -
        k[1326]*y[IDX_GHCNI];
    ydot[IDX_GHNCI] = 0.0 + k[1305]*y[IDX_HNCI] - k[1327]*y[IDX_GHNCI] -
        k[1328]*y[IDX_GHNCI] - k[1329]*y[IDX_GHNCI] - k[1330]*y[IDX_GHNCI];
    ydot[IDX_GHNCOI] = 0.0 + k[1224]*y[IDX_HNCOI] + k[1225]*y[IDX_OCNI] -
        k[1375]*y[IDX_GHNCOI] - k[1376]*y[IDX_GHNCOI] - k[1377]*y[IDX_GHNCOI] -
        k[1378]*y[IDX_GHNCOI];
    ydot[IDX_GHNOI] = 0.0 + k[1294]*y[IDX_HNOI] + k[1295]*y[IDX_HNOII] +
        k[1296]*y[IDX_H2NOII] - k[1351]*y[IDX_GHNOI] - k[1352]*y[IDX_GHNOI] -
        k[1353]*y[IDX_GHNOI] - k[1354]*y[IDX_GHNOI];
    ydot[IDX_GMgI] = 0.0 + k[1299]*y[IDX_MgII] + k[1300]*y[IDX_MgI] -
        k[1319]*y[IDX_GMgI] - k[1320]*y[IDX_GMgI] - k[1321]*y[IDX_GMgI] -
        k[1322]*y[IDX_GMgI];
    ydot[IDX_GN2I] = 0.0 + k[1262]*y[IDX_N2I] + k[1271]*y[IDX_N2II] +
        k[1302]*y[IDX_N2HII] - k[1335]*y[IDX_GN2I] - k[1336]*y[IDX_GN2I] -
        k[1337]*y[IDX_GN2I] - k[1338]*y[IDX_GN2I];
    ydot[IDX_GNH3I] = 0.0 + k[1265]*y[IDX_NHI] + k[1267]*y[IDX_NH3I] +
        k[1268]*y[IDX_NII] + k[1273]*y[IDX_NHII] + k[1279]*y[IDX_NH2II] +
        k[1283]*y[IDX_NH3II] + k[1289]*y[IDX_NH2I] + k[1290]*y[IDX_NI] -
        k[1311]*y[IDX_GNH3I] - k[1312]*y[IDX_GNH3I] - k[1313]*y[IDX_GNH3I] -
        k[1314]*y[IDX_GNH3I];
    ydot[IDX_GNOI] = 0.0 + k[1255]*y[IDX_NOI] + k[1277]*y[IDX_NOII] -
        k[1343]*y[IDX_GNOI] - k[1344]*y[IDX_GNOI] - k[1345]*y[IDX_GNOI] -
        k[1346]*y[IDX_GNOI];
    ydot[IDX_GNO2I] = 0.0 + k[1293]*y[IDX_NO2I] - k[1387]*y[IDX_GNO2I] -
        k[1388]*y[IDX_GNO2I] - k[1389]*y[IDX_GNO2I] - k[1390]*y[IDX_GNO2I];
    ydot[IDX_GO2I] = 0.0 + k[1270]*y[IDX_O2II] + k[1292]*y[IDX_O2I] -
        k[1355]*y[IDX_GO2I] - k[1356]*y[IDX_GO2I] - k[1357]*y[IDX_GO2I] -
        k[1358]*y[IDX_GO2I];
    ydot[IDX_GO2HI] = 0.0 + k[1297]*y[IDX_O2HII] + k[1303]*y[IDX_O2HI] -
        k[1367]*y[IDX_GO2HI] - k[1368]*y[IDX_GO2HI] - k[1369]*y[IDX_GO2HI] -
        k[1370]*y[IDX_GO2HI];
    ydot[IDX_GSiCI] = 0.0 + k[1239]*y[IDX_SiCI] + k[1241]*y[IDX_SiCII] -
        k[1371]*y[IDX_GSiCI] - k[1372]*y[IDX_GSiCI] - k[1373]*y[IDX_GSiCI] -
        k[1374]*y[IDX_GSiCI];
    ydot[IDX_GSiC2I] = 0.0 + k[1240]*y[IDX_SiC2I] + k[1242]*y[IDX_SiC2II] -
        k[1395]*y[IDX_GSiC2I] - k[1396]*y[IDX_GSiC2I] - k[1397]*y[IDX_GSiC2I] -
        k[1398]*y[IDX_GSiC2I];
    ydot[IDX_GSiC3I] = 0.0 + k[1243]*y[IDX_SiC3I] + k[1244]*y[IDX_SiC3II] -
        k[1399]*y[IDX_GSiC3I] - k[1400]*y[IDX_GSiC3I] - k[1401]*y[IDX_GSiC3I] -
        k[1402]*y[IDX_GSiC3I];
    ydot[IDX_GSiH4I] = 0.0 + k[1228]*y[IDX_SiI] + k[1230]*y[IDX_SiHI] +
        k[1231]*y[IDX_SiII] + k[1232]*y[IDX_SiHII] + k[1233]*y[IDX_SiH2I] +
        k[1234]*y[IDX_SiH2II] + k[1235]*y[IDX_SiH3I] + k[1236]*y[IDX_SiH3II] +
        k[1237]*y[IDX_SiH4I] + k[1238]*y[IDX_SiH4II] + k[1248]*y[IDX_SiH5II] -
        k[1363]*y[IDX_GSiH4I] - k[1364]*y[IDX_GSiH4I] - k[1365]*y[IDX_GSiH4I] -
        k[1366]*y[IDX_GSiH4I];
    ydot[IDX_GSiOI] = 0.0 - k[1379]*y[IDX_GSiOI] - k[1380]*y[IDX_GSiOI] -
        k[1381]*y[IDX_GSiOI] - k[1382]*y[IDX_GSiOI];
    ydot[IDX_CI] = 0.0 + k[2]*y[IDX_H2I]*y[IDX_CHI] +
        k[9]*y[IDX_HI]*y[IDX_CHI] + k[14]*y[IDX_CII]*y[IDX_CH2I] +
        k[15]*y[IDX_CII]*y[IDX_CHI] + k[16]*y[IDX_CII]*y[IDX_H2COI] +
        k[17]*y[IDX_CII]*y[IDX_HCOI] + k[18]*y[IDX_CII]*y[IDX_MgI] +
        k[19]*y[IDX_CII]*y[IDX_NH3I] + k[20]*y[IDX_CII]*y[IDX_NOI] +
        k[21]*y[IDX_CII]*y[IDX_SiI] + k[22]*y[IDX_CII]*y[IDX_SiC2I] +
        k[23]*y[IDX_CII]*y[IDX_SiC3I] + k[24]*y[IDX_CII]*y[IDX_SiCI] +
        k[25]*y[IDX_CII]*y[IDX_SiH2I] + k[26]*y[IDX_CII]*y[IDX_SiH3I] -
        k[27]*y[IDX_CI]*y[IDX_CNII] - k[28]*y[IDX_CI]*y[IDX_COII] -
        k[29]*y[IDX_CI]*y[IDX_N2II] - k[30]*y[IDX_CI]*y[IDX_O2II] -
        k[139]*y[IDX_HeII]*y[IDX_CI] - k[231]*y[IDX_CI] - k[240]*y[IDX_CI] +
        k[250]*y[IDX_CHI] + k[251]*y[IDX_CNI] + k[253]*y[IDX_COI] +
        k[286]*y[IDX_SiC2I] + k[287]*y[IDX_SiC3I] + k[288]*y[IDX_SiCI] +
        k[294]*y[IDX_CHII]*y[IDX_EM] + k[295]*y[IDX_CH2II]*y[IDX_EM] +
        k[296]*y[IDX_CH2II]*y[IDX_EM] + k[303]*y[IDX_CNII]*y[IDX_EM] +
        k[304]*y[IDX_COII]*y[IDX_EM] + k[349]*y[IDX_SiCII]*y[IDX_EM] +
        k[350]*y[IDX_SiC2II]*y[IDX_EM] + k[351]*y[IDX_SiC3II]*y[IDX_EM] -
        k[383]*y[IDX_CI]*y[IDX_H2OII] - k[384]*y[IDX_CI]*y[IDX_H3OII] -
        k[385]*y[IDX_CI]*y[IDX_HCNII] - k[386]*y[IDX_CI]*y[IDX_HCOII] -
        k[387]*y[IDX_CI]*y[IDX_HCO2II] - k[388]*y[IDX_CI]*y[IDX_HNOII] -
        k[389]*y[IDX_CI]*y[IDX_N2HII] - k[390]*y[IDX_CI]*y[IDX_NHII] -
        k[391]*y[IDX_CI]*y[IDX_O2II] - k[392]*y[IDX_CI]*y[IDX_O2HII] -
        k[393]*y[IDX_CI]*y[IDX_OHII] - k[394]*y[IDX_CI]*y[IDX_SiHII] -
        k[395]*y[IDX_CI]*y[IDX_SiOII] + k[400]*y[IDX_CHII]*y[IDX_H2COI] +
        k[403]*y[IDX_CHII]*y[IDX_H2OI] + k[405]*y[IDX_CHII]*y[IDX_HCNI] +
        k[407]*y[IDX_CHII]*y[IDX_HNCI] + k[458]*y[IDX_CHI]*y[IDX_COII] -
        k[509]*y[IDX_H2II]*y[IDX_CI] - k[576]*y[IDX_H3II]*y[IDX_CI] +
        k[663]*y[IDX_HeII]*y[IDX_CNI] + k[667]*y[IDX_HeII]*y[IDX_CO2I] +
        k[685]*y[IDX_HeII]*y[IDX_HNCI] + k[700]*y[IDX_HeII]*y[IDX_SiC3I] +
        k[701]*y[IDX_HeII]*y[IDX_SiCI] + k[720]*y[IDX_NII]*y[IDX_COI] +
        k[737]*y[IDX_NI]*y[IDX_CNII] + k[811]*y[IDX_OII]*y[IDX_CNI] +
        k[831]*y[IDX_OI]*y[IDX_SiCII] - k[863]*y[IDX_CI]*y[IDX_CH2I] -
        k[864]*y[IDX_CI]*y[IDX_HCOI] - k[865]*y[IDX_CI]*y[IDX_N2I] -
        k[866]*y[IDX_CI]*y[IDX_NH2I] - k[867]*y[IDX_CI]*y[IDX_NH2I] -
        k[868]*y[IDX_CI]*y[IDX_NH2I] - k[869]*y[IDX_CI]*y[IDX_NHI] -
        k[870]*y[IDX_CI]*y[IDX_NHI] - k[871]*y[IDX_CI]*y[IDX_NOI] -
        k[872]*y[IDX_CI]*y[IDX_NOI] - k[873]*y[IDX_CI]*y[IDX_O2I] -
        k[874]*y[IDX_CI]*y[IDX_OCNI] - k[875]*y[IDX_CI]*y[IDX_OHI] -
        k[876]*y[IDX_CI]*y[IDX_OHI] - k[877]*y[IDX_CI]*y[IDX_SiHI] +
        k[929]*y[IDX_CHI]*y[IDX_NI] + k[940]*y[IDX_CHI]*y[IDX_OI] -
        k[955]*y[IDX_H2I]*y[IDX_CI] + k[970]*y[IDX_HI]*y[IDX_CHI] +
        k[972]*y[IDX_HI]*y[IDX_COI] - k[1004]*y[IDX_HNCOI]*y[IDX_CI] +
        k[1011]*y[IDX_NI]*y[IDX_CNI] + k[1058]*y[IDX_OI]*y[IDX_CNI] +
        k[1084]*y[IDX_OI]*y[IDX_SiCI] + k[1104]*y[IDX_SiI]*y[IDX_COI] -
        k[1107]*y[IDX_CI] + k[1108]*y[IDX_CHII] + k[1128]*y[IDX_CHI] +
        k[1130]*y[IDX_CNI] + k[1133]*y[IDX_COI] + k[1177]*y[IDX_SiC3I] +
        k[1178]*y[IDX_SiCI] - k[1194]*y[IDX_CI]*y[IDX_NI] -
        k[1195]*y[IDX_CI]*y[IDX_OII] - k[1196]*y[IDX_CI]*y[IDX_OI] -
        k[1200]*y[IDX_H2I]*y[IDX_CI] - k[1206]*y[IDX_HI]*y[IDX_CI] +
        k[1214]*y[IDX_CII]*y[IDX_EM] - k[1250]*y[IDX_CI];
    ydot[IDX_CII] = 0.0 - k[14]*y[IDX_CII]*y[IDX_CH2I] -
        k[15]*y[IDX_CII]*y[IDX_CHI] - k[16]*y[IDX_CII]*y[IDX_H2COI] -
        k[17]*y[IDX_CII]*y[IDX_HCOI] - k[18]*y[IDX_CII]*y[IDX_MgI] -
        k[19]*y[IDX_CII]*y[IDX_NH3I] - k[20]*y[IDX_CII]*y[IDX_NOI] -
        k[21]*y[IDX_CII]*y[IDX_SiI] - k[22]*y[IDX_CII]*y[IDX_SiC2I] -
        k[23]*y[IDX_CII]*y[IDX_SiC3I] - k[24]*y[IDX_CII]*y[IDX_SiCI] -
        k[25]*y[IDX_CII]*y[IDX_SiH2I] - k[26]*y[IDX_CII]*y[IDX_SiH3I] +
        k[27]*y[IDX_CI]*y[IDX_CNII] + k[28]*y[IDX_CI]*y[IDX_COII] +
        k[29]*y[IDX_CI]*y[IDX_N2II] + k[30]*y[IDX_CI]*y[IDX_O2II] +
        k[139]*y[IDX_HeII]*y[IDX_CI] + k[231]*y[IDX_CI] + k[240]*y[IDX_CI] +
        k[241]*y[IDX_CHII] - k[365]*y[IDX_CII]*y[IDX_CH3OHI] -
        k[366]*y[IDX_CII]*y[IDX_CH3OHI] - k[367]*y[IDX_CII]*y[IDX_CO2I] -
        k[368]*y[IDX_CII]*y[IDX_H2COI] - k[369]*y[IDX_CII]*y[IDX_H2COI] -
        k[370]*y[IDX_CII]*y[IDX_H2OI] - k[371]*y[IDX_CII]*y[IDX_H2OI] -
        k[372]*y[IDX_CII]*y[IDX_HCOI] - k[373]*y[IDX_CII]*y[IDX_NH2I] -
        k[374]*y[IDX_CII]*y[IDX_NH3I] - k[375]*y[IDX_CII]*y[IDX_NHI] -
        k[376]*y[IDX_CII]*y[IDX_O2I] - k[377]*y[IDX_CII]*y[IDX_O2I] -
        k[378]*y[IDX_CII]*y[IDX_OCNI] - k[379]*y[IDX_CII]*y[IDX_OHI] -
        k[380]*y[IDX_CII]*y[IDX_SiH2I] - k[381]*y[IDX_CII]*y[IDX_SiHI] -
        k[382]*y[IDX_CII]*y[IDX_SiOI] - k[528]*y[IDX_H2I]*y[IDX_CII] +
        k[614]*y[IDX_HI]*y[IDX_CHII] + k[653]*y[IDX_HeII]*y[IDX_CH2I] +
        k[662]*y[IDX_HeII]*y[IDX_CHI] + k[664]*y[IDX_HeII]*y[IDX_CNI] +
        k[668]*y[IDX_HeII]*y[IDX_CO2I] + k[669]*y[IDX_HeII]*y[IDX_COI] +
        k[678]*y[IDX_HeII]*y[IDX_HCNI] + k[684]*y[IDX_HeII]*y[IDX_HNCI] +
        k[702]*y[IDX_HeII]*y[IDX_SiCI] + k[1107]*y[IDX_CI] +
        k[1109]*y[IDX_CH2II] + k[1131]*y[IDX_COII] -
        k[1192]*y[IDX_CII]*y[IDX_NI] - k[1193]*y[IDX_CII]*y[IDX_OI] -
        k[1199]*y[IDX_H2I]*y[IDX_CII] - k[1205]*y[IDX_HI]*y[IDX_CII] -
        k[1214]*y[IDX_CII]*y[IDX_EM] - k[1264]*y[IDX_CII];
    ydot[IDX_CHI] = 0.0 - k[0]*y[IDX_CHI]*y[IDX_OI] -
        k[2]*y[IDX_H2I]*y[IDX_CHI] - k[9]*y[IDX_HI]*y[IDX_CHI] -
        k[15]*y[IDX_CII]*y[IDX_CHI] + k[31]*y[IDX_CHII]*y[IDX_HCOI] +
        k[32]*y[IDX_CHII]*y[IDX_MgI] + k[33]*y[IDX_CHII]*y[IDX_NH3I] +
        k[34]*y[IDX_CHII]*y[IDX_NOI] + k[35]*y[IDX_CHII]*y[IDX_SiI] -
        k[53]*y[IDX_CHI]*y[IDX_CNII] - k[54]*y[IDX_CHI]*y[IDX_COII] -
        k[55]*y[IDX_CHI]*y[IDX_H2COII] - k[56]*y[IDX_CHI]*y[IDX_H2OII] -
        k[57]*y[IDX_CHI]*y[IDX_NII] - k[58]*y[IDX_CHI]*y[IDX_N2II] -
        k[59]*y[IDX_CHI]*y[IDX_NH2II] - k[60]*y[IDX_CHI]*y[IDX_OII] -
        k[61]*y[IDX_CHI]*y[IDX_O2II] - k[62]*y[IDX_CHI]*y[IDX_OHII] -
        k[78]*y[IDX_HII]*y[IDX_CHI] - k[102]*y[IDX_H2II]*y[IDX_CHI] -
        k[141]*y[IDX_HeII]*y[IDX_CHI] + k[243]*y[IDX_CH2I] + k[246]*y[IDX_CH3I]
        - k[250]*y[IDX_CHI] + k[297]*y[IDX_CH2II]*y[IDX_EM] +
        k[299]*y[IDX_CH3II]*y[IDX_EM] + k[300]*y[IDX_CH3II]*y[IDX_EM] +
        k[318]*y[IDX_H3COII]*y[IDX_EM] + k[365]*y[IDX_CII]*y[IDX_CH3OHI] +
        k[369]*y[IDX_CII]*y[IDX_H2COI] + k[422]*y[IDX_CH2I]*y[IDX_COII] -
        k[458]*y[IDX_CHI]*y[IDX_COII] - k[459]*y[IDX_CHI]*y[IDX_H2COII] -
        k[460]*y[IDX_CHI]*y[IDX_H2OII] - k[461]*y[IDX_CHI]*y[IDX_H3COII] -
        k[462]*y[IDX_CHI]*y[IDX_H3OII] - k[463]*y[IDX_CHI]*y[IDX_HCNII] -
        k[464]*y[IDX_CHI]*y[IDX_HCNHII] - k[465]*y[IDX_CHI]*y[IDX_HCNHII] -
        k[466]*y[IDX_CHI]*y[IDX_HCOII] - k[467]*y[IDX_CHI]*y[IDX_HNOII] -
        k[468]*y[IDX_CHI]*y[IDX_NII] - k[469]*y[IDX_CHI]*y[IDX_N2HII] -
        k[470]*y[IDX_CHI]*y[IDX_NHII] - k[471]*y[IDX_CHI]*y[IDX_NH2II] -
        k[472]*y[IDX_CHI]*y[IDX_OII] - k[473]*y[IDX_CHI]*y[IDX_O2II] -
        k[474]*y[IDX_CHI]*y[IDX_O2HII] - k[475]*y[IDX_CHI]*y[IDX_OHII] -
        k[476]*y[IDX_CHI]*y[IDX_SiII] - k[477]*y[IDX_CHI]*y[IDX_SiHII] -
        k[478]*y[IDX_CHI]*y[IDX_SiOII] - k[512]*y[IDX_H2II]*y[IDX_CHI] -
        k[580]*y[IDX_H3II]*y[IDX_CHI] - k[662]*y[IDX_HeII]*y[IDX_CHI] +
        k[677]*y[IDX_HeII]*y[IDX_HCNI] + k[815]*y[IDX_OII]*y[IDX_HCNI] +
        k[863]*y[IDX_CI]*y[IDX_CH2I] + k[863]*y[IDX_CI]*y[IDX_CH2I] +
        k[864]*y[IDX_CI]*y[IDX_HCOI] + k[868]*y[IDX_CI]*y[IDX_NH2I] +
        k[870]*y[IDX_CI]*y[IDX_NHI] + k[876]*y[IDX_CI]*y[IDX_OHI] +
        k[878]*y[IDX_CH2I]*y[IDX_CH2I] + k[880]*y[IDX_CH2I]*y[IDX_CNI] +
        k[897]*y[IDX_CH2I]*y[IDX_OI] + k[899]*y[IDX_CH2I]*y[IDX_OHI] -
        k[923]*y[IDX_CHI]*y[IDX_CO2I] - k[924]*y[IDX_CHI]*y[IDX_H2COI] -
        k[925]*y[IDX_CHI]*y[IDX_HCOI] - k[926]*y[IDX_CHI]*y[IDX_HNOI] -
        k[927]*y[IDX_CHI]*y[IDX_N2I] - k[928]*y[IDX_CHI]*y[IDX_NI] -
        k[929]*y[IDX_CHI]*y[IDX_NI] - k[930]*y[IDX_CHI]*y[IDX_NOI] -
        k[931]*y[IDX_CHI]*y[IDX_NOI] - k[932]*y[IDX_CHI]*y[IDX_NOI] -
        k[933]*y[IDX_CHI]*y[IDX_O2I] - k[934]*y[IDX_CHI]*y[IDX_O2I] -
        k[935]*y[IDX_CHI]*y[IDX_O2I] - k[936]*y[IDX_CHI]*y[IDX_O2I] -
        k[937]*y[IDX_CHI]*y[IDX_O2HI] - k[938]*y[IDX_CHI]*y[IDX_O2HI] -
        k[939]*y[IDX_CHI]*y[IDX_OI] - k[940]*y[IDX_CHI]*y[IDX_OI] -
        k[941]*y[IDX_CHI]*y[IDX_OHI] + k[955]*y[IDX_H2I]*y[IDX_CI] -
        k[958]*y[IDX_H2I]*y[IDX_CHI] + k[967]*y[IDX_HI]*y[IDX_CH2I] -
        k[970]*y[IDX_HI]*y[IDX_CHI] + k[1007]*y[IDX_NI]*y[IDX_CH2I] +
        k[1111]*y[IDX_CH2II] + k[1113]*y[IDX_CH2I] + k[1118]*y[IDX_CH3I] +
        k[1127]*y[IDX_CH4I] - k[1128]*y[IDX_CHI] - k[1129]*y[IDX_CHI] -
        k[1201]*y[IDX_H2I]*y[IDX_CHI] + k[1206]*y[IDX_HI]*y[IDX_CI] -
        k[1253]*y[IDX_CHI];
    ydot[IDX_CHII] = 0.0 + k[15]*y[IDX_CII]*y[IDX_CHI] -
        k[31]*y[IDX_CHII]*y[IDX_HCOI] - k[32]*y[IDX_CHII]*y[IDX_MgI] -
        k[33]*y[IDX_CHII]*y[IDX_NH3I] - k[34]*y[IDX_CHII]*y[IDX_NOI] -
        k[35]*y[IDX_CHII]*y[IDX_SiI] + k[53]*y[IDX_CHI]*y[IDX_CNII] +
        k[54]*y[IDX_CHI]*y[IDX_COII] + k[55]*y[IDX_CHI]*y[IDX_H2COII] +
        k[56]*y[IDX_CHI]*y[IDX_H2OII] + k[57]*y[IDX_CHI]*y[IDX_NII] +
        k[58]*y[IDX_CHI]*y[IDX_N2II] + k[59]*y[IDX_CHI]*y[IDX_NH2II] +
        k[60]*y[IDX_CHI]*y[IDX_OII] + k[61]*y[IDX_CHI]*y[IDX_O2II] +
        k[62]*y[IDX_CHI]*y[IDX_OHII] + k[78]*y[IDX_HII]*y[IDX_CHI] +
        k[102]*y[IDX_H2II]*y[IDX_CHI] + k[141]*y[IDX_HeII]*y[IDX_CHI] -
        k[241]*y[IDX_CHII] - k[294]*y[IDX_CHII]*y[IDX_EM] +
        k[372]*y[IDX_CII]*y[IDX_HCOI] + k[383]*y[IDX_CI]*y[IDX_H2OII] +
        k[385]*y[IDX_CI]*y[IDX_HCNII] + k[386]*y[IDX_CI]*y[IDX_HCOII] +
        k[387]*y[IDX_CI]*y[IDX_HCO2II] + k[388]*y[IDX_CI]*y[IDX_HNOII] +
        k[389]*y[IDX_CI]*y[IDX_N2HII] + k[390]*y[IDX_CI]*y[IDX_NHII] +
        k[392]*y[IDX_CI]*y[IDX_O2HII] + k[393]*y[IDX_CI]*y[IDX_OHII] -
        k[396]*y[IDX_CHII]*y[IDX_CH3OHI] - k[397]*y[IDX_CHII]*y[IDX_CH3OHI] -
        k[398]*y[IDX_CHII]*y[IDX_CO2I] - k[399]*y[IDX_CHII]*y[IDX_H2COI] -
        k[400]*y[IDX_CHII]*y[IDX_H2COI] - k[401]*y[IDX_CHII]*y[IDX_H2COI] -
        k[402]*y[IDX_CHII]*y[IDX_H2OI] - k[403]*y[IDX_CHII]*y[IDX_H2OI] -
        k[404]*y[IDX_CHII]*y[IDX_H2OI] - k[405]*y[IDX_CHII]*y[IDX_HCNI] -
        k[406]*y[IDX_CHII]*y[IDX_HCOI] - k[407]*y[IDX_CHII]*y[IDX_HNCI] -
        k[408]*y[IDX_CHII]*y[IDX_NI] - k[409]*y[IDX_CHII]*y[IDX_NH2I] -
        k[410]*y[IDX_CHII]*y[IDX_NHI] - k[411]*y[IDX_CHII]*y[IDX_O2I] -
        k[412]*y[IDX_CHII]*y[IDX_O2I] - k[413]*y[IDX_CHII]*y[IDX_O2I] -
        k[414]*y[IDX_CHII]*y[IDX_OI] - k[415]*y[IDX_CHII]*y[IDX_OHI] +
        k[491]*y[IDX_HII]*y[IDX_CH2I] + k[509]*y[IDX_H2II]*y[IDX_CI] +
        k[528]*y[IDX_H2I]*y[IDX_CII] - k[529]*y[IDX_H2I]*y[IDX_CHII] +
        k[576]*y[IDX_H3II]*y[IDX_CI] - k[614]*y[IDX_HI]*y[IDX_CHII] +
        k[615]*y[IDX_HI]*y[IDX_CH2II] + k[654]*y[IDX_HeII]*y[IDX_CH2I] +
        k[655]*y[IDX_HeII]*y[IDX_CH3I] + k[658]*y[IDX_HeII]*y[IDX_CH4I] +
        k[679]*y[IDX_HeII]*y[IDX_HCNI] + k[682]*y[IDX_HeII]*y[IDX_HCOI] -
        k[1108]*y[IDX_CHII] + k[1110]*y[IDX_CH2II] + k[1114]*y[IDX_CH3II] +
        k[1129]*y[IDX_CHI] + k[1205]*y[IDX_HI]*y[IDX_CII] - k[1272]*y[IDX_CHII];
    ydot[IDX_CH2I] = 0.0 - k[14]*y[IDX_CII]*y[IDX_CH2I] +
        k[36]*y[IDX_CH2II]*y[IDX_NOI] - k[37]*y[IDX_CH2I]*y[IDX_CNII] -
        k[38]*y[IDX_CH2I]*y[IDX_COII] - k[39]*y[IDX_CH2I]*y[IDX_H2COII] -
        k[40]*y[IDX_CH2I]*y[IDX_H2OII] - k[41]*y[IDX_CH2I]*y[IDX_N2II] -
        k[42]*y[IDX_CH2I]*y[IDX_NH2II] - k[43]*y[IDX_CH2I]*y[IDX_OII] -
        k[44]*y[IDX_CH2I]*y[IDX_O2II] - k[45]*y[IDX_CH2I]*y[IDX_OHII] -
        k[75]*y[IDX_HII]*y[IDX_CH2I] - k[100]*y[IDX_H2II]*y[IDX_CH2I] -
        k[155]*y[IDX_NII]*y[IDX_CH2I] - k[242]*y[IDX_CH2I] - k[243]*y[IDX_CH2I]
        + k[244]*y[IDX_CH3I] + k[249]*y[IDX_CH4I] +
        k[298]*y[IDX_CH3II]*y[IDX_EM] + k[301]*y[IDX_CH4II]*y[IDX_EM] +
        k[306]*y[IDX_H2COII]*y[IDX_EM] + k[317]*y[IDX_H3COII]*y[IDX_EM] +
        k[397]*y[IDX_CHII]*y[IDX_CH3OHI] + k[401]*y[IDX_CHII]*y[IDX_H2COI] -
        k[422]*y[IDX_CH2I]*y[IDX_COII] - k[423]*y[IDX_CH2I]*y[IDX_H2COII] -
        k[424]*y[IDX_CH2I]*y[IDX_H2OII] - k[425]*y[IDX_CH2I]*y[IDX_H3OII] -
        k[426]*y[IDX_CH2I]*y[IDX_HCNII] - k[427]*y[IDX_CH2I]*y[IDX_HCNHII] -
        k[428]*y[IDX_CH2I]*y[IDX_HCNHII] - k[429]*y[IDX_CH2I]*y[IDX_HCOII] -
        k[430]*y[IDX_CH2I]*y[IDX_HNOII] - k[431]*y[IDX_CH2I]*y[IDX_N2HII] -
        k[432]*y[IDX_CH2I]*y[IDX_NHII] - k[433]*y[IDX_CH2I]*y[IDX_NH2II] -
        k[434]*y[IDX_CH2I]*y[IDX_NH3II] - k[435]*y[IDX_CH2I]*y[IDX_O2II] -
        k[436]*y[IDX_CH2I]*y[IDX_O2HII] - k[437]*y[IDX_CH2I]*y[IDX_OHII] -
        k[438]*y[IDX_CH2I]*y[IDX_SiOII] + k[457]*y[IDX_CH4I]*y[IDX_OHII] -
        k[491]*y[IDX_HII]*y[IDX_CH2I] - k[510]*y[IDX_H2II]*y[IDX_CH2I] -
        k[577]*y[IDX_H3II]*y[IDX_CH2I] - k[653]*y[IDX_HeII]*y[IDX_CH2I] -
        k[654]*y[IDX_HeII]*y[IDX_CH2I] + k[722]*y[IDX_NII]*y[IDX_H2COI] -
        k[863]*y[IDX_CI]*y[IDX_CH2I] - k[878]*y[IDX_CH2I]*y[IDX_CH2I] -
        k[878]*y[IDX_CH2I]*y[IDX_CH2I] - k[879]*y[IDX_CH2I]*y[IDX_CH4I] -
        k[880]*y[IDX_CH2I]*y[IDX_CNI] - k[881]*y[IDX_CH2I]*y[IDX_H2COI] -
        k[882]*y[IDX_CH2I]*y[IDX_HCOI] - k[883]*y[IDX_CH2I]*y[IDX_HNOI] -
        k[884]*y[IDX_CH2I]*y[IDX_N2I] - k[885]*y[IDX_CH2I]*y[IDX_NO2I] -
        k[886]*y[IDX_CH2I]*y[IDX_NOI] - k[887]*y[IDX_CH2I]*y[IDX_NOI] -
        k[888]*y[IDX_CH2I]*y[IDX_NOI] - k[889]*y[IDX_CH2I]*y[IDX_O2I] -
        k[890]*y[IDX_CH2I]*y[IDX_O2I] - k[891]*y[IDX_CH2I]*y[IDX_O2I] -
        k[892]*y[IDX_CH2I]*y[IDX_O2I] - k[893]*y[IDX_CH2I]*y[IDX_O2I] -
        k[894]*y[IDX_CH2I]*y[IDX_OI] - k[895]*y[IDX_CH2I]*y[IDX_OI] -
        k[896]*y[IDX_CH2I]*y[IDX_OI] - k[897]*y[IDX_CH2I]*y[IDX_OI] -
        k[898]*y[IDX_CH2I]*y[IDX_OHI] - k[899]*y[IDX_CH2I]*y[IDX_OHI] -
        k[900]*y[IDX_CH2I]*y[IDX_OHI] + k[901]*y[IDX_CH3I]*y[IDX_CH3I] +
        k[902]*y[IDX_CH3I]*y[IDX_CNI] + k[913]*y[IDX_CH3I]*y[IDX_O2I] +
        k[919]*y[IDX_CH3I]*y[IDX_OHI] + k[924]*y[IDX_CHI]*y[IDX_H2COI] +
        k[925]*y[IDX_CHI]*y[IDX_HCOI] + k[926]*y[IDX_CHI]*y[IDX_HNOI] +
        k[938]*y[IDX_CHI]*y[IDX_O2HI] - k[956]*y[IDX_H2I]*y[IDX_CH2I] +
        k[958]*y[IDX_H2I]*y[IDX_CHI] - k[967]*y[IDX_HI]*y[IDX_CH2I] +
        k[968]*y[IDX_HI]*y[IDX_CH3I] + k[978]*y[IDX_HI]*y[IDX_HCOI] -
        k[1005]*y[IDX_NI]*y[IDX_CH2I] - k[1006]*y[IDX_NI]*y[IDX_CH2I] -
        k[1007]*y[IDX_NI]*y[IDX_CH2I] - k[1112]*y[IDX_CH2I] -
        k[1113]*y[IDX_CH2I] + k[1116]*y[IDX_CH3I] + k[1124]*y[IDX_CH4I] +
        k[1200]*y[IDX_H2I]*y[IDX_CI] - k[1256]*y[IDX_CH2I];
    ydot[IDX_CH2II] = 0.0 + k[14]*y[IDX_CII]*y[IDX_CH2I] -
        k[36]*y[IDX_CH2II]*y[IDX_NOI] + k[37]*y[IDX_CH2I]*y[IDX_CNII] +
        k[38]*y[IDX_CH2I]*y[IDX_COII] + k[39]*y[IDX_CH2I]*y[IDX_H2COII] +
        k[40]*y[IDX_CH2I]*y[IDX_H2OII] + k[41]*y[IDX_CH2I]*y[IDX_N2II] +
        k[42]*y[IDX_CH2I]*y[IDX_NH2II] + k[43]*y[IDX_CH2I]*y[IDX_OII] +
        k[44]*y[IDX_CH2I]*y[IDX_O2II] + k[45]*y[IDX_CH2I]*y[IDX_OHII] +
        k[75]*y[IDX_HII]*y[IDX_CH2I] + k[100]*y[IDX_H2II]*y[IDX_CH2I] +
        k[155]*y[IDX_NII]*y[IDX_CH2I] + k[242]*y[IDX_CH2I] -
        k[295]*y[IDX_CH2II]*y[IDX_EM] - k[296]*y[IDX_CH2II]*y[IDX_EM] -
        k[297]*y[IDX_CH2II]*y[IDX_EM] + k[368]*y[IDX_CII]*y[IDX_H2COI] +
        k[406]*y[IDX_CHII]*y[IDX_HCOI] - k[416]*y[IDX_CH2II]*y[IDX_CO2I] -
        k[417]*y[IDX_CH2II]*y[IDX_H2COI] - k[418]*y[IDX_CH2II]*y[IDX_H2OI] -
        k[419]*y[IDX_CH2II]*y[IDX_HCOI] - k[420]*y[IDX_CH2II]*y[IDX_O2I] -
        k[421]*y[IDX_CH2II]*y[IDX_OI] + k[455]*y[IDX_CH4I]*y[IDX_N2II] +
        k[459]*y[IDX_CHI]*y[IDX_H2COII] + k[460]*y[IDX_CHI]*y[IDX_H2OII] +
        k[461]*y[IDX_CHI]*y[IDX_H3COII] + k[462]*y[IDX_CHI]*y[IDX_H3OII] +
        k[463]*y[IDX_CHI]*y[IDX_HCNII] + k[464]*y[IDX_CHI]*y[IDX_HCNHII] +
        k[465]*y[IDX_CHI]*y[IDX_HCNHII] + k[466]*y[IDX_CHI]*y[IDX_HCOII] +
        k[467]*y[IDX_CHI]*y[IDX_HNOII] + k[469]*y[IDX_CHI]*y[IDX_N2HII] +
        k[470]*y[IDX_CHI]*y[IDX_NHII] + k[471]*y[IDX_CHI]*y[IDX_NH2II] +
        k[474]*y[IDX_CHI]*y[IDX_O2HII] + k[475]*y[IDX_CHI]*y[IDX_OHII] +
        k[477]*y[IDX_CHI]*y[IDX_SiHII] + k[512]*y[IDX_H2II]*y[IDX_CHI] +
        k[529]*y[IDX_H2I]*y[IDX_CHII] - k[530]*y[IDX_H2I]*y[IDX_CH2II] +
        k[580]*y[IDX_H3II]*y[IDX_CHI] - k[615]*y[IDX_HI]*y[IDX_CH2II] +
        k[616]*y[IDX_HI]*y[IDX_CH3II] + k[659]*y[IDX_HeII]*y[IDX_CH4I] +
        k[672]*y[IDX_HeII]*y[IDX_H2COI] - k[736]*y[IDX_NI]*y[IDX_CH2II] -
        k[1109]*y[IDX_CH2II] - k[1110]*y[IDX_CH2II] - k[1111]*y[IDX_CH2II] +
        k[1112]*y[IDX_CH2I] + k[1115]*y[IDX_CH3II] + k[1122]*y[IDX_CH4II] +
        k[1199]*y[IDX_H2I]*y[IDX_CII] - k[1278]*y[IDX_CH2II];
    ydot[IDX_CH3I] = 0.0 + k[46]*y[IDX_CH3II]*y[IDX_HCOI] +
        k[47]*y[IDX_CH3II]*y[IDX_MgI] + k[48]*y[IDX_CH3II]*y[IDX_NOI] -
        k[76]*y[IDX_HII]*y[IDX_CH3I] - k[244]*y[IDX_CH3I] - k[245]*y[IDX_CH3I] -
        k[246]*y[IDX_CH3I] + k[248]*y[IDX_CH3OHI] +
        k[302]*y[IDX_CH4II]*y[IDX_EM] + k[417]*y[IDX_CH2II]*y[IDX_H2COI] +
        k[447]*y[IDX_CH4II]*y[IDX_CO2I] + k[448]*y[IDX_CH4II]*y[IDX_COI] +
        k[449]*y[IDX_CH4II]*y[IDX_H2COI] + k[450]*y[IDX_CH4II]*y[IDX_H2OI] +
        k[451]*y[IDX_CH4I]*y[IDX_COII] + k[452]*y[IDX_CH4I]*y[IDX_H2COII] +
        k[453]*y[IDX_CH4I]*y[IDX_H2OII] + k[454]*y[IDX_CH4I]*y[IDX_HCNII] -
        k[578]*y[IDX_H3II]*y[IDX_CH3I] - k[655]*y[IDX_HeII]*y[IDX_CH3I] +
        k[656]*y[IDX_HeII]*y[IDX_CH3OHI] + k[661]*y[IDX_HeII]*y[IDX_CH4I] +
        k[714]*y[IDX_NII]*y[IDX_CH3OHI] + k[860]*y[IDX_SiII]*y[IDX_CH3OHI] +
        k[878]*y[IDX_CH2I]*y[IDX_CH2I] + k[879]*y[IDX_CH2I]*y[IDX_CH4I] +
        k[879]*y[IDX_CH2I]*y[IDX_CH4I] + k[881]*y[IDX_CH2I]*y[IDX_H2COI] +
        k[882]*y[IDX_CH2I]*y[IDX_HCOI] + k[883]*y[IDX_CH2I]*y[IDX_HNOI] +
        k[900]*y[IDX_CH2I]*y[IDX_OHI] - k[901]*y[IDX_CH3I]*y[IDX_CH3I] -
        k[901]*y[IDX_CH3I]*y[IDX_CH3I] - k[902]*y[IDX_CH3I]*y[IDX_CNI] -
        k[903]*y[IDX_CH3I]*y[IDX_H2COI] - k[904]*y[IDX_CH3I]*y[IDX_H2OI] -
        k[905]*y[IDX_CH3I]*y[IDX_HCOI] - k[906]*y[IDX_CH3I]*y[IDX_HNOI] -
        k[907]*y[IDX_CH3I]*y[IDX_NH2I] - k[908]*y[IDX_CH3I]*y[IDX_NH3I] -
        k[909]*y[IDX_CH3I]*y[IDX_NO2I] - k[910]*y[IDX_CH3I]*y[IDX_NOI] -
        k[911]*y[IDX_CH3I]*y[IDX_O2I] - k[912]*y[IDX_CH3I]*y[IDX_O2I] -
        k[913]*y[IDX_CH3I]*y[IDX_O2I] - k[914]*y[IDX_CH3I]*y[IDX_O2HI] -
        k[915]*y[IDX_CH3I]*y[IDX_OI] - k[916]*y[IDX_CH3I]*y[IDX_OI] -
        k[917]*y[IDX_CH3I]*y[IDX_OHI] - k[918]*y[IDX_CH3I]*y[IDX_OHI] -
        k[919]*y[IDX_CH3I]*y[IDX_OHI] + k[920]*y[IDX_CH4I]*y[IDX_CNI] +
        k[921]*y[IDX_CH4I]*y[IDX_O2I] + k[922]*y[IDX_CH4I]*y[IDX_OHI] +
        k[956]*y[IDX_H2I]*y[IDX_CH2I] - k[957]*y[IDX_H2I]*y[IDX_CH3I] -
        k[968]*y[IDX_HI]*y[IDX_CH3I] + k[969]*y[IDX_HI]*y[IDX_CH4I] -
        k[1008]*y[IDX_NI]*y[IDX_CH3I] - k[1009]*y[IDX_NI]*y[IDX_CH3I] -
        k[1010]*y[IDX_NI]*y[IDX_CH3I] + k[1028]*y[IDX_NH2I]*y[IDX_CH4I] +
        k[1034]*y[IDX_NHI]*y[IDX_CH4I] + k[1056]*y[IDX_OI]*y[IDX_CH4I] -
        k[1116]*y[IDX_CH3I] - k[1117]*y[IDX_CH3I] - k[1118]*y[IDX_CH3I] +
        k[1121]*y[IDX_CH3OHI] + k[1125]*y[IDX_CH4I] +
        k[1201]*y[IDX_H2I]*y[IDX_CHI] + k[1215]*y[IDX_CH3II]*y[IDX_EM] -
        k[1259]*y[IDX_CH3I];
    ydot[IDX_CH3II] = 0.0 - k[46]*y[IDX_CH3II]*y[IDX_HCOI] -
        k[47]*y[IDX_CH3II]*y[IDX_MgI] - k[48]*y[IDX_CH3II]*y[IDX_NOI] +
        k[76]*y[IDX_HII]*y[IDX_CH3I] + k[245]*y[IDX_CH3I] -
        k[298]*y[IDX_CH3II]*y[IDX_EM] - k[299]*y[IDX_CH3II]*y[IDX_EM] -
        k[300]*y[IDX_CH3II]*y[IDX_EM] + k[366]*y[IDX_CII]*y[IDX_CH3OHI] +
        k[396]*y[IDX_CHII]*y[IDX_CH3OHI] + k[399]*y[IDX_CHII]*y[IDX_H2COI] +
        k[419]*y[IDX_CH2II]*y[IDX_HCOI] + k[423]*y[IDX_CH2I]*y[IDX_H2COII] +
        k[424]*y[IDX_CH2I]*y[IDX_H2OII] + k[425]*y[IDX_CH2I]*y[IDX_H3OII] +
        k[426]*y[IDX_CH2I]*y[IDX_HCNII] + k[427]*y[IDX_CH2I]*y[IDX_HCNHII] +
        k[428]*y[IDX_CH2I]*y[IDX_HCNHII] + k[429]*y[IDX_CH2I]*y[IDX_HCOII] +
        k[430]*y[IDX_CH2I]*y[IDX_HNOII] + k[431]*y[IDX_CH2I]*y[IDX_N2HII] +
        k[432]*y[IDX_CH2I]*y[IDX_NHII] + k[433]*y[IDX_CH2I]*y[IDX_NH2II] +
        k[434]*y[IDX_CH2I]*y[IDX_NH3II] + k[436]*y[IDX_CH2I]*y[IDX_O2HII] +
        k[437]*y[IDX_CH2I]*y[IDX_OHII] - k[439]*y[IDX_CH3II]*y[IDX_CH3OHI] -
        k[440]*y[IDX_CH3II]*y[IDX_H2COI] - k[441]*y[IDX_CH3II]*y[IDX_HCOI] -
        k[442]*y[IDX_CH3II]*y[IDX_O2I] - k[443]*y[IDX_CH3II]*y[IDX_OI] -
        k[444]*y[IDX_CH3II]*y[IDX_OI] - k[445]*y[IDX_CH3II]*y[IDX_OHI] -
        k[446]*y[IDX_CH3II]*y[IDX_SiH4I] + k[456]*y[IDX_CH4I]*y[IDX_N2II] +
        k[492]*y[IDX_HII]*y[IDX_CH3OHI] + k[495]*y[IDX_HII]*y[IDX_CH4I] +
        k[510]*y[IDX_H2II]*y[IDX_CH2I] + k[511]*y[IDX_H2II]*y[IDX_CH4I] +
        k[530]*y[IDX_H2I]*y[IDX_CH2II] + k[577]*y[IDX_H3II]*y[IDX_CH2I] +
        k[579]*y[IDX_H3II]*y[IDX_CH3OHI] - k[616]*y[IDX_HI]*y[IDX_CH3II] +
        k[617]*y[IDX_HI]*y[IDX_CH4II] + k[657]*y[IDX_HeII]*y[IDX_CH3OHI] +
        k[660]*y[IDX_HeII]*y[IDX_CH4I] + k[715]*y[IDX_NII]*y[IDX_CH3OHI] +
        k[716]*y[IDX_NII]*y[IDX_CH4I] - k[794]*y[IDX_NHI]*y[IDX_CH3II] +
        k[810]*y[IDX_OII]*y[IDX_CH4I] + k[822]*y[IDX_OI]*y[IDX_CH4II] -
        k[1114]*y[IDX_CH3II] - k[1115]*y[IDX_CH3II] + k[1117]*y[IDX_CH3I] +
        k[1123]*y[IDX_CH4II] - k[1215]*y[IDX_CH3II]*y[IDX_EM] -
        k[1285]*y[IDX_CH3II];
    ydot[IDX_CH3OHI] = 0.0 - k[247]*y[IDX_CH3OHI] - k[248]*y[IDX_CH3OHI] -
        k[365]*y[IDX_CII]*y[IDX_CH3OHI] - k[366]*y[IDX_CII]*y[IDX_CH3OHI] -
        k[396]*y[IDX_CHII]*y[IDX_CH3OHI] - k[397]*y[IDX_CHII]*y[IDX_CH3OHI] -
        k[439]*y[IDX_CH3II]*y[IDX_CH3OHI] - k[492]*y[IDX_HII]*y[IDX_CH3OHI] -
        k[493]*y[IDX_HII]*y[IDX_CH3OHI] - k[494]*y[IDX_HII]*y[IDX_CH3OHI] -
        k[579]*y[IDX_H3II]*y[IDX_CH3OHI] - k[656]*y[IDX_HeII]*y[IDX_CH3OHI] -
        k[657]*y[IDX_HeII]*y[IDX_CH3OHI] - k[712]*y[IDX_NII]*y[IDX_CH3OHI] -
        k[713]*y[IDX_NII]*y[IDX_CH3OHI] - k[714]*y[IDX_NII]*y[IDX_CH3OHI] -
        k[715]*y[IDX_NII]*y[IDX_CH3OHI] - k[808]*y[IDX_OII]*y[IDX_CH3OHI] -
        k[809]*y[IDX_OII]*y[IDX_CH3OHI] - k[820]*y[IDX_O2II]*y[IDX_CH3OHI] -
        k[860]*y[IDX_SiII]*y[IDX_CH3OHI] - k[1119]*y[IDX_CH3OHI] -
        k[1120]*y[IDX_CH3OHI] - k[1121]*y[IDX_CH3OHI] - k[1223]*y[IDX_CH3OHI] +
        k[1359]*y[IDX_GCH3OHI] + k[1360]*y[IDX_GCH3OHI] + k[1361]*y[IDX_GCH3OHI]
        + k[1362]*y[IDX_GCH3OHI];
    ydot[IDX_CH4I] = 0.0 + k[49]*y[IDX_CH4II]*y[IDX_H2COI] +
        k[50]*y[IDX_CH4II]*y[IDX_NH3I] + k[51]*y[IDX_CH4II]*y[IDX_O2I] -
        k[52]*y[IDX_CH4I]*y[IDX_COII] - k[77]*y[IDX_HII]*y[IDX_CH4I] -
        k[101]*y[IDX_H2II]*y[IDX_CH4I] - k[140]*y[IDX_HeII]*y[IDX_CH4I] -
        k[156]*y[IDX_NII]*y[IDX_CH4I] - k[207]*y[IDX_OII]*y[IDX_CH4I] -
        k[249]*y[IDX_CH4I] + k[439]*y[IDX_CH3II]*y[IDX_CH3OHI] +
        k[440]*y[IDX_CH3II]*y[IDX_H2COI] + k[446]*y[IDX_CH3II]*y[IDX_SiH4I] -
        k[451]*y[IDX_CH4I]*y[IDX_COII] - k[452]*y[IDX_CH4I]*y[IDX_H2COII] -
        k[453]*y[IDX_CH4I]*y[IDX_H2OII] - k[454]*y[IDX_CH4I]*y[IDX_HCNII] -
        k[455]*y[IDX_CH4I]*y[IDX_N2II] - k[456]*y[IDX_CH4I]*y[IDX_N2II] -
        k[457]*y[IDX_CH4I]*y[IDX_OHII] - k[495]*y[IDX_HII]*y[IDX_CH4I] -
        k[511]*y[IDX_H2II]*y[IDX_CH4I] - k[658]*y[IDX_HeII]*y[IDX_CH4I] -
        k[659]*y[IDX_HeII]*y[IDX_CH4I] - k[660]*y[IDX_HeII]*y[IDX_CH4I] -
        k[661]*y[IDX_HeII]*y[IDX_CH4I] - k[716]*y[IDX_NII]*y[IDX_CH4I] -
        k[717]*y[IDX_NII]*y[IDX_CH4I] - k[718]*y[IDX_NII]*y[IDX_CH4I] -
        k[810]*y[IDX_OII]*y[IDX_CH4I] - k[879]*y[IDX_CH2I]*y[IDX_CH4I] +
        k[901]*y[IDX_CH3I]*y[IDX_CH3I] + k[903]*y[IDX_CH3I]*y[IDX_H2COI] +
        k[904]*y[IDX_CH3I]*y[IDX_H2OI] + k[905]*y[IDX_CH3I]*y[IDX_HCOI] +
        k[906]*y[IDX_CH3I]*y[IDX_HNOI] + k[907]*y[IDX_CH3I]*y[IDX_NH2I] +
        k[908]*y[IDX_CH3I]*y[IDX_NH3I] + k[914]*y[IDX_CH3I]*y[IDX_O2HI] +
        k[917]*y[IDX_CH3I]*y[IDX_OHI] - k[920]*y[IDX_CH4I]*y[IDX_CNI] -
        k[921]*y[IDX_CH4I]*y[IDX_O2I] - k[922]*y[IDX_CH4I]*y[IDX_OHI] +
        k[957]*y[IDX_H2I]*y[IDX_CH3I] - k[969]*y[IDX_HI]*y[IDX_CH4I] -
        k[1028]*y[IDX_NH2I]*y[IDX_CH4I] - k[1034]*y[IDX_NHI]*y[IDX_CH4I] -
        k[1056]*y[IDX_OI]*y[IDX_CH4I] - k[1124]*y[IDX_CH4I] -
        k[1125]*y[IDX_CH4I] - k[1126]*y[IDX_CH4I] - k[1127]*y[IDX_CH4I] -
        k[1260]*y[IDX_CH4I] + k[1307]*y[IDX_GCH4I] + k[1308]*y[IDX_GCH4I] +
        k[1309]*y[IDX_GCH4I] + k[1310]*y[IDX_GCH4I];
    ydot[IDX_CH4II] = 0.0 - k[49]*y[IDX_CH4II]*y[IDX_H2COI] -
        k[50]*y[IDX_CH4II]*y[IDX_NH3I] - k[51]*y[IDX_CH4II]*y[IDX_O2I] +
        k[52]*y[IDX_CH4I]*y[IDX_COII] + k[77]*y[IDX_HII]*y[IDX_CH4I] +
        k[101]*y[IDX_H2II]*y[IDX_CH4I] + k[140]*y[IDX_HeII]*y[IDX_CH4I] +
        k[156]*y[IDX_NII]*y[IDX_CH4I] + k[207]*y[IDX_OII]*y[IDX_CH4I] -
        k[301]*y[IDX_CH4II]*y[IDX_EM] - k[302]*y[IDX_CH4II]*y[IDX_EM] +
        k[441]*y[IDX_CH3II]*y[IDX_HCOI] - k[447]*y[IDX_CH4II]*y[IDX_CO2I] -
        k[448]*y[IDX_CH4II]*y[IDX_COI] - k[449]*y[IDX_CH4II]*y[IDX_H2COI] -
        k[450]*y[IDX_CH4II]*y[IDX_H2OI] + k[578]*y[IDX_H3II]*y[IDX_CH3I] -
        k[617]*y[IDX_HI]*y[IDX_CH4II] - k[822]*y[IDX_OI]*y[IDX_CH4II] -
        k[1122]*y[IDX_CH4II] - k[1123]*y[IDX_CH4II] + k[1126]*y[IDX_CH4I] -
        k[1288]*y[IDX_CH4II];
    ydot[IDX_CNI] = 0.0 + k[27]*y[IDX_CI]*y[IDX_CNII] +
        k[37]*y[IDX_CH2I]*y[IDX_CNII] + k[53]*y[IDX_CHI]*y[IDX_CNII] +
        k[63]*y[IDX_CNII]*y[IDX_COI] + k[64]*y[IDX_CNII]*y[IDX_H2COI] +
        k[65]*y[IDX_CNII]*y[IDX_HCNI] + k[66]*y[IDX_CNII]*y[IDX_HCOI] +
        k[67]*y[IDX_CNII]*y[IDX_NOI] + k[68]*y[IDX_CNII]*y[IDX_O2I] -
        k[69]*y[IDX_CNI]*y[IDX_N2II] - k[103]*y[IDX_H2II]*y[IDX_CNI] +
        k[126]*y[IDX_HI]*y[IDX_CNII] - k[157]*y[IDX_NII]*y[IDX_CNI] +
        k[183]*y[IDX_NH2I]*y[IDX_CNII] + k[199]*y[IDX_NHI]*y[IDX_CNII] +
        k[216]*y[IDX_OI]*y[IDX_CNII] + k[225]*y[IDX_OHI]*y[IDX_CNII] -
        k[251]*y[IDX_CNI] + k[259]*y[IDX_HCNI] + k[262]*y[IDX_HNCI] +
        k[283]*y[IDX_OCNI] + k[326]*y[IDX_HCNII]*y[IDX_EM] +
        k[327]*y[IDX_HCNHII]*y[IDX_EM] + k[378]*y[IDX_CII]*y[IDX_OCNI] +
        k[385]*y[IDX_CI]*y[IDX_HCNII] + k[426]*y[IDX_CH2I]*y[IDX_HCNII] +
        k[463]*y[IDX_CHI]*y[IDX_HCNII] - k[482]*y[IDX_CNI]*y[IDX_HNOII] -
        k[483]*y[IDX_CNI]*y[IDX_O2HII] - k[513]*y[IDX_H2II]*y[IDX_CNI] +
        k[565]*y[IDX_H2OI]*y[IDX_HCNII] - k[581]*y[IDX_H3II]*y[IDX_CNI] +
        k[620]*y[IDX_HCNII]*y[IDX_CO2I] + k[621]*y[IDX_HCNII]*y[IDX_COI] +
        k[622]*y[IDX_HCNII]*y[IDX_H2COI] + k[623]*y[IDX_HCNII]*y[IDX_HCNI] +
        k[624]*y[IDX_HCNII]*y[IDX_HCOI] + k[626]*y[IDX_HCNII]*y[IDX_HNCI] -
        k[663]*y[IDX_HeII]*y[IDX_CNI] - k[664]*y[IDX_HeII]*y[IDX_CNI] +
        k[698]*y[IDX_HeII]*y[IDX_OCNI] + k[744]*y[IDX_NI]*y[IDX_SiCII] -
        k[747]*y[IDX_NHII]*y[IDX_CNI] + k[784]*y[IDX_NH2I]*y[IDX_HCNII] +
        k[798]*y[IDX_NHI]*y[IDX_HCNII] - k[811]*y[IDX_OII]*y[IDX_CNI] -
        k[836]*y[IDX_OHII]*y[IDX_CNI] + k[853]*y[IDX_OHI]*y[IDX_HCNII] +
        k[865]*y[IDX_CI]*y[IDX_N2I] + k[869]*y[IDX_CI]*y[IDX_NHI] +
        k[871]*y[IDX_CI]*y[IDX_NOI] + k[874]*y[IDX_CI]*y[IDX_OCNI] -
        k[880]*y[IDX_CH2I]*y[IDX_CNI] - k[902]*y[IDX_CH3I]*y[IDX_CNI] -
        k[920]*y[IDX_CH4I]*y[IDX_CNI] + k[928]*y[IDX_CHI]*y[IDX_NI] -
        k[942]*y[IDX_CNI]*y[IDX_H2COI] - k[943]*y[IDX_CNI]*y[IDX_HCOI] -
        k[944]*y[IDX_CNI]*y[IDX_HNOI] - k[945]*y[IDX_CNI]*y[IDX_NO2I] -
        k[946]*y[IDX_CNI]*y[IDX_NOI] - k[947]*y[IDX_CNI]*y[IDX_NOI] -
        k[948]*y[IDX_CNI]*y[IDX_O2I] - k[949]*y[IDX_CNI]*y[IDX_O2I] -
        k[950]*y[IDX_CNI]*y[IDX_SiH4I] - k[959]*y[IDX_H2I]*y[IDX_CNI] +
        k[976]*y[IDX_HI]*y[IDX_HCNI] + k[995]*y[IDX_HI]*y[IDX_OCNI] -
        k[1011]*y[IDX_NI]*y[IDX_CNI] + k[1027]*y[IDX_NI]*y[IDX_SiCI] -
        k[1033]*y[IDX_NH3I]*y[IDX_CNI] - k[1035]*y[IDX_NHI]*y[IDX_CNI] -
        k[1057]*y[IDX_OI]*y[IDX_CNI] - k[1058]*y[IDX_OI]*y[IDX_CNI] +
        k[1063]*y[IDX_OI]*y[IDX_HCNI] + k[1079]*y[IDX_OI]*y[IDX_OCNI] -
        k[1090]*y[IDX_OHI]*y[IDX_CNI] - k[1091]*y[IDX_OHI]*y[IDX_CNI] +
        k[1094]*y[IDX_OHI]*y[IDX_HCNI] - k[1130]*y[IDX_CNI] +
        k[1147]*y[IDX_HCNI] + k[1151]*y[IDX_HNCI] + k[1172]*y[IDX_OCNI] +
        k[1194]*y[IDX_CI]*y[IDX_NI] - k[1263]*y[IDX_CNI];
    ydot[IDX_CNII] = 0.0 - k[27]*y[IDX_CI]*y[IDX_CNII] -
        k[37]*y[IDX_CH2I]*y[IDX_CNII] - k[53]*y[IDX_CHI]*y[IDX_CNII] -
        k[63]*y[IDX_CNII]*y[IDX_COI] - k[64]*y[IDX_CNII]*y[IDX_H2COI] -
        k[65]*y[IDX_CNII]*y[IDX_HCNI] - k[66]*y[IDX_CNII]*y[IDX_HCOI] -
        k[67]*y[IDX_CNII]*y[IDX_NOI] - k[68]*y[IDX_CNII]*y[IDX_O2I] +
        k[69]*y[IDX_CNI]*y[IDX_N2II] + k[103]*y[IDX_H2II]*y[IDX_CNI] -
        k[126]*y[IDX_HI]*y[IDX_CNII] + k[157]*y[IDX_NII]*y[IDX_CNI] -
        k[183]*y[IDX_NH2I]*y[IDX_CNII] - k[199]*y[IDX_NHI]*y[IDX_CNII] -
        k[216]*y[IDX_OI]*y[IDX_CNII] - k[225]*y[IDX_OHI]*y[IDX_CNII] -
        k[303]*y[IDX_CNII]*y[IDX_EM] + k[375]*y[IDX_CII]*y[IDX_NHI] +
        k[408]*y[IDX_CHII]*y[IDX_NI] + k[410]*y[IDX_CHII]*y[IDX_NHI] +
        k[468]*y[IDX_CHI]*y[IDX_NII] - k[479]*y[IDX_CNII]*y[IDX_H2COI] -
        k[480]*y[IDX_CNII]*y[IDX_HCOI] - k[481]*y[IDX_CNII]*y[IDX_O2I] -
        k[531]*y[IDX_H2I]*y[IDX_CNII] - k[560]*y[IDX_H2OI]*y[IDX_CNII] -
        k[561]*y[IDX_H2OI]*y[IDX_CNII] + k[676]*y[IDX_HeII]*y[IDX_HCNI] +
        k[683]*y[IDX_HeII]*y[IDX_HNCI] + k[697]*y[IDX_HeII]*y[IDX_OCNI] -
        k[737]*y[IDX_NI]*y[IDX_CNII] + k[1192]*y[IDX_CII]*y[IDX_NI] -
        k[1276]*y[IDX_CNII];
    ydot[IDX_COI] = 0.0 + k[28]*y[IDX_CI]*y[IDX_COII] +
        k[38]*y[IDX_CH2I]*y[IDX_COII] + k[52]*y[IDX_CH4I]*y[IDX_COII] +
        k[54]*y[IDX_CHI]*y[IDX_COII] - k[63]*y[IDX_CNII]*y[IDX_COI] +
        k[70]*y[IDX_COII]*y[IDX_H2COI] + k[71]*y[IDX_COII]*y[IDX_HCOI] +
        k[72]*y[IDX_COII]*y[IDX_NOI] + k[73]*y[IDX_COII]*y[IDX_O2I] -
        k[74]*y[IDX_COI]*y[IDX_N2II] - k[104]*y[IDX_H2II]*y[IDX_COI] +
        k[123]*y[IDX_H2OI]*y[IDX_COII] + k[127]*y[IDX_HI]*y[IDX_COII] +
        k[134]*y[IDX_HCNI]*y[IDX_COII] - k[158]*y[IDX_NII]*y[IDX_COI] +
        k[184]*y[IDX_NH2I]*y[IDX_COII] + k[193]*y[IDX_NH3I]*y[IDX_COII] +
        k[200]*y[IDX_NHI]*y[IDX_COII] - k[208]*y[IDX_OII]*y[IDX_COI] +
        k[217]*y[IDX_OI]*y[IDX_COII] + k[226]*y[IDX_OHI]*y[IDX_COII] -
        k[232]*y[IDX_COI] + k[252]*y[IDX_CO2I] - k[253]*y[IDX_COI] +
        k[255]*y[IDX_H2COI] + k[260]*y[IDX_HCOI] + k[263]*y[IDX_HNCOI] +
        k[307]*y[IDX_H2COII]*y[IDX_EM] + k[308]*y[IDX_H2COII]*y[IDX_EM] +
        k[319]*y[IDX_H3COII]*y[IDX_EM] + k[330]*y[IDX_HCOII]*y[IDX_EM] +
        k[332]*y[IDX_HCO2II]*y[IDX_EM] + k[333]*y[IDX_HCO2II]*y[IDX_EM] +
        k[335]*y[IDX_HOCII]*y[IDX_EM] + k[367]*y[IDX_CII]*y[IDX_CO2I] +
        k[368]*y[IDX_CII]*y[IDX_H2COI] + k[372]*y[IDX_CII]*y[IDX_HCOI] +
        k[377]*y[IDX_CII]*y[IDX_O2I] + k[382]*y[IDX_CII]*y[IDX_SiOI] +
        k[386]*y[IDX_CI]*y[IDX_HCOII] + k[395]*y[IDX_CI]*y[IDX_SiOII] +
        k[398]*y[IDX_CHII]*y[IDX_CO2I] + k[399]*y[IDX_CHII]*y[IDX_H2COI] +
        k[406]*y[IDX_CHII]*y[IDX_HCOI] + k[416]*y[IDX_CH2II]*y[IDX_CO2I] +
        k[419]*y[IDX_CH2II]*y[IDX_HCOI] + k[429]*y[IDX_CH2I]*y[IDX_HCOII] +
        k[441]*y[IDX_CH3II]*y[IDX_HCOI] - k[448]*y[IDX_CH4II]*y[IDX_COI] +
        k[466]*y[IDX_CHI]*y[IDX_HCOII] + k[480]*y[IDX_CNII]*y[IDX_HCOI] +
        k[481]*y[IDX_CNII]*y[IDX_O2I] - k[485]*y[IDX_COI]*y[IDX_HCO2II] -
        k[486]*y[IDX_COI]*y[IDX_HNOII] - k[487]*y[IDX_COI]*y[IDX_N2HII] -
        k[488]*y[IDX_COI]*y[IDX_O2HII] - k[489]*y[IDX_COI]*y[IDX_SiH4II] -
        k[490]*y[IDX_COI]*y[IDX_SiOII] + k[501]*y[IDX_HII]*y[IDX_HCOI] +
        k[502]*y[IDX_HII]*y[IDX_HNCOI] - k[515]*y[IDX_H2II]*y[IDX_COI] +
        k[519]*y[IDX_H2II]*y[IDX_HCOI] - k[553]*y[IDX_H2OII]*y[IDX_COI] +
        k[557]*y[IDX_H2OII]*y[IDX_HCOI] + k[566]*y[IDX_H2OI]*y[IDX_HCOII] -
        k[583]*y[IDX_H3II]*y[IDX_COI] - k[584]*y[IDX_H3II]*y[IDX_COI] -
        k[621]*y[IDX_HCNII]*y[IDX_COI] + k[625]*y[IDX_HCNII]*y[IDX_HCOI] +
        k[629]*y[IDX_HCNI]*y[IDX_HCOII] + k[635]*y[IDX_HCOII]*y[IDX_H2COI] +
        k[636]*y[IDX_HCOII]*y[IDX_HCOI] + k[637]*y[IDX_HCOII]*y[IDX_SiH2I] +
        k[638]*y[IDX_HCOII]*y[IDX_SiH4I] + k[639]*y[IDX_HCOII]*y[IDX_SiHI] +
        k[640]*y[IDX_HCOII]*y[IDX_SiOI] + k[641]*y[IDX_HCOI]*y[IDX_H2COII] +
        k[644]*y[IDX_HCOI]*y[IDX_O2II] + k[648]*y[IDX_HNCI]*y[IDX_HCOII] +
        k[666]*y[IDX_HeII]*y[IDX_CO2I] - k[669]*y[IDX_HeII]*y[IDX_COI] +
        k[681]*y[IDX_HeII]*y[IDX_HCOI] - k[720]*y[IDX_NII]*y[IDX_COI] +
        k[723]*y[IDX_NII]*y[IDX_HCOI] + k[731]*y[IDX_N2II]*y[IDX_HCOI] +
        k[749]*y[IDX_NHII]*y[IDX_CO2I] - k[751]*y[IDX_NHII]*y[IDX_COI] +
        k[787]*y[IDX_NH2I]*y[IDX_HCOII] + k[799]*y[IDX_NHI]*y[IDX_HCOII] +
        k[812]*y[IDX_OII]*y[IDX_CO2I] + k[816]*y[IDX_OII]*y[IDX_HCOI] -
        k[838]*y[IDX_OHII]*y[IDX_COI] + k[842]*y[IDX_OHII]*y[IDX_HCOI] +
        k[854]*y[IDX_OHI]*y[IDX_HCOII] + k[861]*y[IDX_SiI]*y[IDX_HCOII] +
        k[864]*y[IDX_CI]*y[IDX_HCOI] + k[872]*y[IDX_CI]*y[IDX_NOI] +
        k[873]*y[IDX_CI]*y[IDX_O2I] + k[874]*y[IDX_CI]*y[IDX_OCNI] +
        k[875]*y[IDX_CI]*y[IDX_OHI] + k[882]*y[IDX_CH2I]*y[IDX_HCOI] +
        k[891]*y[IDX_CH2I]*y[IDX_O2I] + k[894]*y[IDX_CH2I]*y[IDX_OI] +
        k[895]*y[IDX_CH2I]*y[IDX_OI] + k[905]*y[IDX_CH3I]*y[IDX_HCOI] +
        k[915]*y[IDX_CH3I]*y[IDX_OI] + k[923]*y[IDX_CHI]*y[IDX_CO2I] +
        k[925]*y[IDX_CHI]*y[IDX_HCOI] + k[934]*y[IDX_CHI]*y[IDX_O2I] +
        k[935]*y[IDX_CHI]*y[IDX_O2I] + k[939]*y[IDX_CHI]*y[IDX_OI] +
        k[943]*y[IDX_CNI]*y[IDX_HCOI] + k[946]*y[IDX_CNI]*y[IDX_NOI] +
        k[948]*y[IDX_CNI]*y[IDX_O2I] - k[951]*y[IDX_COI]*y[IDX_HNOI] -
        k[952]*y[IDX_COI]*y[IDX_NO2I] - k[953]*y[IDX_COI]*y[IDX_O2I] -
        k[954]*y[IDX_COI]*y[IDX_O2HI] + k[971]*y[IDX_HI]*y[IDX_CO2I] -
        k[972]*y[IDX_HI]*y[IDX_COI] + k[977]*y[IDX_HI]*y[IDX_HCOI] +
        k[994]*y[IDX_HI]*y[IDX_OCNI] + k[997]*y[IDX_HCOI]*y[IDX_HCOI] +
        k[997]*y[IDX_HCOI]*y[IDX_HCOI] + k[998]*y[IDX_HCOI]*y[IDX_HCOI] +
        k[1000]*y[IDX_HCOI]*y[IDX_NOI] + k[1002]*y[IDX_HCOI]*y[IDX_O2I] +
        k[1004]*y[IDX_HNCOI]*y[IDX_CI] + k[1012]*y[IDX_NI]*y[IDX_CO2I] +
        k[1014]*y[IDX_NI]*y[IDX_HCOI] + k[1055]*y[IDX_O2I]*y[IDX_OCNI] +
        k[1057]*y[IDX_OI]*y[IDX_CNI] + k[1059]*y[IDX_OI]*y[IDX_CO2I] +
        k[1064]*y[IDX_OI]*y[IDX_HCNI] + k[1067]*y[IDX_OI]*y[IDX_HCOI] +
        k[1078]*y[IDX_OI]*y[IDX_OCNI] + k[1081]*y[IDX_OI]*y[IDX_SiC2I] +
        k[1082]*y[IDX_OI]*y[IDX_SiC3I] + k[1083]*y[IDX_OI]*y[IDX_SiCI] -
        k[1092]*y[IDX_OHI]*y[IDX_COI] + k[1095]*y[IDX_OHI]*y[IDX_HCNI] +
        k[1096]*y[IDX_OHI]*y[IDX_HCOI] + k[1103]*y[IDX_SiI]*y[IDX_CO2I] -
        k[1104]*y[IDX_SiI]*y[IDX_COI] + k[1132]*y[IDX_CO2I] - k[1133]*y[IDX_COI]
        + k[1136]*y[IDX_H2COI] + k[1137]*y[IDX_H2COI] + k[1149]*y[IDX_HCOI] +
        k[1152]*y[IDX_HNCOI] + k[1196]*y[IDX_CI]*y[IDX_OI] - k[1227]*y[IDX_COI]
        - k[1251]*y[IDX_COI] - k[1304]*y[IDX_COI] + k[1331]*y[IDX_GCOI] +
        k[1332]*y[IDX_GCOI] + k[1333]*y[IDX_GCOI] + k[1334]*y[IDX_GCOI];
    ydot[IDX_COII] = 0.0 - k[28]*y[IDX_CI]*y[IDX_COII] -
        k[38]*y[IDX_CH2I]*y[IDX_COII] - k[52]*y[IDX_CH4I]*y[IDX_COII] -
        k[54]*y[IDX_CHI]*y[IDX_COII] + k[63]*y[IDX_CNII]*y[IDX_COI] -
        k[70]*y[IDX_COII]*y[IDX_H2COI] - k[71]*y[IDX_COII]*y[IDX_HCOI] -
        k[72]*y[IDX_COII]*y[IDX_NOI] - k[73]*y[IDX_COII]*y[IDX_O2I] +
        k[74]*y[IDX_COI]*y[IDX_N2II] + k[104]*y[IDX_H2II]*y[IDX_COI] -
        k[123]*y[IDX_H2OI]*y[IDX_COII] - k[127]*y[IDX_HI]*y[IDX_COII] -
        k[134]*y[IDX_HCNI]*y[IDX_COII] + k[158]*y[IDX_NII]*y[IDX_COI] -
        k[184]*y[IDX_NH2I]*y[IDX_COII] - k[193]*y[IDX_NH3I]*y[IDX_COII] -
        k[200]*y[IDX_NHI]*y[IDX_COII] + k[208]*y[IDX_OII]*y[IDX_COI] -
        k[217]*y[IDX_OI]*y[IDX_COII] - k[226]*y[IDX_OHI]*y[IDX_COII] +
        k[232]*y[IDX_COI] - k[304]*y[IDX_COII]*y[IDX_EM] +
        k[367]*y[IDX_CII]*y[IDX_CO2I] + k[376]*y[IDX_CII]*y[IDX_O2I] +
        k[378]*y[IDX_CII]*y[IDX_OCNI] + k[379]*y[IDX_CII]*y[IDX_OHI] +
        k[391]*y[IDX_CI]*y[IDX_O2II] + k[411]*y[IDX_CHII]*y[IDX_O2I] +
        k[414]*y[IDX_CHII]*y[IDX_OI] + k[415]*y[IDX_CHII]*y[IDX_OHI] -
        k[422]*y[IDX_CH2I]*y[IDX_COII] - k[451]*y[IDX_CH4I]*y[IDX_COII] -
        k[458]*y[IDX_CHI]*y[IDX_COII] + k[472]*y[IDX_CHI]*y[IDX_OII] -
        k[484]*y[IDX_COII]*y[IDX_H2COI] + k[497]*y[IDX_HII]*y[IDX_H2COI] +
        k[500]*y[IDX_HII]*y[IDX_HCOI] - k[532]*y[IDX_H2I]*y[IDX_COII] -
        k[533]*y[IDX_H2I]*y[IDX_COII] - k[562]*y[IDX_H2OI]*y[IDX_COII] +
        k[665]*y[IDX_HeII]*y[IDX_CO2I] + k[670]*y[IDX_HeII]*y[IDX_H2COI] +
        k[680]*y[IDX_HeII]*y[IDX_HCOI] + k[719]*y[IDX_NII]*y[IDX_CO2I] -
        k[779]*y[IDX_NH2I]*y[IDX_COII] - k[792]*y[IDX_NH3I]*y[IDX_COII] -
        k[795]*y[IDX_NHI]*y[IDX_COII] - k[851]*y[IDX_OHI]*y[IDX_COII] -
        k[1131]*y[IDX_COII] + k[1148]*y[IDX_HCOII] +
        k[1193]*y[IDX_CII]*y[IDX_OI] + k[1195]*y[IDX_CI]*y[IDX_OII] -
        k[1275]*y[IDX_COII];
    ydot[IDX_CO2I] = 0.0 - k[252]*y[IDX_CO2I] +
        k[331]*y[IDX_HCO2II]*y[IDX_EM] - k[367]*y[IDX_CII]*y[IDX_CO2I] +
        k[387]*y[IDX_CI]*y[IDX_HCO2II] - k[398]*y[IDX_CHII]*y[IDX_CO2I] -
        k[416]*y[IDX_CH2II]*y[IDX_CO2I] - k[447]*y[IDX_CH4II]*y[IDX_CO2I] +
        k[485]*y[IDX_COI]*y[IDX_HCO2II] + k[490]*y[IDX_COI]*y[IDX_SiOII] -
        k[496]*y[IDX_HII]*y[IDX_CO2I] - k[514]*y[IDX_H2II]*y[IDX_CO2I] +
        k[567]*y[IDX_H2OI]*y[IDX_HCO2II] - k[582]*y[IDX_H3II]*y[IDX_CO2I] -
        k[620]*y[IDX_HCNII]*y[IDX_CO2I] - k[652]*y[IDX_HNOII]*y[IDX_CO2I] -
        k[665]*y[IDX_HeII]*y[IDX_CO2I] - k[666]*y[IDX_HeII]*y[IDX_CO2I] -
        k[667]*y[IDX_HeII]*y[IDX_CO2I] - k[668]*y[IDX_HeII]*y[IDX_CO2I] -
        k[719]*y[IDX_NII]*y[IDX_CO2I] - k[734]*y[IDX_N2HII]*y[IDX_CO2I] -
        k[748]*y[IDX_NHII]*y[IDX_CO2I] - k[749]*y[IDX_NHII]*y[IDX_CO2I] -
        k[750]*y[IDX_NHII]*y[IDX_CO2I] - k[812]*y[IDX_OII]*y[IDX_CO2I] -
        k[821]*y[IDX_O2HII]*y[IDX_CO2I] - k[837]*y[IDX_OHII]*y[IDX_CO2I] +
        k[889]*y[IDX_CH2I]*y[IDX_O2I] + k[890]*y[IDX_CH2I]*y[IDX_O2I] -
        k[923]*y[IDX_CHI]*y[IDX_CO2I] + k[933]*y[IDX_CHI]*y[IDX_O2I] +
        k[951]*y[IDX_COI]*y[IDX_HNOI] + k[952]*y[IDX_COI]*y[IDX_NO2I] +
        k[953]*y[IDX_COI]*y[IDX_O2I] + k[954]*y[IDX_COI]*y[IDX_O2HI] -
        k[971]*y[IDX_HI]*y[IDX_CO2I] + k[1001]*y[IDX_HCOI]*y[IDX_O2I] -
        k[1012]*y[IDX_NI]*y[IDX_CO2I] + k[1053]*y[IDX_NOI]*y[IDX_OCNI] +
        k[1054]*y[IDX_O2I]*y[IDX_OCNI] - k[1059]*y[IDX_OI]*y[IDX_CO2I] +
        k[1066]*y[IDX_OI]*y[IDX_HCOI] + k[1092]*y[IDX_OHI]*y[IDX_COI] -
        k[1103]*y[IDX_SiI]*y[IDX_CO2I] - k[1132]*y[IDX_CO2I] -
        k[1258]*y[IDX_CO2I] + k[1383]*y[IDX_GCO2I] + k[1384]*y[IDX_GCO2I] +
        k[1385]*y[IDX_GCO2I] + k[1386]*y[IDX_GCO2I];
    ydot[IDX_EM] = 0.0 + k[0]*y[IDX_CHI]*y[IDX_OI] -
        k[8]*y[IDX_H2I]*y[IDX_EM] + k[8]*y[IDX_H2I]*y[IDX_EM] + k[231]*y[IDX_CI]
        + k[232]*y[IDX_COI] + k[233]*y[IDX_H2I] + k[234]*y[IDX_H2I] +
        k[236]*y[IDX_HI] + k[237]*y[IDX_HeI] + k[238]*y[IDX_NI] +
        k[239]*y[IDX_OI] + k[240]*y[IDX_CI] + k[242]*y[IDX_CH2I] +
        k[245]*y[IDX_CH3I] + k[258]*y[IDX_HI] + k[261]*y[IDX_HCOI] +
        k[265]*y[IDX_HeI] + k[266]*y[IDX_MgI] + k[268]*y[IDX_NI] +
        k[269]*y[IDX_NH2I] + k[272]*y[IDX_NH3I] + k[275]*y[IDX_NHI] +
        k[277]*y[IDX_NOI] + k[279]*y[IDX_O2I] + k[282]*y[IDX_OI] +
        k[285]*y[IDX_SiI] - k[294]*y[IDX_CHII]*y[IDX_EM] -
        k[295]*y[IDX_CH2II]*y[IDX_EM] - k[296]*y[IDX_CH2II]*y[IDX_EM] -
        k[297]*y[IDX_CH2II]*y[IDX_EM] - k[298]*y[IDX_CH3II]*y[IDX_EM] -
        k[299]*y[IDX_CH3II]*y[IDX_EM] - k[300]*y[IDX_CH3II]*y[IDX_EM] -
        k[301]*y[IDX_CH4II]*y[IDX_EM] - k[302]*y[IDX_CH4II]*y[IDX_EM] -
        k[303]*y[IDX_CNII]*y[IDX_EM] - k[304]*y[IDX_COII]*y[IDX_EM] -
        k[305]*y[IDX_H2II]*y[IDX_EM] - k[306]*y[IDX_H2COII]*y[IDX_EM] -
        k[307]*y[IDX_H2COII]*y[IDX_EM] - k[308]*y[IDX_H2COII]*y[IDX_EM] -
        k[309]*y[IDX_H2COII]*y[IDX_EM] - k[310]*y[IDX_H2NOII]*y[IDX_EM] -
        k[311]*y[IDX_H2NOII]*y[IDX_EM] - k[312]*y[IDX_H2OII]*y[IDX_EM] -
        k[313]*y[IDX_H2OII]*y[IDX_EM] - k[314]*y[IDX_H2OII]*y[IDX_EM] -
        k[315]*y[IDX_H3II]*y[IDX_EM] - k[316]*y[IDX_H3II]*y[IDX_EM] -
        k[317]*y[IDX_H3COII]*y[IDX_EM] - k[318]*y[IDX_H3COII]*y[IDX_EM] -
        k[319]*y[IDX_H3COII]*y[IDX_EM] - k[320]*y[IDX_H3COII]*y[IDX_EM] -
        k[321]*y[IDX_H3COII]*y[IDX_EM] - k[322]*y[IDX_H3OII]*y[IDX_EM] -
        k[323]*y[IDX_H3OII]*y[IDX_EM] - k[324]*y[IDX_H3OII]*y[IDX_EM] -
        k[325]*y[IDX_H3OII]*y[IDX_EM] - k[326]*y[IDX_HCNII]*y[IDX_EM] -
        k[327]*y[IDX_HCNHII]*y[IDX_EM] - k[328]*y[IDX_HCNHII]*y[IDX_EM] -
        k[329]*y[IDX_HCNHII]*y[IDX_EM] - k[330]*y[IDX_HCOII]*y[IDX_EM] -
        k[331]*y[IDX_HCO2II]*y[IDX_EM] - k[332]*y[IDX_HCO2II]*y[IDX_EM] -
        k[333]*y[IDX_HCO2II]*y[IDX_EM] - k[334]*y[IDX_HNOII]*y[IDX_EM] -
        k[335]*y[IDX_HOCII]*y[IDX_EM] - k[336]*y[IDX_HeHII]*y[IDX_EM] -
        k[337]*y[IDX_N2II]*y[IDX_EM] - k[338]*y[IDX_N2HII]*y[IDX_EM] -
        k[339]*y[IDX_N2HII]*y[IDX_EM] - k[340]*y[IDX_NHII]*y[IDX_EM] -
        k[341]*y[IDX_NH2II]*y[IDX_EM] - k[342]*y[IDX_NH2II]*y[IDX_EM] -
        k[343]*y[IDX_NH3II]*y[IDX_EM] - k[344]*y[IDX_NH3II]*y[IDX_EM] -
        k[345]*y[IDX_NOII]*y[IDX_EM] - k[346]*y[IDX_O2II]*y[IDX_EM] -
        k[347]*y[IDX_O2HII]*y[IDX_EM] - k[348]*y[IDX_OHII]*y[IDX_EM] -
        k[349]*y[IDX_SiCII]*y[IDX_EM] - k[350]*y[IDX_SiC2II]*y[IDX_EM] -
        k[351]*y[IDX_SiC3II]*y[IDX_EM] - k[352]*y[IDX_SiHII]*y[IDX_EM] -
        k[353]*y[IDX_SiH2II]*y[IDX_EM] - k[354]*y[IDX_SiH2II]*y[IDX_EM] -
        k[355]*y[IDX_SiH2II]*y[IDX_EM] - k[356]*y[IDX_SiH3II]*y[IDX_EM] -
        k[357]*y[IDX_SiH3II]*y[IDX_EM] - k[358]*y[IDX_SiH4II]*y[IDX_EM] -
        k[359]*y[IDX_SiH4II]*y[IDX_EM] - k[360]*y[IDX_SiH5II]*y[IDX_EM] -
        k[361]*y[IDX_SiH5II]*y[IDX_EM] - k[362]*y[IDX_SiOII]*y[IDX_EM] -
        k[363]*y[IDX_SiOHII]*y[IDX_EM] - k[364]*y[IDX_SiOHII]*y[IDX_EM] +
        k[1107]*y[IDX_CI] + k[1112]*y[IDX_CH2I] + k[1117]*y[IDX_CH3I] +
        k[1120]*y[IDX_CH3OHI] + k[1126]*y[IDX_CH4I] + k[1129]*y[IDX_CHI] +
        k[1138]*y[IDX_H2COI] + k[1139]*y[IDX_H2COI] + k[1141]*y[IDX_H2OI] +
        k[1150]*y[IDX_HCOI] + k[1154]*y[IDX_MgI] + k[1157]*y[IDX_NH2I] +
        k[1160]*y[IDX_NH3I] + k[1163]*y[IDX_NHI] + k[1165]*y[IDX_NOI] +
        k[1168]*y[IDX_O2I] + k[1175]*y[IDX_OHI] + k[1176]*y[IDX_SiI] +
        k[1180]*y[IDX_SiH2I] + k[1183]*y[IDX_SiH3I] + k[1191]*y[IDX_SiOI] -
        k[1214]*y[IDX_CII]*y[IDX_EM] - k[1215]*y[IDX_CH3II]*y[IDX_EM] -
        k[1216]*y[IDX_HII]*y[IDX_EM] - k[1217]*y[IDX_H2COII]*y[IDX_EM] -
        k[1218]*y[IDX_HeII]*y[IDX_EM] - k[1219]*y[IDX_MgII]*y[IDX_EM] -
        k[1220]*y[IDX_NII]*y[IDX_EM] - k[1221]*y[IDX_OII]*y[IDX_EM] -
        k[1222]*y[IDX_SiII]*y[IDX_EM] - k[1306]*y[IDX_EM];
    ydot[IDX_HI] = 0.0 + k[2]*y[IDX_H2I]*y[IDX_CHI] +
        k[3]*y[IDX_H2I]*y[IDX_H2I] + k[3]*y[IDX_H2I]*y[IDX_H2I] +
        k[4]*y[IDX_H2I]*y[IDX_H2OI] + k[7]*y[IDX_H2I]*y[IDX_OHI] +
        k[8]*y[IDX_H2I]*y[IDX_EM] + k[8]*y[IDX_H2I]*y[IDX_EM] -
        k[9]*y[IDX_HI]*y[IDX_CHI] + k[9]*y[IDX_HI]*y[IDX_CHI] +
        k[9]*y[IDX_HI]*y[IDX_CHI] - k[10]*y[IDX_HI]*y[IDX_H2I] +
        k[10]*y[IDX_HI]*y[IDX_H2I] + k[10]*y[IDX_HI]*y[IDX_H2I] +
        k[10]*y[IDX_HI]*y[IDX_H2I] - k[11]*y[IDX_HI]*y[IDX_H2OI] +
        k[11]*y[IDX_HI]*y[IDX_H2OI] + k[11]*y[IDX_HI]*y[IDX_H2OI] -
        k[12]*y[IDX_HI]*y[IDX_O2I] + k[12]*y[IDX_HI]*y[IDX_O2I] -
        k[13]*y[IDX_HI]*y[IDX_OHI] + k[13]*y[IDX_HI]*y[IDX_OHI] +
        k[13]*y[IDX_HI]*y[IDX_OHI] + k[75]*y[IDX_HII]*y[IDX_CH2I] +
        k[76]*y[IDX_HII]*y[IDX_CH3I] + k[77]*y[IDX_HII]*y[IDX_CH4I] +
        k[78]*y[IDX_HII]*y[IDX_CHI] + k[79]*y[IDX_HII]*y[IDX_H2COI] +
        k[80]*y[IDX_HII]*y[IDX_H2OI] + k[81]*y[IDX_HII]*y[IDX_HCNI] +
        k[82]*y[IDX_HII]*y[IDX_HCOI] + k[83]*y[IDX_HII]*y[IDX_MgI] +
        k[84]*y[IDX_HII]*y[IDX_NH2I] + k[85]*y[IDX_HII]*y[IDX_NH3I] +
        k[86]*y[IDX_HII]*y[IDX_NHI] + k[87]*y[IDX_HII]*y[IDX_NOI] +
        k[88]*y[IDX_HII]*y[IDX_O2I] + k[89]*y[IDX_HII]*y[IDX_OI] +
        k[90]*y[IDX_HII]*y[IDX_OHI] + k[91]*y[IDX_HII]*y[IDX_SiI] +
        k[92]*y[IDX_HII]*y[IDX_SiC2I] + k[93]*y[IDX_HII]*y[IDX_SiC3I] +
        k[94]*y[IDX_HII]*y[IDX_SiCI] + k[95]*y[IDX_HII]*y[IDX_SiH2I] +
        k[96]*y[IDX_HII]*y[IDX_SiH3I] + k[97]*y[IDX_HII]*y[IDX_SiH4I] +
        k[98]*y[IDX_HII]*y[IDX_SiHI] + k[99]*y[IDX_HII]*y[IDX_SiOI] -
        k[126]*y[IDX_HI]*y[IDX_CNII] - k[127]*y[IDX_HI]*y[IDX_COII] -
        k[128]*y[IDX_HI]*y[IDX_H2II] - k[129]*y[IDX_HI]*y[IDX_HCNII] -
        k[130]*y[IDX_HI]*y[IDX_HeII] - k[131]*y[IDX_HI]*y[IDX_OII] +
        k[233]*y[IDX_H2I] + k[235]*y[IDX_H2I] + k[235]*y[IDX_H2I] -
        k[236]*y[IDX_HI] + k[241]*y[IDX_CHII] + k[243]*y[IDX_CH2I] +
        k[244]*y[IDX_CH3I] + k[250]*y[IDX_CHI] + k[254]*y[IDX_H2CNI] +
        k[256]*y[IDX_H2OI] - k[258]*y[IDX_HI] + k[259]*y[IDX_HCNI] +
        k[260]*y[IDX_HCOI] + k[262]*y[IDX_HNCI] + k[264]*y[IDX_HNOI] +
        k[270]*y[IDX_NH2I] + k[271]*y[IDX_NH3I] + k[274]*y[IDX_NHI] +
        k[281]*y[IDX_O2HI] + k[284]*y[IDX_OHI] + k[289]*y[IDX_SiH2I] +
        k[290]*y[IDX_SiH3I] + k[292]*y[IDX_SiHI] + k[294]*y[IDX_CHII]*y[IDX_EM]
        + k[296]*y[IDX_CH2II]*y[IDX_EM] + k[296]*y[IDX_CH2II]*y[IDX_EM] +
        k[297]*y[IDX_CH2II]*y[IDX_EM] + k[298]*y[IDX_CH3II]*y[IDX_EM] +
        k[300]*y[IDX_CH3II]*y[IDX_EM] + k[300]*y[IDX_CH3II]*y[IDX_EM] +
        k[301]*y[IDX_CH4II]*y[IDX_EM] + k[301]*y[IDX_CH4II]*y[IDX_EM] +
        k[302]*y[IDX_CH4II]*y[IDX_EM] + k[305]*y[IDX_H2II]*y[IDX_EM] +
        k[305]*y[IDX_H2II]*y[IDX_EM] + k[308]*y[IDX_H2COII]*y[IDX_EM] +
        k[308]*y[IDX_H2COII]*y[IDX_EM] + k[309]*y[IDX_H2COII]*y[IDX_EM] +
        k[310]*y[IDX_H2NOII]*y[IDX_EM] + k[313]*y[IDX_H2OII]*y[IDX_EM] +
        k[313]*y[IDX_H2OII]*y[IDX_EM] + k[314]*y[IDX_H2OII]*y[IDX_EM] +
        k[315]*y[IDX_H3II]*y[IDX_EM] + k[316]*y[IDX_H3II]*y[IDX_EM] +
        k[316]*y[IDX_H3II]*y[IDX_EM] + k[316]*y[IDX_H3II]*y[IDX_EM] +
        k[319]*y[IDX_H3COII]*y[IDX_EM] + k[320]*y[IDX_H3COII]*y[IDX_EM] +
        k[321]*y[IDX_H3COII]*y[IDX_EM] + k[321]*y[IDX_H3COII]*y[IDX_EM] +
        k[322]*y[IDX_H3OII]*y[IDX_EM] + k[323]*y[IDX_H3OII]*y[IDX_EM] +
        k[325]*y[IDX_H3OII]*y[IDX_EM] + k[325]*y[IDX_H3OII]*y[IDX_EM] +
        k[326]*y[IDX_HCNII]*y[IDX_EM] + k[327]*y[IDX_HCNHII]*y[IDX_EM] +
        k[327]*y[IDX_HCNHII]*y[IDX_EM] + k[328]*y[IDX_HCNHII]*y[IDX_EM] +
        k[329]*y[IDX_HCNHII]*y[IDX_EM] + k[330]*y[IDX_HCOII]*y[IDX_EM] +
        k[331]*y[IDX_HCO2II]*y[IDX_EM] + k[332]*y[IDX_HCO2II]*y[IDX_EM] +
        k[334]*y[IDX_HNOII]*y[IDX_EM] + k[335]*y[IDX_HOCII]*y[IDX_EM] +
        k[336]*y[IDX_HeHII]*y[IDX_EM] + k[338]*y[IDX_N2HII]*y[IDX_EM] +
        k[340]*y[IDX_NHII]*y[IDX_EM] + k[341]*y[IDX_NH2II]*y[IDX_EM] +
        k[341]*y[IDX_NH2II]*y[IDX_EM] + k[342]*y[IDX_NH2II]*y[IDX_EM] +
        k[343]*y[IDX_NH3II]*y[IDX_EM] + k[344]*y[IDX_NH3II]*y[IDX_EM] +
        k[344]*y[IDX_NH3II]*y[IDX_EM] + k[347]*y[IDX_O2HII]*y[IDX_EM] +
        k[348]*y[IDX_OHII]*y[IDX_EM] + k[352]*y[IDX_SiHII]*y[IDX_EM] +
        k[354]*y[IDX_SiH2II]*y[IDX_EM] + k[354]*y[IDX_SiH2II]*y[IDX_EM] +
        k[355]*y[IDX_SiH2II]*y[IDX_EM] + k[356]*y[IDX_SiH3II]*y[IDX_EM] +
        k[359]*y[IDX_SiH4II]*y[IDX_EM] + k[361]*y[IDX_SiH5II]*y[IDX_EM] +
        k[364]*y[IDX_SiOHII]*y[IDX_EM] + k[370]*y[IDX_CII]*y[IDX_H2OI] +
        k[371]*y[IDX_CII]*y[IDX_H2OI] + k[373]*y[IDX_CII]*y[IDX_NH2I] +
        k[375]*y[IDX_CII]*y[IDX_NHI] + k[379]*y[IDX_CII]*y[IDX_OHI] +
        k[381]*y[IDX_CII]*y[IDX_SiHI] + k[394]*y[IDX_CI]*y[IDX_SiHII] +
        k[402]*y[IDX_CHII]*y[IDX_H2OI] + k[408]*y[IDX_CHII]*y[IDX_NI] +
        k[414]*y[IDX_CHII]*y[IDX_OI] + k[418]*y[IDX_CH2II]*y[IDX_H2OI] +
        k[421]*y[IDX_CH2II]*y[IDX_OI] + k[443]*y[IDX_CH3II]*y[IDX_OI] +
        k[456]*y[IDX_CH4I]*y[IDX_N2II] + k[468]*y[IDX_CHI]*y[IDX_NII] +
        k[472]*y[IDX_CHI]*y[IDX_OII] + k[476]*y[IDX_CHI]*y[IDX_SiII] +
        k[497]*y[IDX_HII]*y[IDX_H2COI] + k[509]*y[IDX_H2II]*y[IDX_CI] +
        k[510]*y[IDX_H2II]*y[IDX_CH2I] + k[511]*y[IDX_H2II]*y[IDX_CH4I] +
        k[512]*y[IDX_H2II]*y[IDX_CHI] + k[513]*y[IDX_H2II]*y[IDX_CNI] +
        k[514]*y[IDX_H2II]*y[IDX_CO2I] + k[515]*y[IDX_H2II]*y[IDX_COI] +
        k[516]*y[IDX_H2II]*y[IDX_H2I] + k[517]*y[IDX_H2II]*y[IDX_H2COI] +
        k[518]*y[IDX_H2II]*y[IDX_H2OI] + k[520]*y[IDX_H2II]*y[IDX_HeI] +
        k[521]*y[IDX_H2II]*y[IDX_N2I] + k[522]*y[IDX_H2II]*y[IDX_NI] +
        k[523]*y[IDX_H2II]*y[IDX_NHI] + k[524]*y[IDX_H2II]*y[IDX_NOI] +
        k[525]*y[IDX_H2II]*y[IDX_O2I] + k[526]*y[IDX_H2II]*y[IDX_OI] +
        k[527]*y[IDX_H2II]*y[IDX_OHI] + k[528]*y[IDX_H2I]*y[IDX_CII] +
        k[529]*y[IDX_H2I]*y[IDX_CHII] + k[530]*y[IDX_H2I]*y[IDX_CH2II] +
        k[531]*y[IDX_H2I]*y[IDX_CNII] + k[532]*y[IDX_H2I]*y[IDX_COII] +
        k[533]*y[IDX_H2I]*y[IDX_COII] + k[534]*y[IDX_H2I]*y[IDX_H2OII] +
        k[535]*y[IDX_H2I]*y[IDX_HCNII] + k[536]*y[IDX_H2I]*y[IDX_HeII] +
        k[538]*y[IDX_H2I]*y[IDX_NII] + k[539]*y[IDX_H2I]*y[IDX_N2II] +
        k[541]*y[IDX_H2I]*y[IDX_NHII] + k[542]*y[IDX_H2I]*y[IDX_NH2II] +
        k[543]*y[IDX_H2I]*y[IDX_OII] + k[545]*y[IDX_H2I]*y[IDX_OHII] +
        k[546]*y[IDX_H2I]*y[IDX_SiH4II] + k[547]*y[IDX_H2I]*y[IDX_SiOII] +
        k[551]*y[IDX_H2COI]*y[IDX_O2II] + k[572]*y[IDX_H2OI]*y[IDX_SiII] +
        k[591]*y[IDX_H3II]*y[IDX_MgI] + k[598]*y[IDX_H3II]*y[IDX_OI] -
        k[614]*y[IDX_HI]*y[IDX_CHII] - k[615]*y[IDX_HI]*y[IDX_CH2II] -
        k[616]*y[IDX_HI]*y[IDX_CH3II] - k[617]*y[IDX_HI]*y[IDX_CH4II] -
        k[618]*y[IDX_HI]*y[IDX_HeHII] - k[619]*y[IDX_HI]*y[IDX_SiHII] +
        k[654]*y[IDX_HeII]*y[IDX_CH2I] + k[658]*y[IDX_HeII]*y[IDX_CH4I] +
        k[660]*y[IDX_HeII]*y[IDX_CH4I] + k[662]*y[IDX_HeII]*y[IDX_CHI] +
        k[671]*y[IDX_HeII]*y[IDX_H2COI] + k[673]*y[IDX_HeII]*y[IDX_H2OI] +
        k[675]*y[IDX_HeII]*y[IDX_H2SiOI] + k[676]*y[IDX_HeII]*y[IDX_HCNI] +
        k[678]*y[IDX_HeII]*y[IDX_HCNI] + k[680]*y[IDX_HeII]*y[IDX_HCOI] +
        k[683]*y[IDX_HeII]*y[IDX_HNCI] + k[684]*y[IDX_HeII]*y[IDX_HNCI] +
        k[686]*y[IDX_HeII]*y[IDX_HNOI] + k[690]*y[IDX_HeII]*y[IDX_NH2I] +
        k[692]*y[IDX_HeII]*y[IDX_NH3I] + k[693]*y[IDX_HeII]*y[IDX_NHI] +
        k[699]*y[IDX_HeII]*y[IDX_OHI] + k[704]*y[IDX_HeII]*y[IDX_SiH2I] +
        k[706]*y[IDX_HeII]*y[IDX_SiH3I] + k[708]*y[IDX_HeII]*y[IDX_SiH4I] +
        k[709]*y[IDX_HeII]*y[IDX_SiHI] + k[712]*y[IDX_NII]*y[IDX_CH3OHI] +
        k[714]*y[IDX_NII]*y[IDX_CH3OHI] + k[715]*y[IDX_NII]*y[IDX_CH3OHI] +
        k[716]*y[IDX_NII]*y[IDX_CH4I] + k[717]*y[IDX_NII]*y[IDX_CH4I] +
        k[718]*y[IDX_NII]*y[IDX_CH4I] + k[718]*y[IDX_NII]*y[IDX_CH4I] +
        k[726]*y[IDX_NII]*y[IDX_NHI] + k[730]*y[IDX_N2II]*y[IDX_H2COI] +
        k[736]*y[IDX_NI]*y[IDX_CH2II] + k[738]*y[IDX_NI]*y[IDX_H2OII] +
        k[740]*y[IDX_NI]*y[IDX_NHII] + k[741]*y[IDX_NI]*y[IDX_NH2II] +
        k[743]*y[IDX_NI]*y[IDX_OHII] + k[803]*y[IDX_NHI]*y[IDX_OII] +
        k[819]*y[IDX_OII]*y[IDX_OHI] + k[820]*y[IDX_O2II]*y[IDX_CH3OHI] +
        k[827]*y[IDX_OI]*y[IDX_NH2II] + k[830]*y[IDX_OI]*y[IDX_OHII] +
        k[832]*y[IDX_OI]*y[IDX_SiHII] + k[833]*y[IDX_OI]*y[IDX_SiH2II] +
        k[855]*y[IDX_OHI]*y[IDX_HCOII] + k[859]*y[IDX_OHI]*y[IDX_SiII] +
        k[866]*y[IDX_CI]*y[IDX_NH2I] + k[867]*y[IDX_CI]*y[IDX_NH2I] +
        k[869]*y[IDX_CI]*y[IDX_NHI] + k[875]*y[IDX_CI]*y[IDX_OHI] +
        k[877]*y[IDX_CI]*y[IDX_SiHI] + k[888]*y[IDX_CH2I]*y[IDX_NOI] +
        k[890]*y[IDX_CH2I]*y[IDX_O2I] + k[890]*y[IDX_CH2I]*y[IDX_O2I] +
        k[895]*y[IDX_CH2I]*y[IDX_OI] + k[895]*y[IDX_CH2I]*y[IDX_OI] +
        k[896]*y[IDX_CH2I]*y[IDX_OI] + k[898]*y[IDX_CH2I]*y[IDX_OHI] +
        k[915]*y[IDX_CH3I]*y[IDX_OI] + k[916]*y[IDX_CH3I]*y[IDX_OI] +
        k[928]*y[IDX_CHI]*y[IDX_NI] + k[932]*y[IDX_CHI]*y[IDX_NOI] +
        k[933]*y[IDX_CHI]*y[IDX_O2I] + k[934]*y[IDX_CHI]*y[IDX_O2I] +
        k[939]*y[IDX_CHI]*y[IDX_OI] + k[941]*y[IDX_CHI]*y[IDX_OHI] +
        k[955]*y[IDX_H2I]*y[IDX_CI] + k[956]*y[IDX_H2I]*y[IDX_CH2I] +
        k[957]*y[IDX_H2I]*y[IDX_CH3I] + k[958]*y[IDX_H2I]*y[IDX_CHI] +
        k[959]*y[IDX_H2I]*y[IDX_CNI] + k[960]*y[IDX_H2I]*y[IDX_NI] +
        k[961]*y[IDX_H2I]*y[IDX_NH2I] + k[962]*y[IDX_H2I]*y[IDX_NHI] +
        k[963]*y[IDX_H2I]*y[IDX_O2I] + k[965]*y[IDX_H2I]*y[IDX_OI] +
        k[966]*y[IDX_H2I]*y[IDX_OHI] - k[967]*y[IDX_HI]*y[IDX_CH2I] -
        k[968]*y[IDX_HI]*y[IDX_CH3I] - k[969]*y[IDX_HI]*y[IDX_CH4I] -
        k[970]*y[IDX_HI]*y[IDX_CHI] - k[971]*y[IDX_HI]*y[IDX_CO2I] -
        k[972]*y[IDX_HI]*y[IDX_COI] - k[973]*y[IDX_HI]*y[IDX_H2CNI] -
        k[974]*y[IDX_HI]*y[IDX_H2COI] - k[975]*y[IDX_HI]*y[IDX_H2OI] -
        k[976]*y[IDX_HI]*y[IDX_HCNI] - k[977]*y[IDX_HI]*y[IDX_HCOI] -
        k[978]*y[IDX_HI]*y[IDX_HCOI] - k[979]*y[IDX_HI]*y[IDX_HNCI] +
        k[979]*y[IDX_HI]*y[IDX_HNCI] - k[980]*y[IDX_HI]*y[IDX_HNOI] -
        k[981]*y[IDX_HI]*y[IDX_HNOI] - k[982]*y[IDX_HI]*y[IDX_HNOI] -
        k[983]*y[IDX_HI]*y[IDX_NH2I] - k[984]*y[IDX_HI]*y[IDX_NH3I] -
        k[985]*y[IDX_HI]*y[IDX_NHI] - k[986]*y[IDX_HI]*y[IDX_NO2I] -
        k[987]*y[IDX_HI]*y[IDX_NOI] - k[988]*y[IDX_HI]*y[IDX_NOI] -
        k[989]*y[IDX_HI]*y[IDX_O2I] - k[990]*y[IDX_HI]*y[IDX_O2HI] -
        k[991]*y[IDX_HI]*y[IDX_O2HI] - k[992]*y[IDX_HI]*y[IDX_O2HI] -
        k[993]*y[IDX_HI]*y[IDX_OCNI] - k[994]*y[IDX_HI]*y[IDX_OCNI] -
        k[995]*y[IDX_HI]*y[IDX_OCNI] - k[996]*y[IDX_HI]*y[IDX_OHI] +
        k[1005]*y[IDX_NI]*y[IDX_CH2I] + k[1006]*y[IDX_NI]*y[IDX_CH2I] +
        k[1008]*y[IDX_NI]*y[IDX_CH3I] + k[1010]*y[IDX_NI]*y[IDX_CH3I] +
        k[1010]*y[IDX_NI]*y[IDX_CH3I] + k[1016]*y[IDX_NI]*y[IDX_HCOI] +
        k[1018]*y[IDX_NI]*y[IDX_NHI] + k[1025]*y[IDX_NI]*y[IDX_OHI] +
        k[1030]*y[IDX_NH2I]*y[IDX_NOI] + k[1039]*y[IDX_NHI]*y[IDX_NHI] +
        k[1039]*y[IDX_NHI]*y[IDX_NHI] + k[1042]*y[IDX_NHI]*y[IDX_NOI] +
        k[1046]*y[IDX_NHI]*y[IDX_OI] + k[1049]*y[IDX_NHI]*y[IDX_OHI] +
        k[1065]*y[IDX_OI]*y[IDX_HCNI] + k[1066]*y[IDX_OI]*y[IDX_HCOI] +
        k[1068]*y[IDX_OI]*y[IDX_HNOI] + k[1072]*y[IDX_OI]*y[IDX_NH2I] +
        k[1080]*y[IDX_OI]*y[IDX_OHI] + k[1086]*y[IDX_OI]*y[IDX_SiH2I] +
        k[1086]*y[IDX_OI]*y[IDX_SiH2I] + k[1087]*y[IDX_OI]*y[IDX_SiH3I] +
        k[1089]*y[IDX_OI]*y[IDX_SiHI] + k[1091]*y[IDX_OHI]*y[IDX_CNI] +
        k[1092]*y[IDX_OHI]*y[IDX_COI] + k[1099]*y[IDX_OHI]*y[IDX_NOI] +
        k[1102]*y[IDX_OHI]*y[IDX_SiI] + k[1110]*y[IDX_CH2II] +
        k[1113]*y[IDX_CH2I] + k[1115]*y[IDX_CH3II] + k[1116]*y[IDX_CH3I] +
        k[1120]*y[IDX_CH3OHI] + k[1123]*y[IDX_CH4II] + k[1125]*y[IDX_CH4I] +
        k[1127]*y[IDX_CH4I] + k[1128]*y[IDX_CHI] + k[1134]*y[IDX_H2II] +
        k[1135]*y[IDX_H2CNI] + k[1137]*y[IDX_H2COI] + k[1137]*y[IDX_H2COI] +
        k[1139]*y[IDX_H2COI] + k[1140]*y[IDX_H2OII] + k[1142]*y[IDX_H2OI] +
        k[1144]*y[IDX_H2SiOI] + k[1144]*y[IDX_H2SiOI] + k[1145]*y[IDX_H3II] +
        k[1147]*y[IDX_HCNI] + k[1148]*y[IDX_HCOII] + k[1149]*y[IDX_HCOI] +
        k[1151]*y[IDX_HNCI] + k[1153]*y[IDX_HNOI] + k[1158]*y[IDX_NH2I] +
        k[1159]*y[IDX_NH3I] + k[1162]*y[IDX_NHI] + k[1170]*y[IDX_O2HI] +
        k[1173]*y[IDX_OHII] + k[1174]*y[IDX_OHI] + k[1179]*y[IDX_SiHII] +
        k[1181]*y[IDX_SiH2I] + k[1182]*y[IDX_SiH3I] + k[1186]*y[IDX_SiH4I] +
        k[1187]*y[IDX_SiH4I] + k[1188]*y[IDX_SiHI] -
        k[1197]*y[IDX_HII]*y[IDX_HI] - k[1205]*y[IDX_HI]*y[IDX_CII] -
        k[1206]*y[IDX_HI]*y[IDX_CI] - k[1207]*y[IDX_HI]*y[IDX_OI] -
        k[1208]*y[IDX_HI]*y[IDX_OHI] - k[1209]*y[IDX_HI]*y[IDX_SiII] +
        k[1216]*y[IDX_HII]*y[IDX_EM] + k[1248]*y[IDX_SiH5II] +
        k[1286]*y[IDX_H3OII] + k[1287]*y[IDX_HCO2II] + k[1296]*y[IDX_H2NOII] +
        k[1301]*y[IDX_HCNHII] + k[1302]*y[IDX_N2HII];
    ydot[IDX_HII] = 0.0 - k[1]*y[IDX_HII]*y[IDX_HNCI] +
        k[1]*y[IDX_HII]*y[IDX_HNCI] - k[75]*y[IDX_HII]*y[IDX_CH2I] -
        k[76]*y[IDX_HII]*y[IDX_CH3I] - k[77]*y[IDX_HII]*y[IDX_CH4I] -
        k[78]*y[IDX_HII]*y[IDX_CHI] - k[79]*y[IDX_HII]*y[IDX_H2COI] -
        k[80]*y[IDX_HII]*y[IDX_H2OI] - k[81]*y[IDX_HII]*y[IDX_HCNI] -
        k[82]*y[IDX_HII]*y[IDX_HCOI] - k[83]*y[IDX_HII]*y[IDX_MgI] -
        k[84]*y[IDX_HII]*y[IDX_NH2I] - k[85]*y[IDX_HII]*y[IDX_NH3I] -
        k[86]*y[IDX_HII]*y[IDX_NHI] - k[87]*y[IDX_HII]*y[IDX_NOI] -
        k[88]*y[IDX_HII]*y[IDX_O2I] - k[89]*y[IDX_HII]*y[IDX_OI] -
        k[90]*y[IDX_HII]*y[IDX_OHI] - k[91]*y[IDX_HII]*y[IDX_SiI] -
        k[92]*y[IDX_HII]*y[IDX_SiC2I] - k[93]*y[IDX_HII]*y[IDX_SiC3I] -
        k[94]*y[IDX_HII]*y[IDX_SiCI] - k[95]*y[IDX_HII]*y[IDX_SiH2I] -
        k[96]*y[IDX_HII]*y[IDX_SiH3I] - k[97]*y[IDX_HII]*y[IDX_SiH4I] -
        k[98]*y[IDX_HII]*y[IDX_SiHI] - k[99]*y[IDX_HII]*y[IDX_SiOI] +
        k[126]*y[IDX_HI]*y[IDX_CNII] + k[127]*y[IDX_HI]*y[IDX_COII] +
        k[128]*y[IDX_HI]*y[IDX_H2II] + k[129]*y[IDX_HI]*y[IDX_HCNII] +
        k[130]*y[IDX_HI]*y[IDX_HeII] + k[131]*y[IDX_HI]*y[IDX_OII] +
        k[233]*y[IDX_H2I] + k[236]*y[IDX_HI] + k[258]*y[IDX_HI] -
        k[491]*y[IDX_HII]*y[IDX_CH2I] - k[492]*y[IDX_HII]*y[IDX_CH3OHI] -
        k[493]*y[IDX_HII]*y[IDX_CH3OHI] - k[494]*y[IDX_HII]*y[IDX_CH3OHI] -
        k[495]*y[IDX_HII]*y[IDX_CH4I] - k[496]*y[IDX_HII]*y[IDX_CO2I] -
        k[497]*y[IDX_HII]*y[IDX_H2COI] - k[498]*y[IDX_HII]*y[IDX_H2COI] -
        k[499]*y[IDX_HII]*y[IDX_H2SiOI] - k[500]*y[IDX_HII]*y[IDX_HCOI] -
        k[501]*y[IDX_HII]*y[IDX_HCOI] - k[502]*y[IDX_HII]*y[IDX_HNCOI] -
        k[503]*y[IDX_HII]*y[IDX_HNOI] - k[504]*y[IDX_HII]*y[IDX_NO2I] -
        k[505]*y[IDX_HII]*y[IDX_SiH2I] - k[506]*y[IDX_HII]*y[IDX_SiH3I] -
        k[507]*y[IDX_HII]*y[IDX_SiH4I] - k[508]*y[IDX_HII]*y[IDX_SiHI] +
        k[536]*y[IDX_H2I]*y[IDX_HeII] + k[661]*y[IDX_HeII]*y[IDX_CH4I] +
        k[674]*y[IDX_HeII]*y[IDX_H2OI] + k[687]*y[IDX_HeII]*y[IDX_HNOI] +
        k[1108]*y[IDX_CHII] + k[1111]*y[IDX_CH2II] + k[1134]*y[IDX_H2II] +
        k[1146]*y[IDX_H3II] + k[1156]*y[IDX_NHII] - k[1197]*y[IDX_HII]*y[IDX_HI]
        - k[1198]*y[IDX_HII]*y[IDX_HeI] - k[1216]*y[IDX_HII]*y[IDX_EM];
    ydot[IDX_H2I] = 0.0 - k[2]*y[IDX_H2I]*y[IDX_CHI] +
        k[2]*y[IDX_H2I]*y[IDX_CHI] - k[3]*y[IDX_H2I]*y[IDX_H2I] -
        k[3]*y[IDX_H2I]*y[IDX_H2I] + k[3]*y[IDX_H2I]*y[IDX_H2I] -
        k[4]*y[IDX_H2I]*y[IDX_H2OI] + k[4]*y[IDX_H2I]*y[IDX_H2OI] -
        k[5]*y[IDX_H2I]*y[IDX_HOCII] + k[5]*y[IDX_H2I]*y[IDX_HOCII] -
        k[6]*y[IDX_H2I]*y[IDX_O2I] + k[6]*y[IDX_H2I]*y[IDX_O2I] -
        k[7]*y[IDX_H2I]*y[IDX_OHI] + k[7]*y[IDX_H2I]*y[IDX_OHI] -
        k[8]*y[IDX_H2I]*y[IDX_EM] - k[10]*y[IDX_HI]*y[IDX_H2I] +
        k[100]*y[IDX_H2II]*y[IDX_CH2I] + k[101]*y[IDX_H2II]*y[IDX_CH4I] +
        k[102]*y[IDX_H2II]*y[IDX_CHI] + k[103]*y[IDX_H2II]*y[IDX_CNI] +
        k[104]*y[IDX_H2II]*y[IDX_COI] + k[105]*y[IDX_H2II]*y[IDX_H2COI] +
        k[106]*y[IDX_H2II]*y[IDX_H2OI] + k[107]*y[IDX_H2II]*y[IDX_HCNI] +
        k[108]*y[IDX_H2II]*y[IDX_HCOI] + k[109]*y[IDX_H2II]*y[IDX_NH2I] +
        k[110]*y[IDX_H2II]*y[IDX_NH3I] + k[111]*y[IDX_H2II]*y[IDX_NHI] +
        k[112]*y[IDX_H2II]*y[IDX_NOI] + k[113]*y[IDX_H2II]*y[IDX_O2I] +
        k[114]*y[IDX_H2II]*y[IDX_OHI] - k[115]*y[IDX_H2I]*y[IDX_HeII] +
        k[128]*y[IDX_HI]*y[IDX_H2II] - k[233]*y[IDX_H2I] - k[234]*y[IDX_H2I] -
        k[235]*y[IDX_H2I] + k[246]*y[IDX_CH3I] + k[247]*y[IDX_CH3OHI] +
        k[249]*y[IDX_CH4I] + k[255]*y[IDX_H2COI] + k[257]*y[IDX_H2SiOI] +
        k[273]*y[IDX_NH3I] + k[291]*y[IDX_SiH4I] + k[295]*y[IDX_CH2II]*y[IDX_EM]
        + k[299]*y[IDX_CH3II]*y[IDX_EM] + k[307]*y[IDX_H2COII]*y[IDX_EM] +
        k[311]*y[IDX_H2NOII]*y[IDX_EM] + k[312]*y[IDX_H2OII]*y[IDX_EM] +
        k[315]*y[IDX_H3II]*y[IDX_EM] + k[319]*y[IDX_H3COII]*y[IDX_EM] +
        k[323]*y[IDX_H3OII]*y[IDX_EM] + k[324]*y[IDX_H3OII]*y[IDX_EM] +
        k[353]*y[IDX_SiH2II]*y[IDX_EM] + k[357]*y[IDX_SiH3II]*y[IDX_EM] +
        k[358]*y[IDX_SiH4II]*y[IDX_EM] + k[360]*y[IDX_SiH5II]*y[IDX_EM] +
        k[374]*y[IDX_CII]*y[IDX_NH3I] + k[380]*y[IDX_CII]*y[IDX_SiH2I] +
        k[384]*y[IDX_CI]*y[IDX_H3OII] + k[404]*y[IDX_CHII]*y[IDX_H2OI] +
        k[409]*y[IDX_CHII]*y[IDX_NH2I] + k[410]*y[IDX_CHII]*y[IDX_NHI] +
        k[415]*y[IDX_CHII]*y[IDX_OHI] + k[444]*y[IDX_CH3II]*y[IDX_OI] +
        k[445]*y[IDX_CH3II]*y[IDX_OHI] + k[455]*y[IDX_CH4I]*y[IDX_N2II] +
        k[491]*y[IDX_HII]*y[IDX_CH2I] + k[493]*y[IDX_HII]*y[IDX_CH3OHI] +
        k[494]*y[IDX_HII]*y[IDX_CH3OHI] + k[494]*y[IDX_HII]*y[IDX_CH3OHI] +
        k[495]*y[IDX_HII]*y[IDX_CH4I] + k[497]*y[IDX_HII]*y[IDX_H2COI] +
        k[498]*y[IDX_HII]*y[IDX_H2COI] + k[499]*y[IDX_HII]*y[IDX_H2SiOI] +
        k[500]*y[IDX_HII]*y[IDX_HCOI] + k[503]*y[IDX_HII]*y[IDX_HNOI] +
        k[505]*y[IDX_HII]*y[IDX_SiH2I] + k[506]*y[IDX_HII]*y[IDX_SiH3I] +
        k[507]*y[IDX_HII]*y[IDX_SiH4I] + k[508]*y[IDX_HII]*y[IDX_SiHI] +
        k[511]*y[IDX_H2II]*y[IDX_CH4I] - k[516]*y[IDX_H2II]*y[IDX_H2I] +
        k[517]*y[IDX_H2II]*y[IDX_H2COI] - k[528]*y[IDX_H2I]*y[IDX_CII] -
        k[529]*y[IDX_H2I]*y[IDX_CHII] - k[530]*y[IDX_H2I]*y[IDX_CH2II] -
        k[531]*y[IDX_H2I]*y[IDX_CNII] - k[532]*y[IDX_H2I]*y[IDX_COII] -
        k[533]*y[IDX_H2I]*y[IDX_COII] - k[534]*y[IDX_H2I]*y[IDX_H2OII] -
        k[535]*y[IDX_H2I]*y[IDX_HCNII] - k[536]*y[IDX_H2I]*y[IDX_HeII] -
        k[537]*y[IDX_H2I]*y[IDX_HeHII] - k[538]*y[IDX_H2I]*y[IDX_NII] -
        k[539]*y[IDX_H2I]*y[IDX_N2II] - k[540]*y[IDX_H2I]*y[IDX_NHII] -
        k[541]*y[IDX_H2I]*y[IDX_NHII] - k[542]*y[IDX_H2I]*y[IDX_NH2II] -
        k[543]*y[IDX_H2I]*y[IDX_OII] - k[544]*y[IDX_H2I]*y[IDX_O2HII] -
        k[545]*y[IDX_H2I]*y[IDX_OHII] - k[546]*y[IDX_H2I]*y[IDX_SiH4II] -
        k[547]*y[IDX_H2I]*y[IDX_SiOII] + k[576]*y[IDX_H3II]*y[IDX_CI] +
        k[577]*y[IDX_H3II]*y[IDX_CH2I] + k[578]*y[IDX_H3II]*y[IDX_CH3I] +
        k[579]*y[IDX_H3II]*y[IDX_CH3OHI] + k[580]*y[IDX_H3II]*y[IDX_CHI] +
        k[581]*y[IDX_H3II]*y[IDX_CNI] + k[582]*y[IDX_H3II]*y[IDX_CO2I] +
        k[583]*y[IDX_H3II]*y[IDX_COI] + k[584]*y[IDX_H3II]*y[IDX_COI] +
        k[585]*y[IDX_H3II]*y[IDX_H2COI] + k[586]*y[IDX_H3II]*y[IDX_H2OI] +
        k[587]*y[IDX_H3II]*y[IDX_HCNI] + k[588]*y[IDX_H3II]*y[IDX_HCOI] +
        k[589]*y[IDX_H3II]*y[IDX_HNCI] + k[590]*y[IDX_H3II]*y[IDX_HNOI] +
        k[591]*y[IDX_H3II]*y[IDX_MgI] + k[592]*y[IDX_H3II]*y[IDX_N2I] +
        k[593]*y[IDX_H3II]*y[IDX_NH2I] + k[594]*y[IDX_H3II]*y[IDX_NHI] +
        k[595]*y[IDX_H3II]*y[IDX_NO2I] + k[596]*y[IDX_H3II]*y[IDX_NOI] +
        k[597]*y[IDX_H3II]*y[IDX_O2I] + k[599]*y[IDX_H3II]*y[IDX_OI] +
        k[600]*y[IDX_H3II]*y[IDX_OHI] + k[601]*y[IDX_H3II]*y[IDX_SiI] +
        k[602]*y[IDX_H3II]*y[IDX_SiH2I] + k[603]*y[IDX_H3II]*y[IDX_SiH3I] +
        k[604]*y[IDX_H3II]*y[IDX_SiH4I] + k[605]*y[IDX_H3II]*y[IDX_SiHI] +
        k[606]*y[IDX_H3II]*y[IDX_SiOI] + k[614]*y[IDX_HI]*y[IDX_CHII] +
        k[615]*y[IDX_HI]*y[IDX_CH2II] + k[616]*y[IDX_HI]*y[IDX_CH3II] +
        k[617]*y[IDX_HI]*y[IDX_CH4II] + k[619]*y[IDX_HI]*y[IDX_SiHII] +
        k[653]*y[IDX_HeII]*y[IDX_CH2I] + k[655]*y[IDX_HeII]*y[IDX_CH3I] +
        k[658]*y[IDX_HeII]*y[IDX_CH4I] + k[659]*y[IDX_HeII]*y[IDX_CH4I] +
        k[670]*y[IDX_HeII]*y[IDX_H2COI] + k[689]*y[IDX_HeII]*y[IDX_NH2I] +
        k[691]*y[IDX_HeII]*y[IDX_NH3I] + k[703]*y[IDX_HeII]*y[IDX_SiH2I] +
        k[705]*y[IDX_HeII]*y[IDX_SiH3I] + k[707]*y[IDX_HeII]*y[IDX_SiH4I] +
        k[707]*y[IDX_HeII]*y[IDX_SiH4I] + k[708]*y[IDX_HeII]*y[IDX_SiH4I] +
        k[717]*y[IDX_NII]*y[IDX_CH4I] + k[724]*y[IDX_NII]*y[IDX_NH3I] +
        k[739]*y[IDX_NI]*y[IDX_H2OII] + k[755]*y[IDX_NHII]*y[IDX_H2OI] +
        k[794]*y[IDX_NHI]*y[IDX_CH3II] + k[823]*y[IDX_OI]*y[IDX_H2OII] +
        k[828]*y[IDX_OI]*y[IDX_NH3II] + k[834]*y[IDX_OI]*y[IDX_SiH3II] +
        k[889]*y[IDX_CH2I]*y[IDX_O2I] + k[894]*y[IDX_CH2I]*y[IDX_OI] +
        k[915]*y[IDX_CH3I]*y[IDX_OI] + k[918]*y[IDX_CH3I]*y[IDX_OHI] -
        k[955]*y[IDX_H2I]*y[IDX_CI] - k[956]*y[IDX_H2I]*y[IDX_CH2I] -
        k[957]*y[IDX_H2I]*y[IDX_CH3I] - k[958]*y[IDX_H2I]*y[IDX_CHI] -
        k[959]*y[IDX_H2I]*y[IDX_CNI] - k[960]*y[IDX_H2I]*y[IDX_NI] -
        k[961]*y[IDX_H2I]*y[IDX_NH2I] - k[962]*y[IDX_H2I]*y[IDX_NHI] -
        k[963]*y[IDX_H2I]*y[IDX_O2I] - k[964]*y[IDX_H2I]*y[IDX_O2I] -
        k[965]*y[IDX_H2I]*y[IDX_OI] - k[966]*y[IDX_H2I]*y[IDX_OHI] +
        k[967]*y[IDX_HI]*y[IDX_CH2I] + k[968]*y[IDX_HI]*y[IDX_CH3I] +
        k[969]*y[IDX_HI]*y[IDX_CH4I] + k[970]*y[IDX_HI]*y[IDX_CHI] +
        k[973]*y[IDX_HI]*y[IDX_H2CNI] + k[974]*y[IDX_HI]*y[IDX_H2COI] +
        k[975]*y[IDX_HI]*y[IDX_H2OI] + k[976]*y[IDX_HI]*y[IDX_HCNI] +
        k[977]*y[IDX_HI]*y[IDX_HCOI] + k[981]*y[IDX_HI]*y[IDX_HNOI] +
        k[983]*y[IDX_HI]*y[IDX_NH2I] + k[984]*y[IDX_HI]*y[IDX_NH3I] +
        k[985]*y[IDX_HI]*y[IDX_NHI] + k[991]*y[IDX_HI]*y[IDX_O2HI] +
        k[996]*y[IDX_HI]*y[IDX_OHI] + k[997]*y[IDX_HCOI]*y[IDX_HCOI] +
        k[1009]*y[IDX_NI]*y[IDX_CH3I] + k[1038]*y[IDX_NHI]*y[IDX_NHI] +
        k[1060]*y[IDX_OI]*y[IDX_H2CNI] + k[1085]*y[IDX_OI]*y[IDX_SiH2I] +
        k[1109]*y[IDX_CH2II] + k[1114]*y[IDX_CH3II] + k[1118]*y[IDX_CH3I] +
        k[1119]*y[IDX_CH3OHI] + k[1122]*y[IDX_CH4II] + k[1124]*y[IDX_CH4I] +
        k[1127]*y[IDX_CH4I] + k[1136]*y[IDX_H2COI] + k[1143]*y[IDX_H2SiOI] +
        k[1146]*y[IDX_H3II] + k[1161]*y[IDX_NH3I] + k[1184]*y[IDX_SiH3I] +
        k[1185]*y[IDX_SiH4I] + k[1187]*y[IDX_SiH4I] -
        k[1199]*y[IDX_H2I]*y[IDX_CII] - k[1200]*y[IDX_H2I]*y[IDX_CI] -
        k[1201]*y[IDX_H2I]*y[IDX_CHI] - k[1202]*y[IDX_H2I]*y[IDX_SiII] -
        k[1203]*y[IDX_H2I]*y[IDX_SiHII] - k[1204]*y[IDX_H2I]*y[IDX_SiH3II];
    ydot[IDX_H2II] = 0.0 - k[100]*y[IDX_H2II]*y[IDX_CH2I] -
        k[101]*y[IDX_H2II]*y[IDX_CH4I] - k[102]*y[IDX_H2II]*y[IDX_CHI] -
        k[103]*y[IDX_H2II]*y[IDX_CNI] - k[104]*y[IDX_H2II]*y[IDX_COI] -
        k[105]*y[IDX_H2II]*y[IDX_H2COI] - k[106]*y[IDX_H2II]*y[IDX_H2OI] -
        k[107]*y[IDX_H2II]*y[IDX_HCNI] - k[108]*y[IDX_H2II]*y[IDX_HCOI] -
        k[109]*y[IDX_H2II]*y[IDX_NH2I] - k[110]*y[IDX_H2II]*y[IDX_NH3I] -
        k[111]*y[IDX_H2II]*y[IDX_NHI] - k[112]*y[IDX_H2II]*y[IDX_NOI] -
        k[113]*y[IDX_H2II]*y[IDX_O2I] - k[114]*y[IDX_H2II]*y[IDX_OHI] +
        k[115]*y[IDX_H2I]*y[IDX_HeII] - k[128]*y[IDX_HI]*y[IDX_H2II] +
        k[234]*y[IDX_H2I] - k[305]*y[IDX_H2II]*y[IDX_EM] +
        k[501]*y[IDX_HII]*y[IDX_HCOI] - k[509]*y[IDX_H2II]*y[IDX_CI] -
        k[510]*y[IDX_H2II]*y[IDX_CH2I] - k[511]*y[IDX_H2II]*y[IDX_CH4I] -
        k[512]*y[IDX_H2II]*y[IDX_CHI] - k[513]*y[IDX_H2II]*y[IDX_CNI] -
        k[514]*y[IDX_H2II]*y[IDX_CO2I] - k[515]*y[IDX_H2II]*y[IDX_COI] -
        k[516]*y[IDX_H2II]*y[IDX_H2I] - k[517]*y[IDX_H2II]*y[IDX_H2COI] -
        k[518]*y[IDX_H2II]*y[IDX_H2OI] - k[519]*y[IDX_H2II]*y[IDX_HCOI] -
        k[520]*y[IDX_H2II]*y[IDX_HeI] - k[521]*y[IDX_H2II]*y[IDX_N2I] -
        k[522]*y[IDX_H2II]*y[IDX_NI] - k[523]*y[IDX_H2II]*y[IDX_NHI] -
        k[524]*y[IDX_H2II]*y[IDX_NOI] - k[525]*y[IDX_H2II]*y[IDX_O2I] -
        k[526]*y[IDX_H2II]*y[IDX_OI] - k[527]*y[IDX_H2II]*y[IDX_OHI] +
        k[618]*y[IDX_HI]*y[IDX_HeHII] - k[1134]*y[IDX_H2II] +
        k[1145]*y[IDX_H3II] + k[1197]*y[IDX_HII]*y[IDX_HI];
    ydot[IDX_H2CNI] = 0.0 - k[254]*y[IDX_H2CNI] -
        k[973]*y[IDX_HI]*y[IDX_H2CNI] + k[1008]*y[IDX_NI]*y[IDX_CH3I] -
        k[1013]*y[IDX_NI]*y[IDX_H2CNI] - k[1060]*y[IDX_OI]*y[IDX_H2CNI] -
        k[1135]*y[IDX_H2CNI] - k[1298]*y[IDX_H2CNI] + k[1339]*y[IDX_GH2CNI] +
        k[1340]*y[IDX_GH2CNI] + k[1341]*y[IDX_GH2CNI] + k[1342]*y[IDX_GH2CNI];
    ydot[IDX_H2COI] = 0.0 - k[16]*y[IDX_CII]*y[IDX_H2COI] +
        k[39]*y[IDX_CH2I]*y[IDX_H2COII] - k[49]*y[IDX_CH4II]*y[IDX_H2COI] +
        k[55]*y[IDX_CHI]*y[IDX_H2COII] - k[64]*y[IDX_CNII]*y[IDX_H2COI] -
        k[70]*y[IDX_COII]*y[IDX_H2COI] - k[79]*y[IDX_HII]*y[IDX_H2COI] -
        k[105]*y[IDX_H2II]*y[IDX_H2COI] - k[116]*y[IDX_H2COI]*y[IDX_O2II] -
        k[117]*y[IDX_H2OII]*y[IDX_H2COI] + k[136]*y[IDX_HCOI]*y[IDX_H2COII] -
        k[142]*y[IDX_HeII]*y[IDX_H2COI] + k[148]*y[IDX_MgI]*y[IDX_H2COII] -
        k[159]*y[IDX_NII]*y[IDX_H2COI] - k[170]*y[IDX_N2II]*y[IDX_H2COI] -
        k[175]*y[IDX_NHII]*y[IDX_H2COI] + k[194]*y[IDX_NH3I]*y[IDX_H2COII] +
        k[203]*y[IDX_NOI]*y[IDX_H2COII] - k[209]*y[IDX_OII]*y[IDX_H2COI] -
        k[219]*y[IDX_OHII]*y[IDX_H2COI] + k[228]*y[IDX_SiI]*y[IDX_H2COII] +
        k[247]*y[IDX_CH3OHI] - k[255]*y[IDX_H2COI] +
        k[320]*y[IDX_H3COII]*y[IDX_EM] - k[368]*y[IDX_CII]*y[IDX_H2COI] -
        k[369]*y[IDX_CII]*y[IDX_H2COI] + k[396]*y[IDX_CHII]*y[IDX_CH3OHI] -
        k[399]*y[IDX_CHII]*y[IDX_H2COI] - k[400]*y[IDX_CHII]*y[IDX_H2COI] -
        k[401]*y[IDX_CHII]*y[IDX_H2COI] - k[417]*y[IDX_CH2II]*y[IDX_H2COI] +
        k[438]*y[IDX_CH2I]*y[IDX_SiOII] - k[440]*y[IDX_CH3II]*y[IDX_H2COI] -
        k[449]*y[IDX_CH4II]*y[IDX_H2COI] + k[461]*y[IDX_CHI]*y[IDX_H3COII] -
        k[479]*y[IDX_CNII]*y[IDX_H2COI] - k[484]*y[IDX_COII]*y[IDX_H2COI] -
        k[497]*y[IDX_HII]*y[IDX_H2COI] - k[498]*y[IDX_HII]*y[IDX_H2COI] -
        k[517]*y[IDX_H2II]*y[IDX_H2COI] - k[548]*y[IDX_H2COII]*y[IDX_H2COI] -
        k[550]*y[IDX_H2COI]*y[IDX_HNOII] - k[551]*y[IDX_H2COI]*y[IDX_O2II] -
        k[552]*y[IDX_H2COI]*y[IDX_O2HII] - k[554]*y[IDX_H2OII]*y[IDX_H2COI] +
        k[564]*y[IDX_H2OI]*y[IDX_H3COII] - k[585]*y[IDX_H3II]*y[IDX_H2COI] -
        k[607]*y[IDX_H3OII]*y[IDX_H2COI] - k[622]*y[IDX_HCNII]*y[IDX_H2COI] +
        k[628]*y[IDX_HCNI]*y[IDX_H3COII] - k[633]*y[IDX_HCNHII]*y[IDX_H2COI] -
        k[634]*y[IDX_HCNHII]*y[IDX_H2COI] - k[635]*y[IDX_HCOII]*y[IDX_H2COI] +
        k[647]*y[IDX_HNCI]*y[IDX_H3COII] - k[670]*y[IDX_HeII]*y[IDX_H2COI] -
        k[671]*y[IDX_HeII]*y[IDX_H2COI] - k[672]*y[IDX_HeII]*y[IDX_H2COI] -
        k[721]*y[IDX_NII]*y[IDX_H2COI] - k[722]*y[IDX_NII]*y[IDX_H2COI] -
        k[730]*y[IDX_N2II]*y[IDX_H2COI] - k[735]*y[IDX_N2HII]*y[IDX_H2COI] -
        k[752]*y[IDX_NHII]*y[IDX_H2COI] - k[753]*y[IDX_NHII]*y[IDX_H2COI] -
        k[769]*y[IDX_NH2II]*y[IDX_H2COI] - k[770]*y[IDX_NH2II]*y[IDX_H2COI] +
        k[782]*y[IDX_NH2I]*y[IDX_H3COII] - k[813]*y[IDX_OII]*y[IDX_H2COI] -
        k[839]*y[IDX_OHII]*y[IDX_H2COI] - k[881]*y[IDX_CH2I]*y[IDX_H2COI] +
        k[885]*y[IDX_CH2I]*y[IDX_NO2I] + k[886]*y[IDX_CH2I]*y[IDX_NOI] +
        k[892]*y[IDX_CH2I]*y[IDX_O2I] + k[898]*y[IDX_CH2I]*y[IDX_OHI] -
        k[903]*y[IDX_CH3I]*y[IDX_H2COI] + k[909]*y[IDX_CH3I]*y[IDX_NO2I] +
        k[911]*y[IDX_CH3I]*y[IDX_O2I] + k[916]*y[IDX_CH3I]*y[IDX_OI] +
        k[918]*y[IDX_CH3I]*y[IDX_OHI] - k[924]*y[IDX_CHI]*y[IDX_H2COI] -
        k[942]*y[IDX_CNI]*y[IDX_H2COI] - k[974]*y[IDX_HI]*y[IDX_H2COI] +
        k[998]*y[IDX_HCOI]*y[IDX_HCOI] + k[999]*y[IDX_HCOI]*y[IDX_HNOI] +
        k[1003]*y[IDX_HCOI]*y[IDX_O2HI] - k[1061]*y[IDX_OI]*y[IDX_H2COI] -
        k[1093]*y[IDX_OHI]*y[IDX_H2COI] + k[1119]*y[IDX_CH3OHI] -
        k[1136]*y[IDX_H2COI] - k[1137]*y[IDX_H2COI] - k[1138]*y[IDX_H2COI] -
        k[1139]*y[IDX_H2COI] + k[1217]*y[IDX_H2COII]*y[IDX_EM] -
        k[1252]*y[IDX_H2COI] + k[1347]*y[IDX_GH2COI] + k[1348]*y[IDX_GH2COI] +
        k[1349]*y[IDX_GH2COI] + k[1350]*y[IDX_GH2COI];
    ydot[IDX_H2COII] = 0.0 + k[16]*y[IDX_CII]*y[IDX_H2COI] -
        k[39]*y[IDX_CH2I]*y[IDX_H2COII] + k[49]*y[IDX_CH4II]*y[IDX_H2COI] -
        k[55]*y[IDX_CHI]*y[IDX_H2COII] + k[64]*y[IDX_CNII]*y[IDX_H2COI] +
        k[70]*y[IDX_COII]*y[IDX_H2COI] + k[79]*y[IDX_HII]*y[IDX_H2COI] +
        k[105]*y[IDX_H2II]*y[IDX_H2COI] + k[116]*y[IDX_H2COI]*y[IDX_O2II] +
        k[117]*y[IDX_H2OII]*y[IDX_H2COI] - k[136]*y[IDX_HCOI]*y[IDX_H2COII] +
        k[142]*y[IDX_HeII]*y[IDX_H2COI] - k[148]*y[IDX_MgI]*y[IDX_H2COII] +
        k[159]*y[IDX_NII]*y[IDX_H2COI] + k[170]*y[IDX_N2II]*y[IDX_H2COI] +
        k[175]*y[IDX_NHII]*y[IDX_H2COI] - k[194]*y[IDX_NH3I]*y[IDX_H2COII] -
        k[203]*y[IDX_NOI]*y[IDX_H2COII] + k[209]*y[IDX_OII]*y[IDX_H2COI] +
        k[219]*y[IDX_OHII]*y[IDX_H2COI] - k[228]*y[IDX_SiI]*y[IDX_H2COII] -
        k[306]*y[IDX_H2COII]*y[IDX_EM] - k[307]*y[IDX_H2COII]*y[IDX_EM] -
        k[308]*y[IDX_H2COII]*y[IDX_EM] - k[309]*y[IDX_H2COII]*y[IDX_EM] +
        k[402]*y[IDX_CHII]*y[IDX_H2OI] + k[416]*y[IDX_CH2II]*y[IDX_CO2I] -
        k[423]*y[IDX_CH2I]*y[IDX_H2COII] + k[435]*y[IDX_CH2I]*y[IDX_O2II] +
        k[443]*y[IDX_CH3II]*y[IDX_OI] + k[445]*y[IDX_CH3II]*y[IDX_OHI] -
        k[452]*y[IDX_CH4I]*y[IDX_H2COII] - k[459]*y[IDX_CHI]*y[IDX_H2COII] -
        k[548]*y[IDX_H2COII]*y[IDX_H2COI] - k[549]*y[IDX_H2COII]*y[IDX_O2I] +
        k[558]*y[IDX_H2OII]*y[IDX_HCOI] - k[563]*y[IDX_H2OI]*y[IDX_H2COII] +
        k[588]*y[IDX_H3II]*y[IDX_HCOI] + k[624]*y[IDX_HCNII]*y[IDX_HCOI] -
        k[627]*y[IDX_HCNI]*y[IDX_H2COII] + k[636]*y[IDX_HCOII]*y[IDX_HCOI] -
        k[641]*y[IDX_HCOI]*y[IDX_H2COII] + k[642]*y[IDX_HCOI]*y[IDX_HNOII] +
        k[643]*y[IDX_HCOI]*y[IDX_N2HII] + k[645]*y[IDX_HCOI]*y[IDX_O2HII] -
        k[646]*y[IDX_HNCI]*y[IDX_H2COII] + k[712]*y[IDX_NII]*y[IDX_CH3OHI] +
        k[759]*y[IDX_NHII]*y[IDX_HCOI] + k[774]*y[IDX_NH2II]*y[IDX_HCOI] -
        k[780]*y[IDX_NH2I]*y[IDX_H2COII] - k[796]*y[IDX_NHI]*y[IDX_H2COII] +
        k[808]*y[IDX_OII]*y[IDX_CH3OHI] + k[843]*y[IDX_OHII]*y[IDX_HCOI] +
        k[1138]*y[IDX_H2COI] - k[1217]*y[IDX_H2COII]*y[IDX_EM] -
        k[1284]*y[IDX_H2COII];
    ydot[IDX_H2NOII] = 0.0 - k[310]*y[IDX_H2NOII]*y[IDX_EM] -
        k[311]*y[IDX_H2NOII]*y[IDX_EM] + k[590]*y[IDX_H3II]*y[IDX_HNOI] +
        k[777]*y[IDX_NH2II]*y[IDX_O2I] - k[1296]*y[IDX_H2NOII];
    ydot[IDX_H2OI] = 0.0 - k[4]*y[IDX_H2I]*y[IDX_H2OI] -
        k[11]*y[IDX_HI]*y[IDX_H2OI] + k[40]*y[IDX_CH2I]*y[IDX_H2OII] +
        k[56]*y[IDX_CHI]*y[IDX_H2OII] - k[80]*y[IDX_HII]*y[IDX_H2OI] -
        k[106]*y[IDX_H2II]*y[IDX_H2OI] + k[117]*y[IDX_H2OII]*y[IDX_H2COI] +
        k[118]*y[IDX_H2OII]*y[IDX_HCOI] + k[119]*y[IDX_H2OII]*y[IDX_MgI] +
        k[120]*y[IDX_H2OII]*y[IDX_NOI] + k[121]*y[IDX_H2OII]*y[IDX_O2I] +
        k[122]*y[IDX_H2OII]*y[IDX_SiI] - k[123]*y[IDX_H2OI]*y[IDX_COII] -
        k[124]*y[IDX_H2OI]*y[IDX_HCNII] - k[125]*y[IDX_H2OI]*y[IDX_N2II] -
        k[143]*y[IDX_HeII]*y[IDX_H2OI] - k[160]*y[IDX_NII]*y[IDX_H2OI] -
        k[176]*y[IDX_NHII]*y[IDX_H2OI] + k[185]*y[IDX_NH2I]*y[IDX_H2OII] +
        k[195]*y[IDX_NH3I]*y[IDX_H2OII] - k[210]*y[IDX_OII]*y[IDX_H2OI] -
        k[220]*y[IDX_OHII]*y[IDX_H2OI] - k[256]*y[IDX_H2OI] +
        k[318]*y[IDX_H3COII]*y[IDX_EM] + k[322]*y[IDX_H3OII]*y[IDX_EM] -
        k[370]*y[IDX_CII]*y[IDX_H2OI] - k[371]*y[IDX_CII]*y[IDX_H2OI] -
        k[402]*y[IDX_CHII]*y[IDX_H2OI] - k[403]*y[IDX_CHII]*y[IDX_H2OI] -
        k[404]*y[IDX_CHII]*y[IDX_H2OI] - k[418]*y[IDX_CH2II]*y[IDX_H2OI] +
        k[425]*y[IDX_CH2I]*y[IDX_H3OII] - k[450]*y[IDX_CH4II]*y[IDX_H2OI] +
        k[462]*y[IDX_CHI]*y[IDX_H3OII] + k[492]*y[IDX_HII]*y[IDX_CH3OHI] -
        k[518]*y[IDX_H2II]*y[IDX_H2OI] - k[555]*y[IDX_H2OII]*y[IDX_H2OI] -
        k[560]*y[IDX_H2OI]*y[IDX_CNII] - k[561]*y[IDX_H2OI]*y[IDX_CNII] -
        k[562]*y[IDX_H2OI]*y[IDX_COII] - k[563]*y[IDX_H2OI]*y[IDX_H2COII] -
        k[564]*y[IDX_H2OI]*y[IDX_H3COII] - k[565]*y[IDX_H2OI]*y[IDX_HCNII] -
        k[566]*y[IDX_H2OI]*y[IDX_HCOII] - k[567]*y[IDX_H2OI]*y[IDX_HCO2II] -
        k[568]*y[IDX_H2OI]*y[IDX_HNOII] - k[569]*y[IDX_H2OI]*y[IDX_N2II] -
        k[570]*y[IDX_H2OI]*y[IDX_N2HII] - k[571]*y[IDX_H2OI]*y[IDX_O2HII] -
        k[572]*y[IDX_H2OI]*y[IDX_SiII] - k[573]*y[IDX_H2OI]*y[IDX_SiHII] -
        k[574]*y[IDX_H2OI]*y[IDX_SiH4II] - k[575]*y[IDX_H2OI]*y[IDX_SiH5II] +
        k[579]*y[IDX_H3II]*y[IDX_CH3OHI] - k[586]*y[IDX_H3II]*y[IDX_H2OI] +
        k[607]*y[IDX_H3OII]*y[IDX_H2COI] + k[608]*y[IDX_H3OII]*y[IDX_HCNI] +
        k[609]*y[IDX_H3OII]*y[IDX_HNCI] + k[610]*y[IDX_H3OII]*y[IDX_SiI] +
        k[611]*y[IDX_H3OII]*y[IDX_SiH2I] + k[612]*y[IDX_H3OII]*y[IDX_SiHI] +
        k[613]*y[IDX_H3OII]*y[IDX_SiOI] - k[673]*y[IDX_HeII]*y[IDX_H2OI] -
        k[674]*y[IDX_HeII]*y[IDX_H2OI] - k[754]*y[IDX_NHII]*y[IDX_H2OI] -
        k[755]*y[IDX_NHII]*y[IDX_H2OI] - k[756]*y[IDX_NHII]*y[IDX_H2OI] -
        k[757]*y[IDX_NHII]*y[IDX_H2OI] - k[771]*y[IDX_NH2II]*y[IDX_H2OI] -
        k[772]*y[IDX_NH2II]*y[IDX_H2OI] + k[783]*y[IDX_NH2I]*y[IDX_H3OII] +
        k[808]*y[IDX_OII]*y[IDX_CH3OHI] - k[840]*y[IDX_OHII]*y[IDX_H2OI] +
        k[891]*y[IDX_CH2I]*y[IDX_O2I] + k[899]*y[IDX_CH2I]*y[IDX_OHI] -
        k[904]*y[IDX_CH3I]*y[IDX_H2OI] + k[910]*y[IDX_CH3I]*y[IDX_NOI] +
        k[912]*y[IDX_CH3I]*y[IDX_O2I] + k[919]*y[IDX_CH3I]*y[IDX_OHI] +
        k[922]*y[IDX_CH4I]*y[IDX_OHI] + k[966]*y[IDX_H2I]*y[IDX_OHI] -
        k[975]*y[IDX_HI]*y[IDX_H2OI] + k[990]*y[IDX_HI]*y[IDX_O2HI] +
        k[1029]*y[IDX_NH2I]*y[IDX_NOI] + k[1031]*y[IDX_NH2I]*y[IDX_OHI] -
        k[1036]*y[IDX_NHI]*y[IDX_H2OI] + k[1048]*y[IDX_NHI]*y[IDX_OHI] -
        k[1062]*y[IDX_OI]*y[IDX_H2OI] + k[1093]*y[IDX_OHI]*y[IDX_H2COI] +
        k[1094]*y[IDX_OHI]*y[IDX_HCNI] + k[1096]*y[IDX_OHI]*y[IDX_HCOI] +
        k[1097]*y[IDX_OHI]*y[IDX_HNOI] + k[1098]*y[IDX_OHI]*y[IDX_NH3I] +
        k[1100]*y[IDX_OHI]*y[IDX_O2HI] + k[1101]*y[IDX_OHI]*y[IDX_OHI] -
        k[1141]*y[IDX_H2OI] - k[1142]*y[IDX_H2OI] + k[1208]*y[IDX_HI]*y[IDX_OHI]
        - k[1257]*y[IDX_H2OI] + k[1315]*y[IDX_GH2OI] + k[1316]*y[IDX_GH2OI] +
        k[1317]*y[IDX_GH2OI] + k[1318]*y[IDX_GH2OI];
    ydot[IDX_H2OII] = 0.0 - k[40]*y[IDX_CH2I]*y[IDX_H2OII] -
        k[56]*y[IDX_CHI]*y[IDX_H2OII] + k[80]*y[IDX_HII]*y[IDX_H2OI] +
        k[106]*y[IDX_H2II]*y[IDX_H2OI] - k[117]*y[IDX_H2OII]*y[IDX_H2COI] -
        k[118]*y[IDX_H2OII]*y[IDX_HCOI] - k[119]*y[IDX_H2OII]*y[IDX_MgI] -
        k[120]*y[IDX_H2OII]*y[IDX_NOI] - k[121]*y[IDX_H2OII]*y[IDX_O2I] -
        k[122]*y[IDX_H2OII]*y[IDX_SiI] + k[123]*y[IDX_H2OI]*y[IDX_COII] +
        k[124]*y[IDX_H2OI]*y[IDX_HCNII] + k[125]*y[IDX_H2OI]*y[IDX_N2II] +
        k[143]*y[IDX_HeII]*y[IDX_H2OI] + k[160]*y[IDX_NII]*y[IDX_H2OI] +
        k[176]*y[IDX_NHII]*y[IDX_H2OI] - k[185]*y[IDX_NH2I]*y[IDX_H2OII] -
        k[195]*y[IDX_NH3I]*y[IDX_H2OII] + k[210]*y[IDX_OII]*y[IDX_H2OI] +
        k[220]*y[IDX_OHII]*y[IDX_H2OI] - k[312]*y[IDX_H2OII]*y[IDX_EM] -
        k[313]*y[IDX_H2OII]*y[IDX_EM] - k[314]*y[IDX_H2OII]*y[IDX_EM] -
        k[383]*y[IDX_CI]*y[IDX_H2OII] - k[424]*y[IDX_CH2I]*y[IDX_H2OII] -
        k[453]*y[IDX_CH4I]*y[IDX_H2OII] - k[460]*y[IDX_CHI]*y[IDX_H2OII] +
        k[527]*y[IDX_H2II]*y[IDX_OHI] - k[534]*y[IDX_H2I]*y[IDX_H2OII] +
        k[545]*y[IDX_H2I]*y[IDX_OHII] - k[553]*y[IDX_H2OII]*y[IDX_COI] -
        k[554]*y[IDX_H2OII]*y[IDX_H2COI] - k[555]*y[IDX_H2OII]*y[IDX_H2OI] -
        k[556]*y[IDX_H2OII]*y[IDX_HCNI] - k[557]*y[IDX_H2OII]*y[IDX_HCOI] -
        k[558]*y[IDX_H2OII]*y[IDX_HCOI] - k[559]*y[IDX_H2OII]*y[IDX_HNCI] +
        k[598]*y[IDX_H3II]*y[IDX_OI] + k[600]*y[IDX_H3II]*y[IDX_OHI] -
        k[738]*y[IDX_NI]*y[IDX_H2OII] - k[739]*y[IDX_NI]*y[IDX_H2OII] +
        k[768]*y[IDX_NHII]*y[IDX_OHI] - k[781]*y[IDX_NH2I]*y[IDX_H2OII] -
        k[797]*y[IDX_NHI]*y[IDX_H2OII] - k[823]*y[IDX_OI]*y[IDX_H2OII] +
        k[842]*y[IDX_OHII]*y[IDX_HCOI] + k[847]*y[IDX_OHII]*y[IDX_OHI] -
        k[852]*y[IDX_OHI]*y[IDX_H2OII] + k[853]*y[IDX_OHI]*y[IDX_HCNII] +
        k[854]*y[IDX_OHI]*y[IDX_HCOII] + k[856]*y[IDX_OHI]*y[IDX_HNOII] +
        k[857]*y[IDX_OHI]*y[IDX_N2HII] + k[858]*y[IDX_OHI]*y[IDX_O2HII] -
        k[1140]*y[IDX_H2OII] + k[1141]*y[IDX_H2OI] - k[1280]*y[IDX_H2OII];
    ydot[IDX_H2SiOI] = 0.0 - k[257]*y[IDX_H2SiOI] -
        k[499]*y[IDX_HII]*y[IDX_H2SiOI] - k[675]*y[IDX_HeII]*y[IDX_H2SiOI] +
        k[1087]*y[IDX_OI]*y[IDX_SiH3I] - k[1143]*y[IDX_H2SiOI] -
        k[1144]*y[IDX_H2SiOI] - k[1247]*y[IDX_H2SiOI] + k[1391]*y[IDX_GH2SiOI] +
        k[1392]*y[IDX_GH2SiOI] + k[1393]*y[IDX_GH2SiOI] +
        k[1394]*y[IDX_GH2SiOI];
    ydot[IDX_H3II] = 0.0 - k[315]*y[IDX_H3II]*y[IDX_EM] -
        k[316]*y[IDX_H3II]*y[IDX_EM] + k[516]*y[IDX_H2II]*y[IDX_H2I] +
        k[519]*y[IDX_H2II]*y[IDX_HCOI] + k[537]*y[IDX_H2I]*y[IDX_HeHII] +
        k[540]*y[IDX_H2I]*y[IDX_NHII] + k[544]*y[IDX_H2I]*y[IDX_O2HII] -
        k[576]*y[IDX_H3II]*y[IDX_CI] - k[577]*y[IDX_H3II]*y[IDX_CH2I] -
        k[578]*y[IDX_H3II]*y[IDX_CH3I] - k[579]*y[IDX_H3II]*y[IDX_CH3OHI] -
        k[580]*y[IDX_H3II]*y[IDX_CHI] - k[581]*y[IDX_H3II]*y[IDX_CNI] -
        k[582]*y[IDX_H3II]*y[IDX_CO2I] - k[583]*y[IDX_H3II]*y[IDX_COI] -
        k[584]*y[IDX_H3II]*y[IDX_COI] - k[585]*y[IDX_H3II]*y[IDX_H2COI] -
        k[586]*y[IDX_H3II]*y[IDX_H2OI] - k[587]*y[IDX_H3II]*y[IDX_HCNI] -
        k[588]*y[IDX_H3II]*y[IDX_HCOI] - k[589]*y[IDX_H3II]*y[IDX_HNCI] -
        k[590]*y[IDX_H3II]*y[IDX_HNOI] - k[591]*y[IDX_H3II]*y[IDX_MgI] -
        k[592]*y[IDX_H3II]*y[IDX_N2I] - k[593]*y[IDX_H3II]*y[IDX_NH2I] -
        k[594]*y[IDX_H3II]*y[IDX_NHI] - k[595]*y[IDX_H3II]*y[IDX_NO2I] -
        k[596]*y[IDX_H3II]*y[IDX_NOI] - k[597]*y[IDX_H3II]*y[IDX_O2I] -
        k[598]*y[IDX_H3II]*y[IDX_OI] - k[599]*y[IDX_H3II]*y[IDX_OI] -
        k[600]*y[IDX_H3II]*y[IDX_OHI] - k[601]*y[IDX_H3II]*y[IDX_SiI] -
        k[602]*y[IDX_H3II]*y[IDX_SiH2I] - k[603]*y[IDX_H3II]*y[IDX_SiH3I] -
        k[604]*y[IDX_H3II]*y[IDX_SiH4I] - k[605]*y[IDX_H3II]*y[IDX_SiHI] -
        k[606]*y[IDX_H3II]*y[IDX_SiOI] - k[1145]*y[IDX_H3II] -
        k[1146]*y[IDX_H3II];
    ydot[IDX_H3COII] = 0.0 - k[317]*y[IDX_H3COII]*y[IDX_EM] -
        k[318]*y[IDX_H3COII]*y[IDX_EM] - k[319]*y[IDX_H3COII]*y[IDX_EM] -
        k[320]*y[IDX_H3COII]*y[IDX_EM] - k[321]*y[IDX_H3COII]*y[IDX_EM] +
        k[365]*y[IDX_CII]*y[IDX_CH3OHI] + k[397]*y[IDX_CHII]*y[IDX_CH3OHI] +
        k[400]*y[IDX_CHII]*y[IDX_H2COI] + k[418]*y[IDX_CH2II]*y[IDX_H2OI] +
        k[439]*y[IDX_CH3II]*y[IDX_CH3OHI] + k[442]*y[IDX_CH3II]*y[IDX_O2I] +
        k[449]*y[IDX_CH4II]*y[IDX_H2COI] + k[452]*y[IDX_CH4I]*y[IDX_H2COII] -
        k[461]*y[IDX_CHI]*y[IDX_H3COII] + k[493]*y[IDX_HII]*y[IDX_CH3OHI] +
        k[548]*y[IDX_H2COII]*y[IDX_H2COI] + k[550]*y[IDX_H2COI]*y[IDX_HNOII] +
        k[552]*y[IDX_H2COI]*y[IDX_O2HII] + k[554]*y[IDX_H2OII]*y[IDX_H2COI] -
        k[564]*y[IDX_H2OI]*y[IDX_H3COII] + k[585]*y[IDX_H3II]*y[IDX_H2COI] +
        k[607]*y[IDX_H3OII]*y[IDX_H2COI] + k[622]*y[IDX_HCNII]*y[IDX_H2COI] -
        k[628]*y[IDX_HCNI]*y[IDX_H3COII] + k[633]*y[IDX_HCNHII]*y[IDX_H2COI] +
        k[634]*y[IDX_HCNHII]*y[IDX_H2COI] + k[635]*y[IDX_HCOII]*y[IDX_H2COI] +
        k[641]*y[IDX_HCOI]*y[IDX_H2COII] - k[647]*y[IDX_HNCI]*y[IDX_H3COII] +
        k[713]*y[IDX_NII]*y[IDX_CH3OHI] + k[735]*y[IDX_N2HII]*y[IDX_H2COI] +
        k[752]*y[IDX_NHII]*y[IDX_H2COI] + k[769]*y[IDX_NH2II]*y[IDX_H2COI] -
        k[782]*y[IDX_NH2I]*y[IDX_H3COII] + k[796]*y[IDX_NHI]*y[IDX_H2COII] +
        k[809]*y[IDX_OII]*y[IDX_CH3OHI] + k[820]*y[IDX_O2II]*y[IDX_CH3OHI] +
        k[839]*y[IDX_OHII]*y[IDX_H2COI] + k[1120]*y[IDX_CH3OHI] -
        k[1249]*y[IDX_H3COII];
    ydot[IDX_H3OII] = 0.0 - k[322]*y[IDX_H3OII]*y[IDX_EM] -
        k[323]*y[IDX_H3OII]*y[IDX_EM] - k[324]*y[IDX_H3OII]*y[IDX_EM] -
        k[325]*y[IDX_H3OII]*y[IDX_EM] - k[384]*y[IDX_CI]*y[IDX_H3OII] +
        k[403]*y[IDX_CHII]*y[IDX_H2OI] - k[425]*y[IDX_CH2I]*y[IDX_H3OII] +
        k[450]*y[IDX_CH4II]*y[IDX_H2OI] + k[453]*y[IDX_CH4I]*y[IDX_H2OII] +
        k[457]*y[IDX_CH4I]*y[IDX_OHII] - k[462]*y[IDX_CHI]*y[IDX_H3OII] +
        k[518]*y[IDX_H2II]*y[IDX_H2OI] + k[534]*y[IDX_H2I]*y[IDX_H2OII] +
        k[555]*y[IDX_H2OII]*y[IDX_H2OI] + k[557]*y[IDX_H2OII]*y[IDX_HCOI] +
        k[563]*y[IDX_H2OI]*y[IDX_H2COII] + k[564]*y[IDX_H2OI]*y[IDX_H3COII] +
        k[565]*y[IDX_H2OI]*y[IDX_HCNII] + k[566]*y[IDX_H2OI]*y[IDX_HCOII] +
        k[567]*y[IDX_H2OI]*y[IDX_HCO2II] + k[568]*y[IDX_H2OI]*y[IDX_HNOII] +
        k[570]*y[IDX_H2OI]*y[IDX_N2HII] + k[571]*y[IDX_H2OI]*y[IDX_O2HII] +
        k[573]*y[IDX_H2OI]*y[IDX_SiHII] + k[574]*y[IDX_H2OI]*y[IDX_SiH4II] +
        k[575]*y[IDX_H2OI]*y[IDX_SiH5II] + k[586]*y[IDX_H3II]*y[IDX_H2OI] -
        k[607]*y[IDX_H3OII]*y[IDX_H2COI] - k[608]*y[IDX_H3OII]*y[IDX_HCNI] -
        k[609]*y[IDX_H3OII]*y[IDX_HNCI] - k[610]*y[IDX_H3OII]*y[IDX_SiI] -
        k[611]*y[IDX_H3OII]*y[IDX_SiH2I] - k[612]*y[IDX_H3OII]*y[IDX_SiHI] -
        k[613]*y[IDX_H3OII]*y[IDX_SiOI] + k[754]*y[IDX_NHII]*y[IDX_H2OI] +
        k[771]*y[IDX_NH2II]*y[IDX_H2OI] - k[783]*y[IDX_NH2I]*y[IDX_H3OII] +
        k[797]*y[IDX_NHI]*y[IDX_H2OII] + k[840]*y[IDX_OHII]*y[IDX_H2OI] +
        k[852]*y[IDX_OHI]*y[IDX_H2OII] - k[1286]*y[IDX_H3OII];
    ydot[IDX_HCNI] = 0.0 + k[1]*y[IDX_HII]*y[IDX_HNCI] -
        k[65]*y[IDX_CNII]*y[IDX_HCNI] - k[81]*y[IDX_HII]*y[IDX_HCNI] -
        k[107]*y[IDX_H2II]*y[IDX_HCNI] + k[124]*y[IDX_H2OI]*y[IDX_HCNII] +
        k[129]*y[IDX_HI]*y[IDX_HCNII] + k[132]*y[IDX_HCNII]*y[IDX_NOI] +
        k[133]*y[IDX_HCNII]*y[IDX_O2I] - k[134]*y[IDX_HCNI]*y[IDX_COII] -
        k[135]*y[IDX_HCNI]*y[IDX_N2II] - k[161]*y[IDX_NII]*y[IDX_HCNI] +
        k[196]*y[IDX_NH3I]*y[IDX_HCNII] + k[254]*y[IDX_H2CNI] -
        k[259]*y[IDX_HCNI] + k[328]*y[IDX_HCNHII]*y[IDX_EM] -
        k[405]*y[IDX_CHII]*y[IDX_HCNI] + k[427]*y[IDX_CH2I]*y[IDX_HCNHII] +
        k[464]*y[IDX_CHI]*y[IDX_HCNHII] + k[479]*y[IDX_CNII]*y[IDX_H2COI] -
        k[556]*y[IDX_H2OII]*y[IDX_HCNI] - k[587]*y[IDX_H3II]*y[IDX_HCNI] -
        k[608]*y[IDX_H3OII]*y[IDX_HCNI] - k[623]*y[IDX_HCNII]*y[IDX_HCNI] -
        k[627]*y[IDX_HCNI]*y[IDX_H2COII] - k[628]*y[IDX_HCNI]*y[IDX_H3COII] -
        k[629]*y[IDX_HCNI]*y[IDX_HCOII] - k[630]*y[IDX_HCNI]*y[IDX_HNOII] -
        k[631]*y[IDX_HCNI]*y[IDX_N2HII] - k[632]*y[IDX_HCNI]*y[IDX_O2HII] +
        k[633]*y[IDX_HCNHII]*y[IDX_H2COI] - k[676]*y[IDX_HeII]*y[IDX_HCNI] -
        k[677]*y[IDX_HeII]*y[IDX_HCNI] - k[678]*y[IDX_HeII]*y[IDX_HCNI] -
        k[679]*y[IDX_HeII]*y[IDX_HCNI] - k[758]*y[IDX_NHII]*y[IDX_HCNI] -
        k[773]*y[IDX_NH2II]*y[IDX_HCNI] + k[785]*y[IDX_NH2I]*y[IDX_HCNHII] -
        k[814]*y[IDX_OII]*y[IDX_HCNI] - k[815]*y[IDX_OII]*y[IDX_HCNI] -
        k[841]*y[IDX_OHII]*y[IDX_HCNI] + k[866]*y[IDX_CI]*y[IDX_NH2I] +
        k[880]*y[IDX_CH2I]*y[IDX_CNI] + k[884]*y[IDX_CH2I]*y[IDX_N2I] +
        k[887]*y[IDX_CH2I]*y[IDX_NOI] + k[902]*y[IDX_CH3I]*y[IDX_CNI] +
        k[910]*y[IDX_CH3I]*y[IDX_NOI] + k[920]*y[IDX_CH4I]*y[IDX_CNI] +
        k[927]*y[IDX_CHI]*y[IDX_N2I] + k[930]*y[IDX_CHI]*y[IDX_NOI] +
        k[942]*y[IDX_CNI]*y[IDX_H2COI] + k[943]*y[IDX_CNI]*y[IDX_HCOI] +
        k[944]*y[IDX_CNI]*y[IDX_HNOI] + k[950]*y[IDX_CNI]*y[IDX_SiH4I] +
        k[959]*y[IDX_H2I]*y[IDX_CNI] + k[973]*y[IDX_HI]*y[IDX_H2CNI] -
        k[976]*y[IDX_HI]*y[IDX_HCNI] + k[979]*y[IDX_HI]*y[IDX_HNCI] +
        k[993]*y[IDX_HI]*y[IDX_OCNI] + k[1005]*y[IDX_NI]*y[IDX_CH2I] +
        k[1009]*y[IDX_NI]*y[IDX_CH3I] + k[1010]*y[IDX_NI]*y[IDX_CH3I] +
        k[1013]*y[IDX_NI]*y[IDX_H2CNI] + k[1015]*y[IDX_NI]*y[IDX_HCOI] +
        k[1033]*y[IDX_NH3I]*y[IDX_CNI] + k[1035]*y[IDX_NHI]*y[IDX_CNI] -
        k[1063]*y[IDX_OI]*y[IDX_HCNI] - k[1064]*y[IDX_OI]*y[IDX_HCNI] -
        k[1065]*y[IDX_OI]*y[IDX_HCNI] + k[1090]*y[IDX_OHI]*y[IDX_CNI] -
        k[1094]*y[IDX_OHI]*y[IDX_HCNI] - k[1095]*y[IDX_OHI]*y[IDX_HCNI] +
        k[1135]*y[IDX_H2CNI] - k[1147]*y[IDX_HCNI] - k[1266]*y[IDX_HCNI] +
        k[1323]*y[IDX_GHCNI] + k[1324]*y[IDX_GHCNI] + k[1325]*y[IDX_GHCNI] +
        k[1326]*y[IDX_GHCNI];
    ydot[IDX_HCNII] = 0.0 + k[65]*y[IDX_CNII]*y[IDX_HCNI] +
        k[81]*y[IDX_HII]*y[IDX_HCNI] + k[107]*y[IDX_H2II]*y[IDX_HCNI] -
        k[124]*y[IDX_H2OI]*y[IDX_HCNII] - k[129]*y[IDX_HI]*y[IDX_HCNII] -
        k[132]*y[IDX_HCNII]*y[IDX_NOI] - k[133]*y[IDX_HCNII]*y[IDX_O2I] +
        k[134]*y[IDX_HCNI]*y[IDX_COII] + k[135]*y[IDX_HCNI]*y[IDX_N2II] +
        k[161]*y[IDX_NII]*y[IDX_HCNI] - k[196]*y[IDX_NH3I]*y[IDX_HCNII] -
        k[326]*y[IDX_HCNII]*y[IDX_EM] + k[373]*y[IDX_CII]*y[IDX_NH2I] +
        k[374]*y[IDX_CII]*y[IDX_NH3I] - k[385]*y[IDX_CI]*y[IDX_HCNII] +
        k[409]*y[IDX_CHII]*y[IDX_NH2I] - k[426]*y[IDX_CH2I]*y[IDX_HCNII] -
        k[454]*y[IDX_CH4I]*y[IDX_HCNII] - k[463]*y[IDX_CHI]*y[IDX_HCNII] +
        k[480]*y[IDX_CNII]*y[IDX_HCOI] + k[482]*y[IDX_CNI]*y[IDX_HNOII] +
        k[483]*y[IDX_CNI]*y[IDX_O2HII] + k[513]*y[IDX_H2II]*y[IDX_CNI] +
        k[531]*y[IDX_H2I]*y[IDX_CNII] - k[535]*y[IDX_H2I]*y[IDX_HCNII] +
        k[560]*y[IDX_H2OI]*y[IDX_CNII] - k[565]*y[IDX_H2OI]*y[IDX_HCNII] +
        k[581]*y[IDX_H3II]*y[IDX_CNI] - k[620]*y[IDX_HCNII]*y[IDX_CO2I] -
        k[621]*y[IDX_HCNII]*y[IDX_COI] - k[622]*y[IDX_HCNII]*y[IDX_H2COI] -
        k[623]*y[IDX_HCNII]*y[IDX_HCNI] - k[624]*y[IDX_HCNII]*y[IDX_HCOI] -
        k[625]*y[IDX_HCNII]*y[IDX_HCOI] - k[626]*y[IDX_HCNII]*y[IDX_HNCI] +
        k[717]*y[IDX_NII]*y[IDX_CH4I] + k[736]*y[IDX_NI]*y[IDX_CH2II] +
        k[747]*y[IDX_NHII]*y[IDX_CNI] - k[784]*y[IDX_NH2I]*y[IDX_HCNII] -
        k[793]*y[IDX_NH3I]*y[IDX_HCNII] - k[798]*y[IDX_NHI]*y[IDX_HCNII] +
        k[836]*y[IDX_OHII]*y[IDX_CNI] - k[853]*y[IDX_OHI]*y[IDX_HCNII] -
        k[1282]*y[IDX_HCNII];
    ydot[IDX_HCNHII] = 0.0 - k[327]*y[IDX_HCNHII]*y[IDX_EM] -
        k[328]*y[IDX_HCNHII]*y[IDX_EM] - k[329]*y[IDX_HCNHII]*y[IDX_EM] +
        k[405]*y[IDX_CHII]*y[IDX_HCNI] + k[407]*y[IDX_CHII]*y[IDX_HNCI] -
        k[427]*y[IDX_CH2I]*y[IDX_HCNHII] - k[428]*y[IDX_CH2I]*y[IDX_HCNHII] +
        k[454]*y[IDX_CH4I]*y[IDX_HCNII] - k[464]*y[IDX_CHI]*y[IDX_HCNHII] -
        k[465]*y[IDX_CHI]*y[IDX_HCNHII] + k[535]*y[IDX_H2I]*y[IDX_HCNII] +
        k[556]*y[IDX_H2OII]*y[IDX_HCNI] + k[559]*y[IDX_H2OII]*y[IDX_HNCI] +
        k[587]*y[IDX_H3II]*y[IDX_HCNI] + k[589]*y[IDX_H3II]*y[IDX_HNCI] +
        k[608]*y[IDX_H3OII]*y[IDX_HCNI] + k[609]*y[IDX_H3OII]*y[IDX_HNCI] +
        k[623]*y[IDX_HCNII]*y[IDX_HCNI] + k[625]*y[IDX_HCNII]*y[IDX_HCOI] +
        k[626]*y[IDX_HCNII]*y[IDX_HNCI] + k[627]*y[IDX_HCNI]*y[IDX_H2COII] +
        k[628]*y[IDX_HCNI]*y[IDX_H3COII] + k[629]*y[IDX_HCNI]*y[IDX_HCOII] +
        k[630]*y[IDX_HCNI]*y[IDX_HNOII] + k[631]*y[IDX_HCNI]*y[IDX_N2HII] +
        k[632]*y[IDX_HCNI]*y[IDX_O2HII] - k[633]*y[IDX_HCNHII]*y[IDX_H2COI] -
        k[634]*y[IDX_HCNHII]*y[IDX_H2COI] + k[646]*y[IDX_HNCI]*y[IDX_H2COII] +
        k[647]*y[IDX_HNCI]*y[IDX_H3COII] + k[648]*y[IDX_HNCI]*y[IDX_HCOII] +
        k[649]*y[IDX_HNCI]*y[IDX_HNOII] + k[650]*y[IDX_HNCI]*y[IDX_N2HII] +
        k[651]*y[IDX_HNCI]*y[IDX_O2HII] + k[718]*y[IDX_NII]*y[IDX_CH4I] +
        k[758]*y[IDX_NHII]*y[IDX_HCNI] + k[760]*y[IDX_NHII]*y[IDX_HNCI] +
        k[773]*y[IDX_NH2II]*y[IDX_HCNI] + k[775]*y[IDX_NH2II]*y[IDX_HNCI] -
        k[785]*y[IDX_NH2I]*y[IDX_HCNHII] - k[786]*y[IDX_NH2I]*y[IDX_HCNHII] +
        k[793]*y[IDX_NH3I]*y[IDX_HCNII] + k[794]*y[IDX_NHI]*y[IDX_CH3II] +
        k[841]*y[IDX_OHII]*y[IDX_HCNI] + k[844]*y[IDX_OHII]*y[IDX_HNCI] -
        k[1301]*y[IDX_HCNHII];
    ydot[IDX_HCOI] = 0.0 - k[17]*y[IDX_CII]*y[IDX_HCOI] -
        k[31]*y[IDX_CHII]*y[IDX_HCOI] - k[46]*y[IDX_CH3II]*y[IDX_HCOI] -
        k[66]*y[IDX_CNII]*y[IDX_HCOI] - k[71]*y[IDX_COII]*y[IDX_HCOI] -
        k[82]*y[IDX_HII]*y[IDX_HCOI] - k[108]*y[IDX_H2II]*y[IDX_HCOI] -
        k[118]*y[IDX_H2OII]*y[IDX_HCOI] - k[136]*y[IDX_HCOI]*y[IDX_H2COII] -
        k[137]*y[IDX_HCOI]*y[IDX_O2II] - k[138]*y[IDX_HCOI]*y[IDX_SiOII] +
        k[149]*y[IDX_MgI]*y[IDX_HCOII] - k[162]*y[IDX_NII]*y[IDX_HCOI] -
        k[171]*y[IDX_N2II]*y[IDX_HCOI] - k[180]*y[IDX_NH2II]*y[IDX_HCOI] -
        k[189]*y[IDX_NH3II]*y[IDX_HCOI] - k[211]*y[IDX_OII]*y[IDX_HCOI] -
        k[221]*y[IDX_OHII]*y[IDX_HCOI] - k[260]*y[IDX_HCOI] - k[261]*y[IDX_HCOI]
        + k[309]*y[IDX_H2COII]*y[IDX_EM] + k[321]*y[IDX_H3COII]*y[IDX_EM] +
        k[366]*y[IDX_CII]*y[IDX_CH3OHI] - k[372]*y[IDX_CII]*y[IDX_HCOI] -
        k[406]*y[IDX_CHII]*y[IDX_HCOI] + k[413]*y[IDX_CHII]*y[IDX_O2I] -
        k[419]*y[IDX_CH2II]*y[IDX_HCOI] + k[423]*y[IDX_CH2I]*y[IDX_H2COII] -
        k[441]*y[IDX_CH3II]*y[IDX_HCOI] + k[459]*y[IDX_CHI]*y[IDX_H2COII] -
        k[480]*y[IDX_CNII]*y[IDX_HCOI] + k[484]*y[IDX_COII]*y[IDX_H2COI] -
        k[500]*y[IDX_HII]*y[IDX_HCOI] - k[501]*y[IDX_HII]*y[IDX_HCOI] -
        k[519]*y[IDX_H2II]*y[IDX_HCOI] + k[548]*y[IDX_H2COII]*y[IDX_H2COI] -
        k[557]*y[IDX_H2OII]*y[IDX_HCOI] - k[558]*y[IDX_H2OII]*y[IDX_HCOI] +
        k[563]*y[IDX_H2OI]*y[IDX_H2COII] - k[588]*y[IDX_H3II]*y[IDX_HCOI] -
        k[624]*y[IDX_HCNII]*y[IDX_HCOI] - k[625]*y[IDX_HCNII]*y[IDX_HCOI] +
        k[627]*y[IDX_HCNI]*y[IDX_H2COII] - k[636]*y[IDX_HCOII]*y[IDX_HCOI] -
        k[641]*y[IDX_HCOI]*y[IDX_H2COII] - k[642]*y[IDX_HCOI]*y[IDX_HNOII] -
        k[643]*y[IDX_HCOI]*y[IDX_N2HII] - k[644]*y[IDX_HCOI]*y[IDX_O2II] -
        k[645]*y[IDX_HCOI]*y[IDX_O2HII] + k[646]*y[IDX_HNCI]*y[IDX_H2COII] -
        k[680]*y[IDX_HeII]*y[IDX_HCOI] - k[681]*y[IDX_HeII]*y[IDX_HCOI] -
        k[682]*y[IDX_HeII]*y[IDX_HCOI] - k[723]*y[IDX_NII]*y[IDX_HCOI] -
        k[731]*y[IDX_N2II]*y[IDX_HCOI] + k[750]*y[IDX_NHII]*y[IDX_CO2I] -
        k[759]*y[IDX_NHII]*y[IDX_HCOI] + k[770]*y[IDX_NH2II]*y[IDX_H2COI] -
        k[774]*y[IDX_NH2II]*y[IDX_HCOI] + k[780]*y[IDX_NH2I]*y[IDX_H2COII] -
        k[816]*y[IDX_OII]*y[IDX_HCOI] - k[842]*y[IDX_OHII]*y[IDX_HCOI] -
        k[843]*y[IDX_OHII]*y[IDX_HCOI] - k[864]*y[IDX_CI]*y[IDX_HCOI] +
        k[881]*y[IDX_CH2I]*y[IDX_H2COI] - k[882]*y[IDX_CH2I]*y[IDX_HCOI] +
        k[893]*y[IDX_CH2I]*y[IDX_O2I] + k[896]*y[IDX_CH2I]*y[IDX_OI] +
        k[903]*y[IDX_CH3I]*y[IDX_H2COI] - k[905]*y[IDX_CH3I]*y[IDX_HCOI] +
        k[912]*y[IDX_CH3I]*y[IDX_O2I] + k[923]*y[IDX_CHI]*y[IDX_CO2I] +
        k[924]*y[IDX_CHI]*y[IDX_H2COI] - k[925]*y[IDX_CHI]*y[IDX_HCOI] +
        k[931]*y[IDX_CHI]*y[IDX_NOI] + k[936]*y[IDX_CHI]*y[IDX_O2I] +
        k[937]*y[IDX_CHI]*y[IDX_O2HI] + k[941]*y[IDX_CHI]*y[IDX_OHI] +
        k[942]*y[IDX_CNI]*y[IDX_H2COI] - k[943]*y[IDX_CNI]*y[IDX_HCOI] +
        k[974]*y[IDX_HI]*y[IDX_H2COI] - k[977]*y[IDX_HI]*y[IDX_HCOI] -
        k[978]*y[IDX_HI]*y[IDX_HCOI] - k[997]*y[IDX_HCOI]*y[IDX_HCOI] -
        k[997]*y[IDX_HCOI]*y[IDX_HCOI] - k[998]*y[IDX_HCOI]*y[IDX_HCOI] -
        k[998]*y[IDX_HCOI]*y[IDX_HCOI] - k[999]*y[IDX_HCOI]*y[IDX_HNOI] -
        k[1000]*y[IDX_HCOI]*y[IDX_NOI] - k[1001]*y[IDX_HCOI]*y[IDX_O2I] -
        k[1002]*y[IDX_HCOI]*y[IDX_O2I] - k[1003]*y[IDX_HCOI]*y[IDX_O2HI] -
        k[1014]*y[IDX_NI]*y[IDX_HCOI] - k[1015]*y[IDX_NI]*y[IDX_HCOI] -
        k[1016]*y[IDX_NI]*y[IDX_HCOI] + k[1061]*y[IDX_OI]*y[IDX_H2COI] -
        k[1066]*y[IDX_OI]*y[IDX_HCOI] - k[1067]*y[IDX_OI]*y[IDX_HCOI] +
        k[1093]*y[IDX_OHI]*y[IDX_H2COI] - k[1096]*y[IDX_OHI]*y[IDX_HCOI] -
        k[1149]*y[IDX_HCOI] - k[1150]*y[IDX_HCOI] - k[1261]*y[IDX_HCOI];
    ydot[IDX_HCOII] = 0.0 + k[0]*y[IDX_CHI]*y[IDX_OI] +
        k[5]*y[IDX_H2I]*y[IDX_HOCII] + k[17]*y[IDX_CII]*y[IDX_HCOI] +
        k[31]*y[IDX_CHII]*y[IDX_HCOI] + k[46]*y[IDX_CH3II]*y[IDX_HCOI] +
        k[66]*y[IDX_CNII]*y[IDX_HCOI] + k[71]*y[IDX_COII]*y[IDX_HCOI] +
        k[82]*y[IDX_HII]*y[IDX_HCOI] + k[108]*y[IDX_H2II]*y[IDX_HCOI] +
        k[118]*y[IDX_H2OII]*y[IDX_HCOI] + k[136]*y[IDX_HCOI]*y[IDX_H2COII] +
        k[137]*y[IDX_HCOI]*y[IDX_O2II] + k[138]*y[IDX_HCOI]*y[IDX_SiOII] -
        k[149]*y[IDX_MgI]*y[IDX_HCOII] + k[162]*y[IDX_NII]*y[IDX_HCOI] +
        k[171]*y[IDX_N2II]*y[IDX_HCOI] + k[180]*y[IDX_NH2II]*y[IDX_HCOI] +
        k[189]*y[IDX_NH3II]*y[IDX_HCOI] + k[211]*y[IDX_OII]*y[IDX_HCOI] +
        k[221]*y[IDX_OHII]*y[IDX_HCOI] + k[261]*y[IDX_HCOI] -
        k[330]*y[IDX_HCOII]*y[IDX_EM] + k[369]*y[IDX_CII]*y[IDX_H2COI] +
        k[370]*y[IDX_CII]*y[IDX_H2OI] + k[384]*y[IDX_CI]*y[IDX_H3OII] -
        k[386]*y[IDX_CI]*y[IDX_HCOII] + k[398]*y[IDX_CHII]*y[IDX_CO2I] +
        k[401]*y[IDX_CHII]*y[IDX_H2COI] + k[404]*y[IDX_CHII]*y[IDX_H2OI] +
        k[412]*y[IDX_CHII]*y[IDX_O2I] + k[417]*y[IDX_CH2II]*y[IDX_H2COI] +
        k[420]*y[IDX_CH2II]*y[IDX_O2I] + k[421]*y[IDX_CH2II]*y[IDX_OI] +
        k[422]*y[IDX_CH2I]*y[IDX_COII] - k[429]*y[IDX_CH2I]*y[IDX_HCOII] +
        k[440]*y[IDX_CH3II]*y[IDX_H2COI] + k[444]*y[IDX_CH3II]*y[IDX_OI] +
        k[448]*y[IDX_CH4II]*y[IDX_COI] + k[451]*y[IDX_CH4I]*y[IDX_COII] +
        k[458]*y[IDX_CHI]*y[IDX_COII] - k[466]*y[IDX_CHI]*y[IDX_HCOII] +
        k[473]*y[IDX_CHI]*y[IDX_O2II] + k[478]*y[IDX_CHI]*y[IDX_SiOII] +
        k[479]*y[IDX_CNII]*y[IDX_H2COI] + k[484]*y[IDX_COII]*y[IDX_H2COI] +
        k[485]*y[IDX_COI]*y[IDX_HCO2II] + k[486]*y[IDX_COI]*y[IDX_HNOII] +
        k[487]*y[IDX_COI]*y[IDX_N2HII] + k[488]*y[IDX_COI]*y[IDX_O2HII] +
        k[489]*y[IDX_COI]*y[IDX_SiH4II] + k[494]*y[IDX_HII]*y[IDX_CH3OHI] +
        k[496]*y[IDX_HII]*y[IDX_CO2I] + k[498]*y[IDX_HII]*y[IDX_H2COI] +
        k[515]*y[IDX_H2II]*y[IDX_COI] + k[517]*y[IDX_H2II]*y[IDX_H2COI] +
        k[532]*y[IDX_H2I]*y[IDX_COII] + k[549]*y[IDX_H2COII]*y[IDX_O2I] +
        k[551]*y[IDX_H2COI]*y[IDX_O2II] + k[553]*y[IDX_H2OII]*y[IDX_COI] +
        k[561]*y[IDX_H2OI]*y[IDX_CNII] + k[562]*y[IDX_H2OI]*y[IDX_COII] -
        k[566]*y[IDX_H2OI]*y[IDX_HCOII] + k[583]*y[IDX_H3II]*y[IDX_COI] +
        k[621]*y[IDX_HCNII]*y[IDX_COI] - k[629]*y[IDX_HCNI]*y[IDX_HCOII] -
        k[635]*y[IDX_HCOII]*y[IDX_H2COI] - k[636]*y[IDX_HCOII]*y[IDX_HCOI] -
        k[637]*y[IDX_HCOII]*y[IDX_SiH2I] - k[638]*y[IDX_HCOII]*y[IDX_SiH4I] -
        k[639]*y[IDX_HCOII]*y[IDX_SiHI] - k[640]*y[IDX_HCOII]*y[IDX_SiOI] -
        k[648]*y[IDX_HNCI]*y[IDX_HCOII] + k[671]*y[IDX_HeII]*y[IDX_H2COI] +
        k[721]*y[IDX_NII]*y[IDX_H2COI] + k[730]*y[IDX_N2II]*y[IDX_H2COI] +
        k[751]*y[IDX_NHII]*y[IDX_COI] + k[753]*y[IDX_NHII]*y[IDX_H2COI] +
        k[779]*y[IDX_NH2I]*y[IDX_COII] - k[787]*y[IDX_NH2I]*y[IDX_HCOII] +
        k[792]*y[IDX_NH3I]*y[IDX_COII] + k[795]*y[IDX_NHI]*y[IDX_COII] -
        k[799]*y[IDX_NHI]*y[IDX_HCOII] + k[813]*y[IDX_OII]*y[IDX_H2COI] +
        k[814]*y[IDX_OII]*y[IDX_HCNI] + k[824]*y[IDX_OI]*y[IDX_HCO2II] +
        k[838]*y[IDX_OHII]*y[IDX_COI] + k[851]*y[IDX_OHI]*y[IDX_COII] -
        k[854]*y[IDX_OHI]*y[IDX_HCOII] - k[855]*y[IDX_OHI]*y[IDX_HCOII] -
        k[861]*y[IDX_SiI]*y[IDX_HCOII] + k[1139]*y[IDX_H2COI] -
        k[1148]*y[IDX_HCOII] + k[1150]*y[IDX_HCOI] - k[1281]*y[IDX_HCOII];
    ydot[IDX_HCO2II] = 0.0 - k[331]*y[IDX_HCO2II]*y[IDX_EM] -
        k[332]*y[IDX_HCO2II]*y[IDX_EM] - k[333]*y[IDX_HCO2II]*y[IDX_EM] -
        k[387]*y[IDX_CI]*y[IDX_HCO2II] + k[447]*y[IDX_CH4II]*y[IDX_CO2I] -
        k[485]*y[IDX_COI]*y[IDX_HCO2II] + k[514]*y[IDX_H2II]*y[IDX_CO2I] -
        k[567]*y[IDX_H2OI]*y[IDX_HCO2II] + k[582]*y[IDX_H3II]*y[IDX_CO2I] +
        k[620]*y[IDX_HCNII]*y[IDX_CO2I] + k[652]*y[IDX_HNOII]*y[IDX_CO2I] +
        k[734]*y[IDX_N2HII]*y[IDX_CO2I] + k[748]*y[IDX_NHII]*y[IDX_CO2I] +
        k[821]*y[IDX_O2HII]*y[IDX_CO2I] - k[824]*y[IDX_OI]*y[IDX_HCO2II] +
        k[837]*y[IDX_OHII]*y[IDX_CO2I] + k[855]*y[IDX_OHI]*y[IDX_HCOII] -
        k[1287]*y[IDX_HCO2II];
    ydot[IDX_HeI] = 0.0 + k[115]*y[IDX_H2I]*y[IDX_HeII] +
        k[130]*y[IDX_HI]*y[IDX_HeII] + k[139]*y[IDX_HeII]*y[IDX_CI] +
        k[140]*y[IDX_HeII]*y[IDX_CH4I] + k[141]*y[IDX_HeII]*y[IDX_CHI] +
        k[142]*y[IDX_HeII]*y[IDX_H2COI] + k[143]*y[IDX_HeII]*y[IDX_H2OI] +
        k[144]*y[IDX_HeII]*y[IDX_N2I] + k[145]*y[IDX_HeII]*y[IDX_NH3I] +
        k[146]*y[IDX_HeII]*y[IDX_O2I] + k[147]*y[IDX_HeII]*y[IDX_SiI] -
        k[237]*y[IDX_HeI] - k[265]*y[IDX_HeI] + k[336]*y[IDX_HeHII]*y[IDX_EM] -
        k[520]*y[IDX_H2II]*y[IDX_HeI] + k[536]*y[IDX_H2I]*y[IDX_HeII] +
        k[537]*y[IDX_H2I]*y[IDX_HeHII] + k[618]*y[IDX_HI]*y[IDX_HeHII] +
        k[653]*y[IDX_HeII]*y[IDX_CH2I] + k[654]*y[IDX_HeII]*y[IDX_CH2I] +
        k[655]*y[IDX_HeII]*y[IDX_CH3I] + k[656]*y[IDX_HeII]*y[IDX_CH3OHI] +
        k[657]*y[IDX_HeII]*y[IDX_CH3OHI] + k[658]*y[IDX_HeII]*y[IDX_CH4I] +
        k[659]*y[IDX_HeII]*y[IDX_CH4I] + k[660]*y[IDX_HeII]*y[IDX_CH4I] +
        k[661]*y[IDX_HeII]*y[IDX_CH4I] + k[662]*y[IDX_HeII]*y[IDX_CHI] +
        k[663]*y[IDX_HeII]*y[IDX_CNI] + k[664]*y[IDX_HeII]*y[IDX_CNI] +
        k[665]*y[IDX_HeII]*y[IDX_CO2I] + k[666]*y[IDX_HeII]*y[IDX_CO2I] +
        k[667]*y[IDX_HeII]*y[IDX_CO2I] + k[668]*y[IDX_HeII]*y[IDX_CO2I] +
        k[669]*y[IDX_HeII]*y[IDX_COI] + k[670]*y[IDX_HeII]*y[IDX_H2COI] +
        k[671]*y[IDX_HeII]*y[IDX_H2COI] + k[672]*y[IDX_HeII]*y[IDX_H2COI] +
        k[673]*y[IDX_HeII]*y[IDX_H2OI] + k[674]*y[IDX_HeII]*y[IDX_H2OI] +
        k[675]*y[IDX_HeII]*y[IDX_H2SiOI] + k[676]*y[IDX_HeII]*y[IDX_HCNI] +
        k[677]*y[IDX_HeII]*y[IDX_HCNI] + k[678]*y[IDX_HeII]*y[IDX_HCNI] +
        k[679]*y[IDX_HeII]*y[IDX_HCNI] + k[680]*y[IDX_HeII]*y[IDX_HCOI] +
        k[682]*y[IDX_HeII]*y[IDX_HCOI] + k[683]*y[IDX_HeII]*y[IDX_HNCI] +
        k[684]*y[IDX_HeII]*y[IDX_HNCI] + k[685]*y[IDX_HeII]*y[IDX_HNCI] +
        k[686]*y[IDX_HeII]*y[IDX_HNOI] + k[687]*y[IDX_HeII]*y[IDX_HNOI] +
        k[688]*y[IDX_HeII]*y[IDX_N2I] + k[689]*y[IDX_HeII]*y[IDX_NH2I] +
        k[690]*y[IDX_HeII]*y[IDX_NH2I] + k[691]*y[IDX_HeII]*y[IDX_NH3I] +
        k[692]*y[IDX_HeII]*y[IDX_NH3I] + k[693]*y[IDX_HeII]*y[IDX_NHI] +
        k[694]*y[IDX_HeII]*y[IDX_NOI] + k[695]*y[IDX_HeII]*y[IDX_NOI] +
        k[696]*y[IDX_HeII]*y[IDX_O2I] + k[697]*y[IDX_HeII]*y[IDX_OCNI] +
        k[698]*y[IDX_HeII]*y[IDX_OCNI] + k[699]*y[IDX_HeII]*y[IDX_OHI] +
        k[700]*y[IDX_HeII]*y[IDX_SiC3I] + k[701]*y[IDX_HeII]*y[IDX_SiCI] +
        k[702]*y[IDX_HeII]*y[IDX_SiCI] + k[703]*y[IDX_HeII]*y[IDX_SiH2I] +
        k[704]*y[IDX_HeII]*y[IDX_SiH2I] + k[705]*y[IDX_HeII]*y[IDX_SiH3I] +
        k[706]*y[IDX_HeII]*y[IDX_SiH3I] + k[707]*y[IDX_HeII]*y[IDX_SiH4I] +
        k[708]*y[IDX_HeII]*y[IDX_SiH4I] + k[709]*y[IDX_HeII]*y[IDX_SiHI] +
        k[710]*y[IDX_HeII]*y[IDX_SiOI] + k[711]*y[IDX_HeII]*y[IDX_SiOI] -
        k[1198]*y[IDX_HII]*y[IDX_HeI] + k[1218]*y[IDX_HeII]*y[IDX_EM];
    ydot[IDX_HeII] = 0.0 - k[115]*y[IDX_H2I]*y[IDX_HeII] -
        k[130]*y[IDX_HI]*y[IDX_HeII] - k[139]*y[IDX_HeII]*y[IDX_CI] -
        k[140]*y[IDX_HeII]*y[IDX_CH4I] - k[141]*y[IDX_HeII]*y[IDX_CHI] -
        k[142]*y[IDX_HeII]*y[IDX_H2COI] - k[143]*y[IDX_HeII]*y[IDX_H2OI] -
        k[144]*y[IDX_HeII]*y[IDX_N2I] - k[145]*y[IDX_HeII]*y[IDX_NH3I] -
        k[146]*y[IDX_HeII]*y[IDX_O2I] - k[147]*y[IDX_HeII]*y[IDX_SiI] +
        k[237]*y[IDX_HeI] + k[265]*y[IDX_HeI] - k[536]*y[IDX_H2I]*y[IDX_HeII] -
        k[653]*y[IDX_HeII]*y[IDX_CH2I] - k[654]*y[IDX_HeII]*y[IDX_CH2I] -
        k[655]*y[IDX_HeII]*y[IDX_CH3I] - k[656]*y[IDX_HeII]*y[IDX_CH3OHI] -
        k[657]*y[IDX_HeII]*y[IDX_CH3OHI] - k[658]*y[IDX_HeII]*y[IDX_CH4I] -
        k[659]*y[IDX_HeII]*y[IDX_CH4I] - k[660]*y[IDX_HeII]*y[IDX_CH4I] -
        k[661]*y[IDX_HeII]*y[IDX_CH4I] - k[662]*y[IDX_HeII]*y[IDX_CHI] -
        k[663]*y[IDX_HeII]*y[IDX_CNI] - k[664]*y[IDX_HeII]*y[IDX_CNI] -
        k[665]*y[IDX_HeII]*y[IDX_CO2I] - k[666]*y[IDX_HeII]*y[IDX_CO2I] -
        k[667]*y[IDX_HeII]*y[IDX_CO2I] - k[668]*y[IDX_HeII]*y[IDX_CO2I] -
        k[669]*y[IDX_HeII]*y[IDX_COI] - k[670]*y[IDX_HeII]*y[IDX_H2COI] -
        k[671]*y[IDX_HeII]*y[IDX_H2COI] - k[672]*y[IDX_HeII]*y[IDX_H2COI] -
        k[673]*y[IDX_HeII]*y[IDX_H2OI] - k[674]*y[IDX_HeII]*y[IDX_H2OI] -
        k[675]*y[IDX_HeII]*y[IDX_H2SiOI] - k[676]*y[IDX_HeII]*y[IDX_HCNI] -
        k[677]*y[IDX_HeII]*y[IDX_HCNI] - k[678]*y[IDX_HeII]*y[IDX_HCNI] -
        k[679]*y[IDX_HeII]*y[IDX_HCNI] - k[680]*y[IDX_HeII]*y[IDX_HCOI] -
        k[681]*y[IDX_HeII]*y[IDX_HCOI] - k[682]*y[IDX_HeII]*y[IDX_HCOI] -
        k[683]*y[IDX_HeII]*y[IDX_HNCI] - k[684]*y[IDX_HeII]*y[IDX_HNCI] -
        k[685]*y[IDX_HeII]*y[IDX_HNCI] - k[686]*y[IDX_HeII]*y[IDX_HNOI] -
        k[687]*y[IDX_HeII]*y[IDX_HNOI] - k[688]*y[IDX_HeII]*y[IDX_N2I] -
        k[689]*y[IDX_HeII]*y[IDX_NH2I] - k[690]*y[IDX_HeII]*y[IDX_NH2I] -
        k[691]*y[IDX_HeII]*y[IDX_NH3I] - k[692]*y[IDX_HeII]*y[IDX_NH3I] -
        k[693]*y[IDX_HeII]*y[IDX_NHI] - k[694]*y[IDX_HeII]*y[IDX_NOI] -
        k[695]*y[IDX_HeII]*y[IDX_NOI] - k[696]*y[IDX_HeII]*y[IDX_O2I] -
        k[697]*y[IDX_HeII]*y[IDX_OCNI] - k[698]*y[IDX_HeII]*y[IDX_OCNI] -
        k[699]*y[IDX_HeII]*y[IDX_OHI] - k[700]*y[IDX_HeII]*y[IDX_SiC3I] -
        k[701]*y[IDX_HeII]*y[IDX_SiCI] - k[702]*y[IDX_HeII]*y[IDX_SiCI] -
        k[703]*y[IDX_HeII]*y[IDX_SiH2I] - k[704]*y[IDX_HeII]*y[IDX_SiH2I] -
        k[705]*y[IDX_HeII]*y[IDX_SiH3I] - k[706]*y[IDX_HeII]*y[IDX_SiH3I] -
        k[707]*y[IDX_HeII]*y[IDX_SiH4I] - k[708]*y[IDX_HeII]*y[IDX_SiH4I] -
        k[709]*y[IDX_HeII]*y[IDX_SiHI] - k[710]*y[IDX_HeII]*y[IDX_SiOI] -
        k[711]*y[IDX_HeII]*y[IDX_SiOI] - k[1218]*y[IDX_HeII]*y[IDX_EM];
    ydot[IDX_HeHII] = 0.0 - k[336]*y[IDX_HeHII]*y[IDX_EM] +
        k[520]*y[IDX_H2II]*y[IDX_HeI] - k[537]*y[IDX_H2I]*y[IDX_HeHII] -
        k[618]*y[IDX_HI]*y[IDX_HeHII] + k[681]*y[IDX_HeII]*y[IDX_HCOI] +
        k[1198]*y[IDX_HII]*y[IDX_HeI];
    ydot[IDX_HNCI] = 0.0 - k[1]*y[IDX_HII]*y[IDX_HNCI] - k[262]*y[IDX_HNCI]
        + k[329]*y[IDX_HCNHII]*y[IDX_EM] - k[407]*y[IDX_CHII]*y[IDX_HNCI] +
        k[428]*y[IDX_CH2I]*y[IDX_HCNHII] + k[465]*y[IDX_CHI]*y[IDX_HCNHII] -
        k[559]*y[IDX_H2OII]*y[IDX_HNCI] - k[589]*y[IDX_H3II]*y[IDX_HNCI] -
        k[609]*y[IDX_H3OII]*y[IDX_HNCI] - k[626]*y[IDX_HCNII]*y[IDX_HNCI] +
        k[634]*y[IDX_HCNHII]*y[IDX_H2COI] - k[646]*y[IDX_HNCI]*y[IDX_H2COII] -
        k[647]*y[IDX_HNCI]*y[IDX_H3COII] - k[648]*y[IDX_HNCI]*y[IDX_HCOII] -
        k[649]*y[IDX_HNCI]*y[IDX_HNOII] - k[650]*y[IDX_HNCI]*y[IDX_N2HII] -
        k[651]*y[IDX_HNCI]*y[IDX_O2HII] - k[683]*y[IDX_HeII]*y[IDX_HNCI] -
        k[684]*y[IDX_HeII]*y[IDX_HNCI] - k[685]*y[IDX_HeII]*y[IDX_HNCI] -
        k[760]*y[IDX_NHII]*y[IDX_HNCI] - k[775]*y[IDX_NH2II]*y[IDX_HNCI] +
        k[786]*y[IDX_NH2I]*y[IDX_HCNHII] - k[844]*y[IDX_OHII]*y[IDX_HNCI] +
        k[867]*y[IDX_CI]*y[IDX_NH2I] - k[979]*y[IDX_HI]*y[IDX_HNCI] +
        k[1004]*y[IDX_HNCOI]*y[IDX_CI] + k[1006]*y[IDX_NI]*y[IDX_CH2I] -
        k[1151]*y[IDX_HNCI] - k[1305]*y[IDX_HNCI] + k[1327]*y[IDX_GHNCI] +
        k[1328]*y[IDX_GHNCI] + k[1329]*y[IDX_GHNCI] + k[1330]*y[IDX_GHNCI];
    ydot[IDX_HNCOI] = 0.0 - k[263]*y[IDX_HNCOI] -
        k[502]*y[IDX_HII]*y[IDX_HNCOI] + k[888]*y[IDX_CH2I]*y[IDX_NOI] -
        k[1004]*y[IDX_HNCOI]*y[IDX_CI] - k[1152]*y[IDX_HNCOI] -
        k[1224]*y[IDX_HNCOI] + k[1375]*y[IDX_GHNCOI] + k[1376]*y[IDX_GHNCOI] +
        k[1377]*y[IDX_GHNCOI] + k[1378]*y[IDX_GHNCOI];
    ydot[IDX_HNOI] = 0.0 + k[204]*y[IDX_NOI]*y[IDX_HNOII] -
        k[264]*y[IDX_HNOI] + k[310]*y[IDX_H2NOII]*y[IDX_EM] -
        k[503]*y[IDX_HII]*y[IDX_HNOI] - k[590]*y[IDX_H3II]*y[IDX_HNOI] -
        k[686]*y[IDX_HeII]*y[IDX_HNOI] - k[687]*y[IDX_HeII]*y[IDX_HNOI] -
        k[883]*y[IDX_CH2I]*y[IDX_HNOI] - k[906]*y[IDX_CH3I]*y[IDX_HNOI] +
        k[909]*y[IDX_CH3I]*y[IDX_NO2I] - k[926]*y[IDX_CHI]*y[IDX_HNOI] -
        k[944]*y[IDX_CNI]*y[IDX_HNOI] - k[951]*y[IDX_COI]*y[IDX_HNOI] -
        k[980]*y[IDX_HI]*y[IDX_HNOI] - k[981]*y[IDX_HI]*y[IDX_HNOI] -
        k[982]*y[IDX_HI]*y[IDX_HNOI] - k[999]*y[IDX_HCOI]*y[IDX_HNOI] +
        k[1000]*y[IDX_HCOI]*y[IDX_NOI] - k[1017]*y[IDX_NI]*y[IDX_HNOI] +
        k[1041]*y[IDX_NHI]*y[IDX_NO2I] + k[1044]*y[IDX_NHI]*y[IDX_O2I] +
        k[1049]*y[IDX_NHI]*y[IDX_OHI] - k[1068]*y[IDX_OI]*y[IDX_HNOI] -
        k[1069]*y[IDX_OI]*y[IDX_HNOI] - k[1070]*y[IDX_OI]*y[IDX_HNOI] +
        k[1072]*y[IDX_OI]*y[IDX_NH2I] - k[1097]*y[IDX_OHI]*y[IDX_HNOI] -
        k[1153]*y[IDX_HNOI] - k[1294]*y[IDX_HNOI] + k[1351]*y[IDX_GHNOI] +
        k[1352]*y[IDX_GHNOI] + k[1353]*y[IDX_GHNOI] + k[1354]*y[IDX_GHNOI];
    ydot[IDX_HNOII] = 0.0 - k[204]*y[IDX_NOI]*y[IDX_HNOII] -
        k[334]*y[IDX_HNOII]*y[IDX_EM] - k[388]*y[IDX_CI]*y[IDX_HNOII] -
        k[430]*y[IDX_CH2I]*y[IDX_HNOII] - k[467]*y[IDX_CHI]*y[IDX_HNOII] -
        k[482]*y[IDX_CNI]*y[IDX_HNOII] - k[486]*y[IDX_COI]*y[IDX_HNOII] +
        k[524]*y[IDX_H2II]*y[IDX_NOI] - k[550]*y[IDX_H2COI]*y[IDX_HNOII] -
        k[568]*y[IDX_H2OI]*y[IDX_HNOII] + k[596]*y[IDX_H3II]*y[IDX_NOI] -
        k[630]*y[IDX_HCNI]*y[IDX_HNOII] - k[642]*y[IDX_HCOI]*y[IDX_HNOII] -
        k[649]*y[IDX_HNCI]*y[IDX_HNOII] - k[652]*y[IDX_HNOII]*y[IDX_CO2I] -
        k[732]*y[IDX_N2I]*y[IDX_HNOII] + k[738]*y[IDX_NI]*y[IDX_H2OII] +
        k[749]*y[IDX_NHII]*y[IDX_CO2I] + k[755]*y[IDX_NHII]*y[IDX_H2OI] +
        k[778]*y[IDX_NH2II]*y[IDX_O2I] - k[788]*y[IDX_NH2I]*y[IDX_HNOII] -
        k[800]*y[IDX_NHI]*y[IDX_HNOII] + k[804]*y[IDX_NHI]*y[IDX_O2II] +
        k[807]*y[IDX_NOI]*y[IDX_O2HII] + k[827]*y[IDX_OI]*y[IDX_NH2II] +
        k[828]*y[IDX_OI]*y[IDX_NH3II] + k[846]*y[IDX_OHII]*y[IDX_NOI] -
        k[856]*y[IDX_OHI]*y[IDX_HNOII] - k[1295]*y[IDX_HNOII];
    ydot[IDX_HOCII] = 0.0 - k[5]*y[IDX_H2I]*y[IDX_HOCII] -
        k[335]*y[IDX_HOCII]*y[IDX_EM] + k[371]*y[IDX_CII]*y[IDX_H2OI] +
        k[533]*y[IDX_H2I]*y[IDX_COII] + k[584]*y[IDX_H3II]*y[IDX_COI] -
        k[1226]*y[IDX_HOCII];
    ydot[IDX_MgI] = 0.0 - k[18]*y[IDX_CII]*y[IDX_MgI] -
        k[32]*y[IDX_CHII]*y[IDX_MgI] - k[47]*y[IDX_CH3II]*y[IDX_MgI] -
        k[83]*y[IDX_HII]*y[IDX_MgI] - k[119]*y[IDX_H2OII]*y[IDX_MgI] -
        k[148]*y[IDX_MgI]*y[IDX_H2COII] - k[149]*y[IDX_MgI]*y[IDX_HCOII] -
        k[150]*y[IDX_MgI]*y[IDX_N2II] - k[151]*y[IDX_MgI]*y[IDX_NOII] -
        k[152]*y[IDX_MgI]*y[IDX_O2II] - k[153]*y[IDX_MgI]*y[IDX_SiII] -
        k[154]*y[IDX_MgI]*y[IDX_SiOII] - k[163]*y[IDX_NII]*y[IDX_MgI] -
        k[190]*y[IDX_NH3II]*y[IDX_MgI] - k[266]*y[IDX_MgI] -
        k[591]*y[IDX_H3II]*y[IDX_MgI] - k[1154]*y[IDX_MgI] +
        k[1219]*y[IDX_MgII]*y[IDX_EM] - k[1300]*y[IDX_MgI] + k[1319]*y[IDX_GMgI]
        + k[1320]*y[IDX_GMgI] + k[1321]*y[IDX_GMgI] + k[1322]*y[IDX_GMgI];
    ydot[IDX_MgII] = 0.0 + k[18]*y[IDX_CII]*y[IDX_MgI] +
        k[32]*y[IDX_CHII]*y[IDX_MgI] + k[47]*y[IDX_CH3II]*y[IDX_MgI] +
        k[83]*y[IDX_HII]*y[IDX_MgI] + k[119]*y[IDX_H2OII]*y[IDX_MgI] +
        k[148]*y[IDX_MgI]*y[IDX_H2COII] + k[149]*y[IDX_MgI]*y[IDX_HCOII] +
        k[150]*y[IDX_MgI]*y[IDX_N2II] + k[151]*y[IDX_MgI]*y[IDX_NOII] +
        k[152]*y[IDX_MgI]*y[IDX_O2II] + k[153]*y[IDX_MgI]*y[IDX_SiII] +
        k[154]*y[IDX_MgI]*y[IDX_SiOII] + k[163]*y[IDX_NII]*y[IDX_MgI] +
        k[190]*y[IDX_NH3II]*y[IDX_MgI] + k[266]*y[IDX_MgI] +
        k[591]*y[IDX_H3II]*y[IDX_MgI] + k[1154]*y[IDX_MgI] -
        k[1219]*y[IDX_MgII]*y[IDX_EM] - k[1299]*y[IDX_MgII];
    ydot[IDX_NI] = 0.0 + k[57]*y[IDX_CHI]*y[IDX_NII] +
        k[155]*y[IDX_NII]*y[IDX_CH2I] + k[156]*y[IDX_NII]*y[IDX_CH4I] +
        k[157]*y[IDX_NII]*y[IDX_CNI] + k[158]*y[IDX_NII]*y[IDX_COI] +
        k[159]*y[IDX_NII]*y[IDX_H2COI] + k[160]*y[IDX_NII]*y[IDX_H2OI] +
        k[161]*y[IDX_NII]*y[IDX_HCNI] + k[162]*y[IDX_NII]*y[IDX_HCOI] +
        k[163]*y[IDX_NII]*y[IDX_MgI] + k[164]*y[IDX_NII]*y[IDX_NH2I] +
        k[165]*y[IDX_NII]*y[IDX_NH3I] + k[166]*y[IDX_NII]*y[IDX_NHI] +
        k[167]*y[IDX_NII]*y[IDX_NOI] + k[168]*y[IDX_NII]*y[IDX_O2I] +
        k[169]*y[IDX_NII]*y[IDX_OHI] - k[174]*y[IDX_NI]*y[IDX_N2II] -
        k[238]*y[IDX_NI] + k[251]*y[IDX_CNI] + k[267]*y[IDX_N2I] +
        k[267]*y[IDX_N2I] - k[268]*y[IDX_NI] + k[274]*y[IDX_NHI] +
        k[278]*y[IDX_NOI] + k[303]*y[IDX_CNII]*y[IDX_EM] +
        k[337]*y[IDX_N2II]*y[IDX_EM] + k[337]*y[IDX_N2II]*y[IDX_EM] +
        k[339]*y[IDX_N2HII]*y[IDX_EM] + k[340]*y[IDX_NHII]*y[IDX_EM] +
        k[341]*y[IDX_NH2II]*y[IDX_EM] + k[345]*y[IDX_NOII]*y[IDX_EM] +
        k[390]*y[IDX_CI]*y[IDX_NHII] - k[408]*y[IDX_CHII]*y[IDX_NI] +
        k[432]*y[IDX_CH2I]*y[IDX_NHII] + k[470]*y[IDX_CHI]*y[IDX_NHII] -
        k[522]*y[IDX_H2II]*y[IDX_NI] + k[540]*y[IDX_H2I]*y[IDX_NHII] +
        k[664]*y[IDX_HeII]*y[IDX_CNI] + k[678]*y[IDX_HeII]*y[IDX_HCNI] +
        k[679]*y[IDX_HeII]*y[IDX_HCNI] + k[684]*y[IDX_HeII]*y[IDX_HNCI] +
        k[688]*y[IDX_HeII]*y[IDX_N2I] + k[694]*y[IDX_HeII]*y[IDX_NOI] +
        k[716]*y[IDX_NII]*y[IDX_CH4I] - k[736]*y[IDX_NI]*y[IDX_CH2II] -
        k[737]*y[IDX_NI]*y[IDX_CNII] - k[738]*y[IDX_NI]*y[IDX_H2OII] -
        k[739]*y[IDX_NI]*y[IDX_H2OII] - k[740]*y[IDX_NI]*y[IDX_NHII] -
        k[741]*y[IDX_NI]*y[IDX_NH2II] - k[742]*y[IDX_NI]*y[IDX_O2II] -
        k[743]*y[IDX_NI]*y[IDX_OHII] - k[744]*y[IDX_NI]*y[IDX_SiCII] -
        k[745]*y[IDX_NI]*y[IDX_SiOII] - k[746]*y[IDX_NI]*y[IDX_SiOII] +
        k[747]*y[IDX_NHII]*y[IDX_CNI] + k[748]*y[IDX_NHII]*y[IDX_CO2I] +
        k[751]*y[IDX_NHII]*y[IDX_COI] + k[752]*y[IDX_NHII]*y[IDX_H2COI] +
        k[754]*y[IDX_NHII]*y[IDX_H2OI] + k[758]*y[IDX_NHII]*y[IDX_HCNI] +
        k[759]*y[IDX_NHII]*y[IDX_HCOI] + k[760]*y[IDX_NHII]*y[IDX_HNCI] +
        k[761]*y[IDX_NHII]*y[IDX_N2I] + k[762]*y[IDX_NHII]*y[IDX_NH2I] +
        k[763]*y[IDX_NHII]*y[IDX_NHI] + k[766]*y[IDX_NHII]*y[IDX_O2I] +
        k[767]*y[IDX_NHII]*y[IDX_OI] + k[768]*y[IDX_NHII]*y[IDX_OHI] +
        k[795]*y[IDX_NHI]*y[IDX_COII] + k[796]*y[IDX_NHI]*y[IDX_H2COII] +
        k[797]*y[IDX_NHI]*y[IDX_H2OII] + k[802]*y[IDX_NHI]*y[IDX_NH2II] +
        k[814]*y[IDX_OII]*y[IDX_HCNI] + k[817]*y[IDX_OII]*y[IDX_N2I] +
        k[825]*y[IDX_OI]*y[IDX_N2II] + k[865]*y[IDX_CI]*y[IDX_N2I] +
        k[870]*y[IDX_CI]*y[IDX_NHI] + k[872]*y[IDX_CI]*y[IDX_NOI] +
        k[886]*y[IDX_CH2I]*y[IDX_NOI] + k[927]*y[IDX_CHI]*y[IDX_N2I] -
        k[928]*y[IDX_CHI]*y[IDX_NI] - k[929]*y[IDX_CHI]*y[IDX_NI] +
        k[931]*y[IDX_CHI]*y[IDX_NOI] + k[947]*y[IDX_CNI]*y[IDX_NOI] -
        k[960]*y[IDX_H2I]*y[IDX_NI] + k[985]*y[IDX_HI]*y[IDX_NHI] +
        k[988]*y[IDX_HI]*y[IDX_NOI] - k[1005]*y[IDX_NI]*y[IDX_CH2I] -
        k[1006]*y[IDX_NI]*y[IDX_CH2I] - k[1007]*y[IDX_NI]*y[IDX_CH2I] -
        k[1008]*y[IDX_NI]*y[IDX_CH3I] - k[1009]*y[IDX_NI]*y[IDX_CH3I] -
        k[1010]*y[IDX_NI]*y[IDX_CH3I] - k[1011]*y[IDX_NI]*y[IDX_CNI] -
        k[1012]*y[IDX_NI]*y[IDX_CO2I] - k[1013]*y[IDX_NI]*y[IDX_H2CNI] -
        k[1014]*y[IDX_NI]*y[IDX_HCOI] - k[1015]*y[IDX_NI]*y[IDX_HCOI] -
        k[1016]*y[IDX_NI]*y[IDX_HCOI] - k[1017]*y[IDX_NI]*y[IDX_HNOI] -
        k[1018]*y[IDX_NI]*y[IDX_NHI] - k[1019]*y[IDX_NI]*y[IDX_NO2I] -
        k[1020]*y[IDX_NI]*y[IDX_NO2I] - k[1021]*y[IDX_NI]*y[IDX_NO2I] -
        k[1022]*y[IDX_NI]*y[IDX_NOI] - k[1023]*y[IDX_NI]*y[IDX_O2I] -
        k[1024]*y[IDX_NI]*y[IDX_O2HI] - k[1025]*y[IDX_NI]*y[IDX_OHI] -
        k[1026]*y[IDX_NI]*y[IDX_OHI] - k[1027]*y[IDX_NI]*y[IDX_SiCI] +
        k[1035]*y[IDX_NHI]*y[IDX_CNI] + k[1040]*y[IDX_NHI]*y[IDX_NHI] +
        k[1047]*y[IDX_NHI]*y[IDX_OI] + k[1048]*y[IDX_NHI]*y[IDX_OHI] +
        k[1057]*y[IDX_OI]*y[IDX_CNI] + k[1071]*y[IDX_OI]*y[IDX_N2I] +
        k[1076]*y[IDX_OI]*y[IDX_NOI] + k[1105]*y[IDX_SiI]*y[IDX_NOI] +
        k[1130]*y[IDX_CNI] + k[1155]*y[IDX_N2I] + k[1155]*y[IDX_N2I] +
        k[1156]*y[IDX_NHII] + k[1162]*y[IDX_NHI] + k[1166]*y[IDX_NOI] -
        k[1192]*y[IDX_CII]*y[IDX_NI] - k[1194]*y[IDX_CI]*y[IDX_NI] -
        k[1210]*y[IDX_NII]*y[IDX_NI] + k[1220]*y[IDX_NII]*y[IDX_EM] -
        k[1290]*y[IDX_NI];
    ydot[IDX_NII] = 0.0 - k[57]*y[IDX_CHI]*y[IDX_NII] -
        k[155]*y[IDX_NII]*y[IDX_CH2I] - k[156]*y[IDX_NII]*y[IDX_CH4I] -
        k[157]*y[IDX_NII]*y[IDX_CNI] - k[158]*y[IDX_NII]*y[IDX_COI] -
        k[159]*y[IDX_NII]*y[IDX_H2COI] - k[160]*y[IDX_NII]*y[IDX_H2OI] -
        k[161]*y[IDX_NII]*y[IDX_HCNI] - k[162]*y[IDX_NII]*y[IDX_HCOI] -
        k[163]*y[IDX_NII]*y[IDX_MgI] - k[164]*y[IDX_NII]*y[IDX_NH2I] -
        k[165]*y[IDX_NII]*y[IDX_NH3I] - k[166]*y[IDX_NII]*y[IDX_NHI] -
        k[167]*y[IDX_NII]*y[IDX_NOI] - k[168]*y[IDX_NII]*y[IDX_O2I] -
        k[169]*y[IDX_NII]*y[IDX_OHI] + k[174]*y[IDX_NI]*y[IDX_N2II] +
        k[238]*y[IDX_NI] + k[268]*y[IDX_NI] - k[468]*y[IDX_CHI]*y[IDX_NII] -
        k[538]*y[IDX_H2I]*y[IDX_NII] + k[663]*y[IDX_HeII]*y[IDX_CNI] +
        k[677]*y[IDX_HeII]*y[IDX_HCNI] + k[688]*y[IDX_HeII]*y[IDX_N2I] +
        k[689]*y[IDX_HeII]*y[IDX_NH2I] + k[693]*y[IDX_HeII]*y[IDX_NHI] +
        k[695]*y[IDX_HeII]*y[IDX_NOI] - k[712]*y[IDX_NII]*y[IDX_CH3OHI] -
        k[713]*y[IDX_NII]*y[IDX_CH3OHI] - k[714]*y[IDX_NII]*y[IDX_CH3OHI] -
        k[715]*y[IDX_NII]*y[IDX_CH3OHI] - k[716]*y[IDX_NII]*y[IDX_CH4I] -
        k[717]*y[IDX_NII]*y[IDX_CH4I] - k[718]*y[IDX_NII]*y[IDX_CH4I] -
        k[719]*y[IDX_NII]*y[IDX_CO2I] - k[720]*y[IDX_NII]*y[IDX_COI] -
        k[721]*y[IDX_NII]*y[IDX_H2COI] - k[722]*y[IDX_NII]*y[IDX_H2COI] -
        k[723]*y[IDX_NII]*y[IDX_HCOI] - k[724]*y[IDX_NII]*y[IDX_NH3I] -
        k[725]*y[IDX_NII]*y[IDX_NH3I] - k[726]*y[IDX_NII]*y[IDX_NHI] -
        k[727]*y[IDX_NII]*y[IDX_NOI] - k[728]*y[IDX_NII]*y[IDX_O2I] -
        k[729]*y[IDX_NII]*y[IDX_O2I] - k[1210]*y[IDX_NII]*y[IDX_NI] -
        k[1220]*y[IDX_NII]*y[IDX_EM] - k[1268]*y[IDX_NII];
    ydot[IDX_N2I] = 0.0 + k[29]*y[IDX_CI]*y[IDX_N2II] +
        k[41]*y[IDX_CH2I]*y[IDX_N2II] + k[58]*y[IDX_CHI]*y[IDX_N2II] +
        k[69]*y[IDX_CNI]*y[IDX_N2II] + k[74]*y[IDX_COI]*y[IDX_N2II] +
        k[125]*y[IDX_H2OI]*y[IDX_N2II] + k[135]*y[IDX_HCNI]*y[IDX_N2II] -
        k[144]*y[IDX_HeII]*y[IDX_N2I] + k[150]*y[IDX_MgI]*y[IDX_N2II] +
        k[170]*y[IDX_N2II]*y[IDX_H2COI] + k[171]*y[IDX_N2II]*y[IDX_HCOI] +
        k[172]*y[IDX_N2II]*y[IDX_NOI] + k[173]*y[IDX_N2II]*y[IDX_O2I] +
        k[174]*y[IDX_NI]*y[IDX_N2II] + k[186]*y[IDX_NH2I]*y[IDX_N2II] +
        k[197]*y[IDX_NH3I]*y[IDX_N2II] + k[201]*y[IDX_NHI]*y[IDX_N2II] +
        k[218]*y[IDX_OI]*y[IDX_N2II] + k[227]*y[IDX_OHI]*y[IDX_N2II] -
        k[267]*y[IDX_N2I] + k[338]*y[IDX_N2HII]*y[IDX_EM] +
        k[389]*y[IDX_CI]*y[IDX_N2HII] + k[431]*y[IDX_CH2I]*y[IDX_N2HII] +
        k[455]*y[IDX_CH4I]*y[IDX_N2II] + k[456]*y[IDX_CH4I]*y[IDX_N2II] +
        k[469]*y[IDX_CHI]*y[IDX_N2HII] + k[487]*y[IDX_COI]*y[IDX_N2HII] -
        k[521]*y[IDX_H2II]*y[IDX_N2I] + k[570]*y[IDX_H2OI]*y[IDX_N2HII] -
        k[592]*y[IDX_H3II]*y[IDX_N2I] + k[631]*y[IDX_HCNI]*y[IDX_N2HII] +
        k[643]*y[IDX_HCOI]*y[IDX_N2HII] + k[650]*y[IDX_HNCI]*y[IDX_N2HII] -
        k[688]*y[IDX_HeII]*y[IDX_N2I] + k[730]*y[IDX_N2II]*y[IDX_H2COI] -
        k[732]*y[IDX_N2I]*y[IDX_HNOII] - k[733]*y[IDX_N2I]*y[IDX_O2HII] +
        k[734]*y[IDX_N2HII]*y[IDX_CO2I] + k[735]*y[IDX_N2HII]*y[IDX_H2COI] -
        k[761]*y[IDX_NHII]*y[IDX_N2I] + k[789]*y[IDX_NH2I]*y[IDX_N2HII] +
        k[801]*y[IDX_NHI]*y[IDX_N2HII] - k[817]*y[IDX_OII]*y[IDX_N2I] +
        k[826]*y[IDX_OI]*y[IDX_N2HII] - k[845]*y[IDX_OHII]*y[IDX_N2I] +
        k[857]*y[IDX_OHI]*y[IDX_N2HII] - k[865]*y[IDX_CI]*y[IDX_N2I] -
        k[884]*y[IDX_CH2I]*y[IDX_N2I] - k[927]*y[IDX_CHI]*y[IDX_N2I] +
        k[946]*y[IDX_CNI]*y[IDX_NOI] + k[1011]*y[IDX_NI]*y[IDX_CNI] +
        k[1018]*y[IDX_NI]*y[IDX_NHI] + k[1019]*y[IDX_NI]*y[IDX_NO2I] +
        k[1021]*y[IDX_NI]*y[IDX_NO2I] + k[1022]*y[IDX_NI]*y[IDX_NOI] +
        k[1029]*y[IDX_NH2I]*y[IDX_NOI] + k[1030]*y[IDX_NH2I]*y[IDX_NOI] +
        k[1038]*y[IDX_NHI]*y[IDX_NHI] + k[1039]*y[IDX_NHI]*y[IDX_NHI] +
        k[1042]*y[IDX_NHI]*y[IDX_NOI] + k[1043]*y[IDX_NHI]*y[IDX_NOI] +
        k[1051]*y[IDX_NOI]*y[IDX_NOI] + k[1053]*y[IDX_NOI]*y[IDX_OCNI] -
        k[1071]*y[IDX_OI]*y[IDX_N2I] - k[1155]*y[IDX_N2I] - k[1262]*y[IDX_N2I] +
        k[1335]*y[IDX_GN2I] + k[1336]*y[IDX_GN2I] + k[1337]*y[IDX_GN2I] +
        k[1338]*y[IDX_GN2I];
    ydot[IDX_N2II] = 0.0 - k[29]*y[IDX_CI]*y[IDX_N2II] -
        k[41]*y[IDX_CH2I]*y[IDX_N2II] - k[58]*y[IDX_CHI]*y[IDX_N2II] -
        k[69]*y[IDX_CNI]*y[IDX_N2II] - k[74]*y[IDX_COI]*y[IDX_N2II] -
        k[125]*y[IDX_H2OI]*y[IDX_N2II] - k[135]*y[IDX_HCNI]*y[IDX_N2II] +
        k[144]*y[IDX_HeII]*y[IDX_N2I] - k[150]*y[IDX_MgI]*y[IDX_N2II] -
        k[170]*y[IDX_N2II]*y[IDX_H2COI] - k[171]*y[IDX_N2II]*y[IDX_HCOI] -
        k[172]*y[IDX_N2II]*y[IDX_NOI] - k[173]*y[IDX_N2II]*y[IDX_O2I] -
        k[174]*y[IDX_NI]*y[IDX_N2II] - k[186]*y[IDX_NH2I]*y[IDX_N2II] -
        k[197]*y[IDX_NH3I]*y[IDX_N2II] - k[201]*y[IDX_NHI]*y[IDX_N2II] -
        k[218]*y[IDX_OI]*y[IDX_N2II] - k[227]*y[IDX_OHI]*y[IDX_N2II] -
        k[337]*y[IDX_N2II]*y[IDX_EM] - k[455]*y[IDX_CH4I]*y[IDX_N2II] -
        k[456]*y[IDX_CH4I]*y[IDX_N2II] - k[539]*y[IDX_H2I]*y[IDX_N2II] -
        k[569]*y[IDX_H2OI]*y[IDX_N2II] + k[726]*y[IDX_NII]*y[IDX_NHI] +
        k[727]*y[IDX_NII]*y[IDX_NOI] - k[730]*y[IDX_N2II]*y[IDX_H2COI] -
        k[731]*y[IDX_N2II]*y[IDX_HCOI] + k[737]*y[IDX_NI]*y[IDX_CNII] +
        k[740]*y[IDX_NI]*y[IDX_NHII] - k[825]*y[IDX_OI]*y[IDX_N2II] +
        k[1210]*y[IDX_NII]*y[IDX_NI] - k[1271]*y[IDX_N2II];
    ydot[IDX_N2HII] = 0.0 - k[338]*y[IDX_N2HII]*y[IDX_EM] -
        k[339]*y[IDX_N2HII]*y[IDX_EM] - k[389]*y[IDX_CI]*y[IDX_N2HII] -
        k[431]*y[IDX_CH2I]*y[IDX_N2HII] - k[469]*y[IDX_CHI]*y[IDX_N2HII] -
        k[487]*y[IDX_COI]*y[IDX_N2HII] + k[521]*y[IDX_H2II]*y[IDX_N2I] +
        k[539]*y[IDX_H2I]*y[IDX_N2II] + k[569]*y[IDX_H2OI]*y[IDX_N2II] -
        k[570]*y[IDX_H2OI]*y[IDX_N2HII] + k[592]*y[IDX_H3II]*y[IDX_N2I] -
        k[631]*y[IDX_HCNI]*y[IDX_N2HII] - k[643]*y[IDX_HCOI]*y[IDX_N2HII] -
        k[650]*y[IDX_HNCI]*y[IDX_N2HII] + k[724]*y[IDX_NII]*y[IDX_NH3I] +
        k[731]*y[IDX_N2II]*y[IDX_HCOI] + k[732]*y[IDX_N2I]*y[IDX_HNOII] +
        k[733]*y[IDX_N2I]*y[IDX_O2HII] - k[734]*y[IDX_N2HII]*y[IDX_CO2I] -
        k[735]*y[IDX_N2HII]*y[IDX_H2COI] + k[741]*y[IDX_NI]*y[IDX_NH2II] +
        k[761]*y[IDX_NHII]*y[IDX_N2I] + k[764]*y[IDX_NHII]*y[IDX_NOI] -
        k[789]*y[IDX_NH2I]*y[IDX_N2HII] - k[801]*y[IDX_NHI]*y[IDX_N2HII] -
        k[826]*y[IDX_OI]*y[IDX_N2HII] + k[845]*y[IDX_OHII]*y[IDX_N2I] -
        k[857]*y[IDX_OHI]*y[IDX_N2HII] - k[1302]*y[IDX_N2HII];
    ydot[IDX_NHI] = 0.0 - k[86]*y[IDX_HII]*y[IDX_NHI] -
        k[111]*y[IDX_H2II]*y[IDX_NHI] - k[166]*y[IDX_NII]*y[IDX_NHI] +
        k[175]*y[IDX_NHII]*y[IDX_H2COI] + k[176]*y[IDX_NHII]*y[IDX_H2OI] +
        k[177]*y[IDX_NHII]*y[IDX_NH3I] + k[178]*y[IDX_NHII]*y[IDX_NOI] +
        k[179]*y[IDX_NHII]*y[IDX_O2I] - k[199]*y[IDX_NHI]*y[IDX_CNII] -
        k[200]*y[IDX_NHI]*y[IDX_COII] - k[201]*y[IDX_NHI]*y[IDX_N2II] -
        k[202]*y[IDX_NHI]*y[IDX_OII] + k[263]*y[IDX_HNCOI] + k[270]*y[IDX_NH2I]
        + k[273]*y[IDX_NH3I] - k[274]*y[IDX_NHI] - k[275]*y[IDX_NHI] +
        k[339]*y[IDX_N2HII]*y[IDX_EM] + k[342]*y[IDX_NH2II]*y[IDX_EM] +
        k[344]*y[IDX_NH3II]*y[IDX_EM] - k[375]*y[IDX_CII]*y[IDX_NHI] -
        k[410]*y[IDX_CHII]*y[IDX_NHI] + k[433]*y[IDX_CH2I]*y[IDX_NH2II] +
        k[471]*y[IDX_CHI]*y[IDX_NH2II] - k[523]*y[IDX_H2II]*y[IDX_NHI] +
        k[561]*y[IDX_H2OI]*y[IDX_CNII] - k[594]*y[IDX_H3II]*y[IDX_NHI] -
        k[693]*y[IDX_HeII]*y[IDX_NHI] + k[712]*y[IDX_NII]*y[IDX_CH3OHI] +
        k[713]*y[IDX_NII]*y[IDX_CH3OHI] + k[721]*y[IDX_NII]*y[IDX_H2COI] +
        k[725]*y[IDX_NII]*y[IDX_NH3I] - k[726]*y[IDX_NII]*y[IDX_NHI] -
        k[763]*y[IDX_NHII]*y[IDX_NHI] + k[769]*y[IDX_NH2II]*y[IDX_H2COI] +
        k[771]*y[IDX_NH2II]*y[IDX_H2OI] + k[773]*y[IDX_NH2II]*y[IDX_HCNI] +
        k[774]*y[IDX_NH2II]*y[IDX_HCOI] + k[775]*y[IDX_NH2II]*y[IDX_HNCI] +
        k[776]*y[IDX_NH2II]*y[IDX_NH2I] + k[779]*y[IDX_NH2I]*y[IDX_COII] -
        k[794]*y[IDX_NHI]*y[IDX_CH3II] - k[795]*y[IDX_NHI]*y[IDX_COII] -
        k[796]*y[IDX_NHI]*y[IDX_H2COII] - k[797]*y[IDX_NHI]*y[IDX_H2OII] -
        k[798]*y[IDX_NHI]*y[IDX_HCNII] - k[799]*y[IDX_NHI]*y[IDX_HCOII] -
        k[800]*y[IDX_NHI]*y[IDX_HNOII] - k[801]*y[IDX_NHI]*y[IDX_N2HII] -
        k[802]*y[IDX_NHI]*y[IDX_NH2II] - k[803]*y[IDX_NHI]*y[IDX_OII] -
        k[804]*y[IDX_NHI]*y[IDX_O2II] - k[805]*y[IDX_NHI]*y[IDX_O2HII] -
        k[806]*y[IDX_NHI]*y[IDX_OHII] + k[868]*y[IDX_CI]*y[IDX_NH2I] -
        k[869]*y[IDX_CI]*y[IDX_NHI] - k[870]*y[IDX_CI]*y[IDX_NHI] +
        k[884]*y[IDX_CH2I]*y[IDX_N2I] + k[907]*y[IDX_CH3I]*y[IDX_NH2I] +
        k[929]*y[IDX_CHI]*y[IDX_NI] + k[951]*y[IDX_COI]*y[IDX_HNOI] +
        k[960]*y[IDX_H2I]*y[IDX_NI] - k[962]*y[IDX_H2I]*y[IDX_NHI] +
        k[982]*y[IDX_HI]*y[IDX_HNOI] + k[983]*y[IDX_HI]*y[IDX_NH2I] -
        k[985]*y[IDX_HI]*y[IDX_NHI] + k[987]*y[IDX_HI]*y[IDX_NOI] +
        k[994]*y[IDX_HI]*y[IDX_OCNI] + k[1007]*y[IDX_NI]*y[IDX_CH2I] +
        k[1013]*y[IDX_NI]*y[IDX_H2CNI] + k[1014]*y[IDX_NI]*y[IDX_HCOI] +
        k[1017]*y[IDX_NI]*y[IDX_HNOI] - k[1018]*y[IDX_NI]*y[IDX_NHI] +
        k[1024]*y[IDX_NI]*y[IDX_O2HI] + k[1026]*y[IDX_NI]*y[IDX_OHI] +
        k[1031]*y[IDX_NH2I]*y[IDX_OHI] - k[1034]*y[IDX_NHI]*y[IDX_CH4I] -
        k[1035]*y[IDX_NHI]*y[IDX_CNI] - k[1036]*y[IDX_NHI]*y[IDX_H2OI] -
        k[1037]*y[IDX_NHI]*y[IDX_NH3I] - k[1038]*y[IDX_NHI]*y[IDX_NHI] -
        k[1038]*y[IDX_NHI]*y[IDX_NHI] - k[1039]*y[IDX_NHI]*y[IDX_NHI] -
        k[1039]*y[IDX_NHI]*y[IDX_NHI] - k[1040]*y[IDX_NHI]*y[IDX_NHI] -
        k[1040]*y[IDX_NHI]*y[IDX_NHI] - k[1041]*y[IDX_NHI]*y[IDX_NO2I] -
        k[1042]*y[IDX_NHI]*y[IDX_NOI] - k[1043]*y[IDX_NHI]*y[IDX_NOI] -
        k[1044]*y[IDX_NHI]*y[IDX_O2I] - k[1045]*y[IDX_NHI]*y[IDX_O2I] -
        k[1046]*y[IDX_NHI]*y[IDX_OI] - k[1047]*y[IDX_NHI]*y[IDX_OI] -
        k[1048]*y[IDX_NHI]*y[IDX_OHI] - k[1049]*y[IDX_NHI]*y[IDX_OHI] -
        k[1050]*y[IDX_NHI]*y[IDX_OHI] + k[1064]*y[IDX_OI]*y[IDX_HCNI] +
        k[1070]*y[IDX_OI]*y[IDX_HNOI] + k[1073]*y[IDX_OI]*y[IDX_NH2I] +
        k[1152]*y[IDX_HNCOI] + k[1158]*y[IDX_NH2I] + k[1161]*y[IDX_NH3I] -
        k[1162]*y[IDX_NHI] - k[1163]*y[IDX_NHI] - k[1265]*y[IDX_NHI];
    ydot[IDX_NHII] = 0.0 + k[86]*y[IDX_HII]*y[IDX_NHI] +
        k[111]*y[IDX_H2II]*y[IDX_NHI] + k[166]*y[IDX_NII]*y[IDX_NHI] -
        k[175]*y[IDX_NHII]*y[IDX_H2COI] - k[176]*y[IDX_NHII]*y[IDX_H2OI] -
        k[177]*y[IDX_NHII]*y[IDX_NH3I] - k[178]*y[IDX_NHII]*y[IDX_NOI] -
        k[179]*y[IDX_NHII]*y[IDX_O2I] + k[199]*y[IDX_NHI]*y[IDX_CNII] +
        k[200]*y[IDX_NHI]*y[IDX_COII] + k[201]*y[IDX_NHI]*y[IDX_N2II] +
        k[202]*y[IDX_NHI]*y[IDX_OII] + k[275]*y[IDX_NHI] -
        k[340]*y[IDX_NHII]*y[IDX_EM] - k[390]*y[IDX_CI]*y[IDX_NHII] -
        k[432]*y[IDX_CH2I]*y[IDX_NHII] - k[470]*y[IDX_CHI]*y[IDX_NHII] +
        k[522]*y[IDX_H2II]*y[IDX_NI] + k[538]*y[IDX_H2I]*y[IDX_NII] -
        k[540]*y[IDX_H2I]*y[IDX_NHII] - k[541]*y[IDX_H2I]*y[IDX_NHII] +
        k[685]*y[IDX_HeII]*y[IDX_HNCI] + k[690]*y[IDX_HeII]*y[IDX_NH2I] +
        k[691]*y[IDX_HeII]*y[IDX_NH3I] + k[723]*y[IDX_NII]*y[IDX_HCOI] -
        k[740]*y[IDX_NI]*y[IDX_NHII] - k[747]*y[IDX_NHII]*y[IDX_CNI] -
        k[748]*y[IDX_NHII]*y[IDX_CO2I] - k[749]*y[IDX_NHII]*y[IDX_CO2I] -
        k[750]*y[IDX_NHII]*y[IDX_CO2I] - k[751]*y[IDX_NHII]*y[IDX_COI] -
        k[752]*y[IDX_NHII]*y[IDX_H2COI] - k[753]*y[IDX_NHII]*y[IDX_H2COI] -
        k[754]*y[IDX_NHII]*y[IDX_H2OI] - k[755]*y[IDX_NHII]*y[IDX_H2OI] -
        k[756]*y[IDX_NHII]*y[IDX_H2OI] - k[757]*y[IDX_NHII]*y[IDX_H2OI] -
        k[758]*y[IDX_NHII]*y[IDX_HCNI] - k[759]*y[IDX_NHII]*y[IDX_HCOI] -
        k[760]*y[IDX_NHII]*y[IDX_HNCI] - k[761]*y[IDX_NHII]*y[IDX_N2I] -
        k[762]*y[IDX_NHII]*y[IDX_NH2I] - k[763]*y[IDX_NHII]*y[IDX_NHI] -
        k[764]*y[IDX_NHII]*y[IDX_NOI] - k[765]*y[IDX_NHII]*y[IDX_O2I] -
        k[766]*y[IDX_NHII]*y[IDX_O2I] - k[767]*y[IDX_NHII]*y[IDX_OI] -
        k[768]*y[IDX_NHII]*y[IDX_OHI] - k[1156]*y[IDX_NHII] + k[1163]*y[IDX_NHI]
        - k[1273]*y[IDX_NHII];
    ydot[IDX_NH2I] = 0.0 + k[42]*y[IDX_CH2I]*y[IDX_NH2II] +
        k[59]*y[IDX_CHI]*y[IDX_NH2II] - k[84]*y[IDX_HII]*y[IDX_NH2I] -
        k[109]*y[IDX_H2II]*y[IDX_NH2I] - k[164]*y[IDX_NII]*y[IDX_NH2I] +
        k[180]*y[IDX_NH2II]*y[IDX_HCOI] + k[181]*y[IDX_NH2II]*y[IDX_NH3I] +
        k[182]*y[IDX_NH2II]*y[IDX_NOI] - k[183]*y[IDX_NH2I]*y[IDX_CNII] -
        k[184]*y[IDX_NH2I]*y[IDX_COII] - k[185]*y[IDX_NH2I]*y[IDX_H2OII] -
        k[186]*y[IDX_NH2I]*y[IDX_N2II] - k[187]*y[IDX_NH2I]*y[IDX_O2II] -
        k[188]*y[IDX_NH2I]*y[IDX_OHII] - k[212]*y[IDX_OII]*y[IDX_NH2I] -
        k[269]*y[IDX_NH2I] - k[270]*y[IDX_NH2I] + k[271]*y[IDX_NH3I] +
        k[343]*y[IDX_NH3II]*y[IDX_EM] - k[373]*y[IDX_CII]*y[IDX_NH2I] -
        k[409]*y[IDX_CHII]*y[IDX_NH2I] + k[434]*y[IDX_CH2I]*y[IDX_NH3II] -
        k[593]*y[IDX_H3II]*y[IDX_NH2I] - k[689]*y[IDX_HeII]*y[IDX_NH2I] -
        k[690]*y[IDX_HeII]*y[IDX_NH2I] + k[753]*y[IDX_NHII]*y[IDX_H2COI] -
        k[762]*y[IDX_NHII]*y[IDX_NH2I] - k[776]*y[IDX_NH2II]*y[IDX_NH2I] -
        k[779]*y[IDX_NH2I]*y[IDX_COII] - k[780]*y[IDX_NH2I]*y[IDX_H2COII] -
        k[781]*y[IDX_NH2I]*y[IDX_H2OII] - k[782]*y[IDX_NH2I]*y[IDX_H3COII] -
        k[783]*y[IDX_NH2I]*y[IDX_H3OII] - k[784]*y[IDX_NH2I]*y[IDX_HCNII] -
        k[785]*y[IDX_NH2I]*y[IDX_HCNHII] - k[786]*y[IDX_NH2I]*y[IDX_HCNHII] -
        k[787]*y[IDX_NH2I]*y[IDX_HCOII] - k[788]*y[IDX_NH2I]*y[IDX_HNOII] -
        k[789]*y[IDX_NH2I]*y[IDX_N2HII] - k[790]*y[IDX_NH2I]*y[IDX_O2HII] -
        k[791]*y[IDX_NH2I]*y[IDX_OHII] + k[792]*y[IDX_NH3I]*y[IDX_COII] +
        k[793]*y[IDX_NH3I]*y[IDX_HCNII] - k[866]*y[IDX_CI]*y[IDX_NH2I] -
        k[867]*y[IDX_CI]*y[IDX_NH2I] - k[868]*y[IDX_CI]*y[IDX_NH2I] -
        k[907]*y[IDX_CH3I]*y[IDX_NH2I] + k[908]*y[IDX_CH3I]*y[IDX_NH3I] -
        k[961]*y[IDX_H2I]*y[IDX_NH2I] + k[962]*y[IDX_H2I]*y[IDX_NHI] +
        k[980]*y[IDX_HI]*y[IDX_HNOI] - k[983]*y[IDX_HI]*y[IDX_NH2I] +
        k[984]*y[IDX_HI]*y[IDX_NH3I] - k[1028]*y[IDX_NH2I]*y[IDX_CH4I] -
        k[1029]*y[IDX_NH2I]*y[IDX_NOI] - k[1030]*y[IDX_NH2I]*y[IDX_NOI] -
        k[1031]*y[IDX_NH2I]*y[IDX_OHI] - k[1032]*y[IDX_NH2I]*y[IDX_OHI] +
        k[1033]*y[IDX_NH3I]*y[IDX_CNI] + k[1034]*y[IDX_NHI]*y[IDX_CH4I] +
        k[1036]*y[IDX_NHI]*y[IDX_H2OI] + k[1037]*y[IDX_NHI]*y[IDX_NH3I] +
        k[1037]*y[IDX_NHI]*y[IDX_NH3I] + k[1040]*y[IDX_NHI]*y[IDX_NHI] +
        k[1050]*y[IDX_NHI]*y[IDX_OHI] - k[1072]*y[IDX_OI]*y[IDX_NH2I] -
        k[1073]*y[IDX_OI]*y[IDX_NH2I] + k[1074]*y[IDX_OI]*y[IDX_NH3I] +
        k[1095]*y[IDX_OHI]*y[IDX_HCNI] + k[1098]*y[IDX_OHI]*y[IDX_NH3I] -
        k[1157]*y[IDX_NH2I] - k[1158]*y[IDX_NH2I] + k[1159]*y[IDX_NH3I] -
        k[1289]*y[IDX_NH2I];
    ydot[IDX_NH2II] = 0.0 - k[42]*y[IDX_CH2I]*y[IDX_NH2II] -
        k[59]*y[IDX_CHI]*y[IDX_NH2II] + k[84]*y[IDX_HII]*y[IDX_NH2I] +
        k[109]*y[IDX_H2II]*y[IDX_NH2I] + k[164]*y[IDX_NII]*y[IDX_NH2I] -
        k[180]*y[IDX_NH2II]*y[IDX_HCOI] - k[181]*y[IDX_NH2II]*y[IDX_NH3I] -
        k[182]*y[IDX_NH2II]*y[IDX_NOI] + k[183]*y[IDX_NH2I]*y[IDX_CNII] +
        k[184]*y[IDX_NH2I]*y[IDX_COII] + k[185]*y[IDX_NH2I]*y[IDX_H2OII] +
        k[186]*y[IDX_NH2I]*y[IDX_N2II] + k[187]*y[IDX_NH2I]*y[IDX_O2II] +
        k[188]*y[IDX_NH2I]*y[IDX_OHII] + k[212]*y[IDX_OII]*y[IDX_NH2I] +
        k[269]*y[IDX_NH2I] - k[341]*y[IDX_NH2II]*y[IDX_EM] -
        k[342]*y[IDX_NH2II]*y[IDX_EM] - k[433]*y[IDX_CH2I]*y[IDX_NH2II] -
        k[471]*y[IDX_CHI]*y[IDX_NH2II] + k[502]*y[IDX_HII]*y[IDX_HNCOI] +
        k[523]*y[IDX_H2II]*y[IDX_NHI] + k[541]*y[IDX_H2I]*y[IDX_NHII] -
        k[542]*y[IDX_H2I]*y[IDX_NH2II] + k[594]*y[IDX_H3II]*y[IDX_NHI] +
        k[692]*y[IDX_HeII]*y[IDX_NH3I] + k[725]*y[IDX_NII]*y[IDX_NH3I] -
        k[741]*y[IDX_NI]*y[IDX_NH2II] + k[757]*y[IDX_NHII]*y[IDX_H2OI] +
        k[763]*y[IDX_NHII]*y[IDX_NHI] - k[769]*y[IDX_NH2II]*y[IDX_H2COI] -
        k[770]*y[IDX_NH2II]*y[IDX_H2COI] - k[771]*y[IDX_NH2II]*y[IDX_H2OI] -
        k[772]*y[IDX_NH2II]*y[IDX_H2OI] - k[773]*y[IDX_NH2II]*y[IDX_HCNI] -
        k[774]*y[IDX_NH2II]*y[IDX_HCOI] - k[775]*y[IDX_NH2II]*y[IDX_HNCI] -
        k[776]*y[IDX_NH2II]*y[IDX_NH2I] - k[777]*y[IDX_NH2II]*y[IDX_O2I] -
        k[778]*y[IDX_NH2II]*y[IDX_O2I] + k[798]*y[IDX_NHI]*y[IDX_HCNII] +
        k[799]*y[IDX_NHI]*y[IDX_HCOII] + k[800]*y[IDX_NHI]*y[IDX_HNOII] +
        k[801]*y[IDX_NHI]*y[IDX_N2HII] - k[802]*y[IDX_NHI]*y[IDX_NH2II] +
        k[805]*y[IDX_NHI]*y[IDX_O2HII] + k[806]*y[IDX_NHI]*y[IDX_OHII] -
        k[827]*y[IDX_OI]*y[IDX_NH2II] + k[1157]*y[IDX_NH2I] -
        k[1279]*y[IDX_NH2II];
    ydot[IDX_NH3I] = 0.0 - k[19]*y[IDX_CII]*y[IDX_NH3I] -
        k[33]*y[IDX_CHII]*y[IDX_NH3I] - k[50]*y[IDX_CH4II]*y[IDX_NH3I] -
        k[85]*y[IDX_HII]*y[IDX_NH3I] - k[110]*y[IDX_H2II]*y[IDX_NH3I] -
        k[145]*y[IDX_HeII]*y[IDX_NH3I] - k[165]*y[IDX_NII]*y[IDX_NH3I] -
        k[177]*y[IDX_NHII]*y[IDX_NH3I] - k[181]*y[IDX_NH2II]*y[IDX_NH3I] +
        k[189]*y[IDX_NH3II]*y[IDX_HCOI] + k[190]*y[IDX_NH3II]*y[IDX_MgI] +
        k[191]*y[IDX_NH3II]*y[IDX_NOI] + k[192]*y[IDX_NH3II]*y[IDX_SiI] -
        k[193]*y[IDX_NH3I]*y[IDX_COII] - k[194]*y[IDX_NH3I]*y[IDX_H2COII] -
        k[195]*y[IDX_NH3I]*y[IDX_H2OII] - k[196]*y[IDX_NH3I]*y[IDX_HCNII] -
        k[197]*y[IDX_NH3I]*y[IDX_N2II] - k[198]*y[IDX_NH3I]*y[IDX_O2II] -
        k[213]*y[IDX_OII]*y[IDX_NH3I] - k[222]*y[IDX_OHII]*y[IDX_NH3I] -
        k[271]*y[IDX_NH3I] - k[272]*y[IDX_NH3I] - k[273]*y[IDX_NH3I] -
        k[374]*y[IDX_CII]*y[IDX_NH3I] - k[691]*y[IDX_HeII]*y[IDX_NH3I] -
        k[692]*y[IDX_HeII]*y[IDX_NH3I] - k[724]*y[IDX_NII]*y[IDX_NH3I] -
        k[725]*y[IDX_NII]*y[IDX_NH3I] - k[792]*y[IDX_NH3I]*y[IDX_COII] -
        k[793]*y[IDX_NH3I]*y[IDX_HCNII] - k[908]*y[IDX_CH3I]*y[IDX_NH3I] +
        k[961]*y[IDX_H2I]*y[IDX_NH2I] - k[984]*y[IDX_HI]*y[IDX_NH3I] +
        k[1028]*y[IDX_NH2I]*y[IDX_CH4I] + k[1032]*y[IDX_NH2I]*y[IDX_OHI] -
        k[1033]*y[IDX_NH3I]*y[IDX_CNI] - k[1037]*y[IDX_NHI]*y[IDX_NH3I] -
        k[1074]*y[IDX_OI]*y[IDX_NH3I] - k[1098]*y[IDX_OHI]*y[IDX_NH3I] -
        k[1159]*y[IDX_NH3I] - k[1160]*y[IDX_NH3I] - k[1161]*y[IDX_NH3I] -
        k[1267]*y[IDX_NH3I] + k[1311]*y[IDX_GNH3I] + k[1312]*y[IDX_GNH3I] +
        k[1313]*y[IDX_GNH3I] + k[1314]*y[IDX_GNH3I];
    ydot[IDX_NH3II] = 0.0 + k[19]*y[IDX_CII]*y[IDX_NH3I] +
        k[33]*y[IDX_CHII]*y[IDX_NH3I] + k[50]*y[IDX_CH4II]*y[IDX_NH3I] +
        k[85]*y[IDX_HII]*y[IDX_NH3I] + k[110]*y[IDX_H2II]*y[IDX_NH3I] +
        k[145]*y[IDX_HeII]*y[IDX_NH3I] + k[165]*y[IDX_NII]*y[IDX_NH3I] +
        k[177]*y[IDX_NHII]*y[IDX_NH3I] + k[181]*y[IDX_NH2II]*y[IDX_NH3I] -
        k[189]*y[IDX_NH3II]*y[IDX_HCOI] - k[190]*y[IDX_NH3II]*y[IDX_MgI] -
        k[191]*y[IDX_NH3II]*y[IDX_NOI] - k[192]*y[IDX_NH3II]*y[IDX_SiI] +
        k[193]*y[IDX_NH3I]*y[IDX_COII] + k[194]*y[IDX_NH3I]*y[IDX_H2COII] +
        k[195]*y[IDX_NH3I]*y[IDX_H2OII] + k[196]*y[IDX_NH3I]*y[IDX_HCNII] +
        k[197]*y[IDX_NH3I]*y[IDX_N2II] + k[198]*y[IDX_NH3I]*y[IDX_O2II] +
        k[213]*y[IDX_OII]*y[IDX_NH3I] + k[222]*y[IDX_OHII]*y[IDX_NH3I] +
        k[272]*y[IDX_NH3I] - k[343]*y[IDX_NH3II]*y[IDX_EM] -
        k[344]*y[IDX_NH3II]*y[IDX_EM] - k[434]*y[IDX_CH2I]*y[IDX_NH3II] +
        k[542]*y[IDX_H2I]*y[IDX_NH2II] + k[593]*y[IDX_H3II]*y[IDX_NH2I] +
        k[756]*y[IDX_NHII]*y[IDX_H2OI] + k[762]*y[IDX_NHII]*y[IDX_NH2I] +
        k[770]*y[IDX_NH2II]*y[IDX_H2COI] + k[772]*y[IDX_NH2II]*y[IDX_H2OI] +
        k[776]*y[IDX_NH2II]*y[IDX_NH2I] + k[780]*y[IDX_NH2I]*y[IDX_H2COII] +
        k[781]*y[IDX_NH2I]*y[IDX_H2OII] + k[782]*y[IDX_NH2I]*y[IDX_H3COII] +
        k[783]*y[IDX_NH2I]*y[IDX_H3OII] + k[784]*y[IDX_NH2I]*y[IDX_HCNII] +
        k[785]*y[IDX_NH2I]*y[IDX_HCNHII] + k[786]*y[IDX_NH2I]*y[IDX_HCNHII] +
        k[787]*y[IDX_NH2I]*y[IDX_HCOII] + k[788]*y[IDX_NH2I]*y[IDX_HNOII] +
        k[789]*y[IDX_NH2I]*y[IDX_N2HII] + k[790]*y[IDX_NH2I]*y[IDX_O2HII] +
        k[791]*y[IDX_NH2I]*y[IDX_OHII] + k[802]*y[IDX_NHI]*y[IDX_NH2II] -
        k[828]*y[IDX_OI]*y[IDX_NH3II] + k[1160]*y[IDX_NH3I] -
        k[1283]*y[IDX_NH3II];
    ydot[IDX_NOI] = 0.0 - k[20]*y[IDX_CII]*y[IDX_NOI] -
        k[34]*y[IDX_CHII]*y[IDX_NOI] - k[36]*y[IDX_CH2II]*y[IDX_NOI] -
        k[48]*y[IDX_CH3II]*y[IDX_NOI] - k[67]*y[IDX_CNII]*y[IDX_NOI] -
        k[72]*y[IDX_COII]*y[IDX_NOI] - k[87]*y[IDX_HII]*y[IDX_NOI] -
        k[112]*y[IDX_H2II]*y[IDX_NOI] - k[120]*y[IDX_H2OII]*y[IDX_NOI] -
        k[132]*y[IDX_HCNII]*y[IDX_NOI] + k[151]*y[IDX_MgI]*y[IDX_NOII] -
        k[167]*y[IDX_NII]*y[IDX_NOI] - k[172]*y[IDX_N2II]*y[IDX_NOI] -
        k[178]*y[IDX_NHII]*y[IDX_NOI] - k[182]*y[IDX_NH2II]*y[IDX_NOI] -
        k[191]*y[IDX_NH3II]*y[IDX_NOI] - k[203]*y[IDX_NOI]*y[IDX_H2COII] -
        k[204]*y[IDX_NOI]*y[IDX_HNOII] - k[205]*y[IDX_NOI]*y[IDX_O2II] -
        k[206]*y[IDX_NOI]*y[IDX_SiOII] - k[223]*y[IDX_OHII]*y[IDX_NOI] +
        k[229]*y[IDX_SiI]*y[IDX_NOII] + k[264]*y[IDX_HNOI] + k[276]*y[IDX_NO2I]
        - k[277]*y[IDX_NOI] - k[278]*y[IDX_NOI] + k[311]*y[IDX_H2NOII]*y[IDX_EM]
        + k[334]*y[IDX_HNOII]*y[IDX_EM] + k[388]*y[IDX_CI]*y[IDX_HNOII] +
        k[430]*y[IDX_CH2I]*y[IDX_HNOII] + k[467]*y[IDX_CHI]*y[IDX_HNOII] +
        k[482]*y[IDX_CNI]*y[IDX_HNOII] + k[486]*y[IDX_COI]*y[IDX_HNOII] -
        k[524]*y[IDX_H2II]*y[IDX_NOI] + k[550]*y[IDX_H2COI]*y[IDX_HNOII] +
        k[568]*y[IDX_H2OI]*y[IDX_HNOII] - k[596]*y[IDX_H3II]*y[IDX_NOI] +
        k[630]*y[IDX_HCNI]*y[IDX_HNOII] + k[642]*y[IDX_HCOI]*y[IDX_HNOII] +
        k[649]*y[IDX_HNCI]*y[IDX_HNOII] + k[652]*y[IDX_HNOII]*y[IDX_CO2I] +
        k[687]*y[IDX_HeII]*y[IDX_HNOI] - k[694]*y[IDX_HeII]*y[IDX_NOI] -
        k[695]*y[IDX_HeII]*y[IDX_NOI] + k[715]*y[IDX_NII]*y[IDX_CH3OHI] +
        k[719]*y[IDX_NII]*y[IDX_CO2I] - k[727]*y[IDX_NII]*y[IDX_NOI] +
        k[729]*y[IDX_NII]*y[IDX_O2I] + k[732]*y[IDX_N2I]*y[IDX_HNOII] +
        k[746]*y[IDX_NI]*y[IDX_SiOII] - k[764]*y[IDX_NHII]*y[IDX_NOI] +
        k[788]*y[IDX_NH2I]*y[IDX_HNOII] + k[800]*y[IDX_NHI]*y[IDX_HNOII] -
        k[807]*y[IDX_NOI]*y[IDX_O2HII] - k[846]*y[IDX_OHII]*y[IDX_NOI] +
        k[856]*y[IDX_OHI]*y[IDX_HNOII] - k[871]*y[IDX_CI]*y[IDX_NOI] -
        k[872]*y[IDX_CI]*y[IDX_NOI] + k[883]*y[IDX_CH2I]*y[IDX_HNOI] +
        k[885]*y[IDX_CH2I]*y[IDX_NO2I] - k[886]*y[IDX_CH2I]*y[IDX_NOI] -
        k[887]*y[IDX_CH2I]*y[IDX_NOI] - k[888]*y[IDX_CH2I]*y[IDX_NOI] +
        k[906]*y[IDX_CH3I]*y[IDX_HNOI] - k[910]*y[IDX_CH3I]*y[IDX_NOI] +
        k[926]*y[IDX_CHI]*y[IDX_HNOI] - k[930]*y[IDX_CHI]*y[IDX_NOI] -
        k[931]*y[IDX_CHI]*y[IDX_NOI] - k[932]*y[IDX_CHI]*y[IDX_NOI] +
        k[944]*y[IDX_CNI]*y[IDX_HNOI] + k[945]*y[IDX_CNI]*y[IDX_NO2I] -
        k[946]*y[IDX_CNI]*y[IDX_NOI] - k[947]*y[IDX_CNI]*y[IDX_NOI] +
        k[948]*y[IDX_CNI]*y[IDX_O2I] + k[952]*y[IDX_COI]*y[IDX_NO2I] +
        k[981]*y[IDX_HI]*y[IDX_HNOI] + k[986]*y[IDX_HI]*y[IDX_NO2I] -
        k[987]*y[IDX_HI]*y[IDX_NOI] - k[988]*y[IDX_HI]*y[IDX_NOI] +
        k[999]*y[IDX_HCOI]*y[IDX_HNOI] - k[1000]*y[IDX_HCOI]*y[IDX_NOI] +
        k[1012]*y[IDX_NI]*y[IDX_CO2I] + k[1017]*y[IDX_NI]*y[IDX_HNOI] +
        k[1020]*y[IDX_NI]*y[IDX_NO2I] + k[1020]*y[IDX_NI]*y[IDX_NO2I] -
        k[1022]*y[IDX_NI]*y[IDX_NOI] + k[1023]*y[IDX_NI]*y[IDX_O2I] +
        k[1025]*y[IDX_NI]*y[IDX_OHI] - k[1029]*y[IDX_NH2I]*y[IDX_NOI] -
        k[1030]*y[IDX_NH2I]*y[IDX_NOI] + k[1041]*y[IDX_NHI]*y[IDX_NO2I] -
        k[1042]*y[IDX_NHI]*y[IDX_NOI] - k[1043]*y[IDX_NHI]*y[IDX_NOI] +
        k[1045]*y[IDX_NHI]*y[IDX_O2I] + k[1046]*y[IDX_NHI]*y[IDX_OI] -
        k[1051]*y[IDX_NOI]*y[IDX_NOI] - k[1051]*y[IDX_NOI]*y[IDX_NOI] -
        k[1052]*y[IDX_NOI]*y[IDX_O2I] - k[1053]*y[IDX_NOI]*y[IDX_OCNI] +
        k[1054]*y[IDX_O2I]*y[IDX_OCNI] + k[1058]*y[IDX_OI]*y[IDX_CNI] +
        k[1069]*y[IDX_OI]*y[IDX_HNOI] + k[1071]*y[IDX_OI]*y[IDX_N2I] +
        k[1075]*y[IDX_OI]*y[IDX_NO2I] - k[1076]*y[IDX_OI]*y[IDX_NOI] +
        k[1078]*y[IDX_OI]*y[IDX_OCNI] + k[1097]*y[IDX_OHI]*y[IDX_HNOI] -
        k[1099]*y[IDX_OHI]*y[IDX_NOI] - k[1105]*y[IDX_SiI]*y[IDX_NOI] +
        k[1153]*y[IDX_HNOI] + k[1164]*y[IDX_NO2I] - k[1165]*y[IDX_NOI] -
        k[1166]*y[IDX_NOI] - k[1255]*y[IDX_NOI] + k[1343]*y[IDX_GNOI] +
        k[1344]*y[IDX_GNOI] + k[1345]*y[IDX_GNOI] + k[1346]*y[IDX_GNOI];
    ydot[IDX_NOII] = 0.0 + k[20]*y[IDX_CII]*y[IDX_NOI] +
        k[34]*y[IDX_CHII]*y[IDX_NOI] + k[36]*y[IDX_CH2II]*y[IDX_NOI] +
        k[48]*y[IDX_CH3II]*y[IDX_NOI] + k[67]*y[IDX_CNII]*y[IDX_NOI] +
        k[72]*y[IDX_COII]*y[IDX_NOI] + k[87]*y[IDX_HII]*y[IDX_NOI] +
        k[112]*y[IDX_H2II]*y[IDX_NOI] + k[120]*y[IDX_H2OII]*y[IDX_NOI] +
        k[132]*y[IDX_HCNII]*y[IDX_NOI] - k[151]*y[IDX_MgI]*y[IDX_NOII] +
        k[167]*y[IDX_NII]*y[IDX_NOI] + k[172]*y[IDX_N2II]*y[IDX_NOI] +
        k[178]*y[IDX_NHII]*y[IDX_NOI] + k[182]*y[IDX_NH2II]*y[IDX_NOI] +
        k[191]*y[IDX_NH3II]*y[IDX_NOI] + k[203]*y[IDX_NOI]*y[IDX_H2COII] +
        k[204]*y[IDX_NOI]*y[IDX_HNOII] + k[205]*y[IDX_NOI]*y[IDX_O2II] +
        k[206]*y[IDX_NOI]*y[IDX_SiOII] + k[223]*y[IDX_OHII]*y[IDX_NOI] -
        k[229]*y[IDX_SiI]*y[IDX_NOII] + k[277]*y[IDX_NOI] -
        k[345]*y[IDX_NOII]*y[IDX_EM] + k[481]*y[IDX_CNII]*y[IDX_O2I] +
        k[503]*y[IDX_HII]*y[IDX_HNOI] + k[504]*y[IDX_HII]*y[IDX_NO2I] +
        k[595]*y[IDX_H3II]*y[IDX_NO2I] + k[686]*y[IDX_HeII]*y[IDX_HNOI] +
        k[714]*y[IDX_NII]*y[IDX_CH3OHI] + k[720]*y[IDX_NII]*y[IDX_COI] +
        k[722]*y[IDX_NII]*y[IDX_H2COI] + k[728]*y[IDX_NII]*y[IDX_O2I] +
        k[739]*y[IDX_NI]*y[IDX_H2OII] + k[742]*y[IDX_NI]*y[IDX_O2II] +
        k[743]*y[IDX_NI]*y[IDX_OHII] + k[745]*y[IDX_NI]*y[IDX_SiOII] +
        k[750]*y[IDX_NHII]*y[IDX_CO2I] + k[765]*y[IDX_NHII]*y[IDX_O2I] +
        k[803]*y[IDX_NHI]*y[IDX_OII] + k[811]*y[IDX_OII]*y[IDX_CNI] +
        k[815]*y[IDX_OII]*y[IDX_HCNI] + k[817]*y[IDX_OII]*y[IDX_N2I] +
        k[818]*y[IDX_OII]*y[IDX_NO2I] + k[825]*y[IDX_OI]*y[IDX_N2II] +
        k[1165]*y[IDX_NOI] - k[1277]*y[IDX_NOII];
    ydot[IDX_NO2I] = 0.0 - k[276]*y[IDX_NO2I] -
        k[504]*y[IDX_HII]*y[IDX_NO2I] - k[595]*y[IDX_H3II]*y[IDX_NO2I] -
        k[818]*y[IDX_OII]*y[IDX_NO2I] - k[885]*y[IDX_CH2I]*y[IDX_NO2I] -
        k[909]*y[IDX_CH3I]*y[IDX_NO2I] - k[945]*y[IDX_CNI]*y[IDX_NO2I] -
        k[952]*y[IDX_COI]*y[IDX_NO2I] - k[986]*y[IDX_HI]*y[IDX_NO2I] -
        k[1019]*y[IDX_NI]*y[IDX_NO2I] - k[1020]*y[IDX_NI]*y[IDX_NO2I] -
        k[1021]*y[IDX_NI]*y[IDX_NO2I] - k[1041]*y[IDX_NHI]*y[IDX_NO2I] +
        k[1052]*y[IDX_NOI]*y[IDX_O2I] + k[1055]*y[IDX_O2I]*y[IDX_OCNI] +
        k[1068]*y[IDX_OI]*y[IDX_HNOI] - k[1075]*y[IDX_OI]*y[IDX_NO2I] +
        k[1099]*y[IDX_OHI]*y[IDX_NOI] - k[1164]*y[IDX_NO2I] -
        k[1293]*y[IDX_NO2I] + k[1387]*y[IDX_GNO2I] + k[1388]*y[IDX_GNO2I] +
        k[1389]*y[IDX_GNO2I] + k[1390]*y[IDX_GNO2I];
    ydot[IDX_OI] = 0.0 - k[0]*y[IDX_CHI]*y[IDX_OI] +
        k[6]*y[IDX_H2I]*y[IDX_O2I] + k[6]*y[IDX_H2I]*y[IDX_O2I] +
        k[7]*y[IDX_H2I]*y[IDX_OHI] + k[12]*y[IDX_HI]*y[IDX_O2I] +
        k[12]*y[IDX_HI]*y[IDX_O2I] + k[13]*y[IDX_HI]*y[IDX_OHI] +
        k[43]*y[IDX_CH2I]*y[IDX_OII] + k[60]*y[IDX_CHI]*y[IDX_OII] -
        k[89]*y[IDX_HII]*y[IDX_OI] + k[131]*y[IDX_HI]*y[IDX_OII] +
        k[202]*y[IDX_NHI]*y[IDX_OII] + k[207]*y[IDX_OII]*y[IDX_CH4I] +
        k[208]*y[IDX_OII]*y[IDX_COI] + k[209]*y[IDX_OII]*y[IDX_H2COI] +
        k[210]*y[IDX_OII]*y[IDX_H2OI] + k[211]*y[IDX_OII]*y[IDX_HCOI] +
        k[212]*y[IDX_OII]*y[IDX_NH2I] + k[213]*y[IDX_OII]*y[IDX_NH3I] +
        k[214]*y[IDX_OII]*y[IDX_O2I] + k[215]*y[IDX_OII]*y[IDX_OHI] -
        k[216]*y[IDX_OI]*y[IDX_CNII] - k[217]*y[IDX_OI]*y[IDX_COII] -
        k[218]*y[IDX_OI]*y[IDX_N2II] - k[239]*y[IDX_OI] + k[252]*y[IDX_CO2I] +
        k[253]*y[IDX_COI] + k[276]*y[IDX_NO2I] + k[278]*y[IDX_NOI] +
        k[280]*y[IDX_O2I] + k[280]*y[IDX_O2I] - k[282]*y[IDX_OI] +
        k[283]*y[IDX_OCNI] + k[284]*y[IDX_OHI] + k[293]*y[IDX_SiOI] +
        k[304]*y[IDX_COII]*y[IDX_EM] + k[306]*y[IDX_H2COII]*y[IDX_EM] +
        k[312]*y[IDX_H2OII]*y[IDX_EM] + k[313]*y[IDX_H2OII]*y[IDX_EM] +
        k[323]*y[IDX_H3OII]*y[IDX_EM] + k[332]*y[IDX_HCO2II]*y[IDX_EM] +
        k[345]*y[IDX_NOII]*y[IDX_EM] + k[346]*y[IDX_O2II]*y[IDX_EM] +
        k[346]*y[IDX_O2II]*y[IDX_EM] + k[348]*y[IDX_OHII]*y[IDX_EM] +
        k[362]*y[IDX_SiOII]*y[IDX_EM] + k[376]*y[IDX_CII]*y[IDX_O2I] +
        k[391]*y[IDX_CI]*y[IDX_O2II] + k[393]*y[IDX_CI]*y[IDX_OHII] +
        k[412]*y[IDX_CHII]*y[IDX_O2I] - k[414]*y[IDX_CHII]*y[IDX_OI] -
        k[421]*y[IDX_CH2II]*y[IDX_OI] + k[435]*y[IDX_CH2I]*y[IDX_O2II] +
        k[437]*y[IDX_CH2I]*y[IDX_OHII] + k[442]*y[IDX_CH3II]*y[IDX_O2I] -
        k[443]*y[IDX_CH3II]*y[IDX_OI] - k[444]*y[IDX_CH3II]*y[IDX_OI] +
        k[473]*y[IDX_CHI]*y[IDX_O2II] + k[475]*y[IDX_CHI]*y[IDX_OHII] +
        k[496]*y[IDX_HII]*y[IDX_CO2I] - k[526]*y[IDX_H2II]*y[IDX_OI] -
        k[598]*y[IDX_H3II]*y[IDX_OI] - k[599]*y[IDX_H3II]*y[IDX_OI] +
        k[665]*y[IDX_HeII]*y[IDX_CO2I] + k[669]*y[IDX_HeII]*y[IDX_COI] +
        k[672]*y[IDX_HeII]*y[IDX_H2COI] + k[682]*y[IDX_HeII]*y[IDX_HCOI] +
        k[695]*y[IDX_HeII]*y[IDX_NOI] + k[696]*y[IDX_HeII]*y[IDX_O2I] +
        k[697]*y[IDX_HeII]*y[IDX_OCNI] + k[710]*y[IDX_HeII]*y[IDX_SiOI] +
        k[727]*y[IDX_NII]*y[IDX_NOI] + k[728]*y[IDX_NII]*y[IDX_O2I] +
        k[742]*y[IDX_NI]*y[IDX_O2II] + k[756]*y[IDX_NHII]*y[IDX_H2OI] +
        k[764]*y[IDX_NHII]*y[IDX_NOI] - k[767]*y[IDX_NHII]*y[IDX_OI] +
        k[777]*y[IDX_NH2II]*y[IDX_O2I] + k[791]*y[IDX_NH2I]*y[IDX_OHII] +
        k[804]*y[IDX_NHI]*y[IDX_O2II] + k[806]*y[IDX_NHI]*y[IDX_OHII] -
        k[822]*y[IDX_OI]*y[IDX_CH4II] - k[823]*y[IDX_OI]*y[IDX_H2OII] -
        k[824]*y[IDX_OI]*y[IDX_HCO2II] - k[825]*y[IDX_OI]*y[IDX_N2II] -
        k[826]*y[IDX_OI]*y[IDX_N2HII] - k[827]*y[IDX_OI]*y[IDX_NH2II] -
        k[828]*y[IDX_OI]*y[IDX_NH3II] - k[829]*y[IDX_OI]*y[IDX_O2HII] -
        k[830]*y[IDX_OI]*y[IDX_OHII] - k[831]*y[IDX_OI]*y[IDX_SiCII] -
        k[832]*y[IDX_OI]*y[IDX_SiHII] - k[833]*y[IDX_OI]*y[IDX_SiH2II] -
        k[834]*y[IDX_OI]*y[IDX_SiH3II] - k[835]*y[IDX_OI]*y[IDX_SiOII] +
        k[836]*y[IDX_OHII]*y[IDX_CNI] + k[837]*y[IDX_OHII]*y[IDX_CO2I] +
        k[838]*y[IDX_OHII]*y[IDX_COI] + k[839]*y[IDX_OHII]*y[IDX_H2COI] +
        k[840]*y[IDX_OHII]*y[IDX_H2OI] + k[841]*y[IDX_OHII]*y[IDX_HCNI] +
        k[843]*y[IDX_OHII]*y[IDX_HCOI] + k[844]*y[IDX_OHII]*y[IDX_HNCI] +
        k[845]*y[IDX_OHII]*y[IDX_N2I] + k[846]*y[IDX_OHII]*y[IDX_NOI] +
        k[847]*y[IDX_OHII]*y[IDX_OHI] + k[848]*y[IDX_OHII]*y[IDX_SiI] +
        k[849]*y[IDX_OHII]*y[IDX_SiHI] + k[850]*y[IDX_OHII]*y[IDX_SiOI] +
        k[851]*y[IDX_OHI]*y[IDX_COII] + k[852]*y[IDX_OHI]*y[IDX_H2OII] +
        k[871]*y[IDX_CI]*y[IDX_NOI] + k[873]*y[IDX_CI]*y[IDX_O2I] +
        k[876]*y[IDX_CI]*y[IDX_OHI] + k[892]*y[IDX_CH2I]*y[IDX_O2I] -
        k[894]*y[IDX_CH2I]*y[IDX_OI] - k[895]*y[IDX_CH2I]*y[IDX_OI] -
        k[896]*y[IDX_CH2I]*y[IDX_OI] - k[897]*y[IDX_CH2I]*y[IDX_OI] +
        k[900]*y[IDX_CH2I]*y[IDX_OHI] - k[915]*y[IDX_CH3I]*y[IDX_OI] -
        k[916]*y[IDX_CH3I]*y[IDX_OI] + k[917]*y[IDX_CH3I]*y[IDX_OHI] +
        k[930]*y[IDX_CHI]*y[IDX_NOI] + k[934]*y[IDX_CHI]*y[IDX_O2I] +
        k[936]*y[IDX_CHI]*y[IDX_O2I] - k[939]*y[IDX_CHI]*y[IDX_OI] -
        k[940]*y[IDX_CHI]*y[IDX_OI] + k[949]*y[IDX_CNI]*y[IDX_O2I] +
        k[953]*y[IDX_COI]*y[IDX_O2I] - k[965]*y[IDX_H2I]*y[IDX_OI] +
        k[978]*y[IDX_HI]*y[IDX_HCOI] + k[980]*y[IDX_HI]*y[IDX_HNOI] +
        k[987]*y[IDX_HI]*y[IDX_NOI] + k[989]*y[IDX_HI]*y[IDX_O2I] +
        k[990]*y[IDX_HI]*y[IDX_O2HI] + k[993]*y[IDX_HI]*y[IDX_OCNI] +
        k[996]*y[IDX_HI]*y[IDX_OHI] + k[1015]*y[IDX_NI]*y[IDX_HCOI] +
        k[1019]*y[IDX_NI]*y[IDX_NO2I] + k[1019]*y[IDX_NI]*y[IDX_NO2I] +
        k[1022]*y[IDX_NI]*y[IDX_NOI] + k[1023]*y[IDX_NI]*y[IDX_O2I] +
        k[1026]*y[IDX_NI]*y[IDX_OHI] + k[1032]*y[IDX_NH2I]*y[IDX_OHI] +
        k[1042]*y[IDX_NHI]*y[IDX_NOI] + k[1044]*y[IDX_NHI]*y[IDX_O2I] -
        k[1046]*y[IDX_NHI]*y[IDX_OI] - k[1047]*y[IDX_NHI]*y[IDX_OI] +
        k[1050]*y[IDX_NHI]*y[IDX_OHI] + k[1052]*y[IDX_NOI]*y[IDX_O2I] -
        k[1056]*y[IDX_OI]*y[IDX_CH4I] - k[1057]*y[IDX_OI]*y[IDX_CNI] -
        k[1058]*y[IDX_OI]*y[IDX_CNI] - k[1059]*y[IDX_OI]*y[IDX_CO2I] -
        k[1060]*y[IDX_OI]*y[IDX_H2CNI] - k[1061]*y[IDX_OI]*y[IDX_H2COI] -
        k[1062]*y[IDX_OI]*y[IDX_H2OI] - k[1063]*y[IDX_OI]*y[IDX_HCNI] -
        k[1064]*y[IDX_OI]*y[IDX_HCNI] - k[1065]*y[IDX_OI]*y[IDX_HCNI] -
        k[1066]*y[IDX_OI]*y[IDX_HCOI] - k[1067]*y[IDX_OI]*y[IDX_HCOI] -
        k[1068]*y[IDX_OI]*y[IDX_HNOI] - k[1069]*y[IDX_OI]*y[IDX_HNOI] -
        k[1070]*y[IDX_OI]*y[IDX_HNOI] - k[1071]*y[IDX_OI]*y[IDX_N2I] -
        k[1072]*y[IDX_OI]*y[IDX_NH2I] - k[1073]*y[IDX_OI]*y[IDX_NH2I] -
        k[1074]*y[IDX_OI]*y[IDX_NH3I] - k[1075]*y[IDX_OI]*y[IDX_NO2I] -
        k[1076]*y[IDX_OI]*y[IDX_NOI] - k[1077]*y[IDX_OI]*y[IDX_O2HI] -
        k[1078]*y[IDX_OI]*y[IDX_OCNI] - k[1079]*y[IDX_OI]*y[IDX_OCNI] -
        k[1080]*y[IDX_OI]*y[IDX_OHI] - k[1081]*y[IDX_OI]*y[IDX_SiC2I] -
        k[1082]*y[IDX_OI]*y[IDX_SiC3I] - k[1083]*y[IDX_OI]*y[IDX_SiCI] -
        k[1084]*y[IDX_OI]*y[IDX_SiCI] - k[1085]*y[IDX_OI]*y[IDX_SiH2I] -
        k[1086]*y[IDX_OI]*y[IDX_SiH2I] - k[1087]*y[IDX_OI]*y[IDX_SiH3I] -
        k[1088]*y[IDX_OI]*y[IDX_SiH4I] - k[1089]*y[IDX_OI]*y[IDX_SiHI] +
        k[1090]*y[IDX_OHI]*y[IDX_CNI] + k[1101]*y[IDX_OHI]*y[IDX_OHI] +
        k[1106]*y[IDX_SiI]*y[IDX_O2I] + k[1131]*y[IDX_COII] +
        k[1132]*y[IDX_CO2I] + k[1133]*y[IDX_COI] + k[1164]*y[IDX_NO2I] +
        k[1166]*y[IDX_NOI] + k[1167]*y[IDX_O2II] + k[1169]*y[IDX_O2I] +
        k[1169]*y[IDX_O2I] + k[1171]*y[IDX_O2HI] + k[1172]*y[IDX_OCNI] +
        k[1174]*y[IDX_OHI] + k[1189]*y[IDX_SiOII] + k[1190]*y[IDX_SiOI] -
        k[1193]*y[IDX_CII]*y[IDX_OI] - k[1196]*y[IDX_CI]*y[IDX_OI] -
        k[1207]*y[IDX_HI]*y[IDX_OI] - k[1211]*y[IDX_OI]*y[IDX_OI] -
        k[1211]*y[IDX_OI]*y[IDX_OI] - k[1212]*y[IDX_OI]*y[IDX_SiII] -
        k[1213]*y[IDX_OI]*y[IDX_SiI] + k[1221]*y[IDX_OII]*y[IDX_EM] -
        k[1291]*y[IDX_OI];
    ydot[IDX_OII] = 0.0 - k[43]*y[IDX_CH2I]*y[IDX_OII] -
        k[60]*y[IDX_CHI]*y[IDX_OII] + k[89]*y[IDX_HII]*y[IDX_OI] -
        k[131]*y[IDX_HI]*y[IDX_OII] - k[202]*y[IDX_NHI]*y[IDX_OII] -
        k[207]*y[IDX_OII]*y[IDX_CH4I] - k[208]*y[IDX_OII]*y[IDX_COI] -
        k[209]*y[IDX_OII]*y[IDX_H2COI] - k[210]*y[IDX_OII]*y[IDX_H2OI] -
        k[211]*y[IDX_OII]*y[IDX_HCOI] - k[212]*y[IDX_OII]*y[IDX_NH2I] -
        k[213]*y[IDX_OII]*y[IDX_NH3I] - k[214]*y[IDX_OII]*y[IDX_O2I] -
        k[215]*y[IDX_OII]*y[IDX_OHI] + k[216]*y[IDX_OI]*y[IDX_CNII] +
        k[217]*y[IDX_OI]*y[IDX_COII] + k[218]*y[IDX_OI]*y[IDX_N2II] +
        k[239]*y[IDX_OI] + k[282]*y[IDX_OI] + k[377]*y[IDX_CII]*y[IDX_O2I] +
        k[413]*y[IDX_CHII]*y[IDX_O2I] - k[472]*y[IDX_CHI]*y[IDX_OII] -
        k[543]*y[IDX_H2I]*y[IDX_OII] + k[666]*y[IDX_HeII]*y[IDX_CO2I] +
        k[694]*y[IDX_HeII]*y[IDX_NOI] + k[696]*y[IDX_HeII]*y[IDX_O2I] +
        k[698]*y[IDX_HeII]*y[IDX_OCNI] + k[699]*y[IDX_HeII]*y[IDX_OHI] +
        k[711]*y[IDX_HeII]*y[IDX_SiOI] + k[729]*y[IDX_NII]*y[IDX_O2I] -
        k[803]*y[IDX_NHI]*y[IDX_OII] - k[808]*y[IDX_OII]*y[IDX_CH3OHI] -
        k[809]*y[IDX_OII]*y[IDX_CH3OHI] - k[810]*y[IDX_OII]*y[IDX_CH4I] -
        k[811]*y[IDX_OII]*y[IDX_CNI] - k[812]*y[IDX_OII]*y[IDX_CO2I] -
        k[813]*y[IDX_OII]*y[IDX_H2COI] - k[814]*y[IDX_OII]*y[IDX_HCNI] -
        k[815]*y[IDX_OII]*y[IDX_HCNI] - k[816]*y[IDX_OII]*y[IDX_HCOI] -
        k[817]*y[IDX_OII]*y[IDX_N2I] - k[818]*y[IDX_OII]*y[IDX_NO2I] -
        k[819]*y[IDX_OII]*y[IDX_OHI] + k[1167]*y[IDX_O2II] + k[1173]*y[IDX_OHII]
        - k[1195]*y[IDX_CI]*y[IDX_OII] - k[1221]*y[IDX_OII]*y[IDX_EM] -
        k[1269]*y[IDX_OII];
    ydot[IDX_O2I] = 0.0 - k[6]*y[IDX_H2I]*y[IDX_O2I] -
        k[12]*y[IDX_HI]*y[IDX_O2I] + k[30]*y[IDX_CI]*y[IDX_O2II] +
        k[44]*y[IDX_CH2I]*y[IDX_O2II] - k[51]*y[IDX_CH4II]*y[IDX_O2I] +
        k[61]*y[IDX_CHI]*y[IDX_O2II] - k[68]*y[IDX_CNII]*y[IDX_O2I] -
        k[73]*y[IDX_COII]*y[IDX_O2I] - k[88]*y[IDX_HII]*y[IDX_O2I] -
        k[113]*y[IDX_H2II]*y[IDX_O2I] + k[116]*y[IDX_H2COI]*y[IDX_O2II] -
        k[121]*y[IDX_H2OII]*y[IDX_O2I] - k[133]*y[IDX_HCNII]*y[IDX_O2I] +
        k[137]*y[IDX_HCOI]*y[IDX_O2II] - k[146]*y[IDX_HeII]*y[IDX_O2I] +
        k[152]*y[IDX_MgI]*y[IDX_O2II] - k[168]*y[IDX_NII]*y[IDX_O2I] -
        k[173]*y[IDX_N2II]*y[IDX_O2I] - k[179]*y[IDX_NHII]*y[IDX_O2I] +
        k[187]*y[IDX_NH2I]*y[IDX_O2II] + k[198]*y[IDX_NH3I]*y[IDX_O2II] +
        k[205]*y[IDX_NOI]*y[IDX_O2II] - k[214]*y[IDX_OII]*y[IDX_O2I] -
        k[224]*y[IDX_OHII]*y[IDX_O2I] + k[230]*y[IDX_SiI]*y[IDX_O2II] -
        k[279]*y[IDX_O2I] - k[280]*y[IDX_O2I] + k[281]*y[IDX_O2HI] +
        k[347]*y[IDX_O2HII]*y[IDX_EM] - k[376]*y[IDX_CII]*y[IDX_O2I] -
        k[377]*y[IDX_CII]*y[IDX_O2I] + k[392]*y[IDX_CI]*y[IDX_O2HII] -
        k[411]*y[IDX_CHII]*y[IDX_O2I] - k[412]*y[IDX_CHII]*y[IDX_O2I] -
        k[413]*y[IDX_CHII]*y[IDX_O2I] - k[420]*y[IDX_CH2II]*y[IDX_O2I] +
        k[436]*y[IDX_CH2I]*y[IDX_O2HII] - k[442]*y[IDX_CH3II]*y[IDX_O2I] +
        k[474]*y[IDX_CHI]*y[IDX_O2HII] - k[481]*y[IDX_CNII]*y[IDX_O2I] +
        k[483]*y[IDX_CNI]*y[IDX_O2HII] + k[488]*y[IDX_COI]*y[IDX_O2HII] -
        k[525]*y[IDX_H2II]*y[IDX_O2I] + k[544]*y[IDX_H2I]*y[IDX_O2HII] -
        k[549]*y[IDX_H2COII]*y[IDX_O2I] + k[551]*y[IDX_H2COI]*y[IDX_O2II] +
        k[552]*y[IDX_H2COI]*y[IDX_O2HII] + k[571]*y[IDX_H2OI]*y[IDX_O2HII] -
        k[597]*y[IDX_H3II]*y[IDX_O2I] + k[632]*y[IDX_HCNI]*y[IDX_O2HII] +
        k[645]*y[IDX_HCOI]*y[IDX_O2HII] + k[651]*y[IDX_HNCI]*y[IDX_O2HII] +
        k[668]*y[IDX_HeII]*y[IDX_CO2I] - k[696]*y[IDX_HeII]*y[IDX_O2I] -
        k[728]*y[IDX_NII]*y[IDX_O2I] - k[729]*y[IDX_NII]*y[IDX_O2I] +
        k[733]*y[IDX_N2I]*y[IDX_O2HII] - k[765]*y[IDX_NHII]*y[IDX_O2I] -
        k[766]*y[IDX_NHII]*y[IDX_O2I] - k[777]*y[IDX_NH2II]*y[IDX_O2I] -
        k[778]*y[IDX_NH2II]*y[IDX_O2I] + k[790]*y[IDX_NH2I]*y[IDX_O2HII] +
        k[805]*y[IDX_NHI]*y[IDX_O2HII] + k[807]*y[IDX_NOI]*y[IDX_O2HII] +
        k[818]*y[IDX_OII]*y[IDX_NO2I] + k[820]*y[IDX_O2II]*y[IDX_CH3OHI] +
        k[821]*y[IDX_O2HII]*y[IDX_CO2I] + k[824]*y[IDX_OI]*y[IDX_HCO2II] +
        k[829]*y[IDX_OI]*y[IDX_O2HII] + k[835]*y[IDX_OI]*y[IDX_SiOII] +
        k[858]*y[IDX_OHI]*y[IDX_O2HII] - k[862]*y[IDX_SiH2II]*y[IDX_O2I] -
        k[873]*y[IDX_CI]*y[IDX_O2I] - k[889]*y[IDX_CH2I]*y[IDX_O2I] -
        k[890]*y[IDX_CH2I]*y[IDX_O2I] - k[891]*y[IDX_CH2I]*y[IDX_O2I] -
        k[892]*y[IDX_CH2I]*y[IDX_O2I] - k[893]*y[IDX_CH2I]*y[IDX_O2I] -
        k[911]*y[IDX_CH3I]*y[IDX_O2I] - k[912]*y[IDX_CH3I]*y[IDX_O2I] -
        k[913]*y[IDX_CH3I]*y[IDX_O2I] + k[914]*y[IDX_CH3I]*y[IDX_O2HI] -
        k[921]*y[IDX_CH4I]*y[IDX_O2I] - k[933]*y[IDX_CHI]*y[IDX_O2I] -
        k[934]*y[IDX_CHI]*y[IDX_O2I] - k[935]*y[IDX_CHI]*y[IDX_O2I] -
        k[936]*y[IDX_CHI]*y[IDX_O2I] + k[938]*y[IDX_CHI]*y[IDX_O2HI] -
        k[948]*y[IDX_CNI]*y[IDX_O2I] - k[949]*y[IDX_CNI]*y[IDX_O2I] -
        k[953]*y[IDX_COI]*y[IDX_O2I] - k[963]*y[IDX_H2I]*y[IDX_O2I] -
        k[964]*y[IDX_H2I]*y[IDX_O2I] - k[989]*y[IDX_HI]*y[IDX_O2I] +
        k[991]*y[IDX_HI]*y[IDX_O2HI] - k[1001]*y[IDX_HCOI]*y[IDX_O2I] -
        k[1002]*y[IDX_HCOI]*y[IDX_O2I] + k[1003]*y[IDX_HCOI]*y[IDX_O2HI] +
        k[1021]*y[IDX_NI]*y[IDX_NO2I] - k[1023]*y[IDX_NI]*y[IDX_O2I] +
        k[1024]*y[IDX_NI]*y[IDX_O2HI] - k[1044]*y[IDX_NHI]*y[IDX_O2I] -
        k[1045]*y[IDX_NHI]*y[IDX_O2I] + k[1051]*y[IDX_NOI]*y[IDX_NOI] -
        k[1052]*y[IDX_NOI]*y[IDX_O2I] - k[1054]*y[IDX_O2I]*y[IDX_OCNI] -
        k[1055]*y[IDX_O2I]*y[IDX_OCNI] + k[1059]*y[IDX_OI]*y[IDX_CO2I] +
        k[1070]*y[IDX_OI]*y[IDX_HNOI] + k[1075]*y[IDX_OI]*y[IDX_NO2I] +
        k[1076]*y[IDX_OI]*y[IDX_NOI] + k[1077]*y[IDX_OI]*y[IDX_O2HI] +
        k[1079]*y[IDX_OI]*y[IDX_OCNI] + k[1080]*y[IDX_OI]*y[IDX_OHI] +
        k[1100]*y[IDX_OHI]*y[IDX_O2HI] - k[1106]*y[IDX_SiI]*y[IDX_O2I] -
        k[1168]*y[IDX_O2I] - k[1169]*y[IDX_O2I] + k[1170]*y[IDX_O2HI] +
        k[1211]*y[IDX_OI]*y[IDX_OI] - k[1292]*y[IDX_O2I] + k[1355]*y[IDX_GO2I] +
        k[1356]*y[IDX_GO2I] + k[1357]*y[IDX_GO2I] + k[1358]*y[IDX_GO2I];
    ydot[IDX_O2II] = 0.0 - k[30]*y[IDX_CI]*y[IDX_O2II] -
        k[44]*y[IDX_CH2I]*y[IDX_O2II] + k[51]*y[IDX_CH4II]*y[IDX_O2I] -
        k[61]*y[IDX_CHI]*y[IDX_O2II] + k[68]*y[IDX_CNII]*y[IDX_O2I] +
        k[73]*y[IDX_COII]*y[IDX_O2I] + k[88]*y[IDX_HII]*y[IDX_O2I] +
        k[113]*y[IDX_H2II]*y[IDX_O2I] - k[116]*y[IDX_H2COI]*y[IDX_O2II] +
        k[121]*y[IDX_H2OII]*y[IDX_O2I] + k[133]*y[IDX_HCNII]*y[IDX_O2I] -
        k[137]*y[IDX_HCOI]*y[IDX_O2II] + k[146]*y[IDX_HeII]*y[IDX_O2I] -
        k[152]*y[IDX_MgI]*y[IDX_O2II] + k[168]*y[IDX_NII]*y[IDX_O2I] +
        k[173]*y[IDX_N2II]*y[IDX_O2I] + k[179]*y[IDX_NHII]*y[IDX_O2I] -
        k[187]*y[IDX_NH2I]*y[IDX_O2II] - k[198]*y[IDX_NH3I]*y[IDX_O2II] -
        k[205]*y[IDX_NOI]*y[IDX_O2II] + k[214]*y[IDX_OII]*y[IDX_O2I] +
        k[224]*y[IDX_OHII]*y[IDX_O2I] - k[230]*y[IDX_SiI]*y[IDX_O2II] +
        k[279]*y[IDX_O2I] - k[346]*y[IDX_O2II]*y[IDX_EM] -
        k[391]*y[IDX_CI]*y[IDX_O2II] - k[435]*y[IDX_CH2I]*y[IDX_O2II] -
        k[473]*y[IDX_CHI]*y[IDX_O2II] - k[551]*y[IDX_H2COI]*y[IDX_O2II] -
        k[644]*y[IDX_HCOI]*y[IDX_O2II] + k[667]*y[IDX_HeII]*y[IDX_CO2I] -
        k[742]*y[IDX_NI]*y[IDX_O2II] - k[804]*y[IDX_NHI]*y[IDX_O2II] +
        k[812]*y[IDX_OII]*y[IDX_CO2I] + k[819]*y[IDX_OII]*y[IDX_OHI] -
        k[820]*y[IDX_O2II]*y[IDX_CH3OHI] + k[823]*y[IDX_OI]*y[IDX_H2OII] +
        k[830]*y[IDX_OI]*y[IDX_OHII] - k[1167]*y[IDX_O2II] + k[1168]*y[IDX_O2I]
        - k[1270]*y[IDX_O2II];
    ydot[IDX_O2HI] = 0.0 - k[281]*y[IDX_O2HI] +
        k[549]*y[IDX_H2COII]*y[IDX_O2I] + k[913]*y[IDX_CH3I]*y[IDX_O2I] -
        k[914]*y[IDX_CH3I]*y[IDX_O2HI] + k[921]*y[IDX_CH4I]*y[IDX_O2I] -
        k[937]*y[IDX_CHI]*y[IDX_O2HI] - k[938]*y[IDX_CHI]*y[IDX_O2HI] -
        k[954]*y[IDX_COI]*y[IDX_O2HI] + k[963]*y[IDX_H2I]*y[IDX_O2I] -
        k[990]*y[IDX_HI]*y[IDX_O2HI] - k[991]*y[IDX_HI]*y[IDX_O2HI] -
        k[992]*y[IDX_HI]*y[IDX_O2HI] + k[1002]*y[IDX_HCOI]*y[IDX_O2I] -
        k[1003]*y[IDX_HCOI]*y[IDX_O2HI] - k[1024]*y[IDX_NI]*y[IDX_O2HI] -
        k[1077]*y[IDX_OI]*y[IDX_O2HI] - k[1100]*y[IDX_OHI]*y[IDX_O2HI] -
        k[1170]*y[IDX_O2HI] - k[1171]*y[IDX_O2HI] - k[1303]*y[IDX_O2HI] +
        k[1367]*y[IDX_GO2HI] + k[1368]*y[IDX_GO2HI] + k[1369]*y[IDX_GO2HI] +
        k[1370]*y[IDX_GO2HI];
    ydot[IDX_O2HII] = 0.0 - k[347]*y[IDX_O2HII]*y[IDX_EM] -
        k[392]*y[IDX_CI]*y[IDX_O2HII] - k[436]*y[IDX_CH2I]*y[IDX_O2HII] -
        k[474]*y[IDX_CHI]*y[IDX_O2HII] - k[483]*y[IDX_CNI]*y[IDX_O2HII] -
        k[488]*y[IDX_COI]*y[IDX_O2HII] + k[525]*y[IDX_H2II]*y[IDX_O2I] -
        k[544]*y[IDX_H2I]*y[IDX_O2HII] - k[552]*y[IDX_H2COI]*y[IDX_O2HII] -
        k[571]*y[IDX_H2OI]*y[IDX_O2HII] + k[597]*y[IDX_H3II]*y[IDX_O2I] -
        k[632]*y[IDX_HCNI]*y[IDX_O2HII] + k[644]*y[IDX_HCOI]*y[IDX_O2II] -
        k[645]*y[IDX_HCOI]*y[IDX_O2HII] - k[651]*y[IDX_HNCI]*y[IDX_O2HII] -
        k[733]*y[IDX_N2I]*y[IDX_O2HII] + k[766]*y[IDX_NHII]*y[IDX_O2I] -
        k[790]*y[IDX_NH2I]*y[IDX_O2HII] - k[805]*y[IDX_NHI]*y[IDX_O2HII] -
        k[807]*y[IDX_NOI]*y[IDX_O2HII] - k[821]*y[IDX_O2HII]*y[IDX_CO2I] -
        k[829]*y[IDX_OI]*y[IDX_O2HII] - k[858]*y[IDX_OHI]*y[IDX_O2HII] -
        k[1297]*y[IDX_O2HII];
    ydot[IDX_OCNI] = 0.0 - k[283]*y[IDX_OCNI] -
        k[378]*y[IDX_CII]*y[IDX_OCNI] - k[697]*y[IDX_HeII]*y[IDX_OCNI] -
        k[698]*y[IDX_HeII]*y[IDX_OCNI] - k[874]*y[IDX_CI]*y[IDX_OCNI] +
        k[932]*y[IDX_CHI]*y[IDX_NOI] + k[945]*y[IDX_CNI]*y[IDX_NO2I] +
        k[947]*y[IDX_CNI]*y[IDX_NOI] + k[949]*y[IDX_CNI]*y[IDX_O2I] -
        k[993]*y[IDX_HI]*y[IDX_OCNI] - k[994]*y[IDX_HI]*y[IDX_OCNI] -
        k[995]*y[IDX_HI]*y[IDX_OCNI] + k[1016]*y[IDX_NI]*y[IDX_HCOI] -
        k[1053]*y[IDX_NOI]*y[IDX_OCNI] - k[1054]*y[IDX_O2I]*y[IDX_OCNI] -
        k[1055]*y[IDX_O2I]*y[IDX_OCNI] + k[1060]*y[IDX_OI]*y[IDX_H2CNI] +
        k[1065]*y[IDX_OI]*y[IDX_HCNI] - k[1078]*y[IDX_OI]*y[IDX_OCNI] -
        k[1079]*y[IDX_OI]*y[IDX_OCNI] + k[1091]*y[IDX_OHI]*y[IDX_CNI] -
        k[1172]*y[IDX_OCNI] - k[1225]*y[IDX_OCNI];
    ydot[IDX_OHI] = 0.0 + k[4]*y[IDX_H2I]*y[IDX_H2OI] -
        k[7]*y[IDX_H2I]*y[IDX_OHI] + k[11]*y[IDX_HI]*y[IDX_H2OI] -
        k[13]*y[IDX_HI]*y[IDX_OHI] + k[45]*y[IDX_CH2I]*y[IDX_OHII] +
        k[62]*y[IDX_CHI]*y[IDX_OHII] - k[90]*y[IDX_HII]*y[IDX_OHI] -
        k[114]*y[IDX_H2II]*y[IDX_OHI] - k[169]*y[IDX_NII]*y[IDX_OHI] +
        k[188]*y[IDX_NH2I]*y[IDX_OHII] - k[215]*y[IDX_OII]*y[IDX_OHI] +
        k[219]*y[IDX_OHII]*y[IDX_H2COI] + k[220]*y[IDX_OHII]*y[IDX_H2OI] +
        k[221]*y[IDX_OHII]*y[IDX_HCOI] + k[222]*y[IDX_OHII]*y[IDX_NH3I] +
        k[223]*y[IDX_OHII]*y[IDX_NOI] + k[224]*y[IDX_OHII]*y[IDX_O2I] -
        k[225]*y[IDX_OHI]*y[IDX_CNII] - k[226]*y[IDX_OHI]*y[IDX_COII] -
        k[227]*y[IDX_OHI]*y[IDX_N2II] + k[248]*y[IDX_CH3OHI] +
        k[256]*y[IDX_H2OI] - k[284]*y[IDX_OHI] + k[314]*y[IDX_H2OII]*y[IDX_EM] +
        k[317]*y[IDX_H3COII]*y[IDX_EM] + k[324]*y[IDX_H3OII]*y[IDX_EM] +
        k[325]*y[IDX_H3OII]*y[IDX_EM] + k[333]*y[IDX_HCO2II]*y[IDX_EM] +
        k[363]*y[IDX_SiOHII]*y[IDX_EM] - k[379]*y[IDX_CII]*y[IDX_OHI] +
        k[383]*y[IDX_CI]*y[IDX_H2OII] + k[411]*y[IDX_CHII]*y[IDX_O2I] -
        k[415]*y[IDX_CHII]*y[IDX_OHI] + k[420]*y[IDX_CH2II]*y[IDX_O2I] +
        k[424]*y[IDX_CH2I]*y[IDX_H2OII] - k[445]*y[IDX_CH3II]*y[IDX_OHI] +
        k[460]*y[IDX_CHI]*y[IDX_H2OII] + k[504]*y[IDX_HII]*y[IDX_NO2I] -
        k[527]*y[IDX_H2II]*y[IDX_OHI] + k[553]*y[IDX_H2OII]*y[IDX_COI] +
        k[554]*y[IDX_H2OII]*y[IDX_H2COI] + k[555]*y[IDX_H2OII]*y[IDX_H2OI] +
        k[556]*y[IDX_H2OII]*y[IDX_HCNI] + k[558]*y[IDX_H2OII]*y[IDX_HCOI] +
        k[559]*y[IDX_H2OII]*y[IDX_HNCI] + k[560]*y[IDX_H2OI]*y[IDX_CNII] +
        k[562]*y[IDX_H2OI]*y[IDX_COII] + k[569]*y[IDX_H2OI]*y[IDX_N2II] +
        k[595]*y[IDX_H3II]*y[IDX_NO2I] - k[600]*y[IDX_H3II]*y[IDX_OHI] +
        k[657]*y[IDX_HeII]*y[IDX_CH3OHI] + k[674]*y[IDX_HeII]*y[IDX_H2OI] -
        k[699]*y[IDX_HeII]*y[IDX_OHI] + k[757]*y[IDX_NHII]*y[IDX_H2OI] +
        k[765]*y[IDX_NHII]*y[IDX_O2I] - k[768]*y[IDX_NHII]*y[IDX_OHI] +
        k[772]*y[IDX_NH2II]*y[IDX_H2OI] + k[778]*y[IDX_NH2II]*y[IDX_O2I] +
        k[781]*y[IDX_NH2I]*y[IDX_H2OII] + k[809]*y[IDX_OII]*y[IDX_CH3OHI] +
        k[810]*y[IDX_OII]*y[IDX_CH4I] + k[813]*y[IDX_OII]*y[IDX_H2COI] -
        k[819]*y[IDX_OII]*y[IDX_OHI] + k[822]*y[IDX_OI]*y[IDX_CH4II] -
        k[847]*y[IDX_OHII]*y[IDX_OHI] - k[851]*y[IDX_OHI]*y[IDX_COII] -
        k[852]*y[IDX_OHI]*y[IDX_H2OII] - k[853]*y[IDX_OHI]*y[IDX_HCNII] -
        k[854]*y[IDX_OHI]*y[IDX_HCOII] - k[855]*y[IDX_OHI]*y[IDX_HCOII] -
        k[856]*y[IDX_OHI]*y[IDX_HNOII] - k[857]*y[IDX_OHI]*y[IDX_N2HII] -
        k[858]*y[IDX_OHI]*y[IDX_O2HII] - k[859]*y[IDX_OHI]*y[IDX_SiII] +
        k[862]*y[IDX_SiH2II]*y[IDX_O2I] - k[875]*y[IDX_CI]*y[IDX_OHI] -
        k[876]*y[IDX_CI]*y[IDX_OHI] + k[887]*y[IDX_CH2I]*y[IDX_NOI] +
        k[893]*y[IDX_CH2I]*y[IDX_O2I] + k[897]*y[IDX_CH2I]*y[IDX_OI] -
        k[898]*y[IDX_CH2I]*y[IDX_OHI] - k[899]*y[IDX_CH2I]*y[IDX_OHI] -
        k[900]*y[IDX_CH2I]*y[IDX_OHI] + k[904]*y[IDX_CH3I]*y[IDX_H2OI] +
        k[911]*y[IDX_CH3I]*y[IDX_O2I] - k[917]*y[IDX_CH3I]*y[IDX_OHI] -
        k[918]*y[IDX_CH3I]*y[IDX_OHI] - k[919]*y[IDX_CH3I]*y[IDX_OHI] -
        k[922]*y[IDX_CH4I]*y[IDX_OHI] + k[935]*y[IDX_CHI]*y[IDX_O2I] +
        k[937]*y[IDX_CHI]*y[IDX_O2HI] + k[940]*y[IDX_CHI]*y[IDX_OI] -
        k[941]*y[IDX_CHI]*y[IDX_OHI] + k[954]*y[IDX_COI]*y[IDX_O2HI] +
        k[964]*y[IDX_H2I]*y[IDX_O2I] + k[964]*y[IDX_H2I]*y[IDX_O2I] +
        k[965]*y[IDX_H2I]*y[IDX_OI] - k[966]*y[IDX_H2I]*y[IDX_OHI] +
        k[971]*y[IDX_HI]*y[IDX_CO2I] + k[972]*y[IDX_HI]*y[IDX_COI] +
        k[975]*y[IDX_HI]*y[IDX_H2OI] + k[982]*y[IDX_HI]*y[IDX_HNOI] +
        k[986]*y[IDX_HI]*y[IDX_NO2I] + k[988]*y[IDX_HI]*y[IDX_NOI] +
        k[989]*y[IDX_HI]*y[IDX_O2I] + k[992]*y[IDX_HI]*y[IDX_O2HI] +
        k[992]*y[IDX_HI]*y[IDX_O2HI] + k[995]*y[IDX_HI]*y[IDX_OCNI] -
        k[996]*y[IDX_HI]*y[IDX_OHI] + k[1001]*y[IDX_HCOI]*y[IDX_O2I] -
        k[1025]*y[IDX_NI]*y[IDX_OHI] - k[1026]*y[IDX_NI]*y[IDX_OHI] +
        k[1030]*y[IDX_NH2I]*y[IDX_NOI] - k[1031]*y[IDX_NH2I]*y[IDX_OHI] -
        k[1032]*y[IDX_NH2I]*y[IDX_OHI] + k[1036]*y[IDX_NHI]*y[IDX_H2OI] +
        k[1043]*y[IDX_NHI]*y[IDX_NOI] + k[1045]*y[IDX_NHI]*y[IDX_O2I] +
        k[1047]*y[IDX_NHI]*y[IDX_OI] - k[1048]*y[IDX_NHI]*y[IDX_OHI] -
        k[1049]*y[IDX_NHI]*y[IDX_OHI] - k[1050]*y[IDX_NHI]*y[IDX_OHI] +
        k[1056]*y[IDX_OI]*y[IDX_CH4I] + k[1061]*y[IDX_OI]*y[IDX_H2COI] +
        k[1062]*y[IDX_OI]*y[IDX_H2OI] + k[1062]*y[IDX_OI]*y[IDX_H2OI] +
        k[1063]*y[IDX_OI]*y[IDX_HCNI] + k[1067]*y[IDX_OI]*y[IDX_HCOI] +
        k[1069]*y[IDX_OI]*y[IDX_HNOI] + k[1073]*y[IDX_OI]*y[IDX_NH2I] +
        k[1074]*y[IDX_OI]*y[IDX_NH3I] + k[1077]*y[IDX_OI]*y[IDX_O2HI] -
        k[1080]*y[IDX_OI]*y[IDX_OHI] + k[1088]*y[IDX_OI]*y[IDX_SiH4I] -
        k[1090]*y[IDX_OHI]*y[IDX_CNI] - k[1091]*y[IDX_OHI]*y[IDX_CNI] -
        k[1092]*y[IDX_OHI]*y[IDX_COI] - k[1093]*y[IDX_OHI]*y[IDX_H2COI] -
        k[1094]*y[IDX_OHI]*y[IDX_HCNI] - k[1095]*y[IDX_OHI]*y[IDX_HCNI] -
        k[1096]*y[IDX_OHI]*y[IDX_HCOI] - k[1097]*y[IDX_OHI]*y[IDX_HNOI] -
        k[1098]*y[IDX_OHI]*y[IDX_NH3I] - k[1099]*y[IDX_OHI]*y[IDX_NOI] -
        k[1100]*y[IDX_OHI]*y[IDX_O2HI] - k[1101]*y[IDX_OHI]*y[IDX_OHI] -
        k[1101]*y[IDX_OHI]*y[IDX_OHI] - k[1102]*y[IDX_OHI]*y[IDX_SiI] +
        k[1121]*y[IDX_CH3OHI] + k[1142]*y[IDX_H2OI] + k[1171]*y[IDX_O2HI] -
        k[1174]*y[IDX_OHI] - k[1175]*y[IDX_OHI] + k[1207]*y[IDX_HI]*y[IDX_OI] -
        k[1208]*y[IDX_HI]*y[IDX_OHI] - k[1254]*y[IDX_OHI];
    ydot[IDX_OHII] = 0.0 - k[45]*y[IDX_CH2I]*y[IDX_OHII] -
        k[62]*y[IDX_CHI]*y[IDX_OHII] + k[90]*y[IDX_HII]*y[IDX_OHI] +
        k[114]*y[IDX_H2II]*y[IDX_OHI] + k[169]*y[IDX_NII]*y[IDX_OHI] -
        k[188]*y[IDX_NH2I]*y[IDX_OHII] + k[215]*y[IDX_OII]*y[IDX_OHI] -
        k[219]*y[IDX_OHII]*y[IDX_H2COI] - k[220]*y[IDX_OHII]*y[IDX_H2OI] -
        k[221]*y[IDX_OHII]*y[IDX_HCOI] - k[222]*y[IDX_OHII]*y[IDX_NH3I] -
        k[223]*y[IDX_OHII]*y[IDX_NOI] - k[224]*y[IDX_OHII]*y[IDX_O2I] +
        k[225]*y[IDX_OHI]*y[IDX_CNII] + k[226]*y[IDX_OHI]*y[IDX_COII] +
        k[227]*y[IDX_OHI]*y[IDX_N2II] - k[348]*y[IDX_OHII]*y[IDX_EM] -
        k[393]*y[IDX_CI]*y[IDX_OHII] - k[437]*y[IDX_CH2I]*y[IDX_OHII] -
        k[457]*y[IDX_CH4I]*y[IDX_OHII] - k[475]*y[IDX_CHI]*y[IDX_OHII] +
        k[526]*y[IDX_H2II]*y[IDX_OI] + k[543]*y[IDX_H2I]*y[IDX_OII] -
        k[545]*y[IDX_H2I]*y[IDX_OHII] + k[599]*y[IDX_H3II]*y[IDX_OI] +
        k[656]*y[IDX_HeII]*y[IDX_CH3OHI] + k[673]*y[IDX_HeII]*y[IDX_H2OI] -
        k[743]*y[IDX_NI]*y[IDX_OHII] + k[767]*y[IDX_NHII]*y[IDX_OI] -
        k[791]*y[IDX_NH2I]*y[IDX_OHII] - k[806]*y[IDX_NHI]*y[IDX_OHII] +
        k[816]*y[IDX_OII]*y[IDX_HCOI] + k[826]*y[IDX_OI]*y[IDX_N2HII] +
        k[829]*y[IDX_OI]*y[IDX_O2HII] - k[830]*y[IDX_OI]*y[IDX_OHII] -
        k[836]*y[IDX_OHII]*y[IDX_CNI] - k[837]*y[IDX_OHII]*y[IDX_CO2I] -
        k[838]*y[IDX_OHII]*y[IDX_COI] - k[839]*y[IDX_OHII]*y[IDX_H2COI] -
        k[840]*y[IDX_OHII]*y[IDX_H2OI] - k[841]*y[IDX_OHII]*y[IDX_HCNI] -
        k[842]*y[IDX_OHII]*y[IDX_HCOI] - k[843]*y[IDX_OHII]*y[IDX_HCOI] -
        k[844]*y[IDX_OHII]*y[IDX_HNCI] - k[845]*y[IDX_OHII]*y[IDX_N2I] -
        k[846]*y[IDX_OHII]*y[IDX_NOI] - k[847]*y[IDX_OHII]*y[IDX_OHI] -
        k[848]*y[IDX_OHII]*y[IDX_SiI] - k[849]*y[IDX_OHII]*y[IDX_SiHI] -
        k[850]*y[IDX_OHII]*y[IDX_SiOI] + k[1140]*y[IDX_H2OII] -
        k[1173]*y[IDX_OHII] + k[1175]*y[IDX_OHI] - k[1274]*y[IDX_OHII];
    ydot[IDX_SiI] = 0.0 - k[21]*y[IDX_CII]*y[IDX_SiI] -
        k[35]*y[IDX_CHII]*y[IDX_SiI] - k[91]*y[IDX_HII]*y[IDX_SiI] -
        k[122]*y[IDX_H2OII]*y[IDX_SiI] - k[147]*y[IDX_HeII]*y[IDX_SiI] +
        k[153]*y[IDX_MgI]*y[IDX_SiII] - k[192]*y[IDX_NH3II]*y[IDX_SiI] -
        k[228]*y[IDX_SiI]*y[IDX_H2COII] - k[229]*y[IDX_SiI]*y[IDX_NOII] -
        k[230]*y[IDX_SiI]*y[IDX_O2II] - k[285]*y[IDX_SiI] + k[288]*y[IDX_SiCI] +
        k[292]*y[IDX_SiHI] + k[293]*y[IDX_SiOI] + k[349]*y[IDX_SiCII]*y[IDX_EM]
        + k[352]*y[IDX_SiHII]*y[IDX_EM] + k[353]*y[IDX_SiH2II]*y[IDX_EM] +
        k[354]*y[IDX_SiH2II]*y[IDX_EM] + k[362]*y[IDX_SiOII]*y[IDX_EM] +
        k[363]*y[IDX_SiOHII]*y[IDX_EM] + k[477]*y[IDX_CHI]*y[IDX_SiHII] +
        k[478]*y[IDX_CHI]*y[IDX_SiOII] + k[573]*y[IDX_H2OI]*y[IDX_SiHII] -
        k[601]*y[IDX_H3II]*y[IDX_SiI] - k[610]*y[IDX_H3OII]*y[IDX_SiI] +
        k[702]*y[IDX_HeII]*y[IDX_SiCI] + k[711]*y[IDX_HeII]*y[IDX_SiOI] +
        k[745]*y[IDX_NI]*y[IDX_SiOII] - k[848]*y[IDX_OHII]*y[IDX_SiI] -
        k[861]*y[IDX_SiI]*y[IDX_HCOII] + k[1027]*y[IDX_NI]*y[IDX_SiCI] +
        k[1083]*y[IDX_OI]*y[IDX_SiCI] - k[1102]*y[IDX_OHI]*y[IDX_SiI] -
        k[1103]*y[IDX_SiI]*y[IDX_CO2I] - k[1104]*y[IDX_SiI]*y[IDX_COI] -
        k[1105]*y[IDX_SiI]*y[IDX_NOI] - k[1106]*y[IDX_SiI]*y[IDX_O2I] -
        k[1176]*y[IDX_SiI] + k[1178]*y[IDX_SiCI] + k[1188]*y[IDX_SiHI] +
        k[1190]*y[IDX_SiOI] - k[1213]*y[IDX_OI]*y[IDX_SiI] +
        k[1222]*y[IDX_SiII]*y[IDX_EM] - k[1228]*y[IDX_SiI];
    ydot[IDX_SiII] = 0.0 + k[21]*y[IDX_CII]*y[IDX_SiI] +
        k[35]*y[IDX_CHII]*y[IDX_SiI] + k[91]*y[IDX_HII]*y[IDX_SiI] +
        k[122]*y[IDX_H2OII]*y[IDX_SiI] + k[147]*y[IDX_HeII]*y[IDX_SiI] -
        k[153]*y[IDX_MgI]*y[IDX_SiII] + k[192]*y[IDX_NH3II]*y[IDX_SiI] +
        k[228]*y[IDX_SiI]*y[IDX_H2COII] + k[229]*y[IDX_SiI]*y[IDX_NOII] +
        k[230]*y[IDX_SiI]*y[IDX_O2II] + k[285]*y[IDX_SiI] +
        k[382]*y[IDX_CII]*y[IDX_SiOI] + k[395]*y[IDX_CI]*y[IDX_SiOII] +
        k[438]*y[IDX_CH2I]*y[IDX_SiOII] - k[476]*y[IDX_CHI]*y[IDX_SiII] +
        k[490]*y[IDX_COI]*y[IDX_SiOII] + k[508]*y[IDX_HII]*y[IDX_SiHI] -
        k[572]*y[IDX_H2OI]*y[IDX_SiII] + k[619]*y[IDX_HI]*y[IDX_SiHII] +
        k[701]*y[IDX_HeII]*y[IDX_SiCI] + k[703]*y[IDX_HeII]*y[IDX_SiH2I] +
        k[707]*y[IDX_HeII]*y[IDX_SiH4I] + k[709]*y[IDX_HeII]*y[IDX_SiHI] +
        k[710]*y[IDX_HeII]*y[IDX_SiOI] + k[744]*y[IDX_NI]*y[IDX_SiCII] +
        k[746]*y[IDX_NI]*y[IDX_SiOII] + k[835]*y[IDX_OI]*y[IDX_SiOII] -
        k[859]*y[IDX_OHI]*y[IDX_SiII] - k[860]*y[IDX_SiII]*y[IDX_CH3OHI] +
        k[1176]*y[IDX_SiI] + k[1179]*y[IDX_SiHII] + k[1189]*y[IDX_SiOII] -
        k[1202]*y[IDX_H2I]*y[IDX_SiII] - k[1209]*y[IDX_HI]*y[IDX_SiII] -
        k[1212]*y[IDX_OI]*y[IDX_SiII] - k[1222]*y[IDX_SiII]*y[IDX_EM] -
        k[1231]*y[IDX_SiII];
    ydot[IDX_SiCI] = 0.0 - k[24]*y[IDX_CII]*y[IDX_SiCI] -
        k[94]*y[IDX_HII]*y[IDX_SiCI] + k[286]*y[IDX_SiC2I] - k[288]*y[IDX_SiCI]
        + k[350]*y[IDX_SiC2II]*y[IDX_EM] - k[701]*y[IDX_HeII]*y[IDX_SiCI] -
        k[702]*y[IDX_HeII]*y[IDX_SiCI] + k[877]*y[IDX_CI]*y[IDX_SiHI] -
        k[1027]*y[IDX_NI]*y[IDX_SiCI] + k[1081]*y[IDX_OI]*y[IDX_SiC2I] -
        k[1083]*y[IDX_OI]*y[IDX_SiCI] - k[1084]*y[IDX_OI]*y[IDX_SiCI] -
        k[1178]*y[IDX_SiCI] - k[1239]*y[IDX_SiCI] + k[1371]*y[IDX_GSiCI] +
        k[1372]*y[IDX_GSiCI] + k[1373]*y[IDX_GSiCI] + k[1374]*y[IDX_GSiCI];
    ydot[IDX_SiCII] = 0.0 + k[24]*y[IDX_CII]*y[IDX_SiCI] +
        k[94]*y[IDX_HII]*y[IDX_SiCI] - k[349]*y[IDX_SiCII]*y[IDX_EM] +
        k[380]*y[IDX_CII]*y[IDX_SiH2I] + k[381]*y[IDX_CII]*y[IDX_SiHI] +
        k[394]*y[IDX_CI]*y[IDX_SiHII] + k[476]*y[IDX_CHI]*y[IDX_SiII] -
        k[744]*y[IDX_NI]*y[IDX_SiCII] - k[831]*y[IDX_OI]*y[IDX_SiCII] -
        k[1241]*y[IDX_SiCII];
    ydot[IDX_SiC2I] = 0.0 - k[22]*y[IDX_CII]*y[IDX_SiC2I] -
        k[92]*y[IDX_HII]*y[IDX_SiC2I] - k[286]*y[IDX_SiC2I] +
        k[287]*y[IDX_SiC3I] + k[351]*y[IDX_SiC3II]*y[IDX_EM] -
        k[1081]*y[IDX_OI]*y[IDX_SiC2I] + k[1082]*y[IDX_OI]*y[IDX_SiC3I] +
        k[1177]*y[IDX_SiC3I] - k[1240]*y[IDX_SiC2I] + k[1395]*y[IDX_GSiC2I] +
        k[1396]*y[IDX_GSiC2I] + k[1397]*y[IDX_GSiC2I] + k[1398]*y[IDX_GSiC2I];
    ydot[IDX_SiC2II] = 0.0 + k[22]*y[IDX_CII]*y[IDX_SiC2I] +
        k[92]*y[IDX_HII]*y[IDX_SiC2I] - k[350]*y[IDX_SiC2II]*y[IDX_EM] +
        k[700]*y[IDX_HeII]*y[IDX_SiC3I] - k[1242]*y[IDX_SiC2II];
    ydot[IDX_SiC3I] = 0.0 - k[23]*y[IDX_CII]*y[IDX_SiC3I] -
        k[93]*y[IDX_HII]*y[IDX_SiC3I] - k[287]*y[IDX_SiC3I] -
        k[700]*y[IDX_HeII]*y[IDX_SiC3I] - k[1082]*y[IDX_OI]*y[IDX_SiC3I] -
        k[1177]*y[IDX_SiC3I] - k[1243]*y[IDX_SiC3I] + k[1399]*y[IDX_GSiC3I] +
        k[1400]*y[IDX_GSiC3I] + k[1401]*y[IDX_GSiC3I] + k[1402]*y[IDX_GSiC3I];
    ydot[IDX_SiC3II] = 0.0 + k[23]*y[IDX_CII]*y[IDX_SiC3I] +
        k[93]*y[IDX_HII]*y[IDX_SiC3I] - k[351]*y[IDX_SiC3II]*y[IDX_EM] -
        k[1244]*y[IDX_SiC3II];
    ydot[IDX_SiHI] = 0.0 - k[98]*y[IDX_HII]*y[IDX_SiHI] +
        k[289]*y[IDX_SiH2I] - k[292]*y[IDX_SiHI] +
        k[355]*y[IDX_SiH2II]*y[IDX_EM] + k[357]*y[IDX_SiH3II]*y[IDX_EM] -
        k[381]*y[IDX_CII]*y[IDX_SiHI] - k[508]*y[IDX_HII]*y[IDX_SiHI] -
        k[605]*y[IDX_H3II]*y[IDX_SiHI] - k[612]*y[IDX_H3OII]*y[IDX_SiHI] -
        k[639]*y[IDX_HCOII]*y[IDX_SiHI] - k[709]*y[IDX_HeII]*y[IDX_SiHI] -
        k[849]*y[IDX_OHII]*y[IDX_SiHI] - k[877]*y[IDX_CI]*y[IDX_SiHI] -
        k[1089]*y[IDX_OI]*y[IDX_SiHI] + k[1181]*y[IDX_SiH2I] +
        k[1184]*y[IDX_SiH3I] + k[1187]*y[IDX_SiH4I] - k[1188]*y[IDX_SiHI] -
        k[1230]*y[IDX_SiHI];
    ydot[IDX_SiHII] = 0.0 + k[98]*y[IDX_HII]*y[IDX_SiHI] -
        k[352]*y[IDX_SiHII]*y[IDX_EM] - k[394]*y[IDX_CI]*y[IDX_SiHII] -
        k[477]*y[IDX_CHI]*y[IDX_SiHII] + k[505]*y[IDX_HII]*y[IDX_SiH2I] -
        k[573]*y[IDX_H2OI]*y[IDX_SiHII] + k[601]*y[IDX_H3II]*y[IDX_SiI] +
        k[610]*y[IDX_H3OII]*y[IDX_SiI] - k[619]*y[IDX_HI]*y[IDX_SiHII] +
        k[704]*y[IDX_HeII]*y[IDX_SiH2I] + k[705]*y[IDX_HeII]*y[IDX_SiH3I] +
        k[708]*y[IDX_HeII]*y[IDX_SiH4I] - k[832]*y[IDX_OI]*y[IDX_SiHII] +
        k[848]*y[IDX_OHII]*y[IDX_SiI] + k[861]*y[IDX_SiI]*y[IDX_HCOII] -
        k[1179]*y[IDX_SiHII] - k[1203]*y[IDX_H2I]*y[IDX_SiHII] +
        k[1209]*y[IDX_HI]*y[IDX_SiII] - k[1232]*y[IDX_SiHII];
    ydot[IDX_SiH2I] = 0.0 - k[25]*y[IDX_CII]*y[IDX_SiH2I] -
        k[95]*y[IDX_HII]*y[IDX_SiH2I] - k[289]*y[IDX_SiH2I] +
        k[290]*y[IDX_SiH3I] + k[291]*y[IDX_SiH4I] +
        k[356]*y[IDX_SiH3II]*y[IDX_EM] + k[358]*y[IDX_SiH4II]*y[IDX_EM] -
        k[380]*y[IDX_CII]*y[IDX_SiH2I] - k[505]*y[IDX_HII]*y[IDX_SiH2I] -
        k[602]*y[IDX_H3II]*y[IDX_SiH2I] - k[611]*y[IDX_H3OII]*y[IDX_SiH2I] -
        k[637]*y[IDX_HCOII]*y[IDX_SiH2I] - k[703]*y[IDX_HeII]*y[IDX_SiH2I] -
        k[704]*y[IDX_HeII]*y[IDX_SiH2I] - k[1085]*y[IDX_OI]*y[IDX_SiH2I] -
        k[1086]*y[IDX_OI]*y[IDX_SiH2I] - k[1180]*y[IDX_SiH2I] -
        k[1181]*y[IDX_SiH2I] + k[1182]*y[IDX_SiH3I] + k[1185]*y[IDX_SiH4I] -
        k[1233]*y[IDX_SiH2I];
    ydot[IDX_SiH2II] = 0.0 + k[25]*y[IDX_CII]*y[IDX_SiH2I] +
        k[95]*y[IDX_HII]*y[IDX_SiH2I] - k[353]*y[IDX_SiH2II]*y[IDX_EM] -
        k[354]*y[IDX_SiH2II]*y[IDX_EM] - k[355]*y[IDX_SiH2II]*y[IDX_EM] +
        k[506]*y[IDX_HII]*y[IDX_SiH3I] + k[605]*y[IDX_H3II]*y[IDX_SiHI] +
        k[612]*y[IDX_H3OII]*y[IDX_SiHI] + k[639]*y[IDX_HCOII]*y[IDX_SiHI] +
        k[706]*y[IDX_HeII]*y[IDX_SiH3I] - k[833]*y[IDX_OI]*y[IDX_SiH2II] +
        k[849]*y[IDX_OHII]*y[IDX_SiHI] - k[862]*y[IDX_SiH2II]*y[IDX_O2I] +
        k[1180]*y[IDX_SiH2I] + k[1202]*y[IDX_H2I]*y[IDX_SiII] -
        k[1234]*y[IDX_SiH2II];
    ydot[IDX_SiH3I] = 0.0 - k[26]*y[IDX_CII]*y[IDX_SiH3I] -
        k[96]*y[IDX_HII]*y[IDX_SiH3I] - k[290]*y[IDX_SiH3I] +
        k[359]*y[IDX_SiH4II]*y[IDX_EM] + k[360]*y[IDX_SiH5II]*y[IDX_EM] +
        k[489]*y[IDX_COI]*y[IDX_SiH4II] - k[506]*y[IDX_HII]*y[IDX_SiH3I] +
        k[574]*y[IDX_H2OI]*y[IDX_SiH4II] - k[603]*y[IDX_H3II]*y[IDX_SiH3I] -
        k[705]*y[IDX_HeII]*y[IDX_SiH3I] - k[706]*y[IDX_HeII]*y[IDX_SiH3I] +
        k[950]*y[IDX_CNI]*y[IDX_SiH4I] - k[1087]*y[IDX_OI]*y[IDX_SiH3I] +
        k[1088]*y[IDX_OI]*y[IDX_SiH4I] - k[1182]*y[IDX_SiH3I] -
        k[1183]*y[IDX_SiH3I] - k[1184]*y[IDX_SiH3I] + k[1186]*y[IDX_SiH4I] -
        k[1235]*y[IDX_SiH3I];
    ydot[IDX_SiH3II] = 0.0 + k[26]*y[IDX_CII]*y[IDX_SiH3I] +
        k[96]*y[IDX_HII]*y[IDX_SiH3I] - k[356]*y[IDX_SiH3II]*y[IDX_EM] -
        k[357]*y[IDX_SiH3II]*y[IDX_EM] + k[446]*y[IDX_CH3II]*y[IDX_SiH4I] +
        k[507]*y[IDX_HII]*y[IDX_SiH4I] + k[602]*y[IDX_H3II]*y[IDX_SiH2I] +
        k[611]*y[IDX_H3OII]*y[IDX_SiH2I] + k[637]*y[IDX_HCOII]*y[IDX_SiH2I] -
        k[834]*y[IDX_OI]*y[IDX_SiH3II] + k[1183]*y[IDX_SiH3I] +
        k[1203]*y[IDX_H2I]*y[IDX_SiHII] - k[1204]*y[IDX_H2I]*y[IDX_SiH3II] -
        k[1236]*y[IDX_SiH3II];
    ydot[IDX_SiH4I] = 0.0 - k[97]*y[IDX_HII]*y[IDX_SiH4I] -
        k[291]*y[IDX_SiH4I] + k[361]*y[IDX_SiH5II]*y[IDX_EM] -
        k[446]*y[IDX_CH3II]*y[IDX_SiH4I] - k[507]*y[IDX_HII]*y[IDX_SiH4I] +
        k[575]*y[IDX_H2OI]*y[IDX_SiH5II] - k[604]*y[IDX_H3II]*y[IDX_SiH4I] -
        k[638]*y[IDX_HCOII]*y[IDX_SiH4I] - k[707]*y[IDX_HeII]*y[IDX_SiH4I] -
        k[708]*y[IDX_HeII]*y[IDX_SiH4I] - k[950]*y[IDX_CNI]*y[IDX_SiH4I] -
        k[1088]*y[IDX_OI]*y[IDX_SiH4I] - k[1185]*y[IDX_SiH4I] -
        k[1186]*y[IDX_SiH4I] - k[1187]*y[IDX_SiH4I] - k[1237]*y[IDX_SiH4I] +
        k[1363]*y[IDX_GSiH4I] + k[1364]*y[IDX_GSiH4I] + k[1365]*y[IDX_GSiH4I] +
        k[1366]*y[IDX_GSiH4I];
    ydot[IDX_SiH4II] = 0.0 + k[97]*y[IDX_HII]*y[IDX_SiH4I] -
        k[358]*y[IDX_SiH4II]*y[IDX_EM] - k[359]*y[IDX_SiH4II]*y[IDX_EM] -
        k[489]*y[IDX_COI]*y[IDX_SiH4II] - k[546]*y[IDX_H2I]*y[IDX_SiH4II] -
        k[574]*y[IDX_H2OI]*y[IDX_SiH4II] + k[603]*y[IDX_H3II]*y[IDX_SiH3I] -
        k[1238]*y[IDX_SiH4II];
    ydot[IDX_SiH5II] = 0.0 - k[360]*y[IDX_SiH5II]*y[IDX_EM] -
        k[361]*y[IDX_SiH5II]*y[IDX_EM] + k[546]*y[IDX_H2I]*y[IDX_SiH4II] -
        k[575]*y[IDX_H2OI]*y[IDX_SiH5II] + k[604]*y[IDX_H3II]*y[IDX_SiH4I] +
        k[638]*y[IDX_HCOII]*y[IDX_SiH4I] + k[1204]*y[IDX_H2I]*y[IDX_SiH3II] -
        k[1248]*y[IDX_SiH5II];
    ydot[IDX_SiOI] = 0.0 - k[99]*y[IDX_HII]*y[IDX_SiOI] +
        k[138]*y[IDX_HCOI]*y[IDX_SiOII] + k[154]*y[IDX_MgI]*y[IDX_SiOII] +
        k[206]*y[IDX_NOI]*y[IDX_SiOII] + k[257]*y[IDX_H2SiOI] -
        k[293]*y[IDX_SiOI] + k[364]*y[IDX_SiOHII]*y[IDX_EM] -
        k[382]*y[IDX_CII]*y[IDX_SiOI] - k[606]*y[IDX_H3II]*y[IDX_SiOI] -
        k[613]*y[IDX_H3OII]*y[IDX_SiOI] - k[640]*y[IDX_HCOII]*y[IDX_SiOI] -
        k[710]*y[IDX_HeII]*y[IDX_SiOI] - k[711]*y[IDX_HeII]*y[IDX_SiOI] -
        k[850]*y[IDX_OHII]*y[IDX_SiOI] + k[1084]*y[IDX_OI]*y[IDX_SiCI] +
        k[1085]*y[IDX_OI]*y[IDX_SiH2I] + k[1086]*y[IDX_OI]*y[IDX_SiH2I] +
        k[1089]*y[IDX_OI]*y[IDX_SiHI] + k[1102]*y[IDX_OHI]*y[IDX_SiI] +
        k[1103]*y[IDX_SiI]*y[IDX_CO2I] + k[1104]*y[IDX_SiI]*y[IDX_COI] +
        k[1105]*y[IDX_SiI]*y[IDX_NOI] + k[1106]*y[IDX_SiI]*y[IDX_O2I] +
        k[1143]*y[IDX_H2SiOI] + k[1144]*y[IDX_H2SiOI] - k[1190]*y[IDX_SiOI] -
        k[1191]*y[IDX_SiOI] + k[1213]*y[IDX_OI]*y[IDX_SiI] - k[1229]*y[IDX_SiOI]
        + k[1379]*y[IDX_GSiOI] + k[1380]*y[IDX_GSiOI] + k[1381]*y[IDX_GSiOI] +
        k[1382]*y[IDX_GSiOI];
    ydot[IDX_SiOII] = 0.0 + k[99]*y[IDX_HII]*y[IDX_SiOI] -
        k[138]*y[IDX_HCOI]*y[IDX_SiOII] - k[154]*y[IDX_MgI]*y[IDX_SiOII] -
        k[206]*y[IDX_NOI]*y[IDX_SiOII] - k[362]*y[IDX_SiOII]*y[IDX_EM] -
        k[395]*y[IDX_CI]*y[IDX_SiOII] - k[438]*y[IDX_CH2I]*y[IDX_SiOII] -
        k[478]*y[IDX_CHI]*y[IDX_SiOII] - k[490]*y[IDX_COI]*y[IDX_SiOII] -
        k[547]*y[IDX_H2I]*y[IDX_SiOII] - k[745]*y[IDX_NI]*y[IDX_SiOII] -
        k[746]*y[IDX_NI]*y[IDX_SiOII] + k[831]*y[IDX_OI]*y[IDX_SiCII] +
        k[832]*y[IDX_OI]*y[IDX_SiHII] - k[835]*y[IDX_OI]*y[IDX_SiOII] +
        k[859]*y[IDX_OHI]*y[IDX_SiII] - k[1189]*y[IDX_SiOII] +
        k[1191]*y[IDX_SiOI] + k[1212]*y[IDX_OI]*y[IDX_SiII] -
        k[1245]*y[IDX_SiOII];
    ydot[IDX_SiOHII] = 0.0 - k[363]*y[IDX_SiOHII]*y[IDX_EM] -
        k[364]*y[IDX_SiOHII]*y[IDX_EM] + k[499]*y[IDX_HII]*y[IDX_H2SiOI] +
        k[547]*y[IDX_H2I]*y[IDX_SiOII] + k[572]*y[IDX_H2OI]*y[IDX_SiII] +
        k[606]*y[IDX_H3II]*y[IDX_SiOI] + k[613]*y[IDX_H3OII]*y[IDX_SiOI] +
        k[640]*y[IDX_HCOII]*y[IDX_SiOI] + k[675]*y[IDX_HeII]*y[IDX_H2SiOI] +
        k[833]*y[IDX_OI]*y[IDX_SiH2II] + k[834]*y[IDX_OI]*y[IDX_SiH3II] +
        k[850]*y[IDX_OHII]*y[IDX_SiOI] + k[860]*y[IDX_SiII]*y[IDX_CH3OHI] +
        k[862]*y[IDX_SiH2II]*y[IDX_O2I] - k[1246]*y[IDX_SiOHII];
    
    ydot[IDX_H2I] += H2formation*y[IDX_HI]*nH - H2dissociation*y[IDX_H2I];
    ydot[IDX_HI] += 2.0*(H2dissociation*y[IDX_H2I] - H2formation*y[IDX_HI]*nH);
    
// clang-format on

    /* */

    return NAUNET_SUCCESS;
}