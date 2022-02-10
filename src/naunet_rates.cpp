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

#define IJth(A, i, j) SM_ELEMENT_D(A, i, j)

// clang-format off
int EvalRates(realtype *k, realtype *y, NaunetData *u_data) {

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

    // Some variable definitions from krome
    realtype Te      = Tgas * 8.617343e-5;            // Tgas in eV (eV)
    realtype lnTe    = log(Te);                       // ln of Te (#)
    realtype T32     = Tgas * 0.0033333333333333335;  // Tgas/(300 K) (#)
    realtype invT    = 1.0 / Tgas;                    // inverse of T (1/K)
    realtype invTe   = 1.0 / Te;                      // inverse of T (1/eV)
    realtype sqrTgas = sqrt(Tgas);  // Tgas rootsquare (K**0.5)

    // reaaction rate (k) of each reaction
    // clang-format off
    if (Tgas>10.0 && Tgas<2500.0) { k[0] = 1.09e-11 * pow(Tgas/300.0, -2.19)
        * exp(-165.1/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>1340.0 && Tgas<41000.0) { k[2] = 6e-09 * pow(Tgas/300.0, 0.0) *
        exp(-40200.0/Tgas);  }
        
    if (Tgas>2803.0 && Tgas<41000.0) { k[3] = 1e-08 * pow(Tgas/300.0, 0.0) *
        exp(-84100.0/Tgas);  }
        
    if (Tgas>1763.0 && Tgas<41000.0) { k[4] = 5.8e-09 * pow(Tgas/300.0, 0.0)
        * exp(-52900.0/Tgas);  }
        
    if (Tgas>20.0 && Tgas<41000.0) { k[5] = 3.8e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>1743.0 && Tgas<41000.0) { k[6] = 6e-09 * pow(Tgas/300.0, 0.0) *
        exp(-52300.0/Tgas);  }
        
    if (Tgas>1696.0 && Tgas<41000.0) { k[7] = 6e-09 * pow(Tgas/300.0, 0.0) *
        exp(-50900.0/Tgas);  }
        
    if (Tgas>3400.0 && Tgas<41000.0) { k[8] = 3.22e-09 * pow(Tgas/300.0,
        0.35) * exp(-102000.0/Tgas);  }
        
    if (Tgas>1340.0 && Tgas<41000.0) { k[9] = 6e-09 * pow(Tgas/300.0, 0.0) *
        exp(-40200.0/Tgas);  }
        
    if (Tgas>1833.0 && Tgas<41000.0) { k[10] = 4.67e-07 * pow(Tgas/300.0,
        -1.0) * exp(-55000.0/Tgas);  }
        
    if (Tgas>1763.0 && Tgas<41000.0) { k[11] = 5.8e-09 * pow(Tgas/300.0,
        0.0) * exp(-52900.0/Tgas);  }
        
    if (Tgas>1743.0 && Tgas<41000.0) { k[12] = 6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-52300.0/Tgas);  }
        
    if (Tgas>1696.0 && Tgas<41000.0) { k[13] = 6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-50900.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[14] = 5.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[15] = 3.8e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[16] = 7.8e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[17] = 4.8e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[18] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[19] = 6.72e-10 * pow(Tgas/300.0, 0.0)
        * exp(+0.5/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[20] = 7.05e-10 * pow(Tgas/300.0,
        -0.03) * exp(+16.7/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[21] = 2.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[22] = 2e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[23] = 2e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[24] = 2.5e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[25] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[26] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[27] = 1.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[28] = 1.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[29] = 1.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[30] = 5.2e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[31] = 4.6e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[32] = 3.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[33] = 4.59e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[34] = 7.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[35] = 2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[36] = 4.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[37] = 8.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[38] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[39] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[40] = 4.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[41] = 8.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[42] = 4.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[43] = 9.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[44] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[45] = 4.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[46] = 4.4e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[47] = 3.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[48] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[49] = 1.62e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[50] = 1.65e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[51] = 3.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[52] = 7.93e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[53] = 6.4e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[54] = 3.2e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[55] = 3.1e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[56] = 3.4e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[57] = 3.6e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[58] = 6.3e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[59] = 3.5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[60] = 3.5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[61] = 3.1e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[62] = 3.5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[63] = 6.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[64] = 5.2e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[65] = 1.79e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[66] = 3.7e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[67] = 5.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[68] = 2.58e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[69] = 1e-10 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[70] = 1.35e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[71] = 7.4e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[72] = 3.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[73] = 1.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[74] = 7.4e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[75] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[76] = 3.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[77] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[78] = 1.9e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[79] = 2.96e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[80] = 6.9e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>205.0 && Tgas<565.0) { k[81] = 1.05e-08 * pow(Tgas/300.0,
        -0.13) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[82] = 9.4e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[83] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[84] = 2.9e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[85] = 3.7e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[86] = 2.1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[87] = 2.9e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[88] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[89] = 6.86e-10 * pow(Tgas/300.0,
        0.26) * exp(-224.3/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[90] = 2.1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[91] = 9.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[92] = 3e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[93] = 3e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[94] = 3e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[95] = 1.5e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[96] = 1.5e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[97] = 1.5e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[98] = 1.7e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[99] = 3.3e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[100] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[101] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[102] = 7.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[103] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[104] = 6.44e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[105] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[106] = 3.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[107] = 2.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[108] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[109] = 2.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[110] = 5.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[111] = 7.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[112] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[113] = 8e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[114] = 7.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[115] = 7.2e-15 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[116] = 2.07e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[117] = 1.41e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[118] = 2.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[119] = 2.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[120] = 2.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[121] = 4.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[122] = 3e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[123] = 1.72e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[124] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[125] = 2.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[126] = 6.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[127] = 7.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[128] = 6.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[129] = 3.7e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[130] = 1.2e-15 * pow(Tgas/300.0,
        0.25) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[131] = 5.66e-10 * pow(Tgas/300.0,
        0.36) * exp(+8.6/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[132] = 8.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[133] = 3.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[134] = 3.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[135] = 3.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[136] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[137] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[138] = 6.6e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[139] = 6.3e-15 * pow(Tgas/300.0, 0.75)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[140] = 5.1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[141] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[142] = 9.69e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[143] = 6.05e-11 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[144] = 6.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[145] = 2.64e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[146] = 3.3e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[147] = 3.3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[148] = 2.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[149] = 2.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[150] = 7e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[151] = 8.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[152] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[153] = 2.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[154] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[155] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[156] = 2.8e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[157] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[158] = 8.25e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[159] = 1.88e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[160] = 2.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[161] = 3.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[162] = 4.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[163] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[164] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[165] = 1.97e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[166] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[167] = 4.51e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[168] = 3.11e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[169] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[170] = 3.77e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[171] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[172] = 4.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[173] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[174] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[175] = 9.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[176] = 1.05e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[177] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[178] = 7.12e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[179] = 4.51e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[180] = 4.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[181] = 6.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[182] = 7e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[183] = 9.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[184] = 4.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[185] = 4.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[186] = 8.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[187] = 8.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[188] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[189] = 4.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[190] = 3.3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[191] = 7.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[192] = 1.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[193] = 2.02e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[194] = 4.25e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[195] = 2.21e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[196] = 1.68e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[197] = 1.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[198] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[199] = 6.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[200] = 3.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[201] = 6.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[202] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[203] = 7.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[204] = 7e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[205] = 4.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[206] = 7.2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[207] = 8.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>2000.0 && Tgas<10000.0) { k[208] = 4.9e-12 * pow(Tgas/300.0,
        0.5) * exp(-4580.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[209] = 2.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[210] = 3.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[211] = 4.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[212] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[213] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[214] = 1.9e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[215] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[216] = 6.5e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[217] = 1.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[218] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[219] = 7.44e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[220] = 1.59e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[221] = 2.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[222] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[223] = 3.59e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[224] = 5.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[225] = 6.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[226] = 3.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[227] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[228] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[229] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[230] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[231] = 2.3e-17 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[232] = 3.9e-17 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[233] = 2.86e-19 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[234] = 1.2e-17 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[235] = 1.3e-18 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[236] = 5.98e-18 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[237] = 6.5e-18 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[238] = 2.7e-17 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[239] = 3.4e-17 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[240] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 255.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[241] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 88.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[242] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[243] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[244] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[245] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[246] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[247] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1584.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[248] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 752.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[249] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1169.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[250] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 365.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[251] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 5290.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[252] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 854.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[253] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 1.17) * 105.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[254] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[255] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1329.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[256] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 485.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[257] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[258] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 0.2 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[259] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1557.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[260] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 210.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[261] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 584.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[262] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1500.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[263] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1500.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[264] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 500.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[265] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 0.2 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[266] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 66.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[267] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 25.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[268] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1.1 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[269] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 324.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[270] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 40.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[271] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 657.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[272] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 288.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[273] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 270.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[274] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[275] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[276] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[277] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 247.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[278] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 231.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[279] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 58.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[280] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 375.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[281] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 375.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[282] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1.4 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[283] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[284] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 254.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[285] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 2115.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[286] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[287] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[288] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[289] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[290] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[291] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[292] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[293] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[294] = 1.5e-07 * pow(Tgas/300.0, -0.42)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[295] = 7.68e-08 * pow(Tgas/300.0,
        -0.6) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[296] = 4.03e-07 * pow(Tgas/300.0,
        -0.6) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[297] = 1.6e-07 * pow(Tgas/300.0, -0.6)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[298] = 7.75e-08 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[299] = 1.95e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[300] = 2e-07 * pow(Tgas/300.0, -0.4) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[301] = 1.75e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[302] = 1.75e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[303] = 1.8e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[304] = 2e-07 * pow(Tgas/300.0, -0.48)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[305] = 1.6e-08 * pow(Tgas/300.0, -0.43)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[306] = 2.5e-08 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[307] = 7.5e-08 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[308] = 2.5e-07 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[309] = 1.6e-07 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[310] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[311] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[312] = 3.9e-08 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[313] = 3.05e-07 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[314] = 8.6e-08 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[315] = 2.34e-08 * pow(Tgas/300.0,
        -0.52) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[316] = 4.36e-08 * pow(Tgas/300.0,
        -0.52) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[317] = 4.2e-08 * pow(Tgas/300.0,
        -0.78) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[318] = 1.4e-08 * pow(Tgas/300.0,
        -0.78) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[319] = 2.1e-07 * pow(Tgas/300.0,
        -0.78) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[320] = 2.17e-07 * pow(Tgas/300.0,
        -0.78) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[321] = 2.17e-07 * pow(Tgas/300.0,
        -0.78) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[322] = 7.09e-08 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[323] = 5.6e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[324] = 5.37e-08 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[325] = 3.05e-07 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[326] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[327] = 9.3e-08 * pow(Tgas/300.0, -0.65)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[328] = 9.5e-08 * pow(Tgas/300.0, -0.65)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[329] = 9.5e-08 * pow(Tgas/300.0, -0.65)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[330] = 2.4e-07 * pow(Tgas/300.0, -0.69)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[331] = 6e-08 * pow(Tgas/300.0, -0.64) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[332] = 8.1e-07 * pow(Tgas/300.0, -0.64)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[333] = 3.2e-07 * pow(Tgas/300.0, -0.64)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[334] = 3e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[335] = 1.1e-07 * pow(Tgas/300.0, -1.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[336] = 1e-08 * pow(Tgas/300.0, -0.6) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[337] = 1.7e-07 * pow(Tgas/300.0, -0.3)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<150.0) { k[338] = 2.77e-07 * pow(Tgas/300.0,
        -0.74) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<150.0) { k[339] = 2.09e-08 * pow(Tgas/300.0,
        -0.74) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[340] = 4.3e-08 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>12.0 && Tgas<12400.0) { k[341] = 1.78e-07 * pow(Tgas/300.0,
        -0.8) * exp(-17.1/Tgas);  }
        
    if (Tgas>12.0 && Tgas<12400.0) { k[342] = 9.21e-08 * pow(Tgas/300.0,
        -0.79) * exp(-17.1/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[343] = 1.55e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[344] = 1.55e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[345] = 4.3e-07 * pow(Tgas/300.0, -0.37)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[346] = 1.95e-07 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[347] = 3e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[348] = 3.75e-08 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[349] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[350] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[351] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[352] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[353] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[354] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[355] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[356] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[357] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[358] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[359] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[360] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[361] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[362] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[363] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[364] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[365] = 5.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[366] = 2.08e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[367] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[368] = 2.34e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[369] = 7.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[370] = 9e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[371] = 2.09e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[372] = 4.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[373] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[374] = 1.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(+0.5/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[375] = 7.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[376] = 3.42e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[377] = 4.54e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[378] = 3.8e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[379] = 7.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[380] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[381] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[382] = 5.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[383] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[384] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[385] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[386] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[387] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[388] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[389] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[390] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[391] = 5.2e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[392] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[393] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[394] = 2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[395] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[396] = 1.45e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[397] = 2.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[398] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[399] = 9.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[400] = 9.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[401] = 9.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[402] = 5.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[403] = 5.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[404] = 2.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[405] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[406] = 4.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[407] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[408] = 1.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[409] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[410] = 7.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[411] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[412] = 9.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[413] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[414] = 3.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[415] = 7.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[416] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[417] = 2.81e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[418] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[419] = 4.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[420] = 9.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[421] = 7.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[422] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[423] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[424] = 4.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[425] = 9.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[426] = 8.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[427] = 4.35e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[428] = 4.35e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[429] = 8.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[430] = 8.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[431] = 8.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[432] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[433] = 4.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[434] = 9.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[435] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[436] = 8.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[437] = 4.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[438] = 8.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[439] = 2.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[440] = 1.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[441] = 4.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[442] = 5e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[443] = 4e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[444] = 4e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[445] = 7.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[446] = 1.8e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[447] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[448] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[449] = 1.98e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[450] = 2.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[451] = 4.55e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[452] = 9.35e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[453] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[454] = 1.04e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[455] = 7e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[456] = 9.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[457] = 1.31e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[458] = 3.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[459] = 3.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[460] = 3.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[461] = 6.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[462] = 6.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[463] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[464] = 3.15e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[465] = 3.15e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[466] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[467] = 6.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[468] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[469] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[470] = 9.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[471] = 3.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[472] = 3.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[473] = 3.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[474] = 6.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[475] = 3.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[476] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[477] = 6e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[478] = 5.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[479] = 5.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[480] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[481] = 8.6e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[482] = 8.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[483] = 8.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[484] = 1.65e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[485] = 7.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[486] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[487] = 8.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[488] = 8.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[489] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[490] = 7.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[491] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[492] = 5.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[493] = 3.84e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[494] = 8.85e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[495] = 2.3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[496] = 3.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[497] = 1.06e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[498] = 3.57e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[499] = 1.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[500] = 9.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[501] = 9.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[502] = 7.94e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[503] = 4e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[504] = 1.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[505] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[506] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[507] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[508] = 1.7e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[509] = 2.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[510] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[511] = 2.3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[512] = 7.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[513] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[514] = 2.35e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[515] = 2.16e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[516] = 2.08e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[517] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[518] = 3.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[519] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[520] = 1.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[521] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[522] = 1.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[523] = 7.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[524] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[525] = 1.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[526] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[527] = 7.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>154.0 && Tgas<3000.0) { k[528] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-4640.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[529] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[530] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[531] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[532] = 7.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[533] = 7.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[534] = 6.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[535] = 9e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[536] = 3.7e-14 * pow(Tgas/300.0, 0.0) *
        exp(-35.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[537] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[538] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-85.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[539] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[540] = 2.25e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[541] = 1.28e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[542] = 2.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[543] = 1.7e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>80.0 && Tgas<300.0) { k[544] = 6.4e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[545] = 1.01e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[546] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[547] = 3.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[548] = 3.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[549] = 7.7e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[550] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[551] = 2.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[552] = 9.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[553] = 5e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[554] = 6.62e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[555] = 2.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[556] = 2.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[557] = 2.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[558] = 2.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[559] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[560] = 1.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[561] = 1.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[562] = 8.84e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[563] = 2.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[564] = 2.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[565] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[566] = 2.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[567] = 2.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[568] = 2.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[569] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[570] = 2.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[571] = 8.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[572] = 2.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[573] = 8e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[574] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[575] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[576] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[577] = 1.7e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[578] = 2.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[579] = 3.71e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[580] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[581] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[582] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<400.0) { k[583] = 1.36e-09 * pow(Tgas/300.0,
        -0.14) * exp(+3.4/Tgas);  }
        
    if (Tgas>10.0 && Tgas<400.0) { k[584] = 8.49e-10 * pow(Tgas/300.0, 0.07)
        * exp(-5.2/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[585] = 6.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[586] = 5.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[587] = 8.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[588] = 1.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[589] = 8.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[590] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[591] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[592] = 1.8e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[593] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[594] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[595] = 7e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[596] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<3000.0) { k[597] = 9.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-100.0/Tgas);  }
        
    if (Tgas>5.0 && Tgas<400.0) { k[598] = 3.42e-10 * pow(Tgas/300.0, -0.16)
        * exp(-1.4/Tgas);  }
        
    if (Tgas>5.0 && Tgas<400.0) { k[599] = 7.98e-10 * pow(Tgas/300.0, -0.16)
        * exp(-1.4/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[600] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[601] = 3.7e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[602] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[603] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[604] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[605] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[606] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[607] = 3.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[608] = 3.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[609] = 4e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[610] = 1.8e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[611] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[612] = 9.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[613] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[614] = 9.06e-10 * pow(Tgas/300.0,
        -0.37) * exp(-29.1/Tgas);  }
        
    if (Tgas>236.0 && Tgas<300.0) { k[615] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-7080.0/Tgas);  }
        
    if (Tgas>352.0 && Tgas<41000.0) { k[616] = 7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-10560.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[617] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[618] = 9.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[619] = 1.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[620] = 2.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[621] = 1.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[622] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[623] = 1.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[624] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[625] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[626] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[627] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[628] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>205.0 && Tgas<565.0) { k[629] = 3.1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[630] = 9.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[631] = 3.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[632] = 9.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[633] = 1.05e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[634] = 1.05e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[635] = 3.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[636] = 7.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[637] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[638] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[639] = 8.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[640] = 7.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[641] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[642] = 7.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[643] = 7.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[644] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[645] = 7.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[646] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[647] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[648] = 3.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[649] = 9.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[650] = 3.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[651] = 9.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[652] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[653] = 7.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[654] = 7.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[655] = 1.8e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[656] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[657] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[658] = 2.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[659] = 9.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[660] = 8.5e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[661] = 4.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[662] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[663] = 8.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[664] = 8.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[665] = 8.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[666] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[667] = 1.1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[668] = 4e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[669] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[670] = 1.88e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[671] = 1.14e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[672] = 1.71e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[673] = 2.86e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[674] = 2.04e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[675] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[676] = 1.46e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[677] = 2.17e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[678] = 7.75e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[679] = 6.51e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[680] = 4.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[681] = 3e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[682] = 4.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[683] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[684] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[685] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[686] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[687] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[688] = 9.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[689] = 8e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[690] = 8e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[691] = 1.76e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[692] = 1.76e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[693] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[694] = 2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[695] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[696] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[697] = 3e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[698] = 3e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[699] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[700] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[701] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[702] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[703] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[704] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[705] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[706] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[707] = 1.41e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[708] = 9.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[709] = 1.8e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[710] = 8.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[711] = 8.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[712] = 9.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[713] = 4.96e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[714] = 3.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[715] = 1.24e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[716] = 4.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[717] = 5.6e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[718] = 3.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[719] = 2.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[720] = 1.45e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[721] = 7.25e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[722] = 2.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[723] = 4.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[724] = 2.16e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[725] = 2.16e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[726] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[727] = 7.9e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[728] = 2.63e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[729] = 3.66e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[730] = 2.52e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[731] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[732] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[733] = 8e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[734] = 9.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[735] = 3.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[736] = 2.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[737] = 6.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[738] = 1.12e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[739] = 2.8e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[740] = 1.3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[741] = 9.1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[742] = 1.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[743] = 8.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[744] = 7.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[745] = 9e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[746] = 2.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[747] = 1.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[748] = 3.85e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[749] = 3.85e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[750] = 3.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[751] = 4.41e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[752] = 4.95e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[753] = 1.82e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[754] = 1.05e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[755] = 3.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[756] = 1.75e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[757] = 8.75e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[758] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[759] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[760] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[761] = 6.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[762] = 1.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[763] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[764] = 1.78e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[765] = 2.05e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[766] = 1.64e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[767] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[768] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[769] = 2.24e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[770] = 5.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[771] = 2.76e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[772] = 1e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[773] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[774] = 4.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[775] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[776] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[777] = 1.19e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[778] = 2.1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[779] = 4.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[780] = 8.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[781] = 4.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[782] = 8.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[783] = 9.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[784] = 9e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[785] = 4.45e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[786] = 4.45e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[787] = 8.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[788] = 8.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[789] = 8.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[790] = 8.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[791] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[792] = 4.08e-11 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[793] = 8.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[794] = 7.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[795] = 3.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[796] = 6.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[797] = 7.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[798] = 6.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[799] = 6.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[800] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[801] = 6.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[802] = 7.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[803] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[804] = 3.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[805] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[806] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[807] = 7.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[808] = 9.5e-11 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[809] = 1.33e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[810] = 1.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[811] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[812] = 9.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[813] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>205.0 && Tgas<565.0) { k[814] = 1.2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>20.0 && Tgas<5565.0) { k[815] = 1.2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[816] = 4.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>23.0 && Tgas<41000.0) { k[817] = 2.42e-12 * pow(Tgas/300.0,
        -0.21) * exp(+44.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[818] = 8.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[819] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[820] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[821] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[822] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[823] = 4e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[824] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[825] = 1.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[826] = 1.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[827] = 7.2e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[828] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[829] = 6.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[830] = 7.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[831] = 6e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[832] = 4e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[833] = 6.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[834] = 2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[835] = 2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[836] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[837] = 1.44e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[838] = 1.05e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[839] = 1.12e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[840] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[841] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[842] = 2.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[843] = 2.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[844] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[845] = 3.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[846] = 6.11e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[847] = 7e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[848] = 1.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[849] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[850] = 9.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[851] = 3.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[852] = 6.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[853] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[854] = 6.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[855] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[856] = 6.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[857] = 6.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[858] = 6.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[859] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[860] = 1.65e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[861] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[862] = 2.4e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<4000.0) { k[863] = 2.69e-12 * pow(Tgas/300.0,
        0.0) * exp(-23550.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[864] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>2000.0 && Tgas<5000.0) { k[865] = 8.69e-11 * pow(Tgas/300.0,
        0.0) * exp(-22600.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[866] = 3.26e-11 * pow(Tgas/300.0, -0.1)
        * exp(+9.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[867] = 3.26e-11 * pow(Tgas/300.0, -0.1)
        * exp(+9.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<4000.0) { k[868] = 9.62e-13 * pow(Tgas/300.0,
        0.0) * exp(-10517.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[869] = 1.2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>133.0 && Tgas<300.0) { k[870] = 1.73e-11 * pow(Tgas/300.0, 0.5)
        * exp(-4000.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[871] = 6e-11 * pow(Tgas/300.0, -0.16)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[872] = 9e-11 * pow(Tgas/300.0, -0.16) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<8000.0) { k[873] = 5.56e-11 * pow(Tgas/300.0,
        0.41) * exp(+26.9/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[874] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[875] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>493.0 && Tgas<41000.0) { k[876] = 2.25e-11 * pow(Tgas/300.0,
        0.5) * exp(-14800.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[877] = 6.59e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[878] = 4e-10 * pow(Tgas/300.0, 0.0) *
        exp(-5000.0/Tgas);  }
        
    if (Tgas>296.0 && Tgas<2500.0) { k[879] = 7.13e-12 * pow(Tgas/300.0,
        0.0) * exp(-5050.0/Tgas);  }
        
    if (Tgas>83.0 && Tgas<300.0) { k[880] = 5.3e-12 * pow(Tgas/300.0, 0.0) *
        exp(-2500.0/Tgas);  }
        
    if (Tgas>109.0 && Tgas<300.0) { k[881] = 3.3e-13 * pow(Tgas/300.0, 0.0)
        * exp(-3270.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[882] = 3e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[883] = 1.7e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>1500.0 && Tgas<2000.0) { k[884] = 8e-12 * pow(Tgas/300.0, 0.0)
        * exp(-18000.0/Tgas);  }
        
    k[885] = 6.91e-11 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>116.0 && Tgas<300.0) { k[886] = 2.7e-12 * pow(Tgas/300.0, 0.0)
        * exp(-3500.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2000.0) { k[887] = 3.65e-12 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[888] = 3.65e-12 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<1828.0) { k[889] = 2.92e-11 * pow(Tgas/300.0,
        -3.3) * exp(-1443.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<1828.0) { k[890] = 3.65e-11 * pow(Tgas/300.0,
        -3.3) * exp(-1443.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<1828.0) { k[891] = 2.48e-10 * pow(Tgas/300.0,
        -3.3) * exp(-1443.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<1828.0) { k[892] = 3.65e-11 * pow(Tgas/300.0,
        -3.3) * exp(-1443.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[893] = 4.1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-750.0/Tgas);  }
        
    if (Tgas>1900.0 && Tgas<2600.0) { k[894] = 8e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[895] = 1.33e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>1200.0 && Tgas<1812.0) { k[896] = 5.01e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>1900.0 && Tgas<2300.0) { k[897] = 4.98e-10 * pow(Tgas/300.0,
        0.0) * exp(-6000.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[898] = 3e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>100.0 && Tgas<300.0) { k[899] = 1.44e-11 * pow(Tgas/300.0, 0.5)
        * exp(-3000.0/Tgas);  }
        
    if (Tgas>100.0 && Tgas<300.0) { k[900] = 1.44e-11 * pow(Tgas/300.0, 0.5)
        * exp(-3000.0/Tgas);  }
        
    if (Tgas>1950.0 && Tgas<2300.0) { k[901] = 7.13e-12 * pow(Tgas/300.0,
        0.0) * exp(-5052.0/Tgas);  }
        
    if (Tgas>50.0 && Tgas<300.0) { k[902] = 9.21e-12 * pow(Tgas/300.0, 0.7)
        * exp(-1500.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[903] = 1.34e-15 * pow(Tgas/300.0,
        5.05) * exp(-1636.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<3000.0) { k[904] = 2.3e-15 * pow(Tgas/300.0,
        3.47) * exp(-6681.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[905] = 2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<913.0) { k[906] = 3.32e-12 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2000.0) { k[907] = 4.76e-17 * pow(Tgas/300.0,
        5.77) * exp(+151.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[908] = 9.55e-14 * pow(Tgas/300.0,
        0.0) * exp(-4890.0/Tgas);  }
        
    k[909] = 5.41e-12 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>1100.0 && Tgas<2080.0) { k[910] = 4e-12 * pow(Tgas/300.0, 0.0)
        * exp(-7900.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3000.0) { k[911] = 5.64e-13 * pow(Tgas/300.0,
        0.0) * exp(-4500.0/Tgas);  }
        
    if (Tgas>1700.0 && Tgas<2000.0) { k[912] = 1.66e-12 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>1165.0 && Tgas<41000.0) { k[913] = 5.3e-12 * pow(Tgas/300.0,
        0.0) * exp(-34975.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[914] = 6e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>354.0 && Tgas<925.0) { k[915] = 3.6e-11 * pow(Tgas/300.0, 0.0)
        * exp(-202.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[916] = 1.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2500.0) { k[917] = 3.27e-14 * pow(Tgas/300.0,
        2.2) * exp(-2240.0/Tgas);  }
        
    k[918] = 1.7e-12 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>300.0 && Tgas<1000.0) { k[919] = 1.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-1400.0/Tgas);  }
        
    if (Tgas>160.0 && Tgas<2500.0) { k[920] = 3.14e-12 * pow(Tgas/300.0,
        1.53) * exp(-504.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[921] = 6.7e-11 * pow(Tgas/300.0, 0.0)
        * exp(-28640.0/Tgas);  }
        
    if (Tgas>178.0 && Tgas<3000.0) { k[922] = 3.77e-13 * pow(Tgas/300.0,
        2.42) * exp(-1162.0/Tgas);  }
        
    if (Tgas>100.0 && Tgas<300.0) { k[923] = 2.94e-13 * pow(Tgas/300.0, 0.5)
        * exp(-3000.0/Tgas);  }
        
    if (Tgas>66.0 && Tgas<300.0) { k[924] = 9.21e-12 * pow(Tgas/300.0, 0.7)
        * exp(-2000.0/Tgas);  }
        
    if (Tgas>16.0 && Tgas<300.0) { k[925] = 2.87e-12 * pow(Tgas/300.0, 0.7)
        * exp(-500.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[926] = 1.73e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<4000.0) { k[927] = 5.6e-13 * pow(Tgas/300.0,
        0.88) * exp(-10128.0/Tgas);  }
        
    if (Tgas>222.0 && Tgas<584.0) { k[928] = 1.66e-10 * pow(Tgas/300.0,
        -0.09) * exp(-0.0/Tgas);  }
        
    if (Tgas>990.0 && Tgas<1100.0) { k[929] = 3.03e-11 * pow(Tgas/300.0,
        0.65) * exp(-1207.0/Tgas);  }
        
    if (Tgas>13.0 && Tgas<708.0) { k[930] = 1.2e-10 * pow(Tgas/300.0, -0.13)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>13.0 && Tgas<708.0) { k[931] = 1.16e-11 * pow(Tgas/300.0,
        -0.13) * exp(-0.0/Tgas);  }
        
    if (Tgas>13.0 && Tgas<708.0) { k[932] = 3.49e-11 * pow(Tgas/300.0,
        -0.13) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[933] = 1.14e-11 * pow(Tgas/300.0,
        -0.48) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[934] = 1.14e-11 * pow(Tgas/300.0,
        -0.48) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[935] = 7.6e-12 * pow(Tgas/300.0, -0.48)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[936] = 7.6e-12 * pow(Tgas/300.0, -0.48)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>100.0 && Tgas<300.0) { k[937] = 1.44e-11 * pow(Tgas/300.0, 0.5)
        * exp(-3000.0/Tgas);  }
        
    if (Tgas>251.0 && Tgas<300.0) { k[938] = 2.94e-13 * pow(Tgas/300.0, 0.5)
        * exp(-7550.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2000.0) { k[939] = 6.02e-11 * pow(Tgas/300.0, 0.1)
        * exp(+4.5/Tgas);  }
        
    if (Tgas>10.0 && Tgas<6000.0) { k[940] = 2.52e-11 * pow(Tgas/300.0, 0.0)
        * exp(-2381.0/Tgas);  }
        
    if (Tgas>166.0 && Tgas<300.0) { k[941] = 1.44e-11 * pow(Tgas/300.0, 0.5)
        * exp(-5000.0/Tgas);  }
        
    if (Tgas>297.0 && Tgas<2500.0) { k[942] = 2.6e-10 * pow(Tgas/300.0,
        -0.47) * exp(-826.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[943] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[944] = 3e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>297.0 && Tgas<2500.0) { k[945] = 7.02e-11 * pow(Tgas/300.0,
        -0.27) * exp(-8.3/Tgas);  }
        
    if (Tgas>300.0 && Tgas<1500.0) { k[946] = 1.6e-13 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<4430.0) { k[947] = 1.62e-10 * pow(Tgas/300.0,
        0.0) * exp(-21205.0/Tgas);  }
        
    if (Tgas>13.0 && Tgas<1565.0) { k[948] = 5.12e-12 * pow(Tgas/300.0,
        -0.49) * exp(+5.2/Tgas);  }
        
    if (Tgas>13.0 && Tgas<4526.0) { k[949] = 2.02e-11 * pow(Tgas/300.0,
        -0.19) * exp(+31.9/Tgas);  }
        
    k[950] = 2.2e-10 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>10.0 && Tgas<1900.0) { k[951] = 3.32e-12 * pow(Tgas/300.0, 0.0)
        * exp(-6170.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[952] = 1.48e-10 * pow(Tgas/300.0,
        0.0) * exp(-17000.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<6000.0) { k[953] = 5.99e-12 * pow(Tgas/300.0,
        0.0) * exp(-24075.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<2500.0) { k[954] = 5.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-12160.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[955] = 6.64e-10 * pow(Tgas/300.0,
        0.0) * exp(-11700.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[956] = 5.18e-11 * pow(Tgas/300.0,
        0.17) * exp(-6400.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[957] = 6.86e-14 * pow(Tgas/300.0,
        2.74) * exp(-4740.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[958] = 5.46e-10 * pow(Tgas/300.0,
        0.0) * exp(-1943.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<3500.0) { k[959] = 4.04e-13 * pow(Tgas/300.0,
        2.87) * exp(-820.0/Tgas);  }
        
    if (Tgas>1600.0 && Tgas<2850.0) { k[960] = 1.69e-09 * pow(Tgas/300.0,
        0.0) * exp(-18095.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[961] = 2.05e-15 * pow(Tgas/300.0,
        3.89) * exp(-1400.0/Tgas);  }
        
    if (Tgas>2601.0 && Tgas<2788.0) { k[962] = 5.96e-11 * pow(Tgas/300.0,
        0.0) * exp(-7782.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[963] = 2.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-28500.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[964] = 3.16e-10 * pow(Tgas/300.0,
        0.0) * exp(-21890.0/Tgas);  }
        
    if (Tgas>297.0 && Tgas<3532.0) { k[965] = 3.14e-13 * pow(Tgas/300.0,
        2.7) * exp(-3150.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<2581.0) { k[966] = 2.05e-12 * pow(Tgas/300.0,
        1.52) * exp(-1736.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[967] = 2.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[968] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-7600.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[969] = 5.94e-13 * pow(Tgas/300.0,
        3.0) * exp(-4045.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2000.0) { k[970] = 1.31e-10 * pow(Tgas/300.0,
        0.0) * exp(-80.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<15223.0) { k[971] = 3.38e-10 * pow(Tgas/300.0,
        0.0) * exp(-13163.0/Tgas);  }
        
    if (Tgas>2590.0 && Tgas<41000.0) { k[972] = 1.1e-10 * pow(Tgas/300.0,
        0.5) * exp(-77700.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[973] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<5000.0) { k[974] = 4.85e-12 * pow(Tgas/300.0,
        1.9) * exp(-1379.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<3000.0) { k[975] = 1.59e-11 * pow(Tgas/300.0,
        1.2) * exp(-9610.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[976] = 6.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-12500.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[977] = 1.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>1200.0 && Tgas<1812.0) { k[978] = 6.61e-11 * pow(Tgas/300.0,
        0.0) * exp(-51598.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[979] = 1.14e-13 * pow(Tgas/300.0,
        4.23) * exp(+114.6/Tgas);  }
        
    if (Tgas>550.0 && Tgas<3000.0) { k[980] = 1.05e-09 * pow(Tgas/300.0,
        -0.3) * exp(-14730.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<3000.0) { k[981] = 4.5e-11 * pow(Tgas/300.0,
        0.72) * exp(-329.0/Tgas);  }
        
    if (Tgas>350.0 && Tgas<3000.0) { k[982] = 2.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-9010.0/Tgas);  }
        
    if (Tgas>73.0 && Tgas<3000.0) { k[983] = 4.56e-12 * pow(Tgas/300.0,
        1.02) * exp(-2161.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[984] = 7.8e-13 * pow(Tgas/300.0, 2.4)
        * exp(-4990.0/Tgas);  }
        
    if (Tgas>80.0 && Tgas<300.0) { k[985] = 1.73e-11 * pow(Tgas/300.0, 0.5)
        * exp(-2400.0/Tgas);  }
        
    if (Tgas>24.0 && Tgas<300.0) { k[986] = 1.4e-10 * pow(Tgas/300.0, 0.0) *
        exp(-740.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<3000.0) { k[987] = 9.29e-10 * pow(Tgas/300.0,
        -0.1) * exp(-35220.0/Tgas);  }
        
    if (Tgas>1500.0 && Tgas<4524.0) { k[988] = 3.6e-10 * pow(Tgas/300.0,
        0.0) * exp(-24910.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<4000.0) { k[989] = 2.61e-10 * pow(Tgas/300.0,
        0.0) * exp(-8156.0/Tgas);  }
        
    if (Tgas>245.0 && Tgas<2500.0) { k[990] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-866.0/Tgas);  }
        
    if (Tgas>230.0 && Tgas<2500.0) { k[991] = 2.06e-11 * pow(Tgas/300.0,
        0.84) * exp(-277.0/Tgas);  }
        
    if (Tgas>230.0 && Tgas<2500.0) { k[992] = 1.66e-10 * pow(Tgas/300.0,
        0.0) * exp(-413.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<2500.0) { k[993] = 1.87e-11 * pow(Tgas/300.0,
        0.9) * exp(-2924.0/Tgas);  }
        
    if (Tgas>295.0 && Tgas<1490.0) { k[994] = 1.26e-10 * pow(Tgas/300.0,
        0.0) * exp(-515.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[995] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[996] = 6.99e-14 * pow(Tgas/300.0,
        2.8) * exp(-1950.0/Tgas);  }
        
    if (Tgas>303.0 && Tgas<376.0) { k[997] = 3.6e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[998] = 3e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[999] = 1e-12 * pow(Tgas/300.0, 0.0) *
        exp(-1000.0/Tgas);  }
        
    if (Tgas>295.0 && Tgas<2500.0) { k[1000] = 1.2e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    k[1001] = 7.6e-13 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1002] = 4.64e-12 * pow(Tgas/300.0,
        0.7) * exp(+25.6/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1003] = 5e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1004] = 1e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1005] = 3.95e-11 * pow(Tgas/300.0,
        0.17) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1006] = 3.95e-11 * pow(Tgas/300.0,
        0.17) * exp(-0.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<4000.0) { k[1007] = 9.96e-13 * pow(Tgas/300.0,
        0.0) * exp(-20380.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2000.0) { k[1008] = 7.4e-11 * pow(Tgas/300.0,
        0.26) * exp(-8.4/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1009] = 1.3e-11 * pow(Tgas/300.0, 0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>253.0 && Tgas<352.0) { k[1010] = 3.32e-13 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<299.0) { k[1011] = 1e-10 * pow(Tgas/300.0, 0.18) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>291.0 && Tgas<523.0) { k[1012] = 3.2e-13 * pow(Tgas/300.0, 0.0)
        * exp(-1710.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1013] = 1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-200.0/Tgas);  }
        
    if (Tgas>33.0 && Tgas<300.0) { k[1014] = 5.71e-12 * pow(Tgas/300.0, 0.5)
        * exp(-1000.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1015] = 1.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1016] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>33.0 && Tgas<300.0) { k[1017] = 2.94e-12 * pow(Tgas/300.0, 0.5)
        * exp(-1000.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1400.0) { k[1018] = 4.98e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    k[1019] = 2.41e-12 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>10.0 && Tgas<300.0) { k[1020] = 1e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1021] = 1e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>100.0 && Tgas<2500.0) { k[1022] = 3.38e-11 * pow(Tgas/300.0,
        -0.17) * exp(+2.8/Tgas);  }
        
    if (Tgas>200.0 && Tgas<14000.0) { k[1023] = 2.26e-12 * pow(Tgas/300.0,
        0.86) * exp(-3134.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1024] = 1.7e-13 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>5.0 && Tgas<2500.0) { k[1025] = 6.05e-11 * pow(Tgas/300.0,
        -0.23) * exp(-14.9/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1026] = 1.88e-11 * pow(Tgas/300.0,
        0.1) * exp(-10700.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1027] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2500.0) { k[1028] = 2.9e-13 * pow(Tgas/300.0,
        2.87) * exp(-5380.0/Tgas);  }
        
    if (Tgas>210.0 && Tgas<3000.0) { k[1029] = 4.27e-11 * pow(Tgas/300.0,
        -2.5) * exp(-331.0/Tgas);  }
        
    if (Tgas>1500.0 && Tgas<2150.0) { k[1030] = 1.49e-12 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<3000.0) { k[1031] = 1.35e-12 * pow(Tgas/300.0,
        1.25) * exp(+43.5/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2500.0) { k[1032] = 2.08e-13 * pow(Tgas/300.0,
        0.76) * exp(-262.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1033] = 2.75e-11 * pow(Tgas/300.0,
        -1.14) * exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2000.0) { k[1034] = 6.63e-16 * pow(Tgas/300.0,
        6.13) * exp(-5895.0/Tgas);  }
        
    if (Tgas>33.0 && Tgas<300.0) { k[1035] = 2.94e-12 * pow(Tgas/300.0, 0.5)
        * exp(-1000.0/Tgas);  }
        
    if (Tgas>600.0 && Tgas<3000.0) { k[1036] = 1.83e-12 * pow(Tgas/300.0,
        1.6) * exp(-14090.0/Tgas);  }
        
    if (Tgas>1300.0 && Tgas<1700.0) { k[1037] = 5.25e-10 * pow(Tgas/300.0,
        0.0) * exp(-13470.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1038] = 1.7e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    k[1039] = 1.16e-09 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>300.0 && Tgas<3000.0) { k[1040] = 1.81e-13 * pow(Tgas/300.0,
        1.8) * exp(+70.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1041] = 2.44e-11 * pow(Tgas/300.0,
        -1.94) * exp(-56.9/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3300.0) { k[1042] = 7.4e-10 * pow(Tgas/300.0,
        0.0) * exp(-10540.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3300.0) { k[1043] = 1.33e-11 * pow(Tgas/300.0,
        -0.78) * exp(-40.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<3500.0) { k[1044] = 6.88e-14 * pow(Tgas/300.0,
        2.07) * exp(-3281.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3300.0) { k[1045] = 2.54e-14 * pow(Tgas/300.0,
        1.18) * exp(-312.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<299.0) { k[1046] = 6.6e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<3000.0) { k[1047] = 1.16e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3000.0) { k[1048] = 3.11e-12 * pow(Tgas/300.0,
        1.2) * exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3000.0) { k[1049] = 3.32e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3000.0) { k[1050] = 2.93e-12 * pow(Tgas/300.0,
        0.1) * exp(-5800.0/Tgas);  }
        
    if (Tgas>1573.0 && Tgas<4700.0) { k[1051] = 2.51e-11 * pow(Tgas/300.0,
        0.0) * exp(-30653.0/Tgas);  }
        
    if (Tgas>780.0 && Tgas<41000.0) { k[1052] = 2.8e-12 * pow(Tgas/300.0,
        0.0) * exp(-23400.0/Tgas);  }
        
    if (Tgas>290.0 && Tgas<2660.0) { k[1053] = 4.55e-11 * pow(Tgas/300.0,
        -1.33) * exp(-242.0/Tgas);  }
        
    k[1054] = 1.32e-12 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>25.0 && Tgas<300.0) { k[1055] = 8.1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-773.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2500.0) { k[1056] = 2.29e-12 * pow(Tgas/300.0,
        2.2) * exp(-3820.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<294.0) { k[1057] = 2.54e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<5000.0) { k[1058] = 5.37e-11 * pow(Tgas/300.0,
        0.0) * exp(-13800.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<6000.0) { k[1059] = 2.46e-11 * pow(Tgas/300.0,
        0.0) * exp(-26567.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1060] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<2500.0) { k[1061] = 1.07e-11 * pow(Tgas/300.0,
        1.17) * exp(-1242.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1062] = 1.85e-11 * pow(Tgas/300.0,
        0.95) * exp(-8571.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<4000.0) { k[1063] = 6.21e-10 * pow(Tgas/300.0,
        0.0) * exp(-12439.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2600.0) { k[1064] = 7.3e-13 * pow(Tgas/300.0,
        1.14) * exp(-3742.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2600.0) { k[1065] = 1.36e-12 * pow(Tgas/300.0,
        1.38) * exp(-3693.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1066] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1067] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1068] = 1e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1069] = 3.8e-11 * pow(Tgas/300.0,
        -0.08) * exp(-0.0/Tgas);  }
        
    if (Tgas>116.0 && Tgas<300.0) { k[1070] = 2.94e-12 * pow(Tgas/300.0,
        0.5) * exp(-3500.0/Tgas);  }
        
    if (Tgas>1400.0 && Tgas<4700.0) { k[1071] = 2.51e-10 * pow(Tgas/300.0,
        0.0) * exp(-38602.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<3000.0) { k[1072] = 6.3e-11 * pow(Tgas/300.0,
        -0.1) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<3000.0) { k[1073] = 7e-12 * pow(Tgas/300.0, -0.1)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2500.0) { k[1074] = 1.89e-11 * pow(Tgas/300.0,
        0.0) * exp(-4003.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1075] = 9.82e-12 * pow(Tgas/300.0,
        -0.21) * exp(-5.2/Tgas);  }
        
    if (Tgas>200.0 && Tgas<5000.0) { k[1076] = 1.18e-11 * pow(Tgas/300.0,
        0.0) * exp(-20413.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1077] = 5.76e-11 * pow(Tgas/300.0,
        -0.3) * exp(-7.5/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1078] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<2500.0) { k[1079] = 4.02e-10 * pow(Tgas/300.0,
        -1.43) * exp(-3501.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<500.0) { k[1080] = 3.69e-11 * pow(Tgas/300.0,
        -0.27) * exp(-12.9/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1081] = 4e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1082] = 4e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1083] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1084] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1085] = 8e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1086] = 1.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1087] = 1.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>280.0 && Tgas<1000.0) { k[1088] = 1.98e-11 * pow(Tgas/300.0,
        0.0) * exp(-1183.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1089] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<2500.0) { k[1090] = 1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-1000.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<2500.0) { k[1091] = 7e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>80.0 && Tgas<3150.0) { k[1092] = 2.81e-13 * pow(Tgas/300.0,
        0.0) * exp(-176.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<3000.0) { k[1093] = 7.76e-12 * pow(Tgas/300.0,
        0.82) * exp(+30.6/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2840.0) { k[1094] = 1.87e-13 * pow(Tgas/300.0,
        1.5) * exp(-3887.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<2500.0) { k[1095] = 1.07e-13 * pow(Tgas/300.0,
        0.0) * exp(-5892.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1096] = 1.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<4000.0) { k[1097] = 6.17e-12 * pow(Tgas/300.0,
        1.23) * exp(+44.3/Tgas);  }
        
    if (Tgas>200.0 && Tgas<3000.0) { k[1098] = 1.47e-13 * pow(Tgas/300.0,
        2.05) * exp(-7.0/Tgas);  }
        
    if (Tgas>503.0 && Tgas<41000.0) { k[1099] = 5.2e-12 * pow(Tgas/300.0,
        0.0) * exp(-15100.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1100] = 8.58e-11 * pow(Tgas/300.0,
        -0.56) * exp(-14.8/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1101] = 1.65e-12 * pow(Tgas/300.0,
        1.14) * exp(-50.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<3560.0) { k[1102] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<3160.0) { k[1103] = 2.72e-11 * pow(Tgas/300.0,
        0.0) * exp(-282.0/Tgas);  }
        
    if (Tgas>2720.0 && Tgas<5190.0) { k[1104] = 1.3e-09 * pow(Tgas/300.0,
        0.0) * exp(-34513.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1105] = 9e-11 * pow(Tgas/300.0, -0.96)
        * exp(-28.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1106] = 1.72e-10 * pow(Tgas/300.0,
        -0.53) * exp(-17.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1107] = G0 * 3.1e-10 * exp(-3.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1108] = G0 * 3.3e-10 * exp(-2.9*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1109] = G0 * 4.67e-11 * exp(-2.2*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1110] = G0 * 4.67e-11 * exp(-2.2*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1111] = G0 * 4.67e-11 * exp(-2.2*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1112] = G0 * 1e-09 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1113] = G0 * 5.8e-10 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1114] = G0 * 1e-09 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1115] = G0 * 1e-09 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1116] = G0 * 1.35e-10 * exp(-2.3*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1117] = G0 * 1e-10 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1118] = G0 * 1.35e-10 * exp(-2.3*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1119] = G0 * 7e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1120] = G0 * 1.3e-10 * exp(-2.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1121] = G0 * 7e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1122] = G0 * 2.27e-10 * exp(-2.7*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1123] = G0 * 5.33e-11 * exp(-2.7*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1124] = G0 * 9.8e-10 * exp(-2.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1125] = G0 * 2.2e-10 * exp(-2.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1126] = G0 * 6.8e-12 * exp(-3.9*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1127] = G0 * 2.2e-10 * exp(-2.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1128] = G0 * 9.2e-10 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1129] = G0 * 7.6e-10 * exp(-3.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1130] = G0 * 2.9e-10 * exp(-3.5*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1131] = G0 * 1e-10 * exp(-2.5*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1132] = G0 * 8.9e-10 * exp(-3.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1133] = (2.0e-10) * G0 *
        GetShieldingFactor(IDX_COI, h2col, cocol, Tgas, 1) *
        GetGrainScattering(Av, lamdabar) / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1134] = G0 * 5.7e-10 * exp(-2.4*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1135] = G0 * 5.48e-10 * exp(-2.0*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1136] = G0 * 1e-09 * exp(-2.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1137] = G0 * 7e-10 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1138] = G0 * 4.7e-10 * exp(-2.8*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1139] = G0 * 1.4e-11 * exp(-3.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1140] = G0 * 1e-12 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1141] = G0 * 3.1e-11 * exp(-3.9*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1142] = G0 * 8e-10 * exp(-2.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1143] = G0 * 4.4e-10 * exp(-1.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1144] = G0 * 4.4e-10 * exp(-1.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1145] = G0 * 5e-15 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1146] = G0 * 5e-15 * exp(-1.8*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1147] = G0 * 1.6e-09 * exp(-2.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1148] = G0 * 5.4e-12 * exp(-3.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1149] = G0 * 1.1e-09 * exp(-1.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1150] = G0 * 5.6e-10 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1151] = G0 * 1.5e-09 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1152] = G0 * 1e-09 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1153] = G0 * 1.7e-10 * exp(-0.5*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1154] = G0 * 7.9e-11 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1155] = G0 * 2.3e-10 * exp(-3.9*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1156] = G0 * 5.4e-11 * exp(-1.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1157] = G0 * 1.73e-10 * exp(-2.6*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1158] = G0 * 7.5e-10 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1159] = G0 * 9.23e-10 * exp(-2.1*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1160] = G0 * 2.8e-10 * exp(-3.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1161] = G0 * 2.76e-10 * exp(-2.1*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1162] = G0 * 5e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1163] = G0 * 1e-11 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1164] = G0 * 1.4e-09 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1165] = G0 * 2.6e-10 * exp(-2.9*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1166] = G0 * 4.7e-10 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1167] = G0 * 3.5e-11 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1168] = G0 * 7.6e-11 * exp(-3.9*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1169] = G0 * 7.9e-10 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1170] = G0 * 3.35e-10 * exp(-2.1*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1171] = G0 * 3.35e-10 * exp(-2.1*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1172] = G0 * 1e-11 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1173] = G0 * 1.1e-11 * exp(-3.5*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1174] = G0 * 3.9e-10 * exp(-2.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1175] = G0 * 1.6e-12 * exp(-3.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1176] = G0 * 3.1e-09 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1177] = G0 * 2e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1178] = G0 * 1e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1179] = G0 * 2.7e-09 * exp(-1.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1180] = G0 * 1e-09 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1181] = G0 * 5e-11 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1182] = G0 * 3e-11 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1183] = G0 * 1e-10 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1184] = G0 * 3e-11 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1185] = G0 * 4.8e-10 * exp(-2.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1186] = G0 * 1.6e-10 * exp(-2.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1187] = G0 * 1.6e-10 * exp(-2.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1188] = G0 * 2.8e-09 * exp(-1.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1189] = G0 * 1e-10 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1190] = G0 * 1.6e-09 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1191] = G0 * 2.4e-10 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>300.0 && Tgas<14700.0) { k[1192] = 1.08e-18 * pow(Tgas/300.0,
        0.07) * exp(-57.5/Tgas);  }
        
    if (Tgas>10.0 && Tgas<13900.0) { k[1193] = 3.14e-18 * pow(Tgas/300.0,
        -0.15) * exp(-68.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<14700.0) { k[1194] = 5.72e-19 * pow(Tgas/300.0,
        0.37) * exp(-51.0/Tgas);  }
        
    if (Tgas>2000.0 && Tgas<10000.0) { k[1195] = 5e-10 * pow(Tgas/300.0,
        -3.7) * exp(-800.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<14700.0) { k[1196] = 4.69e-19 * pow(Tgas/300.0,
        1.52) * exp(+50.5/Tgas);  }
        
    if (Tgas>200.0 && Tgas<32000.0) { k[1197] = 1.15e-18 * pow(Tgas/300.0,
        1.49) * exp(-228.0/Tgas);  }
        
    if (Tgas>16.0 && Tgas<100.0) { k[1198] = 5.26e-20 * pow(Tgas/300.0,
        -0.51) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1199] = 2e-16 * pow(Tgas/300.0, -1.3) *
        exp(-23.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1200] = 1e-17 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1201] = 5.09e-18 * pow(Tgas/300.0,
        -0.71) * exp(-11.6/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1202] = 3e-18 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1203] = 3e-17 * pow(Tgas/300.0, -1.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1204] = 1e-18 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1205] = 1.7e-17 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1206] = 1e-17 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1207] = 9.9e-19 * pow(Tgas/300.0,
        -0.38) * exp(-0.0/Tgas);  }
        
    if (Tgas>20.0 && Tgas<300.0) { k[1208] = 5.26e-18 * pow(Tgas/300.0,
        -5.22) * exp(-90.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<10000.0) { k[1209] = 1.17e-17 * pow(Tgas/300.0,
        -0.14) * exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<14700.0) { k[1210] = 3.71e-18 * pow(Tgas/300.0,
        0.24) * exp(-26.1/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1211] = 4.9e-20 * pow(Tgas/300.0, 1.58)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<14700.0) { k[1212] = 9.22e-19 * pow(Tgas/300.0,
        -0.08) * exp(+21.2/Tgas);  }
        
    if (Tgas>300.0 && Tgas<14700.0) { k[1213] = 3.23e-17 * pow(Tgas/300.0,
        0.31) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1214] = 2.36e-12 * pow(Tgas/300.0,
        -0.29) * exp(+17.6/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1215] = 1.1e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<20000.0) { k[1216] = 3.5e-12 * pow(Tgas/300.0,
        -0.75) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1217] = 1.1e-10 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[1218] = 5.36e-12 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[1219] = 2.78e-12 * pow(Tgas/300.0,
        -0.68) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1220] = 3.5e-12 * pow(Tgas/300.0,
        -0.53) * exp(+3.2/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1221] = 3.24e-12 * pow(Tgas/300.0,
        -0.66) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[1222] = 4.26e-12 * pow(Tgas/300.0,
        -0.62) * exp(-0.0/Tgas);  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1223] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1224] = 4.57e4 * 1.0 * sqrt(Tgas / 43.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1225] = 4.57e4 * 1.0 * sqrt(Tgas / 42.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1226] = 4.57e4 * 1.0 * sqrt(Tgas / 29.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1227] = 4.57e4 * 0.0 * sqrt(Tgas / 28.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1228] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1229] = 4.57e4 * 1.0 * sqrt(Tgas / 44.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1230] = 4.57e4 * 1.0 * sqrt(Tgas / 29.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1231] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1232] = 4.57e4 * 1.0 * sqrt(Tgas / 29.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1233] = 4.57e4 * 1.0 * sqrt(Tgas / 30.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1234] = 4.57e4 * 1.0 * sqrt(Tgas / 30.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1235] = 4.57e4 * 1.0 * sqrt(Tgas / 31.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1236] = 4.57e4 * 1.0 * sqrt(Tgas / 31.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1237] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1238] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1239] = 4.57e4 * 1.0 * sqrt(Tgas / 40.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1240] = 4.57e4 * 1.0 * sqrt(Tgas / 52.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1241] = 4.57e4 * 1.0 * sqrt(Tgas / 40.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1242] = 4.57e4 * 1.0 * sqrt(Tgas / 52.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1243] = 4.57e4 * 1.0 * sqrt(Tgas / 64.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1244] = 4.57e4 * 1.0 * sqrt(Tgas / 64.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1245] = 4.57e4 * 1.0 * sqrt(Tgas / 44.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1246] = 4.57e4 * 1.0 * sqrt(Tgas / 45.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1247] = 4.57e4 * 1.0 * sqrt(Tgas / 46.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1248] = 4.57e4 * 1.0 * sqrt(Tgas / 33.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1249] = 4.57e4 * 1.0 * sqrt(Tgas / 31.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1250] = 4.57e4 * 1.0 * sqrt(Tgas / 12.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1251] = 4.57e4 * 0.9 * sqrt(Tgas / 28.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1252] = 4.57e4 * 1.0 * sqrt(Tgas / 30.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1253] = 4.57e4 * 1.0 * sqrt(Tgas / 13.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1254] = 4.57e4 * 1.0 * sqrt(Tgas / 17.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1255] = 4.57e4 * 1.0 * sqrt(Tgas / 30.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1256] = 4.57e4 * 1.0 * sqrt(Tgas / 14.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1257] = 4.57e4 * 1.0 * sqrt(Tgas / 18.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1258] = 4.57e4 * 1.0 * sqrt(Tgas / 44.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1259] = 4.57e4 * 1.0 * sqrt(Tgas / 15.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1260] = 4.57e4 * 1.0 * sqrt(Tgas / 16.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1261] = 4.57e4 * 1.0 * sqrt(Tgas / 29.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1262] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1263] = 4.57e4 * 1.0 * sqrt(Tgas / 26.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1264] = 4.57e4 * 1.0 * sqrt(Tgas / 12.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1265] = 4.57e4 * 1.0 * sqrt(Tgas / 15.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1266] = 4.57e4 * 1.0 * sqrt(Tgas / 27.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1267] = 4.57e4 * 1.0 * sqrt(Tgas / 17.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1268] = 4.57e4 * 1.0 * sqrt(Tgas / 14.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1269] = 4.57e4 * 1.0 * sqrt(Tgas / 16.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1270] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1271] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1272] = 4.57e4 * 1.0 * sqrt(Tgas / 13.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1273] = 4.57e4 * 1.0 * sqrt(Tgas / 15.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1274] = 4.57e4 * 1.0 * sqrt(Tgas / 17.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1275] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1276] = 4.57e4 * 1.0 * sqrt(Tgas / 26.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1277] = 4.57e4 * 1.0 * sqrt(Tgas / 30.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1278] = 4.57e4 * 1.0 * sqrt(Tgas / 14.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1279] = 4.57e4 * 1.0 * sqrt(Tgas / 16.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1280] = 4.57e4 * 1.0 * sqrt(Tgas / 18.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1281] = 4.57e4 * 1.0 * sqrt(Tgas / 29.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1282] = 4.57e4 * 1.0 * sqrt(Tgas / 27.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1283] = 4.57e4 * 1.0 * sqrt(Tgas / 17.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1284] = 4.57e4 * 1.0 * sqrt(Tgas / 30.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1285] = 4.57e4 * 1.0 * sqrt(Tgas / 15.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1286] = 4.57e4 * 1.0 * sqrt(Tgas / 19.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1287] = 4.57e4 * 1.0 * sqrt(Tgas / 45.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1288] = 4.57e4 * 1.0 * sqrt(Tgas / 16.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1289] = 4.57e4 * 1.0 * sqrt(Tgas / 16.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1290] = 4.57e4 * 1.0 * sqrt(Tgas / 14.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1291] = 4.57e4 * 1.0 * sqrt(Tgas / 16.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1292] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1293] = 4.57e4 * 1.0 * sqrt(Tgas / 46.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1294] = 4.57e4 * 1.0 * sqrt(Tgas / 31.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1295] = 4.57e4 * 1.0 * sqrt(Tgas / 31.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1296] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1297] = 4.57e4 * 1.0 * sqrt(Tgas / 33.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1298] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1299] = 4.57e4 * 1.0 * sqrt(Tgas / 24.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1300] = 4.57e4 * 1.0 * sqrt(Tgas / 24.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1301] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1302] = 4.57e4 * 1.0 * sqrt(Tgas / 29.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1303] = 4.57e4 * 1.0 * sqrt(Tgas / 33.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1304] = 4.57e4 * 0.1 * sqrt(Tgas / 28.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1305] = 4.57e4 * 1.0 * sqrt(Tgas / 27.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[1306] = 4.57e4 * 1.0 * garea * fr * ( 1.0
        + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1307] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1308] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1309] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1310] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GCH4I/(pi*pi*amu*16.0)) * 2.0 * densites *
        exp(-eb_GCH4I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1311] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1312] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1313] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1314] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GNH3I/(pi*pi*amu*17.0)) * 2.0 * densites *
        exp(-eb_GNH3I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1315] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1316] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1317] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1318] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GH2OI/(pi*pi*amu*18.0)) * 2.0 * densites *
        exp(-eb_GH2OI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1319] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1320] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1321] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1322] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GMgI/(pi*pi*amu*24.0)) * 2.0 * densites *
        exp(-eb_GMgI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1323] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1324] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1325] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1326] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GHCNI/(pi*pi*amu*27.0)) * 2.0 * densites *
        exp(-eb_GHCNI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1327] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1328] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1329] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1330] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GHNCI/(pi*pi*amu*27.0)) * 2.0 * densites *
        exp(-eb_GHNCI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1331] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1332] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1333] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1334] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GCOI/(pi*pi*amu*28.0)) * 2.0 * densites *
        exp(-eb_GCOI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1335] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1336] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1337] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1338] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GN2I/(pi*pi*amu*28.0)) * 2.0 * densites *
        exp(-eb_GN2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1339] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1340] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1341] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1342] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GH2CNI/(pi*pi*amu*28.0)) * 2.0 * densites *
        exp(-eb_GH2CNI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1343] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1344] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1345] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1346] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GNOI/(pi*pi*amu*30.0)) * 2.0 * densites *
        exp(-eb_GNOI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1347] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1348] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1349] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1350] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GH2COI/(pi*pi*amu*30.0)) * 2.0 * densites *
        exp(-eb_GH2COI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1351] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1352] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1353] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1354] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GHNOI/(pi*pi*amu*31.0)) * 2.0 * densites *
        exp(-eb_GHNOI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1355] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1356] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1357] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1358] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GO2I/(pi*pi*amu*32.0)) * 2.0 * densites *
        exp(-eb_GO2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1359] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1360] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1361] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1362] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GCH3OHI/(pi*pi*amu*32.0)) * 2.0 * densites *
        exp(-eb_GCH3OHI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1363] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1364] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1365] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1366] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GSiH4I/(pi*pi*amu*32.0)) * 2.0 * densites *
        exp(-eb_GSiH4I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1367] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1368] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1369] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1370] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GO2HI/(pi*pi*amu*33.0)) * 2.0 * densites *
        exp(-eb_GO2HI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1371] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1372] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1373] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1374] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GSiCI/(pi*pi*amu*40.0)) * 2.0 * densites *
        exp(-eb_GSiCI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1375] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1376] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1377] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1378] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GHNCOI/(pi*pi*amu*43.0)) * 2.0 * densites *
        exp(-eb_GHNCOI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1379] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1380] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1381] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1382] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GSiOI/(pi*pi*amu*44.0)) * 2.0 * densites *
        exp(-eb_GSiOI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1383] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1384] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1385] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1386] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GCO2I/(pi*pi*amu*44.0)) * 2.0 * densites *
        exp(-eb_GCO2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1387] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1388] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1389] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1390] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GNO2I/(pi*pi*amu*46.0)) * 2.0 * densites *
        exp(-eb_GNO2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1391] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1392] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1393] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1394] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GH2SiOI/(pi*pi*amu*46.0)) * 2.0 * densites *
        exp(-eb_GH2SiOI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1395] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1396] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1397] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1398] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GSiC2I/(pi*pi*amu*52.0)) * 2.0 * densites *
        exp(-eb_GSiC2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1399] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1400] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1401] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[1402] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GSiC3I/(pi*pi*amu*64.0)) * 2.0 * densites *
        exp(-eb_GSiC3I/Tgas)) : 0.0;  }
        
    
        // clang-format on

    return NAUNET_SUCCESS;
}

// clang-format off
int EvalHeatingRates(realtype *kh, realtype *y, NaunetData *u_data) {

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

    // Some variable definitions from krome
    realtype Te      = Tgas * 8.617343e-5;            // Tgas in eV (eV)
    realtype lnTe    = log(Te);                       // ln of Te (#)
    realtype T32     = Tgas * 0.0033333333333333335;  // Tgas/(300 K) (#)
    realtype invT    = 1.0 / Tgas;                    // inverse of T (1/K)
    realtype invTe   = 1.0 / Te;                      // inverse of T (1/eV)
    realtype sqrTgas = sqrt(Tgas);  // Tgas rootsquare (K**0.5)

    // reaaction rate (k) of each reaction
    // clang-format off
    
    // clang-format on

    return NAUNET_SUCCESS;
}

// clang-format off
int EvalCoolingRates(realtype *kc, realtype *y, NaunetData *u_data) {

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

    // Some variable definitions from krome
    realtype Te      = Tgas * 8.617343e-5;            // Tgas in eV (eV)
    realtype lnTe    = log(Te);                       // ln of Te (#)
    realtype T32     = Tgas * 0.0033333333333333335;  // Tgas/(300 K) (#)
    realtype invT    = 1.0 / Tgas;                    // inverse of T (1/K)
    realtype invTe   = 1.0 / Te;                      // inverse of T (1/eV)
    realtype sqrTgas = sqrt(Tgas);  // Tgas rootsquare (K**0.5)

    // reaaction rate (k) of each reaction
    // clang-format off
    
    // clang-format on

    return NAUNET_SUCCESS;
}
