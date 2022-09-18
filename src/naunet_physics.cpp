// 
#include <math.h>
#include <stdio.h>

#include <algorithm>

#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_physics.h"
#include "naunet_utilities.h"

// clang-format off
double GetElementAbund(double *y, int elemidx) {
    if (elemidx == IDX_ELEM_MG) {
        return 1.0*y[IDX_GMgI] + 1.0*y[IDX_MgI] + 1.0*y[IDX_MgII] + 0.0;
    }
    if (elemidx == IDX_ELEM_SI) {
        return 1.0*y[IDX_GSiOI] + 1.0*y[IDX_GSiCI] + 1.0*y[IDX_GSiC2I] + 1.0*y[IDX_GSiC3I] + 
               1.0*y[IDX_GH2SiOI] + 1.0*y[IDX_SiC3II] + 1.0*y[IDX_H2SiOI] + 1.0*y[IDX_SiC2II] + 
               1.0*y[IDX_GSiH4I] + 1.0*y[IDX_SiC2I] + 1.0*y[IDX_SiC3I] + 1.0*y[IDX_SiH5II] + 
               1.0*y[IDX_SiH4II] + 1.0*y[IDX_SiCII] + 1.0*y[IDX_SiCI] + 1.0*y[IDX_SiH3II] + 
               1.0*y[IDX_SiH2II] + 1.0*y[IDX_SiH2I] + 1.0*y[IDX_SiOHII] + 1.0*y[IDX_SiHII] + 
               1.0*y[IDX_SiH4I] + 1.0*y[IDX_SiHI] + 1.0*y[IDX_SiH3I] + 1.0*y[IDX_SiOII] + 
               1.0*y[IDX_SiOI] + 1.0*y[IDX_SiII] + 1.0*y[IDX_SiI] + 0.0;
    }
    if (elemidx == IDX_ELEM_HE) {
        return 1.0*y[IDX_HeHII] + 1.0*y[IDX_HeII] + 1.0*y[IDX_HeI] + 0.0;
    }
    if (elemidx == IDX_ELEM_N) {
        return 1.0*y[IDX_GH2CNI] + 1.0*y[IDX_GHNCI] + 1.0*y[IDX_GNO2I] + 1.0*y[IDX_GHNCOI] + 
               1.0*y[IDX_GNOI] + 1.0*y[IDX_GHNOI] + 2.0*y[IDX_GN2I] + 1.0*y[IDX_GHCNI] + 
               1.0*y[IDX_GNH3I] + 1.0*y[IDX_H2CNI] + 1.0*y[IDX_H2NOII] + 1.0*y[IDX_HNCOI] + 
               1.0*y[IDX_NO2I] + 1.0*y[IDX_OCNI] + 1.0*y[IDX_HNOI] + 1.0*y[IDX_CNII] + 
               1.0*y[IDX_HCNHII] + 2.0*y[IDX_N2HII] + 1.0*y[IDX_HNCI] + 1.0*y[IDX_HNOII] + 
               2.0*y[IDX_N2II] + 1.0*y[IDX_NH3I] + 1.0*y[IDX_NII] + 1.0*y[IDX_HCNII] + 
               1.0*y[IDX_NH2II] + 1.0*y[IDX_NHII] + 1.0*y[IDX_NH2I] + 1.0*y[IDX_NH3II] + 
               1.0*y[IDX_NOII] + 2.0*y[IDX_N2I] + 1.0*y[IDX_HCNI] + 1.0*y[IDX_NHI] + 
               1.0*y[IDX_CNI] + 1.0*y[IDX_NOI] + 1.0*y[IDX_NI] + 0.0;
    }
    if (elemidx == IDX_ELEM_C) {
        return 1.0*y[IDX_GH2CNI] + 1.0*y[IDX_GHNCI] + 1.0*y[IDX_GCOI] + 1.0*y[IDX_GHNCOI] + 
               1.0*y[IDX_GSiCI] + 2.0*y[IDX_GSiC2I] + 3.0*y[IDX_GSiC3I] + 1.0*y[IDX_GCH3OHI] + 
               1.0*y[IDX_GCO2I] + 1.0*y[IDX_GH2COI] + 1.0*y[IDX_GHCNI] + 3.0*y[IDX_SiC3II] + 
               1.0*y[IDX_H2CNI] + 1.0*y[IDX_GCH4I] + 1.0*y[IDX_HNCOI] + 1.0*y[IDX_HOCII] + 
               2.0*y[IDX_SiC2II] + 2.0*y[IDX_SiC2I] + 3.0*y[IDX_SiC3I] + 1.0*y[IDX_SiCII] + 
               1.0*y[IDX_SiCI] + 1.0*y[IDX_OCNI] + 1.0*y[IDX_HCO2II] + 1.0*y[IDX_CH3OHI] + 
               1.0*y[IDX_CH4II] + 1.0*y[IDX_CNII] + 1.0*y[IDX_HCNHII] + 1.0*y[IDX_HNCI] + 
               1.0*y[IDX_H3COII] + 1.0*y[IDX_CH4I] + 1.0*y[IDX_COII] + 1.0*y[IDX_CH3I] + 
               1.0*y[IDX_CO2I] + 1.0*y[IDX_HCNII] + 1.0*y[IDX_CH3II] + 1.0*y[IDX_CH2II] + 
               1.0*y[IDX_CII] + 1.0*y[IDX_HCNI] + 1.0*y[IDX_CHII] + 1.0*y[IDX_CH2I] + 
               1.0*y[IDX_H2COII] + 1.0*y[IDX_CNI] + 1.0*y[IDX_H2COI] + 1.0*y[IDX_HCOI] + 
               1.0*y[IDX_CHI] + 1.0*y[IDX_CI] + 1.0*y[IDX_HCOII] + 1.0*y[IDX_COI] + 0.0;
    }
    if (elemidx == IDX_ELEM_O) {
        return 2.0*y[IDX_GNO2I] + 1.0*y[IDX_GSiOI] + 1.0*y[IDX_GCOI] + 1.0*y[IDX_GHNCOI] + 
               1.0*y[IDX_GNOI] + 2.0*y[IDX_GO2I] + 2.0*y[IDX_GO2HI] + 1.0*y[IDX_GCH3OHI] + 
               2.0*y[IDX_GCO2I] + 1.0*y[IDX_GH2SiOI] + 1.0*y[IDX_GHNOI] + 1.0*y[IDX_GH2COI] + 
               1.0*y[IDX_GH2OI] + 1.0*y[IDX_H2NOII] + 1.0*y[IDX_H2SiOI] + 1.0*y[IDX_HNCOI] + 
               1.0*y[IDX_HOCII] + 2.0*y[IDX_O2HI] + 2.0*y[IDX_NO2I] + 1.0*y[IDX_OCNI] + 
               1.0*y[IDX_SiOHII] + 1.0*y[IDX_SiOII] + 2.0*y[IDX_HCO2II] + 1.0*y[IDX_HNOI] + 
               1.0*y[IDX_CH3OHI] + 1.0*y[IDX_SiOI] + 2.0*y[IDX_O2HII] + 1.0*y[IDX_HNOII] + 
               1.0*y[IDX_H3COII] + 1.0*y[IDX_COII] + 2.0*y[IDX_CO2I] + 1.0*y[IDX_OII] + 
               2.0*y[IDX_O2II] + 1.0*y[IDX_H2OII] + 1.0*y[IDX_NOII] + 1.0*y[IDX_H3OII] + 
               1.0*y[IDX_H2COII] + 1.0*y[IDX_OHII] + 1.0*y[IDX_H2COI] + 1.0*y[IDX_HCOI] + 
               1.0*y[IDX_NOI] + 1.0*y[IDX_OHI] + 2.0*y[IDX_O2I] + 1.0*y[IDX_HCOII] + 
               1.0*y[IDX_H2OI] + 1.0*y[IDX_OI] + 1.0*y[IDX_COI] + 0.0;
    }
    if (elemidx == IDX_ELEM_H) {
        return 2.0*y[IDX_GH2CNI] + 1.0*y[IDX_GHNCI] + 1.0*y[IDX_GHNCOI] + 1.0*y[IDX_GO2HI] + 
               4.0*y[IDX_GCH3OHI] + 2.0*y[IDX_GH2SiOI] + 1.0*y[IDX_GHNOI] + 2.0*y[IDX_GH2COI] + 
               1.0*y[IDX_GHCNI] + 2.0*y[IDX_GH2OI] + 3.0*y[IDX_GNH3I] + 2.0*y[IDX_H2CNI] + 
               4.0*y[IDX_GCH4I] + 2.0*y[IDX_H2NOII] + 2.0*y[IDX_H2SiOI] + 1.0*y[IDX_HeHII] + 
               1.0*y[IDX_HNCOI] + 1.0*y[IDX_HOCII] + 4.0*y[IDX_GSiH4I] + 5.0*y[IDX_SiH5II] + 
               4.0*y[IDX_SiH4II] + 1.0*y[IDX_O2HI] + 3.0*y[IDX_SiH3II] + 2.0*y[IDX_SiH2II] + 
               2.0*y[IDX_SiH2I] + 1.0*y[IDX_SiOHII] + 1.0*y[IDX_SiHII] + 4.0*y[IDX_SiH4I] + 
               1.0*y[IDX_SiHI] + 3.0*y[IDX_SiH3I] + 1.0*y[IDX_HCO2II] + 1.0*y[IDX_HNOI] + 
               4.0*y[IDX_CH3OHI] + 4.0*y[IDX_CH4II] + 2.0*y[IDX_HCNHII] + 1.0*y[IDX_N2HII] + 
               1.0*y[IDX_O2HII] + 1.0*y[IDX_HNCI] + 1.0*y[IDX_HNOII] + 3.0*y[IDX_H3COII] + 
               4.0*y[IDX_CH4I] + 2.0*y[IDX_H2II] + 3.0*y[IDX_NH3I] + 3.0*y[IDX_CH3I] + 
               1.0*y[IDX_HCNII] + 2.0*y[IDX_NH2II] + 1.0*y[IDX_NHII] + 3.0*y[IDX_CH3II] + 
               2.0*y[IDX_NH2I] + 2.0*y[IDX_CH2II] + 2.0*y[IDX_H2OII] + 3.0*y[IDX_NH3II] + 
               3.0*y[IDX_H3OII] + 1.0*y[IDX_HCNI] + 1.0*y[IDX_CHII] + 2.0*y[IDX_CH2I] + 
               2.0*y[IDX_H2COII] + 1.0*y[IDX_NHI] + 1.0*y[IDX_OHII] + 2.0*y[IDX_H2COI] + 
               1.0*y[IDX_HCOI] + 1.0*y[IDX_CHI] + 3.0*y[IDX_H3II] + 1.0*y[IDX_OHI] + 
               1.0*y[IDX_HII] + 1.0*y[IDX_HCOII] + 2.0*y[IDX_H2OI] + 2.0*y[IDX_H2I] + 
               1.0*y[IDX_HI] + 0.0;
    }
    
}

double GetMantleDens(double *y) {
    return y[IDX_GH2CNI] + y[IDX_GHNCI] + y[IDX_GNO2I] + y[IDX_GSiOI] + y[IDX_GCOI]
        + y[IDX_GHNCOI] + y[IDX_GMgI] + y[IDX_GNOI] + y[IDX_GO2I] + y[IDX_GO2HI]
        + y[IDX_GSiCI] + y[IDX_GSiC2I] + y[IDX_GSiC3I] + y[IDX_GCH3OHI] +
        y[IDX_GCO2I] + y[IDX_GH2SiOI] + y[IDX_GHNOI] + y[IDX_GN2I] +
        y[IDX_GH2COI] + y[IDX_GHCNI] + y[IDX_GH2OI] + y[IDX_GNH3I] +
        y[IDX_GCH4I] + y[IDX_GSiH4I] + 0.0;
}

double GetHNuclei(double *y) {
#ifdef IDX_ELEM_H
    return GetElementAbund(y, IDX_ELEM_H);
#else
    return 0.0;
#endif
}

double GetMu(double *y) {
    // TODO: exclude electron, grain?
    double mass = 28.0*y[IDX_GH2CNI] + 27.0*y[IDX_GHNCI] + 46.0*y[IDX_GNO2I] + 44.0*y[IDX_GSiOI] + 
                  28.0*y[IDX_GCOI] + 43.0*y[IDX_GHNCOI] + 24.0*y[IDX_GMgI] + 30.0*y[IDX_GNOI] + 
                  32.0*y[IDX_GO2I] + 33.0*y[IDX_GO2HI] + 40.0*y[IDX_GSiCI] + 52.0*y[IDX_GSiC2I] + 
                  64.0*y[IDX_GSiC3I] + 32.0*y[IDX_GCH3OHI] + 44.0*y[IDX_GCO2I] + 46.0*y[IDX_GH2SiOI] + 
                  31.0*y[IDX_GHNOI] + 28.0*y[IDX_GN2I] + 30.0*y[IDX_GH2COI] + 27.0*y[IDX_GHCNI] + 
                  18.0*y[IDX_GH2OI] + 17.0*y[IDX_GNH3I] + 64.0*y[IDX_SiC3II] + 28.0*y[IDX_H2CNI] + 
                  16.0*y[IDX_GCH4I] + 32.0*y[IDX_H2NOII] + 46.0*y[IDX_H2SiOI] + 5.0*y[IDX_HeHII] + 
                  43.0*y[IDX_HNCOI] + 29.0*y[IDX_HOCII] + 52.0*y[IDX_SiC2II] + 32.0*y[IDX_GSiH4I] + 
                  52.0*y[IDX_SiC2I] + 64.0*y[IDX_SiC3I] + 33.0*y[IDX_SiH5II] + 32.0*y[IDX_SiH4II] + 
                  40.0*y[IDX_SiCII] + 33.0*y[IDX_O2HI] + 40.0*y[IDX_SiCI] + 46.0*y[IDX_NO2I] + 
                  31.0*y[IDX_SiH3II] + 30.0*y[IDX_SiH2II] + 42.0*y[IDX_OCNI] + 30.0*y[IDX_SiH2I] + 
                  45.0*y[IDX_SiOHII] + 29.0*y[IDX_SiHII] + 32.0*y[IDX_SiH4I] + 29.0*y[IDX_SiHI] + 
                  31.0*y[IDX_SiH3I] + 44.0*y[IDX_SiOII] + 45.0*y[IDX_HCO2II] + 31.0*y[IDX_HNOI] + 
                  32.0*y[IDX_CH3OHI] + 24.0*y[IDX_MgI] + 24.0*y[IDX_MgII] + 16.0*y[IDX_CH4II] + 
                  44.0*y[IDX_SiOI] + 26.0*y[IDX_CNII] + 28.0*y[IDX_HCNHII] + 29.0*y[IDX_N2HII] + 
                  33.0*y[IDX_O2HII] + 28.0*y[IDX_SiII] + 28.0*y[IDX_SiI] + 27.0*y[IDX_HNCI] + 
                  31.0*y[IDX_HNOII] + 28.0*y[IDX_N2II] + 31.0*y[IDX_H3COII] + 16.0*y[IDX_CH4I] + 
                  28.0*y[IDX_COII] + 2.0*y[IDX_H2II] + 17.0*y[IDX_NH3I] + 15.0*y[IDX_CH3I] + 
                  44.0*y[IDX_CO2I] + 14.0*y[IDX_NII] + 16.0*y[IDX_OII] + 27.0*y[IDX_HCNII] + 
                  16.0*y[IDX_NH2II] + 15.0*y[IDX_NHII] + 32.0*y[IDX_O2II] + 15.0*y[IDX_CH3II] + 
                  16.0*y[IDX_NH2I] + 14.0*y[IDX_CH2II] + 18.0*y[IDX_H2OII] + 17.0*y[IDX_NH3II] + 
                  30.0*y[IDX_NOII] + 19.0*y[IDX_H3OII] + 28.0*y[IDX_N2I] + 12.0*y[IDX_CII] + 
                  27.0*y[IDX_HCNI] + 13.0*y[IDX_CHII] + 14.0*y[IDX_CH2I] + 30.0*y[IDX_H2COII] + 
                  15.0*y[IDX_NHI] + 17.0*y[IDX_OHII] + 26.0*y[IDX_CNI] + 30.0*y[IDX_H2COI] + 
                  29.0*y[IDX_HCOI] + 4.0*y[IDX_HeII] + 13.0*y[IDX_CHI] + 3.0*y[IDX_H3II] + 
                  4.0*y[IDX_HeI] + 30.0*y[IDX_NOI] + 14.0*y[IDX_NI] + 17.0*y[IDX_OHI] + 
                  32.0*y[IDX_O2I] + 12.0*y[IDX_CI] + 1.0*y[IDX_HII] + 29.0*y[IDX_HCOII] + 
                  18.0*y[IDX_H2OI] + 16.0*y[IDX_OI] + 0.0*y[IDX_EM] + 28.0*y[IDX_COI] + 
                  2.0*y[IDX_H2I] + 1.0*y[IDX_HI] + 0.0;
    double num = y[IDX_GH2CNI] + y[IDX_GHNCI] + y[IDX_GNO2I] + y[IDX_GSiOI] +
                 y[IDX_GCOI] + y[IDX_GHNCOI] + y[IDX_GMgI] + y[IDX_GNOI] +
                 y[IDX_GO2I] + y[IDX_GO2HI] + y[IDX_GSiCI] + y[IDX_GSiC2I] +
                 y[IDX_GSiC3I] + y[IDX_GCH3OHI] + y[IDX_GCO2I] + y[IDX_GH2SiOI]
                 + y[IDX_GHNOI] + y[IDX_GN2I] + y[IDX_GH2COI] + y[IDX_GHCNI] +
                 y[IDX_GH2OI] + y[IDX_GNH3I] + y[IDX_SiC3II] + y[IDX_H2CNI] +
                 y[IDX_GCH4I] + y[IDX_H2NOII] + y[IDX_H2SiOI] + y[IDX_HeHII] +
                 y[IDX_HNCOI] + y[IDX_HOCII] + y[IDX_SiC2II] + y[IDX_GSiH4I] +
                 y[IDX_SiC2I] + y[IDX_SiC3I] + y[IDX_SiH5II] + y[IDX_SiH4II] +
                 y[IDX_SiCII] + y[IDX_O2HI] + y[IDX_SiCI] + y[IDX_NO2I] +
                 y[IDX_SiH3II] + y[IDX_SiH2II] + y[IDX_OCNI] + y[IDX_SiH2I] +
                 y[IDX_SiOHII] + y[IDX_SiHII] + y[IDX_SiH4I] + y[IDX_SiHI] +
                 y[IDX_SiH3I] + y[IDX_SiOII] + y[IDX_HCO2II] + y[IDX_HNOI] +
                 y[IDX_CH3OHI] + y[IDX_MgI] + y[IDX_MgII] + y[IDX_CH4II] +
                 y[IDX_SiOI] + y[IDX_CNII] + y[IDX_HCNHII] + y[IDX_N2HII] +
                 y[IDX_O2HII] + y[IDX_SiII] + y[IDX_SiI] + y[IDX_HNCI] +
                 y[IDX_HNOII] + y[IDX_N2II] + y[IDX_H3COII] + y[IDX_CH4I] +
                 y[IDX_COII] + y[IDX_H2II] + y[IDX_NH3I] + y[IDX_CH3I] +
                 y[IDX_CO2I] + y[IDX_NII] + y[IDX_OII] + y[IDX_HCNII] +
                 y[IDX_NH2II] + y[IDX_NHII] + y[IDX_O2II] + y[IDX_CH3II] +
                 y[IDX_NH2I] + y[IDX_CH2II] + y[IDX_H2OII] + y[IDX_NH3II] +
                 y[IDX_NOII] + y[IDX_H3OII] + y[IDX_N2I] + y[IDX_CII] +
                 y[IDX_HCNI] + y[IDX_CHII] + y[IDX_CH2I] + y[IDX_H2COII] +
                 y[IDX_NHI] + y[IDX_OHII] + y[IDX_CNI] + y[IDX_H2COI] +
                 y[IDX_HCOI] + y[IDX_HeII] + y[IDX_CHI] + y[IDX_H3II] +
                 y[IDX_HeI] + y[IDX_NOI] + y[IDX_NI] + y[IDX_OHI] + y[IDX_O2I] +
                 y[IDX_CI] + y[IDX_HII] + y[IDX_HCOII] + y[IDX_H2OI] + y[IDX_OI]
                 + y[IDX_EM] + y[IDX_COI] + y[IDX_H2I] + y[IDX_HI] + 0.0;

    return mass / num;
}

double GetGamma(double *y) {
    // TODO: different ways to get adiabatic index
    return 5.0 / 3.0;
}

double GetNumDens(double *y) {
    double numdens = 0.0;

    for (int i = 0; i < NSPECIES; i++) numdens += y[i];
    return numdens;
}
// clang-format on

// clang-format off
double GetShieldingFactor(int specidx, double h2coldens, double spcoldens,
                          double tgas, int method) {
    // clang-format on
    double factor;
#ifdef IDX_H2I
    if (specidx == IDX_H2I) {
        factor = GetH2shielding(h2coldens, method);
    }
#endif
#ifdef IDX_COI
    if (specidx == IDX_COI) {
        factor = GetCOshielding(tgas, h2coldens, spcoldens, method);
    }
#endif
#ifdef IDX_N2I
    if (specidx == IDX_N2I) {
        factor = GetN2shielding(tgas, h2coldens, spcoldens, method);
    }
#endif

    return factor;
}

// clang-format off
double GetH2shielding(double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetH2shieldingInt(coldens);
            break;
        case 1:
            shielding = GetH2shieldingFGK(coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// clang-format off
double GetCOshielding(double tgas, double h2col, double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetCOshieldingInt(tgas, h2col, coldens);
            break;
        case 1:
            shielding = GetCOshieldingInt1(h2col, coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// clang-format off
double GetN2shielding(double tgas, double h2col, double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetN2shieldingInt(tgas, h2col, coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
double GetH2shieldingInt(double coldens) {
    // clang-format on

    double shielding = -1.0;

    /* */

    printf("WARNING!! Not Implemented! Return H2 shielding = -1.\n");

    /* */

    return shielding;
}

// Calculates the line self shielding function
// Ref: Federman et al. apj vol.227 p.466.
// Originally implemented in UCLCHEM
// clang-format off
double GetH2shieldingFGK(double coldens) {
    // clang-format on

    const double dopplerwidth       = 3.0e10;
    const double radiativewidth     = 8.0e7;
    const double oscillatorstrength = 1.0e-2;

    double shielding                = -1.0;

    double taud = 0.5 * coldens * 1.5e-2 * oscillatorstrength / dopplerwidth;

    // Calculate wing contribution of self shielding function sr
    if (taud < 0.0) taud = 0.0;

    double sr = 0.0;
    if (radiativewidth != 0.0) {
        double r = radiativewidth / (1.7724539 * dopplerwidth);
        double t = 3.02 * pow(1000.0 * r, -0.064);
        double u = pow(taud * r, 0.5) / t;
        sr       = pow((u * u + 0.78539816), -0.5) * r / t;
    }

    // Calculate doppler contribution of self shielding function sj
    double sj = 0.0;
    if (taud == 0.0) {
        sj = 1.0;
    } else if (taud < 2.0) {
        sj = exp(-0.6666667 * taud);
    } else if (taud < 10.0) {
        sj = 0.638 * pow(taud, -1.25);
    } else if (taud < 100.0) {
        sj = 0.505 * pow(taud, -1.15);
    } else {
        sj = 0.344 * pow(taud, -1.0667);
    }

    shielding = sj + sr;

    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
double GetCOshieldingInt(double tgas, double h2col, double coldens) {
    // clang-format on
    double shielding = -1.0;

    /* */

    printf("WARNING!! Not Implemented! Return CO shielding = -1.\n");

    /* */

    return shielding;
}

// clang-format off
double GetCOshieldingInt1(double h2col, double coldens) {
    // clang-format on
    double shielding = -1.0;

    /* */
    double logh2     = std::min(std::max(log10(h2col), COShieldingTableX[0]),
                                COShieldingTableX[5]);
    double logco     = std::min(std::max(log10(coldens), COShieldingTableY[0]),
                                COShieldingTableY[6]);

    /* */

    double *x1  = naunet_util::vector(1, 6);
    double *x2  = naunet_util::vector(1, 7);
    double **y  = naunet_util::matrix(1, 6, 1, 7);
    double **y2 = naunet_util::matrix(1, 6, 1, 7);

    for (int i = 1; i <= 6; i++) x1[i] = COShieldingTableX[i - 1];
    for (int i = 1; i <= 7; i++) x2[i] = COShieldingTableY[i - 1];

    for (int i = 1; i <= 6; i++) {
        for (int j = 1; j <= 7; j++) {
            y[i][j] = COShieldingTable[i - 1][j - 1];
        }
    }

    naunet_util::splie2(x1, x2, y, 6, 7, y2);

    naunet_util::splin2(x1, x2, y, y2, 6, 7, logh2, logco, &shielding);

    shielding = pow(10.0, shielding);

    naunet_util::free_vector(x1, 1, 6);
    naunet_util::free_vector(x2, 1, 7);
    naunet_util::free_matrix(y, 1, 6, 1, 7);
    naunet_util::free_matrix(y2, 1, 6, 1, 7);

    /* */

    /* */

    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
double GetN2shieldingInt(double tgas, double h2col, double coldens) {
    // clang-format on

    double shielding = -1.0;

    /* */

    printf("WARNING!! Not Implemented! Return N2 shielding = -1.\n");

    /* */

    return shielding;
}

// Calculate xlamda := tau(lambda) / tau(visual)
// tau(lambda) is the opt. depth for dust extinction at
// wavelength x (cf. b.d.savage and j.s.mathis, annual
// review of astronomy and astrophysics vol.17(1979),p.84)
// clang-format off
double xlamda(double wavelength) {
    // clang-format on
    double x[29] = {910.0,  950.0,  1000.0,  1050.0,  1110.0, 1180.0,
                    1250.0, 1390.0, 1490.0,  1600.0,  1700.0, 1800.0,
                    1900.0, 2000.0, 2100.0,  2190.0,  2300.0, 2400.0,
                    2500.0, 2740.0, 3440.0,  4000.0,  4400.0, 5500.0,
                    7000.0, 9000.0, 12500.0, 22000.0, 34000.0};

    double y[29] = {5.76, 5.18, 4.65, 4.16, 3.73, 3.4,  3.11, 2.74, 2.63, 2.62,
                    2.54, 2.5,  2.58, 2.78, 3.01, 3.12, 2.86, 2.58, 2.35, 2.0,
                    1.58, 1.42, 1.32, 1.0,  0.75, 0.48, 0.28, 0.12, 0.05};

    if (wavelength < x[0]) {
        return 5.76;
    }

    else if (wavelength >= x[28]) {
        return 0.05 - 5.16e-11 * (wavelength - x[28]);
    }

    for (int i = 0; i < 28; i++) {
        if (wavelength >= x[i] && wavelength < x[i + 1]) {
            return y[i] +
                   (y[i + 1] - y[i]) * (wavelength - x[i]) / (x[i + 1] - x[i]);
        }
    }

    return 0.0;
}

// Calculate the influence of dust extinction (g=0.8, omega=0.3)
// Ref: Wagenblast & Hartquist, mnras237, 1019 (1989)
// Adapted from UCLCHEM
// clang-format off
double GetGrainScattering(double av, double wavelength) {
    // clang-format on
    double c[6] = {1.0e0, 2.006e0, -1.438e0, 7.364e-1, -5.076e-1, -5.920e-2};
    double k[6] = {7.514e-1, 8.490e-1, 1.013e0, 1.282e0, 2.005e0, 5.832e0};

    double tv   = av / 1.086;
    double tl   = tv * xlamda(wavelength);

    double scat = 0.0;
    double expo;
    if (tl < 1.0) {
        expo = k[0] * tl;
        if (expo < 35.0) {
            scat = c[0] * exp(-expo);
        }
    } else {
        for (int i = 1; i < 6; i++) {
            expo = k[i] * tl;
            if (expo < 35.0) {
                scat = scat + c[i] * exp(-expo);
            }
        }
    }

    return scat;
}

// Calculate lambda bar (in a) according to equ. 4 of van dishoeck
// and black, apj 334, p771 (1988)
// Adapted from UCLCHEM
// clang-format off
double GetCharactWavelength(double h2col, double cocol) {
    // clang-format on
    double logco = log10(abs(cocol) + 1.0);
    double logh2 = log10(abs(h2col) + 1.0);

    double lbar  = (5675.0 - 200.6 * logh2) - (571.6 - 24.09 * logh2) * logco +
                  (18.22 - 0.7664 * logh2) * pow(logco, 2.0);

    // lbar represents the mean of the wavelengths of the 33
    // dissociating bands weighted by their fractional contribution
    // to the total rate of each depth. lbar cannot be larger than
    // the wavelength of band 33 (1076.1a) and not be smaller than
    // the wavelength of band 1 (913.6a).

    /* */
    lbar = std::min(1076.0, std::max(913.0, lbar));
    /* */
    return lbar;
}