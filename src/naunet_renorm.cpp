#include "naunet_macros.h"
#include "naunet_physics.h"
#include "naunet_renorm.h"

// clang-format off
int InitRenorm(realtype *ab, SUNMatrix A) {
    // clang-format on
    realtype Hnuclei = GetHNuclei(ab);

    // clang-format off
            
    IJth(A, IDX_ELEM_MG, IDX_ELEM_MG) = 0.0 + 24.0 * ab[IDX_GMgI] / 24.0 / Hnuclei +
                                    24.0 * ab[IDX_MgI] / 24.0 / Hnuclei + 24.0 *
                                    ab[IDX_MgII] / 24.0 / Hnuclei;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_SI) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_H) = 0.0;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_SI) = 0.0 + 28.0 * ab[IDX_GSiOI] / 44.0 / Hnuclei
                                    + 28.0 * ab[IDX_GSiCI] / 40.0 / Hnuclei +
                                    28.0 * ab[IDX_GSiC2I] / 52.0 / Hnuclei +
                                    28.0 * ab[IDX_GSiC3I] / 64.0 / Hnuclei +
                                    28.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei +
                                    28.0 * ab[IDX_SiC3II] / 64.0 / Hnuclei +
                                    28.0 * ab[IDX_H2SiOI] / 46.0 / Hnuclei +
                                    28.0 * ab[IDX_SiC2II] / 52.0 / Hnuclei +
                                    28.0 * ab[IDX_GSiH4I] / 32.0 / Hnuclei +
                                    28.0 * ab[IDX_SiC2I] / 52.0 / Hnuclei + 28.0
                                    * ab[IDX_SiC3I] / 64.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH5II] / 33.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH4II] / 32.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCII] / 40.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCI] / 40.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH3II] / 31.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH2II] / 30.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH2I] / 30.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiHII] / 29.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH4I] / 32.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiHI] / 29.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH3I] / 31.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOII] / 44.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOI] / 44.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiII] / 28.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiI] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_GSiCI] / 40.0 / Hnuclei
                                    + 24.0 * ab[IDX_GSiC2I] / 52.0 / Hnuclei +
                                    36.0 * ab[IDX_GSiC3I] / 64.0 / Hnuclei +
                                    36.0 * ab[IDX_SiC3II] / 64.0 / Hnuclei +
                                    24.0 * ab[IDX_SiC2II] / 52.0 / Hnuclei +
                                    24.0 * ab[IDX_SiC2I] / 52.0 / Hnuclei + 36.0
                                    * ab[IDX_SiC3I] / 64.0 / Hnuclei + 12.0 *
                                    ab[IDX_SiCII] / 40.0 / Hnuclei + 12.0 *
                                    ab[IDX_SiCI] / 40.0 / Hnuclei;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GSiOI] / 44.0 / Hnuclei
                                    + 16.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei +
                                    16.0 * ab[IDX_H2SiOI] / 46.0 / Hnuclei +
                                    16.0 * ab[IDX_SiOHII] / 45.0 / Hnuclei +
                                    16.0 * ab[IDX_SiOII] / 44.0 / Hnuclei + 16.0
                                    * ab[IDX_SiOI] / 44.0 / Hnuclei;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_H) = 0.0 + 2.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei
                                    + 2.0 * ab[IDX_H2SiOI] / 46.0 / Hnuclei +
                                    4.0 * ab[IDX_GSiH4I] / 32.0 / Hnuclei + 5.0
                                    * ab[IDX_SiH5II] / 33.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiH4II] / 32.0 / Hnuclei + 3.0 *
                                    ab[IDX_SiH3II] / 31.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiH2II] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiH2I] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiHII] / 29.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiH4I] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiHI] / 29.0 / Hnuclei + 3.0 *
                                    ab[IDX_SiH3I] / 31.0 / Hnuclei;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_SI) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_HE) = 0.0 + 4.0 * ab[IDX_HeHII] / 5.0 / Hnuclei +
                                    4.0 * ab[IDX_HeII] / 4.0 / Hnuclei + 4.0 *
                                    ab[IDX_HeI] / 4.0 / Hnuclei;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_HeHII] / 5.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_SI) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_GH2CNI] / 28.0 / Hnuclei
                                    + 14.0 * ab[IDX_GHNCI] / 27.0 / Hnuclei +
                                    14.0 * ab[IDX_GNO2I] / 46.0 / Hnuclei + 14.0
                                    * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 14.0 *
                                    ab[IDX_GNOI] / 30.0 / Hnuclei + 14.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 56.0 *
                                    ab[IDX_GN2I] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_GHCNI] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_GNH3I] / 17.0 / Hnuclei + 14.0 *
                                    ab[IDX_H2CNI] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 14.0 *
                                    ab[IDX_NO2I] / 46.0 / Hnuclei + 14.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_CNII] / 26.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 56.0 *
                                    ab[IDX_N2HII] / 29.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 56.0 *
                                    ab[IDX_N2II] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_NH3I] / 17.0 / Hnuclei + 14.0 *
                                    ab[IDX_NII] / 14.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_NH2II] / 16.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHII] / 15.0 / Hnuclei + 14.0 *
                                    ab[IDX_NH2I] / 16.0 / Hnuclei + 14.0 *
                                    ab[IDX_NH3II] / 17.0 / Hnuclei + 14.0 *
                                    ab[IDX_NOII] / 30.0 / Hnuclei + 56.0 *
                                    ab[IDX_N2I] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHI] / 15.0 / Hnuclei + 14.0 *
                                    ab[IDX_CNI] / 26.0 / Hnuclei + 14.0 *
                                    ab[IDX_NOI] / 30.0 / Hnuclei + 14.0 *
                                    ab[IDX_NI] / 14.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_GH2CNI] / 28.0 / Hnuclei
                                    + 12.0 * ab[IDX_GHNCI] / 27.0 / Hnuclei +
                                    12.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    12.0 * ab[IDX_GHCNI] / 27.0 / Hnuclei + 12.0
                                    * ab[IDX_H2CNI] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 12.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNII] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNI] / 26.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_O) = 0.0 + 32.0 * ab[IDX_GNO2I] / 46.0 / Hnuclei
                                    + 16.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    16.0 * ab[IDX_GNOI] / 30.0 / Hnuclei + 16.0
                                    * ab[IDX_GHNOI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 32.0 *
                                    ab[IDX_NO2I] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOI] / 30.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_H) = 0.0 + 2.0 * ab[IDX_GH2CNI] / 28.0 / Hnuclei
                                    + 1.0 * ab[IDX_GHNCI] / 27.0 / Hnuclei + 1.0
                                    * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCNI] / 27.0 / Hnuclei + 3.0 *
                                    ab[IDX_GNH3I] / 17.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CNI] / 28.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 2.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 2.0 *
                                    ab[IDX_N2HII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 3.0 *
                                    ab[IDX_NH3I] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 2.0 *
                                    ab[IDX_NH2II] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHII] / 15.0 / Hnuclei + 2.0 *
                                    ab[IDX_NH2I] / 16.0 / Hnuclei + 3.0 *
                                    ab[IDX_NH3II] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHI] / 15.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_SI) = 0.0 + 28.0 * ab[IDX_GSiCI] / 40.0 / Hnuclei
                                    + 56.0 * ab[IDX_GSiC2I] / 52.0 / Hnuclei +
                                    84.0 * ab[IDX_GSiC3I] / 64.0 / Hnuclei +
                                    84.0 * ab[IDX_SiC3II] / 64.0 / Hnuclei +
                                    56.0 * ab[IDX_SiC2II] / 52.0 / Hnuclei +
                                    56.0 * ab[IDX_SiC2I] / 52.0 / Hnuclei + 84.0
                                    * ab[IDX_SiC3I] / 64.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCII] / 40.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCI] / 40.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_GH2CNI] / 28.0 / Hnuclei
                                    + 14.0 * ab[IDX_GHNCI] / 27.0 / Hnuclei +
                                    14.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    14.0 * ab[IDX_GHCNI] / 27.0 / Hnuclei + 14.0
                                    * ab[IDX_H2CNI] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 14.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 14.0 *
                                    ab[IDX_CNII] / 26.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_CNI] / 26.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_GH2CNI] / 28.0 / Hnuclei
                                    + 12.0 * ab[IDX_GHNCI] / 27.0 / Hnuclei +
                                    12.0 * ab[IDX_GCOI] / 28.0 / Hnuclei + 12.0
                                    * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 12.0 *
                                    ab[IDX_GSiCI] / 40.0 / Hnuclei + 48.0 *
                                    ab[IDX_GSiC2I] / 52.0 / Hnuclei + 108.0 *
                                    ab[IDX_GSiC3I] / 64.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCO2I] / 44.0 / Hnuclei + 12.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_GHCNI] / 27.0 / Hnuclei + 108.0 *
                                    ab[IDX_SiC3II] / 64.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2CNI] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCH4I] / 16.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 12.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 48.0 *
                                    ab[IDX_SiC2II] / 52.0 / Hnuclei + 48.0 *
                                    ab[IDX_SiC2I] / 52.0 / Hnuclei + 108.0 *
                                    ab[IDX_SiC3I] / 64.0 / Hnuclei + 12.0 *
                                    ab[IDX_SiCII] / 40.0 / Hnuclei + 12.0 *
                                    ab[IDX_SiCI] / 40.0 / Hnuclei + 12.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH4II] / 16.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNII] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH4I] / 16.0 / Hnuclei + 12.0 *
                                    ab[IDX_COII] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3I] / 15.0 / Hnuclei + 12.0 *
                                    ab[IDX_CO2I] / 44.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3II] / 15.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 12.0 *
                                    ab[IDX_CII] / 12.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNI] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei + 12.0 *
                                    ab[IDX_CI] / 12.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_COI] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GCOI] / 28.0 / Hnuclei +
                                    16.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    16.0 * ab[IDX_GCH3OHI] / 32.0 / Hnuclei +
                                    32.0 * ab[IDX_GCO2I] / 44.0 / Hnuclei + 16.0
                                    * ab[IDX_GH2COI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 32.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_COII] / 28.0 / Hnuclei + 32.0 *
                                    ab[IDX_CO2I] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_COI] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_H) = 0.0 + 2.0 * ab[IDX_GH2CNI] / 28.0 / Hnuclei
                                    + 1.0 * ab[IDX_GHNCI] / 27.0 / Hnuclei + 1.0
                                    * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCNI] / 27.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CNI] / 28.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH4I] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH4II] / 16.0 / Hnuclei + 2.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH4I] / 16.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH3I] / 15.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH3II] / 15.0 / Hnuclei + 2.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 2.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_SI) = 0.0 + 28.0 * ab[IDX_GSiOI] / 44.0 / Hnuclei
                                    + 28.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei +
                                    28.0 * ab[IDX_H2SiOI] / 46.0 / Hnuclei +
                                    28.0 * ab[IDX_SiOHII] / 45.0 / Hnuclei +
                                    28.0 * ab[IDX_SiOII] / 44.0 / Hnuclei + 28.0
                                    * ab[IDX_SiOI] / 44.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_N) = 0.0 + 28.0 * ab[IDX_GNO2I] / 46.0 / Hnuclei
                                    + 14.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    14.0 * ab[IDX_GNOI] / 30.0 / Hnuclei + 14.0
                                    * ab[IDX_GHNOI] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 28.0 *
                                    ab[IDX_NO2I] / 46.0 / Hnuclei + 14.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_NOII] / 30.0 / Hnuclei + 14.0 *
                                    ab[IDX_NOI] / 30.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_GCOI] / 28.0 / Hnuclei +
                                    12.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    12.0 * ab[IDX_GCH3OHI] / 32.0 / Hnuclei +
                                    24.0 * ab[IDX_GCO2I] / 44.0 / Hnuclei + 12.0
                                    * ab[IDX_GH2COI] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 12.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 24.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 12.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 12.0 *
                                    ab[IDX_COII] / 28.0 / Hnuclei + 24.0 *
                                    ab[IDX_CO2I] / 44.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_COI] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_O) = 0.0 + 64.0 * ab[IDX_GNO2I] / 46.0 / Hnuclei
                                    + 16.0 * ab[IDX_GSiOI] / 44.0 / Hnuclei +
                                    16.0 * ab[IDX_GCOI] / 28.0 / Hnuclei + 16.0
                                    * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_GNOI] / 30.0 / Hnuclei + 64.0 *
                                    ab[IDX_GO2I] / 32.0 / Hnuclei + 64.0 *
                                    ab[IDX_GO2HI] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 64.0 *
                                    ab[IDX_GCO2I] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_GH2SiOI] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_GH2OI] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 64.0 *
                                    ab[IDX_O2HI] / 33.0 / Hnuclei + 64.0 *
                                    ab[IDX_NO2I] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiOII] / 44.0 / Hnuclei + 64.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiOI] / 44.0 / Hnuclei + 64.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_COII] / 28.0 / Hnuclei + 64.0 *
                                    ab[IDX_CO2I] / 44.0 / Hnuclei + 16.0 *
                                    ab[IDX_OII] / 16.0 / Hnuclei + 64.0 *
                                    ab[IDX_O2II] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei + 64.0 *
                                    ab[IDX_O2I] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei + 16.0 *
                                    ab[IDX_OI] / 16.0 / Hnuclei + 16.0 *
                                    ab[IDX_COI] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei
                                    + 2.0 * ab[IDX_GO2HI] / 33.0 / Hnuclei + 4.0
                                    * ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2SiOI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2OI] / 18.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 2.0 *
                                    ab[IDX_O2HI] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 2.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_H, IDX_ELEM_SI) = 0.0 + 56.0 * ab[IDX_GH2SiOI] / 46.0 /
                                    Hnuclei + 56.0 * ab[IDX_H2SiOI] / 46.0 /
                                    Hnuclei + 112.0 * ab[IDX_GSiH4I] / 32.0 /
                                    Hnuclei + 140.0 * ab[IDX_SiH5II] / 33.0 /
                                    Hnuclei + 112.0 * ab[IDX_SiH4II] / 32.0 /
                                    Hnuclei + 84.0 * ab[IDX_SiH3II] / 31.0 /
                                    Hnuclei + 56.0 * ab[IDX_SiH2II] / 30.0 /
                                    Hnuclei + 56.0 * ab[IDX_SiH2I] / 30.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiOHII] / 45.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiHII] / 29.0 /
                                    Hnuclei + 112.0 * ab[IDX_SiH4I] / 32.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiHI] / 29.0 /
                                    Hnuclei + 84.0 * ab[IDX_SiH3I] / 31.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_HE) = 0.0 + 4.0 * ab[IDX_HeHII] / 5.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_N) = 0.0 + 28.0 * ab[IDX_GH2CNI] / 28.0 / Hnuclei
                                    + 14.0 * ab[IDX_GHNCI] / 27.0 / Hnuclei +
                                    14.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    14.0 * ab[IDX_GHNOI] / 31.0 / Hnuclei + 14.0
                                    * ab[IDX_GHCNI] / 27.0 / Hnuclei + 42.0 *
                                    ab[IDX_GNH3I] / 17.0 / Hnuclei + 28.0 *
                                    ab[IDX_H2CNI] / 28.0 / Hnuclei + 28.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 28.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 28.0 *
                                    ab[IDX_N2HII] / 29.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 42.0 *
                                    ab[IDX_NH3I] / 17.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 28.0 *
                                    ab[IDX_NH2II] / 16.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHII] / 15.0 / Hnuclei + 28.0 *
                                    ab[IDX_NH2I] / 16.0 / Hnuclei + 42.0 *
                                    ab[IDX_NH3II] / 17.0 / Hnuclei + 14.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 14.0 *
                                    ab[IDX_NHI] / 15.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_C) = 0.0 + 24.0 * ab[IDX_GH2CNI] / 28.0 / Hnuclei
                                    + 12.0 * ab[IDX_GHNCI] / 27.0 / Hnuclei +
                                    12.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    48.0 * ab[IDX_GCH3OHI] / 32.0 / Hnuclei +
                                    24.0 * ab[IDX_GH2COI] / 30.0 / Hnuclei +
                                    12.0 * ab[IDX_GHCNI] / 27.0 / Hnuclei + 24.0
                                    * ab[IDX_H2CNI] / 28.0 / Hnuclei + 48.0 *
                                    ab[IDX_GCH4I] / 16.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 12.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH4II] / 16.0 / Hnuclei + 24.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 36.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH4I] / 16.0 / Hnuclei + 36.0 *
                                    ab[IDX_CH3I] / 15.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 36.0 *
                                    ab[IDX_CH3II] / 15.0 / Hnuclei + 24.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 24.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 24.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 24.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei
                                    + 32.0 * ab[IDX_GO2HI] / 33.0 / Hnuclei +
                                    64.0 * ab[IDX_GCH3OHI] / 32.0 / Hnuclei +
                                    32.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei +
                                    16.0 * ab[IDX_GHNOI] / 31.0 / Hnuclei + 32.0
                                    * ab[IDX_GH2COI] / 30.0 / Hnuclei + 32.0 *
                                    ab[IDX_GH2OI] / 18.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 32.0 *
                                    ab[IDX_O2HI] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 32.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 64.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 32.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 48.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 48.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_H) = 0.0 + 4.0 * ab[IDX_GH2CNI] / 28.0 / Hnuclei
                                    + 1.0 * ab[IDX_GHNCI] / 27.0 / Hnuclei + 1.0
                                    * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GO2HI] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2SiOI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCNI] / 27.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2OI] / 18.0 / Hnuclei + 9.0 *
                                    ab[IDX_GNH3I] / 17.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2CNI] / 28.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH4I] / 16.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HeHII] / 5.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_GSiH4I] / 32.0 / Hnuclei + 25.0 *
                                    ab[IDX_SiH5II] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiH4II] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_O2HI] / 33.0 / Hnuclei + 9.0 *
                                    ab[IDX_SiH3II] / 31.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiH2II] / 30.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiH2I] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiHII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiH4I] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiHI] / 29.0 / Hnuclei + 9.0 *
                                    ab[IDX_SiH3I] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH4II] / 16.0 / Hnuclei + 4.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 1.0 *
                                    ab[IDX_N2HII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH4I] / 16.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2II] / 2.0 / Hnuclei + 9.0 *
                                    ab[IDX_NH3I] / 17.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3I] / 15.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 4.0 *
                                    ab[IDX_NH2II] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHII] / 15.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3II] / 15.0 / Hnuclei + 4.0 *
                                    ab[IDX_NH2I] / 16.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 9.0 *
                                    ab[IDX_NH3II] / 17.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHI] / 15.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3II] / 3.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_HII] / 1.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2I] / 2.0 / Hnuclei + 1.0 *
                                    ab[IDX_HI] / 1.0 / Hnuclei;
        // clang-format on

    return NAUNET_SUCCESS;
}

// clang-format off
int RenormAbundance(realtype *rptr, realtype *ab) {
    
    ab[IDX_GH2CNI] = ab[IDX_GH2CNI] * (14.0 * rptr[IDX_ELEM_N] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 2.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_GHNCI] = ab[IDX_GHNCI] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_GNO2I] = ab[IDX_GNO2I] * (14.0 * rptr[IDX_ELEM_N] / 46.0 + 32.0 * rptr[IDX_ELEM_O] / 46.0);
    ab[IDX_GSiOI] = ab[IDX_GSiOI] * (28.0 * rptr[IDX_ELEM_SI] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_GCOI] = ab[IDX_GCOI] * (12.0 * rptr[IDX_ELEM_C] / 28.0 + 16.0 * rptr[IDX_ELEM_O] / 28.0);
    ab[IDX_GHNCOI] = ab[IDX_GHNCOI] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_GMgI] = ab[IDX_GMgI] * (24.0 * rptr[IDX_ELEM_MG] / 24.0);
    ab[IDX_GNOI] = ab[IDX_GNOI] * (14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0);
    ab[IDX_GO2I] = ab[IDX_GO2I] * (32.0 * rptr[IDX_ELEM_O] / 32.0);
    ab[IDX_GO2HI] = ab[IDX_GO2HI] * (32.0 * rptr[IDX_ELEM_O] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_GSiCI] = ab[IDX_GSiCI] * (28.0 * rptr[IDX_ELEM_SI] / 40.0 + 12.0 * rptr[IDX_ELEM_C] / 40.0);
    ab[IDX_GSiC2I] = ab[IDX_GSiC2I] * (28.0 * rptr[IDX_ELEM_SI] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_GSiC3I] = ab[IDX_GSiC3I] * (28.0 * rptr[IDX_ELEM_SI] / 64.0 + 36.0 * rptr[IDX_ELEM_C] / 64.0);
    ab[IDX_GCH3OHI] = ab[IDX_GCH3OHI] * (12.0 * rptr[IDX_ELEM_C] / 32.0 + 16.0 * rptr[IDX_ELEM_O] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_GCO2I] = ab[IDX_GCO2I] * (12.0 * rptr[IDX_ELEM_C] / 44.0 + 32.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_GH2SiOI] = ab[IDX_GH2SiOI] * (28.0 * rptr[IDX_ELEM_SI] / 46.0 + 16.0 * rptr[IDX_ELEM_O] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_GHNOI] = ab[IDX_GHNOI] * (14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_GN2I] = ab[IDX_GN2I] * (28.0 * rptr[IDX_ELEM_N] / 28.0);
    ab[IDX_GH2COI] = ab[IDX_GH2COI] * (12.0 * rptr[IDX_ELEM_C] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_GHCNI] = ab[IDX_GHCNI] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_GH2OI] = ab[IDX_GH2OI] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0);
    ab[IDX_GNH3I] = ab[IDX_GNH3I] * (14.0 * rptr[IDX_ELEM_N] / 17.0 + 3.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_SiC3II] = ab[IDX_SiC3II] * (28.0 * rptr[IDX_ELEM_SI] / 64.0 + 36.0 * rptr[IDX_ELEM_C] / 64.0);
    ab[IDX_H2CNI] = ab[IDX_H2CNI] * (14.0 * rptr[IDX_ELEM_N] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 2.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_GCH4I] = ab[IDX_GCH4I] * (12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_H2NOII] = ab[IDX_H2NOII] * (14.0 * rptr[IDX_ELEM_N] / 32.0 + 16.0 * rptr[IDX_ELEM_O] / 32.0 + 2.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_H2SiOI] = ab[IDX_H2SiOI] * (28.0 * rptr[IDX_ELEM_SI] / 46.0 + 16.0 * rptr[IDX_ELEM_O] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_HeHII] = ab[IDX_HeHII] * (4.0 * rptr[IDX_ELEM_HE] / 5.0 + 1.0 * rptr[IDX_ELEM_H] / 5.0);
    ab[IDX_HNCOI] = ab[IDX_HNCOI] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_HOCII] = ab[IDX_HOCII] * (12.0 * rptr[IDX_ELEM_C] / 29.0 + 16.0 * rptr[IDX_ELEM_O] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_SiC2II] = ab[IDX_SiC2II] * (28.0 * rptr[IDX_ELEM_SI] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_GSiH4I] = ab[IDX_GSiH4I] * (28.0 * rptr[IDX_ELEM_SI] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_SiC2I] = ab[IDX_SiC2I] * (28.0 * rptr[IDX_ELEM_SI] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_SiC3I] = ab[IDX_SiC3I] * (28.0 * rptr[IDX_ELEM_SI] / 64.0 + 36.0 * rptr[IDX_ELEM_C] / 64.0);
    ab[IDX_SiH5II] = ab[IDX_SiH5II] * (28.0 * rptr[IDX_ELEM_SI] / 33.0 + 5.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_SiH4II] = ab[IDX_SiH4II] * (28.0 * rptr[IDX_ELEM_SI] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_SiCII] = ab[IDX_SiCII] * (28.0 * rptr[IDX_ELEM_SI] / 40.0 + 12.0 * rptr[IDX_ELEM_C] / 40.0);
    ab[IDX_O2HI] = ab[IDX_O2HI] * (32.0 * rptr[IDX_ELEM_O] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_SiCI] = ab[IDX_SiCI] * (28.0 * rptr[IDX_ELEM_SI] / 40.0 + 12.0 * rptr[IDX_ELEM_C] / 40.0);
    ab[IDX_NO2I] = ab[IDX_NO2I] * (14.0 * rptr[IDX_ELEM_N] / 46.0 + 32.0 * rptr[IDX_ELEM_O] / 46.0);
    ab[IDX_SiH3II] = ab[IDX_SiH3II] * (28.0 * rptr[IDX_ELEM_SI] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_SiH2II] = ab[IDX_SiH2II] * (28.0 * rptr[IDX_ELEM_SI] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_OCNI] = ab[IDX_OCNI] * (14.0 * rptr[IDX_ELEM_N] / 42.0 + 12.0 * rptr[IDX_ELEM_C] / 42.0 + 16.0 * rptr[IDX_ELEM_O] / 42.0);
    ab[IDX_SiH2I] = ab[IDX_SiH2I] * (28.0 * rptr[IDX_ELEM_SI] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_SiOHII] = ab[IDX_SiOHII] * (28.0 * rptr[IDX_ELEM_SI] / 45.0 + 16.0 * rptr[IDX_ELEM_O] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_SiHII] = ab[IDX_SiHII] * (28.0 * rptr[IDX_ELEM_SI] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_SiH4I] = ab[IDX_SiH4I] * (28.0 * rptr[IDX_ELEM_SI] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_SiHI] = ab[IDX_SiHI] * (28.0 * rptr[IDX_ELEM_SI] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_SiH3I] = ab[IDX_SiH3I] * (28.0 * rptr[IDX_ELEM_SI] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_SiOII] = ab[IDX_SiOII] * (28.0 * rptr[IDX_ELEM_SI] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_HCO2II] = ab[IDX_HCO2II] * (12.0 * rptr[IDX_ELEM_C] / 45.0 + 32.0 * rptr[IDX_ELEM_O] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_HNOI] = ab[IDX_HNOI] * (14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_CH3OHI] = ab[IDX_CH3OHI] * (12.0 * rptr[IDX_ELEM_C] / 32.0 + 16.0 * rptr[IDX_ELEM_O] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_MgI] = ab[IDX_MgI] * (24.0 * rptr[IDX_ELEM_MG] / 24.0);
    ab[IDX_MgII] = ab[IDX_MgII] * (24.0 * rptr[IDX_ELEM_MG] / 24.0);
    ab[IDX_CH4II] = ab[IDX_CH4II] * (12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_SiOI] = ab[IDX_SiOI] * (28.0 * rptr[IDX_ELEM_SI] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_CNII] = ab[IDX_CNII] * (14.0 * rptr[IDX_ELEM_N] / 26.0 + 12.0 * rptr[IDX_ELEM_C] / 26.0);
    ab[IDX_HCNHII] = ab[IDX_HCNHII] * (14.0 * rptr[IDX_ELEM_N] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 2.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_N2HII] = ab[IDX_N2HII] * (28.0 * rptr[IDX_ELEM_N] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_O2HII] = ab[IDX_O2HII] * (32.0 * rptr[IDX_ELEM_O] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_SiII] = ab[IDX_SiII] * (28.0 * rptr[IDX_ELEM_SI] / 28.0);
    ab[IDX_SiI] = ab[IDX_SiI] * (28.0 * rptr[IDX_ELEM_SI] / 28.0);
    ab[IDX_HNCI] = ab[IDX_HNCI] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_HNOII] = ab[IDX_HNOII] * (14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_N2II] = ab[IDX_N2II] * (28.0 * rptr[IDX_ELEM_N] / 28.0);
    ab[IDX_H3COII] = ab[IDX_H3COII] * (12.0 * rptr[IDX_ELEM_C] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_CH4I] = ab[IDX_CH4I] * (12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_COII] = ab[IDX_COII] * (12.0 * rptr[IDX_ELEM_C] / 28.0 + 16.0 * rptr[IDX_ELEM_O] / 28.0);
    ab[IDX_H2II] = ab[IDX_H2II] * (2.0 * rptr[IDX_ELEM_H] / 2.0);
    ab[IDX_NH3I] = ab[IDX_NH3I] * (14.0 * rptr[IDX_ELEM_N] / 17.0 + 3.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_CH3I] = ab[IDX_CH3I] * (12.0 * rptr[IDX_ELEM_C] / 15.0 + 3.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_CO2I] = ab[IDX_CO2I] * (12.0 * rptr[IDX_ELEM_C] / 44.0 + 32.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_NII] = ab[IDX_NII] * (14.0 * rptr[IDX_ELEM_N] / 14.0);
    ab[IDX_OII] = ab[IDX_OII] * (16.0 * rptr[IDX_ELEM_O] / 16.0);
    ab[IDX_HCNII] = ab[IDX_HCNII] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_NH2II] = ab[IDX_NH2II] * (14.0 * rptr[IDX_ELEM_N] / 16.0 + 2.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_NHII] = ab[IDX_NHII] * (14.0 * rptr[IDX_ELEM_N] / 15.0 + 1.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_O2II] = ab[IDX_O2II] * (32.0 * rptr[IDX_ELEM_O] / 32.0);
    ab[IDX_CH3II] = ab[IDX_CH3II] * (12.0 * rptr[IDX_ELEM_C] / 15.0 + 3.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_NH2I] = ab[IDX_NH2I] * (14.0 * rptr[IDX_ELEM_N] / 16.0 + 2.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_CH2II] = ab[IDX_CH2II] * (12.0 * rptr[IDX_ELEM_C] / 14.0 + 2.0 * rptr[IDX_ELEM_H] / 14.0);
    ab[IDX_H2OII] = ab[IDX_H2OII] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0);
    ab[IDX_NH3II] = ab[IDX_NH3II] * (14.0 * rptr[IDX_ELEM_N] / 17.0 + 3.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_NOII] = ab[IDX_NOII] * (14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0);
    ab[IDX_H3OII] = ab[IDX_H3OII] * (16.0 * rptr[IDX_ELEM_O] / 19.0 + 3.0 * rptr[IDX_ELEM_H] / 19.0);
    ab[IDX_N2I] = ab[IDX_N2I] * (28.0 * rptr[IDX_ELEM_N] / 28.0);
    ab[IDX_CII] = ab[IDX_CII] * (12.0 * rptr[IDX_ELEM_C] / 12.0);
    ab[IDX_HCNI] = ab[IDX_HCNI] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_CHII] = ab[IDX_CHII] * (12.0 * rptr[IDX_ELEM_C] / 13.0 + 1.0 * rptr[IDX_ELEM_H] / 13.0);
    ab[IDX_CH2I] = ab[IDX_CH2I] * (12.0 * rptr[IDX_ELEM_C] / 14.0 + 2.0 * rptr[IDX_ELEM_H] / 14.0);
    ab[IDX_H2COII] = ab[IDX_H2COII] * (12.0 * rptr[IDX_ELEM_C] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_NHI] = ab[IDX_NHI] * (14.0 * rptr[IDX_ELEM_N] / 15.0 + 1.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_OHII] = ab[IDX_OHII] * (16.0 * rptr[IDX_ELEM_O] / 17.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_CNI] = ab[IDX_CNI] * (14.0 * rptr[IDX_ELEM_N] / 26.0 + 12.0 * rptr[IDX_ELEM_C] / 26.0);
    ab[IDX_H2COI] = ab[IDX_H2COI] * (12.0 * rptr[IDX_ELEM_C] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_HCOI] = ab[IDX_HCOI] * (12.0 * rptr[IDX_ELEM_C] / 29.0 + 16.0 * rptr[IDX_ELEM_O] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_HeII] = ab[IDX_HeII] * (4.0 * rptr[IDX_ELEM_HE] / 4.0);
    ab[IDX_CHI] = ab[IDX_CHI] * (12.0 * rptr[IDX_ELEM_C] / 13.0 + 1.0 * rptr[IDX_ELEM_H] / 13.0);
    ab[IDX_H3II] = ab[IDX_H3II] * (3.0 * rptr[IDX_ELEM_H] / 3.0);
    ab[IDX_HeI] = ab[IDX_HeI] * (4.0 * rptr[IDX_ELEM_HE] / 4.0);
    ab[IDX_NOI] = ab[IDX_NOI] * (14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0);
    ab[IDX_NI] = ab[IDX_NI] * (14.0 * rptr[IDX_ELEM_N] / 14.0);
    ab[IDX_OHI] = ab[IDX_OHI] * (16.0 * rptr[IDX_ELEM_O] / 17.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_O2I] = ab[IDX_O2I] * (32.0 * rptr[IDX_ELEM_O] / 32.0);
    ab[IDX_CI] = ab[IDX_CI] * (12.0 * rptr[IDX_ELEM_C] / 12.0);
    ab[IDX_HII] = ab[IDX_HII] * (1.0 * rptr[IDX_ELEM_H] / 1.0);
    ab[IDX_HCOII] = ab[IDX_HCOII] * (12.0 * rptr[IDX_ELEM_C] / 29.0 + 16.0 * rptr[IDX_ELEM_O] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_H2OI] = ab[IDX_H2OI] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0);
    ab[IDX_OI] = ab[IDX_OI] * (16.0 * rptr[IDX_ELEM_O] / 16.0);
    ab[IDX_EM] = ab[IDX_EM] * (1.0);
    ab[IDX_COI] = ab[IDX_COI] * (12.0 * rptr[IDX_ELEM_C] / 28.0 + 16.0 * rptr[IDX_ELEM_O] / 28.0);
    ab[IDX_H2I] = ab[IDX_H2I] * (2.0 * rptr[IDX_ELEM_H] / 2.0);
    ab[IDX_HI] = ab[IDX_HI] * (1.0 * rptr[IDX_ELEM_H] / 1.0);
        // clang-format on

    return NAUNET_SUCCESS;
}