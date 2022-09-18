// #include <stdio.h>

#include <stdexcept>
#include <vector>

#include "naunet.h"
#include "naunet_data.h"
#include "naunet_macros.h"
#include "naunet_physics.h"
#include "naunet_ode.h"
#include "naunet_timer.h"

int main() {

    //
    double nH       = 2e4;


    Naunet naunet;

    //
    double y[NEQUATIONS] = {0.0};
    // for (int i = 0; i < NEQUATIONS; i++)
    // {
    //     y[i] = 1.e-40;
    // }
    y[IDX_H2I]           = 0.5 * nH;
    y[IDX_HI]            = 5.0e-5 * nH;
    y[IDX_HeI]           = 9.75e-2 * nH;
    y[IDX_NI]            = 7.5e-5 * nH;
    y[IDX_OI]            = 1.8e-4 * nH;
    y[IDX_COI]           = 1.4e-4 * nH;
    y[IDX_SiI]           = 8.0e-9 * nH;
    y[IDX_MgI]           = 7.0e-9 * nH;


    printf("The reference element abundance.\n");
    double elem[NELEMENTS] = {0.0};
    for (int i = 0; i < NELEMENTS; i++) {
        double elemab = GetElementAbund(y, i);
        double Hnuclei = GetHNuclei(y);
        elem[i] = elemab / Hnuclei;
        printf("    element[%d] / Hnuclei = %13.7e\n", i, elemab / Hnuclei);
    }

    if (naunet.SetReferenceAbund(y, 1) == NAUNET_FAIL) {
        printf("Fail to set reference abundance.\n");
        return 1;
    }

    //
    y[IDX_H2I]           = 0.4 * nH;
    y[IDX_HI]            = 1.0e-3 * nH;
    y[IDX_HeI]           = 1.9e-1 * nH;
    y[IDX_NI]            = 7.5e-5 * nH;
    y[IDX_OI]            = 8.0e-5 * nH;
    y[IDX_COI]           = 2.8e-4 * nH;
    y[IDX_SiI]           = 2.0e-9 * nH;
    y[IDX_MgI]           = 2.0e-9 * nH;


    printf("The updated abundance.\n");
    for (int i = 0; i < NELEMENTS; i++) {
        double elemab = GetElementAbund(y, i);
        double Hnuclei = GetHNuclei(y);
        printf("    element[%d] / Hnuclei = %13.7e\n", i, elemab / Hnuclei);
    }

    int flag = naunet.Renorm(y);

    printf("The renormalized abundance.\n");
    for (int i = 0; i < NELEMENTS; i++) {
        double elemab = GetElementAbund(y, i);
        double Hnuclei = GetHNuclei(y);
        printf("    element[%d] / Hnuclei = %13.7e\n", i, elemab / Hnuclei);
        if (abs(elemab / Hnuclei - elem[i]) > 1e-3) {
            return 1;
        }
    }

    return 0;
}
