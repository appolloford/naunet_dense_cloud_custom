#ifndef __NAUNET_DATA_H__
#define __NAUNET_DATA_H__

// Struct for holding the nessesary additional variables for the problem.
struct NaunetData {
    // clang-format off
    double rG;
    double gdens;
    double sites;
    double fr;
    double opt_thd;
    double opt_crd;
    double opt_h2d;
    double opt_uvd;
    double crdeseff;
    double h2deseff;
    double nH;
    double zeta;
    double Tgas;
    double Av;
    double omega;
    double G0;
    double uvcreff;
    
    // clang-format on
};

#endif