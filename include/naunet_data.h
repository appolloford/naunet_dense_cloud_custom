#ifndef __NAUNET_DATA_H__
#define __NAUNET_DATA_H__

// 
// Struct for holding the nessesary additional variables for the problem.
struct NaunetData {
    // clang-format off
    double nH;
    double Tgas;
    double zeta = 1.300e-17;
    double Av = 1.000e+00;
    double omega = 5.000e-01;
    double G0 = 1.000e+00;
    double gdens = 7.639e-13;
    double rG = 1.000e-05;
    double sites = 1.500e+15;
    double fr = 1.000e+00;
    double opt_crd = 1.000e+00;
    double opt_uvd = 1.000e+00;
    double opt_h2d = 1.000e+00;
    double eb_crd = 1.210e+03;
    double eb_uvd = 1.000e+04;
    double eb_h2d = 1.210e+03;
    double crdeseff = 1.000e+05;
    double uvcreff = 1.000e-03;
    double h2deseff = 1.000e-02;
    double opt_thd = 1.000e+00;
    double ksp = 0.000e+00;
    
    // clang-format on
};
#endif