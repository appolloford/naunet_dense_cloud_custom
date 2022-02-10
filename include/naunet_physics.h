#ifndef __NAUNET_PHYSICS_H__
#define __NAUNET_PHYSICS_H__

/*  */
double GetMantleDens(double *y);
double GetHNuclei(double *y);
double GetMu(double *y);
double GetGamma(double *y);
double GetNumDens(double *y);
double GetShieldingFactor(int specidx, double h2coldens, double spcoldens,
                          double tgas, int method);
double GetH2shielding(double coldens, int method);
double GetCOshielding(double tgas, double h2col, double coldens, int method);
double GetN2shielding(double tgas, double h2col, double coldens, int method);
double GetH2shieldingInt(double coldens);
double GetH2shieldingFGK(double coldens);
double GetCOshieldingInt(double tgas, double h2col, double coldens);
double GetCOshieldingInt1(double h2col, double coldens);
double GetN2shieldingInt(double tgas, double h2col, double coldens);
double GetCharactWavelength(double h2col, double cocol);
double GetGrainScattering(double av, double wavelength);
/*  */

#endif