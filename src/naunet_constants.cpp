// clang-format off
#include "naunet_constants.h"

const double pi              = 3.1415926;

// atomic mass unit (g)
const double amu             = 1.6605402e-24;

// mass of electron (g)
const double me              = 9.1093897e-28;

// mass or electron (u)
const double meu             = 5.48579909e-4;

// mass of proton (g)
const double mp              = 1.6726231e-24;

// mass of neutron (g)
const double mn              = 1.6749286e-24;

// mass of hydrogen (g)
const double mh              = 1.6733e-24;

// electron charge (esu)
const double echarge         = 4.80320425e-10;

// Boltzmann constant (erg/K)
const double kerg            = 1.380658e-16;

const double zism            = 1.3e-17;
const double nmono           = 2.0;
const double eb_GCH3OHI      = 5000.0;
const double eb_GCH4I        = 960.0;
const double eb_GCOI         = 1300.0;
const double eb_GCO2I        = 2600.0;
const double eb_GH2CNI       = 2400.0;
const double eb_GH2COI       = 4500.0;
const double eb_GH2OI        = 5600.0;
const double eb_GH2SiOI      = 4400.0;
const double eb_GHCNI        = 3700.0;
const double eb_GHNCI        = 3800.0;
const double eb_GHNCOI       = 4400.0;
const double eb_GHNOI        = 3000.0;
const double eb_GMgI         = 5300.0;
const double eb_GN2I         = 1100.0;
const double eb_GNH3I        = 5500.0;
const double eb_GNOI         = 1600.0;
const double eb_GNO2I        = 2400.0;
const double eb_GO2I         = 1200.0;
const double eb_GO2HI        = 5000.0;
const double eb_GSiCI        = 3500.0;
const double eb_GSiC2I       = 4300.0;
const double eb_GSiC3I       = 5100.0;
const double eb_GSiH4I       = 13000.0;
const double eb_GSiOI        = 3500.0;






// Table of CO self-shielding factor
// H2 column density
const
double COShieldingTableX[6] = 
{18.0, 19.0, 20.0, 21.0, 22.0, 23.0};

// CO column density
const
double COShieldingTableY[7] = 
{12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0};

const
double COShieldingTable[6][7] = 
{{ 0.000e+00, -1.408e-02, -1.099e-01, -4.400e-01, -1.154e+00, -1.888e+00, -2.760e+00},
 {-8.539e-02, -1.015e-01, -2.104e-01, -5.608e-01, -1.272e+00, -1.973e+00, -2.818e+00},
 {-1.451e-01, -1.612e-01, -2.708e-01, -6.273e-01, -1.355e+00, -2.057e+00, -2.902e+00},
 {-4.559e-01, -4.666e-01, -5.432e-01, -8.665e-01, -1.602e+00, -2.303e+00, -3.146e+00},
 {-1.303e+00, -1.312e+00, -1.367e+00, -1.676e+00, -2.305e+00, -3.034e+00, -3.758e+00},
 {-3.883e+00, -3.888e+00, -3.936e+00, -4.197e+00, -4.739e+00, -5.165e+00, -5.441e+00}};
// {{ 0.     ,-0.08539,-0.1451 ,-0.4559 ,-1.303  ,-3.883  },
//  {-0.01408,-0.1015 ,-0.1612 ,-0.4666 ,-1.312  ,-3.888  },
//  {-0.1099 ,-0.2104 ,-0.2708 ,-0.5432 ,-1.367  ,-3.936  },
//  {-0.44   ,-0.5608 ,-0.6273 ,-0.8665 ,-1.676  ,-4.197  },
//  {-1.154  ,-1.272  ,-1.355  ,-1.602  ,-2.305  ,-4.739  },
//  {-1.888  ,-1.973  ,-2.057  ,-2.303  ,-3.034  ,-5.165  },
//  {-2.76   ,-2.818  ,-2.902  ,-3.146  ,-3.758  ,-5.441  }};

