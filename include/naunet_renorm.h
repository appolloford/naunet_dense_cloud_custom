#ifndef __NAUNET_RENORM_H__
#define __NAUNET_RENORM_H__

#include <sundials/sundials_matrix.h>

// clang-format off
int InitRenorm(realtype *ab, SUNMatrix A);
int RenormAbundance(realtype *rptr, realtype *ab);

#endif