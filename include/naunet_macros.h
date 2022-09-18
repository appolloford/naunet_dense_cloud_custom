#ifndef __NAUNET_MACROS_H__
#define __NAUNET_MACROS_H__

#include <sunmatrix/sunmatrix_dense.h>

// 
// clang-format off
#define NAUNET_SUCCESS 0
#define NAUNET_FAIL 1

#define MAX_NSYSTEMS 1

#define NELEMENTS 7
#define NSPECIES 114
#define NHEATPROCS 0
#define NCOOLPROCS 0
#define THERMAL (NHEATPROCS || NCOOLPROCS)
#if (NSPECIES + THERMAL)
#define NEQUATIONS (NSPECIES + THERMAL)
#else
#define NEQUATIONS 1
#endif
#define NREACTIONS 1403
// non-zero terms in jacobian matrix, used in sparse matrix
#define NNZ 3028

#define IDX_ELEM_MG 0
#define IDX_ELEM_SI 1
#define IDX_ELEM_HE 2
#define IDX_ELEM_N 3
#define IDX_ELEM_C 4
#define IDX_ELEM_O 5
#define IDX_ELEM_H 6

#define IDX_GH2CNI 0
#define IDX_GHNCI 1
#define IDX_GNO2I 2
#define IDX_GSiOI 3
#define IDX_GCOI 4
#define IDX_GHNCOI 5
#define IDX_GMgI 6
#define IDX_GNOI 7
#define IDX_GO2I 8
#define IDX_GO2HI 9
#define IDX_GSiCI 10
#define IDX_GSiC2I 11
#define IDX_GSiC3I 12
#define IDX_GCH3OHI 13
#define IDX_GCO2I 14
#define IDX_GH2SiOI 15
#define IDX_GHNOI 16
#define IDX_GN2I 17
#define IDX_GH2COI 18
#define IDX_GHCNI 19
#define IDX_GH2OI 20
#define IDX_GNH3I 21
#define IDX_SiC3II 22
#define IDX_H2CNI 23
#define IDX_GCH4I 24
#define IDX_H2NOII 25
#define IDX_H2SiOI 26
#define IDX_HeHII 27
#define IDX_HNCOI 28
#define IDX_HOCII 29
#define IDX_SiC2II 30
#define IDX_GSiH4I 31
#define IDX_SiC2I 32
#define IDX_SiC3I 33
#define IDX_SiH5II 34
#define IDX_SiH4II 35
#define IDX_SiCII 36
#define IDX_O2HI 37
#define IDX_SiCI 38
#define IDX_NO2I 39
#define IDX_SiH3II 40
#define IDX_SiH2II 41
#define IDX_OCNI 42
#define IDX_SiH2I 43
#define IDX_SiOHII 44
#define IDX_SiHII 45
#define IDX_SiH4I 46
#define IDX_SiHI 47
#define IDX_SiH3I 48
#define IDX_SiOII 49
#define IDX_HCO2II 50
#define IDX_HNOI 51
#define IDX_CH3OHI 52
#define IDX_MgI 53
#define IDX_MgII 54
#define IDX_CH4II 55
#define IDX_SiOI 56
#define IDX_CNII 57
#define IDX_HCNHII 58
#define IDX_N2HII 59
#define IDX_O2HII 60
#define IDX_SiII 61
#define IDX_SiI 62
#define IDX_HNCI 63
#define IDX_HNOII 64
#define IDX_N2II 65
#define IDX_H3COII 66
#define IDX_CH4I 67
#define IDX_COII 68
#define IDX_H2II 69
#define IDX_NH3I 70
#define IDX_CH3I 71
#define IDX_CO2I 72
#define IDX_NII 73
#define IDX_OII 74
#define IDX_HCNII 75
#define IDX_NH2II 76
#define IDX_NHII 77
#define IDX_O2II 78
#define IDX_CH3II 79
#define IDX_NH2I 80
#define IDX_CH2II 81
#define IDX_H2OII 82
#define IDX_NH3II 83
#define IDX_NOII 84
#define IDX_H3OII 85
#define IDX_N2I 86
#define IDX_CII 87
#define IDX_HCNI 88
#define IDX_CHII 89
#define IDX_CH2I 90
#define IDX_H2COII 91
#define IDX_NHI 92
#define IDX_OHII 93
#define IDX_CNI 94
#define IDX_H2COI 95
#define IDX_HCOI 96
#define IDX_HeII 97
#define IDX_CHI 98
#define IDX_H3II 99
#define IDX_HeI 100
#define IDX_NOI 101
#define IDX_NI 102
#define IDX_OHI 103
#define IDX_O2I 104
#define IDX_CI 105
#define IDX_HII 106
#define IDX_HCOII 107
#define IDX_H2OI 108
#define IDX_OI 109
#define IDX_EM 110
#define IDX_COI 111
#define IDX_H2I 112
#define IDX_HI 113

#if THERMAL
#define IDX_TGAS NSPECIES
#endif
#define IJth(A, i, j) SM_ELEMENT_D(A, i, j)

#endif