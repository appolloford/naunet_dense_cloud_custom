#ifndef __NAUNET_H__
#define __NAUNET_H__

#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_math.h>   // contains the macros ABS, SUNSQR, EXP
#include <sundials/sundials_types.h>  // defs. of realtype, sunindextype
/* */

#include "naunet_data.h"
#include "naunet_macros.h"

#ifdef PYMODULE
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
#endif

class Naunet {
   public:
    Naunet();
    ~Naunet();
    int Finalize();
    int Init(int nsystem = MAX_NSYSTEMS, double atol = 1e-20,
             double rtol = 1e-5, int mxsteps = 500);
    int PrintDebugInfo();
#ifdef IDX_ELEM_H
    // Renormalize the species abundances accroding to `ab_ref_`(private).
    // This conserves the element fractions but does not change the overall
    // density/
    int Renorm(realtype *ab);
#endif
    int Reset(int nsystem = MAX_NSYSTEMS, double atol = 1e-20,
              double rtol = 1e-5, int mxsteps = 500);
#ifdef IDX_ELEM_H
    // Set the reference abundance `ab_ref_`. `opt == 0` assumes that the input
    // is element abundances. `opt == 1` assumes the input is species
    // abundances.
    int SetReferenceAbund(realtype *ref, int opt = 0);
#endif
    int Solve(realtype *ab, realtype dt, NaunetData *data);
#ifdef PYMODULE
#ifdef IDX_ELEM_H
    py::array_t<realtype> PyWrapRenorm(py::array_t<realtype> arr);
    py::array_t<realtype> PyWrapSetReferenceAbund(py::array_t<realtype> arr,
                                                  int opt);
#endif
    py::array_t<realtype> PyWrapSolve(py::array_t<realtype> arr, realtype dt,
                                      NaunetData *data);
#endif

   private:
    int n_system_;
    int mxsteps_;
    realtype atol_;
    realtype rtol_;
    FILE *errfp_;
    realtype ab_ref_[NELEMENTS];

    /*  */

    N_Vector cv_y_;
    SUNMatrix cv_a_;
    void *cv_mem_;
    SUNLinearSolver cv_ls_;

    realtype ab_init_[NEQUATIONS];
    realtype ab_tmp_[NEQUATIONS];  // Temporary state for error handling

    /*  */

    int GetCVStates(void *cv_mem, long int &nst, long int &nfe,
                    long int &nsetups, long int &nje, long int &netf,
                    long int &nge, long int &nni, long int &ncfn);
    int HandleError(int flag, realtype *ab, realtype dt, realtype t0);
    static int CheckFlag(void *flagvalue, const char *funcname, int opt,
                         FILE *errf);
};

#ifdef PYMODULE

PYBIND11_MODULE(PYMODNAME, m) {
    py::class_<Naunet>(m, "Naunet")
        .def(py::init())
        .def("Init", &Naunet::Init, py::arg("nsystem") = 1,
             py::arg("atol") = 1e-20, py::arg("rtol") = 1e-5,
             py::arg("mxsteps") = 500)
        .def("Finalize", &Naunet::Finalize)
#ifdef IDX_ELEM_H
        .def("Renorm", &Naunet::PyWrapRenorm)
#endif
        .def("Reset", &Naunet::Reset, py::arg("nsystem") = 1,
             py::arg("atol") = 1e-20, py::arg("rtol") = 1e-5,
             py::arg("mxsteps") = 500)
#ifdef IDX_ELEM_H
        .def("SetReferenceAbund", &Naunet::PyWrapSetReferenceAbund,
             py::arg("ref"), py::arg("opt") = 0)
#endif
        .def("Solve", &Naunet::PyWrapSolve);

    // clang-format off
    py::class_<NaunetData>(m, "NaunetData")
        .def(py::init())
        .def_readwrite("nH", &NaunetData::nH)
        .def_readwrite("Tgas", &NaunetData::Tgas)
        .def_readwrite("zeta", &NaunetData::zeta)
        .def_readwrite("Av", &NaunetData::Av)
        .def_readwrite("omega", &NaunetData::omega)
        .def_readwrite("G0", &NaunetData::G0)
        .def_readwrite("gdens", &NaunetData::gdens)
        .def_readwrite("rG", &NaunetData::rG)
        .def_readwrite("sites", &NaunetData::sites)
        .def_readwrite("fr", &NaunetData::fr)
        .def_readwrite("opt_crd", &NaunetData::opt_crd)
        .def_readwrite("opt_uvd", &NaunetData::opt_uvd)
        .def_readwrite("opt_h2d", &NaunetData::opt_h2d)
        .def_readwrite("eb_crd", &NaunetData::eb_crd)
        .def_readwrite("eb_uvd", &NaunetData::eb_uvd)
        .def_readwrite("eb_h2d", &NaunetData::eb_h2d)
        .def_readwrite("crdeseff", &NaunetData::crdeseff)
        .def_readwrite("uvcreff", &NaunetData::uvcreff)
        .def_readwrite("h2deseff", &NaunetData::h2deseff)
        .def_readwrite("opt_thd", &NaunetData::opt_thd)
        .def_readwrite("ksp", &NaunetData::ksp)
        ;
    // clang-format on
}

#endif

#endif