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
    int Init(int nsystem = MAX_NSYSTEMS, double atol = 1e-20, double rtol = 1e-5, int mxsteps=500);
    int DebugInfo();
    int Finalize();
    /* */
    int Solve(realtype *ab, realtype dt, NaunetData *data);
#ifdef PYMODULE
    py::array_t<realtype> PyWrapSolve(py::array_t<realtype> arr, realtype dt,
                                      NaunetData *data);
#endif

   private:
    int n_system_;
    int mxsteps_;
    realtype atol_;
    realtype rtol_;

    /*  */

    N_Vector cv_y_;
    SUNMatrix cv_a_;
    void *cv_mem_;
    SUNLinearSolver cv_ls_;

    /*  */
};

#ifdef PYMODULE

PYBIND11_MODULE(PYMODNAME, m) {
    py::class_<Naunet>(m, "Naunet")
        .def(py::init())
        .def("Init", &Naunet::Init, py::arg("nsystem") = 1,
             py::arg("atol") = 1e-20, py::arg("rtol") = 1e-5,
             py::arg("mxsteps") = 500)
        .def("Finalize", &Naunet::Finalize)
#ifdef USE_CUDA
        .def("Reset", &Naunet::Reset, py::arg("nsystem") = 1,
             py::arg("atol") = 1e-20, py::arg("rtol") = 1e-5,
             py::arg("mxsteps") = 500)
#endif
        .def("Solve", &Naunet::PyWrapSolve);

    // clang-format off
    py::class_<NaunetData>(m, "NaunetData")
        .def(py::init())
        .def_readwrite("rG", &NaunetData::rG)
        .def_readwrite("gdens", &NaunetData::gdens)
        .def_readwrite("sites", &NaunetData::sites)
        .def_readwrite("fr", &NaunetData::fr)
        .def_readwrite("opt_thd", &NaunetData::opt_thd)
        .def_readwrite("opt_crd", &NaunetData::opt_crd)
        .def_readwrite("opt_h2d", &NaunetData::opt_h2d)
        .def_readwrite("opt_uvd", &NaunetData::opt_uvd)
        .def_readwrite("crdeseff", &NaunetData::crdeseff)
        .def_readwrite("h2deseff", &NaunetData::h2deseff)
        .def_readwrite("nH", &NaunetData::nH)
        .def_readwrite("zeta", &NaunetData::zeta)
        .def_readwrite("Tgas", &NaunetData::Tgas)
        .def_readwrite("Av", &NaunetData::Av)
        .def_readwrite("omega", &NaunetData::omega)
        .def_readwrite("G0", &NaunetData::G0)
        .def_readwrite("uvcreff", &NaunetData::uvcreff)
        ;
    // clang-format on
}

#endif

#endif