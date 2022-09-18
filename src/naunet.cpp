#include <cvode/cvode.h>  // prototypes for CVODE fcts., consts.
/* */
#include <nvector/nvector_serial.h>     // access to serial N_Vector
#include <sundials/sundials_dense.h>    // use generic dense solver in precond
#include <sunlinsol/sunlinsol_dense.h>  // access to dense SUNLinearSolver
#include <sunmatrix/sunmatrix_dense.h>  // access to dense SUNMatrix
/* */
#include "naunet.h"
#include "naunet_ode.h"
#include "naunet_physics.h"
#include "naunet_renorm.h"
/* */
#include "naunet_constants.h"
#include "naunet_utilities.h"
/* */

Naunet::Naunet(){};

Naunet::~Naunet(){};

// Adaptedfrom the cvDiurnals_ky.c example from the CVODE package.
// Check function return value...
//   opt == 0 means SUNDIALS function allocates memory so check if
//            returned NULL pointer
//   opt == 1 means SUNDIALS function returns a flag so check if
//            flag >= 0
//   opt == 2 means function allocates memory so check if returned
//            NULL pointer
int Naunet::CheckFlag(void *flagvalue, const char *funcname, int opt,
                      FILE *errf) {
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && flagvalue == NULL) {
        fprintf(errf,
                "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return NAUNET_FAIL;
    }

    /* Check if flag < 0 */
    else if (opt == 1) {
        errflag = (int *)flagvalue;
        if (*errflag < 0) {
            fprintf(errf, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                    funcname, *errflag);
            return NAUNET_FAIL;
        }
    }

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && flagvalue == NULL) {
        fprintf(errf, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return NAUNET_FAIL;
    }

    return NAUNET_SUCCESS;
};

int Naunet::Finalize() {
    /* */

    N_VDestroy(cv_y_);
    // N_VFreeEmpty(cv_y_);
    SUNMatDestroy(cv_a_);
    SUNLinSolFree(cv_ls_);
    // delete m_data;

    /*  */

    fclose(errfp_);

    return NAUNET_SUCCESS;
};

int Naunet::GetCVStates(void *cv_mem, long int &nst, long int &nfe,
                        long int &nsetups, long int &nje, long int &netf,
                        long int &nge, long int &nni, long int &ncfn) {
    int flag;

    flag = CVodeGetNumSteps(cv_mem, &nst);
    if (CheckFlag(&flag, "CVodeGetNumSteps", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumRhsEvals(cv_mem, &nfe);
    if (CheckFlag(&flag, "CVodeGetNumRhsEvals", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumLinSolvSetups(cv_mem, &nsetups);
    if (CheckFlag(&flag, "CVodeGetNumLinSolvSetups", 1, errfp_) ==
        NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumErrTestFails(cv_mem, &netf);
    if (CheckFlag(&flag, "CVodeGetNumErrTestFails", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumNonlinSolvIters(cv_mem, &nni);
    if (CheckFlag(&flag, "CVodeGetNumNonlinSolvIters", 1, errfp_) ==
        NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumNonlinSolvConvFails(cv_mem, &ncfn);
    if (CheckFlag(&flag, "CVodeGetNumNonlinSolvConvFails", 1, errfp_) ==
        NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumJacEvals(cv_mem, &nje);
    if (CheckFlag(&flag, "CVodeGetNumJacEvals", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumGEvals(cv_mem, &nge);
    if (CheckFlag(&flag, "CVodeGetNumGEvals", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    return NAUNET_SUCCESS;
};

int Naunet::HandleError(int cvflag, realtype *ab, realtype dt, realtype t0) {
    if (cvflag >= 0) {
        return NAUNET_SUCCESS;
    }

    fprintf(errfp_, "CVode failed in Naunet! Flag = %d\n", cvflag);
    fprintf(errfp_, "Calling HandleError to fix the problem\n");

    /* */

    realtype dt_init = dt;

    for (int level = 1; level < 6; level++) {
        int nsubsteps = 10 * level;

        if (cvflag < 0 && cvflag > -5) {
            for (int i = 0; i < NEQUATIONS; i++) {
                ab_tmp_[i] = ab[i];
            }
            dt -= t0;
        } else if (cvflag == -6) {
            // The state may have something wrong
            // Reset to the initial state and try finer steps
            for (int i = 0; i < NEQUATIONS; i++) {
                ab_tmp_[i] = ab_init_[i];
            }
            dt = dt_init;
        } else if (cvflag < 0) {
            fprintf(
                errfp_,
                "The error cannot be recovered by Naunet! Exit from Naunet!\n");
            fprintf(errfp_, "cvFlag = %d, level = %d\n", cvflag, level);
            return NAUNET_FAIL;
        }

        // Reset initial conditions
        t0 = 0.0;
        for (int i = 0; i < NEQUATIONS; i++) {
            ab[i] = ab_tmp_[i];
        }

        // Reinitialize
        cvflag = CVodeReInit(cv_mem_, t0, cv_y_);
        if (CheckFlag(&cvflag, "CVodeReInit", 1, errfp_) == NAUNET_FAIL) {
            return NAUNET_FAIL;
        }

        for (int step = 0; step < nsubsteps; step++) {
            realtype tout = pow(
                10.0, log10(dt) * (realtype)(step + 1) / (realtype)nsubsteps);
            // printf("tout: %13.7e, step: %d, level: %d\n", tout, step, level);
            // realtype tcur = 0.0;
            // cvflag = CVodeGetCurrentTime(cv_mem_, &tcur);
            cvflag = CVode(cv_mem_, tout, cv_y_, &t0, CV_NORMAL);
            if (cvflag < 0) {
                fprintf(errfp_,
                        "CVode failed in Naunet! Flag = %d in the %dth substep "
                        "of %dth level! \n",
                        cvflag, step, level);
                if (level < 5) {
                    fprintf(errfp_,
                            "Tyring to fix the error in the next level\n");
                }
                // fprintf(errfp_, "Failed to fix the error! cvflag = %d in the
                // %dth substep! \n", cvflag, i);
                break;
            }
        }

        // if CVode succeeded, leave the loop
        if (cvflag >= 0) {
            if (level > 0) {
                fprintf(errfp_,
                        "The error was successfully fixed in %dth level\n",
                        level);
            }
            // break;
            return NAUNET_SUCCESS;
        }
    }

    /* */

    return NAUNET_FAIL;
}

int Naunet::Init(int nsystem, double atol, double rtol, int mxsteps) {
    n_system_ = nsystem;
    mxsteps_  = mxsteps;
    atol_     = atol;
    rtol_     = rtol;
    errfp_    = fopen("naunet_error_record.txt", "a");

    /* */
    if (nsystem != 1) {
        printf("This solver doesn't support nsystem > 1!");
        return NAUNET_FAIL;
    }

    cv_y_  = N_VNewEmpty_Serial((sunindextype)NEQUATIONS);
    cv_a_  = SUNDenseMatrix(NEQUATIONS, NEQUATIONS);
    cv_ls_ = SUNLinSol_Dense(cv_y_, cv_a_);

    /*  */

    // reset the n_vector to empty, maybe not necessary
    /* */

    // N_VDestroy(cv_y_);
    // cv_y_ = N_VNewEmpty_Serial((sunindextype)NEQUATIONS);

    /* */

    /* */

    // TODO: The result is not saved
    // double *x1  = vector(1, 6);
    // double *x2  = vector(1, 7);
    // double **y  = matrix(1, 6, 1, 7);
    // double **y2 = matrix(1, 6, 1, 7);

    // for (int i=1; i<=6; i++) x1[i] = COShieldingTableX[i-1];
    // for (int i=1; i<=7; i++) x2[i] = COShieldingTableY[i-1];

    // for (int i=1; i<=6; i++) {
    //     for (int j=1; j<=7; j++) {
    //         y[i][j] = COShieldingTable[i-1][j-1];
    //     }
    // }

    // splie2(x1, x2, y, 6, 7, y2);

    // free_vector(x1, 1, 6);
    // free_vector(x2, 1, 7);
    // free_matrix(y, 1, 6, 1, 7);
    // free_matrix(y2, 1, 6, 1, 7);

    // splie2(COShieldingTableX, COShieldingTableY, COShieldingTable, 6, 7,
    // COShieldingTableD2);

    /* */

    return NAUNET_SUCCESS;
};

int Naunet::PrintDebugInfo() {
    long int nst, nfe, nsetups, nje, netf, nge, nni, ncfn;
    int flag;

    /* */

    if (GetCVStates(cv_mem_, nst, nfe, nsetups, nje, netf, nge, nni, ncfn) ==
        NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    printf("\nFinal Statistics:\n");
    printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld nje = %ld\n", nst, nfe,
           nsetups, nje);
    printf("nni = %-6ld ncfn = %-6ld netf = %-6ld    nge = %ld\n \n", nni, ncfn,
           netf, nge);

    /*  */

    return NAUNET_SUCCESS;
};

#ifdef IDX_ELEM_H
int Naunet::Renorm(realtype *ab) {
    N_Vector b  = N_VMake_Serial(NELEMENTS, ab_ref_);
    N_Vector r  = N_VNew_Serial(NELEMENTS);
    SUNMatrix A = SUNDenseMatrix(NELEMENTS, NELEMENTS);

    N_VConst(0.0, r);

    InitRenorm(ab, A);

    SUNLinearSolver LS = SUNLinSol_Dense(r, A);

    int flag;
    flag = SUNLinSolSetup(LS, A);
    if (CheckFlag(&flag, "SUNLinSolSetup", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }
    flag = SUNLinSolSolve(LS, A, r, b, 0.0);
    if (CheckFlag(&flag, "SUNLinSolSolve", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    realtype *rptr = N_VGetArrayPointer(r);

    RenormAbundance(rptr, ab);

    N_VDestroy(b);
    N_VDestroy(r);
    SUNMatDestroy(A);
    SUNLinSolFree(LS);

    return NAUNET_SUCCESS;
};
#endif

// To reset the size of cusparse solver
int Naunet::Reset(int nsystem, double atol, double rtol, int mxsteps) {
    n_system_ = nsystem;
    mxsteps_  = mxsteps;
    atol_     = atol;
    rtol_     = rtol;

    /* */
    if (nsystem != 1) {
        printf("This solver doesn't support nsystem > 1!");
        return NAUNET_FAIL;
    }

    // N_VFreeEmpty(cv_y_);
    N_VDestroy(cv_y_);
    SUNMatDestroy(cv_a_);
    SUNLinSolFree(cv_ls_);

    cv_y_  = N_VNewEmpty_Serial((sunindextype)NEQUATIONS);
    cv_a_  = SUNDenseMatrix(NEQUATIONS, NEQUATIONS);
    cv_ls_ = SUNLinSol_Dense(cv_y_, cv_a_);

    /*  */

    return NAUNET_SUCCESS;
};

#ifdef IDX_ELEM_H
int Naunet::SetReferenceAbund(realtype *ref, int opt) {
    if (opt == 0) {
        for (int i = 0; i < NELEMENTS; i++) {
            ab_ref_[i] = ref[i] / ref[IDX_ELEM_H];
        }
    } else if (opt == 1) {
        double Hnuclei = GetHNuclei(ref);
        for (int i = 0; i < NELEMENTS; i++) {
            ab_ref_[i] = GetElementAbund(ref, i) / Hnuclei;
        }
    }

    return NAUNET_SUCCESS;
}
#endif

int Naunet::Solve(realtype *ab, realtype dt, NaunetData *data) {
    /* */

    int cvflag;
    realtype t0 = 0.0;

    for (int i = 0; i < NEQUATIONS; i++) {
        ab_init_[i] = ab[i];
        ab_tmp_[i]  = ab[i];
    }

    // realtype *ydata = N_VGetArrayPointer(cv_y_);
    // for (int i=0; i<NEQUATIONS; i++)
    // {
    //     ydata[i] = ab[i];
    // }
    N_VSetArrayPointer(ab, cv_y_);

    cv_mem_ = CVodeCreate(CV_BDF);

    cvflag  = CVodeSetErrFile(cv_mem_, errfp_);
    if (CheckFlag(&cvflag, "CVodeSetErrFile", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag = CVodeSetMaxNumSteps(cv_mem_, mxsteps_);
    if (CheckFlag(&cvflag, "CVodeSetMaxNumSteps", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag = CVodeInit(cv_mem_, Fex, t0, cv_y_);
    if (CheckFlag(&cvflag, "CVodeInit", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag = CVodeSStolerances(cv_mem_, rtol_, atol_);
    if (CheckFlag(&cvflag, "CVodeSStolerances", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag = CVodeSetLinearSolver(cv_mem_, cv_ls_, cv_a_);
    if (CheckFlag(&cvflag, "CVodeSetLinearSolver", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag = CVodeSetJacFn(cv_mem_, Jac);
    if (CheckFlag(&cvflag, "CVodeSetJacFn", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag = CVodeSetUserData(cv_mem_, data);
    if (CheckFlag(&cvflag, "CVodeSetUserData", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag   = CVode(cv_mem_, dt, cv_y_, &t0, CV_NORMAL);

    // ab   = N_VGetArrayPointer(cv_y_);

    int flag = HandleError(cvflag, ab, dt, t0);

    if (flag == NAUNET_FAIL) {
        fprintf(errfp_, "Some unrecoverable error occurred. cvFlag = %d\n",
                cvflag);
        fprintf(errfp_, "Initial condition: \n");

        // clang-format off
        /* */
        /* */
        fprintf(errfp_, "    data.nH = %13.7e;\n", data->nH);
        /* */
        fprintf(errfp_, "    data.Tgas = %13.7e;\n", data->Tgas);
        /* */
        fprintf(errfp_, "    data.zeta = %13.7e;\n", data->zeta);
        /* */
        fprintf(errfp_, "    data.Av = %13.7e;\n", data->Av);
        /* */
        fprintf(errfp_, "    data.omega = %13.7e;\n", data->omega);
        /* */
        fprintf(errfp_, "    data.G0 = %13.7e;\n", data->G0);
        /* */
        fprintf(errfp_, "    data.gdens = %13.7e;\n", data->gdens);
        /* */
        fprintf(errfp_, "    data.rG = %13.7e;\n", data->rG);
        /* */
        fprintf(errfp_, "    data.sites = %13.7e;\n", data->sites);
        /* */
        fprintf(errfp_, "    data.fr = %13.7e;\n", data->fr);
        /* */
        fprintf(errfp_, "    data.opt_crd = %13.7e;\n", data->opt_crd);
        /* */
        fprintf(errfp_, "    data.opt_uvd = %13.7e;\n", data->opt_uvd);
        /* */
        fprintf(errfp_, "    data.opt_h2d = %13.7e;\n", data->opt_h2d);
        /* */
        fprintf(errfp_, "    data.eb_crd = %13.7e;\n", data->eb_crd);
        /* */
        fprintf(errfp_, "    data.eb_uvd = %13.7e;\n", data->eb_uvd);
        /* */
        fprintf(errfp_, "    data.eb_h2d = %13.7e;\n", data->eb_h2d);
        /* */
        fprintf(errfp_, "    data.crdeseff = %13.7e;\n", data->crdeseff);
        /* */
        fprintf(errfp_, "    data.uvcreff = %13.7e;\n", data->uvcreff);
        /* */
        fprintf(errfp_, "    data.h2deseff = %13.7e;\n", data->h2deseff);
        /* */
        fprintf(errfp_, "    data.opt_thd = %13.7e;\n", data->opt_thd);
        /* */
        fprintf(errfp_, "    data.ksp = %13.7e;\n", data->ksp);
        /*  */
        // clang-format on

        fprintf(errfp_, "\n");

        realtype spy = 365.0 * 86400.0;

        fprintf(errfp_, "    dtyr = %13.7e;\n", dt / spy);
        fprintf(errfp_, "\n");

        for (int i = 0; i < NEQUATIONS; i++) {
            fprintf(errfp_, "    y[%d] = %13.7e;\n", i, ab_init_[i]);
        }

        for (int i = 0; i < NEQUATIONS; i++) {
            fprintf(errfp_, "    y_final[%d] = %13.7e;\n", i, ab[i]);
        }
    }

    CVodeFree(&cv_mem_);

    return flag;

    /* */
};

#ifdef PYMODULE
#ifdef IDX_ELEM_H
py::array_t<realtype> Naunet::PyWrapRenorm(py::array_t<realtype> arr) {
    py::buffer_info info = arr.request();
    realtype *ab         = static_cast<realtype *>(info.ptr);

    int flag             = Renorm(ab);
    if (flag == NAUNET_FAIL) {
        throw std::runtime_error("Fail to renormalization");
    }

    return py::array_t<realtype>(info.shape, ab);
}
py::array_t<realtype> Naunet::PyWrapSetReferenceAbund(py::array_t<realtype> arr,
                                                      int opt) {
    py::buffer_info info = arr.request();
    realtype *ab         = static_cast<realtype *>(info.ptr);

    int flag             = SetReferenceAbund(ab, opt);
    if (flag == NAUNET_FAIL) {
        throw std::runtime_error("Fail to set reference abundance");
    }

    return py::array_t<realtype>(info.shape, ab);
}
#endif
py::array_t<realtype> Naunet::PyWrapSolve(py::array_t<realtype> arr,
                                          realtype dt, NaunetData *data) {
    py::buffer_info info = arr.request();
    realtype *ab         = static_cast<realtype *>(info.ptr);

    int flag             = Solve(ab, dt, data);
    if (flag == NAUNET_FAIL) {
        throw std::runtime_error("Something unrecoverable occurred");
    }

    return py::array_t<realtype>(info.shape, ab);
}
#endif
