cdef extern from "gsl_sf_pow_int.h":

  double  gsl_sf_pow_int(double x, int n)

  int  gsl_sf_pow_int_e(double x, int n, gsl_sf_result * result)
