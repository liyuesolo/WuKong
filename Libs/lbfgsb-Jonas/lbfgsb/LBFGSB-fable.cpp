#include <fem.hpp> // Fortran EMulation library of fable module

namespace lbfgsb {

using namespace fem::major_types;

double
epsilon(...)
{
	return std::numeric_limits<double>::epsilon();
}

using fem::common;

//C
//C======================= The end of mainlb =============================
//C
void
active(
  common& cmn,
  int const& n,
  arr_cref<double> l,
  arr_cref<double> u,
  arr_cref<int> nbd,
  arr_ref<double> x,
  arr_ref<int> iwhere,
  int const& iprint,
  bool& prjctd,
  bool& cnstnd,
  bool& boxed)
{
  l(dimension(n));
  u(dimension(n));
  nbd(dimension(n));
  x(dimension(n));
  iwhere(dimension(n));
  common_write write(cmn);
  //C
  //C     ************
  //C
  //C     Subroutine active
  //C
  //C     This subroutine initializes iwhere and projects the initial x to
  //C       the feasible set if necessary.
  //C
  //C     iwhere is an integer array of dimension n.
  //C       On entry iwhere is unspecified.
  //C       On exit iwhere(i)=-1  if x(i) has no bounds
  //C                         3   if l(i)=u(i)
  //C                         0   otherwise.
  //C       In cauchy, iwhere is given finer gradations.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  //C     Initialize nbdd, prjctd, cnstnd and boxed.
  //C
  int nbdd = 0;
  prjctd = false;
  cnstnd = false;
  boxed = true;
  //C
  //C     Project the initial x to the easible set if necessary.
  //C
  int i = fem::int0;
  FEM_DO_SAFE(i, 1, n) {
    if (nbd(i) > 0) {
      if (nbd(i) <= 2 && x(i) <= l(i)) {
        if (x(i) < l(i)) {
          prjctd = true;
          x(i) = l(i);
        }
        nbdd++;
      }
      else if (nbd(i) >= 2 && x(i) >= u(i)) {
        if (x(i) > u(i)) {
          prjctd = true;
          x(i) = u(i);
        }
        nbdd++;
      }
    }
  }
  //C
  //C     Initialize iwhere and assign values to cnstnd and boxed.
  //C
  const double zero = 0.0e0;
  FEM_DO_SAFE(i, 1, n) {
    if (nbd(i) != 2) {
      boxed = false;
    }
    if (nbd(i) == 0) {
      //C                                this variable is always free
      iwhere(i) = -1;
      //C
      //C           otherwise set x(i)=mid(x(i), u(i), l(i)).
    }
    else {
      cnstnd = true;
      if (nbd(i) == 2 && u(i) - l(i) <= zero) {
        //C                   this variable is always fixed
        iwhere(i) = 3;
      }
      else {
        iwhere(i) = 0;
      }
    }
  }
  //C
  if (iprint >= 0) {
    if (prjctd) {
      write(6, star),
        "The initial X is infeasible.  Restart with its projection.";
    }
    if (!cnstnd) {
      write(6, star), "This problem is unconstrained.";
    }
  }
  //C
  if (iprint > 0) {
    write(6, "(/,'At X0 ',i9,' variables are exactly at the bounds')"), nbdd;
  }
  //C
}

//C
//C====================== The end of dnrm2 ===============================
//C
void
daxpy(
  int const& n,
  double const& da,
  arr_cref<double> dx,
  int const& incx,
  arr_ref<double> dy,
  int const& incy)
{
  dx(dimension(star));
  dy(dimension(star));
  int ix = fem::int0;
  int iy = fem::int0;
  int i = fem::int0;
  int m = fem::int0;
  int mp1 = fem::int0;
  //C
  //C     constant times a vector plus a vector.
  //C     uses unrolled loops for increments equal to one.
  //C     jack dongarra, linpack, 3/11/78.
  //C
  if (n <= 0) {
    return;
  }
  if (da == 0.0e0) {
    return;
  }
  if (incx == 1 && incy == 1) {
    goto statement_20;
  }
  //C
  //C        code for unequal increments or equal increments
  //C          not equal to 1
  //C
  ix = 1;
  iy = 1;
  if (incx < 0) {
    ix = (-n + 1) * incx + 1;
  }
  if (incy < 0) {
    iy = (-n + 1) * incy + 1;
  }
  FEM_DO_SAFE(i, 1, n) {
    dy(iy) += da * dx(ix);
    ix += incx;
    iy += incy;
  }
  return;
  //C
  //C        code for both increments equal to 1
  //C
  //C        clean-up loop
  //C
  statement_20:
  m = fem::mod(n, 4);
  if (m == 0) {
    goto statement_40;
  }
  FEM_DO_SAFE(i, 1, m) {
    dy(i) += da * dx(i);
  }
  if (n < 4) {
    return;
  }
  statement_40:
  mp1 = m + 1;
  FEM_DOSTEP(i, mp1, n, 4) {
    dy(i) += da * dx(i);
    dy(i + 1) += da * dx(i + 1);
    dy(i + 2) += da * dx(i + 2);
    dy(i + 3) += da * dx(i + 3);
  }
}

//C
//C====================== The end of dcopy ===============================
//C
double
ddot(
  int const& n,
  arr_cref<double> dx,
  int const& incx,
  arr_cref<double> dy,
  int const& incy)
{
  double return_value = fem::double0;
  dx(dimension(star));
  dy(dimension(star));
  double dtemp = fem::double0;
  int ix = fem::int0;
  int iy = fem::int0;
  int i = fem::int0;
  int m = fem::int0;
  int mp1 = fem::int0;
  //C
  //C     forms the dot product of two vectors.
  //C     uses unrolled loops for increments equal to one.
  //C     jack dongarra, linpack, 3/11/78.
  //C
  return_value = 0.0e0;
  dtemp = 0.0e0;
  if (n <= 0) {
    return return_value;
  }
  if (incx == 1 && incy == 1) {
    goto statement_20;
  }
  //C
  //C        code for unequal increments or equal increments
  //C          not equal to 1
  //C
  ix = 1;
  iy = 1;
  if (incx < 0) {
    ix = (-n + 1) * incx + 1;
  }
  if (incy < 0) {
    iy = (-n + 1) * incy + 1;
  }
  FEM_DO_SAFE(i, 1, n) {
    dtemp += dx(ix) * dy(iy);
    ix += incx;
    iy += incy;
  }
  return_value = dtemp;
  return return_value;
  //C
  //C        code for both increments equal to 1
  //C
  //C        clean-up loop
  //C
  statement_20:
  m = fem::mod(n, 5);
  if (m == 0) {
    goto statement_40;
  }
  FEM_DO_SAFE(i, 1, m) {
    dtemp += dx(i) * dy(i);
  }
  if (n < 5) {
    goto statement_60;
  }
  statement_40:
  mp1 = m + 1;
  FEM_DOSTEP(i, mp1, n, 5) {
    dtemp += dx(i) * dy(i) + dx(i + 1) * dy(i + 1) + dx(i + 2) * dy(
      i + 2) + dx(i + 3) * dy(i + 3) + dx(i + 4) * dy(i + 4);
  }
  statement_60:
  return_value = dtemp;
  return return_value;
}

//C
//C====================== The end of dpofa ===============================
//C
void
dtrsl(
  arr_cref<double, 2> t,
  int const& ldt,
  int const& n,
  arr_ref<double> b,
  int const& job,
  int& info)
{
  t(dimension(ldt, star));
  b(dimension(star));
  int identifier_case = fem::int0;
  int j = fem::int0;
  double temp = fem::double0;
  int jj = fem::int0;
  //C
  //C     dtrsl solves systems of the form
  //C
  //C                   t * x = b
  //C     or
  //C                   trans(t) * x = b
  //C
  //C     where t is a triangular matrix of order n. here trans(t)
  //C     denotes the transpose of the matrix t.
  //C
  //C     on entry
  //C
  //C         t         double precision(ldt,n)
  //C                   t contains the matrix of the system. the zero
  //C                   elements of the matrix are not referenced, and
  //C                   the corresponding elements of the array can be
  //C                   used to store other information.
  //C
  //C         ldt       integer
  //C                   ldt is the leading dimension of the array t.
  //C
  //C         n         integer
  //C                   n is the order of the system.
  //C
  //C         b         double precision(n).
  //C                   b contains the right hand side of the system.
  //C
  //C         job       integer
  //C                   job specifies what kind of system is to be solved.
  //C                   if job is
  //C
  //C                        00   solve t*x=b, t lower triangular,
  //C                        01   solve t*x=b, t upper triangular,
  //C                        10   solve trans(t)*x=b, t lower triangular,
  //C                        11   solve trans(t)*x=b, t upper triangular.
  //C
  //C     on return
  //C
  //C         b         b contains the solution, if info .eq. 0.
  //C                   otherwise b is unaltered.
  //C
  //C         info      integer
  //C                   info contains zero if the system is nonsingular.
  //C                   otherwise info contains the index of
  //C                   the first zero diagonal element of t.
  //C
  //C     linpack. this version dated 08/14/78 .
  //C     g. w. stewart, university of maryland, argonne national lab.
  //C
  //C     subroutines and functions
  //C
  //C     blas daxpy,ddot
  //C     fortran mod
  //C
  //C     internal variables
  //C
  //C     begin block permitting ...exits to 150
  //C
  //C        check for zero diagonal elements.
  //C
  FEM_DO_SAFE(info, 1, n) {
    //C     ......exit
    if (t(info, info) == 0.0e0) {
      goto statement_150;
    }
  }
  info = 0;
  //C
  //C        determine the task and go to it.
  //C
  identifier_case = 1;
  if (fem::mod(job, 10) != 0) {
    identifier_case = 2;
  }
  if (fem::mod(job, 100) / 10 != 0) {
    identifier_case += 2;
  }
  switch (identifier_case) {
    case 1: goto statement_20;
    case 2: goto statement_50;
    case 3: goto statement_80;
    case 4: goto statement_110;
    default: break;
  }
  //C
  //C        solve t*x=b for t lower triangular
  //C
  statement_20:
  b(1) = b(1) / t(1, 1);
  if (n < 2) {
    goto statement_40;
  }
  FEM_DO_SAFE(j, 2, n) {
    temp = -b(j - 1);
    daxpy(n - j + 1, temp, t(j, j - 1), 1, b(j), 1);
    b(j) = b(j) / t(j, j);
  }
  statement_40:
  goto statement_140;
  //C
  //C        solve t*x=b for t upper triangular.
  //C
  statement_50:
  b(n) = b(n) / t(n, n);
  if (n < 2) {
    goto statement_70;
  }
  FEM_DO_SAFE(jj, 2, n) {
    j = n - jj + 1;
    temp = -b(j + 1);
    daxpy(j, temp, t(1, j + 1), 1, b(1), 1);
    b(j) = b(j) / t(j, j);
  }
  statement_70:
  goto statement_140;
  //C
  //C        solve trans(t)*x=b for t lower triangular.
  //C
  statement_80:
  b(n) = b(n) / t(n, n);
  if (n < 2) {
    goto statement_100;
  }
  FEM_DO_SAFE(jj, 2, n) {
    j = n - jj + 1;
    b(j) = b(j) - ddot(jj - 1, t(j + 1, j), 1, b(j + 1), 1);
    b(j) = b(j) / t(j, j);
  }
  statement_100:
  goto statement_140;
  //C
  //C        solve trans(t)*x=b for t upper triangular.
  //C
  statement_110:
  b(1) = b(1) / t(1, 1);
  if (n < 2) {
    goto statement_130;
  }
  FEM_DO_SAFE(j, 2, n) {
    b(j) = b(j) - ddot(j - 1, t(1, j), 1, b(1), 1);
    b(j) = b(j) / t(j, j);
  }
  statement_130:
  statement_140:
  statement_150:;
}
//C
//C====================== The end of dtrsl ===============================
//C

//C
//C======================= The end of active =============================
//C
void
bmv(
  int const& m,
  arr_cref<double, 2> sy,
  arr_cref<double, 2> wt,
  int const& col,
  arr_cref<double> v,
  arr_ref<double> p,
  int& info)
{
  sy(dimension(m, m));
  wt(dimension(m, m));
  v(dimension(2 * col));
  p(dimension(2 * col));
  //C
  //C     ************
  //C
  //C     Subroutine bmv
  //C
  //C     This subroutine computes the product of the 2m x 2m middle matrix
  //C       in the compact L-BFGS formula of B and a 2m vector v;
  //C       it returns the product in p.
  //C
  //C     m is an integer variable.
  //C       On entry m is the maximum number of variable metric corrections
  //C         used to define the limited memory matrix.
  //C       On exit m is unchanged.
  //C
  //C     sy is a double precision array of dimension m x m.
  //C       On entry sy specifies the matrix S'Y.
  //C       On exit sy is unchanged.
  //C
  //C     wt is a double precision array of dimension m x m.
  //C       On entry wt specifies the upper triangular matrix J' which is
  //C         the Cholesky factor of (thetaS'S+LD^(-1)L').
  //C       On exit wt is unchanged.
  //C
  //C     col is an integer variable.
  //C       On entry col specifies the number of s-vectors (or y-vectors)
  //C         stored in the compact L-BFGS formula.
  //C       On exit col is unchanged.
  //C
  //C     v is a double precision array of dimension 2col.
  //C       On entry v specifies vector v.
  //C       On exit v is unchanged.
  //C
  //C     p is a double precision array of dimension 2col.
  //C       On entry p is unspecified.
  //C       On exit p is the product Mv.
  //C
  //C     info is an integer variable.
  //C       On entry info is unspecified.
  //C       On exit info = 0       for normal return,
  //C                    = nonzero for abnormal return when the system
  //C                                to be solved by dtrsl is singular.
  //C
  //C     Subprograms called:
  //C
  //C       Linpack ... dtrsl.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  if (col == 0) {
    return;
  }
  //C
  //C     PART I: solve [  D^(1/2)      O ] [ p1 ] = [ v1 ]
  //C                   [ -L*D^(-1/2)   J ] [ p2 ]   [ v2 ].
  //C
  //C       solve Jp2=v2+LD^(-1)v1.
  p(col + 1) = v(col + 1);
  int i = fem::int0;
  int i2 = fem::int0;
  double sum = fem::double0;
  int k = fem::int0;
  FEM_DO_SAFE(i, 2, col) {
    i2 = col + i;
    sum = 0.0e0;
    FEM_DO_SAFE(k, 1, i - 1) {
      sum += sy(i, k) * v(k) / sy(k, k);
    }
    p(i2) = v(i2) + sum;
  }
  //C     Solve the triangular system
  dtrsl(wt, m, col, p(col + 1), 11, info);
  if (info != 0) {
    return;
  }
  //C
  //C       solve D^(1/2)p1=v1.
  FEM_DO_SAFE(i, 1, col) {
    p(i) = v(i) / fem::sqrt(sy(i, i));
  }
  //C
  //C     PART II: solve [ -D^(1/2)   D^(-1/2)*L'  ] [ p1 ] = [ p1 ]
  //C                    [  0         J'           ] [ p2 ]   [ p2 ].
  //C
  //C       solve J^Tp2=p2.
  dtrsl(wt, m, col, p(col + 1), 01, info);
  if (info != 0) {
    return;
  }
  //C
  //C       compute p1=-D^(-1/2)(p1-D^(-1/2)L'p2)
  //C                 =-D^(-1/2)p1+D^(-1)L'p2.
  FEM_DO_SAFE(i, 1, col) {
    p(i) = -p(i) / fem::sqrt(sy(i, i));
  }
  FEM_DO_SAFE(i, 1, col) {
    sum = 0.e0;
    FEM_DO_SAFE(k, i + 1, col) {
      sum += sy(k, i) * p(col + k) / sy(i, i);
    }
    p(i) += sum;
  }
  //C
}

//C
//C======================= The end of freev ==============================
//C
void
hpsolb(
  int const& n,
  arr_ref<double> t,
  arr_ref<int> iorder,
  int const& iheap)
{
  t(dimension(n));
  iorder(dimension(n));
  int k = fem::int0;
  double ddum = fem::double0;
  int indxin = fem::int0;
  int i = fem::int0;
  int j = fem::int0;
  double out = fem::double0;
  int indxou = fem::int0;
  //C
  //C     ************
  //C
  //C     Subroutine hpsolb
  //C
  //C     This subroutine sorts out the least element of t, and puts the
  //C       remaining elements of t in a heap.
  //C
  //C     n is an integer variable.
  //C       On entry n is the dimension of the arrays t and iorder.
  //C       On exit n is unchanged.
  //C
  //C     t is a double precision array of dimension n.
  //C       On entry t stores the elements to be sorted,
  //C       On exit t(n) stores the least elements of t, and t(1) to t(n-1)
  //C         stores the remaining elements in the form of a heap.
  //C
  //C     iorder is an integer array of dimension n.
  //C       On entry iorder(i) is the index of t(i).
  //C       On exit iorder(i) is still the index of t(i), but iorder may be
  //C         permuted in accordance with t.
  //C
  //C     iheap is an integer variable specifying the task.
  //C       On entry iheap should be set as follows:
  //C         iheap .eq. 0 if t(1) to t(n) is not in the form of a heap,
  //C         iheap .ne. 0 if otherwise.
  //C       On exit iheap is unchanged.
  //C
  //C     References:
  //C       Algorithm 232 of CACM (J. W. J. Williams): HEAPSORT.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  if (iheap == 0) {
    //C
    //C        Rearrange the elements t(1) to t(n) to form a heap.
    //C
    FEM_DO_SAFE(k, 2, n) {
      ddum = t(k);
      indxin = iorder(k);
      //C
      //C           Add ddum to the heap.
      i = k;
      statement_10:
      if (i > 1) {
        j = i / 2;
        if (ddum < t(j)) {
          t(i) = t(j);
          iorder(i) = iorder(j);
          i = j;
          goto statement_10;
        }
      }
      t(i) = ddum;
      iorder(i) = indxin;
    }
  }
  //C
  //C     Assign to 'out' the value of t(1), the least member of the heap,
  //C        and rearrange the remaining members to form a heap as
  //C        elements 1 to n-1 of t.
  //C
  if (n > 1) {
    i = 1;
    out = t(1);
    indxou = iorder(1);
    ddum = t(n);
    indxin = iorder(n);
    //C
    //C        Restore the heap
    statement_30:
    j = i + i;
    if (j <= n - 1) {
      if (t(j + 1) < t(j)) {
        j++;
      }
      if (t(j) < ddum) {
        t(i) = t(j);
        iorder(i) = iorder(j);
        i = j;
        goto statement_30;
      }
    }
    t(i) = ddum;
    iorder(i) = indxin;
    //C
    //C     Put the least member in t(n).
    //C
    t(n) = out;
    iorder(n) = indxou;
  }
  //C
}

//C
//C====================== The end of daxpy ===============================
//C
void
dcopy(
  int const& n,
  arr_cref<double> dx,
  int const& incx,
  arr_ref<double> dy,
  int const& incy)
{
  dx(dimension(star));
  dy(dimension(star));
  int ix = fem::int0;
  int iy = fem::int0;
  int i = fem::int0;
  int m = fem::int0;
  int mp1 = fem::int0;
  //C
  //C     copies a vector, x, to a vector, y.
  //C     uses unrolled loops for increments equal to one.
  //C     jack dongarra, linpack, 3/11/78.
  //C
  if (n <= 0) {
    return;
  }
  if (incx == 1 && incy == 1) {
    goto statement_20;
  }
  //C
  //C        code for unequal increments or equal increments
  //C          not equal to 1
  //C
  ix = 1;
  iy = 1;
  if (incx < 0) {
    ix = (-n + 1) * incx + 1;
  }
  if (incy < 0) {
    iy = (-n + 1) * incy + 1;
  }
  FEM_DO_SAFE(i, 1, n) {
    dy(iy) = dx(ix);
    ix += incx;
    iy += incy;
  }
  return;
  //C
  //C        code for both increments equal to 1
  //C
  //C        clean-up loop
  //C
  statement_20:
  m = fem::mod(n, 7);
  if (m == 0) {
    goto statement_40;
  }
  FEM_DO_SAFE(i, 1, m) {
    dy(i) = dx(i);
  }
  if (n < 7) {
    return;
  }
  statement_40:
  mp1 = m + 1;
  FEM_DOSTEP(i, mp1, n, 7) {
    dy(i) = dx(i);
    dy(i + 1) = dx(i + 1);
    dy(i + 2) = dx(i + 2);
    dy(i + 3) = dx(i + 3);
    dy(i + 4) = dx(i + 4);
    dy(i + 5) = dx(i + 5);
    dy(i + 6) = dx(i + 6);
  }
}

//C
//C====================== The end of ddot ================================
//C
void
dscal(
  int const& n,
  double const& da,
  arr_ref<double> dx,
  int const& incx)
{
  dx(dimension(star));
  int nincx = fem::int0;
  int i = fem::int0;
  int m = fem::int0;
  int mp1 = fem::int0;
  //C
  //C     scales a vector by a constant.
  //C     uses unrolled loops for increment equal to one.
  //C     jack dongarra, linpack, 3/11/78.
  //C     modified 3/93 to return if incx .le. 0.
  //C
  if (n <= 0 || incx <= 0) {
    return;
  }
  if (incx == 1) {
    goto statement_20;
  }
  //C
  //C        code for increment not equal to 1
  //C
  nincx = n * incx;
  FEM_DOSTEP(i, 1, nincx, incx) {
    dx(i) = da * dx(i);
  }
  return;
  //C
  //C        code for increment equal to 1
  //C
  //C        clean-up loop
  //C
  statement_20:
  m = fem::mod(n, 5);
  if (m == 0) {
    goto statement_40;
  }
  FEM_DO_SAFE(i, 1, m) {
    dx(i) = da * dx(i);
  }
  if (n < 5) {
    return;
  }
  statement_40:
  mp1 = m + 1;
  FEM_DOSTEP(i, mp1, n, 5) {
    dx(i) = da * dx(i);
    dx(i + 1) = da * dx(i + 1);
    dx(i + 2) = da * dx(i + 2);
    dx(i + 3) = da * dx(i + 3);
    dx(i + 4) = da * dx(i + 4);
  }
}
//C
//C====================== The end of dscal ===============================
//C

//C
//C======================== The end of bmv ===============================
//C
void
cauchy(
  common& cmn,
  int const& n,
  arr_cref<double> x,
  arr_cref<double> l,
  arr_cref<double> u,
  arr_cref<int> nbd,
  arr_cref<double> g,
  arr_ref<int> iorder,
  arr_ref<int> iwhere,
  arr_ref<double> t,
  arr_ref<double> d,
  arr_ref<double> xcp,
  int const& m,
  arr_cref<double, 2> wy,
  arr_cref<double, 2> ws,
  arr_cref<double, 2> sy,
  arr_cref<double, 2> wt,
  double const& theta,
  int const& col,
  int const& head,
  arr_ref<double> p,
  arr_ref<double> c,
  arr_ref<double> wbp,
  arr_ref<double> v,
  int& nseg,
  int const& iprint,
  double const& sbgnrm,
  int& info,
  double const& epsmch)
{
  x(dimension(n));
  l(dimension(n));
  u(dimension(n));
  nbd(dimension(n));
  g(dimension(n));
  iorder(dimension(n));
  iwhere(dimension(n));
  t(dimension(n));
  d(dimension(n));
  xcp(dimension(n));
  wy(dimension(n, col));
  ws(dimension(n, col));
  sy(dimension(m, m));
  wt(dimension(m, m));
  p(dimension(2 * m));
  c(dimension(2 * m));
  wbp(dimension(2 * m));
  v(dimension(2 * m));
  common_write write(cmn);
  const double zero = 0.0e0;
  bool bnded = fem::bool0;
  int nfree = fem::int0;
  int nbreak = fem::int0;
  int ibkmin = fem::int0;
  double bkmin = fem::double0;
  int col2 = fem::int0;
  double f1 = fem::double0;
  int i = fem::int0;
  double neggi = fem::double0;
  double tl = fem::double0;
  double tu = fem::double0;
  bool xlower = fem::bool0;
  bool xupper = fem::bool0;
  int pointr = fem::int0;
  int j = fem::int0;
  const double one = 1.0e0;
  double f2 = fem::double0;
  double f2_org = fem::double0;
  double dtm = fem::double0;
  double tsum = fem::double0;
  int nleft = fem::int0;
  int iter = fem::int0;
  double tj = fem::double0;
  double tj0 = fem::double0;
  int ibp = fem::int0;
  double dt = fem::double0;
  double dibp = fem::double0;
  double zibp = fem::double0;
  double dibp2 = fem::double0;
  double wmc = fem::double0;
  double wmp = fem::double0;
  double wmw = fem::double0;
  static const char* format_1010 = "('Cauchy X =  ',/(4x,1p,6(1x,d11.4)))";
  static const char* format_6010 =
    "('Distance to the stationary point =  ',1p,d11.4)";
  //C
  //C     ************
  //C
  //C     Subroutine cauchy
  //C
  //C     For given x, l, u, g (with sbgnrm > 0), and a limited memory
  //C       BFGS matrix B defined in terms of matrices WY, WS, WT, and
  //C       scalars head, col, and theta, this subroutine computes the
  //C       generalized Cauchy point (GCP), defined as the first local
  //C       minimizer of the quadratic
  //C
  //C                  Q(x + s) = g's + 1/2 s'Bs
  //C
  //C       along the projected gradient direction P(x-tg,l,u).
  //C       The routine returns the GCP in xcp.
  //C
  //C     n is an integer variable.
  //C       On entry n is the dimension of the problem.
  //C       On exit n is unchanged.
  //C
  //C     x is a double precision array of dimension n.
  //C       On entry x is the starting point for the GCP computation.
  //C       On exit x is unchanged.
  //C
  //C     l is a double precision array of dimension n.
  //C       On entry l is the lower bound of x.
  //C       On exit l is unchanged.
  //C
  //C     u is a double precision array of dimension n.
  //C       On entry u is the upper bound of x.
  //C       On exit u is unchanged.
  //C
  //C     nbd is an integer array of dimension n.
  //C       On entry nbd represents the type of bounds imposed on the
  //C         variables, and must be specified as follows:
  //C         nbd(i)=0 if x(i) is unbounded,
  //C                1 if x(i) has only a lower bound,
  //C                2 if x(i) has both lower and upper bounds, and
  //C                3 if x(i) has only an upper bound.
  //C       On exit nbd is unchanged.
  //C
  //C     g is a double precision array of dimension n.
  //C       On entry g is the gradient of f(x).  g must be a nonzero vector.
  //C       On exit g is unchanged.
  //C
  //C     iorder is an integer working array of dimension n.
  //C       iorder will be used to store the breakpoints in the piecewise
  //C       linear path and free variables encountered. On exit,
  //C         iorder(1),...,iorder(nleft) are indices of breakpoints
  //C                                which have not been encountered;
  //C         iorder(nleft+1),...,iorder(nbreak) are indices of
  //C                                     encountered breakpoints; and
  //C         iorder(nfree),...,iorder(n) are indices of variables which
  //C                 have no bound constraits along the search direction.
  //C
  //C     iwhere is an integer array of dimension n.
  //C       On entry iwhere indicates only the permanently fixed (iwhere=3)
  //C       or free (iwhere= -1) components of x.
  //C       On exit iwhere records the status of the current x variables.
  //C       iwhere(i)=-3  if x(i) is free and has bounds, but is not moved
  //C                 0   if x(i) is free and has bounds, and is moved
  //C                 1   if x(i) is fixed at l(i), and l(i) .ne. u(i)
  //C                 2   if x(i) is fixed at u(i), and u(i) .ne. l(i)
  //C                 3   if x(i) is always fixed, i.e.,  u(i)=x(i)=l(i)
  //C                 -1  if x(i) is always free, i.e., it has no bounds.
  //C
  //C     t is a double precision working array of dimension n.
  //C       t will be used to store the break points.
  //C
  //C     d is a double precision array of dimension n used to store
  //C       the Cauchy direction P(x-tg)-x.
  //C
  //C     xcp is a double precision array of dimension n used to return the
  //C       GCP on exit.
  //C
  //C     m is an integer variable.
  //C       On entry m is the maximum number of variable metric corrections
  //C         used to define the limited memory matrix.
  //C       On exit m is unchanged.
  //C
  //C     ws, wy, sy, and wt are double precision arrays.
  //C       On entry they store information that defines the
  //C                             limited memory BFGS matrix:
  //C         ws(n,m) stores S, a set of s-vectors;
  //C         wy(n,m) stores Y, a set of y-vectors;
  //C         sy(m,m) stores S'Y;
  //C         wt(m,m) stores the
  //C                 Cholesky factorization of (theta*S'S+LD^(-1)L').
  //C       On exit these arrays are unchanged.
  //C
  //C     theta is a double precision variable.
  //C       On entry theta is the scaling factor specifying B_0 = theta I.
  //C       On exit theta is unchanged.
  //C
  //C     col is an integer variable.
  //C       On entry col is the actual number of variable metric
  //C         corrections stored so far.
  //C       On exit col is unchanged.
  //C
  //C     head is an integer variable.
  //C       On entry head is the location of the first s-vector (or y-vector)
  //C         in S (or Y).
  //C       On exit col is unchanged.
  //C
  //C     p is a double precision working array of dimension 2m.
  //C       p will be used to store the vector p = W^(T)d.
  //C
  //C     c is a double precision working array of dimension 2m.
  //C       c will be used to store the vector c = W^(T)(xcp-x).
  //C
  //C     wbp is a double precision working array of dimension 2m.
  //C       wbp will be used to store the row of W corresponding
  //C         to a breakpoint.
  //C
  //C     v is a double precision working array of dimension 2m.
  //C
  //C     nseg is an integer variable.
  //C       On exit nseg records the number of quadratic segments explored
  //C         in searching for the GCP.
  //C
  //C     sg and yg are double precision arrays of dimension m.
  //C       On entry sg  and yg store S'g and Y'g correspondingly.
  //C       On exit they are unchanged.
  //C
  //C     iprint is an INTEGER variable that must be set by the user.
  //C       It controls the frequency and type of output generated:
  //C        iprint<0    no output is generated;
  //C        iprint=0    print only one line at the last iteration;
  //C        0<iprint<99 print also f and |proj g| every iprint iterations;
  //C        iprint=99   print details of every iteration except n-vectors;
  //C        iprint=100  print also the changes of active set and final x;
  //C        iprint>100  print details of every iteration including x and g;
  //C       When iprint > 0, the file iterate.dat will be created to
  //C                        summarize the iteration.
  //C
  //C     sbgnrm is a double precision variable.
  //C       On entry sbgnrm is the norm of the projected gradient at x.
  //C       On exit sbgnrm is unchanged.
  //C
  //C     info is an integer variable.
  //C       On entry info is 0.
  //C       On exit info = 0       for normal return,
  //C                    = nonzero for abnormal return when the the system
  //C                              used in routine bmv is singular.
  //C
  //C     Subprograms called:
  //C
  //C       L-BFGS-B Library ... hpsolb, bmv.
  //C
  //C       Linpack ... dscal dcopy, daxpy.
  //C
  //C     References:
  //C
  //C       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
  //C       memory algorithm for bound constrained optimization'',
  //C       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.
  //C
  //C       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
  //C       Subroutines for Large Scale Bound Constrained Optimization''
  //C       Tech. Report, NAM-11, EECS Department, Northwestern University,
  //C       1994.
  //C
  //C       (Postscript files of these papers are available via anonymous
  //C        ftp to eecs.nwu.edu in the directory pub/lbfgs/lbfgs_bcm.)
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  //C     Check the status of the variables, reset iwhere(i) if necessary;
  //C       compute the Cauchy direction d and the breakpoints t; initialize
  //C       the derivative f1 and the vector p = W'd (for theta = 1).
  //C
  if (sbgnrm <= zero) {
    if (iprint >= 0) {
      write(6, star), "Subgnorm = 0.  GCP = X.";
    }
    dcopy(n, x, 1, xcp, 1);
    return;
  }
  bnded = true;
  nfree = n + 1;
  nbreak = 0;
  ibkmin = 0;
  bkmin = zero;
  col2 = 2 * col;
  f1 = zero;
  if (iprint >= 99) {
    write(6, "(/,'---------------- CAUCHY entered-------------------')");
  }
  //C
  //C     We set p to zero and build it up as we determine d.
  //C
  FEM_DO_SAFE(i, 1, col2) {
    p(i) = zero;
  }
  //C
  //C     In the following loop we determine for each variable its bound
  //C        status and its breakpoint, and update p accordingly.
  //C        Smallest breakpoint is identified.
  //C
  FEM_DO_SAFE(i, 1, n) {
    neggi = -g(i);
    if (iwhere(i) != 3 && iwhere(i) !=  - 1) {
      //C             if x(i) is not a constant and has bounds,
      //C             compute the difference between x(i) and its bounds.
      if (nbd(i) <= 2) {
        tl = x(i) - l(i);
      }
      if (nbd(i) >= 2) {
        tu = u(i) - x(i);
      }
      //C
      //C           If a variable is close enough to a bound
      //C             we treat it as at bound.
      xlower = nbd(i) <= 2 && tl <= zero;
      xupper = nbd(i) >= 2 && tu <= zero;
      //C
      //C              reset iwhere(i).
      iwhere(i) = 0;
      if (xlower) {
        if (neggi <= zero) {
          iwhere(i) = 1;
        }
      }
      else if (xupper) {
        if (neggi >= zero) {
          iwhere(i) = 2;
        }
      }
      else {
        if (fem::abs(neggi) <= zero) {
          iwhere(i) = -3;
        }
      }
    }
    pointr = head;
    if (iwhere(i) != 0 && iwhere(i) !=  - 1) {
      d(i) = zero;
    }
    else {
      d(i) = neggi;
      f1 = f1 - neggi * neggi;
      //C             calculate p := p - W'e_i* (g_i).
      FEM_DO_SAFE(j, 1, col) {
        p(j) += wy(i, pointr) * neggi;
        p(col + j) += ws(i, pointr) * neggi;
        pointr = fem::mod(pointr, m) + 1;
      }
      if (nbd(i) <= 2 && nbd(i) != 0 && neggi < zero) {
        //C                                 x(i) + d(i) is bounded; compute t(i).
        nbreak++;
        iorder(nbreak) = i;
        t(nbreak) = tl / (-neggi);
        if (nbreak == 1 || t(nbreak) < bkmin) {
          bkmin = t(nbreak);
          ibkmin = nbreak;
        }
      }
      else if (nbd(i) >= 2 && neggi > zero) {
        //C                                 x(i) + d(i) is bounded; compute t(i).
        nbreak++;
        iorder(nbreak) = i;
        t(nbreak) = tu / neggi;
        if (nbreak == 1 || t(nbreak) < bkmin) {
          bkmin = t(nbreak);
          ibkmin = nbreak;
        }
      }
      else {
        //C                x(i) + d(i) is not bounded.
        nfree = nfree - 1;
        iorder(nfree) = i;
        if (fem::abs(neggi) > zero) {
          bnded = false;
        }
      }
    }
  }
  //C
  //C     The indices of the nonzero components of d are now stored
  //C       in iorder(1),...,iorder(nbreak) and iorder(nfree),...,iorder(n).
  //C       The smallest of the nbreak breakpoints is in t(ibkmin)=bkmin.
  //C
  if (theta != one) {
    //C                   complete the initialization of p for theta not= one.
    dscal(col, theta, p(col + 1), 1);
  }
  //C
  //C     Initialize GCP xcp = x.
  //C
  dcopy(n, x, 1, xcp, 1);
  //C
  if (nbreak == 0 && nfree == n + 1) {
    //C                  is a zero vector, return with the initial xcp as GCP.
    if (iprint > 100) {
      {
        write_loop wloop(cmn, 6, format_1010);
        FEM_DO_SAFE(i, 1, n) {
          wloop, xcp(i);
        }
      }
    }
    return;
  }
  //C
  //C     Initialize c = W'(xcp - x) = 0.
  //C
  FEM_DO_SAFE(j, 1, col2) {
    c(j) = zero;
  }
  //C
  //C     Initialize derivative f2.
  //C
  f2 = -theta * f1;
  f2_org = f2;
  if (col > 0) {
    bmv(m, sy, wt, col, p, v, info);
    if (info != 0) {
      return;
    }
    f2 = f2 - ddot(col2, v, 1, p, 1);
  }
  dtm = -f1 / f2;
  tsum = zero;
  nseg = 1;
  if (iprint >= 99) {
    write(6, star), "There are ", nbreak, "  breakpoints ";
  }
  //C
  //C     If there are no breakpoints, locate the GCP and return.
  //C
  if (nbreak == 0) {
    goto statement_888;
  }
  //C
  nleft = nbreak;
  iter = 1;
  //C
  tj = zero;
  //C
  //C------------------- the beginning of the loop -------------------------
  //C
  statement_777:
  //C
  //C     Find the next smallest breakpoint;
  //C       compute dt = t(nleft) - t(nleft + 1).
  //C
  tj0 = tj;
  if (iter == 1) {
    //C         Since we already have the smallest breakpoint we need not do
    //C         heapsort yet. Often only one breakpoint is used and the
    //C         cost of heapsort is avoided.
    tj = bkmin;
    ibp = iorder(ibkmin);
  }
  else {
    if (iter == 2) {
      //C             Replace the already used smallest breakpoint with the
      //C             breakpoint numbered nbreak > nlast, before heapsort call.
      if (ibkmin != nbreak) {
        t(ibkmin) = t(nbreak);
        iorder(ibkmin) = iorder(nbreak);
      }
      //C        Update heap structure of breakpoints
      //C           (if iter=2, initialize heap).
    }
    hpsolb(nleft, t, iorder, iter - 2);
    tj = t(nleft);
    ibp = iorder(nleft);
  }
  //C
  dt = tj - tj0;
  //C
  if (dt != zero && iprint >= 100) {
    write(6,
      "(/,'Piece    ',i3,' --f1, f2 at start point ',1p,2(1x,d11.4))"),
      nseg, f1, f2;
    write(6, "('Distance to the next break point =  ',1p,d11.4)"), dt;
    write(6, format_6010), dtm;
  }
  //C
  //C     If a minimizer is within this interval, locate the GCP and return.
  //C
  if (dtm < dt) {
    goto statement_888;
  }
  //C
  //C     Otherwise fix one variable and
  //C       reset the corresponding component of d to zero.
  //C
  tsum += dt;
  nleft = nleft - 1;
  iter++;
  dibp = d(ibp);
  d(ibp) = zero;
  if (dibp > zero) {
    zibp = u(ibp) - x(ibp);
    xcp(ibp) = u(ibp);
    iwhere(ibp) = 2;
  }
  else {
    zibp = l(ibp) - x(ibp);
    xcp(ibp) = l(ibp);
    iwhere(ibp) = 1;
  }
  if (iprint >= 100) {
    write(6, star), "Variable  ", ibp, "  is fixed.";
  }
  if (nleft == 0 && nbreak == n) {
    //C                                             all n variables are fixed,
    //C                                                return with xcp as GCP.
    dtm = dt;
    goto statement_999;
  }
  //C
  //C     Update the derivative information.
  //C
  nseg++;
  dibp2 = fem::pow2(dibp);
  //C
  //C     Update f1 and f2.
  //C
  //C        temporarily set f1 and f2 for col=0.
  f1 += dt * f2 + dibp2 - theta * dibp * zibp;
  f2 = f2 - theta * dibp2;
  //C
  if (col > 0) {
    //C                          update c = c + dt*p.
    daxpy(col2, dt, p, 1, c, 1);
    //C
    //C           choose wbp,
    //C           the row of W corresponding to the breakpoint encountered.
    pointr = head;
    FEM_DO_SAFE(j, 1, col) {
      wbp(j) = wy(ibp, pointr);
      wbp(col + j) = theta * ws(ibp, pointr);
      pointr = fem::mod(pointr, m) + 1;
    }
    //C
    //C           compute (wbp)Mc, (wbp)Mp, and (wbp)M(wbp)'.
    bmv(m, sy, wt, col, wbp, v, info);
    if (info != 0) {
      return;
    }
    wmc = ddot(col2, c, 1, v, 1);
    wmp = ddot(col2, p, 1, v, 1);
    wmw = ddot(col2, wbp, 1, v, 1);
    //C
    //C           update p = p - dibp*wbp.
    daxpy(col2, -dibp, wbp, 1, p, 1);
    //C
    //C           complete updating f1 and f2 while col > 0.
    f1 += dibp * wmc;
    f2 += 2.0e0 * dibp * wmp - dibp2 * wmw;
  }
  //C
  f2 = fem::max(epsmch * f2_org, f2);
  if (nleft > 0) {
    dtm = -f1 / f2;
    goto statement_777;
    //C                 to repeat the loop for unsearched intervals.
  }
  else if (bnded) {
    f1 = zero;
    f2 = zero;
    dtm = zero;
  }
  else {
    dtm = -f1 / f2;
  }
  //C
  //C------------------- the end of the loop -------------------------------
  //C
  statement_888:
  if (iprint >= 99) {
    write(6, star);
    write(6, star), "GCP found in this segment";
    write(6,
      "('Piece    ',i3,' --f1, f2 at start point ',1p,2(1x,d11.4))"),
      nseg, f1, f2;
    write(6, format_6010), dtm;
  }
  if (dtm <= zero) {
    dtm = zero;
  }
  tsum += dtm;
  //C
  //C     Move free variables (i.e., the ones w/o breakpoints) and
  //C       the variables whose breakpoints haven't been reached.
  //C
  daxpy(n, tsum, d, 1, xcp, 1);
  //C
  statement_999:
  //C
  //C     Update c = c + dtm*p = W'(x^c - x)
  //C       which will be used in computing r = Z'(B(x^c - x) + g).
  //C
  if (col > 0) {
    daxpy(col2, dtm, p, 1, c, 1);
  }
  if (iprint > 100) {
    {
      write_loop wloop(cmn, 6, format_1010);
      FEM_DO_SAFE(i, 1, n) {
        wloop, xcp(i);
      }
    }
  }
  if (iprint >= 99) {
    write(6, "(/,'---------------- exit CAUCHY----------------------',/)");
  }
  //C
}

//C
//C====================== The end of cauchy ==============================
//C
void
cmprlb(
  int const& n,
  int const& m,
  arr_cref<double> x,
  arr_cref<double> g,
  arr_cref<double, 2> ws,
  arr_cref<double, 2> wy,
  arr_cref<double, 2> sy,
  arr_cref<double, 2> wt,
  arr_cref<double> z,
  arr_ref<double> r,
  arr_ref<double> wa,
  arr_cref<int> index,
  double const& theta,
  int const& col,
  int const& head,
  int const& nfree,
  bool const& cnstnd,
  int& info)
{
  x(dimension(n));
  g(dimension(n));
  ws(dimension(n, m));
  wy(dimension(n, m));
  sy(dimension(m, m));
  wt(dimension(m, m));
  z(dimension(n));
  r(dimension(n));
  wa(dimension(4 * m));
  index(dimension(n));
  //C
  //C     ************
  //C
  //C     Subroutine cmprlb
  //C
  //C       This subroutine computes r=-Z'B(xcp-xk)-Z'g by using
  //C         wa(2m+1)=W'(xcp-x) from subroutine cauchy.
  //C
  //C     Subprograms called:
  //C
  //C       L-BFGS-B Library ... bmv.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  int i = fem::int0;
  int k = fem::int0;
  int pointr = fem::int0;
  int j = fem::int0;
  double a1 = fem::double0;
  double a2 = fem::double0;
  if (!cnstnd && col > 0) {
    FEM_DO_SAFE(i, 1, n) {
      r(i) = -g(i);
    }
  }
  else {
    FEM_DO_SAFE(i, 1, nfree) {
      k = index(i);
      r(i) = -theta * (z(k) - x(k)) - g(k);
    }
    bmv(m, sy, wt, col, wa(2 * m + 1), wa(1), info);
    if (info != 0) {
      info = -8;
      return;
    }
    pointr = head;
    FEM_DO_SAFE(j, 1, col) {
      a1 = wa(j);
      a2 = theta * wa(col + j);
      FEM_DO_SAFE(i, 1, nfree) {
        k = index(i);
        r(i) += wy(k, pointr) * a1 + ws(k, pointr) * a2;
      }
      pointr = fem::mod(pointr, m) + 1;
    }
  }
  //C
}

//C
//C======================= The end of cmprlb =============================
//C
void
errclb(
  int const& n,
  int const& m,
  double const& factr,
  arr_cref<double> l,
  arr_cref<double> u,
  arr_cref<int> nbd,
  str_ref task,
  int& info,
  int& k)
{
  l(dimension(n));
  u(dimension(n));
  nbd(dimension(n));
  //C
  //C     ************
  //C
  //C     Subroutine errclb
  //C
  //C     This subroutine checks the validity of the input data.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  //C     Check the input arguments for errors.
  //C
  if (n <= 0) {
    task = "ERROR: N .LE. 0";
  }
  if (m <= 0) {
    task = "ERROR: M .LE. 0";
  }
  const double zero = 0.0e0;
  if (factr < zero) {
    task = "ERROR: FACTR .LT. 0";
  }
  //C
  //C     Check the validity of the arrays nbd(i), u(i), and l(i).
  //C
  int i = fem::int0;
  FEM_DO_SAFE(i, 1, n) {
    if (nbd(i) < 0 || nbd(i) > 3) {
      //C                                                   return
      task = "ERROR: INVALID NBD";
      info = -6;
      k = i;
    }
    if (nbd(i) == 2) {
      if (l(i) > u(i)) {
        //C                                    return
        task = "ERROR: NO FEASIBLE SOLUTION";
        info = -7;
        k = i;
      }
    }
  }
  //C
}

//C
//C  L-BFGS-B is released under the “New BSD License” (aka “Modified BSD License”
//C  or “3-clause license”)
//C  Please read attached file License.txt
//C
void
dpofa(
  arr_ref<double, 2> a,
  int const& lda,
  int const& n,
  int& info)
{
  a(dimension(lda, star));
  int j = fem::int0;
  double s = fem::double0;
  int jm1 = fem::int0;
  int k = fem::int0;
  double t = fem::double0;
  //C
  //C     dpofa factors a double precision symmetric positive definite
  //C     matrix.
  //C
  //C     dpofa is usually called by dpoco, but it can be called
  //C     directly with a saving in time if  rcond  is not needed.
  //C     (time for dpoco) = (1 + 18/n)*(time for dpofa) .
  //C
  //C     on entry
  //C
  //C        a       double precision(lda, n)
  //C                the symmetric matrix to be factored.  only the
  //C                diagonal and upper triangle are used.
  //C
  //C        lda     integer
  //C                the leading dimension of the array  a .
  //C
  //C        n       integer
  //C                the order of the matrix  a .
  //C
  //C     on return
  //C
  //C        a       an upper triangular matrix  r  so that  a = trans(r)*r
  //C                where  trans(r)  is the transpose.
  //C                the strict lower triangle is unaltered.
  //C                if  info .ne. 0 , the factorization is not complete.
  //C
  //C        info    integer
  //C                = 0  for normal return.
  //C                = k  signals an error condition.  the leading minor
  //C                     of order  k  is not positive definite.
  //C
  //C     linpack.  this version dated 08/14/78 .
  //C     cleve moler, university of new mexico, argonne national lab.
  //C
  //C     subroutines and functions
  //C
  //C     blas ddot
  //C     fortran sqrt
  //C
  //C     internal variables
  //C
  //C     begin block with ...exits to 40
  //C
  FEM_DO_SAFE(j, 1, n) {
    info = j;
    s = 0.0e0;
    jm1 = j - 1;
    if (jm1 < 1) {
      goto statement_20;
    }
    FEM_DO_SAFE(k, 1, jm1) {
      t = a(k, j) - ddot(k - 1, a(1, k), 1, a(1, j), 1);
      t = t / a(k, k);
      a(k, j) = t;
      s += t * t;
    }
    statement_20:
    s = a(j, j) - s;
    //C     ......exit
    if (s <= 0.0e0) {
      goto statement_40;
    }
    a(j, j) = fem::sqrt(s);
  }
  info = 0;
  statement_40:;
}

//C
//C======================= The end of errclb =============================
//C
void
formk(
  int const& n,
  int const& nsub,
  arr_cref<int> ind,
  int const& nenter,
  int const& ileave,
  arr_cref<int> indx2,
  int const& iupdat,
  bool const& updatd,
  arr_ref<double, 2> wn,
  arr_ref<double, 2> wn1,
  int const& m,
  arr_cref<double, 2> ws,
  arr_cref<double, 2> wy,
  arr_cref<double, 2> sy,
  double const& theta,
  int const& col,
  int const& head,
  int& info)
{
  ind(dimension(n));
  indx2(dimension(n));
  wn(dimension(2 * m, 2 * m));
  wn1(dimension(2 * m, 2 * m));
  ws(dimension(n, m));
  wy(dimension(n, m));
  sy(dimension(m, m));
  //C
  //C     ************
  //C
  //C     Subroutine formk
  //C
  //C     This subroutine forms  the LEL^T factorization of the indefinite
  //C
  //C       matrix    K = [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
  //C                     [L_a -R_z           theta*S'AA'S ]
  //C                                                    where E = [-I  0]
  //C                                                              [ 0  I]
  //C     The matrix K can be shown to be equal to the matrix M^[-1]N
  //C       occurring in section 5.1 of [1], as well as to the matrix
  //C       Mbar^[-1] Nbar in section 5.3.
  //C
  //C     n is an integer variable.
  //C       On entry n is the dimension of the problem.
  //C       On exit n is unchanged.
  //C
  //C     nsub is an integer variable
  //C       On entry nsub is the number of subspace variables in free set.
  //C       On exit nsub is not changed.
  //C
  //C     ind is an integer array of dimension nsub.
  //C       On entry ind specifies the indices of subspace variables.
  //C       On exit ind is unchanged.
  //C
  //C     nenter is an integer variable.
  //C       On entry nenter is the number of variables entering the
  //C         free set.
  //C       On exit nenter is unchanged.
  //C
  //C     ileave is an integer variable.
  //C       On entry indx2(ileave),...,indx2(n) are the variables leaving
  //C         the free set.
  //C       On exit ileave is unchanged.
  //C
  //C     indx2 is an integer array of dimension n.
  //C       On entry indx2(1),...,indx2(nenter) are the variables entering
  //C         the free set, while indx2(ileave),...,indx2(n) are the
  //C         variables leaving the free set.
  //C       On exit indx2 is unchanged.
  //C
  //C     iupdat is an integer variable.
  //C       On entry iupdat is the total number of BFGS updates made so far.
  //C       On exit iupdat is unchanged.
  //C
  //C     updatd is a logical variable.
  //C       On entry 'updatd' is true if the L-BFGS matrix is updatd.
  //C       On exit 'updatd' is unchanged.
  //C
  //C     wn is a double precision array of dimension 2m x 2m.
  //C       On entry wn is unspecified.
  //C       On exit the upper triangle of wn stores the LEL^T factorization
  //C         of the 2*col x 2*col indefinite matrix
  //C                     [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
  //C                     [L_a -R_z           theta*S'AA'S ]
  //C
  //C     wn1 is a double precision array of dimension 2m x 2m.
  //C       On entry wn1 stores the lower triangular part of
  //C                     [Y' ZZ'Y   L_a'+R_z']
  //C                     [L_a+R_z   S'AA'S   ]
  //C         in the previous iteration.
  //C       On exit wn1 stores the corresponding updated matrices.
  //C       The purpose of wn1 is just to store these inner products
  //C       so they can be easily updated and inserted into wn.
  //C
  //C     m is an integer variable.
  //C       On entry m is the maximum number of variable metric corrections
  //C         used to define the limited memory matrix.
  //C       On exit m is unchanged.
  //C
  //C     ws, wy, sy, and wtyy are double precision arrays;
  //C     theta is a double precision variable;
  //C     col is an integer variable;
  //C     head is an integer variable.
  //C       On entry they store the information defining the
  //C                                          limited memory BFGS matrix:
  //C         ws(n,m) stores S, a set of s-vectors;
  //C         wy(n,m) stores Y, a set of y-vectors;
  //C         sy(m,m) stores S'Y;
  //C         wtyy(m,m) stores the Cholesky factorization
  //C                                   of (theta*S'S+LD^(-1)L')
  //C         theta is the scaling factor specifying B_0 = theta I;
  //C         col is the number of variable metric corrections stored;
  //C         head is the location of the 1st s- (or y-) vector in S (or Y).
  //C       On exit they are unchanged.
  //C
  //C     info is an integer variable.
  //C       On entry info is unspecified.
  //C       On exit info =  0 for normal return;
  //C                    = -1 when the 1st Cholesky factorization failed;
  //C                    = -2 when the 2st Cholesky factorization failed.
  //C
  //C     Subprograms called:
  //C
  //C       Linpack ... dcopy, dpofa, dtrsl.
  //C
  //C     References:
  //C       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
  //C       memory algorithm for bound constrained optimization'',
  //C       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.
  //C
  //C       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: a
  //C       limited memory FORTRAN code for solving bound constrained
  //C       optimization problems'', Tech. Report, NAM-11, EECS Department,
  //C       Northwestern University, 1994.
  //C
  //C       (Postscript files of these papers are available via anonymous
  //C        ftp to eecs.nwu.edu in the directory pub/lbfgs/lbfgs_bcm.)
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  //C     Form the lower triangular part of
  //C               WN1 = [Y' ZZ'Y   L_a'+R_z']
  //C                     [L_a+R_z   S'AA'S   ]
  //C        where L_a is the strictly lower triangular part of S'AA'Y
  //C              R_z is the upper triangular part of S'ZZ'Y.
  //C
  int jy = fem::int0;
  int js = fem::int0;
  int pbegin = fem::int0;
  int pend = fem::int0;
  int dbegin = fem::int0;
  int dend = fem::int0;
  int iy = fem::int0;
  int is = fem::int0;
  int ipntr = fem::int0;
  int jpntr = fem::int0;
  const double zero = 0.0e0;
  double temp1 = fem::double0;
  double temp2 = fem::double0;
  double temp3 = fem::double0;
  int k = fem::int0;
  int k1 = fem::int0;
  int i = fem::int0;
  int upcl = fem::int0;
  if (updatd) {
    if (iupdat > m) {
      //C                                 shift old part of WN1.
      FEM_DO_SAFE(jy, 1, m - 1) {
        js = m + jy;
        dcopy(m - jy, wn1(jy + 1, jy + 1), 1, wn1(jy, jy), 1);
        dcopy(m - jy, wn1(js + 1, js + 1), 1, wn1(js, js), 1);
        dcopy(m - 1, wn1(m + 2, jy + 1), 1, wn1(m + 1, jy), 1);
      }
    }
    //C
    //C          put new rows in blocks (1,1), (2,1) and (2,2).
    pbegin = 1;
    pend = nsub;
    dbegin = nsub + 1;
    dend = n;
    iy = col;
    is = m + col;
    ipntr = head + col - 1;
    if (ipntr > m) {
      ipntr = ipntr - m;
    }
    jpntr = head;
    FEM_DO_SAFE(jy, 1, col) {
      js = m + jy;
      temp1 = zero;
      temp2 = zero;
      temp3 = zero;
      //C             compute element jy of row 'col' of Y'ZZ'Y
      FEM_DO_SAFE(k, pbegin, pend) {
        k1 = ind(k);
        temp1 += wy(k1, ipntr) * wy(k1, jpntr);
      }
      //C             compute elements jy of row 'col' of L_a and S'AA'S
      FEM_DO_SAFE(k, dbegin, dend) {
        k1 = ind(k);
        temp2 += ws(k1, ipntr) * ws(k1, jpntr);
        temp3 += ws(k1, ipntr) * wy(k1, jpntr);
      }
      wn1(iy, jy) = temp1;
      wn1(is, js) = temp2;
      wn1(is, jy) = temp3;
      jpntr = fem::mod(jpntr, m) + 1;
    }
    //C
    //C          put new column in block (2,1).
    jy = col;
    jpntr = head + col - 1;
    if (jpntr > m) {
      jpntr = jpntr - m;
    }
    ipntr = head;
    FEM_DO_SAFE(i, 1, col) {
      is = m + i;
      temp3 = zero;
      //C             compute element i of column 'col' of R_z
      FEM_DO_SAFE(k, pbegin, pend) {
        k1 = ind(k);
        temp3 += ws(k1, ipntr) * wy(k1, jpntr);
      }
      ipntr = fem::mod(ipntr, m) + 1;
      wn1(is, jy) = temp3;
    }
    upcl = col - 1;
  }
  else {
    upcl = col;
  }
  //C
  //C       modify the old parts in blocks (1,1) and (2,2) due to changes
  //C       in the set of free variables.
  ipntr = head;
  double temp4 = fem::double0;
  FEM_DO_SAFE(iy, 1, upcl) {
    is = m + iy;
    jpntr = head;
    FEM_DO_SAFE(jy, 1, iy) {
      js = m + jy;
      temp1 = zero;
      temp2 = zero;
      temp3 = zero;
      temp4 = zero;
      FEM_DO_SAFE(k, 1, nenter) {
        k1 = indx2(k);
        temp1 += wy(k1, ipntr) * wy(k1, jpntr);
        temp2 += ws(k1, ipntr) * ws(k1, jpntr);
      }
      FEM_DO_SAFE(k, ileave, n) {
        k1 = indx2(k);
        temp3 += wy(k1, ipntr) * wy(k1, jpntr);
        temp4 += ws(k1, ipntr) * ws(k1, jpntr);
      }
      wn1(iy, jy) += temp1 - temp3;
      wn1(is, js) = wn1(is, js) - temp2 + temp4;
      jpntr = fem::mod(jpntr, m) + 1;
    }
    ipntr = fem::mod(ipntr, m) + 1;
  }
  //C
  //C       modify the old parts in block (2,1).
  ipntr = head;
  FEM_DO_SAFE(is, m + 1, m + upcl) {
    jpntr = head;
    FEM_DO_SAFE(jy, 1, upcl) {
      temp1 = zero;
      temp3 = zero;
      FEM_DO_SAFE(k, 1, nenter) {
        k1 = indx2(k);
        temp1 += ws(k1, ipntr) * wy(k1, jpntr);
      }
      FEM_DO_SAFE(k, ileave, n) {
        k1 = indx2(k);
        temp3 += ws(k1, ipntr) * wy(k1, jpntr);
      }
      if (is <= jy + m) {
        wn1(is, jy) += temp1 - temp3;
      }
      else {
        wn1(is, jy) = wn1(is, jy) - temp1 + temp3;
      }
      jpntr = fem::mod(jpntr, m) + 1;
    }
    ipntr = fem::mod(ipntr, m) + 1;
  }
  //C
  //C     Form the upper triangle of WN = [D+Y' ZZ'Y/theta   -L_a'+R_z' ]
  //C                                     [-L_a +R_z        S'AA'S*theta]
  //C
  int m2 = 2 * m;
  int is1 = fem::int0;
  int js1 = fem::int0;
  FEM_DO_SAFE(iy, 1, col) {
    is = col + iy;
    is1 = m + iy;
    FEM_DO_SAFE(jy, 1, iy) {
      js = col + jy;
      js1 = m + jy;
      wn(jy, iy) = wn1(iy, jy) / theta;
      wn(js, is) = wn1(is1, js1) * theta;
    }
    FEM_DO_SAFE(jy, 1, iy - 1) {
      wn(jy, is) = -wn1(is1, jy);
    }
    FEM_DO_SAFE(jy, iy, col) {
      wn(jy, is) = wn1(is1, jy);
    }
    wn(iy, iy) += sy(iy, iy);
  }
  //C
  //C     Form the upper triangle of WN= [  LL'            L^-1(-L_a'+R_z')]
  //C                                    [(-L_a +R_z)L'^-1   S'AA'S*theta  ]
  //C
  //C        first Cholesky factor (1,1) block of wn to get LL'
  //C                          with L' stored in the upper triangle of wn.
  dpofa(wn, m2, col, info);
  if (info != 0) {
    info = -1;
    return;
  }
  //C        then form L^-1(-L_a'+R_z') in the (1,2) block.
  int col2 = 2 * col;
  FEM_DO_SAFE(js, col + 1, col2) {
    dtrsl(wn, m2, col, wn(1, js), 11, info);
  }
  //C
  //C     Form S'AA'S*theta + (L^-1(-L_a'+R_z'))'L^-1(-L_a'+R_z') in the
  //C        upper triangle of (2,2) block of wn.
  //C
  FEM_DO_SAFE(is, col + 1, col2) {
    FEM_DO_SAFE(js, is, col2) {
      wn(is, js) += ddot(col, wn(1, is), 1, wn(1, js), 1);
    }
  }
  //C
  //C     Cholesky factorization of (2,2) block of wn.
  //C
  dpofa(wn(col + 1, col + 1), m2, col, info);
  if (info != 0) {
    info = -2;
    return;
  }
  //C
}

//C
//C======================= The end of formk ==============================
//C
void
formt(
  int const& m,
  arr_ref<double, 2> wt,
  arr_cref<double, 2> sy,
  arr_cref<double, 2> ss,
  int const& col,
  double const& theta,
  int& info)
{
  wt(dimension(m, m));
  sy(dimension(m, m));
  ss(dimension(m, m));
  //C
  //C     ************
  //C
  //C     Subroutine formt
  //C
  //C       This subroutine forms the upper half of the pos. def. and symm.
  //C         T = theta*SS + L*D^(-1)*L', stores T in the upper triangle
  //C         of the array wt, and performs the Cholesky factorization of T
  //C         to produce J*J', with J' stored in the upper triangle of wt.
  //C
  //C     Subprograms called:
  //C
  //C       Linpack ... dpofa.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  //C     Form the upper half of  T = theta*SS + L*D^(-1)*L',
  //C        store T in the upper triangle of the array wt.
  //C
  int j = fem::int0;
  FEM_DO_SAFE(j, 1, col) {
    wt(1, j) = theta * ss(1, j);
  }
  int i = fem::int0;
  int k1 = fem::int0;
  const double zero = 0.0e0;
  double ddum = fem::double0;
  int k = fem::int0;
  FEM_DO_SAFE(i, 2, col) {
    FEM_DO_SAFE(j, i, col) {
      k1 = fem::min(i, j) - 1;
      ddum = zero;
      FEM_DO_SAFE(k, 1, k1) {
        ddum += sy(i, k) * sy(j, k) / sy(k, k);
      }
      wt(i, j) = ddum + theta * ss(i, j);
    }
  }
  //C
  //C     Cholesky factorize T to J*J' with
  //C        J' stored in the upper triangle of wt.
  //C
  dpofa(wt, m, col, info);
  if (info != 0) {
    info = -3;
  }
  //C
}

//C
//C======================= The end of formt ==============================
//C
void
freev(
  common& cmn,
  int const& n,
  int& nfree,
  arr_ref<int> index,
  int& nenter,
  int& ileave,
  arr_ref<int> indx2,
  arr_cref<int> iwhere,
  bool& wrk,
  bool const& updatd,
  bool const& cnstnd,
  int const& iprint,
  int const& iter)
{
  index(dimension(n));
  indx2(dimension(n));
  iwhere(dimension(n));
  common_write write(cmn);
  //C
  //C     ************
  //C
  //C     Subroutine freev
  //C
  //C     This subroutine counts the entering and leaving variables when
  //C       iter > 0, and finds the index set of free and active variables
  //C       at the GCP.
  //C
  //C     cnstnd is a logical variable indicating whether bounds are present
  //C
  //C     index is an integer array of dimension n
  //C       for i=1,...,nfree, index(i) are the indices of free variables
  //C       for i=nfree+1,...,n, index(i) are the indices of bound variables
  //C       On entry after the first iteration, index gives
  //C         the free variables at the previous iteration.
  //C       On exit it gives the free variables based on the determination
  //C         in cauchy using the array iwhere.
  //C
  //C     indx2 is an integer array of dimension n
  //C       On entry indx2 is unspecified.
  //C       On exit with iter>0, indx2 indicates which variables
  //C          have changed status since the previous iteration.
  //C       For i= 1,...,nenter, indx2(i) have changed from bound to free.
  //C       For i= ileave+1,...,n, indx2(i) have changed from free to bound.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  nenter = 0;
  ileave = n + 1;
  int i = fem::int0;
  int k = fem::int0;
  if (iter > 0 && cnstnd) {
    //C                           count the entering and leaving variables.
    FEM_DO_SAFE(i, 1, nfree) {
      k = index(i);
      //C
      //C            write(6,*) ' k  = index(i) ', k
      //C            write(6,*) ' index = ', i
      //C
      if (iwhere(k) > 0) {
        ileave = ileave - 1;
        indx2(ileave) = k;
        if (iprint >= 100) {
          write(6, star), "Variable ", k, " leaves the set of free variables";
        }
      }
    }
    FEM_DO_SAFE(i, 1 + nfree, n) {
      k = index(i);
      if (iwhere(k) <= 0) {
        nenter++;
        indx2(nenter) = k;
        if (iprint >= 100) {
          write(6, star), "Variable ", k, " enters the set of free variables";
        }
      }
    }
    if (iprint >= 99) {
      write(6, star), n + 1 - ileave, " variables leave; ", nenter,
        " variables enter";
    }
  }
  wrk = (ileave < n + 1) || (nenter > 0) || updatd;
  //C
  //C     Find the index set of free and active variables at the GCP.
  //C
  nfree = 0;
  int iact = n + 1;
  FEM_DO_SAFE(i, 1, n) {
    if (iwhere(i) <= 0) {
      nfree++;
      index(nfree) = i;
    }
    else {
      iact = iact - 1;
      index(iact) = i;
    }
  }
  if (iprint >= 99) {
    write(6, star), nfree, " variables are free at GCP ", iter + 1;
  }
  //C
}

//C
//C====================== The end of dcsrch ==============================
//C
void
dcstep(
  double& stx,
  double& fx,
  double& dx,
  double& sty,
  double& fy,
  double& dy,
  double& stp,
  double const& fp,
  double const& dp,
  bool& brackt,
  double const& stpmin,
  double const& stpmax)
{
  //C     **********
  //C
  //C     Subroutine dcstep
  //C
  //C     This subroutine computes a safeguarded step for a search
  //C     procedure and updates an interval that contains a step that
  //C     satisfies a sufficient decrease and a curvature condition.
  //C
  //C     The parameter stx contains the step with the least function
  //C     value. If brackt is set to .true. then a minimizer has
  //C     been bracketed in an interval with endpoints stx and sty.
  //C     The parameter stp contains the current step.
  //C     The subroutine assumes that if brackt is set to .true. then
  //C
  //C           min(stx,sty) < stp < max(stx,sty),
  //C
  //C     and that the derivative at stx is negative in the direction
  //C     of the step.
  //C
  //C     The subroutine statement is
  //C
  //C       subroutine dcstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,
  //C                         stpmin,stpmax)
  //C
  //C     where
  //C
  //C       stx is a double precision variable.
  //C         On entry stx is the best step obtained so far and is an
  //C            endpoint of the interval that contains the minimizer.
  //C         On exit stx is the updated best step.
  //C
  //C       fx is a double precision variable.
  //C         On entry fx is the function at stx.
  //C         On exit fx is the function at stx.
  //C
  //C       dx is a double precision variable.
  //C         On entry dx is the derivative of the function at
  //C            stx. The derivative must be negative in the direction of
  //C            the step, that is, dx and stp - stx must have opposite
  //C            signs.
  //C         On exit dx is the derivative of the function at stx.
  //C
  //C       sty is a double precision variable.
  //C         On entry sty is the second endpoint of the interval that
  //C            contains the minimizer.
  //C         On exit sty is the updated endpoint of the interval that
  //C            contains the minimizer.
  //C
  //C       fy is a double precision variable.
  //C         On entry fy is the function at sty.
  //C         On exit fy is the function at sty.
  //C
  //C       dy is a double precision variable.
  //C         On entry dy is the derivative of the function at sty.
  //C         On exit dy is the derivative of the function at the exit sty.
  //C
  //C       stp is a double precision variable.
  //C         On entry stp is the current step. If brackt is set to .true.
  //C            then on input stp must be between stx and sty.
  //C         On exit stp is a new trial step.
  //C
  //C       fp is a double precision variable.
  //C         On entry fp is the function at stp
  //C         On exit fp is unchanged.
  //C
  //C       dp is a double precision variable.
  //C         On entry dp is the the derivative of the function at stp.
  //C         On exit dp is unchanged.
  //C
  //C       brackt is an logical variable.
  //C         On entry brackt specifies if a minimizer has been bracketed.
  //C            Initially brackt must be set to .false.
  //C         On exit brackt specifies if a minimizer has been bracketed.
  //C            When a minimizer is bracketed brackt is set to .true.
  //C
  //C       stpmin is a double precision variable.
  //C         On entry stpmin is a lower bound for the step.
  //C         On exit stpmin is unchanged.
  //C
  //C       stpmax is a double precision variable.
  //C         On entry stpmax is an upper bound for the step.
  //C         On exit stpmax is unchanged.
  //C
  //C     MINPACK-1 Project. June 1983
  //C     Argonne National Laboratory.
  //C     Jorge J. More' and David J. Thuente.
  //C
  //C     MINPACK-2 Project. October 1993.
  //C     Argonne National Laboratory and University of Minnesota.
  //C     Brett M. Averick and Jorge J. More'.
  //C
  //C     **********
  //C
  double sgnd = dp * (dx / fem::abs(dx));
  //C
  //C     First case: A higher function value. The minimum is bracketed.
  //C     If the cubic step is closer to stx than the quadratic step, the
  //C     cubic step is taken, otherwise the average of the cubic and
  //C     quadratic steps is taken.
  //C
  const double three = 3.0e0;
  double theta = fem::double0;
  double s = fem::double0;
  double gamma = fem::double0;
  double p = fem::double0;
  double q = fem::double0;
  double r = fem::double0;
  double stpc = fem::double0;
  const double two = 2.0e0;
  double stpq = fem::double0;
  double stpf = fem::double0;
  const double zero = 0.0e0;
  const double p66 = 0.66e0;
  if (fp > fx) {
    theta = three * (fx - fp) / (stp - stx) + dx + dp;
    s = fem::max(fem::abs(theta), fem::abs(dx), fem::abs(dp));
    gamma = s * fem::sqrt(fem::pow2((theta / s)) - (dx / s) * (dp / s));
    if (stp < stx) {
      gamma = -gamma;
    }
    p = (gamma - dx) + theta;
    q = ((gamma - dx) + gamma) + dp;
    r = p / q;
    stpc = stx + r * (stp - stx);
    stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / two) * (stp - stx);
    if (fem::abs(stpc - stx) < fem::abs(stpq - stx)) {
      stpf = stpc;
    }
    else {
      stpf = stpc + (stpq - stpc) / two;
    }
    brackt = true;
    //C
    //C     Second case: A lower function value and derivatives of opposite
    //C     sign. The minimum is bracketed. If the cubic step is farther from
    //C     stp than the secant step, the cubic step is taken, otherwise the
    //C     secant step is taken.
    //C
  }
  else if (sgnd < zero) {
    theta = three * (fx - fp) / (stp - stx) + dx + dp;
    s = fem::max(fem::abs(theta), fem::abs(dx), fem::abs(dp));
    gamma = s * fem::sqrt(fem::pow2((theta / s)) - (dx / s) * (dp / s));
    if (stp > stx) {
      gamma = -gamma;
    }
    p = (gamma - dp) + theta;
    q = ((gamma - dp) + gamma) + dx;
    r = p / q;
    stpc = stp + r * (stx - stp);
    stpq = stp + (dp / (dp - dx)) * (stx - stp);
    if (fem::abs(stpc - stp) > fem::abs(stpq - stp)) {
      stpf = stpc;
    }
    else {
      stpf = stpq;
    }
    brackt = true;
    //C
    //C     Third case: A lower function value, derivatives of the same sign,
    //C     and the magnitude of the derivative decreases.
    //C
  }
  else if (fem::abs(dp) < fem::abs(dx)) {
    //C
    //C        The cubic step is computed only if the cubic tends to infinity
    //C        in the direction of the step or if the minimum of the cubic
    //C        is beyond stp. Otherwise the cubic step is defined to be the
    //C        secant step.
    //C
    theta = three * (fx - fp) / (stp - stx) + dx + dp;
    s = fem::max(fem::abs(theta), fem::abs(dx), fem::abs(dp));
    //C
    //C        The case gamma = 0 only arises if the cubic does not tend
    //C        to infinity in the direction of the step.
    //C
    gamma = s * fem::sqrt(fem::max(zero, fem::pow2((theta / s)) - (
      dx / s) * (dp / s)));
    if (stp > stx) {
      gamma = -gamma;
    }
    p = (gamma - dp) + theta;
    q = (gamma + (dx - dp)) + gamma;
    r = p / q;
    if (r < zero && gamma != zero) {
      stpc = stp + r * (stx - stp);
    }
    else if (stp > stx) {
      stpc = stpmax;
    }
    else {
      stpc = stpmin;
    }
    stpq = stp + (dp / (dp - dx)) * (stx - stp);
    //C
    if (brackt) {
      //C
      //C           A minimizer has been bracketed. If the cubic step is
      //C           closer to stp than the secant step, the cubic step is
      //C           taken, otherwise the secant step is taken.
      //C
      if (fem::abs(stpc - stp) < fem::abs(stpq - stp)) {
        stpf = stpc;
      }
      else {
        stpf = stpq;
      }
      if (stp > stx) {
        stpf = fem::min(stp + p66 * (sty - stp), stpf);
      }
      else {
        stpf = fem::max(stp + p66 * (sty - stp), stpf);
      }
    }
    else {
      //C
      //C           A minimizer has not been bracketed. If the cubic step is
      //C           farther from stp than the secant step, the cubic step is
      //C           taken, otherwise the secant step is taken.
      //C
      if (fem::abs(stpc - stp) > fem::abs(stpq - stp)) {
        stpf = stpc;
      }
      else {
        stpf = stpq;
      }
      stpf = fem::min(stpmax, stpf);
      stpf = fem::max(stpmin, stpf);
    }
    //C
    //C     Fourth case: A lower function value, derivatives of the same sign,
    //C     and the magnitude of the derivative does not decrease. If the
    //C     minimum is not bracketed, the step is either stpmin or stpmax,
    //C     otherwise the cubic step is taken.
    //C
  }
  else {
    if (brackt) {
      theta = three * (fp - fy) / (sty - stp) + dy + dp;
      s = fem::max(fem::abs(theta), fem::abs(dy), fem::abs(dp));
      gamma = s * fem::sqrt(fem::pow2((theta / s)) - (dy / s) * (dp / s));
      if (stp > sty) {
        gamma = -gamma;
      }
      p = (gamma - dp) + theta;
      q = ((gamma - dp) + gamma) + dy;
      r = p / q;
      stpc = stp + r * (sty - stp);
      stpf = stpc;
    }
    else if (stp > stx) {
      stpf = stpmax;
    }
    else {
      stpf = stpmin;
    }
  }
  //C
  //C     Update the interval which contains a minimizer.
  //C
  if (fp > fx) {
    sty = stp;
    fy = fp;
    dy = dp;
  }
  else {
    if (sgnd < zero) {
      sty = stx;
      fy = fx;
      dy = dx;
    }
    stx = stp;
    fx = fp;
    dx = dp;
  }
  //C
  //C     Compute the new step.
  //C
  stp = stpf;
  //C
}

//C====================== The end of subsm ===============================
//C
void
dcsrch(
  double const& f,
  double const& g,
  double& stp,
  double const& ftol,
  double const& gtol,
  double const& xtol,
  double const& stpmin,
  double const& stpmax,
  str_ref task,
  arr_ref<int> isave,
  arr_ref<double> dsave)
{
  isave(dimension(2));
  dsave(dimension(13));
  const double zero = 0.0e0;
  bool brackt = fem::bool0;
  int stage = fem::int0;
  double finit = fem::double0;
  double ginit = fem::double0;
  double gtest = fem::double0;
  double width = fem::double0;
  const double p5 = 0.5e0;
  double width1 = fem::double0;
  double stx = fem::double0;
  double fx = fem::double0;
  double gx = fem::double0;
  double sty = fem::double0;
  double fy = fem::double0;
  double gy = fem::double0;
  double stmin = fem::double0;
  const double xtrapu = 4.0e0;
  double stmax = fem::double0;
  double ftest = fem::double0;
  double fm = fem::double0;
  double fxm = fem::double0;
  double fym = fem::double0;
  double gm = fem::double0;
  double gxm = fem::double0;
  double gym = fem::double0;
  const double p66 = 0.66e0;
  const double xtrapl = 1.1e0;
  //C     **********
  //C
  //C     Subroutine dcsrch
  //C
  //C     This subroutine finds a step that satisfies a sufficient
  //C     decrease condition and a curvature condition.
  //C
  //C     Each call of the subroutine updates an interval with
  //C     endpoints stx and sty. The interval is initially chosen
  //C     so that it contains a minimizer of the modified function
  //C
  //C           psi(stp) = f(stp) - f(0) - ftol*stp*f'(0).
  //C
  //C     If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
  //C     interval is chosen so that it contains a minimizer of f.
  //C
  //C     The algorithm is designed to find a step that satisfies
  //C     the sufficient decrease condition
  //C
  //C           f(stp) <= f(0) + ftol*stp*f'(0),
  //C
  //C     and the curvature condition
  //C
  //C           abs(f'(stp)) <= gtol*abs(f'(0)).
  //C
  //C     If ftol is less than gtol and if, for example, the function
  //C     is bounded below, then there is always a step which satisfies
  //C     both conditions.
  //C
  //C     If no step can be found that satisfies both conditions, then
  //C     the algorithm stops with a warning. In this case stp only
  //C     satisfies the sufficient decrease condition.
  //C
  //C     A typical invocation of dcsrch has the following outline:
  //C
  //C     task = 'START'
  //C  10 continue
  //C        call dcsrch( ... )
  //C        if (task .eq. 'FG') then
  //C           Evaluate the function and the gradient at stp
  //C           goto 10
  //C           end if
  //C
  //C     NOTE: The user must no alter work arrays between calls.
  //C
  //C     The subroutine statement is
  //C
  //C        subroutine dcsrch(f,g,stp,ftol,gtol,xtol,stpmin,stpmax,
  //C                          task,isave,dsave)
  //C     where
  //C
  //C       f is a double precision variable.
  //C         On initial entry f is the value of the function at 0.
  //C            On subsequent entries f is the value of the
  //C            function at stp.
  //C         On exit f is the value of the function at stp.
  //C
  //C       g is a double precision variable.
  //C         On initial entry g is the derivative of the function at 0.
  //C            On subsequent entries g is the derivative of the
  //C            function at stp.
  //C         On exit g is the derivative of the function at stp.
  //C
  //C       stp is a double precision variable.
  //C         On entry stp is the current estimate of a satisfactory
  //C            step. On initial entry, a positive initial estimate
  //C            must be provided.
  //C         On exit stp is the current estimate of a satisfactory step
  //C            if task = 'FG'. If task = 'CONV' then stp satisfies
  //C            the sufficient decrease and curvature condition.
  //C
  //C       ftol is a double precision variable.
  //C         On entry ftol specifies a nonnegative tolerance for the
  //C            sufficient decrease condition.
  //C         On exit ftol is unchanged.
  //C
  //C       gtol is a double precision variable.
  //C         On entry gtol specifies a nonnegative tolerance for the
  //C            curvature condition.
  //C         On exit gtol is unchanged.
  //C
  //C       xtol is a double precision variable.
  //C         On entry xtol specifies a nonnegative relative tolerance
  //C            for an acceptable step. The subroutine exits with a
  //C            warning if the relative difference between sty and stx
  //C            is less than xtol.
  //C         On exit xtol is unchanged.
  //C
  //C       stpmin is a double precision variable.
  //C         On entry stpmin is a nonnegative lower bound for the step.
  //C         On exit stpmin is unchanged.
  //C
  //C       stpmax is a double precision variable.
  //C         On entry stpmax is a nonnegative upper bound for the step.
  //C         On exit stpmax is unchanged.
  //C
  //C       task is a character variable of length at least 60.
  //C         On initial entry task must be set to 'START'.
  //C         On exit task indicates the required action:
  //C
  //C            If task(1:2) = 'FG' then evaluate the function and
  //C            derivative at stp and call dcsrch again.
  //C
  //C            If task(1:4) = 'CONV' then the search is successful.
  //C
  //C            If task(1:4) = 'WARN' then the subroutine is not able
  //C            to satisfy the convergence conditions. The exit value of
  //C            stp contains the best point found during the search.
  //C
  //C            If task(1:5) = 'ERROR' then there is an error in the
  //C            input arguments.
  //C
  //C         On exit with convergence, a warning or an error, the
  //C            variable task contains additional information.
  //C
  //C       isave is an integer work array of dimension 2.
  //C
  //C       dsave is a double precision work array of dimension 13.
  //C
  //C     Subprograms called
  //C
  //C       MINPACK-2 ... dcstep
  //C
  //C     MINPACK-1 Project. June 1983.
  //C     Argonne National Laboratory.
  //C     Jorge J. More' and David J. Thuente.
  //C
  //C     MINPACK-2 Project. October 1993.
  //C     Argonne National Laboratory and University of Minnesota.
  //C     Brett M. Averick, Richard G. Carter, and Jorge J. More'.
  //C
  //C     **********
  //C
  //C     Initialization block.
  //C
  if (task(1, 5) == "START") {
    //C
    //C        Check the input arguments for errors.
    //C
    if (stp < stpmin) {
      task = "ERROR: STP .LT. STPMIN";
    }
    if (stp > stpmax) {
      task = "ERROR: STP .GT. STPMAX";
    }
    if (g >= zero) {
      task = "ERROR: INITIAL G .GE. ZERO";
    }
    if (ftol < zero) {
      task = "ERROR: FTOL .LT. ZERO";
    }
    if (gtol < zero) {
      task = "ERROR: GTOL .LT. ZERO";
    }
    if (xtol < zero) {
      task = "ERROR: XTOL .LT. ZERO";
    }
    if (stpmin < zero) {
      task = "ERROR: STPMIN .LT. ZERO";
    }
    if (stpmax < stpmin) {
      task = "ERROR: STPMAX .LT. STPMIN";
    }
    //C
    //C        Exit if there are errors on input.
    //C
    if (task(1, 5) == "ERROR") {
      return;
    }
    //C
    //C        Initialize local variables.
    //C
    brackt = false;
    stage = 1;
    finit = f;
    ginit = g;
    gtest = ftol * ginit;
    width = stpmax - stpmin;
    width1 = width / p5;
    //C
    //C        The variables stx, fx, gx contain the values of the step,
    //C        function, and derivative at the best step.
    //C        The variables sty, fy, gy contain the value of the step,
    //C        function, and derivative at sty.
    //C        The variables stp, f, g contain the values of the step,
    //C        function, and derivative at stp.
    //C
    stx = zero;
    fx = finit;
    gx = ginit;
    sty = zero;
    fy = finit;
    gy = ginit;
    stmin = zero;
    stmax = stp + xtrapu * stp;
    task = "FG";
    //C
    goto statement_1000;
    //C
  }
  else {
    //C
    //C        Restore local variables.
    //C
    if (isave(1) == 1) {
      brackt = true;
    }
    else {
      brackt = false;
    }
    stage = isave(2);
    ginit = dsave(1);
    gtest = dsave(2);
    gx = dsave(3);
    gy = dsave(4);
    finit = dsave(5);
    fx = dsave(6);
    fy = dsave(7);
    stx = dsave(8);
    sty = dsave(9);
    stmin = dsave(10);
    stmax = dsave(11);
    width = dsave(12);
    width1 = dsave(13);
    //C
  }
  //C
  //C     If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
  //C     algorithm enters the second stage.
  //C
  ftest = finit + stp * gtest;
  if (stage == 1 && f <= ftest && g >= zero) {
    stage = 2;
  }
  //C
  //C     Test for warnings.
  //C
  if (brackt && (stp <= stmin || stp >= stmax)) {
    task = "WARNING: ROUNDING ERRORS PREVENT PROGRESS";
  }
  if (brackt && stmax - stmin <= xtol * stmax) {
    task = "WARNING: XTOL TEST SATISFIED";
  }
  if (stp == stpmax && f <= ftest && g <= gtest) {
    task = "WARNING: STP = STPMAX";
  }
  if (stp == stpmin && (f > ftest || g >= gtest)) {
    task = "WARNING: STP = STPMIN";
  }
  //C
  //C     Test for convergence.
  //C
  if (f <= ftest && fem::abs(g) <= gtol * (-ginit)) {
    task = "CONVERGENCE";
  }
  //C
  //C     Test for termination.
  //C
  if (task(1, 4) == "WARN" || task(1, 4) == "CONV") {
    goto statement_1000;
  }
  //C
  //C     A modified function is used to predict the step during the
  //C     first stage if a lower function value has been obtained but
  //C     the decrease is not sufficient.
  //C
  if (stage == 1 && f <= fx && f > ftest) {
    //C
    //C        Define the modified function and derivative values.
    //C
    fm = f - stp * gtest;
    fxm = fx - stx * gtest;
    fym = fy - sty * gtest;
    gm = g - gtest;
    gxm = gx - gtest;
    gym = gy - gtest;
    //C
    //C        Call dcstep to update stx, sty, and to compute the new step.
    //C
    dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax);
    //C
    //C        Reset the function and derivative values for f.
    //C
    fx = fxm + stx * gtest;
    fy = fym + sty * gtest;
    gx = gxm + gtest;
    gy = gym + gtest;
    //C
  }
  else {
    //C
    //C       Call dcstep to update stx, sty, and to compute the new step.
    //C
    dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax);
    //C
  }
  //C
  //C     Decide if a bisection step is needed.
  //C
  if (brackt) {
    if (fem::abs(sty - stx) >= p66 * width1) {
      stp = stx + p5 * (sty - stx);
    }
    width1 = width;
    width = fem::abs(sty - stx);
  }
  //C
  //C     Set the minimum and maximum steps allowed for stp.
  //C
  if (brackt) {
    stmin = fem::min(stx, sty);
    stmax = fem::max(stx, sty);
  }
  else {
    stmin = stp + xtrapl * (stp - stx);
    stmax = stp + xtrapu * (stp - stx);
  }
  //C
  //C     Force the step to be within the bounds stpmax and stpmin.
  //C
  stp = fem::max(stp, stpmin);
  stp = fem::min(stp, stpmax);
  //C
  //C     If further progress is not possible, let stp be the best
  //C     point obtained during the search.
  //C
  if (brackt && (stp <= stmin || stp >= stmax) || (brackt && stmax -
      stmin <= xtol * stmax)) {
    stp = stx;
  }
  //C
  //C     Obtain another function and derivative.
  //C
  task = "FG";
  //C
  statement_1000:
  //C
  //C     Save local variables.
  //C
  if (brackt) {
    isave(1) = 1;
  }
  else {
    isave(1) = 0;
  }
  isave(2) = stage;
  dsave(1) = ginit;
  dsave(2) = gtest;
  dsave(3) = gx;
  dsave(4) = gy;
  dsave(5) = finit;
  dsave(6) = fx;
  dsave(7) = fy;
  dsave(8) = stx;
  dsave(9) = sty;
  dsave(10) = stmin;
  dsave(11) = stmax;
  dsave(12) = width;
  dsave(13) = width1;
  //C
}

//C
//C====================== The end of hpsolb ==============================
//C
void
lnsrlb(
  common& cmn,
  int const& n,
  arr_cref<double> l,
  arr_cref<double> u,
  arr_cref<int> nbd,
  arr_ref<double> x,
  double const& f,
  double& fold,
  double& gd,
  double& gdold,
  arr_cref<double> g,
  arr_cref<double> d,
  arr_ref<double> r,
  arr_ref<double> t,
  arr_cref<double> z,
  double& stp,
  double& dnorm,
  double& dtd,
  double& xstep,
  double& stpmx,
  int const& iter,
  int& ifun,
  int& iback,
  int& nfgv,
  int& info,
  str_ref task,
  bool const& boxed,
  bool const& cnstnd,
  str_ref csave,
  arr_ref<int> isave,
  arr_ref<double> dsave)
{
  l(dimension(n));
  u(dimension(n));
  nbd(dimension(n));
  x(dimension(n));
  g(dimension(n));
  d(dimension(n));
  r(dimension(n));
  t(dimension(n));
  z(dimension(n));
  isave(dimension(2));
  dsave(dimension(13));
  common_write write(cmn);
  const double big = 1.0e+10;
  const double one = 1.0e0;
  int i = fem::int0;
  double a1 = fem::double0;
  const double zero = 0.0e0;
  double a2 = fem::double0;
  const double ftol = 1.0e-3;
  const double gtol = 0.9e0;
  const double xtol = 0.1e0;
  //C
  //C     **********
  //C
  //C     Subroutine lnsrlb
  //C
  //C     This subroutine calls subroutine dcsrch from the Minpack2 library
  //C       to perform the line search.  Subroutine dscrch is safeguarded so
  //C       that all trial points lie within the feasible region.
  //C
  //C     Subprograms called:
  //C
  //C       Minpack2 Library ... dcsrch.
  //C
  //C       Linpack ... dtrsl, ddot.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     **********
  //C
  if (task(1, 5) == "FG_LN") {
    goto statement_556;
  }
  //C
  dtd = ddot(n, d, 1, d, 1);
  dnorm = fem::sqrt(dtd);
  //C
  //C     Determine the maximum step length.
  //C
  stpmx = big;
  if (cnstnd) {
    if (iter == 0) {
      stpmx = one;
    }
    else {
      FEM_DO_SAFE(i, 1, n) {
        a1 = d(i);
        if (nbd(i) != 0) {
          if (a1 < zero && nbd(i) <= 2) {
            a2 = l(i) - x(i);
            if (a2 >= zero) {
              stpmx = zero;
            }
            else if (a1 * stpmx < a2) {
              stpmx = a2 / a1;
            }
          }
          else if (a1 > zero && nbd(i) >= 2) {
            a2 = u(i) - x(i);
            if (a2 <= zero) {
              stpmx = zero;
            }
            else if (a1 * stpmx > a2) {
              stpmx = a2 / a1;
            }
          }
        }
      }
    }
  }
  //C
  if (iter == 0 && !boxed) {
    stp = fem::min(one / dnorm, stpmx);
  }
  else {
    stp = one;
  }
  //C
  dcopy(n, x, 1, t, 1);
  dcopy(n, g, 1, r, 1);
  fold = f;
  ifun = 0;
  iback = 0;
  csave = "START";
  statement_556:
  gd = ddot(n, g, 1, d, 1);
  if (ifun == 0) {
    gdold = gd;
    if (gd >= zero) {
      //C                               the directional derivative >=0.
      //C                               Line search is impossible.
      write(6, star), " ascent direction in projection gd = ", gd;
      info = -4;
      return;
    }
  }
  //C
  dcsrch(f, gd, stp, ftol, gtol, xtol, zero, stpmx, csave, isave, dsave);
  //C
  xstep = stp * dnorm;
  if (csave(1, 4) != "CONV" && csave(1, 4) != "WARN") {
    task = "FG_LNSRCH";
    ifun++;
    nfgv++;
    iback = ifun - 1;
    if (stp == one) {
      dcopy(n, z, 1, x, 1);
    }
    else {
      FEM_DO_SAFE(i, 1, n) {
        x(i) = stp * d(i) + t(i);
      }
    }
  }
  else {
    task = "NEW_X";
  }
  //C
}

//C
//C======================= The end of lnsrlb =============================
//C
void
matupd(
  int const& n,
  int const& m,
  arr_ref<double, 2> ws,
  arr_ref<double, 2> wy,
  arr_ref<double, 2> sy,
  arr_ref<double, 2> ss,
  arr_cref<double> d,
  arr_cref<double> r,
  int& itail,
  int const& iupdat,
  int& col,
  int& head,
  double& theta,
  double const& rr,
  double const& dr,
  double const& stp,
  double const& dtd)
{
  ws(dimension(n, m));
  wy(dimension(n, m));
  sy(dimension(m, m));
  ss(dimension(m, m));
  d(dimension(n));
  r(dimension(n));
  //C
  //C     ************
  //C
  //C     Subroutine matupd
  //C
  //C       This subroutine updates matrices WS and WY, and forms the
  //C         middle matrix in B.
  //C
  //C     Subprograms called:
  //C
  //C       Linpack ... dcopy, ddot.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  //C     Set pointers for matrices WS and WY.
  //C
  if (iupdat <= m) {
    col = iupdat;
    itail = fem::mod(head + iupdat - 2, m) + 1;
  }
  else {
    itail = fem::mod(itail, m) + 1;
    head = fem::mod(head, m) + 1;
  }
  //C
  //C     Update matrices WS and WY.
  //C
  dcopy(n, d, 1, ws(1, itail), 1);
  dcopy(n, r, 1, wy(1, itail), 1);
  //C
  //C     Set theta=yy/ys.
  //C
  theta = rr / dr;
  //C
  //C     Form the middle matrix in B.
  //C
  //C        update the upper triangle of SS,
  //C                                         and the lower triangle of SY:
  int j = fem::int0;
  if (iupdat > m) {
    //C                              move old information
    FEM_DO_SAFE(j, 1, col - 1) {
      dcopy(j, ss(2, j + 1), 1, ss(1, j), 1);
      dcopy(col - j, sy(j + 1, j + 1), 1, sy(j, j), 1);
    }
  }
  //C        add new information: the last row of SY
  //C                                             and the last column of SS:
  int pointr = head;
  FEM_DO_SAFE(j, 1, col - 1) {
    sy(col, j) = ddot(n, d, 1, wy(1, pointr), 1);
    ss(j, col) = ddot(n, ws(1, pointr), 1, d, 1);
    pointr = fem::mod(pointr, m) + 1;
  }
  const double one = 1.0e0;
  if (stp == one) {
    ss(col, col) = dtd;
  }
  else {
    ss(col, col) = stp * stp * dtd;
  }
  sy(col, col) = dr;
  //C
}

//C
//C======================= The end of matupd =============================
//C
void
prn1lb(
  common& cmn,
  int const& n,
  int const& m,
  arr_cref<double> l,
  arr_cref<double> u,
  arr_cref<double> x,
  int const& iprint,
  int const& itfile,
  double const& epsmch)
{
  l(dimension(n));
  u(dimension(n));
  x(dimension(n));
  common_write write(cmn);
  static const char* format_1004 = "(/,a4,1p,6(1x,d11.4),/(4x,1p,6(1x,d11.4)))";
  //C
  //C     ************
  //C
  //C     Subroutine prn1lb
  //C
  //C     This subroutine prints the input data, initial point, upper and
  //C       lower bounds of each variable, machine precision, as well as
  //C       the headings of the output.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  int i = fem::int0;
  if (iprint >= 0) {
    write(6,
      "('RUNNING THE L-BFGS-B CODE',/,/,'           * * *',/,/,"
      "'Machine precision =',1p,d10.3)"),
      epsmch;
    write(6, star), "N = ", n, "    M = ", m;
    if (iprint >= 1) {
      write(itfile,
        "('RUNNING THE L-BFGS-B CODE',/,/,'it    = iteration number',/,"
        "'nf    = number of function evaluations',/,"
        "'nseg  = number of segments explored during the Cauchy search',/,"
        "'nact  = number of active bounds at the generalized Cauchy point',/,"
        "'sub   = manner in which the subspace minimization terminated:',/,"
        "'        con = converged, bnd = a bound was reached',/,"
        "'itls  = number of iterations performed in the line search',/,"
        "'stepl = step length used',/,"
        "'tstep = norm of the displacement (total step)',/,"
        "'projg = norm of the projected gradient',/,'f     = function value',"
        "/,/,'           * * *',/,/,'Machine precision =',1p,d10.3)"),
        epsmch;
      write(itfile, star), "N = ", n, "    M = ", m;
      write(itfile,
        "(/,3x,'it',3x,'nf',2x,'nseg',2x,'nact',2x,'sub',2x,'itls',2x,'stepl',"
        "4x,'tstep',5x,'projg',8x,'f')");
      if (iprint > 100) {
        {
          write_loop wloop(cmn, 6, format_1004);
          wloop, "L =";
          FEM_DO_SAFE(i, 1, n) {
            wloop, l(i);
          }
        }
        {
          write_loop wloop(cmn, 6, format_1004);
          wloop, "X0 =";
          FEM_DO_SAFE(i, 1, n) {
            wloop, x(i);
          }
        }
        {
          write_loop wloop(cmn, 6, format_1004);
          wloop, "U =";
          FEM_DO_SAFE(i, 1, n) {
            wloop, u(i);
          }
        }
      }
    }
  }
  //C
}

//C
//C======================= The end of prn1lb =============================
//C
void
prn2lb(
  common& cmn,
  int const& n,
  arr_cref<double> x,
  double const& f,
  arr_cref<double> g,
  int const& iprint,
  int const& itfile,
  int const& iter,
  int const& nfgv,
  int const& nact,
  double const& sbgnrm,
  int const& nseg,
  str_ref word,
  int const& iword,
  int const& iback,
  double const& stp,
  double const& xstep)
{
  x(dimension(n));
  g(dimension(n));
  common_write write(cmn);
  static const char* format_1004 = "(/,a4,1p,6(1x,d11.4),/(4x,1p,6(1x,d11.4)))";
  static const char* format_2001 =
    "(/,'At iterate',i5,4x,'f= ',1p,d12.5,4x,'|proj g|= ',1p,d12.5)";
  //C
  //C     ************
  //C
  //C     Subroutine prn2lb
  //C
  //C     This subroutine prints out new information after a successful
  //C       line search.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  //C           'word' records the status of subspace solutions.
  if (iword == 0) {
    //C                            the subspace minimization converged.
    word = "con";
  }
  else if (iword == 1) {
    //C                          the subspace minimization stopped at a bound.
    word = "bnd";
  }
  else if (iword == 5) {
    //C                             the truncated Newton step has been used.
    word = "TNT";
  }
  else {
    word = "---";
  }
  int i = fem::int0;
  int imod = fem::int0;
  if (iprint >= 99) {
    write(6, star), "LINE SEARCH", iback, " times; norm of step = ", xstep;
    write(6, format_2001), iter, f, sbgnrm;
    if (iprint > 100) {
      {
        write_loop wloop(cmn, 6, format_1004);
        wloop, "X =";
        FEM_DO_SAFE(i, 1, n) {
          wloop, x(i);
        }
      }
      {
        write_loop wloop(cmn, 6, format_1004);
        wloop, "G =";
        FEM_DO_SAFE(i, 1, n) {
          wloop, g(i);
        }
      }
    }
  }
  else if (iprint > 0) {
    imod = fem::mod(iter, iprint);
    if (imod == 0) {
      write(6, format_2001), iter, f, sbgnrm;
    }
  }
  if (iprint >= 1) {
    write(itfile,
      "(2(1x,i4),2(1x,i5),2x,a3,1x,i4,1p,2(2x,d7.1),1p,2(1x,d10.3))"),
      iter, nfgv, nseg, nact, word, iback, stp, xstep, sbgnrm, f;
  }
  //C
}

//C
//C======================= The end of prn2lb =============================
//C
void
prn3lb(
  common& cmn,
  int const& n,
  arr_cref<double> x,
  double const& f,
  str_cref task,
  int const& iprint,
  int const& info,
  int const& itfile,
  int const& iter,
  int const& nfgv,
  int const& nintol,
  int const& nskip,
  int const& nact,
  double const& sbgnrm,
  double const& time,
  int const& nseg,
  str_cref word,
  int const& iback,
  double const& stp,
  double const& xstep,
  int const& k,
  double const& cachyt,
  double const& sbtime,
  double const& lnscht)
{
  x(dimension(n));
  common_write write(cmn);
  int i = fem::int0;
  static const char* format_3008 =
    "(/,' Total User time',1p,e10.3,' seconds.',/)";
  static const char* format_3009 = "(/,a60)";
  static const char* format_9011 =
    "(/,' Matrix in 1st Cholesky factorization in formk is not Pos. Def.')";
  static const char* format_9012 =
    "(/,' Matrix in 2st Cholesky factorization in formk is not Pos. Def.')";
  static const char* format_9013 =
    "(/,' Matrix in the Cholesky factorization in formt is not Pos. Def.')";
  static const char* format_9014 =
    "(/,' Derivative >= 0, backtracking line search impossible.',/,"
    "'   Previous x, f and g restored.',/,"
    "' Possible causes: 1 error in function or gradient evaluation;',/,"
    "'                  2 rounding errors dominate computation.')";
  static const char* format_9015 =
    "(/,' Warning:  more than 10 function and gradient',/,"
    "'   evaluations in the last line search.  Termination',/,"
    "'   may possibly be caused by a bad search direction.')";
  static const char* format_9018 = "(/,' The triangular system is singular.')";
  static const char* format_9019 =
    "(/,' Line search cannot locate an adequate point after 20 function',/,"
    "'  and gradient evaluations.  Previous x, f and g restored.',/,"
    "' Possible causes: 1 error in function or gradient evaluation;',/,"
    "'                  2 rounding error dominate computation.')";
  //C
  //C     ************
  //C
  //C     Subroutine prn3lb
  //C
  //C     This subroutine prints out information when either a built-in
  //C       convergence test is satisfied or when an error message is
  //C       generated.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  if (task(1, 5) == "ERROR") {
    goto statement_999;
  }
  //C
  if (iprint >= 0) {
    write(6,
      "(/,'           * * *',/,/,'Tit   = total number of iterations',/,"
      "'Tnf   = total number of function evaluations',/,"
      "'Tnint = total number of segments explored during',' Cauchy searches',"
      "/,'Skip  = number of BFGS updates skipped',/,"
      "'Nact  = number of active bounds at final generalized',' Cauchy point',"
      "/,'Projg = norm of the final projected gradient',/,"
      "'F     = final function value',/,/,'           * * *')");
    write(6,
      "(/,3x,'N',4x,'Tit',5x,'Tnf',2x,'Tnint',2x,'Skip',2x,'Nact',5x,'Projg',"
      "8x,'F')");
    write(6, "(i5,2(1x,i6)(1x,i6)(2x,i4)(1x,i5),1p,2(2x,d10.3))"), n,
      iter, nfgv, nintol, nskip, nact, sbgnrm, f;
    if (iprint >= 100) {
      {
        write_loop wloop(cmn, 6, "(/,a4,1p,6(1x,d11.4),/(4x,1p,6(1x,d11.4)))");
        wloop, "X =";
        FEM_DO_SAFE(i, 1, n) {
          wloop, x(i);
        }
      }
    }
    if (iprint >= 1) {
      write(6, star), " F =", f;
    }
  }
  statement_999:
  if (iprint >= 0) {
    write(6, format_3009), task;
    if (info != 0) {
      if (info ==  - 1) {
        write(6, format_9011);
      }
      if (info ==  - 2) {
        write(6, format_9012);
      }
      if (info ==  - 3) {
        write(6, format_9013);
      }
      if (info ==  - 4) {
        write(6, format_9014);
      }
      if (info ==  - 5) {
        write(6, format_9015);
      }
      if (info ==  - 6) {
        write(6, star), " Input nbd(", k, ") is invalid.";
      }
      if (info ==  - 7) {
        write(6, star), " l(", k, ") > u(", k, ").  No feasible solution.";
      }
      if (info ==  - 8) {
        write(6, format_9018);
      }
      if (info ==  - 9) {
        write(6, format_9019);
      }
    }
    if (iprint >= 1) {
      write(6,
        "(/,' Cauchy                time',1p,e10.3,' seconds.',/,"
        "' Subspace minimization time',1p,e10.3,' seconds.',/,"
        "' Line search           time',1p,e10.3,' seconds.')"),
        cachyt, sbtime, lnscht;
    }
    write(6, format_3008), time;
    if (iprint >= 1) {
      if (info ==  - 4 || info ==  - 9) {
        write(itfile,
          "(2(1x,i4),2(1x,i5),2x,a3,1x,i4,1p,2(2x,d7.1),6x,'-',10x,'-')"),
          iter, nfgv, nseg, nact, word, iback, stp, xstep;
      }
      write(itfile, format_3009), task;
      if (info != 0) {
        if (info ==  - 1) {
          write(itfile, format_9011);
        }
        if (info ==  - 2) {
          write(itfile, format_9012);
        }
        if (info ==  - 3) {
          write(itfile, format_9013);
        }
        if (info ==  - 4) {
          write(itfile, format_9014);
        }
        if (info ==  - 5) {
          write(itfile, format_9015);
        }
        if (info ==  - 8) {
          write(itfile, format_9018);
        }
        if (info ==  - 9) {
          write(itfile, format_9019);
        }
      }
      write(itfile, format_3008), fem::time;
    }
  }
  //C
}

//C
//C======================= The end of prn3lb =============================
//C
void
projgr(
  int const& n,
  arr_cref<double> l,
  arr_cref<double> u,
  arr_cref<int> nbd,
  arr_cref<double> x,
  arr_cref<double> g,
  double& sbgnrm)
{
  l(dimension(n));
  u(dimension(n));
  nbd(dimension(n));
  x(dimension(n));
  g(dimension(n));
  //C
  //C     ************
  //C
  //C     Subroutine projgr
  //C
  //C     This subroutine computes the infinity norm of the projected
  //C       gradient.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  const double zero = 0.0e0;
  sbgnrm = zero;
  int i = fem::int0;
  double gi = fem::double0;
  FEM_DO_SAFE(i, 1, n) {
    gi = g(i);
    if (nbd(i) != 0) {
      if (gi < zero) {
        if (nbd(i) >= 2) {
          gi = fem::max((x(i) - u(i)), gi);
        }
      }
      else {
        if (nbd(i) <= 2) {
          gi = fem::min((x(i) - l(i)), gi);
        }
      }
    }
    sbgnrm = fem::max(sbgnrm, fem::abs(gi));
  }
  //C
}

//C
//C======================= The end of projgr =============================
//C
void
subsm(
  common& cmn,
  int const& n,
  int const& m,
  int const& nsub,
  arr_cref<int> ind,
  arr_cref<double> l,
  arr_cref<double> u,
  arr_cref<int> nbd,
  arr_ref<double> x,
  arr_ref<double> d,
  arr_ref<double> xp,
  arr_cref<double, 2> ws,
  arr_cref<double, 2> wy,
  double const& theta,
  arr_cref<double> xx,
  arr_cref<double> gg,
  int const& col,
  int const& head,
  int& iword,
  arr_ref<double> wv,
  arr_cref<double, 2> wn,
  int const& iprint,
  int& info)
{
  ind(dimension(nsub));
  l(dimension(n));
  u(dimension(n));
  nbd(dimension(n));
  x(dimension(n));
  d(dimension(n));
  xp(dimension(n));
  ws(dimension(n, m));
  wy(dimension(n, m));
  xx(dimension(n));
  gg(dimension(n));
  wv(dimension(2 * m));
  wn(dimension(2 * m, 2 * m));
  common_write write(cmn);
  int pointr = fem::int0;
  int i = fem::int0;
  const double zero = 0.0e0;
  double temp1 = fem::double0;
  double temp2 = fem::double0;
  int j = fem::int0;
  int k = fem::int0;
  int m2 = fem::int0;
  int col2 = fem::int0;
  int jy = fem::int0;
  int js = fem::int0;
  const double one = 1.0e0;
  double dk = fem::double0;
  double xk = fem::double0;
  double dd_p = fem::double0;
  double alpha = fem::double0;
  int ibd = fem::int0;
  //C
  //C     **********************************************************************
  //C
  //C     This routine contains the major changes in the updated version.
  //C     The changes are described in the accompanying paper
  //C
  //C      Jose Luis Morales, Jorge Nocedal
  //C      "Remark On Algorithm 788: L-BFGS-B: Fortran Subroutines for Large-Scale
  //C       Bound Constrained Optimization". Decemmber 27, 2010.
  //C
  //C             J.L. Morales  Departamento de Matematicas,
  //C                           Instituto Tecnologico Autonomo de Mexico
  //C                           Mexico D.F.
  //C
  //C             J, Nocedal    Department of Electrical Engineering and
  //C                           Computer Science.
  //C                           Northwestern University. Evanston, IL. USA
  //C
  //C                           January 17, 2011
  //C
  //C      **********************************************************************
  //C
  //C     Subroutine subsm
  //C
  //C     Given xcp, l, u, r, an index set that specifies
  //C       the active set at xcp, and an l-BFGS matrix B
  //C       (in terms of WY, WS, SY, WT, head, col, and theta),
  //C       this subroutine computes an approximate solution
  //C       of the subspace problem
  //C
  //C       (P)   min Q(x) = r'(x-xcp) + 1/2 (x-xcp)' B (x-xcp)
  //C
  //C             subject to l<=x<=u
  //C                       x_i=xcp_i for all i in A(xcp)
  //C
  //C       along the subspace unconstrained Newton direction
  //C
  //C          d = -(Z'BZ)^(-1) r.
  //C
  //C       The formula for the Newton direction, given the L-BFGS matrix
  //C       and the Sherman-Morrison formula, is
  //C
  //C          d = (1/theta)r + (1/theta*2) Z'WK^(-1)W'Z r.
  //C
  //C       where
  //C                 K = [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
  //C                     [L_a -R_z           theta*S'AA'S ]
  //C
  //C     Note that this procedure for computing d differs
  //C     from that described in [1]. One can show that the matrix K is
  //C     equal to the matrix M^[-1]N in that paper.
  //C
  //C     n is an integer variable.
  //C       On entry n is the dimension of the problem.
  //C       On exit n is unchanged.
  //C
  //C     m is an integer variable.
  //C       On entry m is the maximum number of variable metric corrections
  //C         used to define the limited memory matrix.
  //C       On exit m is unchanged.
  //C
  //C     nsub is an integer variable.
  //C       On entry nsub is the number of free variables.
  //C       On exit nsub is unchanged.
  //C
  //C     ind is an integer array of dimension nsub.
  //C       On entry ind specifies the coordinate indices of free variables.
  //C       On exit ind is unchanged.
  //C
  //C     l is a double precision array of dimension n.
  //C       On entry l is the lower bound of x.
  //C       On exit l is unchanged.
  //C
  //C     u is a double precision array of dimension n.
  //C       On entry u is the upper bound of x.
  //C       On exit u is unchanged.
  //C
  //C     nbd is a integer array of dimension n.
  //C       On entry nbd represents the type of bounds imposed on the
  //C         variables, and must be specified as follows:
  //C         nbd(i)=0 if x(i) is unbounded,
  //C                1 if x(i) has only a lower bound,
  //C                2 if x(i) has both lower and upper bounds, and
  //C                3 if x(i) has only an upper bound.
  //C       On exit nbd is unchanged.
  //C
  //C     x is a double precision array of dimension n.
  //C       On entry x specifies the Cauchy point xcp.
  //C       On exit x(i) is the minimizer of Q over the subspace of
  //C                                                        free variables.
  //C
  //C     d is a double precision array of dimension n.
  //C       On entry d is the reduced gradient of Q at xcp.
  //C       On exit d is the Newton direction of Q.
  //C
  //C    xp is a double precision array of dimension n.
  //C       used to safeguard the projected Newton direction
  //C
  //C    xx is a double precision array of dimension n
  //C       On entry it holds the current iterate
  //C       On output it is unchanged
  //C
  //C    gg is a double precision array of dimension n
  //C       On entry it holds the gradient at the current iterate
  //C       On output it is unchanged
  //C
  //C     ws and wy are double precision arrays;
  //C     theta is a double precision variable;
  //C     col is an integer variable;
  //C     head is an integer variable.
  //C       On entry they store the information defining the
  //C                                          limited memory BFGS matrix:
  //C         ws(n,m) stores S, a set of s-vectors;
  //C         wy(n,m) stores Y, a set of y-vectors;
  //C         theta is the scaling factor specifying B_0 = theta I;
  //C         col is the number of variable metric corrections stored;
  //C         head is the location of the 1st s- (or y-) vector in S (or Y).
  //C       On exit they are unchanged.
  //C
  //C     iword is an integer variable.
  //C       On entry iword is unspecified.
  //C       On exit iword specifies the status of the subspace solution.
  //C         iword = 0 if the solution is in the box,
  //C                 1 if some bound is encountered.
  //C
  //C     wv is a double precision working array of dimension 2m.
  //C
  //C     wn is a double precision array of dimension 2m x 2m.
  //C       On entry the upper triangle of wn stores the LEL^T factorization
  //C         of the indefinite matrix
  //C
  //C              K = [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
  //C                  [L_a -R_z           theta*S'AA'S ]
  //C                                                    where E = [-I  0]
  //C                                                              [ 0  I]
  //C       On exit wn is unchanged.
  //C
  //C     iprint is an INTEGER variable that must be set by the user.
  //C       It controls the frequency and type of output generated:
  //C        iprint<0    no output is generated;
  //C        iprint=0    print only one line at the last iteration;
  //C        0<iprint<99 print also f and |proj g| every iprint iterations;
  //C        iprint=99   print details of every iteration except n-vectors;
  //C        iprint=100  print also the changes of active set and final x;
  //C        iprint>100  print details of every iteration including x and g;
  //C       When iprint > 0, the file iterate.dat will be created to
  //C                        summarize the iteration.
  //C
  //C     info is an integer variable.
  //C       On entry info is unspecified.
  //C       On exit info = 0       for normal return,
  //C                    = nonzero for abnormal return
  //C                                  when the matrix K is ill-conditioned.
  //C
  //C     Subprograms called:
  //C
  //C       Linpack dtrsl.
  //C
  //C     References:
  //C
  //C       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
  //C       memory algorithm for bound constrained optimization'',
  //C       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  if (nsub <= 0) {
    return;
  }
  if (iprint >= 99) {
    write(6, "(/,'----------------SUBSM entered-----------------',/)");
  }
  //C
  //C     Compute wv = W'Zd.
  //C
  pointr = head;
  FEM_DO_SAFE(i, 1, col) {
    temp1 = zero;
    temp2 = zero;
    FEM_DO_SAFE(j, 1, nsub) {
      k = ind(j);
      temp1 += wy(k, pointr) * d(j);
      temp2 += ws(k, pointr) * d(j);
    }
    wv(i) = temp1;
    wv(col + i) = theta * temp2;
    pointr = fem::mod(pointr, m) + 1;
  }
  //C
  //C     Compute wv:=K^(-1)wv.
  //C
  m2 = 2 * m;
  col2 = 2 * col;
  dtrsl(wn, m2, col2, wv, 11, info);
  if (info != 0) {
    return;
  }
  FEM_DO_SAFE(i, 1, col) {
    wv(i) = -wv(i);
  }
  dtrsl(wn, m2, col2, wv, 01, info);
  if (info != 0) {
    return;
  }
  //C
  //C     Compute d = (1/theta)d + (1/theta**2)Z'W wv.
  //C
  pointr = head;
  FEM_DO_SAFE(jy, 1, col) {
    js = col + jy;
    FEM_DO_SAFE(i, 1, nsub) {
      k = ind(i);
      d(i) += wy(k, pointr) * wv(jy) / theta + ws(k, pointr) * wv(js);
    }
    pointr = fem::mod(pointr, m) + 1;
  }
  //C
  dscal(nsub, one / theta, d, 1);
  //C
  //C-----------------------------------------------------------------
  //C     Let us try the projection, d is the Newton direction
  //C
  iword = 0;
  //C
  dcopy(n, x, 1, xp, 1);
  //C
  FEM_DO_SAFE(i, 1, nsub) {
    k = ind(i);
    dk = d(i);
    xk = x(k);
    if (nbd(k) != 0) {
      //C
      //C lower bounds only
      if (nbd(k) == 1) {
        x(k) = fem::max(l(k), xk + dk);
        if (x(k) == l(k)) {
          iword = 1;
        }
      }
      else {
        //C
        //C upper and lower bounds
        if (nbd(k) == 2) {
          xk = fem::max(l(k), xk + dk);
          x(k) = fem::min(u(k), xk);
          if (x(k) == l(k) || x(k) == u(k)) {
            iword = 1;
          }
        }
        else {
          //C
          //C upper bounds only
          if (nbd(k) == 3) {
            x(k) = fem::min(u(k), xk + dk);
            if (x(k) == u(k)) {
              iword = 1;
            }
          }
        }
      }
      //C
      //C free variables
    }
    else {
      x(k) = xk + dk;
    }
  }
  //C
  if (iword == 0) {
    goto statement_911;
  }
  //C
  //C     check sign of the directional derivative
  //C
  dd_p = zero;
  FEM_DO_SAFE(i, 1, n) {
    dd_p += (x(i) - xx(i)) * gg(i);
  }
  if (dd_p > zero) {
    dcopy(n, xp, 1, x, 1);
    write(6, star), " Positive dir derivative in projection ";
    write(6, star), " Using the backtracking step ";
  }
  else {
    goto statement_911;
  }
  //C
  //C-----------------------------------------------------------------
  //C
  alpha = one;
  temp1 = alpha;
  ibd = 0;
  FEM_DO_SAFE(i, 1, nsub) {
    k = ind(i);
    dk = d(i);
    if (nbd(k) != 0) {
      if (dk < zero && nbd(k) <= 2) {
        temp2 = l(k) - x(k);
        if (temp2 >= zero) {
          temp1 = zero;
        }
        else if (dk * alpha < temp2) {
          temp1 = temp2 / dk;
        }
      }
      else if (dk > zero && nbd(k) >= 2) {
        temp2 = u(k) - x(k);
        if (temp2 <= zero) {
          temp1 = zero;
        }
        else if (dk * alpha > temp2) {
          temp1 = temp2 / dk;
        }
      }
      if (temp1 < alpha) {
        alpha = temp1;
        ibd = i;
      }
    }
  }
  //C
  if (alpha < one) {
    dk = d(ibd);
    k = ind(ibd);
    if (dk > zero) {
      x(k) = u(k);
      d(ibd) = zero;
    }
    else if (dk < zero) {
      x(k) = l(k);
      d(ibd) = zero;
    }
  }
  FEM_DO_SAFE(i, 1, nsub) {
    k = ind(i);
    x(k) += alpha * d(i);
  }
  //Cccccc
  statement_911:
  //C
  if (iprint >= 99) {
    write(6, "(/,'----------------exit SUBSM --------------------',/)");
  }
  //C
}

//C
//C  L-BFGS-B is released under the “New BSD License” (aka “Modified BSD License”
//C  or “3-clause license”)
//C  Please read attached file License.txt
//C
void
timer(
  double& ttime)
{
  //C
  //C     This routine computes cpu time in double precision; it makes use of
  //C     the intrinsic f90 cpu_time therefore a conversion type is
  //C     needed.
  //C
  //C           J.L Morales  Departamento de Matematicas,
  //C                        Instituto Tecnologico Autonomo de Mexico
  //C                        Mexico D.F.
  //C
  //C           J.L Nocedal  Department of Electrical Engineering and
  //C                        Computer Science.
  //C                        Northwestern University. Evanston, IL. USA
  //C
  //C                        January 21, 2011
  //C
  float temp = fem::sngl(ttime);
  fem::cpu_time(temp);
  ttime = fem::dble(temp);
  //C
}

//C
//C======================= The end of setulb =============================
//C
void
mainlb(
  common& cmn,
  int const& n,
  int const& m,
  arr_ref<double> x,
  arr_cref<double> l,
  arr_cref<double> u,
  arr_cref<int> nbd,
  double& f,
  arr_ref<double> g,
  double const& factr,
  double const& pgtol,
  arr_ref<double, 2> ws,
  arr_ref<double, 2> wy,
  arr_ref<double, 2> sy,
  arr_ref<double, 2> ss,
  arr_ref<double, 2> wt,
  arr_ref<double, 2> wn,
  arr_ref<double, 2> snd,
  arr_ref<double> z,
  arr_ref<double> r,
  arr_ref<double> d,
  arr_ref<double> t,
  arr_ref<double> xp,
  arr_ref<double> wa,
  arr_ref<int> index,
  arr_ref<int> iwhere,
  arr_ref<int> indx2,
  str_ref task,
  int const& iprint,
  str_ref csave,
  arr_ref<bool> lsave,
  arr_ref<int> isave,
  arr_ref<double> dsave)
{
  x(dimension(n));
  l(dimension(n));
  u(dimension(n));
  nbd(dimension(n));
  g(dimension(n));
  ws(dimension(n, m));
  wy(dimension(n, m));
  sy(dimension(m, m));
  ss(dimension(m, m));
  wt(dimension(m, m));
  wn(dimension(2 * m, 2 * m));
  snd(dimension(2 * m, 2 * m));
  z(dimension(n));
  r(dimension(n));
  d(dimension(n));
  t(dimension(n));
  xp(dimension(n));
  wa(dimension(8 * m));
  index(dimension(n));
  iwhere(dimension(n));
  indx2(dimension(n));
  lsave(dimension(4));
  isave(dimension(23));
  dsave(dimension(29));
  common_write write(cmn);
  const double one = 1.0e0;
  double epsmch = fem::double0;
  double time1 = fem::double0;
  int col = fem::int0;
  int head = fem::int0;
  double theta = fem::double0;
  int iupdat = fem::int0;
  bool updatd = fem::bool0;
  int iback = fem::int0;
  int itail = fem::int0;
  int iword = fem::int0;
  int nact = fem::int0;
  int ileave = fem::int0;
  int nenter = fem::int0;
  const double zero = 0.0e0;
  double fold = fem::double0;
  double dnorm = fem::double0;
  double cpu1 = fem::double0;
  double gd = fem::double0;
  double stpmx = fem::double0;
  double sbgnrm = fem::double0;
  double stp = fem::double0;
  double gdold = fem::double0;
  double dtd = fem::double0;
  int iter = fem::int0;
  int nfgv = fem::int0;
  int nseg = fem::int0;
  int nintol = fem::int0;
  int nskip = fem::int0;
  int nfree = fem::int0;
  int ifun = fem::int0;
  double tol = fem::double0;
  double cachyt = fem::double0;
  double sbtime = fem::double0;
  double lnscht = fem::double0;
  fem::str<3> word = fem::char0;
  int info = fem::int0;
  int itfile = fem::int0;
  int k = fem::int0;
  double xstep = fem::double0;
  bool prjctd = fem::bool0;
  bool cnstnd = fem::bool0;
  bool boxed = fem::bool0;
  bool wrk = fem::bool0;
  double cpu2 = fem::double0;
  int i = fem::int0;
  double ddum = fem::double0;
  double rr = fem::double0;
  double dr = fem::double0;
  double time2 = fem::double0;
  double time = fem::double0;
  static const char* format_1005 =
    "(/,' Singular triangular system detected;',/,"
    "'   refresh the lbfgs memory and restart the iteration.')";
  //C-jlm-jn
  //C
  //C     ************
  //C
  //C     Subroutine mainlb
  //C
  //C     This subroutine solves bound constrained optimization problems by
  //C       using the compact formula of the limited memory BFGS updates.
  //C
  //C     n is an integer variable.
  //C       On entry n is the number of variables.
  //C       On exit n is unchanged.
  //C
  //C     m is an integer variable.
  //C       On entry m is the maximum number of variable metric
  //C          corrections allowed in the limited memory matrix.
  //C       On exit m is unchanged.
  //C
  //C     x is a double precision array of dimension n.
  //C       On entry x is an approximation to the solution.
  //C       On exit x is the current approximation.
  //C
  //C     l is a double precision array of dimension n.
  //C       On entry l is the lower bound of x.
  //C       On exit l is unchanged.
  //C
  //C     u is a double precision array of dimension n.
  //C       On entry u is the upper bound of x.
  //C       On exit u is unchanged.
  //C
  //C     nbd is an integer array of dimension n.
  //C       On entry nbd represents the type of bounds imposed on the
  //C         variables, and must be specified as follows:
  //C         nbd(i)=0 if x(i) is unbounded,
  //C                1 if x(i) has only a lower bound,
  //C                2 if x(i) has both lower and upper bounds,
  //C                3 if x(i) has only an upper bound.
  //C       On exit nbd is unchanged.
  //C
  //C     f is a double precision variable.
  //C       On first entry f is unspecified.
  //C       On final exit f is the value of the function at x.
  //C
  //C     g is a double precision array of dimension n.
  //C       On first entry g is unspecified.
  //C       On final exit g is the value of the gradient at x.
  //C
  //C     factr is a double precision variable.
  //C       On entry factr >= 0 is specified by the user.  The iteration
  //C         will stop when
  //C
  //C         (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr*epsmch
  //C
  //C         where epsmch is the machine precision, which is automatically
  //C         generated by the code.
  //C       On exit factr is unchanged.
  //C
  //C     pgtol is a double precision variable.
  //C       On entry pgtol >= 0 is specified by the user.  The iteration
  //C         will stop when
  //C
  //C                 max{|proj g_i | i = 1, ..., n} <= pgtol
  //C
  //C         where pg_i is the ith component of the projected gradient.
  //C       On exit pgtol is unchanged.
  //C
  //C     ws, wy, sy, and wt are double precision working arrays used to
  //C       store the following information defining the limited memory
  //C          BFGS matrix:
  //C          ws, of dimension n x m, stores S, the matrix of s-vectors;
  //C          wy, of dimension n x m, stores Y, the matrix of y-vectors;
  //C          sy, of dimension m x m, stores S'Y;
  //C          ss, of dimension m x m, stores S'S;
  //C          yy, of dimension m x m, stores Y'Y;
  //C          wt, of dimension m x m, stores the Cholesky factorization
  //C                                  of (theta*S'S+LD^(-1)L'); see eq.
  //C                                  (2.26) in [3].
  //C
  //C     wn is a double precision working array of dimension 2m x 2m
  //C       used to store the LEL^T factorization of the indefinite matrix
  //C                 K = [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
  //C                     [L_a -R_z           theta*S'AA'S ]
  //C
  //C       where     E = [-I  0]
  //C                     [ 0  I]
  //C
  //C     snd is a double precision working array of dimension 2m x 2m
  //C       used to store the lower triangular part of
  //C                 N = [Y' ZZ'Y   L_a'+R_z']
  //C                     [L_a +R_z  S'AA'S   ]
  //C
  //C     z(n),r(n),d(n),t(n), xp(n),wa(8*m) are double precision working arrays.
  //C       z  is used at different times to store the Cauchy point and
  //C          the Newton point.
  //C       xp is used to safeguard the projected Newton direction
  //C
  //C     sg(m),sgo(m),yg(m),ygo(m) are double precision working arrays.
  //C
  //C     index is an integer working array of dimension n.
  //C       In subroutine freev, index is used to store the free and fixed
  //C          variables at the Generalized Cauchy Point (GCP).
  //C
  //C     iwhere is an integer working array of dimension n used to record
  //C       the status of the vector x for GCP computation.
  //C       iwhere(i)=0 or -3 if x(i) is free and has bounds,
  //C                 1       if x(i) is fixed at l(i), and l(i) .ne. u(i)
  //C                 2       if x(i) is fixed at u(i), and u(i) .ne. l(i)
  //C                 3       if x(i) is always fixed, i.e.,  u(i)=x(i)=l(i)
  //C                -1       if x(i) is always free, i.e., no bounds on it.
  //C
  //C     indx2 is an integer working array of dimension n.
  //C       Within subroutine cauchy, indx2 corresponds to the array iorder.
  //C       In subroutine freev, a list of variables entering and leaving
  //C       the free set is stored in indx2, and it is passed on to
  //C       subroutine formk with this information.
  //C
  //C     task is a working string of characters of length 60 indicating
  //C       the current job when entering and leaving this subroutine.
  //C
  //C     iprint is an INTEGER variable that must be set by the user.
  //C       It controls the frequency and type of output generated:
  //C        iprint<0    no output is generated;
  //C        iprint=0    print only one line at the last iteration;
  //C        0<iprint<99 print also f and |proj g| every iprint iterations;
  //C        iprint=99   print details of every iteration except n-vectors;
  //C        iprint=100  print also the changes of active set and final x;
  //C        iprint>100  print details of every iteration including x and g;
  //C       When iprint > 0, the file iterate.dat will be created to
  //C                        summarize the iteration.
  //C
  //C     csave is a working string of characters of length 60.
  //C
  //C     lsave is a logical working array of dimension 4.
  //C
  //C     isave is an integer working array of dimension 23.
  //C
  //C     dsave is a double precision working array of dimension 29.
  //C
  //C     Subprograms called
  //C
  //C       L-BFGS-B Library ... cauchy, subsm, lnsrlb, formk,
  //C
  //C        errclb, prn1lb, prn2lb, prn3lb, active, projgr,
  //C
  //C        freev, cmprlb, matupd, formt.
  //C
  //C       Minpack2 Library ... timer
  //C
  //C       Linpack Library ... dcopy, ddot.
  //C
  //C     References:
  //C
  //C       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
  //C       memory algorithm for bound constrained optimization'',
  //C       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.
  //C
  //C       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
  //C       Subroutines for Large Scale Bound Constrained Optimization''
  //C       Tech. Report, NAM-11, EECS Department, Northwestern University,
  //C       1994.
  //C
  //C       [3] R. Byrd, J. Nocedal and R. Schnabel "Representations of
  //C       Quasi-Newton Matrices and their use in Limited Memory Methods'',
  //C       Mathematical Programming 63 (1994), no. 4, pp. 129-156.
  //C
  //C       (Postscript files of these papers are available via anonymous
  //C        ftp to eecs.nwu.edu in the directory pub/lbfgs/lbfgs_bcm.)
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C
  if (task == "START") {
    //C
    epsmch = epsilon(one);
    //C
    timer(time1);
    //C
    //C        Initialize counters and scalars when task='START'.
    //C
    //C           for the limited memory BFGS matrices:
    col = 0;
    head = 1;
    theta = one;
    iupdat = 0;
    updatd = false;
    iback = 0;
    itail = 0;
    iword = 0;
    nact = 0;
    ileave = 0;
    nenter = 0;
    fold = zero;
    dnorm = zero;
    cpu1 = zero;
    gd = zero;
    stpmx = zero;
    sbgnrm = zero;
    stp = zero;
    gdold = zero;
    dtd = zero;
    //C
    //C           for operation counts:
    iter = 0;
    nfgv = 0;
    nseg = 0;
    nintol = 0;
    nskip = 0;
    nfree = n;
    ifun = 0;
    //C           for stopping tolerance:
    tol = factr * epsmch;
    //C
    //C           for measuring running time:
    cachyt = 0;
    sbtime = 0;
    lnscht = 0;
    //C
    //C           'word' records the status of subspace solutions.
    word = "---";
    //C
    //C           'info' records the termination information.
    info = 0;
    //C
    itfile = 8;
    if (iprint >= 1) {
      //C                                open a summary file 'iterate.dat'
      cmn.io.open(8, "iterate.dat")
        .status("unknown");
    }
    //C
    //C        Check the input arguments for errors.
    //C
    errclb(n, m, factr, l, u, nbd, task, info, k);
    if (task(1, 5) == "ERROR") {
      prn3lb(cmn, n, x, f, task, iprint, info, itfile, iter, nfgv,
        nintol, nskip, nact, sbgnrm, zero, nseg, word, iback, stp,
        xstep, k, cachyt, sbtime, lnscht);
      return;
    }
    //C
    prn1lb(cmn, n, m, l, u, x, iprint, itfile, epsmch);
    //C
    //C        Initialize iwhere & project x onto the feasible set.
    //C
    active(cmn, n, l, u, nbd, x, iwhere, iprint, prjctd, cnstnd, boxed);
    //C
    //C        The end of the initialization.
    //C
  }
  else {
    //C          restore local variables.
    //C
    prjctd = lsave(1);
    cnstnd = lsave(2);
    boxed = lsave(3);
    updatd = lsave(4);
    //C
    nintol = isave(1);
    itfile = isave(3);
    iback = isave(4);
    nskip = isave(5);
    head = isave(6);
    col = isave(7);
    itail = isave(8);
    iter = isave(9);
    iupdat = isave(10);
    nseg = isave(12);
    nfgv = isave(13);
    info = isave(14);
    ifun = isave(15);
    iword = isave(16);
    nfree = isave(17);
    nact = isave(18);
    ileave = isave(19);
    nenter = isave(20);
    //C
    theta = dsave(1);
    fold = dsave(2);
    tol = dsave(3);
    dnorm = dsave(4);
    epsmch = dsave(5);
    cpu1 = dsave(6);
    cachyt = dsave(7);
    sbtime = dsave(8);
    lnscht = dsave(9);
    time1 = dsave(10);
    gd = dsave(11);
    stpmx = dsave(12);
    sbgnrm = dsave(13);
    stp = dsave(14);
    gdold = dsave(15);
    dtd = dsave(16);
    //C
    //C        After returning from the driver go to the point where execution
    //C        is to resume.
    //C
    if (task(1, 5) == "FG_LN") {
      goto statement_666;
    }
    if (task(1, 5) == "NEW_X") {
      goto statement_777;
    }
    if (task(1, 5) == "FG_ST") {
      goto statement_111;
    }
    if (task(1, 4) == "STOP") {
      if (task(7, 9) == "CPU") {
        //C                                          restore the previous iterate.
        dcopy(n, t, 1, x, 1);
        dcopy(n, r, 1, g, 1);
        f = fold;
      }
      goto statement_999;
    }
  }
  //C
  //C     Compute f0 and g0.
  //C
  task = "FG_START";
  //C          return to the driver to calculate f and g; reenter at 111.
  goto statement_1000;
  statement_111:
  nfgv = 1;
  //C
  //C     Compute the infinity norm of the (-) projected gradient.
  //C
  projgr(n, l, u, nbd, x, g, sbgnrm);
  //C
  if (iprint >= 1) {
    write(6,
      "(/,'At iterate',i5,4x,'f= ',1p,d12.5,4x,'|proj g|= ',1p,d12.5)"),
      iter, f, sbgnrm;
    write(itfile,
      "(2(1x,i4),5x,'-',5x,'-',3x,'-',5x,'-',5x,'-',8x,'-',3x,1p,2(1x,d10.3))"),
      iter, nfgv, sbgnrm, f;
  }
  if (sbgnrm <= pgtol) {
    //C                                terminate the algorithm.
    task = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL";
    goto statement_999;
  }
  //C
  //C ----------------- the beginning of the loop --------------------------
  //C
  statement_222:
  if (iprint >= 99) {
    write(6, "(/,/,'ITERATION ',i5)"), iter + 1;
  }
  iword = -1;
  //C
  if (!cnstnd && col > 0) {
    //C                                            skip the search for GCP.
    dcopy(n, x, 1, z, 1);
    wrk = updatd;
    nseg = 0;
    goto statement_333;
  }
  //C
  //Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  //C
  //C     Compute the Generalized Cauchy Point (GCP).
  //C
  //Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  //C
  timer(cpu1);
  cauchy(cmn, n, x, l, u, nbd, g, indx2, iwhere, t, d, z, m, wy, ws,
    sy, wt, theta, col, head, wa(1), wa(2 * m + 1), wa(4 * m + 1), wa(
    6 * m + 1), nseg, iprint, sbgnrm, info, epsmch);
  if (info != 0) {
    //C         singular triangular system detected; refresh the lbfgs memory.
    if (iprint >= 1) {
      write(6, format_1005);
    }
    info = 0;
    col = 0;
    head = 1;
    theta = one;
    iupdat = 0;
    updatd = false;
    timer(cpu2);
    cachyt += cpu2 - cpu1;
    goto statement_222;
  }
  timer(cpu2);
  cachyt += cpu2 - cpu1;
  nintol += nseg;
  //C
  //C     Count the entering and leaving variables for iter > 0;
  //C     find the index set of free and active variables at the GCP.
  //C
  freev(cmn, n, nfree, index, nenter, ileave, indx2, iwhere, wrk,
    updatd, cnstnd, iprint, iter);
  nact = n - nfree;
  //C
  statement_333:
  //C
  //C     If there are no free variables or B=theta*I, then
  //C                                        skip the subspace minimization.
  //C
  if (nfree == 0 || col == 0) {
    goto statement_555;
  }
  //C
  //Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  //C
  //C     Subspace minimization.
  //C
  //Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  //C
  timer(cpu1);
  //C
  //C     Form  the LEL^T factorization of the indefinite
  //C       matrix    K = [-D -Y'ZZ'Y/theta     L_a'-R_z'  ]
  //C                     [L_a -R_z           theta*S'AA'S ]
  //C       where     E = [-I  0]
  //C                     [ 0  I]
  //C
  if (wrk) {
    formk(n, nfree, index, nenter, ileave, indx2, iupdat, updatd, wn,
      snd, m, ws, wy, sy, theta, col, head, info);
  }
  if (info != 0) {
    //C          nonpositive definiteness in Cholesky factorization;
    //C          refresh the lbfgs memory and restart the iteration.
    if (iprint >= 1) {
      write(6,
        "(/,' Nonpositive definiteness in Cholesky factorization in formk;',/,"
        "'   refresh the lbfgs memory and restart the iteration.')");
    }
    info = 0;
    col = 0;
    head = 1;
    theta = one;
    iupdat = 0;
    updatd = false;
    timer(cpu2);
    sbtime += cpu2 - cpu1;
    goto statement_222;
  }
  //C
  //C        compute r=-Z'B(xcp-xk)-Z'g (using wa(2m+1)=W'(xcp-x)
  //C                                                   from 'cauchy').
  cmprlb(n, m, x, g, ws, wy, sy, wt, z, r, wa, index, theta, col,
    head, nfree, cnstnd, info);
  if (info != 0) {
    goto statement_444;
  }
  //C
  //C-jlm-jn   call the direct method.
  //C
  subsm(cmn, n, m, nfree, index, l, u, nbd, z, r, xp, ws, wy, theta,
    x, g, col, head, iword, wa, wn, iprint, info);
  statement_444:
  if (info != 0) {
    //C          singular triangular system detected;
    //C          refresh the lbfgs memory and restart the iteration.
    if (iprint >= 1) {
      write(6, format_1005);
    }
    info = 0;
    col = 0;
    head = 1;
    theta = one;
    iupdat = 0;
    updatd = false;
    timer(cpu2);
    sbtime += cpu2 - cpu1;
    goto statement_222;
  }
  //C
  timer(cpu2);
  sbtime += cpu2 - cpu1;
  statement_555:
  //C
  //Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  //C
  //C     Line search and optimality tests.
  //C
  //Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  //C
  //C     Generate the search direction d:=z-x.
  //C
  FEM_DO_SAFE(i, 1, n) {
    d(i) = z(i) - x(i);
  }
  timer(cpu1);
  statement_666:
  lnsrlb(cmn, n, l, u, nbd, x, f, fold, gd, gdold, g, d, r, t, z,
    stp, dnorm, dtd, xstep, stpmx, iter, ifun, iback, nfgv, info,
    task, boxed, cnstnd, csave, isave(22), dsave(17));
  if (info != 0 || iback >= 20) {
    //C          restore the previous iterate.
    dcopy(n, t, 1, x, 1);
    dcopy(n, r, 1, g, 1);
    f = fold;
    if (col == 0) {
      //C             abnormal termination.
      if (info == 0) {
        info = -9;
        //C                restore the actual number of f and g evaluations etc.
        nfgv = nfgv - 1;
        ifun = ifun - 1;
        iback = iback - 1;
      }
      task = "ABNORMAL_TERMINATION_IN_LNSRCH";
      iter++;
      goto statement_999;
    }
    else {
      //C             refresh the lbfgs memory and restart the iteration.
      if (iprint >= 1) {
        write(6,
          "(/,' Bad direction in the line search;',/,"
          "'   refresh the lbfgs memory and restart the iteration.')");
      }
      if (info == 0) {
        nfgv = nfgv - 1;
      }
      info = 0;
      col = 0;
      head = 1;
      theta = one;
      iupdat = 0;
      updatd = false;
      task = "RESTART_FROM_LNSRCH";
      timer(cpu2);
      lnscht += cpu2 - cpu1;
      goto statement_222;
    }
  }
  else if (task(1, 5) == "FG_LN") {
    //C          return to the driver for calculating f and g; reenter at 666.
    goto statement_1000;
  }
  else {
    //C          calculate and print out the quantities related to the new X.
    timer(cpu2);
    lnscht += cpu2 - cpu1;
    iter++;
    //C
    //C        Compute the infinity norm of the projected (-)gradient.
    //C
    projgr(n, l, u, nbd, x, g, sbgnrm);
    //C
    //C        Print iteration information.
    //C
    prn2lb(cmn, n, x, f, g, iprint, itfile, iter, nfgv, nact, sbgnrm,
      nseg, word, iword, iback, stp, xstep);
    goto statement_1000;
  }
  statement_777:
  //C
  //C     Test for termination.
  //C
  if (sbgnrm <= pgtol) {
    //C                                terminate the algorithm.
    task = "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL";
    goto statement_999;
  }
  //C
  ddum = fem::max(fem::abs(fold), fem::abs(f), one);
  if ((fold - f) <= tol * ddum) {
    //C                                        terminate the algorithm.
    task = "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH";
    if (iback >= 10) {
      info = -5;
    }
    //C           i.e., to issue a warning if iback>10 in the line search.
    goto statement_999;
  }
  //C
  //C     Compute d=newx-oldx, r=newg-oldg, rr=y'y and dr=y's.
  //C
  FEM_DO_SAFE(i, 1, n) {
    r(i) = g(i) - r(i);
  }
  rr = ddot(n, r, 1, r, 1);
  if (stp == one) {
    dr = gd - gdold;
    ddum = -gdold;
  }
  else {
    dr = (gd - gdold) * stp;
    dscal(n, stp, d, 1);
    ddum = -gdold * stp;
  }
  //C
  if (dr <= epsmch * ddum) {
    //C                            skip the L-BFGS update.
    nskip++;
    updatd = false;
    if (iprint >= 1) {
      write(6,
        "('  ys=',1p,e10.3,'  -gs=',1p,e10.3,' BFGS update SKIPPED')"),
        dr, ddum;
    }
    goto statement_888;
  }
  //C
  //Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  //C
  //C     Update the L-BFGS matrix.
  //C
  //Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  //C
  updatd = true;
  iupdat++;
  //C
  //C     Update matrices WS and WY and form the middle matrix in B.
  //C
  matupd(n, m, ws, wy, sy, ss, d, r, itail, iupdat, col, head, theta,
    rr, dr, stp, dtd);
  //C
  //C     Form the upper half of the pds T = theta*SS + L*D^(-1)*L';
  //C        Store T in the upper triangular of the array wt;
  //C        Cholesky factorize T to J*J' with
  //C           J' stored in the upper triangular of wt.
  //C
  formt(m, wt, sy, ss, col, theta, info);
  //C
  if (info != 0) {
    //C          nonpositive definiteness in Cholesky factorization;
    //C          refresh the lbfgs memory and restart the iteration.
    if (iprint >= 1) {
      write(6,
        "(/,' Nonpositive definiteness in Cholesky factorization in formt;',/,"
        "'   refresh the lbfgs memory and restart the iteration.')");
    }
    info = 0;
    col = 0;
    head = 1;
    theta = one;
    iupdat = 0;
    updatd = false;
    goto statement_222;
  }
  //C
  //C     Now the inverse of the middle matrix in B is
  //C
  //C       [  D^(1/2)      O ] [ -D^(1/2)  D^(-1/2)*L' ]
  //C       [ -L*D^(-1/2)   J ] [  0        J'          ]
  //C
  statement_888:
  //C
  //C -------------------- the end of the loop -----------------------------
  //C
  goto statement_222;
  statement_999:
  timer(time2);
  time = time2 - time1;
  prn3lb(cmn, n, x, f, task, iprint, info, itfile, iter, nfgv,
    nintol, nskip, nact, sbgnrm, time, nseg, word, iback, stp,
    xstep, k, cachyt, sbtime, lnscht);
  statement_1000:
  //C
  //C     Save local variables.
  //C
  lsave(1) = prjctd;
  lsave(2) = cnstnd;
  lsave(3) = boxed;
  lsave(4) = updatd;
  //C
  isave(1) = nintol;
  isave(3) = itfile;
  isave(4) = iback;
  isave(5) = nskip;
  isave(6) = head;
  isave(7) = col;
  isave(8) = itail;
  isave(9) = iter;
  isave(10) = iupdat;
  isave(12) = nseg;
  isave(13) = nfgv;
  isave(14) = info;
  isave(15) = ifun;
  isave(16) = iword;
  isave(17) = nfree;
  isave(18) = nact;
  isave(19) = ileave;
  isave(20) = nenter;
  //C
  dsave(1) = theta;
  dsave(2) = fold;
  dsave(3) = tol;
  dsave(4) = dnorm;
  dsave(5) = epsmch;
  dsave(6) = cpu1;
  dsave(7) = cachyt;
  dsave(8) = sbtime;
  dsave(9) = lnscht;
  dsave(10) = time1;
  dsave(11) = gd;
  dsave(12) = stpmx;
  dsave(13) = sbgnrm;
  dsave(14) = stp;
  dsave(15) = gdold;
  dsave(16) = dtd;
  //C
}

//C
//C  L-BFGS-B is released under the “New BSD License” (aka “Modified BSD License”
//C  or “3-clause license”)
//C  Please read attached file License.txt
//C
//C===========   L-BFGS-B (version 3.0.  April 25, 2011  ===================
//C
//C     This is a modified version of L-BFGS-B. Minor changes in the updated
//C     code appear preceded by a line comment as follows
//C
//C     c-jlm-jn
//C
//C     Major changes are described in the accompanying paper:
//C
//C         Jorge Nocedal and Jose Luis Morales, Remark on "Algorithm 778:
//C         L-BFGS-B: Fortran Subroutines for Large-Scale Bound Constrained
//C         Optimization"  (2011). To appear in  ACM Transactions on
//C         Mathematical Software,
//C
//C     The paper describes an improvement and a correction to Algorithm 778.
//C     It is shown that the performance of the algorithm can be improved
//C     significantly by making a relatively simple modication to the subspace
//C     minimization phase. The correction concerns an error caused by the use
//C     of routine dpmeps to estimate machine precision.
//C
//C     The total work space **wa** required by the new version is
//C
//C                  2*m*n + 11m*m + 5*n + 8*m
//C
//C     the old version required
//C
//C                  2*m*n + 12m*m + 4*n + 12*m
//C
//C            J. Nocedal  Department of Electrical Engineering and
//C                        Computer Science.
//C                        Northwestern University. Evanston, IL. USA
//C
//C           J.L Morales  Departamento de Matematicas,
//C                        Instituto Tecnologico Autonomo de Mexico
//C                        Mexico D.F. Mexico.
//C
//C                        March  2011
//C
//C=============================================================================
void
setulb(
  common& cmn,
  int const& n,
  int const& m,
  arr_ref<double> x,
  arr_cref<double> l,
  arr_cref<double> u,
  arr_cref<int> nbd,
  double& f,
  arr_ref<double> g,
  double const& factr,
  double const& pgtol,
  arr_ref<double> wa,
  arr_ref<int> iwa,
  str_ref task,
  int const& iprint,
  str_ref csave,
  arr_ref<bool> lsave,
  arr_ref<int> isave,
  arr_ref<double> dsave)
{
  x(dimension(n));
  l(dimension(n));
  u(dimension(n));
  nbd(dimension(n));
  g(dimension(n));
  wa(dimension(2 * m * n + 5 * n + 11 * m * m + 8 * m));
  iwa(dimension(3 * n));
  lsave(dimension(4));
  isave(dimension(44));
  dsave(dimension(29));
  //C
  //C-jlm-jn
  //C
  //C     ************
  //C
  //C     Subroutine setulb
  //C
  //C     This subroutine partitions the working arrays wa and iwa, and
  //C       then uses the limited memory BFGS method to solve the bound
  //C       constrained optimization problem by calling mainlb.
  //C       (The direct method will be used in the subspace minimization.)
  //C
  //C     n is an integer variable.
  //C       On entry n is the dimension of the problem.
  //C       On exit n is unchanged.
  //C
  //C     m is an integer variable.
  //C       On entry m is the maximum number of variable metric corrections
  //C         used to define the limited memory matrix.
  //C       On exit m is unchanged.
  //C
  //C     x is a double precision array of dimension n.
  //C       On entry x is an approximation to the solution.
  //C       On exit x is the current approximation.
  //C
  //C     l is a double precision array of dimension n.
  //C       On entry l is the lower bound on x.
  //C       On exit l is unchanged.
  //C
  //C     u is a double precision array of dimension n.
  //C       On entry u is the upper bound on x.
  //C       On exit u is unchanged.
  //C
  //C     nbd is an integer array of dimension n.
  //C       On entry nbd represents the type of bounds imposed on the
  //C         variables, and must be specified as follows:
  //C         nbd(i)=0 if x(i) is unbounded,
  //C                1 if x(i) has only a lower bound,
  //C                2 if x(i) has both lower and upper bounds, and
  //C                3 if x(i) has only an upper bound.
  //C       On exit nbd is unchanged.
  //C
  //C     f is a double precision variable.
  //C       On first entry f is unspecified.
  //C       On final exit f is the value of the function at x.
  //C
  //C     g is a double precision array of dimension n.
  //C       On first entry g is unspecified.
  //C       On final exit g is the value of the gradient at x.
  //C
  //C     factr is a double precision variable.
  //C       On entry factr >= 0 is specified by the user.  The iteration
  //C         will stop when
  //C
  //C         (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr*epsmch
  //C
  //C         where epsmch is the machine precision, which is automatically
  //C         generated by the code. Typical values for factr: 1.d+12 for
  //C         low accuracy; 1.d+7 for moderate accuracy; 1.d+1 for extremely
  //C         high accuracy.
  //C       On exit factr is unchanged.
  //C
  //C     pgtol is a double precision variable.
  //C       On entry pgtol >= 0 is specified by the user.  The iteration
  //C         will stop when
  //C
  //C                 max{|proj g_i | i = 1, ..., n} <= pgtol
  //C
  //C         where pg_i is the ith component of the projected gradient.
  //C       On exit pgtol is unchanged.
  //C
  //C     wa is a double precision working array of length
  //C       (2mmax + 5)nmax + 12mmax^2 + 12mmax.
  //C
  //C     iwa is an integer working array of length 3nmax.
  //C
  //C     task is a working string of characters of length 60 indicating
  //C       the current job when entering and quitting this subroutine.
  //C
  //C     iprint is an integer variable that must be set by the user.
  //C       It controls the frequency and type of output generated:
  //C        iprint<0    no output is generated;
  //C        iprint=0    print only one line at the last iteration;
  //C        0<iprint<99 print also f and |proj g| every iprint iterations;
  //C        iprint=99   print details of every iteration except n-vectors;
  //C        iprint=100  print also the changes of active set and final x;
  //C        iprint>100  print details of every iteration including x and g;
  //C       When iprint > 0, the file iterate.dat will be created to
  //C                        summarize the iteration.
  //C
  //C     csave is a working string of characters of length 60.
  //C
  //C     lsave is a logical working array of dimension 4.
  //C       On exit with 'task' = NEW_X, the following information is
  //C                                                             available:
  //C         If lsave(1) = .true.  then  the initial X has been replaced by
  //C                                     its projection in the feasible set;
  //C         If lsave(2) = .true.  then  the problem is constrained;
  //C         If lsave(3) = .true.  then  each variable has upper and lower
  //C                                     bounds;
  //C
  //C     isave is an integer working array of dimension 44.
  //C       On exit with 'task' = NEW_X, the following information is
  //C                                                             available:
  //C         isave(22) = the total number of intervals explored in the
  //C                         search of Cauchy points;
  //C         isave(26) = the total number of skipped BFGS updates before
  //C                         the current iteration;
  //C         isave(30) = the number of current iteration;
  //C         isave(31) = the total number of BFGS updates prior the current
  //C                         iteration;
  //C         isave(33) = the number of intervals explored in the search of
  //C                         Cauchy point in the current iteration;
  //C         isave(34) = the total number of function and gradient
  //C                         evaluations;
  //C         isave(36) = the number of function value or gradient
  //C                                  evaluations in the current iteration;
  //C         if isave(37) = 0  then the subspace argmin is within the box;
  //C         if isave(37) = 1  then the subspace argmin is beyond the box;
  //C         isave(38) = the number of free variables in the current
  //C                         iteration;
  //C         isave(39) = the number of active constraints in the current
  //C                         iteration;
  //C         n + 1 - isave(40) = the number of variables leaving the set of
  //C                           active constraints in the current iteration;
  //C         isave(41) = the number of variables entering the set of active
  //C                         constraints in the current iteration.
  //C
  //C     dsave is a double precision working array of dimension 29.
  //C       On exit with 'task' = NEW_X, the following information is
  //C                                                             available:
  //C         dsave(1) = current 'theta' in the BFGS matrix;
  //C         dsave(2) = f(x) in the previous iteration;
  //C         dsave(3) = factr*epsmch;
  //C         dsave(4) = 2-norm of the line search direction vector;
  //C         dsave(5) = the machine precision epsmch generated by the code;
  //C         dsave(7) = the accumulated time spent on searching for
  //C                                                         Cauchy points;
  //C         dsave(8) = the accumulated time spent on
  //C                                                 subspace minimization;
  //C         dsave(9) = the accumulated time spent on line search;
  //C         dsave(11) = the slope of the line search function at
  //C                                  the current point of line search;
  //C         dsave(12) = the maximum relative step length imposed in
  //C                                                           line search;
  //C         dsave(13) = the infinity norm of the projected gradient;
  //C         dsave(14) = the relative step length in the line search;
  //C         dsave(15) = the slope of the line search function at
  //C                                 the starting point of the line search;
  //C         dsave(16) = the square of the 2-norm of the line search
  //C                                                      direction vector.
  //C
  //C     Subprograms called:
  //C
  //C       L-BFGS-B Library ... mainlb.
  //C
  //C     References:
  //C
  //C       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
  //C       memory algorithm for bound constrained optimization'',
  //C       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.
  //C
  //C       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: a
  //C       limited memory FORTRAN code for solving bound constrained
  //C       optimization problems'', Tech. Report, NAM-11, EECS Department,
  //C       Northwestern University, 1994.
  //C
  //C       (Postscript files of these papers are available via anonymous
  //C        ftp to eecs.nwu.edu in the directory pub/lbfgs/lbfgs_bcm.)
  //C
  //C                           *  *  *
  //C
  //C     NEOS, November 1994. (Latest revision June 1996.)
  //C     Optimization Technology Center.
  //C     Argonne National Laboratory and Northwestern University.
  //C     Written by
  //C                        Ciyou Zhu
  //C     in collaboration with R.H. Byrd, P. Lu-Chen and J. Nocedal.
  //C
  //C     ************
  //C-jlm-jn
  //C
  if (task == "START") {
    isave(1) = m * n;
    isave(2) = fem::pow2(m);
    isave(3) = 4 * fem::pow2(m);
    //C ws      m*n
    isave(4) = 1;
    //C wy      m*n
    isave(5) = isave(4) + isave(1);
    //C wsy     m**2
    isave(6) = isave(5) + isave(1);
    //C wss     m**2
    isave(7) = isave(6) + isave(2);
    //C wt      m**2
    isave(8) = isave(7) + isave(2);
    //C wn      4*m**2
    isave(9) = isave(8) + isave(2);
    //C wsnd    4*m**2
    isave(10) = isave(9) + isave(3);
    //C wz      n
    isave(11) = isave(10) + isave(3);
    //C wr      n
    isave(12) = isave(11) + n;
    //C wd      n
    isave(13) = isave(12) + n;
    //C wt      n
    isave(14) = isave(13) + n;
    //C wxp     n
    isave(15) = isave(14) + n;
    //C wa      8*m
    isave(16) = isave(15) + n;
  }
  int lws = isave(4);
  int lwy = isave(5);
  int lsy = isave(6);
  int lss = isave(7);
  int lwt = isave(8);
  int lwn = isave(9);
  int lsnd = isave(10);
  int lz = isave(11);
  int lr = isave(12);
  int ld = isave(13);
  int lt = isave(14);
  int lxp = isave(15);
  int lwa = isave(16);
  //C
  mainlb(cmn, n, m, x, l, u, nbd, f, g, factr, pgtol, wa(lws), wa(lwy),
    wa(lsy), wa(lss), wa(lwt), wa(lwn), wa(lsnd), wa(lz), wa(lr), wa(ld),
    wa(lt), wa(lxp), wa(lwa), iwa(1), iwa(n + 1), iwa(2 * n + 1),
    task, iprint, csave, lsave, isave(22), dsave);
  //C
}

//C
//C  L-BFGS-B is released under the “New BSD License” (aka “Modified BSD License”
//C  or “3-clause license”)
//C  Please read attached file License.txt
//C
double
dnrm2(
  int const& n,
  arr_cref<double> x,
  int const& incx)
{
  double return_value = fem::double0;
  x(dimension(n));
  //C     **********
  //C
  //C     Function dnrm2
  //C
  //C     Given a vector x of length n, this function calculates the
  //C     Euclidean norm of x with stride incx.
  //C
  //C     The function statement is
  //C
  //C       double precision function dnrm2(n,x,incx)
  //C
  //C     where
  //C
  //C       n is a positive integer input variable.
  //C
  //C       x is an input array of length n.
  //C
  //C       incx is a positive integer variable that specifies the
  //C         stride of the vector.
  //C
  //C     Subprograms called
  //C
  //C       FORTRAN-supplied ... abs, max, sqrt
  //C
  //C     MINPACK-2 Project. February 1991.
  //C     Argonne National Laboratory.
  //C     Brett M. Averick.
  //C
  //C     **********
  //C
  return_value = 0.0e0;
  double scale = 0.0e0;
  //C
  int i = fem::int0;
  FEM_DOSTEP(i, 1, n, incx) {
    scale = fem::max(scale, fem::abs(x(i)));
  }
  //C
  if (scale == 0.0e0) {
    return return_value;
  }
  //C
  FEM_DOSTEP(i, 1, n, incx) {
    return_value += fem::pow2((x(i) / scale));
  }
  //C
  return_value = scale * fem::sqrt(return_value);
  //C
  return return_value;
  //C
}

} // namespace lbfgsb
