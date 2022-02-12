# =====================================================================
#       CLASS THAT SETS UP GRID & MATRIX EQUATION
# =====================================================================

import time
from numpy import zeros, size, sqrt, linspace
import scipy.linalg as la


class NonlinearPoisson3D:

    # Constructor/Initializer  (arguments are the Cartesian coordinates)
    def __init__(self, x, y, z):

        self.N = size(x)  # equal grid size on all dimensions x,y,z
        # equal grid spacing h = delta x = delta y = delta z
        self.h = x[1] - x[0]

        self.b_1d = zeros(self.N ** 3)  # initialize b long vector
        self.A = zeros((self.N ** 3, self.N ** 3))  # initialize matrix A
        self.sol = zeros((self.N, self.N, self.N))  # initialize solution
        self.rad = zeros((self.N, self.N, self.N))  # initialize radius

        # compute radius
        for i in range(0, self.N):
            for j in range(0, self.N):
                for k in range(0, self.N):
                    rad2 = (x[i] ** 2) + (y[j] ** 2) + (z[k] ** 2)
                    self.rad[i, j, k] = sqrt(rad2)

    def operator_matrix(self, f_prime):
        """Set up operator matrix A of coefficients (Eq. (4.10))"""

        N = self.N

        """Set Robin BCs (see Eq.(4.11) and the discussion that follows)"""
        i = 0  # lower x-boundary
        for j in range(0, N):
            for k in range(0, N):
                index = self.super_index(i, j, k)
                self.A[index, index] = self.rad[i, j, k]
                self.A[index, index + 1] = -self.rad[i + 1, j, k]

        i = N - 1  # upper x-boundary
        for j in range(0, N):
            for k in range(0, N):
                index = self.super_index(i, j, k)
                self.A[index, index] = self.rad[i, j, k]
                self.A[index, index - 1] = -self.rad[i - 1, j, k]

        j = 0  # lower y-boundary
        for i in range(1, N - 1):
            for k in range(0, N):
                index = self.super_index(i, j, k)
                self.A[index, index] = self.rad[i, j, k]
                self.A[index, index + N] = -self.rad[i, j + 1, k]

        j = N - 1  # upper y-boundary
        for i in range(1, N - 1):
            for k in range(0, N):
                index = self.super_index(i, j, k)
                self.A[index, index] = self.rad[i, j, k]
                self.A[index, index - N] = -self.rad[i, j - 1, k]

        k = 0  # lower z-boundary
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                index = self.super_index(i, j, k)
                self.A[index, index] = self.rad[i, j, k]
                self.A[index, index + N**2] = -self.rad[i, j, k + 1]

        k = N - 1  # upper z-boundary
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                index = self.super_index(i, j, k)
                self.A[index, index] = self.rad[i, j, k]
                self.A[index, index - N**2] = -self.rad[i, j, k - 1]

        """Use Eq. (4.10) to fill matrix A"""

        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    index = self.super_index(i, j, k)

                    # diagonal elements (recall that G in Eq.(4.10) is - f')
                    self.A[index, index] = - \
                        (self.h ** 2) * f_prime[i, j, k] - 6.0

                    # off-diagonal elements
                    self.A[index, index - 1] = 1.0
                    self.A[index, index + 1] = 1.0
                    self.A[index, index - N] = 1.0
                    self.A[index, index + N] = 1.0
                    self.A[index, index - N**2] = 1.0
                    self.A[index, index + N**2] = 1.0

    def rhs(self, b):
        """Setup RHS of matrix equation (4.10)"""
        N = self.N
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    index = self.super_index(i, j, k)
                    self.b_1d[index] = (self.h ** 2) * b[i, j, k]

    def solve(self):

        # Find solution (in long vector (1d) format)
        sol_1d = la.solve(self.A, self.b_1d)

        # Now translate from superindex to 3d format
        for i in range(0, self.N):
            for j in range(0, self.N):
                for k in range(0, self.N):
                    index = self.super_index(i, j, k)
                    self.sol[i, j, k] = sol_1d[index]

        return self.sol  # solution in full 3d format

    def super_index(self, i, j, k):
        """This is the super index we've been using throughout (see Eq.(4.9))"""
        return i + (self.N * j) + ((self.N**2) * k)


# ===========================================================================
#    CLASS THAT CONSTRUCTS PUNCTURE INITIAL DATA FOR A SINGLE BLACK HOLE
# ===========================================================================

class InitialData:

    def __init__(self, bh_loc, ang_mom, lin_mom, N, bd):
        """Constructor/Initializer  (arguments are physical parameters,
           number of gridpoints, and outer boundary)"""

        self.bh_loc = bh_loc  # location of black hole
        self.ang_mom = ang_mom  # angular momentum of black hole
        self.lin_mom = lin_mom  # linear momentum of black hole

        # Set up the grid
        self.N = N  # number of gridpoints
        self.bd = bd  # outer boundary
        self.h = (2.0 * bd)/N  # h = ( bd - (-bd) )/N

        # Set up the coordinates using cell-centered grid
        half_h = self.h/2.0
        self.x = linspace(half_h - bd, bd -
                          half_h, N)
        self.y = linspace(half_h - bd, bd -
                          half_h, N)
        self.z = linspace(half_h - bd, bd -
                          half_h, N)

        # Allocate the elliptic solver using the NonlinearPoisson3D class
        self.solver = NonlinearPoisson3D(self.x, self.y, self.z)

        # Initialize functions u, theta, rho, and residual R (here we call it res)
        self.theta = zeros((N, N, N))
        self.rho = zeros((N, N, N))
        self.u = zeros((N, N, N))
        self.res = zeros((N, N, N))

    def solution(self, tol, it_max):
        """Construct the solution within a user-set tolerance
           and max number of iterations allowed"""

        # theta_rho_functs and residual will be defined shortly
        self.theta_rho_functs()
        res_norm = self.residual()

        # Iterate
        it_step = 0
        while res_norm > tol and it_step < it_max:
            it_step += 1
            self.update_u()  # update_u also to be defined shortly
            res_norm = self.residual()
            print(" Residual after", it_step, "iterations :", res_norm)
        if (res_norm < tol):
            print(" Done. Reached convergence within desired tolerance!")
        else:
            print(" No convergence, unfortunately :( )")

    def update_u(self):
        """Function that implements one iteration: u^{[n+1]} = u^{[n]} + \delta u"""

        N = self.N
        f_prime = zeros((N, N, N))  # initialize f'
        b = zeros((N, N, N))  # initialize b  (=-R)

        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    # calculate f' from Eq.(4.6)
                    var_exp = self.theta[i, j, k] * \
                        (1.0 + self.u[i, j, k]) + 1.0
                    f_prime[i, j, k] = (7.0 * self.rho[i, j, k] *
                                        self.theta[i, j, k])/var_exp**8
                    # Set b=-R (-residual)
                    b[i, j, k] = -self.res[i, j, k]

        # Update the solver feeding it the newly calculated f'
        self.solver.operator_matrix(f_prime)

        # Also update using the newly calculated residual
        self.solver.rhs(b)

        # Solve for \delta u (recall Eq.(4.5))
        delta_u = self.solver.solve()

        # Update u (u^{[n+1]} = u^{[n]} + delta u)
        self.u += delta_u

    def residual(self):
        """Calculate the residual from Eq.(4.2) using updated u values"""

        res_norm = 0.0
        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                for k in range(1, self.N - 1):

                    """Compute the Laplace operator \partial^2
                       (using 2nd order centered difference)"""
                    ddx = (self.u[i + 1, j, k] - 2.0 * self.u[i, j, k] +
                           self.u[i - 1, j, k])
                    ddy = (self.u[i, j + 1, k] - 2.0 * self.u[i, j, k] +
                           self.u[i, j - 1, k])
                    ddz = (self.u[i, j, k + 1] - 2.0 * self.u[i, j, k] +
                           self.u[i, j, k - 1])
                    laplace = (ddx + ddy + ddz)/(self.h ** 2)

                    # Now compute f(u)=  - rho/( theta*(1 + u) + 1)^7
                    # expression inside parentheses in f
                    f_in = self.theta[i, j, k] * (1.0 + self.u[i, j, k]) + 1.0
                    f = - self.rho[i, j, k]/(f_in ** 7)

                    # R = laplace - f (recall Eq.(4.2))
                    self.res[i, j, k] = laplace - f
                    # sum the square of all residuals
                    res_norm += self.res[i, j, k] ** 2

        res_norm = sqrt(res_norm) * (self.h ** 3)  # 2 grid-norm of residual
        return res_norm

    def theta_rho_functs(self):
        """Set up functions theta and rho (see Eqs.(4.13) and (3.28), respectively)"""
        """Here we also compute the Bowen-York curvature, which is needed for rho"""

        N = self.N

        # define momenta
        j_x = self.ang_mom[0]
        j_y = self.ang_mom[1]
        j_z = self.ang_mom[2]

        p_x = self.lin_mom[0]
        p_y = self.lin_mom[1]
        p_z = self.lin_mom[2]

        for i in range(0, N):
            for j in range(0, N):
                for k in range(0, N):

                    # define radial function r_BH (see Eq.(4.12))
                    d_x = self.x[i] - self.bh_loc[0]
                    d_y = self.y[j] - self.bh_loc[1]
                    d_z = self.z[k] - self.bh_loc[2]
                    r_bh = sqrt((d_x ** 2) + (d_y ** 2) + (d_z ** 2))

                    # recall l^i radial unit vector from Eqs. (3.21)
                    l_x = d_x/r_bh
                    l_y = d_y/r_bh
                    l_z = d_z/r_bh

                    # Construct Bowen-York curvature with angular momentum A_J^{ij} (Eq. (3.22a))
                    J_coeff = 3.0/(r_bh**3)
                    Aj_xx = J_coeff * (2.0 * l_x * (j_y * l_z - j_z * l_y))
                    Aj_yy = J_coeff * (2.0 * l_y * (j_z * l_x - j_x * l_z))
                    Aj_zz = J_coeff * (2.0 * l_z * (j_x * l_y - j_y * l_x))
                    Aj_xy = J_coeff * \
                        (l_x * (j_z * l_x - j_x * l_z) +
                         l_y * (j_y * l_z - j_z * l_y))
                    Aj_xz = J_coeff * \
                        (l_x * (j_x * l_y - j_y * l_x) +
                         l_z * (j_y * l_z - j_z * l_y))
                    Aj_yz = J_coeff * \
                        (l_y * (j_x * l_y - j_y * l_x) +
                         l_z * (j_z * l_x - j_x * l_z))

                    # Construct Bowen-York curvature with linear momentum A_p^{ij} (Eq. (3.22b))
                    P_coeff = 3.0/(2.0 * (r_bh**2))
                    lP = (l_x * p_x) + (l_y * p_y) + \
                        (l_z * p_z)  # contraction l_k P^k
                    Ap_xx = P_coeff * (2.0 * p_x * l_x +
                                       lP * (l_x * l_x - 1.0))
                    Ap_yy = P_coeff * (2.0 * p_y * l_y +
                                       lP * (l_y * l_y - 1.0))
                    Ap_zz = P_coeff * (2.0 * p_z * l_z +
                                       lP * (l_z * l_z - 1.0))
                    Ap_xy = P_coeff * (p_x * l_y + p_y * l_x + lP * l_x * l_y)
                    Ap_xz = P_coeff * (p_x * l_z + p_z * l_x + lP * l_x * l_z)
                    Ap_yz = P_coeff * (p_y * l_z + p_z * l_y + lP * l_y * l_z)

                    # Construct full Bowen-York curvature (Eq. (3.22c), with + sign)
                    A_xx = Ap_xx + Aj_xx
                    A_yy = Ap_yy + Aj_yy
                    A_zz = Ap_zz + Aj_zz
                    A_xy = Ap_xy + Aj_xy
                    A_xz = Ap_xz + Aj_xz
                    A_yz = Ap_yz + Aj_yz

                    # Compute A_{ij} A^{ij} term in rho (Eq.(3.28))
                    A2 = (
                        A_xx ** 2 + A_yy ** 2 + A_zz ** 2 +
                        2.0*(A_xy ** 2 + A_xz ** 2 + A_yz ** 2)
                    )

                    # Compute theta (Eq.(4.13)) and rho (Eq.(3.28))
                    self.theta[i, j, k] = 2.0 * r_bh
                    self.rho[i, j, k] = ((self.theta[i, j, k] ** 7)/8.0) * A2

    def write_to_file(self):
        """Write solution to file"""

        filename = "InitialData_" + str(self.N) + "_" + str(self.bd)
        filename = filename + ".data"
        out = open(filename, "w")
        if out:
            k = self.N // 2
            out.write(
                "# Data for black hole at x = (%f,%f,%f)\n"
                % (self.bh_loc[0], self.bh_loc[1], self.bh_loc[2])
            )
            out.write("# with angular momentum P = (%f, %f, %f)\n" %
                      (self.ang_mom))
            out.write("# and linear momentum P = (%f, %f, %f)\n" %
                      (self.lin_mom))
            out.write("# in plane for z = %e \n" % (self.z[k]))
            out.write("# x            y              u              \n")
            out.write("#============================================\n")
            for i in range(0, self.N):
                for j in range(0, self.N):
                    out.write("%e  %e  %e\n" % (self.x[i], self.y[j],
                                                self.u[i, j, k]))
                out.write("\n")
            out.close()
        else:
            print(" Could not open file ", filename,
                  " in write_to_file() function")


# =============================================================
#              MAIN ROUTINE
# =============================================================

def main():
    """Set default values for variables:"""

    # location of black hole:
    loc_x = 0.0
    loc_y = 0.0
    loc_z = 0.0

    # linear momentum entries:
    p_x = 1.0
    p_y = 0.0
    p_z = 0.0

    # angular momentum entries:
    j_x = 0.0
    j_y = 0.0
    j_z = 0.0

    N = 26  # number of grid points
    bd = 8.0  # location of outer boundary
    tol = 1.0e-12  # tolerance
    it_max = 50  # max number of iterations allowed

    bh_loc = (loc_x, loc_y, loc_z)  # location of black hole
    lin_mom = (p_x, p_y, p_z)  # linear momentum
    ang_mom = (j_x, j_y, j_z)  # angular momentum

    # Construct the initial data solver
    bh_initdata = InitialData(bh_loc, ang_mom, lin_mom, N, bd)

    # Build solution
    bh_initdata.solution(tol, it_max)

    # Output to file
    bh_initdata.write_to_file()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("It took %s seconds to run this beautiful code :)" %
          (time.time() - start_time))
