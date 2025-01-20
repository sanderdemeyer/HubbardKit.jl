using Test
using Printf
using Random
using PEPSKit
using TensorKit

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Dcut = 8
particle_symmetry, spin_symmetry = U1Irrep, U1Irrep
N1, N2 = 2, 2
Random.seed!(10)
if symm == Trivial
    Pspace = Vect[fℤ₂](0 => 2, 1 => 2)
    Vspace = Vect[fℤ₂](0 => Dcut / 2, 1 => Dcut / 2)
elseif (particle_symmetry == U1Irrep) && (spin_symmetry == U1Irrep)
    # Pspace = Vect[fℤ₂ ⊠ U1Irrep ⊠ U1Irrep]((0,-1,0) => 1, (1,0,1//2) => 1, (1,0,-1//2) => 1, (0,1,0) => 1)
    Pspace = Vect[fℤ₂ ⊠ U1Irrep ⊠ U1Irrep]((0,0,0) => 1, (1,1,1//2) => 1, (1,1,-1//2) => 1, (0,2,0) => 1)
    Vspace = Vect[fℤ₂ ⊠ U1Irrep ⊠ U1Irrep]((0,0,0) => 1, (1,1,1//2) => 1, (1,1,-1//2) => 1, (0,2,0) => 1, (1,3,1//2) => 1, (1,3,-1//2) => 1)
else
    error("Not implemented")
end

peps = InfinitePEPS(rand, Float64, Pspace, Vspace; unitcell=(N1, N2))
spaces = fill(Pspace, N1, N2)

χ = 5
envs0 = CTMRGEnv(randn, Float64, peps, Vspace)
trscheme = truncerr(1e-9) & truncdim(χ)
ctm_alg = CTMRG(;
    maxiter=40, tol=1e-10, verbosity=3, trscheme=trscheme, ctmrgscheme=:sequential
)
envs = leading_boundary(envs0, peps, ctm_alg)

ham = hubbard_model(Float64, particle_symmetry, spin_symmetry, InfiniteSquare(N1, N2); t=t, U=U, mu=U / 2);


opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=5, gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
    reuse_env=true,
)
result = fixedpoint(peps, ham, opt_alg, envs)
result.E


"""
ham_t = hubbard_model(Float64, particle_symmetry, spin_symmetry, InfiniteSquare(N1, N2); t=t, U=0, mu=0);
ham_U = hubbard_model(Float64, particle_symmetry, spin_symmetry, InfiniteSquare(N1, N2); t=0, U=U, mu=0);
ham_chem = hubbard_model(Float64, particle_symmetry, spin_symmetry, InfiniteSquare(N1, N2); t=0, U=0, mu=U / 2);

n_up_O = MPSKitModels.e_number_up(Float64, particle_symmetry, spin_symmetry)
n_down_O = MPSKitModels.e_number_down(Float64, particle_symmetry, spin_symmetry)
n_up = LocalOperator(spaces, ((idx,) => n_up_O for idx in vertices(lattice))...,)
n_down = LocalOperator(spaces, ((idx,) => n_down_O for idx in vertices(lattice))...,)

E = expectation_value(peps, ham, envs) / (N1 * N2);
E_t = expectation_value(peps, ham_t, envs) / (N1 * N2);
E_U = expectation_value(peps, ham_U, envs) / (N1 * N2);
E_chem = expectation_value(peps, ham_chem, envs) / (N1 * N2);

nup = expectation_value(peps, n_up, envs)
ndown = expectation_value(peps, n_down, envs)

println("E = $(E)")
"""