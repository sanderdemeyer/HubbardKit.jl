using Pkg
using Test
using Printf
using Random
using PEPSKit
using TensorKit
using JLD2
using OptimKit
using KrylovKit

function custom_finalize(name, (peps, envs), f, g, numiter)
    file = jldopen(name * "_num_$(numiter).jld2", "w")
    file["peps"] = peps
    file["envs"] = envs
    file["f"] = f
    file["g"] = g
    return (peps, envs), f, g
end

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Dcut = 2
χenv = 12
t, U = 1, 6

particle_symmetry, spin_symmetry = Trivial, U1Irrep
N1, N2 = 2, 2
Random.seed!(10)

if (particle_symmetry == Trivial) && (spin_symmetry == Trivial)
    Espace = Vect[fℤ₂](0 => χenv / 2, 1 => χenv / 2)
elseif (particle_symmetry == Trivial) && (spin_symmetry == U1Irrep)
    Espace = Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => χenv)
else
    error("Not implemented")
end

ctm_alg = SequentialCTMRG(; maxiter=300, tol=1e-7, projector_alg = HalfInfiniteProjector, trscheme = truncdim(dim(Espace)))

with_CTMRG = false

if with_CTMRG
    name = "Hubbard_SU_t_$(t)_U_$(U)_D_$(Dcut)_chienv_$(χenv)"
    file = jldopen(name * ".jld2", "r")
    peps = file["peps"]
    envs = file["envs"]
else
    name = "Hubbard_SU_t_$(t)_U_$(U)_D_$(Dcut)_chienv_$(χenv)_wo_CTMRG"
    file = jldopen(name * ".jld2", "r")
    peps = file["peps"]
    envs0 = CTMRGEnv(randn, Float64, peps, Espace)
    envs = leading_boundary(envs0, peps, ctm_alg)
end
close(file)

finalize! = ((peps, envs), f, g, numiter) -> custom_finalize(name, (peps, envs), f, g, numiter)

ham = hubbard_model(Float64, particle_symmetry, spin_symmetry, InfiniteSquare(N1, N2); t=t, U=U, mu=U / 2);


opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:diffgauge),
    reuse_env=true,
)
result = fixedpoint(peps, ham, opt_alg, envs; finalize! = finalize!)
println("E = $(result.E)")

file = jldopen(name * "_converged_CTMRG.jld2", "w")
file["peps"] = result.peps
file["envs"] = result.env
file["E"] = result.E
close(file)

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