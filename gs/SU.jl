using Test
using Random
using PEPSKit
using TensorKit
using JLD2
using MPSKitModels
import MPSKitModels: hubbard_space

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Dcut = 4
particle_symmetry, spin_symmetry = Trivial, U1Irrep
N1, N2 = 2, 2
Random.seed!(10)

χenv0, χenv = 12, 24

Pspace = hubbard_space(particle_symmetry, spin_symmetry)

if (particle_symmetry == Trivial) && (spin_symmetry == Trivial)
    Vspace = Vect[fℤ₂](0 => Dcut / 2, 1 => Dcut / 2)
    Espaces = [Vect[fℤ₂](0 => χ / 2, 1 => χ / 2) for χ = [χenv0, χenv]]
elseif (particle_symmetry == Trivial) && (spin_symmetry == U1Irrep)
    Vspace = Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => floor(Dcut/2), (1, 1 // 2) => floor(Dcut/4), (1, -1 // 2) => floor(Dcut/4))
    Espaces = [Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => floor(χ/2), (1, 1 // 2) => floor(χ/4), (1, -1 // 2) => floor(χ/4)) for χ = [χenv0, χenv]]
    Vspace = Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => Dcut)
    Espaces = [Vect[fℤ₂ ⊠ U1Irrep]((0, 0) => χ) for χ = [χenv0, χenv]]
else
    error("Not implemented")
end
peps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(N1, N2))

# normalize vertex tensors
for ind in CartesianIndices(peps.vertices)
    peps.vertices[ind] /= norm(peps.vertices[ind], Inf)
end

# Hubbard model Hamiltonian at half-filling
t, U = 1, 6
ham = hubbard_model(Float64, particle_symmetry, spin_symmetry, InfiniteSquare(N1, N2); t, U, mu=U / 2)

function do_SU(peps, ham, Espaces, Dcut)
    # simple update
    dts = [1e-2, 1e-3, 4e-4, 1e-4]
    tols = [1e-6, 1e-8, 1e-8, 1e-8]
    maxiter = 5000
    for (n, (dt, tol)) in enumerate(zip(dts, tols))
        println(summary(InfinitePEPS(peps)[1,1]))
        trscheme = truncerr(1e-10) & truncdim(Dcut)
        alg = SimpleUpdate(dt, tol, maxiter, trscheme)
        peps, = simpleupdate(peps, ham, alg; bipartite=false)
    end

    # absorb weight into site tensors
    peps = InfinitePEPS(peps)

    # CTMRG
    envs = CTMRGEnv(randn, Float64, peps, Espaces[1])
    ctm_alg = SequentialCTMRG(; maxiter=300, tol=1e-7, projector_alg = HalfInfiniteProjector, trscheme = truncdim(dim(Espaces[1])))
    envs = leading_boundary(envs, peps, ctm_alg)
    println("envs = $(summary(envs.edges[1,1,1]))")
    E = costfun(peps, envs, ham) / (N1 * N2)

    envs = CTMRGEnv(randn, Float64, peps, Espaces[2])
    ctm_alg = SequentialCTMRG(; maxiter=300, tol=1e-7, projector_alg = HalfInfiniteProjector, trscheme = truncdim(dim(Espaces[2])))
    envs = leading_boundary(envs, peps, ctm_alg)
    println("envs = $(summary(envs.edges[1,1,1]))")
    E = costfun(peps, envs, ham) / (N1 * N2)

    return peps, envs, E
end

peps, envs, E = do_SU(peps, ham, Espaces, Dcut)

# Benchmark values of the ground state energy from
# Qin, M., Shi, H., & Zhang, S. (2016). Benchmark study of the two-dimensional Hubbard
# model with auxiliary-field quantum Monte Carlo method. Physical Review B, 94(8), 085103.
Es_exact = Dict(0 => -1.62, 2 => -0.176, 4 => 0.8603, 6 => -0.6567, 8 => -0.5243)
E_exact = Es_exact[U] - U / 2

# measure energy
@info "Energy           = $E"
@info "Benchmark energy = $E_exact"
@test isapprox(E, E_exact; atol=5e-2)

name = "Hubbard_SU_D_$(Dcut)_chienv_$(χenv)"
file = jldopen(name, "w")
file["peps"] = peps
file["envs"] = envs
file["E"] = E
close(file)