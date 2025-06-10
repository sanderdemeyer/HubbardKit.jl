# using Pkg
# Pkg.resolve()
# Pkg.instantiate()
# Pkg.add("TensorKitTensors")
# Pkg.rm("KrylovKit")
# Pkg.rm("OptimKit")
# Pkg.rm("MPSKit")
# Pkg.rm("MPSKitModels")
# Pkg.rm("PEPSKit")
# Pkg.resolve()
# Pkg.instantiate()
# Pkg.add("TensorKit")
# Pkg.add("TensorKitTensors")
# Pkg.add("KrylovKit")
# Pkg.add("OptimKit")
# Pkg.add("MPSKit")
# Pkg.add("MPSKitModels")
# Pkg.add("PEPSKit")

# Pkg.add("ThreadPinning")
# Pkg.add("JLD2")
# Pkg.resolve()
# Pkg.instantiate()

using ThreadPinning
# pinthreads(:cores)

println("Number of threads is $(Threads.nthreads())")
# threadinfo(; slurm = true)
println("beginning of the simulation")

using TensorKit, OptimKit, KrylovKit
using TensorKitTensors.HubbardOperators
using PEPSKit
using JLD2
using MPSKit, MPSKitModels
using PEPSKit: nearest_neighbours, next_nearest_neighbours

function IPEPS_to_IWP(peps::InfinitePEPS)
    Nr, Nc = size(peps)
    weights = map(Iterators.product(1:2, 1:Nr, 1:Nc)) do (d, r, c)
        V = (d == 1 ? domain(peps[r, c])[2] : domain(peps[r, c])[1])
        @assert !isdual(V)
        DiagonalTensorMap(ones(reduceddim(V)), V)
    end
    return InfiniteWeightPEPS(peps.A, SUWeight(weights))
end

function get_Hubbard_ham(H, lattice)
    (Nr, Nc) = size(lattice)
    lat = InfiniteSquare(Nr,Nc)

    terms = []
    for (I,J) in nearest_neighbours(lat)
        if (I[1] + I[2]) % 2 == 0
            push!(terms, (I,J) => H)
        else
            push!(terms, (I,J) => H)
        end
    end
    return PEPSKit.LocalOperator(lattice, terms...)
end

function do_SU(peps, Dbond, H_simple, ctm_alg, V_env)
    IWP_new = IPEPS_to_IWP(peps);
    trscheme = truncdim(Dbond);
    # trscheme = HalfInfiniteProjector; # FixedSpaceTruncation();
    alg = SimpleUpdate(0.0, 1e-6, 1, trscheme);
    x_new, = simpleupdate(IWP_new, H_simple, alg; bipartite=false);

    peps_new = InfinitePEPS(x_new)
    env0_new = CTMRGEnv(peps_new, V_env);
    env_new, = leading_boundary(env0_new, peps_new, ctm_alg);

    return peps_new, env_new
end

function custom_finalize_function(x, f, g, numiter, filename_save)
    println("Saving data to $filename_save")
    file = jldopen(filename_save, "w");
    file["x"] = x;
    file["g"] = g;
    file["numiter"] = numiter;
    close(file)
    return x, f, g
end

T = ComplexF64
I = fℤ₂; # symmetry sector
Dbond = 2
Dv = 2
Dh = 4
χenv = 2
Nr, Nc = 2, 8
t, U = 1, 8
μ = 0.45


particle_symmetry = Trivial
spin_symmetry = Trivial

# @assert P != 1
# @assert mod(Nc, Q - 1) == 0
# @assert P == Q - 1

V_phys = Vect[I](0 => 2, 1 => 2);
V_virt_v = Vect[I](0 => div(Dv,2), 1 => div(Dv,2))
V_virt_h = Vect[I](0 => div(Dh,2), 1 => div(Dh,2))
V_triv = Vect[I](0 => 1)
V_env = Vect[I](0 => div(χenv,2), 1 => div(χenv,2))

pspaces = fill(V_phys, Nr, Nc)
lattice = InfiniteSquare(Nr, Nc)
H_t = - t * HubbardOperators.e_hopping(T, particle_symmetry, spin_symmetry);
H_U = U * HubbardOperators.ud_num(T, particle_symmetry, spin_symmetry);
H_num = HubbardOperators.e_num(T, particle_symmetry, spin_symmetry);
H_num_LO = PEPSKit.LocalOperator(pspaces, ((idx,) => H_num for idx in PEPSKit.vertices(lattice))...,)
H_onesite = H_U - μ * H_num;
H = H_t + (H_onesite ⊗ id(V_phys) + id(V_phys) ⊗ H_onesite) / 2

ham = get_Hubbard_ham(H, pspaces)

PEPS_data = fill(randn(T, V_phys, V_virt_v ⊗ V_virt_h ⊗ V_virt_v' ⊗ V_virt_h'), Nr, Nc)

ψ = InfinitePEPS(PEPS_data);
# Simple Update
# println("Starting with SU")
# peps = IPEPS_to_IWP(ψ)
# dts = [1e-2, 1e-3]#, 4e-4, 1e-4]
# tols = [1e-6, 1e-8]#, 1e-10, 1e-10]
# maxiter = 20
# for (n, (dt, tol)) in enumerate(zip(dts, tols))
#     trscheme = truncdim(Dbond) # & truncerr(1e-10) 
#     alg = SimpleUpdate(dt, tol, maxiter, trscheme)
#     global peps, = simpleupdate(peps, ham, alg; bipartite=false)
# end

# ψ = InfinitePEPS(ψ)

println("Starting with CTMRG")
ctm_alg_AD = SequentialCTMRG(; tol=1e-10, maxiter=5, verbosity=0);
ctm_alg_SU = SequentialCTMRG(; tol=1e-10, maxiter=5, verbosity=0, trscheme = truncdim(χenv));
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg_AD,
    optimizer_alg=LBFGS(4; maxiter=2, gradtol=1e-10, verbosity=0),
    gradient_alg=LinSolver(; iterscheme=:diffgauge),
    reuse_env=true,
);

filename_save = "Hubbard_chempot_$(μ)_chienv_$(χenv)"
iterations = 3
H_simple = ham

env0 = CTMRGEnv(ψ, V_env);
env, = leading_boundary(env0, ψ, ctm_alg_SU);

all_costs = []
all_gradnorms = []

for iter = 1:iterations
    custom_finalize! = (x, f, g, numiter) -> custom_finalize_function(x, f, g, numiter, filename_save * "_iter_$(iter)_LBFGS_$(numiter).jld2")
    global ψ
    global env
    println("Starting with iteration $iter")
    println("AD_CTMRG:")
    ψ, env, cost, info = fixedpoint(ham, ψ, env, opt_alg, (finalize!)= custom_finalize!)
    println("After AD, peps = ")
    println(summary(ψ[1,1]))
    println(summary(ψ[1,2]))
    println(summary(ψ[2,1]))
    println(summary(ψ[2,2]))
    println("Simple Update:")

    file = jldopen(filename_save * "_$(iter)_conv.jld2", "w")
    file["peps"] = copy(ψ)
    file["env"] = env
    file["gradnormhist"] = info.gradnorms
    file["fhist"] = info.costs

    push!(all_costs, info.costs)
    push!(all_gradnorms, info.gradnorms)

    ψ, env = do_SU(ψ, Dbond, H_simple, ctm_alg_SU, V_env)
    file["peps_SU"] = copy(ψ)
    file["env_SU"] = env
    close(file)
end

occupancy = expectation_value(ψ, H_num_LO, env) / (Nr * Nc)

file = jldopen(filename_save * "_converged.jld2", "w")
file["peps"] = copy(ψ)
file["env"] = env
file["gradnormhist"] = all_gradnorms
file["fhist"] = all_costs
file["occupancy"] = occupancy
close(file)

