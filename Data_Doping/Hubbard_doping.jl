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

function get_charges(T, ψ, env, H_num, lattice)
    (Nr, Nc) = size(lattice)
    charges = zeros(T, Nr, Nc)
    terms = []
    for r = 1:Nr, c = 1:Nc
        terms = [(CartesianIndex(r,c),) => H_num]
        ham_num = PEPSKit.LocalOperator(lattice, terms...)
        charge = expectation_value(ψ, ham_num, env)
        charges[r,c] = charge
    end
    return charges
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

function get_charge_sectors(k)
    even_sector = Dict((0,-2*i*Q + k*P) => 1 for i = -1:1)
    odd_sector = Dict((1,-(2*i+1)*Q + k*P) => 1 for i = -1:1)
    return merge(even_sector, odd_sector)
end

T = ComplexF64
I = fℤ₂ ⊠ U1Irrep; # symmetry sector
P, Q = 3, 4
Dbond = 4
χenv = 16
Nr, Nc = 2, 6
t, U = 1, 8

particle_symmetry = U1Irrep
spin_symmetry = Trivial

@assert P != 1
@assert mod(Nc, Q - 1) == 0
@assert P == Q - 1

V_phys = Vect[I]((0,0) => 1, (1,1) => 2, (0,2) => 1);
V_phys_s = Vect[I]((0,-Q) => 1, (1,P-Q) => 2, (0,2*P-Q) => 1);
V_aux = Vect[I]((0,-Q) => 1, (0,P-1-Q) => 1, (0,2*P-2-Q) => 1)
V_triv = Vect[I]((0,0) => 1)
V_env = Vect[I]((0,0) => div(χenv,2), (1,0) => div(χenv,2))

fuser = zeros(V_phys_s ← V_phys ⊗ V_aux) 
fuser[(I(0,-Q), dual(I(0,0)), dual(I(0,-Q)))] .= 1.0
fuser[(I(1,P-Q), dual(I(1,1)), dual(I(0,P-Q-1)))] .= [1.0 0.0; 0.0 1.0]
fuser[(I(0,2*P-Q), dual(I(0,2)), dual(I(0,2*P-Q-2)))] .= 1.0

lattice = fill(V_phys_s, Nr, Nc)

H_t = - t * (HubbardOperators.c_plus_c_min(T, particle_symmetry, spin_symmetry) + HubbardOperators.c_min_c_plus(T, particle_symmetry, spin_symmetry));
H_U = U * HubbardOperators.ud_num(T, particle_symmetry, spin_symmetry);
H_num = HubbardOperators.c_num(T, particle_symmetry, spin_symmetry);

H = H_t + (H_U ⊗ id(V_phys) + id(V_phys) ⊗ H_U) / 2

# @tensor H_t_s[-1 -2; -3 -4] := H_t[1 4; 2 5] * fuser[-1; 1 3] * fuser[-2; 4 6] * 
#     conj(fuser[-3; 2 3]) * conj(fuser[-4; 5 6]);
@tensor H_num_s[-1; -2] := H_num[2; 3] * fuser[-1; 2 1] * conj(fuser[-2; 3 1]); 

@tensor H_s[-1 -2; -3 -4] := H[1 4; 2 5] * fuser[-1; 1 3] * fuser[-2; 4 6] * 
    conj(fuser[-3; 2 3]) * conj(fuser[-4; 5 6]);

ham = get_Hubbard_ham(H_s, lattice)

V_virt_s = [Vect[I](get_charge_sectors(c)) for c = 0:Q-2]

PEPS_data = fill(randn(T, V_phys_s, V_triv ⊗ V_triv ⊗ V_triv' ⊗ V_triv'), Nr, Nc)
for r in 1:Nr, c in 1:Nc
    # PEPS_data[r, c] = randn(T, V_phys_s, V_triv ⊗ V_virt_s[mod1(c+1,Q-1)] ⊗ V_triv' ⊗ V_virt_s[mod1(c,Q-1)]')
    PEPS_data[r, c] = randn(T, V_phys_s, V_phys_s ⊗ V_virt_s[mod1(c+1,Q-1)] ⊗ V_phys_s' ⊗ V_virt_s[mod1(c,Q-1)]')
end

ψ = InfinitePEPS(PEPS_data);
# Simple Update
println("Starting with SU")
peps = IPEPS_to_IWP(ψ)
dts = [1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-6, 1e-8, 1e-10, 1e-10]
maxiter = 20000
for (n, (dt, tol)) in enumerate(zip(dts, tols))
    trscheme = truncdim(Dbond) # & truncerr(1e-10) 
    alg = SimpleUpdate(dt, tol, maxiter, trscheme)
    global peps, = simpleupdate(peps, ham, alg; bipartite=false)
end

ψ = InfinitePEPS(peps)

println("Starting with CTMRG")
ctm_alg_AD = SequentialCTMRG(; tol=1e-10, maxiter=150, verbosity=0);
ctm_alg_SU = SequentialCTMRG(; tol=1e-10, maxiter=150, verbosity=0, trscheme = truncdim(χenv));
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg_AD,
    optimizer_alg=LBFGS(4; maxiter=10, gradtol=1e-10, verbosity=0),
    gradient_alg=LinSolver(; iterscheme=:diffgauge),
    reuse_env=true,
);

filename_save = "Hubbard_testing_chienv_$(χenv)"
iterations = 100
H_simple = ham

env0 = CTMRGEnv(ψ, V_env);
env, = leading_boundary(env0, ψ, ctm_alg_SU);

charges_init = get_charges(T, ψ, env, H_num_s, lattice)

for iter = 1:iterations
    naaam = filename_save * "_$(iter).jld2"
    custom_finalize! = (x, f, g, numiter) -> custom_finalize_function(x, f, g, numiter, filename_save * "_$(iter)_LBFGS_$(numiter).jld2")
    global ψ
    global env
    println("Starting with iteration $iter")
    println("AD_CTMRG:")
    ψ, env, cost, info = fixedpoint(ham, ψ, env, opt_alg, (finalize!)= custom_finalize!)
    println("Simple Update:")

    file = jldopen(filename_save * "_$(iter)_conv.jld2", "w")
    file["peps"] = copy(ψ)
    file["env"] = env
    file["gradnormhist"] = info.gradnorms
    file["fhist"] = info.costs

    ψ, env = do_SU(ψ, Dbond, H_simple, ctm_alg_SU, V_env)
    file["peps_SU"] = copy(ψ)
    file["env_SU"] = env
    close(file)

end

charges = get_charges(T, ψ, env, H_num_s, lattice)