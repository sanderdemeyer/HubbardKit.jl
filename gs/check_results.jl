using Test
using Printf
using Random
using PEPSKit
using TensorKit
using JLD2

t, U = 1, 6
Dcut = 2
χenv = 12
numiter = 24

fs = zeros(numiter)
gs = zeros(numiter)

for i = 1:24
    name = "results U = 6/Hubbard_SU_t_$(t)_U_$(U)_D_$(Dcut)_chienv_$(χenv)_wo_CTMRG_num_$(i).jld2"

    file = jldopen(name, "r")
    peps = file["peps"]
    envs = file["envs"]
    f = file["f"]
    g = file["g"]

    fs[i] = f
    gs[i] = norm(g)
    close(file)
    println("i = $i, f = $(f/4), g = $(norm(g))")
end

using Plots
plt = plot(log.(gs), label = "g")
display(plt)
