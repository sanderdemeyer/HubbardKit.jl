using TensorKit
using PEPSKit
using JLD2

χenv = 16
maxiter = 5

for i = 1:maxiter
    name = "Data_Doping/Hubbard_testing_chienv_$(χenv)_$(i)_conv.jld2"
    file = jldopen(name, "r")

    ψ = file["peps"]
    env = file["env"]
    gradnorms = file["gradnormhist"]
    costs = file["fhist"]
    ψ_SU = file["peps_SU"]
    env_SU = file["env_SU"]
    close(file)

    println("For i = $i, gradnorms = $gradnorms")
    println("For i = $i, costs = $costs")
    println(summary(ψ[1,1]))
end