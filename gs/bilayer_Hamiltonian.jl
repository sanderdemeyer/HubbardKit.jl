using TensorKit
using TensorKitTensors
using PEPSKit

I = fℤ₂ ⊠ U1Irrep
T = ComplexF64
particle_symmetry = U1Irrep
spin_symmetry = Trivial
(Nr, Nc) = (2, 2)
V_phys = Vect[I]((0, 0) => 1, (1, 1) => 2, (0, 2) => 1);

V_z = 0.8
V_xy = 0.7
U_d = 3.4
U_z = 3.2
U_dz = 2.0
Δd = 0.4
t_x = t_y = 0.5

e_hop = HubbardOperators.e_hopping(T, particle_symmetry, spin_symmetry)
U_onsite = HubbardOperators.ud_num(T, particle_symmetry, spin_symmetry)
H_chem = HubbardOperators.e_num(T, particle_symmetry, spin_symmetry)
@tensor U_offsite[-1 -2; -3 -4] := HubbardOperators.e_num(T, particle_symmetry, spin_symmetry)[-1; -3] * HubbardOperators.e_num(T, particle_symmetry, spin_symmetry)[-2; -4]

H_V_z = V_z * U_offsite
H_V_xy = V_xy * U_offsite
H_U_d = U_d * (U_onsite ⊗ id(V_phys) + id(V_phys) ⊗ U_onsite) 
H_U_z = U_z * (U_onsite ⊗ id(V_phys) + id(V_phys) ⊗ U_onsite)
H_U_dz = U_dz * U_offsite
H_chem_d = Δd * H_chem ⊗ id(V_phys)
H_t_x = - t_x * e_hop
H_t_y = - t_y * e_hop

lattice = fill(V_phys, Nr, Nc)

H = PEPSKit.LocalOperator(lattice, 
    (CartesianIndex(1,1), CartesianIndex(1,2)) => H_U_dz + H_chem_d,
    (CartesianIndex(2,1), CartesianIndex(2,2)) => H_U_dz + H_chem_d,
    (CartesianIndex(1,1), CartesianIndex(2,1)) => H_V_z + H_U_d,
    (CartesianIndex(1,2), CartesianIndex(2,2)) => H_U_z,
    (CartesianIndex(1,2), CartesianIndex(1,4)) => H_t_x + H_V_xy,
    (CartesianIndex(1,2), CartesianIndex(3,2)) => H_t_y + H_V_xy,
    (CartesianIndex(2,2), CartesianIndex(2,4)) => H_t_x + H_V_xy,
    (CartesianIndex(2,2), CartesianIndex(4,2)) => H_t_y + H_V_xy,

);

