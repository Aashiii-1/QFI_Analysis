include("Functions.jl")

### Tests

N = 3

X, Z, Id = pauli_matrices()

za = local_operator(Z, 2, N)
println(size(za))