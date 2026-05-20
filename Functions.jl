using DifferentialEquations
using LinearAlgebra
using Plots

using Optim


function pauli_matrices()
X = [0 1; 1 0]
Z = [1 0; 0 -1]
Id = [1 0; 0 1]
return X, Z, Id
end


function kronecker_products(chain)
total = chain[1]
for i in 2:length(chain)
    total = kron(total, chain[i])
    end
return total
end


function local_operator(op_i, i, N)
#I⊗I⊗Z⊗I⊗I, operator at i 
X, Z, Id = pauli_matrices()
chain = [Id for _ in 1:N]
chain[i] = op_i
return kronecker_products(chain)
end



function interaction_btw_sites(i, N, J)
X, Z, Id = pauli_matrices()
Zi = local_operator(Z,i , N) 
Zj =  local_operator(Z, i+1 , N) 
return - J[i] * (Zi * Zj)
end



function TFIM(N, J, Γ)
X, Z, Id = pauli_matrices()
H₀ = zeros(Float64, 2^N, 2^N)
for i in 1:N-1
H₀ += interaction_btw_sites(i, N, J)
end

for i in 1:N
    Xi = local_operator(X, i, N)
    H₀ += -Γ[i] * Xi
end

return H₀
end


# Now we add the open system dynamucs including spatial noise, Lindblad operators, and the master equation.


function spatial_noise(N, Γ₀, p, l)
noise = zeros(Float64, N, N)
for i in 1:N
    for j in 1:N
        r = abs(i-j)
        noise[i, j] = p^r * exp(-r/l)
    end
end
L = cholesky(noise).L
random_vector = randn(N)
return Γ₀ .+ L * random_vector
end


#Uisng jump operators like we did earlier
function jump_operators(N, γ)
X, Z, Id = pauli_matrices()
jump = Matrix{ComplexF64}[]

for i in 1:N
    Xi = local_operator(X, i, N)
    push!(jump, sqrt(γ[i]) * Xi)
end
return jump
end

#Lindblad code from 2 years ago
function make_rhs(N, J, Γ, γ)
    H  = TFIM(N, J, Γ)
    Vs = jump_operators(N, γ)
    dim = 2^N

    function rhs(u, _, t)
        ρ  = reshape(u, dim, dim)
        dρ = -im * (H * ρ - ρ * H)
        for Vi in Vs
            Vd  = adjoint(Vi)
            dρ += Vi * ρ * Vd - 0.5 * (Vd * Vi * ρ + ρ * Vd * Vi)
        end
        return vec(dρ)
    end
    return rhs
end


function time_evolution(rhs, ρ₀, tspan)
    prob = ODEProblem(rhs, vec(ρ₀), tspan)
    sol = solve(prob, Tsit5())
    return sol
end

function plot_dynamics(sol, N)
    t_list = sol.t
    ρ_list = [reshape(u, 2^N, 2^N) for u in sol.u]
    populations = [diag(ρ) for ρ in ρ_list]
    plot(t_list, hcat(populations...)', labels="")
end


function run_dynamics(N, J, Γ, γ, ρ0, tf, τ)
    dim     = 2^N
    tpoints = collect(0.0:τ:tf)
    rhs     = make_rhs(N, J, Γ, γ)
    prob    = ODEProblem(rhs, vec(complex(ρ0)), (0.0, tf))
    sol     = solve(prob, Tsit5(), saveat=tpoints, abstol=1e-8, reltol=1e-6)

    ρ_list = [reshape(u, dim, dim) for u in sol.u]
    return sol.t, ρ_list
end


function plot_populations(t_list, ρ_list, N)
    dim  = 2^N
    pops = [real.(diag(ρ)) for ρ in ρ_list]       
    P    = stack(pops; dims=2)                      

    labels = reshape(["State $i" for i in 0:dim-1], 1, dim)

    plt = plot(t_list, P', xlabel = "Time", ylabel="Population", title="Time Evolution of Populations", labels=labels)
 
    display(plt)
end


function main()
    N  = 4                           
    J  = fill(0.1, N-1)                 
    p  = 0.8                            
    Γ  = spatial_noise(N, 0.5, p, 1.0) 
    γ  = fill(0.1,  N)                   

   dim = 2^N
    ρ0  = zeros(ComplexF64, dim, dim)
  ρ0[dim, dim] = 1.0                   

    tf = 10.0
    τ  = 0.05
  
    t_list, ρ_list = run_dynamics(N, J, Γ, γ, ρ0, tf, τ)
    plot_populations(t_list, ρ_list, N)
end
main()


#Optimising the chain using QFI
function calculate_qfi(ρ, A)
    E = eigen(Hermitian(ρ))
    p = E.values
    states = E.vectors
    
    dim = length(p)
    qfi = 0.0
    
    A_basis = adjoint(states) *A * states
    for m in 1:dim
        for n in 1:dim
            if p[m] + p[n] > 0
                numerator = (p[m] - p[n])^2
                denominator = p[m] + p[n]
                qfi += 2.0 * (numerator / denominator) * abs(A_basis[m, n])^2
            end
        end
    end
    return qfi
end

function opt_prob(J_val)
    N = 3
    p = 0.8
    l = 1
    Γ₀ = 0.5
    Γ = fill(1.0, N)
    γ = fill(0.1, N) 
    tf = 5.0
    τ = 0.1
    
    Γ = spatial_noise(N, Γ₀, p, l)
    X, Z, Id = pauli_matrices()
    A = sum([local_operator(Z, i, N) for i in 1:N])
    

    ρ0 = zeros(ComplexF64, 2^N, 2^N)
    ρ0[2^N, 2^N] = 1.0  
    
    t_list, ρ_list = run_dynamics(N, J_val, Γ, γ, ρ0, tf, τ)
    qfi_trajectory = [calculate_qfi(ρ, A) for ρ in ρ_list]

    return maximum(qfi_trajectory)
end

#optimisation problem

function cost(u, p)
    max_vals = opt_prob(u)
    return -max_vals
end

function run_couplings_optimization()
    N = 3
    p = 0.8
    l = 1
    Γ₀ = 0.5
    γ = fill(0.1, N) 
    tf = 5.0
    τ = 0.1
    
    all_params = (N, Γ₀, p, l, γ, tf, τ)
    initial_J = [0.1, 0.1]
       
    res = optimize(J -> -opt_prob(J), initial_J, NelderMead())
    
    optimal_J = res.minimizer
    max_achieved_qfi = -res.minimum
    
    println("Optimal Couplings for J! : ", round.(optimal_J; digits=4))
    println("Max QFI: ", round(max_achieved_qfi; digits=4))
    
    return optimal_J
end

run_couplings_optimization()