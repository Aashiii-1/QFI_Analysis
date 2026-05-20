## <span style="color:gold"> **Actual Project**

### Summary
- Use MPS to approximate the TFIM
- <span style="color:pink"> Using open boundary conditions
- Spatially corelated noise is used. For now Γ = γsin(xt)
- Lindblad evolution: Diagonal form
     - ∂ρ\∂t = -i[H, ρ] + 𝚺ₖ Γₖ (LₖρLₖ† - 1/2 {LₖLₖ†, ρ}) = Łₚ
    - Lᵢ are a set of jump operators which descibe the dissipative part of the dynamics
    - Γ are the damping rates
    - Work in Liouville space, where you use a Lindbladian superoperator
  
### Step-by-step functions
    - To generate pauli matrices
    - To make operators act on locally in the chain
    - To model interactions between tensors
    - To model TFIM
    - To test the model, maybe simply diagnonalising for a small chain and finding the eigenstates when J = 0 and  Γ = 0
    
    - For noise, this will include jump operators
    - Lindblad Master Equation
    - Calculate QFI of states 
    - Time evolutioon using Linblad, make sure density matrix is vectorised
    ----------------------------------------> done!

    - Generating MPS, this will help compress the structure and can be used to generalise for larger systems
    - TEBD?
    - Optimisation??

### Structure of spatially corelated noise
- Affects qubits diffrently based on their position
- Usually modelled using a spatial correlation function, if i and j are close then the noise they experience would be similar as well
- This coorelated function is calculated using different approaches like using FFT
- Which essentially states that the field at point i is affected by small fluctations at that site. a is the strength of these fluctuations.
- <span style="color:pink"> I am not sure if this is the right formalism. I couldn't find a concrete example on this, so I used Gaussian where C(r) = p*exp(-r/l). This corelation function is then decomposed into lower triangle matrices so that they can be multiplied with pauli matrices


### Structure of Jump Operators and Lindblad ME
- Paper: Open quantum systems — A brief introduction” (Cahiers de l'Institut Pascal, 2026).
- I used the exact operators they've talked about
- V = √γ * Γ

### Optimisation problem
To find the best quantum setups under environmental decay, the system treats the coupling values J across the chain as parameters in an optimization loop. The goal is to maximize the sensitivity of the final evolved state (the Quantum Fisher Information) under continuous noise.

#### Implementation with Optim.jl
The problem is solved using the **Optim.jl** package in Julia using **Nelder-Mead simplex algorithm**. 

## Next Steps: Moving to Matrix Product States (MPS)
While exact diagonalization in Liouville space works well for small systems (N = 4), the dimension of the density matrix scales exponentially. This creates the so called 'cruse of dimensionality.'

I will continue to develop this framework through the following menthods:
- Convert the density matrices and state vectors into Matrix Product States (MPS) using Singular Value Decomposition (SVD) variants.
- Rewrite the noisy Lindblad Hamiltonian dynamics as Matrix Product Operators (MPO).
- Use Time Evolving Block Decimation (TEBD) to evolve the compressed MPS representation through time.













    
