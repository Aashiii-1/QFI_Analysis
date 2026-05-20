# TFI with QFI Analysis
This is a blurb for testing the TFIM with QFI analysis. I will start by writing down the functions and parameters used for this project. All code is written by me:)

## Personal Notes
### TFI:
- This is the quantum analogue of the classical Ising model
    - The Hamiltonian: H = -J ((Σ Zᵢ Zⱼ) - g (Σ Xⱼ))
    - Where i and j are nearest neighbours
    - Xᵢ and Zᵢ are Pauli matrices
    - One-dimensional TFI
         - Each spin is 1/2
         - J is the coupling constant and g is the transverse field strength
         - |g| < 1: ordered phase, value of J determines if it is ferromagnetic or antimagnetic
         - |g| = 1: quantum phase transition
    - Importantly, the quantum representation of this state is ψ₁ to ψₙ
    - Spins can take any value along the z-axis
    - So start by coding up the Hamiltonian and the system, which is a 1D TFI model
 
### MPS
<span style="color:pink"> Note that d and D are different. d refers to the physical dimensions whereas D refers to how entangled the two tensors are. For qubits d is 2 but D is an active choice we make.  

- Use matrix product states to represent the state of the system
- Using periodic boundary conditions first / open boundary conditions second
- Periodic: |ψ> = Σ Tr[A₁(s₁) * Aₙ(sₙ)]|s₁...sₙ>
- Open: |ψ> = Σ A₁(s₁) * Aₙ(sₙ)|s₁...sₙ>
- Aᵢ(sᵢ) is a Dᵢ x Dⱼ matrix where j = i+1
- |sᵢ> is the basis state for the site, so this is site-specific

    - Periodic: the last Dⱼ is just D₁
    - Open: D₁ = 1
    - Parameter D relates to entanglement between particles; for qubits sᵢ = {0,1}
- So this is all about SVD because the goal of MPS is to separate each wavefunction into physical degrees of freedom for each site, so that ψ can be written as the product of N matrices with each matrix corresponding to a specific site
- Ways of representing using MPS:
    - Left canonical decomposition: start by separating the first index s₁
        - |Ψ> = Σₛ (ψₛ₁, \_(s₂...sₙ)) |s₂...sₙ>
        - s₁ is treated as a row index while the others are columns
        - So ψ is a tensor and we use tensor separation

### Tensors: 
Reference paper : https://www.grc.nasa.gov/www/k-12/Numbers/Math/documents/Tensors_TM2002211716.pdf

- Scalars are tensors of rank 0, vectors are rank 1
- Multiplying a vector with a rank-2 tensor results in a vector with new magnitude and direction
- Dyads: dyad product is neither a cross nor dot product
    - U = u₁i + u₂j + u₃k
    - V = v₁i + v₂j + v₃k
    - Dyad product is UV
    - UV = u₁v₁ii + u₂v₂jj + u₃v₃kk + all the cross terms like u₁v₂ij, u₃v₁ki...
    - This turns into a 3x3 matrix, not commutative
     - For 3D space, the dimensions of these tensors are 3ⁿ

#### Some Rules:
1. All scalars are not tensors, although all tensors of rank 0 are scalars
2. All vectors are not tensors, although all tensors of rank 1 are vectors
3. All dyads or matrices are not tensors, although all tensors of rank 2 are dyads or matrices
4. We have examined, in some detail, properties and operating rules for scalars, vectors, dyads, and matrices
5. We now extend these rules to tensors per se. We assert that:
6. Tensors can be multiplied by other tensors to form new tensors
7. The product of a tensor and a scalar (tensor of rank 0) is commutative
8. The pre-multiplication of a given tensor by another tensor produces a different result from post-multiplication; i.e., tensor multiplication in general is not commutative
9. The rank of a new tensor formed by the product of two other tensors is the sum of their individual ranks
10. The inner product of a tensor and a vector or of two tensors is not commutative
11. The rank of a new tensor formed by the inner product of two other tensors is the sum of their individual ranks minus 2
12. A tensor of rank n in three-dimensional space has 3ⁿ components

#### Tensor Contractions
- So if you have an n-dimensional tensor which is just a product of n vectors, you can create a new tensor of dimension n-2 by including a dot between any consecutive terms
- Essentially, for dyads, you go from a matrix to a scalar
- Difference between position vectors can be a tensor
- Tensor analysis needs to account for coordinate independence

#### Covariance and Contravariance
- Recall that in a generalized coordinate system:
- The coordinate axes are general curves – we will call them u, v, w, ... , a, b, c, ....
- The coordinate axes are not necessarily orthogonal
- Pairs of coordinate axes uniquely determine curvilinear surfaces as product spaces. These surfaces are the coordinate surfaces of the system
- We can specify local coordinate axes at any point P in the system just as we can specify local Cartesian axes at any point in a Cartesian system
- Similarly, we can specify local coordinate surfaces at any point P in the system
- We can use the local coordinate curves and the local coordinate surfaces to specify unique sets of unit vectors at P
- We can write any vector quantity V at P as a linear combination of these local unit vectors
    - At any point P in a generalized system, we can specify two related but distinct sets of unit vectors
        - A set tangent to local axes: contravariant
        - A set perpendicular to local axes: covariant

### Time evolution of many-body quantum systems with matrix product operators 

- If the system has local interactions then we can encode these in a tensor network like a matrix product operator
- Magnus expansion: contains nested commutators. The nested commutators appear because the Magnus expansion is solving a specific problem: how do you exponentiate a time-dependent operator when operators at different times do not commute?
    - Since Ω(t) is a matrix, you cannot just take the derivative, you have to use the matrix identity

- Then you approximate the Magnus function's exponential part using Chebyshev polynomials:
    - T₀ = I
    - T₁ = Ω(n)
    - Tᵢ₊₁ = 2Ω(n)Tᵢ + Tᵢ₋₁

**Assumptions made for MPOs**

- Short-range interactions
- N-body system

    - H₀ = Σ W(1)s₁.s₁' ... W(n) sN.sN' |s><s'|
    - |s> is all values of s from 1→n
    - i is {2..n-1}
    - Wⁱ is an order-4 tensor with dimensions dᵢ x d'ᵢ x rᵢ x r'ᵢ
    - W¹ and Wᴺ both impose boundary conditions
    - Since we are working with qubits, the physical dimensions d are 2, so both d and d' are 2
    - This makes it so that sᵢ and s'ᵢ are {0,1} for all i values

- Solution of TDSE in MPO form is:
    - U(T) = Σ T...|s> <s'|

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













    
