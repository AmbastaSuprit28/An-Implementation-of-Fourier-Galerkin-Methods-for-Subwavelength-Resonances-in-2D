# An-Implementation-of-Fourier-Galerkin-Methods-for-Subwavelength-Resonances-in-2D
It's a small project demonstrating the principles in the attached research paper , In what would be simpler terms it's a use of relatively less computationally expensive BEMs over FEMs to calculate the subwavelength resonant frequencies of an 2D acoustic metamaterial.
The code uses boundary integral methods and a Fourier–Galerkin reduction developed in the paper to compute subwavelength resonant frequencies of two-dimensional acoustic metamaterials. By working on the resonator boundaries rather than discretizing the entire domain, the approach can be significantly more computationally efficient than traditional finite element methods (FEM) for this class of problems.
Disclaimer:Being only familiar with C much of the code was written by AI agents as this project was completed as an independent study of the paper.While I would not have been able to write the entire codebase from scratch, I studied the mathematical framework, directed the implementation, validated the numerical results, and can explain the underlying theory and algorithms used( hopefully 😭).
Truth to be told I Overestimated my abilities when I started reading this paper because I already had an A(9/10) grade in my Numerical Methods course I just assumed how hard it could be but the paper was filled with a lot of stuff I didn't knew before it starts from the Helmholtz PDE which the acoustic wave satisfies normally as we do in numerical methods we generate a mesh and solve by various techniques but those are relatively high in cost of computing as the discretize the whole domain and that's where this paper comes in part it asks can we get rid of the domain except the boundary?It starts with Green's Function for 2D Helmholtz PDE which involves the Hankel function and turns every point on the boundary to a tiny source acting on each other so instead of finding the field everywhere we find the strength of the sources on the boundary and take the integral of the Green's Function at point r where we are evaluating the field due to a point r' on the boundary times the weight function at r' say ϕ(r') on the boundary and over the boundary the integral gives us the acoustic pressure at r.
Now when we evaluate this field at a point just outside and inside the boundary say for a point u on the boundary therefore we find the acoustic pressure at u+ε and u-ε and as the the ε shrinks we can apply the transmission conditions involving continuity like when we do on the problem of transmission of waves on one string to other to get out system of boundary integral equations in terms of weight functions at each point on the boundary of some form like A[ϕ]=0 and get rid of our PDE but we still need to find our weight functions we start by writing the weight function in terms of finite fourier modes so each r' on boundary ϕ(r') is determined by Fourier Coefficients so the problem of Solving for N boundary Points become the problem of finding 2F+1 unknow coefficients and the matrix of ϕ=Ba where B[j,n]=exp(i*n*thetha[j]) 
Substituting into the boundary equation:
A[ϕ]=0
ABa=0.
Because we've restricted ourselves to a finite Fourier space, there may be no exact solution.
Instead we get ABa=r,where r is the residual error.Galerkin now demands that this r should not have any component along the fourier modes we have considered in other terms orthogonal to it i.e. (Conjgate Transpose of B)r=0 since r=ABa back substitution gives (B∗)ABa=0 which is further solved to get the fourier coefficients.In my code I have take F=0 for a simpler demonstration so that each weight becomes constant for a given (r,r') now the linear system we obtain earlier to have a non trivial solution det((B∗)AB)=0 so we now try to find the values of omega so that the det becomes zero and from there with some approximations we arrive to the we arrive at the effective matrix formulation used in the paper. In the subwavelength regime (very small omega and contrast parameter δ), the higher Fourier modes contribute much less than the constant mode. Since I took F=0 for a simpler demonstration, the weight function on each resonator is approximated by a constant value and the huge Galerkin system collapses to a tiny effective matrix whose size is simply the number of resonators.For my test geometry consisting of two resonators (a circle and an ellipse), this reduces the problem to solving a 2×2 nonlinear matrix equation of the form Ceff(ω,δ)x=0. The resonance frequencies are then obtained by finding the values of ω for which det(Ceff)=0. Since the resonators can radiate energy into the surrounding medium, these frequencies are generally complex-valued, where the real part corresponds to the oscillation frequency and the imaginary part corresponds to the rate at which the resonance decays. 
I also ploted the variation of contrast factor  to see if the code matches the scaling suggested by the paper (and it also contributed to a sanity check since I didn't write most of the code).
Overall I'd say that this was really an humbling experience and now I feel like I don't know anything about anything for some reason.This project was more of a demonstration of the concept to build something usable in real life acoustic engineering would definately take the time but until then this is it.
Link to The very interesting paper by Jinghao Cao(Computing and Mathematical Sciences Department, California Institute of Technology, Pasadena, CA 91125,
USA ( jinghao.cao@caltech.edu)): https://arxiv.org/abs/2605.23251

Some Plots For a quick visualization
The 2D metamaterial we ran the program on
<img width="1604" height="1638" alt="geometry" src="https://github.com/user-attachments/assets/9f4c7b2b-68f1-4072-8f05-dd684dcda0b2" />
Plot of Resonance Search
<img width="2105" height="1408" alt="resonance_search" src="https://github.com/user-attachments/assets/248240b9-bcfd-467e-840f-f1a0e1f8eb28" />
Plot of Complex  Resonance Map
<img width="2145" height="1638" alt="complex_resonance_map" src="https://github.com/user-attachments/assets/d6804434-d06e-44c9-be13-85a273294f34" />
Plot of |ω| vs  δ (Program Vs Paper's Predictions)
<img width="1876" height="1408" alt="delta_scaling" src="https://github.com/user-attachments/assets/81e8d72d-2ff3-49fc-a66e-e8c33dcb9e27" />
...

## Code Structure
### makecircle.py [makecircle.py](https://github.com/user-attachments/files/29060879/makecircle.py)

generates the boundary coordinates for circular boundary in the 2D metamaterial

### makeellipse.py
generates the boundary coordinates for elliptical boundary in the 2D metamaterial

### geometry.txt
contains the contrast and coordinates of the Bounded domains of the 2D metamaterial 

### geometry_parser.py
Reads boundary coordinates from geometry.txt.

### boundary.py
Computes normals, arc lengths, quadrature weights, area and perimeter.

### operators.py
Constructs boundary integral operators using the Helmholtz Green's function.

### fourier.py
Builds the Fourier basis used in the Galerkin projection.

### galerkin_projection.py
Projects boundary operators onto the Fourier basis.

### reduced_operator.py
Constructs the effective low-frequency operator Ceff(ω,δ).

### solver.py
Computes resonant frequencies by solving Ceff(ω,δ)x = 0.

### main.py
Driver script that loads geometry, assembles matrices, solves for resonances, and generates plots.
