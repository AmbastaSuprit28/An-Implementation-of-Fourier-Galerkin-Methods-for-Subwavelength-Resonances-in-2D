# An-Implementation-of-Fourier-Galerkin-Methods-for-Subwavelength-Resonances-in-2D
It's a small project demonstrating the principles in the attached research paper , In what would be simpler terms it's a use of relatively less computationally expensive BEMs over FEMs to calculate the subwavelength resonant frequencies of an 2D acoustic metamaterial.
The code uses boundary integral methods and a Fourier–Galerkin reduction developed in the paper to compute subwavelength resonant frequencies of two-dimensional acoustic metamaterials. By working on the resonator boundaries rather than discretizing the entire domain, the approach can be significantly more computationally efficient than traditional finite element methods (FEM) for this class of problems.
Disclaimer:Being only familiar with C much of the code was written by AI agents as this project was completed as an independent study of the paper.While I would not have been able to write the entire codebase from scratch, I studied the mathematical framework, directed the implementation, validated the numerical results, and can explain the underlying theory and algorithms used( hopefully 😭).
Truth to be told I Overestimated my abilities when I started reading this paper because I already had an A(9/10) grade in my Numerical Methods course I just assumed how hard it could be but the paper was filled with a lot of stuff I didn't knew before it went from Helmholtz Oscilllators to Green's function to Hankel Functions and the general solution and the using each point on the boundary as a source with a weight with respect to each point to be finally evaluated inside the boundary and outside the boundary and then using continuity conditions at the boundary to arrive at the solution. The main question was how to find out that weight function which they did by using Fourier basis to guess the weights    

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
