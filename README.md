# An-Implementation-of-Fourier-Galerkin-Methods-for-Subwavelength-Resonances-in-2D
It's a small project demonstrating the principles in the attached research paper , In what would be simpler terms it's a use of relatively less computationally expensive BEMs over FEMs to calculate the subwavelength resonant frequencies of an 2D acoustic metamaterial.
The code uses boundary integral methods and a Fourier–Galerkin reduction developed in the paper to compute subwavelength resonant frequencies of two-dimensional acoustic metamaterials. By working on the resonator boundaries rather than discretizing the entire domain, the approach can be significantly more computationally efficient than traditional finite element methods (FEM) for this class of problems.
Disclaimer:Being only familiar with C much of the code was written by AI agents as this project was completed as an independent study of the paper.While I would not have been able to write the entire codebase from scratch, I studied the mathematical framework, directed the implementation, validated the numerical results, and can explain the underlying theory and algorithms used( hopefully 😭).

Link to The very interesting paper by Jinghao Cao(Computing and Mathematical Sciences Department, California Institute of Technology, Pasadena, CA 91125,
USA ( jinghao.cao@caltech.edu)): https://arxiv.org/abs/2605.23251

