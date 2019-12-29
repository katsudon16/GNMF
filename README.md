## Graph regularized NMF implementation

### Notes

- The current structure follows the example files structure in [nimfa](https://github.com/marinkaz/nimfa)
- COIL20 image dataset is used for testing
- The algorithm iterates 100x and manages to produce decreasing objective function values

### Todo list

- Fix divergence update function in NMF code (sparse matrix issue)
- Add divergence update method in GNMF
- Create synthetic dataset
- Make the code more generic (not specific to the dataset used)
- Improve the convergence test
