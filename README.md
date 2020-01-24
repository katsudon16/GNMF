## Graph regularized NMF implementation

### Notes

- The current structure follows the example files structure in [nimfa](https://github.com/marinkaz/nimfa)
- COIL20 image dataset is used as one of the testing dataset
- The algorithm iterates 100x and manages to produce decreasing objective function values

### Main todo list

- Test with different matrix shapes and # iterations to
  i) check if the implementation is correct
  ii) check if the implementation is scalable
- Compare the result with the paper
- Plot and compare the reconstruction errors between GNMF and vanilla NMF methods
- Plot the objective function values vs iterations
