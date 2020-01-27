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

### Variables naming

![image](https://user-images.githubusercontent.com/7066351/73201219-88dcc380-4106-11ea-8bca-cc1dbb3b2cb7.png)

(instead of V = W x H)
