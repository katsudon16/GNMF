## Graph regularized NMF implementation

### Notes

- The current structure follows the example files structure in [nimfa](https://github.com/marinkaz/nimfa)
- COIL20 image dataset is used as one of the testing dataset
- The algorithm iterates 100x and manages to produce decreasing objective function values

### Main todo list

- Test with lambda = 0
- Compare the result with the paper using the accuracy
- Test divergence method
- Create README for testing

### Variables naming

![image](https://user-images.githubusercontent.com/7066351/73201219-88dcc380-4106-11ea-8bca-cc1dbb3b2cb7.png)

(instead of V = W x H)
