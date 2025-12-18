# 2025-12-17
* Paper on generalization: Recht, et al. (2019). [Do ImageNet classifiers generalize to ImageNet?](https://arxiv.org/abs/1902.10811)
   * Test if models that do well on CIFAR-10 and ImageNet test data (i.e. generalize from training data) still do well on new data sets, created to mimic the data distributions of the originals.
   * The all do worse (than on the original test set) but in a predictable way — the rank order remains the same.
   * ![Results image from the Recht, et al. 2017 paper](2025-12-17-recht-paper.png)
   * Their motivated hypothesis is that this is mainly due to distributional shift between the original data set and the one they created, rather than poor generalization.

# 2025-12-16
* [PyTorch "Learn the Basics" tutorial](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
   * Autograd; computing gradings in a neural network; accumulating gradients; resetting gradient accumulators
   * Optimization algorithms; running a train-test loop
   * Saving and loading model weights
* Paper on generalization: Zhang, et al. (2017). [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530).
   * Deep neural networks can memorize input images with randomized labels perfectly in practice, but also provably with at 2n+d (n: training set size; d: input dimension) parameters. They can also memorize randomized input images (random pixel values). So where does generalization come from?
   * Explicit weight regularization accounts for a small amount of generalization but not nearly enough of it to offer a full explanation of generalization.
   * Stochastic gradient descent regularizes implicitly. Unclear how important this is is practice.
   * They have a neat way of finding the global optimum for small data sets (up o 100k images on a 24-core, 256GB workstation) by solving a linear system, which shows excellent generalization without any regularization (Equation 3, p. 9).

# 2025-12-15

* [PyTorch "Learn the Basics" tutorial](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
   * Installed PyTorch
   * Tensor basics
   * Loading, iterating, and batching data for training and test runs: Data, DataLoader
   * Transforms for inputs and outputs/labels
   * Building a neural network with linear, softmax, and ReLU layers by subclassing nn.Module
