# 2026-01-09
* [Experiment](../notebooks/2026-01-07-linear-separability-with-random-labels.ipynb): Does training on randomly labelled data lead to linear separability in the neural network?
  * When we randomize the labels, the network can still perfectly fit the training data but, of course, it is impossible to generalise to the test data.
  * However, does the network still learn something useful? Does linear separability (on the real labels) still increase in the various layers of the network as it learns to memorize randomly labelled data?
  * Yes, but very slowly/weakly. I had to step back from full AlexNet not a smaller version of AlexNet to make this work at all (with 100 epochs of training). But there does seem to be signal, achieving a maximum separability accuracy around 50% in the second convolutional layer of the network.
  * New hypothesis: Using non-random but automatically labelled data (i.e. deterministically computed labels) would provide better signal-to-noise for learning representations and more separability.
  * [As before](../notebooks/2026-01-07-train-and-compare-modified-alexnet.ipynb), the local response norm layers are always detrimental and should probably be removed. [A follow-up experiment](2026-01-09-linear-separability-with-random-labels-sans-local-response-norm.ipynb) shows that the modified network memorizes the training data in fewer than 50 epochs rather than 100, so ~2x faster. There is other weird behavior though. I don't fully understand this yet.

# 2026-01-07
* [Reproduced](../notebooks/2025-12-26-reproduce-alain-2016-understanding.ipynb) linear probe results from Alain & Bengio. (2016). [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644)
  * Learned how to modify the structure of an existing model — to attach, and then optimize the linear probe.
  * Learned that Kaggle provides 30 hours/week of free GPU/TPU compute. (You need phone and identify verification for access to GPUs/TPUs.)
  * Developed two follow-up hypotheses:
    * Dropping the local result normalization layers from AlexNet should improve results (see below for results)
    * Training on randomized labels should improve some layers but not all of them — probably improving earlier, convolutional layers but not later, fully connected layers. If true, this would be a good sign for unsupervised pre-training (since you can just attach random labels to unsupervised inputs).
* [Experiment](../notebooks/2026-01-07-train-and-compare-modified-alexnet.ipynb): Does removing the local response normalization layers from AlexNet improve performance?
   * Accuracy improves significantly faster when removing the local response normalization layers, but ultimately, it still converges to the same accuracy as the original model.
   * The results are more complicated when using loss rather than accuracy as the metric.
     * Training loss is consistently better in the modified model.
     * Validation loss starts of being better and then gets worse than the original model after ~10 epochs of training.
     * If we ignore after how much training the minimum validation loss is achieved, the two models have approximately equal minimum validation loss.

# 2025-12-26
* [Started work on reproducing](../notebooks/2025-12-26-reproduce-alain-2016-understanding.ipynb) linear probe results from Alain & Bengio. (2016). [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644)
  * Created full AlexNet in PyTorch.
  * Learned how to save/load model state (weights/parameters).

# 2025-12-24
* [Reproduced](../notebooks/2025-12-21-reproduce-zhang-2017-understanding.ipynb) Figure 1 from Zhang, et al. (2017). [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530).
  * Learned how to create an MLP and a simplified AlexNet in PyTorch.
  * Learned how to run a training and validation loop in PyTorch.

# 2025-12-20
* Paper on representations: Alain & Bengio. (2016). [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644)
  * They attach a completely linear layer to different hidden layers in a deep net. The linear layer maps from the activations in the hidden layer to the number of classes being predicted.
  * Some layers are really good at predicting the classes, which means the classes are linearly separable at that hidden layer.
  * Some deep nets get monotonically better at linear separation as you go deeper into the layers.

# 2025-12-19
* Paper on representations: Bengio, et al. (2013). [Representation learning: A review and new perspectives](https://arxiv.org/abs/1206.5538)
  * Early, pre-deep-learning required extensive feature engineering to work well. A lot of the engineering work around that time was feature engineering.
  * Deep networks are able to learn their own represetations (or features) directly.
  * Lots of regularization and optimization improvements over a long period of time eventually made training deep networks directly feasible.

# 2025-12-17
* Paper on generalization: Recht, et al. (2019). [Do ImageNet classifiers generalize to ImageNet?](https://arxiv.org/abs/1902.10811)
   * Test if models that do well on CIFAR-10 and ImageNet test data (i.e. generalize from training data) still do well on new data sets, created to mimic the data distributions of the originals.
   * They all do worse (than on the original test set) but in a predictable way — the rank order remains the same.
   * ![Results image from the Recht, et al. 2017 paper](../images/2025-12-17-recht-paper.png)
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
