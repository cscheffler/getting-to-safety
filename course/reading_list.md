# Week-by-week reading list and project ideas for a 12-week course

Each week has:

* Primary papers: Must read
* Background papers: skim/optional
* Exercises: What you have to be able to do

## Week 1: Deep Learning in practice

**Primary**

* Goodfellow et al., *Deep Learning*, Ch. 6 (Optimization) – selective reading
   * TODO
* Bottou et al. (2018), *Optimization Methods for Large-Scale ML*
   * TODO

**Background**

* Wilson & Izmailov (2020), *Bayesian Deep Learning and a Probabilistic Perspective*
   * TODO

**Exercises**

* Implement a clean PyTorch training loop using CIFAR-10 and/or ImageNet (we'll reuse this in Week 2). Try out a few different models — which? (TODO)
   * See [notebooks/2025-12-19-torch-training-loop.ipynb](../notebooks/2025-12-19-torch-training-loop.ipynb)
* Understand why stochastic gradient descent (SGD) is not Bayesian inference (and why we live with it). (TODO)

## Week 2: Generalization, overfitting, and evaluation discipline

**Primary**

* Zhang, et al. (2017). [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530).
   * Deep neural networks can memorize input images with randomized labels perfectly in practice, but also provably with at 2n+d (n: training set size; d: input dimension) parameters. They can also memorize randomized input images (random pixel values). So where does generalization come from?
   * Explicit weight regularization accounts for a small amount of generalization but not nearly enough of it to offer a full explanation of generalization.
   * Stochastic gradient descent regularizes implicitly. Unclear how important this is is practice.
   * They have a neat way of finding the global optimum for small data sets (up o 100k images on a 24-core, 256GB workstation) by solving a linear system, which shows excellent generalization without any regularization (Equation 3, p. 9).
* Recht et al. (2019), [Do ImageNet classifiers generalize to ImageNet?](https://arxiv.org/abs/1902.10811)
   * Test if models that do well on CIFAR-10 and ImageNet test data (i.e. generalize from training data) still do well on new data sets, created to mimic the data distributions of the originals.
   * They all do worse (than on the original test set) but in a predictable way — the rank order remains the same.
   * ![Results image from the Recht, et al. 2017 paper](../images/2025-12-17-recht-paper.png)
   * Their motivated hypothesis is that this is mainly due to distributional shift between the original data set and the one they created, rather than poor generalization.

**Background**

* Arpit et al. (2017), *A Closer Look at Memorization in Deep Nets*
   * TODO

**Exercises**

* Replicate memorization vs generalization behavior on small datasets
* Practice rigorous train/val/test separation

---

## Week 3 — Representations as objects of study

**Primary**

* Bengio et al. (2013), [Representation learning: A review and new perspectives](https://arxiv.org/abs/1206.5538)
   * Early, pre-deep-learning required extensive feature engineering to work well. A lot of the engineering work around that time was feature engineering.
   * Deep networks are able to learn their own "represetations" (or features) directly.
   * Lots of regularization and optimization improvements over a long period of time eventually made training deep networks directly feasible.
* Alain & Bengio (2016), [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644)
   * They attach a completely linear layer to different hidden layers in a deep net. The linear layer maps from the activations in the hidden layer to the number of classes being predicted.
   * Some layers are really good at predicting the classes, which means the classes are linearly separable at that hidden layer.
   * Some deep nets get monotonically better at linear separation as you go deeper into the layers.

**Background**

* Raghu et al. (2017), *SVCCA*
   * TODO

**Applied outcome**

* Train a model and probe intermediate layers
* Treat representations as empirical phenomena

---

## Week 4 — Transformers without mysticism

**Primary**

* Vaswani et al. (2017), [Attention is all you need](https://arxiv.org/abs/1706.03762)
* Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits* (sections 1–2)

**Background**

* Karpathy, *The Annotated Transformer* (blog)

**Applied outcome**

* Load a small transformer
* Inspect attention heads and activations
* No training from scratch

---

## Week 5 — Adversarial robustness as a safety problem

**Primary**

* Szegedy et al. (2014), *Intriguing Properties of Neural Networks*
* Madry et al. (2018), *Towards Deep Learning Models Resistant to Adversarial Attacks*

**Background**

* Goodfellow et al. (2015), *Explaining and Harnessing Adversarial Examples*

**Applied outcome**

* Implement FGSM/PGD
* Reproduce robust vs standard accuracy gaps on CIFAR-10

---

## Week 6 — Distribution shift and dataset fragility

**Primary**

* Hendrycks & Dietterich (2019), *Benchmarking Neural Network Robustness to Common Corruptions*
* Torralba & Efros (2011), *Unbiased Look at Dataset Bias*

**Background**

* Koh et al. (2021), *Wilds*

**Applied outcome**

* Evaluate models under synthetic distribution shift
* Quantify performance degradation

---

## Week 7 — Interpretability and attribution

**Primary**

* Sundararajan et al. (2017), *Axiomatic Attribution for Deep Networks*
* Adebayo et al. (2018), *Sanity Checks for Saliency Maps*

**Background**

* Olah et al. (2018), *The Building Blocks of Interpretability*

**Applied outcome**

* Apply attribution methods using Captum
* Reproduce attribution failures and sanity checks

---

## Week 8 — Mechanistic interpretability (lightweight)

**Primary**

* Olah et al. (2020), *Zoom In: An Introduction to Circuits*
* Nanda & Lieberum (2022), *A Mechanistic Interpretability Analysis of Grokking*

**Background**

* Elhage et al. (2022), *Toy Models of Superposition*

**Applied outcome**

* Identify a simple circuit or feature
* Perform ablations and causal interventions

---

## Week 9 — Safety evaluations and measurement

**Primary**

* Mitchell et al. (2019), *Model Cards for Model Reporting*
* Raji et al. (2020), *Closing the AI Accountability Gap*

**Background**

* Jacobs & Wallach (2021), *Measurement and Fairness*

**Applied outcome**

* Build a safety-relevant evaluation harness
* Justify metrics and error analysis

---

## Week 10 — Prompting, failure modes, and brittle alignment

**Primary**

* Perez et al. (2022), *Red Teaming Language Models with Language Models*
* Wei et al. (2022), *Chain-of-Thought Prompting*

**Background**

* Wallace et al. (2019), *Universal Adversarial Triggers*

**Applied outcome**

* Stress-test small models or prompts
* Discover systematic failure modes

---

## Week 11 — Lightweight alignment techniques

**Primary**

* Ouyang et al. (2022), *Training Language Models to Follow Instructions* (conceptual focus)
* Rafailov et al. (2023), *Direct Preference Optimization* (high-level)

**Background**

* Christiano et al. (2017), *Deep RL from Human Preferences*

**Applied outcome**

* Run a tiny SFT or preference-style fine-tune
* Observe side effects and tradeoffs

---

## Week 12 — Reproducibility as a scientific practice

**Primary**

* Pineau et al. (2021), *Improving Reproducibility in ML Research*
* Lipton & Steinhardt (2019), *Troubling Trends in ML Scholarship*

**Applied outcome**

* Write a critical reproduction report
* Distinguish failure from insight

---

# Canonical Reproduction Projects (Reusable Year-to-Year)

These are **designed to be rotated**, not reinvented.

---

## Project 1 — Adversarial robustness on CIFAR-10

**Paper lineage**

* Goodfellow → Madry → Hendrycks

**Core claim**

> Standard models are highly brittle; robustness trades off with accuracy.

**Reproduction task**

* Train or load a small CNN
* Implement FGSM/PGD
* Reproduce robust vs standard accuracy curves

**Skills taught**

* Training loops
* Evaluation under attack
* Plotting and reporting

---

## Project 2 — Interpretability methods can fail spectacularly

**Paper lineage**

* Integrated Gradients + Sanity Checks

**Core claim**

> Many attribution methods are not actually model-dependent.

**Reproduction task**

* Apply saliency/IG
* Randomize weights or labels
* Replicate failure cases

**Skills taught**

* Model introspection
* Experimental controls
* Scientific skepticism

---

## Project 3 — Representation probing and linear separability

**Paper lineage**

* Linear probes, SVCCA

**Core claim**

> Representations evolve meaningfully across layers.

**Reproduction task**

* Train a small model
* Fit probes to layers
* Measure information emergence

**Skills taught**

* Feature extraction
* Statistical evaluation
* Representation analysis

---

## Project 4 — Safety evaluation under distribution shift

**Paper lineage**

* ImageNet-C / WILDS-style work

**Core claim**

> Reported performance collapses under mild shift.

**Reproduction task**

* Corrupt inputs or shift data
* Quantify degradation
* Diagnose failure modes

**Skills taught**

* Evaluation design
* Robust metrics
* Error analysis

---

## Project 5 (Advanced / optional) — Tiny alignment fine-tuning

**Paper lineage**

* SFT / DPO (scaled down)

**Core claim**

> Alignment methods have unintended side effects.

**Reproduction task**

* LoRA fine-tune a small model
* Measure refusal vs capability loss

**Skills taught**

* Modern fine-tuning
* Tradeoff analysis
* Causal reasoning about interventions
