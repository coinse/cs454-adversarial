# CS454 Genetic Programming for Adversarial Examples

As described in the class, adversarial examples are a manifestation of metamorphic testing on machine learning modules. This coursework aims to help you generate adversarial examples of your own, both of the $\epsilon$-ball mathematical adversarial example variety and the 'realistic' adversarial examples perhaps more important to practical applications (like adding fog).

## Aim

In the first part of this coursework, you will be loosely recreating results from the earlier papers that made adversarial examples, but instead of using their gradient-based optimization schema, you will use a genetic algorithm to find vectors that cause misclassification.

 * [Goodfellow et al., _Explaining and Harnessing Adversarial Examples_](https://arxiv.org/abs/1412.6572)
 * [Carlini and Wagner, _Towards Evaluating the Robustness of Neural Networks_](https://arxiv.org/abs/1608.04644)
 * [Su et al., _One pixel attack for fooling deep neural networks_](https://arxiv.org/abs/1710.08864)

In the second part, you will implement "realistic" perturbations that are nonetheless scalable. The effects are similar to the following paper:

 * [Tian et al., _DeepTest: Automated Testing of Deep-Neural-Network-driven Autonomous Cars_](https://arxiv.org/abs/1708.08559)

Specifically, in this coursework, you will be generating adversarial examples based on the 'seed' images provided in the `images/` directory. In this assignment, an "adversarial example" is defined as an image which is modified from the original image such that the original (correct) label is no longer the most likely prediction according to the model. See the example results in `images/examples/`.

## Setup

 * First, install the non-PyTorch dependencies via `requirements.txt`: `pip install -r requirements.txt`.
 * If you have a GPU and CUDA installed, install PyTorch using `torch_gpu.txt`; otherwise, use `torch_cpu.txt`.
   * Note that grading will be done on a CPU machine for fairness.

## Genetic Algorithm

### $\epsilon$-ball Adversarial Examples

In the first task, you will implement genetic algorithms that find adversarial examples (more specifically, adversarial noise) that differ from the original image by a maximum of $\epsilon$. The specific targets are:

 * The $L_\infty$ norm, which measures the _maximum_ difference between the original pixels and the adversarial example. A good $L_\infty$ adversarial example will change almost all the pixels by a small value.
 * The $L_2$ norm, which measures the Euclidean distance between the original image and the adversarial image.
 * The $L_0$ norm (actually mathematically not a norm) measures the number of pixels which changed between the original and adversarial image. A good $L_0$ adversarial example would change a few pixels by a large amount.

Implement a GA for each norm, including a fitness function, in `epsilon_ball.py`. Your task is to generate noise that can be added to the image tensor, so that the neural network (VGG-11) will mispredict the image. The norms themselves are implemented in `util.py`, so you may make use of them. (The observant of you will notice that e.g. $L_0$ norm does not actually measure the number of pixels changed, but on the number of tensor elements changed. This is intended; use the `get_pixel_norm_for` function if you want to play with that, although it will not be used for evaluation.) Using the `get_viz_for` function, you may generate an image which visualizes the top ten likely classes and their probabilities according to the VGG-11 model. Look to the `images/init_viz/` directory for visualization examples for the original images.

For the first task, you will be evaluated based on the smallest $\epsilon$ you could achieve for each of the five images for each of the three norms (15 numbers in total). Use a table to report your results.

### Bonus: "Foggy" Adversarial Examples

In the second task, instead of generating meaningless noise, you will use [Perlin noise](https://en.wikipedia.org/wiki/Perlin_noise) to implement "fog" that in turn induces DNN misprediction. In this scenario, instead of directly manipulating the noise vectors as you did in the first task, you are to evolve the parameters of Perlin noise to induce misclassification. An example of combining Perlin noise is provided [on the Perlin noise package webpage](https://pypi.org/project/perlin-noise/), also presented here for completeness:

```python
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

noise1 = PerlinNoise(octaves=3)
noise2 = PerlinNoise(octaves=6)
noise3 = PerlinNoise(octaves=12)
noise4 = PerlinNoise(octaves=24)

xpix, ypix = 100, 100
pic = []
for i in range(xpix):
    row = []
    for j in range(ypix):
        noise_val = noise1([i/xpix, j/ypix])
        noise_val += 0.5 * noise2([i/xpix, j/ypix])
        noise_val += 0.25 * noise3([i/xpix, j/ypix])
        noise_val += 0.125 * noise4([i/xpix, j/ypix])

        row.append(noise_val)
    pic.append(row)
```

For this task, evolve a list of [(octave, magnitude)] parameters that will be added as above; for example, the representation for the noise in the code above would be `[(3, 1), (6, 0.5), (12, 0.25), (24, 0.125)]`. 

As we still need some sort of metric to judge how much obscuring the "fog" is doing, the sum of the magnitude parameters is used to evaluate the degree of obscuring. Again, having a "weaker" fog according to this metric and still causing misclassification is the goal.

### Deliverables and Report

Based on the contents of your `epsilon_ball.py` / `perlin_fog.py`, we will evaluate the $\epsilon$ that your solution achieves on a set of held-out images. Make sure not to change the command line interface, as your solution will be automatically graded. Your solution is expected to take three minutes at maximum (on a CPU machine) to generate an adversarial example for `epsilon_ball.py`, and 10 minutes (CPU) for `perlin_fog.py`.

Finally, include a PDF report of your efforts when you are done. In the report, include the following: 

 1. A description of the strategies that you used for each of the norms, and whether you felt compelled to use different strategies in each case
 2. The 15 images that you evolved for each seed image and norm, their $\epsilon$ distance from the original image, and the classification results as visualized by the `get_viz_for` function.
 3. (Bonus) The evolved Perlin noise expression, and a visualization of the best foggy images generated.