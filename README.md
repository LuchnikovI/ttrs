# TTRS - Tensor Train format for Rust
This is an attempt to implement the [Tensor Train](https://sites.pitt.edu/~sjh95/related_papers/tensor_train_decomposition.pdf) (TT) format for processing of large low-rank tensors.
Currently, this repo contains an implementation of the [TTCross](https://www.sciencedirect.com/science/article/pii/S0024379509003747) algorithm for efficient reconstruction of Tensor Train representations from "black-box"
functions, subroutines for setting a TT representation into canonical forms, subroutines for truncation, dot product,
arithmetic operations with Tensor Trains.

We also provide a python wrapper for this package that currently works with only double Complex dtype.

!!!The code is at the early stage of development and might contain severe bugs including safety issues.

# Dependencies/Prerequisites
The code relies on the lapack and [blis](https://github.com/flame/blis) libraries. In particular, it uses src crates [blas-src](https://crates.io/crates/blas-src) with
blis feature and [lapack-src](https://crates.io/crates/lapack-src) with openblas feature. Blis is being downloaded and built at compile time automatically, while openblas
and lapack must be presented on your computer.

# How to instal a version for Python
Clone this repo on your local machine. Create a python virtual environment via

    python -m venv <name>

or use an existing one. Activate the virtual environment. Install maturin in the virtual environment via 

    pip install maturin 

Go to the local directory with the cloned repo. Build the python package by running 

    maturin develop --release

in the cloned repo directory. Now you have ttrs package in your virtual environment.

# Examples of Python code
Currently, in the root directory of the package one can find the following ipynb's with examples:
- optimization_example.ipynb: example of TT based global optimization of 1d function.

# Examples of Rust code
Currently, there are two examples written in Rust:
    - example of TTCross based reconstruction of a Markov decision process (MDP) model that is discussed further here;
    - example of TT based global optimization of 1d function
    To run these examples execute the following command `cargo run --release --example mdp` for the first example and `cargo run --release --example opt` for the second one.

## MDP model reconstruction with TTCross
Let as consider the following controll problem. On has a field where a robot can move in four directions: up, down, left and right. The field contains obstacles
and a target, which is a desirable destination point of a robot. The problem has periodic boundary conditions, i.e. if a robot reaches the rightmost point, it appears
in the leftmost point, preserving its Y-coordinate. The same holds for upmost and downmost points. The field can be schematically represented as
![follows](https://github.com/LuchnikovI/ttrs/blob/main/readmepics/field.pdf). One can controll movements of a robot by four control signals (up, down, left, right).
After each control signal a robot moves by a single cell in a given dirrection. If a robot collides with an obstacle it preserves its position disregarding a control signal.
Robot can take a random movement to a neighboring cell ignoring a control signal with a small probability p = 0.04. The total route of the robot takes 50 subsequent control
signals. The reward is the probability to be in the target cell at the end of the total route, i.e. the reward signal is extremely sparce. Note, that the total number of
possible states (number of cells available for a robot) is equal to 37. Remember this number, it will be of a great interest later.

The first question, that might be asked is "Whether is it possible to reconstruct a model of the field by only running experiments of the following kind:
(1) send a control sequance to a robot; (2) measure a total average reward; (3) repeat this as many times as necessary." At the first glance it seems to be impossible.
Indeed, we measure a single number each time, how is it possible to extract information about the entire field from this? However, it turns out to be
possible to reconstruct a model, that predict reward with machine precision based on TTCross algorithm. The process of (1) sending a control sequance to a robot;
(2) measuring a reward, is equivalent to evaluating a "black-box" function, thus one can use TTCross to reconstruct this function. TTCross has a special
hyper parameter -- maximal TT rank. We set it to 50 hoping that it would be enough. After running the TTCross algorithm, we try to predict reward function with
the reconstructed Tensor Train given a random control signal. We get:

    Exact reward value vs predicted:
    0.037469829764664196 vs 0.03746982976569586
    0.00484408253834081 vs 0.004844082538343779
    0.006747907533891394 vs 0.006747907533936837
    0.15213174893214665 vs 0.15213174892582793
    0.012278823035485338 vs 0.012278823036365053
    0.037839227487945555 vs 0.03783922748968589
    0.016477581590068825 vs 0.01647758158827603
    0.04114224980126864 vs 0.04114224980215756
    0.02552145927703887 vs 0.025521459273494276
    0.0016822098975463973 vs 0.001682209897508306
We got a perfect prediction!
But there is more, let us compress the obtained Tensor Train with accuracy 1e-5 and have a look on its bond dimensions:

    Bond dimensions:
    [3, 6, 10, 17, 25, 35, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 
    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 35, 25, 17, 10, 6, 3]

We already saw, that the number of available states for a robot is 37, here the maximal compressed bond dimension is also 37. This is not an accident.
The TT rank of a compressed Tensor Train is the minimal possible rank that describes observed data correctly. The hidden dimenion of the model is 37, the
hidden dimenion (TT rank) of the Tensor Train should match this value in order to fit the data correctly.

As one can see, we managed not only to build a model, that predicts the reward given a control signal with machine precision, but also understand the complexity of the model (
the number of possible states that the model can take). And all this is reconstructed from extremely sparse data: onle the probability to be in a particular cell at the end
of the robot's route. Ofcourse, to make it possible, one needs to repeate experiment O(time_steps * control_dimension * rank^2) times, not forgetting about the necessity
to avarege the reward signal accrose many runs. But still, the number of experiments scales linearly with number of time steps.

The next step is to add actual reward maximization after model reconstruction. This will be done further.
