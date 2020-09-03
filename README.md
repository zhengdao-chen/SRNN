# SRNN
Code for the paper [*Symplectic Recurrent Neural Networks*](https://arxiv.org/abs/1909.13334), which appeared in ICLR 2020. 

By unrolling a Hamiltonian system with a neural-network-parametrized Hamiltonian function using the leapfrog integrator, and together with initial state optimization, our model is able to learn the dynamics of complex, noisy and stiff physical systems.


## Dependencies
Python 3.7

PyTorch 1.4

Matplotlib 3.1

Numpy 1.18

Scipy 1.4

## Usage
Run `script_chain_new` for the spring-chain experiment

Run `script_3body_new` for the three-body experiment

## License.

This code is made available for research and replication purposes under the CC-BY-NC 4.0 license found in file `LICENSE`.
