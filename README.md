# Direct Poisson Neural Networks

[![Documentation](https://img.shields.io/badge/docs-latest-blue?style=flat-square)](https://enaipi.github.io/direct-Poisson-neural-networks/)

## There are the following scripts available:

## Typical workflow (samples initial conditions, simulates, learns, simulates learned, and compares):

    comparison --generate --steps=100 --implicit --soft --without --model=RB --folder_name=TEST
    plot-compare --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST

## It is also possible to compare just training and validation losses 

    compare-train-errors

## Alternatively, you can do that step by step
First generate dataset for training with:

    simulate --generate --steps=50000 --model=RB

for the rigid body (or HT for heavy top, or P3D for the particle in three dimensions)

Then we train implicit and soft networks:

    learn --method=without --model=RB

(or implicit or soft).

Then choose different initial conditions and see how well our network fits the evolution. If the initial condition is too different we will not get a good fit. If it is the same we will fit perfectly. 

    simulate --steps=500 --generate
    simulate --steps=500 --implicit
    simulate --steps=500 --soft
    simulate --steps=500 --without

## And we can plot and see:

    plot-compare --plot_m --plot_E --plot_L

Check training error and errors while learning. 

## Building the Documentation

Install documentation dependencies:

    pip install -e ".[docs]"

Build HTML documentation:

    cd doc
    make html

The built documentation will be available in `doc/_build/html/`

## Typical arguments used for the training can be found in folder 

    typical_args

Please cite as [M. Šípka, M. Pavelka, O. Esen, and M. Grmela, Direct Poisson neural networks: learning non-symplectic mechanical systems, Journal of Physics A: Mathematical and Theoretical 56(49), 2023.](https://iopscience.iop.org/article/10.1088/1751-8121/ad0803)
