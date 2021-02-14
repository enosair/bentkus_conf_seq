# bentkus_conf_seq

This repo contains demo code for the following paper, to compute the confidence sequence and/or the uniform upper/lower bounds for bounded random variables using Adaptive Bentkus' method.

> Arun Kumar Kuchibhotla\*, Qinqing Zheng\*.  *Near-Optimal Confidence Sequences for Bounded Random Variables*. (\* Equal contribution.) [[arXiv:2006.05022](https://arxiv.org/abs/2006.05022)].


### Introduction

*Let Y1, Y2, . . . be independent real-valued random variables, available sequentially, with mean mu.
A 1-delta confidence sequence is a sequence of confidence intervals {CI_1, CI_2, ...} where CI_n is
constructed on-the-fly after observing data sample Yn, such that
Pr( mu in CI_n for all n >= 1) >= 1 - delta.*

Many inference problems are online nature. Examples includes sequential decision problems like A/B testing, adaptive
sampling schemes like bandit selection, etc. Practitioners often wants to stop data sampling or processing when a
required guarantee is satisfied. Confidence sequence is used to help people determine the number of samples you need, i.e.
the stopping time.

We provide a Bentkus confidence sequence that improves on the exisiting approaches that use Cramér-Chernoff techniques
(e.g. Hoeffding, Bernstein, Bennett).


### Code organization
- `conc_ineq`: Hoeffding-type, Bernstein-type, and Bentkus' methods for pointwise and uniform bounds.
- `ci_expr.py`: code to run a single uniform confidence sequence coverage experiment
- `ada_stop.py`: code to run a single adaptive stopping experiment for (epsilon, delta) mean estimation.
- `best_arm.ipynb`: code to run the best arm identification experiment and generate the related plots.
- `*.ipynb`: code to generate the plots in our paper.

### Code Dependency
- To run the HRMS Bernstein methods, you need to install [[confseq](https://github.com/gostevehoward/confseqi)].
