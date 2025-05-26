<h1 align="center"> Latent Guided Sampling for Combinatorial Optimization </h1>


We introduce a novel latent space model for Combinatorial Optimization and propose Latent Guided Sampling (LGS), an efficient inference method based on Markov Chain Monte Carlo and Stochastic Approximation.
This repository provides a full implementation of the proposed latent space model and the LGS inference method, including all the code required to reproduce the experiments. It also contains problem instances and pretrained models for the Traveling Salesman Problem (TSP) and the Capacitated Vehicle Routing Problem (CVRP). While the model can be trained from scratch, pretrained models are provided to enable efficient inference and reproduction of results without extensive computational cost.

### 🛠 Environment Setup
To set up the environment, execute the following commands:

```shell
conda create -n lgs python=3.9
conda activate lgs
pip install -r requirements.txt
```

### 🚀 Usage Instructions

To train the model from scratch, run:

```shell
python run_train.py
```

To perform inference on instances of size 100, run:
```shell
python run_test.py --problem_size 100
```


### 🙏 Acknowledgments
We gratefully acknowledge the following repositories which served as baselines and inspiration for our implementation: 
* https://github.com/wouterkool/attention-learn-to-route
* https://github.com/yd-kwon/POMO




