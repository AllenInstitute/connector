<h1 align='center'>Connector</h1>
<h2 align='center'>Identifying Connectivity Distributions from Neural Dynamics Using Flows</h2>

The brain uses a set of rules—which we call "dynamics"—to perform computation (e.g., decision-making, working memory, navigation...). How do these rules—or dynamics—arise from the way neurons in our brain are connected to each other? Inferring connectivity from dynamics alone is fundamentally challenging because many different connectivities can generate identical dynamics. To address this, we identified the mathematical source of this degeneracy, and developed a framework that infers, among the set of possible connectivities, the simplest one that is consistent with the data. This helps us tell apart which component of the inferred connectivity is necessary for the observed dynamics, and which is just guesswork. This allows us to focus on the component of the connectivity that matters, and generate sharper, experimentally testable hypotheses about how the brain works. 

# Installation
Run the commands below to install Connector:

```
$ git clone https://github.com/AllenInstitute/connector
$ module load anaconda/2024.10
$ conda create --name connector python=3.12
$ conda activate connector
$ cd connector
$ pip install -e .
```

# Citation

Kim, T.D., Pereira-Obilinovic, U., Wang, Y., Shea-Brown, E., Sümbül, U. (2026). Identifying Connectivity Distributions from Neural Dynamics Using Flows. *Proceedings of the 43rd International Conference on Machine Learning (ICML)*.

```bibtex
@article{kim2026connector,
    author={Timothy Doyeon Kim and Ulises Pereira-Obilinovic and Yiliu Wang and Eric Shea-Brown and Uygar Sümbül},
    title={Identifying Connectivity Distributions from Neural Dynamics Using Flows},
    year={2026},
    journal={Proceedings of the 43rd International Conference on Machine Learning (ICML)}
}
```