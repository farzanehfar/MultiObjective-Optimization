# Multi Objective Optimisation

## This code was written by the following contributors for the publication, "Data-Driven Multi-Objective Optimisation for Electric Vehicle Charging Infrastructure":

### Farzaneh Farhadi (F.Farhadi2@newcastle.ac.uk): Farzaneh holds an MSc in Computer Science and is currently a PhD student at Newcastle University, UK.

### Shixiao Wang (Forainest789@gmail.com): Shixiao has an MSc in Computer Science from Newcastle University, UK.

```bibtex
@article{farhadi2023data,
  title={Data-Driven Multi-Objective Optimization for Electric Vehicle Charging Infrastructure},
  author={Farhadi, Farzaneh and Wang, Shixiao and Palacin, Roberto and Blythe, Phil},
  journal={iScience},
  pages={107737},
  year={2023},
  publisher={Elsevier}
}
```

## Here are the steps for constructing the proposed optimisation model for electric vehicle charging infrastructure:
1. Analysis and processing of data to determine the subject and environment of optimisation.
2. Construct gene vectors, determine genotypes and score types, and construct the initial solution vector space,
i.e. construct the initial population.
3. Through the processing of real data, a resource vector is designed through mapping and an environment
matrix is constructed.
4. Study the specific expression of the design score type through the influencing factors of the real problem, and
complete the construction of multiple objective functions.
5. Use the elite strategy of fuzzy logic to rate the score types.
6. Memorise and eliminate the initial solution vector matrix based on the ratings by constructing a transformation
matrix. This is similar to the memory gate and forgetting gate of LSTM models.
7. Generate the next generation of children based on the rating content and child generation matrix, and add
them to the solution vector matrix space.
8. The environment matrix is regenerated according to the mapping rules.
9. Rating by calculating the score type and then calculating the crowding by the inner product for the exemplar
elite, so that the crowding is in the right zone to ensure species diversity and also the right direction for
optimisation and search.
10. Repeat steps 5 to 9 until the generational requirements are satisfied


# Other Publications by corresponding author "Farzaneh Farhadi": 
## Publication 1 in "Machine Learning for Transport Policy Interventions on Air Quality": https://ieeexplore.ieee.org/document/10114913

```bibtex
@article{farhadi2023machine,
  title={Machine Learning for Transport Policy Interventions on Air Quality},
  author={Farhadi, Farzaneh and Palacin, Roberto and Blythe, Phil},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}
```

## Publication 2 in "Data-driven framework for validating policies: Air quality case study": https://ieeexplore.ieee.org/abstract/document/9922587

```bibtex
@inproceedings{farhadi2022data,
  title={Data-driven framework for validating policies: Air quality case study},
  author={Farhadi, Farzaneh and Palacin, Roberto and Blythe, Phil},
  booktitle={2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC)},
  pages={237--243},
  year={2022},
  organization={IEEE}
}
```
