# Data Driven Multi Objective Optimization

# Journal link: https://www.cell.com/iscience/pdf/S2589-0042(23)01814-X.pdf

## This code was written by the following contributors for the publication, "Data-Driven Multi-Objective optimization for Electric Vehicle Charging Infrastructure" 

### Farzaneh Farhadi (Farhadi.Farzane@gmail.com): Farzaneh holds a PhD in Artificial Intelligence from Newcastle University, UK. 

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

## Here are the steps for constructing the proposed optimization model for electric vehicle charging infrastructure:
1. Analysis and processing of data to determine the subject and environment of optimization.
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
optimization and search.
10. Repeat steps 5 to 9 until the generational requirements are satisfied



