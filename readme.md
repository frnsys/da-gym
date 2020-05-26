# da-gym

_Data Analysis Gym_

Randomly generates data for data analysis exercises. You have access to all the underlying parameters, functions, etc so you can compare your results to the "truth".

### Features

- generates a random causal DAG
- generates discrete and continuous features
- generates the entire population, from which a sample is generated
- related variables are related through randomly generated polynomials
- variables that are root nodes in the DAG are sampled from normal or beta-binomial distribtions
- some variables may be unobservable

### Possible improvements

- the sample generated from the population assume a perfectly random sampling procedure. Could introduce biases into that so that e.g. sub-populations are over/underrepresented in the sample.
- generate specific questions to answer
- time series support
- open to other ideas as well!

## Usage

```python
from da_gym import generate_exercise

sample_df, world = generate_exercise()

# `sample_df` is the dataframe you analyze
# `world` contains info to check against.
# for example, to view the DAG:

import matplotlib.pyplot as plt
nx.draw_networkx(world['dag'])
plt.axis('off')
plt.show()
```