import random
import numpy as np
import pandas as pd
import networkx as nx
from enum import Enum
from scipy import stats

class Type(Enum):
    DISCRETE = 0
    CONTINUOUS = 1

def random_polynomial(n_args, noise_scale=0.01):
    intercept = random.random()
    coefs = [random.random() for _ in range(n_args)]
    pows  = [random.randint(1, 3) for _ in range(n_args)]
    fn = lambda *args: intercept \
        + sum(c*a**p for a, c, p in zip(args, coefs, pows)) \
        + random.random() * noise_scale
    return fn, {
        'intercept': intercept,
        'coefficients': coefs,
        'powers': pows
    }

def random_dag(n, p):
    # Doesn't guarantee the graph is connected,
    # but we want to simulate the possibility of unrelated
    # variables anyways
    graph = nx.fast_gnp_random_graph(n, p, directed=True)
    return nx.DiGraph([(u, v) for (u, v) in graph.edges() if u < v])

def random_dist(typ, discrete_range=(2, 5)):
    if typ is Type.DISCRETE:
        # For now, beta-binomial
        alpha = stats.halfnorm.rvs(loc=0, scale=1)
        beta = stats.halfnorm.rvs(loc=0, scale=1)
        return {
            'name': 'betabinom',
            'params': {
                'n': np.random.randint(*discrete_range),
                'a': alpha,
                'b': beta
            }
        }

    elif typ is Type.CONTINUOUS:
        # For now, normal
        mean = stats.norm.rvs(loc=0, scale=1)
        std = stats.halfnorm.rvs(loc=0, scale=1)
        return {
            'name': 'norm',
            'params': {
                'loc': mean,
                'scale': std
            }
        }


def generate_exercise(n_features=5, discrete_range=(2,5), scale_range=(1, 100), p_discrete=0.5, p_edge=0.2, p_observed=1., pop_size=10000, sample_size=1000):
    """
    - n_features: how many features
    - discrete_range: (min, max) for discrete values
    - scale_range: (min, max) range for scaling of continuous values
    - p_discrete: probability of a discrete feature
    - p_edge: probability of edge forming in DAG
    - p_observed: probability that a feature is observed
    - pop_size: total population of examples
    - sample_size: size of sample drawn from population
    """
    # <https://docs.scipy.org/doc/scipy/reference/stats.html>

    # Generate random DAG
    dag = random_dag(n_features, p_edge)

    # Features
    feats = [{
        'observed': True,
        'type': Type.DISCRETE
            if random.random() < p_discrete
            else Type.CONTINUOUS
    } for _ in range(n_features)]

    # Random scaling for continuous variables
    for f in feats:
        if f['type'] is Type.CONTINUOUS:
            f['scale'] = np.random.randint(*scale_range)
        else:
            f['scale'] = 1.

    # Get roots
    roots = []
    for n in dag.nodes:
        if not nx.ancestors(dag, n):
            roots.append(n)

    # Initialize roots
    for root in roots:
        feat = feats[root]
        dist = random_dist(feat['type'], discrete_range=discrete_range)
        feat['dist'] = dist['name']
        feat['params'] = dist['params']

    # Get isolated variables
    isolated = []
    for i in range(len(feats)):
        if i not in dag.nodes:
            feat = feats[i]
            dist = random_dist(feat['type'], discrete_range=discrete_range)
            feat['dist'] = dist['name']
            feat['params'] = dist['params']
            isolated.append(i)

    # Get intermediary nodes,
    # nodes w/ both parents and children
    # These may or may not be unobserved
    for n in dag.nodes:
        parents = list(dag.predecessors(n))
        children = list(dag.successors(n))
        if len(parents) > 0 and len(children) > 0:
            if random.random() > p_observed:
                feats[n]['observed'] = False

    # Initialize other functions
    for n in dag.nodes():
        if n in roots: continue
        parents = list(dag.predecessors(n))
        fn, params = random_polynomial(len(parents))
        feats[n]['fn'] = fn
        feats[n]['params'] = params

    # Generate population
    samples = {}
    fringe = []
    for n in roots + isolated:
        feat = feats[n]
        dist = getattr(stats, feat['dist'])
        samples[n] = dist.rvs(**feat['params'], size=pop_size)
        if n in roots:
            fringe += dag.successors(n)

    while fringe:
        n = fringe.pop(0)

        if n in samples: continue
        parents = list(dag.predecessors(n))

        # Need to have samples for all parents
        if not all(pa in samples for pa in parents):
            fringe.append(n)
            continue

        vals = [samples[pa] for pa in parents]
        samples[n] = feats[n]['fn'](*vals)

        # If this is a discrete variable, discretize
        # There is probably a better way to do this
        if feats[n]['type'] is Type.DISCRETE:
            n_bins = np.random.randint(*discrete_range)
            _, bins = np.histogram(samples[n], bins=n_bins)
            samples[n] = np.digitize(samples[n], bins=bins)

        fringe += dag.successors(n)

    # Apply scaling
    for f in samples.keys():
        if feats[f]['type'] is Type.CONTINUOUS:
            samples[f] *= feats[f]['scale']

    # Generate complete population dataframe
    pop_df = pd.DataFrame(samples)

    # Hide unobserved variables
    for f in samples.keys():
        if not feats[f]['observed']:
            del samples[f]

    # This assumes a perfectly random
    # sampling procedure. Could introduce
    # bias into the process, e.g. a distribution
    # over costs of sampling a particular
    # sub-population so they are underrepresented
    n_drop = pop_size - sample_size
    sample_df = pd.DataFrame(samples)
    drop_idx = np.random.choice(sample_df.index, n_drop, replace=False)
    sample_df.drop(drop_idx, inplace=True)
    return sample_df, {
        'dag': dag,
        'features': feats,
        'roots': roots,
        'population': pop_df
    }


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    df, world = generate_exercise()
    print(df)
    print(world['features'])
    nx.draw(world['dag'])
    plt.show()