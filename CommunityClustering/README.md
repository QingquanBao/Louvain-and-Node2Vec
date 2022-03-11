## Louvain Algorithm for modularity-based community clustering
Louvain algorithm[^1] is used for clustering community in the graph via maximizing the modularity. The original version of Louvain is tailored to undirected graph. Here for the sake of generalization, we implement a directed one.

Exactly speaking, we modify the gain of modularity in Phase 1 as follows,
$$
\Delta Q(i, c) = \frac{1}{m} \left(
                        d_{i,c} - \frac{1}{m}d_i^{in} \Sigma_{total}^{out} +
                        d_{c,i} - \frac{1}{m}d_i^{out} \Sigma_{total}^{in} 
\right)
$$
where $i$ denotes an arbitrary node and $c$ denotes a community neighboring the node $i$. $d_{i,c}$ involves all edges (with weight) from node $i$ to the community $c$ and $d_{i,c}$ contains the summation of weights from all nodes in $c$ to the node $i$. $d_i^{in}$ is the in-degree of node $i$ and $d_i^{out}$ is the out-degree fo node $i$. $\Sigma_{total}^{in}$ sums all in-degree of every node in the community $c$, $\Sigma_{total}^{out}$ simmilarly all out-degree.

### How to

The algo is implemented in the class `DirectedGraph()` in the file `algo/Louvain_model.py`. To use it, you can refer to the `main.py`.

1. build the graph, e.g. `G = DirectGraph(G)` where `G` is an instance of `networkx.DiGraph` just for efficiency. You can also modify the `__init__()` in in the class `DirectedGraph()` to tailor to your own data structure.
2. run the Louvain algo like `G.louvain(community_num=5, max_iter=30)`, here `community_num` is the number of community you want to cluster, and the `max_iter` is the max iterations of phase 1 in Louvain algorithm.

### What I learn from the implementation

- For the sake of the speed, you should save as many intermediate variables as possible towards graph data;
- `defaultdict` and `Counter` are good tools to use in python;
- When concerning updating parameters or variables in the class members, check **whether this update will influence other update**, or asking yourself, " Can I update directly or have to wait until the end of the loop ?"
- And check **whether one kind of variables will influence another kind of variables**.
- Write down all variables to be updated after a loop, before you start to code :) 

[^1]:  https://arxiv.org/pdf/0803.0476.pdf