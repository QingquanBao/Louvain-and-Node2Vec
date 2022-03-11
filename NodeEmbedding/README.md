## Link prediction with Node2vec

### Demo

To run the demo code, 

```bash
python src/main.py
```

Use the pretrained model, modify the `src/main.py` line 223 and 224. 

### Node2vec and Deepwalk

Node2vec is implemented in `src/embedding.py`. To use it, you can instantialize the class `Node2VecEmbedding(node_size, embedding_size, graph, p, q, sample_len, k)`,  where `node_size` denotes all available nodes in the graph, `embedding_size` is the hidden dimension of the embedding, `graph` is the undirected graph of `networkx`, `p` and `q` corespond to the parameters of Node2vec, `sample_len` is the length of the random walk, `k` is the size of negative samples for each loss.

To train it, use function `emb.train(batch, optimizer, scheduler)`, here `emb` is the instance of the class`Node2VecEmbedding` 