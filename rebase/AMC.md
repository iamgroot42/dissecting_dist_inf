# Affinity Meta-Classifier

1. First, we start with some sample of datapoints (we call this “seed data”).
2. For a given model, we input a datapoint into the model and collect the features (outputs) after every layer. You can think of the model as a composition of functions `...c(b(a(x))))` for some input `x`, and what we essentially do is collect  `a(x)`, `b(a(x))`, `c(a(b(x)))`, ...
3. Compute these features for all of the data that we have and then one layer at a time, compute the cosine similarity between these activations for every `nC2` pair of datapoints. We compute this for all layers of the model. Thus, for every layer we can a list of `nC2` numbers (each is a cosine similarity).
4. Since `nC2` can be pretty large, we use a small model `\phi` that takes the `nC2` numbers as input and outputs a much smaller representation. This model is shared for all the layers.
5. Finally, we concatenate all the layer-wise representations, and then simply train our meta-classifier like we would train any model.
6. In essence, this technique helps track how the “relationship” between feature similarity changes across the layers of the model. Our hypothesis is that this information can help decode information about which of the two training distributions the model had.