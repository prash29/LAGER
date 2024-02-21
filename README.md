# LAGER
### Layout Aware Graph-based Entity Recognition model

## Code will be updated soon!

Paper accepted to LREC-COLING 2024! 

Abstract:
Recent advances of incorporating layout information, typically bounding box coordinates, into pre-trained language models have achieved significant performance in entity recognition from document images. Using coordinates can easily model the absolute position of each token, but they might be sensitive to manipulations in document images (e.g., shifting, rotation or scaling), especially when the training data is limited in few-shot settings. In this paper, we propose to further introduce the topological adjacency relationship among the tokens, emphasizing their relative position information. Specifically, we consider the tokens in the documents as nodes and formulate the edges based on the topological heuristics from the k-nearest bounding boxes. Such adjacency graphs are invariant to affine transformations including shifting, rotations and scaling. We incorporate these graphs into the pre-trained language model by adding graph neural network layers on top of the language model embeddings, leading to a novel model LAGER. Extensive experiments on two benchmark datasets show that LAGER significantly outperforms strong baselines under different few-shot settings and also demonstrate better robustness to manipulations.


