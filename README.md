# Network-Alignment-with-Transferable-Graph-Autoencoders

This repository contains code for the paper Network Alignment with Transferable Graph Autoencoders.

Abstract: 

Network alignment is the task of establishing one-to-one correspondences between the nodes of different graphs and finds a plethora of applications in high-impact domains. However, this task is known to be NP-hard in its general form, and existing algorithms do not scale up as the size of the graphs increases. To tackle both challenges we propose a novel generalized graph autoencoder architecture, designed to extract powerful and robust node embeddings, that are tailored to the alignment task. We prove that the generated embeddings are associated with the eigenvalues and eigenvectors of the graphs and can achieve more accurate alignment compared to classical spectral methods. Our proposed framework also leverages transfer learning and data augmentation to achieve efficient network alignment at a large scale without retraining. Extensive experiments on both network and sub-network alignment with real-world graphs provide corroborating evidence supporting the effectiveness and scalability of the proposed approach.

## Code overview

This repository contains the source code that evaluates the performance of Moment-GNN in:

  - Graph Matching
  - Sub-graph Matching

## Dependencies

To run this code please install Pytorch, numpy, munkres, scipy, netrd, networkx.

We have used torch==2.0.1, numpy==1.23.1, munkres==1.1.4, scipy==1.12.0, networkx==3.2.1

## Data

All data used are inclueded in the data.zip file, please run the following command to unzip the data:

```
unzip data.zip
```

## Graph Matching
To run the experiments for graph matching in Section 5.3 please run:
```
python graphMatching.py
```

To run on specific dataset, perturbation model and level, please run:
```
python graphMatching.py --dataset celegans --model uniform --level 0.01
```
where dataset is one of {celegans, arenas, douban, cora, dblp, coauthor_cs},
perturbation model is one of {uniform, degree},
level is one of {0, 0.01, 0.05}

## Sub-graph Matching
To run the experiments for sub graph matching in Section 5.4, please run:
```
python subgraphMatching.py
```

To select dataset, please run:
```
python subgraphMatching.py --dataset ACM_DBLP
```
where ACM_DBLP is one of {ACM_DBLP, Douban Online_Offline}

## License
MIT

