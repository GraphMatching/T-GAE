# Network-Alignment-with-Transferable-Graph-Autoencoders

This repository contains code for the paper Network Alignment with Transferable Graph Autoencoders submitted at Neurips 2024

Abstract: 

Network alignment is the task of establishing one-to-one correspondences between
2 the nodes of different graphs and finds a plethora of applications in high-impact
3 domains. However, this task is known to be NP-hard in its general form, and
4 existing algorithms do not scale up as the size of the graphs increases. To tackle
5 both challenges we propose a novel generalized graph autoencoder architecture,
6 designed to extract powerful and robust node embeddings, that are tailored to
7 the alignment task. We prove that the generated embeddings are associated with
8 the eigenvalues and eigenvectors of the graphs and can achieve more accurate
9 alignment compared to classical spectral methods. Our proposed framework also
10 leverages transfer learning and data augmentation to achieve efficient network
11 alignment at a large scale without retraining. Extensive experiments on both
12 network and sub-network alignment with real-world graphs provide corroborating
13 evidence supporting the effectiveness and scalability of the proposed approach.
