# A-Survey-on-Mixture-of-Experts


## Algorithm
[[1]](#1). [[2]](#2). [[10]](#10). [[11]](#11). [[12]](#12). [[15]](#15). [[19]](#19). [[20]](#20). [[21]](#21). [[46]](#46). [[48]](#48). 


### Module

#### Experts

##### Dense
[[30]](#30). 

##### Sparse

##### Share
[[18]](#18). [[50]](#50). [[56]](#56). 

##### Placement

##### Activ. Func.


##### Networks
[[35]](#35). 


#### Gating Network

##### Dynamic
[[7]](#7). [[21]](#21). [[1]](#1). [[2]](#2). [[27]](#27).

##### Static
[[4]](#4). [[5]](#5). [[17]](#17). [[44]](#44). 



### Training Paradigm

#### Fully Synchronized

#### Asynchronous
[[13]](#13). [[49]](#49). [[54]](#54). 



### Derivation
[[53]](#53). [[34]](#34). [[26]](#26).




## System

[[6]](#6). [[9]](#9). [[14]](#14). [[18]](#18). [[23]](#23). [[24]](#24). [[25]](#25). [[33]](#33). [[36]](#36). [[38]](#38). [[39]](#39). [[41]](#41). [[47]](#47). 
### Computation


### Memory
[[32]](#32). [[43]](#43). 

### Communication 
[[22]](#22).




## PEFT
[[31]](#31). [[37]](#37). [[40]](#40). [[42]](#42). [[45]](#45). [[51]](#51). [[52]](#52). [[55]](#55). [[57]](#57). [[58]](#58). 

### Attention


### FFN


### Every SubLayer


### Transformer Block





## Application

### NLP
[[8]](#8). [[16]](#16). [[29]](#29). 

### CV
[[3]](#3). [[39]](#39). 

### RecSys


### MultiModal
[[28]](#28). [[35]](#35). 

## References


A Review of Sparse Expert Models in Deep Learning [[ArXiv 2022]](https://arxiv.org/abs/2209.01667) 2022-9-4

Taming Sparsely Activated Transformer with Stochastic Experts [[ICLR 2022]](https://arxiv.org/abs/2110.04260) 2021-10-8

Brainformers: Trading Simplicity for Efficiency [[ICML 2023]](https://arxiv.org/abs/2306.00008) 2023-5-29

Patch-level Routing in Mixture-of-Experts is Provably Sample-efficient for Convolutional Neural Networks [[ICML 2023]](https://arxiv.org/abs/2306.04073) 2023-6-7

Robust Mixture-of-Expert Training for Convolutional Neural Networks [[ICCV 2023]](https://arxiv.org/abs/2308.10110v1) 2023-8-19

Merging Experts into One: Improving Computational Efficiency of Mixture of Experts [[EMNLP 2023]](https://arxiv.org/abs/2310.09832) 2023-10-15

Dense Training, Sparse Inference: Rethinking Training of Mixture-of-Experts Language Models [[ArXiv 2024]](https://arxiv.org/abs/2404.05567) 2024-4-8

Merging Experts into One: Improving Computational Efficiency of Mixture of Experts [[EMNLP 2022]](https://arxiv.org/abs/2210.05144) 2022-10-11

JetMoE: Reaching Llama2 Performance with 0.1M Dollars [[ArXiv 2024]](https://arxiv.org/abs/2404.07413) 2024-4-11

Residual Mixture of Experts [[ArXiv 2022]](https://arxiv.org/abs/2204.09636) 2022-4-20

<a id="1">[1]</a> OUTRAGEOUSLY LARGE NEURAL NETWORKS: THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER	[[ICLR 2017]](https://arxiv.org/abs/1701.06538) 2017-1-23

<a id="2">[2]</a> GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding	[[ICLR 2021]](https://arxiv.org/abs/2006.16668) 2020-6-30

<a id="3">[3]</a> Scaling Vision with Sparse Mixture of Experts	[[NIPS 2021]](https://arxiv.org/abs/2106.05974) 2021-6-10

<a id="4">[4]</a> Hash Layers For Large Sparse Models	[[NIPS 2021]](https://arxiv.org/abs/2106.04426) 2021-6-8

<a id="5">[5]</a> BASE Layers: Simplifying Training of Large, Sparse Models	[[ICML 2021]](https://arxiv.org/abs/2103.16716) 2021-5-30

<a id="6">[6]</a> FASTMOE: A FAST MIXTURE-OF-EXPERT TRAINING SYSTEM	[[ArXiv 2021]](https://arxiv.org/abs/2103.13262) 2021-5-21

<a id="7">[7]</a> EvoMoE: An Evolutional Mixture-of-Experts Training Framework via Dense-To-Sparse Gate	[[ArXiv 2021]](https://arxiv.org/abs/2112.14397) 2021-12-29

<a id="8">[8]</a> GLaM: Efficient Scaling of Language Models with Mixture-of-Experts	[[ICML 2022]](https://arxiv.org/abs/2112.06905) 2021-12-13

<a id="9">[9]</a> FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models	[[PPoPP 2022]](https://dl.acm.org/doi/10.1145/3503221.3508418) 2022-3-28

<a id="10">[10]</a> On the Representation Collapse of Sparse Mixture of Experts	[[NIPS 2022]](https://arxiv.org/abs/2204.09179) 2022-4-20

<a id="11">[11]</a> Task-Specific Expert Pruning for Sparse Mixture-of-Experts	[[ArXiv 2022]](https://arxiv.org/abs/2206.00277) 2022-6-1

<a id="12">[12]</a> A Theoretical View on Sparsely Activated Networks	[[NIPS 2022]](https://arxiv.org/abs/2208.04461) 2022-8-8

<a id="13">[13]</a> Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models	[[NIPS 2022]](https://arxiv.org/abs/2208.03306) 2022-8-5

<a id="14">[14]</a> BaGuaLu: Targeting Brain Scale Pretrained Models with over 37 Million Cores	[[PPoPP 2022]](https://dl.acm.org/doi/10.1145/3503221.3508417) 2022-3-28

<a id="15">[15]</a> Go Wider Instead of Deeper	[[AAAI 2022]](https://arxiv.org/abs/2107.11817) 2021-7-25

<a id="16">[16]</a> Efficient Large Scale Language Modeling with Mixtures of Experts	[[EMNLP 2022]](https://arxiv.org/abs/2112.10684) 2021-12-20

<a id="17">[17]</a> STABLEMOE: Stable Routing Strategy for Mixture of Experts	[[ACL 2022]](https://arxiv.org/abs/2204.08396) 2022-4-18

<a id="18">[18]</a> DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale	[[ICML 2022]](https://arxiv.org/abs/2201.05596) 2022-1-14

<a id="19">[19]</a> UNIFIED SCALING LAWS FOR ROUTED LANGUAGE MODELS	[[ICML 2022]](https://arxiv.org/abs/2202.01169) 2022-2-2

<a id="20">[20]</a> Uni-Perceiver-MoE: Learning Sparse Generalist Models with Conditional MoEs	[[NIPS 2022]](https://arxiv.org/abs/2206.04674) 2022-6-9

<a id="21">[21]</a> Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity	[[ArXiv 2022]](https://arxiv.org/abs/2101.03961) 2021-1-11

<a id="22">[22]</a> TA-MoE: Topology-Aware Large Scale Mixture-of-Expert Training	[[NIPS 2022]](https://arxiv.org/abs/2302.09915) 2023-2-20

<a id="23">[23]</a> Hetu: a highly efficient automatic parallel distributed deep learning system	[[Sci. China Inf. Sci. 2023]](https://link.springer.com/article/10.1007/s11432-022-3581-9) 2022-12

<a id="24">[24]</a> HetuMoE: An Efficient Trillion-scale Mixture-of-Expert Distributed Training System	[[ArXiv 2022]](https://arxiv.org/abs/2203.14685) 2022-3-28

<a id="25">[25]</a> MEGABLOCKS: EFFICIENT SPARSE TRAINING WITH MIXTURE-OF-EXPERTS	[[MLSys 2023]](https://arxiv.org/abs/2211.15841) 2022-11-29

<a id="26">[26]</a> Mixture-of-Experts with Expert Choice Routing	[[NIPS 2022]](https://arxiv.org/abs/2202.09368) 2022-2-18

<a id="27">[27]</a> ST-MOE: DESIGNING STABLE AND TRANSFERABLE SPARSE EXPERT MODELS	[[ArXiv 2022]](https://arxiv.org/abs/2202.08906) 2022-2-17

<a id="28">[28]</a> Multimodal Contrastive Learning with LIMoE: the Language-Image Mixture of Experts	[[NIPS 2022]](https://arxiv.org/abs/2206.02770) 2022-6-6

<a id="29">[29]</a> No Language Left Behind: Scaling Human-Centered Machine Translation	[ArXiv 2022]

<a id="30">[30]</a> Parameter-Efficient Mixture-of-Experts Architecture for Pre-trained Language Models	[ArXiv 2022]

<a id="31">[31]</a> Omni-SMoLA: Boosting Generalist Multimodal Models with Soft Mixture of Low-rank Experts	[ArXiv 2023]

<a id="32">[32]</a> Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference	[ArXiv 2023]

<a id="33">[33]</a> FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via Dynamic Device Placement	[2023]

<a id="34">[34]</a> From Sparse to Soft Mixtures of Experts	[ArXiv 2023]

<a id="35">[35]</a> Scaling Vision-Language Models with Sparse Mixture of Experts	[EMNLP 2023]

<a id="36">[36]</a> SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization	[ATC 2023]

<a id="37">[37]</a> Mixture-of-Experts Meets Instruction Tuning: A Winning Combination for Large Language Models	[ArXiv 2023]

<a id="38">[38]</a> MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism	[IPDPS 2023]

<a id="39">[39]</a> TUTEL: ADAPTIVE MIXTURE-OF-EXPERTS AT SCALE	[MLSys 2023]

<a id="40">[40]</a> Mixture-of-Domain-Adapters: Decoupling and Injecting Domain Knowledge to Pre-trained Language Models’ Memories	[ACL 2023]

<a id="41">[41]</a> A Hybrid Tensor-Expert-Data Parallelism Approach to Optimize Mixture-of-Experts Training	[ICS 2023]

<a id="42">[42]</a> Mixture of Cluster-conditional LoRA Experts for Vision-language Instruction Tuning	[ArXiv 2023]

<a id="43">[43]</a> EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models	[ArXiv 2023]

<a id="44">[44]</a> PANGU-Σ: TOWARDS TRILLION PARAMETER LANGUAGE MODEL WITH SPARSE HETEROGENEOUS COMPUTING	[ArXiv 2023]

<a id="45">[45]</a> Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE for Instruction Tuning	[ArXiv 2023]

<a id="46">[46]</a> SPARSE MOE AS THE NEW DROPOUT: SCALING DENSE AND SELF-SLIMMABLE TRANSFORMERS	[ArXiv 2023]

<a id="47">[47]</a> Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference	[ArXiv 2024]

<a id="48">[48]</a> FUSING MODELS WITH COMPLEMENTARY EXPERTISE	[ICLR 2024]

<a id="49">[49]</a> Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM	[ArXiv 2024]

<a id="50">[50]</a> DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models	[ArXiv 2024]

<a id="51">[51]</a> Higher Layers Need More LoRA Experts	[ArXiv 2024]

<a id="52">[52]</a> LoRAMoE: Alleviate World Knowledge Forgetting in Large Language Models via MoE-Style Plugin	[ArXiv 2024]

<a id="53">[53]</a> Mixture-of-Depths: Dynamically allocating compute in transformer-based language models	[ArXiv 2024]

<a id="54">[54]</a> MoE-LLaVA: Mixture of Experts for Large Vision-Language Models	[ArXiv 2024]

<a id="55">[55]</a> MTLoRA: A Low-Rank Adaptation Approach for Efficient Multi-Task Learning	[ArXiv 2024]

<a id="56">[56]</a> OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models	[ArXiv 2024]

<a id="57">[57]</a> LLaVA-MoLE: Sparse Mixture of LoRA Experts for Mitigating Data Conflicts in Instruction Finetuning MLLMs	[ArXiv 2024]

<a id="58">[58]</a> MOLE: MIXTURE OF LORA EXPERTS	[ICLR 2024]

<a id="59">[59]</a> 





