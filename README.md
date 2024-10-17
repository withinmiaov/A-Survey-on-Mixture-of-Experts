# A-Survey-on-Mixture-of-Experts

## Links to the Survey Paper
ArXiv: https://arxiv.org/abs/2407.06204

TechRxiv: https://www.techrxiv.org/users/799279/articles/1165210-a-survey-on-mixture-of-experts

## Timeline of Mixture of Experts (MoE)

![image](https://github.com/withinmiaov/A-Survey-on-Mixture-of-Experts/blob/main/MoE_Timeline.png)


## Taxonomy of Mixture of Experts (MoE)

![image](https://github.com/withinmiaov/A-Survey-on-Mixture-of-Experts/blob/main/MoE_Taxonomy.png)



_**-Mixture of Experts (MoE)**_

_**--Algorithm**_

_**---Gating Function**_

_**----Dense**_ [EvoMoE[[22]](#22). LoRAMoE[[88]](#88). MoLE[[92]](#92). DS-MoE[[106]](#106).]

_**----Sparse**_

_**-----Token-Choice**_ [Shazeer et al.[[1]](#1). MMoE[[2]](#2). GShard[[3]](#3). Switch Transformer[[5]](#5). Base Layers[[8]](#8). M6-t[[9]](#9). V-MoE[[11]](#11). Z-code M3[[14]](#14). Sentence-level MoE[[15]](#15). DSelect-k[[19]](#19). S-Base[[25]](#25). ST-MoE[[26]](#26). StableMoE[[33]](#33). X-MoE[[35]](#35). Task-level MoE[[38]](#38). Uni-Perceiver-MoE[[42]](#42). NLLB[[43]](#43). MoA[[49]](#49). Mod-Squad[[53]](#53). ModuleFormer[[69]](#69). Mixtral-8x7B[[82]](#82). DeepSeekMoE[[91]](#91). OpenMoE[[95]](#95). Jamba[[102]](#102). DS-MoE[[106]](#106). JetMoE[[107]](#107). Yuan 2.0-M32[[122]](#122). Skywork-MoE[[123]](#123). DBRX. SCMoE[[119]](#119). DYNMoE[[120]](#120). Flextron[[125]](#125).MoH[[127].](#127)

_**-----Non-trainable Token-Choice**_ [M2M-100[[4]](#4). Hash Layer[[10]](#10). DEMix[[13]](#13). Task-MoE[[15]](#15). THOR[[17]](#17). Pangu-∑[[59]](#59).]

_**-----Expert-Choice**_ [Expert-Choice MoE[[27]](#27). Brainformers[[66]](#66).]

_**----Soft**_

_**-----Token Merging**_ [Soft MoE[[72]](#72). HOMOE[[85]](#85).]

_**-----Expert Merging**_ [SMEAR[[67]](#67). MoV[[77]](#77). Omni-SMoLA[[86]](#86). Lory[[114]](#114). MEO[[81]](#81).]

_**---Experts**_

_**----Network Types**_

_**-----FFN**_ [GShard[[3]](#3). Switch Transformer[[5]](#5). MoEfication[[16]](#16). ST-MoE[[26]](#26). Branch-Train-MiX[[100]](#100). DS-MoE[[106]](#106).]

_**-----Attention**_ [MoA[[49]](#49). ModuleFormer[[69]](#69). DS-MoE[[106]](#106). JetMoE[[107]](#107).]

_**-----Others**_ [Chen et al.[[44]](#44). pMoE[[68]](#68). ADVMOE[[74]](#74).]

_**----Hyperparameters**_

_**-----Count**_ [GShard[[3]](#3). Swith Transformer[[5]](#5). GLaM[[20]](#20). Meta-MoE[[21]](#21). DeepSpeed-MoE[[23]](#23). ST-MoE[[26]](#26). Mixtral-8x7B[[87]](#87).]

_**-----Size**_ [GLaM[[20]](#20). Mixtral-8x7B[[87]](#87). LLAMA-MoE[[85]](#85). DeepSeekMoE[[91]](#91). DeepSeek-V2[[115]](#115). PEER[[126]](#126). DBRX. Qwen1.5-MoE.]

_**-----Frequency**_ [V-MoE[[11]](#11). ST-MoE[[26]](#26). Brainformers[[66]](#66). DeepSeekMoE[[91]](#91). OpenMoE[[95]](#95). MoE-LLaVA[[96]](#96). Jamba[[102]](#102).]
            
_**----Activation Function**_

_**----Shared Expert**_ [DeepSpeed-MoE[[23]](#23). NLLB[[43]](#43). MoCLE[[89]](#89). DeepSeekMoE[[91]](#91). OpenMoE[[95]](#95). ScMoE[[105]](#105). PAD-Net[[50]](#50). HyperMoE[[99]](#99).]

_**----PEFT**_

_**-----FFN**_ [AdaMix[[37]](#37). MixDA[[70]](#70). LoRAMoE[[88]](#88). LLaVA-MoLE[[89]](#89). MixLoRA[[109]](#109). EM[[65]](#65).]

_**-----Attention**_ [SiRA[[84]](#84). MoCLE[[89]](#89). MoELoRA.]

_**-----Transformer Block**_ [MoV[[77]](#77). MoLoRA[[77]](#77). Omni-SMoLA[[86]](#86). MoLA[[98]](#98). Intuition-MoR1E[[108]](#108). MeteoRA[[118]](#118). UniPELT. MOELoRA.]

_**-----Every Layer**_ [MoLE[[92]](#92).]

_**---Training & Inference Scheme**_

_**----Original**_ [Shazeer et al.[[1]](#1). GShard[[3]](#3). Switch Transformer[[5]](#5). ST-MoE[[26]](#26).]

_**----Dense2Sparse**_ [MoEfication[[16]](#16). Dua et al.[[18]](#18). EvoMoE[[22]](#22). MoEBERT[[32]](#32). RMoE[[34]](#34). SMoE-Dropout[[56]](#56). LLaMA-MoE[[90]](#90). MoE-LLaVA[[96]](#96). DS-MoE[[106]](#106). Skywork-MoE[[123]](#123). Sparse Upcycling. EMoE[[82]](#82).]
            
_**----Sparse2Dense**_ [OneS[[24]](#24). MoE-Pruning[[39]](#39). ModuleFormer[[65]](#65). EWA[[73]](#73). He et al.[[124]](#124).]

_**----Expert Models Merging**_ [Branch-Train-Merge[[45]](#45). FoE[[78]](#78). Branch-Train-MiX[[100]](#100).]

_**---Derivatives**_ [WideNet[[12]](#12). SMoP[[79]](#79). SUT[[80]](#80). MoD[[104]](#104). Lifelong-MoE. MoT.]

_**--System**_

_**---Computation**_ [FastMoE[[7]](#7). DeepSpeed-MoE[[23]](#23). HetuMoE[[29]](#29). FasterMoE[[30]](#30). Tutel[[41]](#41). SE-MoE[[36]](#36). MegaBlocks[[51]](#51). PIT[[54]](#54). FlexMoE[[60]](#60). SmartMoE[[71]](#71). ScatterMoE[[101]](#101). DeepSpeed-TED[[57]](#57).]

_**---Communication**_ [DeepSpeed-MoE[[23]](#23). HetuMoE[[29]](#29). FasterMoE[[30]](#30). SE-MoE[[36]](#36). Tutel[[41]](#41). TA-MoE[[55]](#55). MPipeMoE[[61]](#61). ExFlow[[93]](#93). ScMoE[[105]](#105). Lancet[[113]](#113). SkyWork-MoE[[123]](#123). DeepSpeed-TED[[57]](#57). PipeMoE[[62]](#62). ScheMoE[[110]](#110). Punniyamurthy et al.[[116]](#116).]

_**---Storage**_ [SE-MoE[[36]](#36). MPipeMoE[[61]](#61). Pre-gated MoE[[75]](#75). EdgeMoE[[76]](#76).]

_**--Application**_

_**---NLP**_ [Shazeer et al.[[1]](#1). GShard[[3]](#3). Swith Transformer[[5]](#5). GLaM[[20]](#20). Meta-MoE[[21]](#21). DeepSpeed-MoE[[23]](#23). ST-MoE[[26]](#26). NLLB[[43]](#43). Mixtral-8x7B[[87]](#87). DeepSeekMoE[[91]](#91). MoGU[[121]](#121).]

_**---CV**_ [V-MoE[[11]](#11). Swin-MoE[[41]](#41). pMoE[[64]](#64). ADVMOE[[70]](#70).]

_**---RecSys**_ [MMoE[[2]](#2). M3oE[[112]](#112) PLE. AdaMCT.]

_**---MultiModal**_ [LIMoE[[40]](#40). Shen et al.[[58]](#58). MoCLE[[89]](#89). LLaVA-MoLE[[94]](#94). MoE-LLaVA[[96]](#96). Uni-MoE[[117]](#117). MM1. PaCE[[64]](#64).]



## References (arranged in order of time)

<a id="127">[127]</a> MoH: Multi-Head Attention as Mixture-of-Head Attention [[ArXiv 2024]](https://arxiv.org/abs/2410.11842v1) 2024-10-15

<a id="126">[126]</a> Mixture of A Million Experts [[ArXiv 2024]](https://arxiv.org/abs/2407.04153) 2024-7-4

<a id="125">[125]</a> Flextron: Many-in-One Flexible Large Language Model [[ICML 2024]](https://arxiv.org/abs/2406.10260) 2024-6-11

<a id="124">[124]</a> Demystifying the Compression of Mixture-of-Experts Through a Unified Framework [[ArXiv 2024]](https://arxiv.org/abs/2406.02500) 2024-6-4

<a id="123">[123]</a> Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models [[ArXiv 2024]](https://github.com/SkyworkAI/Skywork-MoE) 2024-6-3

<a id="122">[122]</a> Yuan 2.0-M32: Mixture of Experts with Attention Router [[ArXiv 2024]](https://arxiv.org/abs/2405.17976) 2024-5-28

<a id="121">[121]</a> MoGU: A Framework for Enhancing Safety of Open-Sourced LLMs While Preserving Their Usability [[ArXiv 2024]](https://arxiv.org/abs/2405.14488) 2024-5-23

<a id="120">[120]</a> Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient Transformer Models [[ArXiv 2024]](https://arxiv.org/abs/2405.14297) 2024-5-23

<a id="119">[119]</a> Unchosen Experts Can Contribute Too: Unleashing MoE Models' Power by Self-Contrast [[ArXiv 2024]](https://arxiv.org/abs/2405.14507) 2024-5-23

<a id="118">[118]</a> MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models [[ArXiv 2024]](https://arxiv.org/abs/2405.13053) 2024-5-19

<a id="117">[117]</a> Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts [[ArXiv 2024]](https://arxiv.org/abs/2405.11273) 2024-5-18

<a id="116">[116]</a> Optimizing Distributed ML Communication with Fused Computation-Collective Operations [[ArXiv 2023]](https://arxiv.org/abs/2305.06942) 2023-5-11

<a id="115">[115]</a> DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model [[ArXiv 2024]](https://arxiv.org/abs/2405.04434) 2024-5-7

<a id="114">[114]</a> Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training [[ArXiv 2024]](https://arxiv.org/abs/2405.03133) 2024-5-6

<a id="113">[113]</a> Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping [[ArXiv 2024]](https://arxiv.org/abs/2404.19429) 2024-4-30

<a id="112">[112]</a> M3oE: Multi-Domain Multi-Task Mixture-of Experts Recommendation Framework [[SIGIR 2024]](https://arxiv.org/abs/2404.18465) 2024-4-29

<a id="111">[111]</a> Multi-Head Mixture-of-Experts [[ArXiv 2024]](https://arxiv.org/abs/2404.15045) 2024-4-23

<a id="110">[110]</a> ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling [[EuroSys 2024]](https://dl.acm.org/doi/10.1145/3627703.3650083) 2024-4-22

<a id="109">[109]</a> MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts [[ArXiv 2024]](https://arxiv.org/abs/2404.15159) 2024-4-22

<a id="108">[108]</a> Intuition-aware Mixture-of-Rank-1-Experts for Parameter Efficient Finetuning [[ArXiv 2024]](https://arxiv.org/abs/2404.08985) 2024-4-13

<a id="107">[107]</a> JetMoE: Reaching Llama2 Performance with 0.1M Dollars [[ArXiv 2024]](https://arxiv.org/abs/2404.07413) 2024-4-11

<a id="106">[106]</a> Dense Training, Sparse Inference: Rethinking Training of Mixture-of-Experts Language Models [[ArXiv 2024]](https://arxiv.org/abs/2404.05567) 2024-4-8

<a id="105">[105]</a> Shortcut-connected Expert Parallelism for Accelerating Mixture-of-Experts [[ArXiv 2024]](https://arxiv.org/abs/2404.05019) 2024-4-7

<a id="104">[104]</a> Mixture-of-Depths: Dynamically allocating compute in transformer-based language models	[[ArXiv 2024]](https://arxiv.org/abs/2404.02258) 2024-4-2

<a id="103">[103]</a> MTLoRA: A Low-Rank Adaptation Approach for Efficient Multi-Task Learning	[[ArXiv 2024]](https://arxiv.org/abs/2403.20320) 2024-3-29

<a id="102">[102]</a> Jamba: A Hybrid Transformer-Mamba Language Model [[ArXiv 2024]](https://arxiv.org/abs/2403.19887) 2024-3-28

<a id="101">[101]</a> Scattered Mixture-of-Experts Implementation [[ArXiv 2024]](https://arxiv.org/abs/2403.08245) 2024-3-13

<a id="100">[100]</a> Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM	[[ArXiv 2024]](https://arxiv.org/abs/2403.07816) 2024-3-12

<a id="99">[99]</a> HyperMoE: Towards Better Mixture of Experts via Transferring Among Experts [[ACL 2024]](https://arxiv.org/abs/2402.12656) 2024-2-20

<a id="98">[98]</a> Higher Layers Need More LoRA Experts	[[ArXiv 2024]](https://arxiv.org/abs/2402.08562) 2024-2-13

<a id="97">[97]</a> FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion [[ArXiv 2024]](https://arxiv.org/abs/2402.03226) 2024-2-5

<a id="96">[96]</a> MoE-LLaVA: Mixture of Experts for Large Vision-Language Models	[[ArXiv 2024]](https://arxiv.org/abs/2401.15947) 2024-1-29

<a id="95">[95]</a> OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models	[[ArXiv 2024]](https://arxiv.org/abs/2402.01739) 2024-1-29

<a id="94">[94]</a> LLaVA-MoLE: Sparse Mixture of LoRA Experts for Mitigating Data Conflicts in Instruction Finetuning MLLMs	[[ArXiv 2024]](https://arxiv.org/abs/2401.16160) 2024-1-29

<a id="93">[93]</a> Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference	[[ArXiv 2024]](https://arxiv.org/abs/2401.08383) 2024-1-16

<a id="92">[92]</a> MOLE: MIXTURE OF LORA EXPERTS	[[ICLR 2024]](https://openreview.net/forum?id=uWvKBCYh4S) 2024-1-16

<a id="91">[91]</a> DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models	[[ArXiv 2024]](https://arxiv.org/abs/2401.06066) 2024-1-11

<a id="90">[90]</a> LLaMA-MoE: Building Mixture-of-Experts from LLaMA with Continual Pre-training [[Github 2023]](https://github.com/pjlab-sys4nlp/llama-moe/blob/main/docs/LLaMA_MoE.pdf) 2023-12

<a id="89">[89]</a> Mixture of Cluster-conditional LoRA Experts for Vision-language Instruction Tuning	[[ArXiv 2023]](https://arxiv.org/abs/2312.12379) 2023-12-19

<a id="88">[88]</a> LoRAMoE: Alleviate World Knowledge Forgetting in Large Language Models via MoE-Style Plugin	[[ArXiv 2023]](https://arxiv.org/abs/2312.09979) 2023-12-15

<a id="87">[87]</a> Mixtral of Experts [[ArXiv 2024]](https://arxiv.org/abs/2401.04088) 2023-12-11

<a id="86">[86]</a> Omni-SMoLA: Boosting Generalist Multimodal Models with Soft Mixture of Low-rank Experts	[[ArXiv 2023]](https://arxiv.org/abs/2312.00968) 2023-12-1

<a id="85">[85]</a> HOMOE: A Memory-Based and Composition-Aware Framework for Zero-Shot Learning with Hopfield Network and Soft Mixture of Experts [[ArXiv 2023]](https://arxiv.org/abs/2311.14747) 2023-11-23

<a id="84">[84]</a> Sira: Sparse mixture of low rank adaptation [[ArXiv 2023]](https://arxiv.org/abs/2311.09179) 2023-11-15

<a id="83">[83]</a> When MOE Meets LLMs: Parameter Efficient Fine-tuning for Multi-task Medical Applications [[SIGIR 2024]](https://arxiv.org/abs/2310.18339) 2023-10-21

<a id="82">[82]</a> Unlocking Emergent Modularity in Large Language Models [[NAACL 2024]](https://arxiv.org/abs/2310.10908) 2023-10-17

<a id="81">[81]</a> Merging Experts into One: Improving Computational Efficiency of Mixture of Experts [[EMNLP 2023]](https://arxiv.org/abs/2310.09832) 2023-10-15

<a id="80">[80]</a> Sparse Universal Transformer [[EMNLP 2023]](https://arxiv.org/abs/2310.07096) 2023-10-11

<a id="79">[79]</a> SMoP: Towards Efficient and Effective Prompt Tuning with Sparse Mixture-of-Prompts [[EMNLP 2023]](https://openreview.net/forum?id=5x5Vxclc1K) 2023-10-8

<a id="78">[78]</a> FUSING MODELS WITH COMPLEMENTARY EXPERTISE	[[ICLR 2024]](https://arxiv.org/abs/2310.01542) 2023-10-2

<a id="77">[77]</a> Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE for Instruction Tuning	[[ICLR 2024]](https://arxiv.org/abs/2309.05444) 2023-9-11

<a id="76">[76]</a> EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models	[[ArXiv 2023]](https://arxiv.org/abs/2308.14352) 2023-8-28

<a id="75">[75]</a> Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference	[[ArXiv 2023]](https://arxiv.org/abs/2308.12066) 2023-8-23

<a id="74">[74]</a> Robust Mixture-of-Expert Training for Convolutional Neural Networks [[ICCV 2023]](https://arxiv.org/abs/2308.10110v1) 2023-8-19

<a id="73">[73]</a> Experts Weights Averaging: A New General Training Scheme for Vision Transformers [[ArXiv 2023]](https://arxiv.org/abs/2308.06093) 2023-8-11

<a id="72">[72]</a> From Sparse to Soft Mixtures of Experts	[ICLR 2024](https://arxiv.org/abs/2308.00951) 2023-8-2

<a id="71">[71]</a> SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization	[[USENIX ATC 2023]](https://www.usenix.org/conference/atc23/presentation/zhai) 2023-7-10

<a id="70">[70]</a> Mixture-of-Domain-Adapters: Decoupling and Injecting Domain Knowledge to Pre-trained Language Models’ Memories	[[ACL 2023]](https://arxiv.org/abs/2306.05406) 2023-6-8

<a id="69">[69]</a> Moduleformer: Learning modular large language models from uncurated data [[ArXiv 2023]](https://arxiv.org/abs/2306.04640) 2023-6-7

<a id="68">[68]</a> Patch-level Routing in Mixture-of-Experts is Provably Sample-efficient for Convolutional Neural Networks [[ICML 2023]](https://arxiv.org/abs/2306.04073) 2023-6-7

<a id="67">[67]</a> Soft Merging of Experts with Adaptive Routing [[TMLR 2024]](https://arxiv.org/abs/2306.03745) 2023-6-6

<a id="66">[66]</a> Brainformers: Trading Simplicity for Efficiency [[ICML 2023]](https://arxiv.org/abs/2306.00008) 2023-5-29

<a id="65">[65]</a>Emergent Modularity in Pre-trained Transformers [[ACL 2023]](https://arxiv.org/abs/2305.18390) 2023-5-28

<a id="64">[64]</a> PaCE: Unified Multi-modal Dialogue Pre-training with Progressive and Compositional Experts [[ACL 2023]](https://arxiv.org/abs/2305.14839) 2023-5-24

<a id="63">[63]</a> Mixture-of-Experts Meets Instruction Tuning: A Winning Combination for Large Language Models	[[ICLR 2024]](https://arxiv.org/abs/2305.14705) 2023-5-24

<a id="62">[62]</a> PipeMoE: Accelerating Mixture-of-Experts through Adaptive Pipelining [[INFOCOM 2023]](https://ieeexplore.ieee.org/document/10228874) 2023-5-17

<a id="61">[61]</a> MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism	[[IPDPS 2023]](https://ieeexplore.ieee.org/document/10177396) 2023-5-15

<a id="60">[60]</a> FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via Dynamic Device Placement	[[Proc. ACM Manag. Data 2023]](https://arxiv.org/abs/2304.03946) 2023-4-8

<a id="59">[59]</a> PANGU-Σ: TOWARDS TRILLION PARAMETER LANGUAGE MODEL WITH SPARSE HETEROGENEOUS COMPUTING	[[ArXiv 2023]](https://arxiv.org/abs/2303.10845) 2023-3-20

<a id="58">[58]</a> Scaling Vision-Language Models with Sparse Mixture of Experts	[EMNLP (Findings) 2023](https://arxiv.org/abs/2303.07226) 2023-3-13

<a id="57">[57]</a> A Hybrid Tensor-Expert-Data Parallelism Approach to Optimize Mixture-of-Experts Training	[[ICS 2023]](https://arxiv.org/abs/2303.06318) 2023-3-11

<a id="56">[56]</a> SPARSE MOE AS THE NEW DROPOUT: SCALING DENSE AND SELF-SLIMMABLE TRANSFORMERS	[[ICLR 2023]](https://arxiv.org/abs/2303.01610) 2023-3-2

<a id="55">[55]</a> TA-MoE: Topology-Aware Large Scale Mixture-of-Expert Training	[[NIPS 2022]](https://arxiv.org/abs/2302.09915) 2023-2-20

<a id="54">[54]</a> PIT: Optimization of Dynamic Sparse Deep Learning Models via Permutation Invariant Transformation [[SOSP 2023]](https://arxiv.org/abs/2301.10936) 2023-1-26

<a id="53">[53]</a> Mod-Squad: Designing Mixture of Experts As Modular Multi-Task Learners [[CVPR 2023]](https://arxiv.org/abs/2212.08066) 2022-12-15

<a id="52">[52]</a> Hetu: a highly efficient automatic parallel distributed deep learning system	[[Sci. China Inf. Sci. 2023]](https://link.springer.com/article/10.1007/s11432-022-3581-9) 2022-12

<a id="51">[51]</a> MEGABLOCKS: EFFICIENT SPARSE TRAINING WITH MIXTURE-OF-EXPERTS	[[MLSys 2023]](https://arxiv.org/abs/2211.15841) 2022-11-29

<a id="50">[50]</a> PAD-Net: An Efficient Framework for Dynamic Networks [[ACL 2023]](https://arxiv.org/abs/2211.05528) 2022-11-10

<a id="49">[49]</a> Mixture of Attention Heads: Selecting Attention Heads Per Token [[EMNLP 2022]](https://arxiv.org/abs/2210.05144) 2022-10-11

<a id="48">[48]</a> Sparsity-Constrained Optimal Transport [[ICLR 2023]](https://arxiv.org/abs/2209.15466) 2022-9-30

<a id="47">[47]</a> A Review of Sparse Expert Models in Deep Learning [[ArXiv 2022]](https://arxiv.org/abs/2209.01667) 2022-9-4

<a id="46">[46]</a> A Theoretical View on Sparsely Activated Networks	[[NIPS 2022]](https://arxiv.org/abs/2208.04461) 2022-8-8

<a id="45">[45]</a> Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models	[[NIPS 2022]](https://arxiv.org/abs/2208.03306) 2022-8-5

<a id="44">[44]</a> Towards Understanding Mixture of Experts in Deep Learning [[ArXiv 2022]](https://arxiv.org/abs/2208.02813) 2022-8-4

<a id="43">[43]</a> No Language Left Behind: Scaling Human-Centered Machine Translation	[[ArXiv 2022]](https://arxiv.org/abs/2207.04672) 2022-7-11

<a id="42">[42]</a> Uni-Perceiver-MoE: Learning Sparse Generalist Models with Conditional MoEs	[[NIPS 2022]](https://arxiv.org/abs/2206.04674) 2022-6-9

<a id="41">[41]</a> TUTEL: ADAPTIVE MIXTURE-OF-EXPERTS AT SCALE	[[MLSys 2023]](https://arxiv.org/abs/2206.03382) 2022-6-7

<a id="40">[40]</a> Multimodal Contrastive Learning with LIMoE: the Language-Image Mixture of Experts	[[NIPS 2022]](https://arxiv.org/abs/2206.02770) 2022-6-6

<a id="39">[39]</a> Task-Specific Expert Pruning for Sparse Mixture-of-Experts	[[ArXiv 2022]](https://arxiv.org/abs/2206.00277) 2022-6-1

<a id="38">[38]</a> Eliciting and Understanding Cross-Task Skills with Task-Level Mixture-of-Experts [[EMNLP 2022]](https://arxiv.org/abs/2205.12701) 2022-5-25

<a id="37">[37]</a> AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning [[EMNLP 2022]](https://arxiv.org/abs/2205.12410) 2022-5-24

<a id="36">[36]</a> SE-MoE: A Scalable and Efficient Mixture-of-Experts Distributed Training and Inference System [[ArXiv 2022]](https://arxiv.org/abs/2205.10034) 2022-5-20

<a id="35">[35]</a> On the Representation Collapse of Sparse Mixture of Experts	[[NIPS 2022]](https://arxiv.org/abs/2204.09179) 2022-4-20

<a id="34">[34]</a> Residual Mixture of Experts [[ArXiv 2022]](https://arxiv.org/abs/2204.09636) 2022-4-20

<a id="33">[33]</a> STABLEMOE: Stable Routing Strategy for Mixture of Experts	[[ACL 2022]](https://arxiv.org/abs/2204.08396) 2022-4-18

<a id="32">[32]</a> MoEBERT: from BERT to Mixture-of-Experts via Importance-Guided Adaptation [[NAACL 2022]](https://arxiv.org/abs/2204.07675) 2022-4-15

<a id="31">[31]</a> BaGuaLu: Targeting Brain Scale Pretrained Models with over 37 Million Cores	[[PPoPP 2022]](https://dl.acm.org/doi/10.1145/3503221.3508417) 2022-3-28

<a id="30">[30]</a> FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models	[[PPoPP 2022]](https://dl.acm.org/doi/10.1145/3503221.3508418) 2022-3-28

<a id="29">[29]</a> HetuMoE: An Efficient Trillion-scale Mixture-of-Expert Distributed Training System	[[ArXiv 2022]](https://arxiv.org/abs/2203.14685) 2022-3-28

<a id="28">[28]</a> Parameter-Efficient Mixture-of-Experts Architecture for Pre-trained Language Models	[[COLING 2022]](https://arxiv.org/abs/2203.01104) 2022-3-2

<a id="27">[27]</a> Mixture-of-Experts with Expert Choice Routing	[[NIPS 2022]](https://arxiv.org/abs/2202.09368) 2022-2-18

<a id="26">[26]</a> ST-MOE: DESIGNING STABLE AND TRANSFERABLE SPARSE EXPERT MODELS	[[ArXiv 2022]](https://arxiv.org/abs/2202.08906) 2022-2-17

<a id="25">[25]</a> UNIFIED SCALING LAWS FOR ROUTED LANGUAGE MODELS	[[ICML 2022]](https://arxiv.org/abs/2202.01169) 2022-2-2

<a id="24">[24]</a> One Student Knows All Experts Know: From Sparse to Dense [[ArXiv 2022]](https://arxiv.org/abs/2201.10890) 2022-1-26

<a id="23">[23]</a> DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale	[[ICML 2022]](https://arxiv.org/abs/2201.05596) 2022-1-14

<a id="22">[22]</a> EvoMoE: An Evolutional Mixture-of-Experts Training Framework via Dense-To-Sparse Gate	[[ArXiv 2021]](https://arxiv.org/abs/2112.14397) 2021-12-29

<a id="21">[21]</a> Efficient Large Scale Language Modeling with Mixtures of Experts	[[EMNLP 2022]](https://arxiv.org/abs/2112.10684) 2021-12-20

<a id="20">[20]</a> GLaM: Efficient Scaling of Language Models with Mixture-of-Experts	[[ICML 2022]](https://arxiv.org/abs/2112.06905) 2021-12-13

<a id="19">[19]</a> Dselect-k: Differentiable selection in the mixture of experts with applications to multi-task learning [[NIPS 2021]](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html) 2021.12.6

<a id="18">[18]</a> Tricks for Training Sparse Translation Models [[NAACL 2022]](https://arxiv.org/abs/2110.08246) 2021-10-15

<a id="17">[17]</a> Taming Sparsely Activated Transformer with Stochastic Experts [[ICLR 2022]](https://arxiv.org/abs/2110.04260) 2021-10-8

<a id="16">[16]</a> MoEfication: Transformer Feed-forward Layers are Mixtures of Experts [[ACL 2022]](https://arxiv.org/abs/2110.01786) 2021-10-5

<a id="15">[15]</a> Beyond distillation: Task-level mixture-of-experts for efficient inference [[EMNLP 2021]](https://arxiv.org/abs/2110.03742) 2021-9-24

<a id="14">[14]</a> Scalable and Efficient MoE Training for Multitask Multilingual Models [[ArXiv 2021]](https://arxiv.org/abs/2109.10465) 2021-9-22

<a id="13">[13]</a> DEMix Layers: Disentangling Domains for Modular Language Modeling [[NAACL 2022]](https://arxiv.org/abs/2108.05036) 2021-8-11

<a id="12">[12]</a> Go Wider Instead of Deeper	[[AAAI 2022]](https://arxiv.org/abs/2107.11817) 2021-7-25

<a id="11">[11]</a> Scaling Vision with Sparse Mixture of Experts	[[NIPS 2021]](https://arxiv.org/abs/2106.05974) 2021-6-10

<a id="10">[10]</a> Hash Layers For Large Sparse Models	[[NIPS 2021]](https://arxiv.org/abs/2106.04426) 2021-6-8

<a id="9">[9]</a> M6-t: Exploring sparse expert models and beyond [[ArXiv 2021]](https://arxiv.org/abs/2105.15082) 2021-5-31

<a id="8">[8]</a> BASE Layers: Simplifying Training of Large, Sparse Models	[[ICML 2021]](https://arxiv.org/abs/2103.16716) 2021-5-30

<a id="7">[7]</a> FASTMOE: A FAST MIXTURE-OF-EXPERT TRAINING SYSTEM	[[ArXiv 2021]](https://arxiv.org/abs/2103.13262) 2021-5-21

<a id="6">[6]</a> CPM-2: Large-scale Cost-effective Pre-trained Language Models [[AI Open 2021]](https://arxiv.org/abs/2106.10715) 2021-1-20

<a id="5">[5]</a> Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity	[[ArXiv 2022]](https://arxiv.org/abs/2101.03961) 2021-1-11

<a id="4">[4]</a> Beyond English-Centric Multilingual Machine Translation [[JMLR 2021]](https://arxiv.org/abs/2010.11125) 2020-10-21

<a id="3">[3]</a> GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding	[[ICLR 2021]](https://arxiv.org/abs/2006.16668) 2020-6-30

<a id="2">[2]</a> Modeling task relationships in multi-task learning with multi-gate mixture-of-experts [[KDD 2018]](https://arxiv.org/abs/2010.11125) 2018-7-19

<a id="1">[1]</a> OUTRAGEOUSLY LARGE NEURAL NETWORKS: THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER	[[ICLR 2017]](https://arxiv.org/abs/1701.06538) 2017-1-23

















