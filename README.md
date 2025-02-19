# MeMo
**Building Language Models by using Associative Memories: Memorization before learning**


Welcome to the **MeMo Lab**. This is the repository where we can all experiment with this new paradigm for building Language Models. 

We are maintaining three different versions:
1) **MeMoCMM.py**: A simple version realized within PyTorch only using vector-matrix operations 
2) **MeMoPyTorch**: A version that exploits the Neural Network modules in order to be ready for the *learning phase* after the *memorization*
3) **MeMoHF**: The HuggingFace version that sees MeMo as a transformer architecture so that it can be exploited in existing solutions

This package contains:
- a **PlayingWithMeMo** for the three versions that show how to memorize, how to retrieve and how to forget
- The **Experiments** proposed in the paper

Enjoy, collaborate, **MeMo**rize!

More details in this [paper](https://arxiv.org/abs/2502.12851): 
```
@misc{zanzotto2025memo,
    title={MeMo: Towards Language Models with Associative Memory Mechanisms},
    author={Fabio Massimo Zanzotto and Elena Sofia Ruzzetti and Giancarlo A. Xompero
        and Leonardo Ranaldi and Davide Venditti and Federico Ranaldi
        and Cristina Giannone and Andrea Favalli and Raniero Romagnoli},
    year={2025},
    eprint={2502.12851},
    archivePrefix={arXiv},
    primaryClass={cs.CL} }
```



