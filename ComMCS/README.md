<h2 align="center"> <a href="https://github.com/HITsz-TMG/TMGNLP/ComMCS">Improving Value-based Process Verifier via Low-Cost Variance Reduction
</a></h2>

<div align="center">

<!-- **Affiliations:** -->

_**Zeitan Sun, Dongfang Li, Baotian Hu, Min Zhang**_

</div>

## 🎏 Overview

<div align=center><img src="./pics/intro.png" height="100%" width="78%"/></div>

**ComMCS** tries to reduce the sampling variance without introducing additional LLM inference cost when optimizing value-based process verifiers. ComMCS reshapes the output of process verifier as the one-step value distribution, and achieve the variance reduction via variance expression, variance approximation and variance comparison.


## 🌈 Results

We perform experiments in GSM8K and MATH-500 on tasks including Best-of-N sampling and beam search.

<div align=center><img src="./pics/result_bon.png" height="100%" width="78%"/></div>

<div align=center><img src="./pics/result_bs.png" height="100%" width="78%"/></div>


## 🌟 Usage




## 📚 Citation
If you find ComMCS useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{sun2025improvingvaluebasedprocessverifier,
      title={Improving Value-based Process Verifier via Low-Cost Variance Reduction}, 
      author={Zetian Sun and Dongfang Li and Baotian Hu and Min Zhang},
      year={2025},
      eprint={2508.10539},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.10539}, 
}
```

