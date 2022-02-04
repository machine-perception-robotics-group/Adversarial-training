# Adversarial training
This repo. includes several adversarial training (AT) methods and some network architectures are avarable.
We list avarable architectures bellow:

|network|# of layers|
| :---: | :---: |
|ResNet|18/34/50/101/152|
|WideResNet| -- |
|VGG|11/13/16/19 (w/ bn or w/o bn)|
|LeNet|5|
|RBF CNNs|3-convs + 1-fc|

* To Do: We will increase the avarable base network including state-of-the-art method.

## Defense methods
We prepare several type of the adversarial training, but most of methods basically defense, which is conscious the decision bounday, on this repo..

- [x] Adversarial logits paring (ALP): H. Kannan, et al., arXiv, 2018. [paper](https://arxiv.org/abs/1803.06373)<br>
- [x] Adversarial vertex mixup (AVmixup)S. Lee, et al., CVPR, 2020. [paper](https://arxiv.org/abs/2003.02484)<br>
- [x] Clean logits paring (CLP): H. Kannan et al., arXiv, 2018. [paper](https://arxiv.org/abs/1803.06373)<br>
- [x] Friendly adversarial training (FAT)[paper](https://arxiv.org/abs/2002.11242)<br>
- [x] Guided-awere instance-reweighting adversarial training (GAIRAT): J. Zhang, et al., ICLR, 2021. [paper](https://arxiv.org/abs/2010.01736)<br>
- [x] Guided adversarial training (GAT): G. Sriramanan et al., NeurIPS, 2020. [paper](https://arxiv.org/abs/2011.14969)<br>
- [x] non-linearlity CNN with the kernel trick: S. A. Taghanaki, et al. CVPR, 2019. [paper](https://arxiv.org/abs/1903.01015)<br>
- [x] Learnable boundary guided adversarial training (LBGAT): J. Cui, et al., ICCV, 2021. [paper](https://arxiv.org/abs/2011.11164)<br>
- [x] Margin aware instance learning (MAIL): Q. Wang, et al., NeurIPS, 2021. [paper](https://arxiv.org/abs/2106.07904#:~:text=Probabilistic%20Margins%20for%20Instance%20Reweighting%20in%20Adversarial%20Training,-Qizhou%20Wang%2C%20Feng&text=Reweighting%20adversarial%20data%20during%20training,critical%20and%20given%20larger%20weights.)<br>
- [x] Misclassification aware adversarial training (MART): J. Zhang, et al., ICLR, 2021. [paper](https://arxiv.org/abs/2010.01736)<br>
- [ ] Margin minimax adversarial training (MMA training)<br>
- [x] Probabilistically compact loss with logits constraints (PC loss with logits constraints): X. Li, et al., AAAI, 2021. [paper](https://arxiv.org/abs/2012.07688)<br>
- [x] Prototype conformity loss (PC loss): A. Mustafa, et al., ICCV, 2019. [paper](https://arxiv.org/abs/1904.00887)<br>
- [x] Adversarial training with PGD attack (Standard AT): A. Madry, et al., ICLR, 2018. [paper](https://arxiv.org/abs/1706.06083)<br>
- [x] Tadeoff-inspired adversarial defense via surrogate-loss minimization (TRADES): H. Zhang, et al., ICML, 2019. [paper](https://arxiv.org/abs/1901.08573)<br>
- [x] Weighted minimax risk (WMMR): H. Zeng, et al., AAAI, 2021. [paper](https://arxiv.org/abs/2010.12989)

## Evaluation metrics for the clusturing of low dimensional feature space
* Silhouette coefficient (Sil.)
* Calkins-Harabasz index (Cal.)
* Mutual information score (MI)
* Homogeneity score (Homo.)
* Completeness score (Comp.)