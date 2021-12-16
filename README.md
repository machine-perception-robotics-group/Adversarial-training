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

where RBF CNNs indicate the convolutional neural networks that assing Radial Basis Function (RBF) kearnel to hidden layers.

* To Do: We will increase the avarable base network including state-of-the-art method.

## Defense methods
We prepare several type of the adversarial training, but most of methods basically defense, which is conscious the decision bounday, on this repo..

- [x] Adversarial logits paring (ALP)<br>
- [x] Adversarial vertex mixup (AVmixup)<br>
- [x] Clean logits paring (CLP)<br>
- [ ] Friendly adversarial training (FAT)<br>
- [x] Guided-awere instance-reweighting adversarial training (GAIRAT)<br>
- [x] Guided adversarial training (GAT)<br>
- [x] non-linearlity CNN with the kernel trick<br>
- [x] Learnable boundary guided adversarial training  (LBGAT)<br>
- [x] Margin aware instance learning (MAIL)<br>
- [x] Misclassification aware adversarial training (MART)<br>
- [ ] Margin minimax adversarial training (MMA training)<br>
- [x] Probabilistically compact loss with logits constraints (PC loss with logits constraints)<br>
- [x] Prototype conformity loss (PC loss)<br>
- [x] Adversarial training with PGD attack (Standard AT)<br>
- [x] Tadeoff-inspired adversarial defense via surrogate-loss minimization (TRADES)<br>
- [x] Weighted minimax risk (WMMR)

## Evaluation metrics for the clusturing of low dimensional feature space
* Silhouette coefficient (Sil.)
* Calkins-Harabasz index (Cal.)
* Mutual information score (MI)
* Homogeneity score (Homo.)
* Completeness score (Comp.)