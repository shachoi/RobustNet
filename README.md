## RobustNet (CVPR 2021 Oral): Official Project Webpage
Codes and paper will release soon.

This repository provides the official PyTorch implementation of the following paper:
> **RobustNet:** Improving Domain Generalization in Urban-Scene Segmentationvia Instance Selective Whitening<br>
> Sungha Choi* (LG AI Research), Sanghun Jung* (KAIST AI), Huiwon Yun (Sogang Univ.)<br>
> Joanne T. Kim (Korea Univ.), Seungryong Kim (Korea Univ.), Jaegul Choo (KAIST AI) (*: equal contribution)<br>
> CVPR 2021, Accepted as Oral Presentation<br>

> Paper : [pdf] [supp] <br>

> **Abstract:** 
*Enhancing the generalization performance of deep neural networks in the real world (i.e., unseen domains) is crucial for safety-critical applications such as autonomous driving.
To address this issue, this paper proposes a novel instance selective whitening loss to improve the robustness of the segmentation networks for unseen domains.
Our approach disentangles the domain-specific style and domain-invariant content encoded in higher-order statistics (i.e., feature covariance) of the feature representations and selectively removes only the style information causing domain shift.
As shown in the below figure, our method provides reasonable predictions for (a) low-illuminated, (b) rainy, and (c) unexpected new scene images.
These types of images are not included in the training dataset that the baseline shows a significant performance drop, contrary to ours.
Being simple but effective, our approach improves the robustness of various backbone networks without additional computational cost. 
We conduct extensive experiments in urban-scene segmentation and show the superiority of our approach over existing work.*<br>

<p align="center">
  <img src="assets/fig_main.png" />
</p>

## Code Contributors
[Sungha Choi](https://www.linkedin.com/in/sungha-choi-1130185a/) (LG AI Research), Sanghun Jung (KAIST AI)

