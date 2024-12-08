---
title: "Paper summary - Making Convolutional Networks Shift-Invariant Again"
datePublished: Sat Nov 30 2024 15:17:01 GMT+0000 (Coordinated Universal Time)
cuid: cm44bfw4x001109jleeeybaro
slug: paper-summary-making-convolutional-networks-shift-invariant-again
tags: ai, computer-vision, aliasing, ai-robustness, ai-natural-robustness, ai-spatial-robustness

---

## TL; DR

Many operations in a typical convolutional neural network down-sample the signal without accounting for aliasing. Aliasing causes frequency content that makes a classifier vulnerable to rotation and translations.

## Aliasing in CNNs

Aliasing is a phenomenon that happens when a signal is sampled with an insufficient sampling frequency. One of the most fundamental results in signal processing is the [sampling theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem), stating that to properly sample a signal, you need to sample with at least double the maximum frequency in the signal. This does not make a lot of intuitive sense for most people, since often the idea of frequency is only associated with audio. Really, everything can be seen as some kind of variating data, with different speeds, including pixel in an image. The classic examples in photography is the brick wall: the brick is a periodic pattern in the pixels. The sampling frequency is the resolution of the camera (i.e. how spatially often the camera takes a sample from the world). Without enough megapixels, you get aliasing, in the form of a [Moiré pattern](https://en.wikipedia.org/wiki/Aliasing). Basically, you create visual content that does not physically exist.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1732448081740/ecb7d093-f6aa-465e-b03e-fa5a3348386c.png align="center")

Why does a CNN care about all this? Well, strided operations are actually (under)sampling a signal by definition. This means that, depending on where the sampling grid ends up, different spurious frequencies will be created. That means that the same images with a small spatial modification (e.g. translation, rotation) will have large differences after an aliasing-inducing operation, potentially being recognized as something very different.

**A model with aliasing is not spatially invariant and thus will have largely different outputs caused by tiny input modifications.**

Difference here can mean a flipped prediction but also a more subtle lowering in the prediction confidence. From the paper:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1732787881512/ccfe56f8-aac4-4167-9b09-a4e2b57b25f9.png align="center")

Granted, it’s CIFAR, but the confidence shift is pretty dramatic

One non-obvious thing is that, even if you blur the initial image (i.e. remove/blur the high frequency content such as textures, fine details, edges), the feature map in intermediate convolutional layer can still contain high frequency, basically wherever a convolutional filter has a strong activation.

To avoid aliasing, you have two solutions: either remove the high frequency content with a low pass filter (i.e. blurring the image/feature map), losing information or use a higher sampling frequency (not really feasible, especially in the middle of a CNN). The paper proposes to go with the first solution.

## Anti-aliased CNNs

The paper proposes a very straightforward solution: for every operation that can creates aliasing (i.e. every down-sampling, so strided pooling and strided convolution for a ResNet), before the actual strided operation, apply a low pass filter. There are results for different filters:

* Rect \\([1, 1]\\) - effectively equivalent to average pooling with filter size = 2
    
* Triangle \\([1, 2, 1]\\)
    
* Binomial \\([1, 4, 6, 4, 1]\\)
    

These correspond to stronger and stronger low pass filtering, so more information loss but less potential aliasing. It’s worth noting that these method do not completely remove aliasing, they mitigate it. So spatial brittleness will still be there, but it should improve substantially, and improve more with a more aggressive filtering (possibly at the expense of normal accuracy).

## Results

There are a lot of results to unpack:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1732977390040/4db7b247-2cc0-454a-a6da-9a8a110db430.png align="center")

Here, consistency appears to be how often (%) the network keeps the same label given an horizontal shift.

* Accuracy improves overall, despite loss of information
    
* Shift invariance is somehow brute-forced by model size (larger networks are more consistent)
    
* Anti-aliasing is still helpful in larger networks
    

There are also results on robustness to image corruption (ImageNet-C \[3\]):

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1732632856537/8e1dd240-d3b2-428e-82c4-65b2ad35319e.png align="center")

Better invariance to high-frequency degradation is expected, but there is also a less predictable resistance to blur. Maybe blurry image can still have aliased feature maps deeper in the network.

Anti-aliasing also seems to help adversarial (i.e. worst-case) spatial perturbations. Here, a weaker adversary is tested with respect to [\[2\]](#Exploring_Robustness) (up to +-16 shift vs +-24), but better test granularity is used. Accuracy protocol is, if any image of the set is misclassified, it counts as wrong so pretty restrictive. Contrary to \[\[2\]\], data augmentation seems to effectively address the issue, but anti-aliasing still helps (the stronger the better).

Finally, they try to learn the blur filter (i.e. initializing the filter to binomial, but then making it adjustable by gradient descent) seems to lose effectiveness over the training, probably because there is no explicit incentive to learn effective anti-aliasing.

### Is this still a problem in modern networks?

These results are beautiful, but is this still relevant today? After all, ResNets have not been state of the art in quite a while.

Looking at modern convolutional networks, it seems like non-anti-aliased strided operations are still there. Looking at ConvNextV2 \[4\] (taken from pytorch image models):

```python
import timm

model = timm.create_model('convnextv2_base.fcmae', pretrained=False)
model.eval()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1732979345808/e92b277e-07d8-4f28-a73d-74580f61f729.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1732979347643/8044e4be-90ba-4206-b262-9c527cdeb7a6.png align="center")

Swin transformers still have a strided conv at the beginning. Plus, the patch merging layer is still down-sampling.

It appears aliasing is alive and well in modern computer vision!

## References

\[1\] [Zhang R. - Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486)

\[2\] [L. Engstrom et al. - Exploring the Landscape of Spatial Robustness](https://arxiv.org/abs/1712.02779)

\[3\] [D. Hendrycks, T. Dietterich - Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261)

\[4\] [S. Woo et al. - ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)