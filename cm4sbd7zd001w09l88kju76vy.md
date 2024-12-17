---
title: "Paper summary -"
datePublished: Sat Dec 14 2024 23:00:00 GMT+0000 (Coordinated Universal Time)
cuid: cm4sbd7zd001w09l88kju76vy
slug: paper-summary
tags: ai, computer-vision, aliasing, ai-robustness, ai-natural-robustness, ai-spatial-robustness

---

## TL; DR: by carefully selecting where to put aliasing mitigation in a CNN it is possible to limit anti-aliasing information loss. Aliasing negatively impact performances in a way that is not evident from validation loss.

Another paper on the effect of aliasing, a very much studied concept in 2020, kinda forgotten after that. It can be seen as a follow up to \[2\], refining the ideas expressed there (check out the [summary](https://robustramblings.hashnode.dev/paper-summary-making-convolutional-networks-shift-invariant-again)! I'm supposed to do that right?).

There are a couple of concepts that are key to this paper, to each we will give a separate sub-section.

### Can CNNs learn low-pass filters from data?

\[2\] said it was not the case, hypothesizing that there is no explicit incentive to learn anti-aliasing. Even a filter initialized as binomial low-pass would not be preserved by training. Here, they say it depends on the filter size, with a pretty solid argument. The key idea comes from Fourier analysis. If you are not familiar with the Fourier transform, an extremely abridged version is that you can view any function as a sum on (complex) sine functions. In real life, every approximation is imperfect, but it get better as you add more and more sine functions (with different parameters). In the figure below, you can see how a rectangle approximation gets better and better with more terms ([source](https://www.researchgate.net/publication/337768291_FFT-based_solver_for_higher-order_and_multi-phase-field_fracture_models_applied_to_strongly_anisotropic_brittle_materials_and_poly-crystals))

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1734429872298/a5ba5857-4e16-497e-aaec-51bf3e4e6520.png align="center")

Basically, without going over all of Fourier analysis, the key idea is:

* You want low-pass filtering to mitigate aliasing.
    
* An ideal anti-alias filter (low pass filter, removing higher frequency that cause aliasing) is a rectangle in the frequency domain.
    
* The more sine waves you add, the more your filter improves and mitigates various sources of noise due to imperfect approximations (e.g. Gibbs phenomenon, basically worse results)
    
* To add another sine wave in the frequency domain, you need to add two to the size of your existing filter. A nice formula is actually given in the paper. Consider the left side of the equation to be normal (spatial) domain (i.e. your CNN filter) and the right side of the equation its Fourier transform.\\(\delta[n-k] +\delta[n+k] = 2 \cos (wk)\\)
    
    basically, to add another sine, you need +2 to your filter size (one term in -k and another in +k in the normal/spatial domain).
    

One can also make the case that, if the anti-aliasing is to be learned by gradient descent, maybe a small CNN filter will not be able to learn effectively. This intuition is used to justify the fact that you need aliasing mitigation where a small (e.g. 3x3 ) CNN filter is followed by sub-sampling but not for larger filter (e.g. the 7x7 convolution at the beginning of as ResNet). This idea is proved empirically (i.e. it works better like this). As far as I can tell, there is no evidence of this beside better accuracy. Given that validation/test loss/accuracy can be misleading, I am not completely convinced, especially given that some other paper seems to show that small conv filter can mitigate anti-aliasing without explicit incentive \[3\]. I guess a better way to put it is, those filters cannot be learned efficiently.

### Can the trade-off between anti-aliasing and information loss be improved with better filter placement?

There are some key intuitions here:

* Beyond high frequencies in the original images, non-linear activation also introduce high frequencies (sharper activations such as ReLu introduce more high-frequecies, but smoother activation such as Swish do not eliminate the problem). Plus, skip connection can propagate high-frequencies through alternative paths.
    
* High frequencies bring aliasing but also information. Adding low-pass filtering at every convolutional layer gives terrible accuracy (61%)
    
* Aliasing is created by sub-sampling feature maps/images with high-frequencies Based on these ideas, its better to add low-pass filtering just before the sub-sapling, allowing the network to use high frequency information for as long as possible.
    

Another interesting idea is introduced: while there are many equivalent methods to add aliasing mitigation when accounting only for the forward pass, the equivalence breaks in the backward pass. Different filter placement are considered, the post filter coming out on top empirically.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1734429888917/2f326900-235b-47ec-9637-f0922a84d4df.png align="center")

They hypothesize that this has a better effect on the gradient, smoothing before up-sampling in the backward pass.

### Can data augmentation make the network learn anti-aliasing?

Here, they provide some comparison between four different models:

* Baseline (no anti-aliasing, infos about data augmentation are not available anymore. Most probably, some combination of random crops+resize, flipping and color jittering).
    
* Anti-alias with non-trainable filters. Trainable filters would need large spatial support (size), and that would basically need to re-design the architecture, which is not really feasible.
    
* Rand Augment - a random set of augmentation, sampled at each epoch uniformly. I think it was state of the art at the time.
    
* Rand Augment + Anti-alias. The results are:
    

| Model | Validation accuracy (%) |
| --- | --- |
| Baseline | 77.36 |
| Anti-alias | 77.76 |
| Rand Augment | 77.32 |
| Rand Augment + Anti-alias | 78.45 |

There is a positive interaction between the two techniques!

They also provide a frequency-based analysis: alternatively remove different frequencies (16 different sub-bands) from the input and see how this affect the network prediction. This should give us an idea of which feature the model is relying on for prediction. The boring result would be a small decrement in performance, more or less equal for all the models.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1734429903146/cd1b0c8b-5dbc-40fd-9f04-0223065bdb7d.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1734429916874/8ccf1d20-02b7-4891-bc6c-4a901898f73c.png align="center")

There are a couple of interesting things in these results:

* As expected, all models are severely impacted by the removal of low frequencies, but rand-augment (with or without anti-aliasing) preserves more accuracy in the lowest bin (still terrible).
    
* Rand-augment alone negatively impact performances when removing pretty much any frequency band! This suggest Rand-augment encourages the model to use features from all the spectrum, while a normally trained model is more biased on low-frequency.
    
* Anti-aliasing improves the model everywhere and makes the model able to leverage Rand-augment without losing performances. In the paper, they justify this in term of shape/texture bias (it is known that normal neural network learn feature biased toward texture, while shape based features tend to be more robust and align more with human perception). This explanation is coherent with the results on blurry images, where Rand Augment and similar techniques are severely impacted (one would guess that a texture-biased model in severely impacted by blurring, which removes textures and other higher-frequency details). It is also coherent with the principle that image textures reside in the medium frequencies, as the medium frequency impact the Rand-augment trained models the most.
    
* Another hypothesis that is cited elsewhere in the paper is that Aliasing can have a double effect: destroy informative content and create artifacts that the network will leverage for brittle generalization (it introduces shortcuts, a concept I will discuss sooner or later).
    

All in all, the conclusion seems to be that anti-aliasing is needed unless you have very large filters everywhere (which is too costly in terms of GPU memory) or very large architectures (and even then, it is not very effective). Seems like anti-aliasing is an inductive bias that one has to encode explicitly in the network.

## References

\[1\] [Vasconcelos C. et al - Impact of Aliasing on Generalization in Deep Convolutional Networks](https://arxiv.org/abs/2108.03489)

\[2\] [Zhang R. - Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486)

\[3\] [Zou X. et al - Delving Deeper into Anti-aliasing in ConvNets](https://arxiv.org/abs/2008.09604)