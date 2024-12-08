---
title: "Does anti-aliasing smooth spatial loss surface?"
datePublished: Sun Dec 08 2024 14:23:22 GMT+0000 (Coordinated Universal Time)
cuid: cm4fp1pyw000n09l83gpl3lji
slug: does-anti-aliasing-smooth-spatial-loss-surface
tags: ai, computer-vision, aliasing, ai-robustness, ai-natural-robustness, ai-spatial-robustness

---

The last two posts ([1](https://robustramblings.hashnode.dev/paper-review-exploring-the-landscape-of-spatial-robustness), [2](https://robustramblings.hashnode.dev/paper-summary-making-convolutional-networks-shift-invariant-again)) I published were on the topic of spatial robustness, i.e. the stability of a neural network prediction with respect to small translation and rotation. A curious observation from \[1\] states that first order adversary seem to not be very effective in finding spatial adversarial examples, whereas they work very well for finding typical \\(l_\infty \\) adversarial examples (i.e. the typical adversarial example based on single pixel modification, with no single modification larger that a set threshold). In the paper, they give the explanation that the loss surface for spatial adversarial examples seems to have a lot of peaks, thus a gradient-based method would not work very well. The curious thing is that, if you look at the figures, the peaks seem almost periodic.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1732006683241/95b023fe-6d90-44e1-981e-06bcbd3a0dc6.png?auto=compress,format&format=webp align="center")

Weird. If you look at \[2\], and specifically when the effect of aliasing on CIFAR is shown, you see something similar

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1733666474254/6de599cf-519f-4057-92fa-3c8dbcb5ccab.png align="center")

The confidence varies with some periodicity with respect to diagonal shift.

Another interesting point made in \[2\] is that aliasing mitigation seems to recover a lot of spatial robustness. Thus, why not try and see if aliasing mitigation makes the loss smoother? To the drawing board!

## Experiment setup

The idea here is to evaluate the same model (a ResNet50) with and without aliasing mitigation on the ImageNet validation set, with the attack model described in \[1\]:

* 31 rotation values equally spaced between -30 and +30 degrees
    
* 5 translation (left to right) values equally spaced between -24 and +24
    

From here, we see if the accuracy improves (as it should), and see if some surface smoothness metrics (brilliantly provided by Claude) show some improvement and plot some of them to see if the effect is visually perceivable.

First some setup:

```python
from glob import glob
from tqdm import tqdm

import antialiased_cnns
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import torch
from torchmetrics.classification import MulticlassAccuracy
from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms.functional import affine
from torchvision.io import ImageReadMode

image_path_list = glob("val/**/*.JPEG", recursive=True)
label_df = pd.read_csv("labels.txt", sep=" ", header=None, names=["filename", "class_id", "class_name"])
label_list = sorted(label_df["filename"].to_list())

antialiased_model = antialiased_cnns.resnet50(pretrained=True)
antialiased_model.eval().to("cuda:0")
# Taken from https://pytorch.org/vision/stable/models.html
base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
base_model.eval().to("cuda:0")
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

# Grid parameters for spatial attacks, see https://arxiv.org/pdf/1712.02779v4
# Rotation from -30 to +30 degrees, 31 values
rotation_grid = np.linspace(-30, +30, 31)
# Translation from -24 to +24 pixels, 5 values - Assuming it is only Left to Right
translation_grid = np.linspace(-24, +24, 5)
```

With this code, we load the ImageNet validation set from a local folder. We take a pretrained normal ResNet50 from Torchvision and recover a pre-trained ResNet50 with anti-aliasing from the awesome repo ([antialiased-cnns](https://github.com/adobe/antialiased-cnns)) provided by the author of \[2\].

Now, we calculate loss and accuracy:

```python
normal_top1_accuracy = MulticlassAccuracy(1000, top_k=1).to("cuda:0")
antialias_top1_accuracy = MulticlassAccuracy(1000, top_k=1).to("cuda:0")

criterion = torch.nn.CrossEntropyLoss()
get_label = lambda x: torch.LongTensor([int(label_list.index(x.split("\\")[-2]))]).to("cuda:0")

normal_losses = np.zeros((len(image_path_list), 
                        rotation_grid.shape[0], 
                        translation_grid.shape[0]))
antialiased_losses = np.zeros((len(image_path_list), 
                        rotation_grid.shape[0], 
                        translation_grid.shape[0]))

for img_idx, img_path in tqdm(enumerate(image_path_list), total=len(image_path_list)):
   with torch.no_grad():
      img = preprocess(
         decode_image(img_path,
                     mode=ImageReadMode.RGB))
      label = get_label(img_path)
      for iidx in range(rotation_grid.shape[0]):
         for jidk in range(translation_grid.shape[0]):
            transformed_img = affine(img.clone(), 
                                    angle=rotation_grid[iidx],
                                    translate=[0, translation_grid[jidk]],
                                    scale=1,
                                    shear=0)[None].to("cuda:0")
            normal_out = base_model(transformed_img)
            normal_losses[img_idx, iidx, jidk] = criterion(
               normal_out, label)
            normal_top1_accuracy.update(normal_out, label)

            antialias_out = antialiased_model(transformed_img)
            antialiased_losses[img_idx, iidx, jidk] = criterion(

               antialias_out, label) 
            antialias_top1_accuracy.update(antialias_out, label)

print(f"Normal top-1 accuracy {normal_top1_accuracy.compute()}") # Gives 0.5601!
print(f"Anti-aliased top-1 accuracy {antialias_top1_accuracy.compute()}") # Gives 0.7898!
```

Awesome, the accuracy does improve by a lot! This accuracy is not really the same calculated in the papers. In the papers, they evaluate adversarial accuracy, i.e. if at least one version of an image is classified incorrectly than the image is considered wrongly classified. I think this calculation (each version of the image is counted once) gives a better a idea of the classification stability, even there is a lot of interdependence in the samples.

Now, on to the surface metrics!

```python
def fast_surface_metrics(surfaces):
    """
    Compute metrics for a pair of surfaces.
    
    Args:
        surfaces: tuple of (Z1, Z2) where each is a 2D numpy array
    Returns:
        dict: Dictionary containing the metrics
    """
    Z1, Z2 = surfaces
    
    # Compute gradients
    gy1, gx1 = np.gradient(Z1)
    gradient_magnitude_1 = np.sqrt(gx1**2 + gy1**2)
    
    gy2, gx2 = np.gradient(Z2)
    gradient_magnitude_2 = np.sqrt(gx2**2 + gy2**2)
    
    return {
        'avg_gradient': (np.mean(gradient_magnitude_1), np.mean(gradient_magnitude_2)),
        'gradient_std': (np.std(gradient_magnitude_1), np.std(gradient_magnitude_2)),
        'total_variation': (
            np.sum(np.abs(np.diff(Z1, axis=0))) + np.sum(np.abs(np.diff(Z1, axis=1))),
            np.sum(np.abs(np.diff(Z2, axis=0))) + np.sum(np.abs(np.diff(Z2, axis=1)))
        )
    }
avg_gradients = []
gradient_stds = []
total_variations = []
for idx in tqdm(range(normal_losses.shape[0]), total=normal_losses.shape[0]):
    r = fast_surface_metrics((normal_losses[idx], antialiased_losses[idx]))
    avg_gradients.append(r["avg_gradient"])
    gradient_stds.append(r["gradient_std"])
    total_variations.append(r["total_variation"])
```

Once again, thanks Claude. The overall smoothness situation seems to improve!

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1733667263642/7c57fe34-2184-4c25-894f-9d1a4efe5b3a.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1733667268076/45ee50e9-8264-45fe-ba9a-784efe671619.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1733667272510/f2c95eb7-3011-4ffa-b717-688226a846e2.png align="center")

I probably should give better name to axis. In any case, right column is anti-aliased, left column is normal. The pattern appears consistent in all three metrics!

Now, let’s plot some loss surfaces

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1733667423864/7f5b0f5a-8ed3-4e68-9b09-03601bc4882d.png align="center")

Visually, it seems less evident. I guess that is why one calculates overall metrics.

Complete code available [here](https://github.com/VictorZeno/robustness-experiments/blob/main/spatial-robustness/aliasing/loss\%20smoothness/antialias_spatial_loss_surface.ipynb), in case anyone wants to play around with it.

## References

\[1\] [**L. Engstrom et al. - Exploring the Landscape of Spatial Robustness**](https://arxiv.org/abs/1712.02779)

[\[2\] **Zhang R. - Making Convolutional Networks Shift-Invariant Again**](https://arxiv.org/abs/1712.02779)