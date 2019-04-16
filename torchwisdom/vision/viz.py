import torch
import numpy as np
import matplotlib.pyplot as plt


def view_classify(image: torch.Tensor, image_title: str, prob: torch.Tensor, labels: list):
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), ncols=2)
    ax1.imshow(image)
    ax1.set_title(image_title)
    ax1.axis('off')

    ps = prob.data.numpy().squeeze().tolist()
    topk = len(ps)
    ax2.barh(np.arange(topk), ps)
    ax2.set_aspect(0.2)
    ax2.set_yticks(np.arange(topk))

    ax2.set_yticklabels(labels, size='large')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()