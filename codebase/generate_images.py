import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import numpy as np


# generating images here

def generateImages(model):
    imglist = []
    for i in range(200):
        m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(10), torch.eye(10))
        z_samp = m.sample()
        x_logit = model.dec.decode(z_samp)
        xprob = torch.sigmoid(x_logit)

        tempx = xprob.detach().numpy()
        tempx = np.reshape(tempx, [28, 28])

        imglist.append(tempx)

    fig = plt.figure(figsize=(8., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(10, 20),  # creates 2x2 grid of axes
                     axes_pad=0.02,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, imglist):
        # Iterating over the grid returns the Axes.
        ax.axis('off')
        ax.imshow(im, cmap='gray')
    plt.axis('off')
    plt.show()