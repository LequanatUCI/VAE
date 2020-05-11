import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        batzh_size = x.size()[0]

        qm, qv = self.enc.encode(x)
        z_samp = ut.sample_gaussian(qm, qv)
        xhat = self.dec.decode(z_samp)

        pm = torch.zeros([batzh_size, self.z_dim], dtype=torch.float, requires_grad=True)
        pv = torch.ones(batzh_size, self.z_dim, requires_grad=True)

        kl = ut.kl_normal(qm, qv, pm, pv)  # require same dimension here, two normal distributions KL(q || p)
        rec = ut.log_bernoulli_with_logits(x, xhat)

        # print(kl.size(), rec.size())
        # print(x.size()[0], type(x.size()[0]))
        kl = torch.sum(kl) / batzh_size
        rec = torch.sum(rec) / batzh_size
        nelbo = kl - rec

        # =========================================================================================================================
        # =========================================================================================================================
        # =========================================================================================================================
        # allkl = []
        # allrec = []
        # for xi in x:
        #     # dim of input xi = 784
        #     xi = torch.reshape(xi, (1, 784))   # reshape to (1, 784)
        #     qm, qv = self.enc.encode(xi)
        #     z_samp = ut.sample_gaussian(qm, qv)
        #
        #     xhat = self.dec.decode(z_samp)
        #
        #     pm = torch.zeros([1, self.z_dim], dtype=torch.float)
        #     pv = torch.ones(self.z_dim)
        #
        #     kli = ut.kl_normal(qm, qv, pm, pv)   # require same dimension here, two normal distributions KL(q || p)
        #     reci = - ut.log_bernoulli_with_logits(xi, xhat)
        #     # print(kli.item(), reci.item())
        #
        #     allkl.append(kli.item())
        #     allrec.append(reci.item())
        #
        # kl = sum(allkl)/len(allkl)
        # rec = sum(allrec)/len(allrec)

        # nelbo = kl + rec

        ################################################################################

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
