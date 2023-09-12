import torch.nn as nn
import torch.functional as F
import torch
import numpy as np
from torch.autograd import Variable
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, device,real_label_val=1.0, fake_label_val=0.0,loss_weight=5e-3):
        super(GANLoss, self).__init__()

        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.device =device
        self.loss = nn.BCEWithLogitsLoss()

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        # if self.gan_type in ['wgan', 'wgan_softplus']:
        #     return target_is_real
        # print(input.size())
        if target_is_real:
            target_val = self.real_label_val
            # target_label = input.new_ones(input.size()) * target_val
            # target_label.to(self.device)
            # target_label = input.new_ones(input.size()) * target_val - torch.from_numpy(np.random.uniform(0,0.02,input.size())).to(self.device)
            # target_label = input.new_ones(input.size()) * target_val
            target_label = Variable(torch.cuda.FloatTensor(np.ones((input.shape[0],1,192,192)) * target_val - np.random.uniform(0,0.2,(input.shape[0],1,192,192))),requires_grad=False)
            target_label = Variable(target_label,requires_grad=False)
        else:
            target_val = self.fake_label_val
            # target_label = input.new_ones(input.size()) * target_val
            # target_label = input.new_ones(input.size()) * target_val + torch.from_numpy(np.random.uniform(0,0.02,input.size())).to(self.device)
            # target_label = input.new_ones(input.size()) * target_val
            # target_label.to(self.device)
            target_label = Variable(torch.cuda.FloatTensor(np.ones((input.shape[0],1,192,192)) * target_val + np.random.uniform(0,0.2,(input.shape[0],1,192,192))),requires_grad=False)
            target_label = Variable(target_label,requires_grad=False)
        return target_label

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        # if self.gan_type == 'hinge':
        #     if is_disc:  # for discriminators in hinge-gan
        #         input = -input if target_is_real else input
        #         loss = self.loss(1 + input).mean()
        #     else:  # for generators in hinge-gan
        #         loss = -input.mean()
        # else:  # other gan types
        loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss

class GANLoss2(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, device,real_label_val=1.0, fake_label_val=0.0,loss_weight=5e-3):
        super(GANLoss2, self).__init__()

        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.device =device
        self.loss = nn.MSELoss()

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        # if self.gan_type in ['wgan', 'wgan_softplus']:
        #     return target_is_real
        # print(input.size())
        if target_is_real:
            target_val = self.real_label_val
            # target_label = Variable(torch.cuda.FloatTensor(np.ones((input.shape[0],1,192,192)) * target_val - np.random.uniform(0,0.2,(input.shape[0],1,192,192))),requires_grad=False)
            # target_label = torch.cuda.FloatTensor(np.ones((input.shape[0], 1, 12, 12)) * target_val) #12 = 192 // 2 ** 4
            target_label = torch.cuda.FloatTensor(
                np.ones((input.shape[0], 1, 192, 192)) * target_val)
            target_label = Variable(target_label,requires_grad=False)
        else:
            target_val = self.fake_label_val
            # target_label = Variable(torch.cuda.FloatTensor(np.ones((input.shape[0],1,192,192)) * target_val + np.random.uniform(0,0.2,(input.shape[0],1,192,192))),requires_grad=False)
            # target_label = torch.cuda.FloatTensor(np.ones((input.shape[0], 1, 12, 12)) * target_val)
            target_label = torch.cuda.FloatTensor(np.ones((input.shape[0], 1, 192, 192)) * target_val)
            target_label = Variable(target_label,requires_grad=False)
        return target_label

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)

        loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss

