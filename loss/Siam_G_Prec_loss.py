import torch
import torch.nn as nn
import numpy as np
from loss.perceptualloss_VAE_v2 import VanillaVAE_ED
from models.VAE2 import VanillaVAE,VanillaVAE_ED_encoder
from loss.loss import weight_mse,MSE
class SiamPerceptualLoss(nn.Module):

    def __init__(self,
                 pretrain_model_path='',
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='weighted_mse'):
        super(SiamPerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.path = pretrain_model_path
        self.style_weight = style_weight

        self.vae_encoder = VanillaVAE(1,96)
        self.vae_encoder = nn.DataParallel(self.vae_encoder)
        self.ck = torch.load(self.path)

        self.vae_encoder.load_state_dict(self.ck,strict=False)
        self.vae_encoder.eval()
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        elif self.criterion_type == 'weighted_mse':
            self.criterion = weight_mse()
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, gt_img,fake_img):
        """Forward function.

        Args:
            fake_img (Tensor): Input tensor with shape (n, c, h, w).
            gt_img (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """


        x_features = self.vae_encoder(fake_img)
        gt_features = self.vae_encoder(gt_img)

        return x_features,gt_features

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

class SiamPerceptualLoss_ED(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 pretrain_model_path='',
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l2'):
        super(SiamPerceptualLoss_ED, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.path = pretrain_model_path
        self.style_weight = style_weight

        self.vae = VanillaVAE_ED_encoder(1,96)
        self.ck = torch.load(self.path)
        self.vae = nn.DataParallel(self.vae)
        self.vae.load_state_dict(self.ck, strict=False)
        for param in self.vae.parameters():
            param.requires_grad = False
        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        elif self.criterion_type == 'weighted_mse':
            self.criterion = MSE()
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, fake_img, gt_img):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vae features
        fake_features= self.vae(fake_img)
        gt_features = self.vae(gt_img)
        data_fake = fake_features.detach().cpu().numpy()
        data_gt = gt_features.detach().cpu().numpy()
        np.save('fake.npy',data_fake)
        np.save('gt.npy', data_gt)
        result = self.criterion(gt_features,fake_features)

        return result

