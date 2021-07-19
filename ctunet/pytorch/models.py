import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat
from torch import cat as ccat
from torch.utils.checkpoint import checkpoint


class UNetBlock(nn.Module):
    """ Encoding block used in the skull reconstruction models.

    :param in_c: Input channels (image).
    :param out_c: Output channels (feature maps).
    :param kern_s_conv: Kernel size.
    :param pad: Padding used in convolutions.
    :param dropout_p: Dropout probability.
    :return: Encoding block.
    """

    def __init__(self, in_c, out_c, kern_s_conv=5, kern_s_uconv=2, pad=2,
                 stride_c=1, stride_upc=2, dropout_p=0, up_block=False):
        super(UNetBlock, self).__init__()

        if not up_block:
            self.block = nn.Sequential(
                nn.Conv3d(in_c, out_c, kern_s_conv, stride_c, pad, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(True),
                nn.Conv3d(out_c, out_c, kern_s_conv, stride_c, pad,
                          bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(True),
                nn.Dropout3d(dropout_p)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose3d(in_c, in_c, kern_s_uconv, stride_upc),
                nn.Conv3d(in_c, out_c, kern_s_conv, stride_c, pad, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(True),
                nn.Conv3d(out_c, out_c, kern_s_conv, stride_c, pad,
                          bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(True),
                nn.Dropout3d(dropout_p)
            )

    def forward(self, x):
        return self.block(x)


class CenterBlock(nn.Module):
    """ U-Net center block

    :param input_channels: Input channels.
    :param output_channels: Output channels.
    :param kern_sz_conv: Size of the convolutional kernel.
    :param padding: Size of the padding.
    :param dropout_p: Dropout p.
    :param fc_block: Return a fully connected block. This parameter must be
    None (default) or a two-element list with the sizes of the input and code.
    :return: Sequential block containing the central convolution.
    """

    def __init__(self, input_channels, output_channels, kern_sz_conv, padding,
                 dropout_p, fc_block=False):
        super(CenterBlock, self).__init__()

        if not fc_block:
            self.block = nn.Sequential(
                nn.Conv3d(input_channels, output_channels,
                          kern_sz_conv, padding=padding,
                          bias=False),
                nn.BatchNorm3d(output_channels),
                nn.ReLU(True),
                nn.Conv3d(output_channels, output_channels,
                          kern_sz_conv, padding=padding,
                          bias=False),
                nn.BatchNorm3d(output_channels),
                nn.ReLU(True),
                nn.Dropout3d(dropout_p)
            )
        elif type(fc_block) in [list, tuple]:
            if len(fc_block) == 2:
                ifc, cfc = fc_block
                self.block = nn.Sequential(nn.Linear(ifc, cfc),
                                           nn.Linear(cfc, ifc),
                                           nn.LeakyReLU(True),
                                           nn.Dropout3d(dropout_p))
        else:
            self.block = nn.Sequential(nn.Linear(121296, 128),
                                       nn.Linear(128, 121296),
                                       nn.LeakyReLU(True),
                                       nn.Dropout3d(dropout_p))

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, kern_sz_conv=5, kern_s_uconv=2,
                 padding=1, stride_conv=1, stride_upconv=2, dropout_p=0,
                 up_block=False):
        super(ResidualBlock, self).__init__()

        self.skip = None
        if not in_c == out_c:
            self.skip = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, stride_conv, bias=False),
                nn.BatchNorm3d(out_c)
            )
            if up_block:
                self.skip = nn.Sequential(
                    nn.ConvTranspose3d(in_c, in_c, kern_s_uconv,
                                       stride_upconv),
                    self.skip,
                )

        if not up_block:  # Downsampling block
            self.block = nn.Sequential(
                nn.Conv3d(in_c, out_c, kern_sz_conv, stride_conv, padding,
                          bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(False),
                nn.Conv3d(out_c, out_c, kern_sz_conv, stride_conv, padding,
                          bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(False),
                nn.Dropout3d(dropout_p)
            )
        else:  # Upsampling block
            self.block = nn.Sequential(
                nn.ConvTranspose3d(in_c, in_c, kern_s_uconv, stride_upconv),
                nn.Conv3d(in_c, out_c, kern_sz_conv, stride_conv, padding,
                          bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(False),
                nn.Conv3d(out_c, out_c, kern_sz_conv, stride_conv, padding,
                          bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(False),
                nn.Dropout3d(dropout_p)
            )

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out


class UNet(nn.Module):
    """ U-Net generic model.

    :param input_channels: Amount of channels in the input.
    :param output_channels: Amount of channels in the output.
    :param down_blocks: Number of downsampling blocks.
    :param up_blocks: Number of upsampling blocks.
    :param kern_sz_conv: Kernel size of the convolutions.
    :param kern_sz_upconv: Kernel size of the upconvolutions.
    :param stride_upconv: Stride of the upconvolutions.
    :param i_size: Initial amount of feature maps.
    :param padding: Padding used in up/down convolutions
    :param dropout_p: Dropout p.
    :param fc_layer: List with the sizes of the FC layer. If it's None.
    (default), a Sequential layer will be used.
    """

    def __init__(self, input_channels=1, out_channels=2, n_blocks=4,
                 kern_sz_conv=3, kern_sz_upconv=2, stride_conv=1,
                 stride_upconv=2, i_size=8, padding=1, dropout_p=0,
                 use_checkpoint=True, fc_layer=None, use_skip_connections=True,
                 apply_softmax=False, apply_sigmoid=True, cat=True,
                 residual=False):
        super(UNet, self).__init__()

        self.chk = use_checkpoint
        self.skip = use_skip_connections
        self.apply_softmax = apply_softmax
        self.apply_sigmoid = apply_sigmoid
        self.fc_layer = fc_layer
        self.cat = cat

        self.mp = nn.MaxPool3d(kernel_size=2, padding=0, stride=2,
                               dilation=1, return_indices=True)

        self.d_blocks = []
        self.u_blocks = []

        for i in range(n_blocks):
            c1 = input_channels if i == 0 else i_size * pow(2, i - 1)
            c2 = i_size * pow(2, i)
            block = ResidualBlock if residual else UNetBlock
            self.d_blocks.append(block(c1, c2, kern_sz_conv, 0, padding,
                                       stride_conv, 0, dropout_p))
        self.d_blocks = nn.ModuleList(self.d_blocks)

        icb, ocb = i_size * pow(2, n_blocks - 1), i_size * pow(2, n_blocks)
        self.cblock = CenterBlock(icb, ocb, kern_sz_conv, padding,
                                  dropout_p, fc_layer)

        for i in range(n_blocks - 1, -1, -1):
            if self.skip or i == n_blocks - 1:
                c1 = i_size * pow(2, i) * (2 if i == (n_blocks - 1) else 4)
                c1 = c1 // 2 if (self.fc_layer and i == (n_blocks - 1)) else c1
                c1 = c1 // 2 if not self.cat or (i == n_blocks - 1) else c1
                c2 = int(i_size * pow(2, i))
            else:
                c1 = i_size * pow(2, i) * 2
                c2 = i_size * pow(2, i)
            block = ResidualBlock if residual else UNetBlock
            self.u_blocks.append(block(c1, c2, kern_sz_conv, kern_sz_upconv,
                                       padding, stride_conv, stride_upconv,
                                       dropout_p, True))
        self.u_blocks = nn.ModuleList(self.u_blocks)

        lc_in = 2 * i_size if (self.skip and self.cat) else i_size
        self.last_conv = nn.Conv3d(lc_in, out_channels, 1)

    def forward(self, x):
        d = []
        u = []
        mps = []
        for i, block in enumerate(self.d_blocks):
            o = x if i == 0 else mps[-1]
            d.append(checkpoint(block, o) if self.chk else block(o))
            mps.append(self.mp(d[-1])[0])

        db_shape = mps[-1].shape
        mps[-1] = mps[-1].view(-1) if self.fc_layer else mps[-1]

        cblock = checkpoint(self.cblock, mps[-1]) if self.chk \
            else self.cblock(mps[-1])

        cblock = cblock.view(db_shape) if self.fc_layer else mps[-1]

        for i, block in enumerate(self.u_blocks):
            o = cblock if i == 0 else u[-1]
            ubl = checkpoint(block, o) if self.chk else block(o)

            if self.skip:
                if self.cat:
                    u.append(ccat((ubl, d[-i - 1]), 1))
                else:
                    u.append(ubl + d[-i - 1])
            else:
                u.append(ubl)

        lc = checkpoint(self.last_conv, u[-1]) if self.chk else \
            self.last_conv(u[-1])

        out = F.softmax(lc, dim=1) if self.apply_softmax else lc
        out = torch.sigmoid(out) if self.apply_sigmoid else out

        return out


# class AE4b1i1o(UNet):
#     """ Denoising AE made from a 4-block U-Net without skip-connections. """
#
#     def __init__(self):
#         super(AE4b1i1o, self).__init__(use_skip_connections=False,
#                                        i_size=5)


class UNet4b2i3o(UNet):
    """ Three-channel output UNet with Shape Priors. """

    def __init__(self):
        super(UNet4b2i3o, self).__init__(i_size=7, input_channels=2,
                                         out_channels=3,
                                         use_checkpoint=True)
class UNet4b1i3o(UNet):
    """ Three-channel output UNet without Shape Priors. """

    def __init__(self):
        super(UNet4b1i3o, self).__init__(i_size=7, input_channels=1,
                                         out_channels=3,
                                         use_checkpoint=True)


class UNetSP(UNet4b2i3o):
    """ Unet with Shape Priors.

    This model uses a standard U-Net with two input channels and three
    output channels for allowing the prediction of both the missing bone
    flap and the full skull.

    The output of this model takes the output of the three-channel UNet,
    and indirectly forces that the channels 1 and 2 are the flap and full
    skull, respectively (the channel 0 is the full skull background). This
    is enforced in the corresponding Loss function (see the ProblemHandler
    FlapRecWithShapePriorDoubleOut.comp_losses_metrics).

    """

    def __init__(self):
        super(UNetSP, self).__init__()

    def forward(self, x):
        # U-Net forward pass
        backg_flap_fullsk = super(UNetSP, self).forward(x)
        backg = backg_flap_fullsk[:, 0:1]
        flap = backg_flap_fullsk[:, 1:2]
        fullsk = backg_flap_fullsk[:, 2:3]

        encoded_full_skull = ccat((backg,
                                   flap + fullsk),
                                  1)
        encoded_flap = ccat((1 - flap,
                             flap),
                            1)
        return encoded_full_skull, encoded_flap


class UNetDO(UNet4b1i3o):
    """ Unet with Double output."""
    def __init__(self):
        super(UNetDO, self).__init__()

    def forward(self, x):
        # U-Net forward pass
        backg_flap_fullsk = super(UNetDO, self).forward(x)
        backg = backg_flap_fullsk[:, 0:1]
        flap = backg_flap_fullsk[:, 1:2]
        fullsk = backg_flap_fullsk[:, 2:3]

        encoded_full_skull = ccat((backg,
                                   flap + fullsk),
                                  1)
        encoded_flap = ccat((1 - flap,
                             flap),
                            1)
        return encoded_full_skull, encoded_flap


# Legacy models (kept for consistency with the submissions, use the generic
# classes for new trained models instead.)

def down_block_cr(in_c, out_c, kern_s, pad, dropout_p=0.5):
    """ Encoding block used in the skull reconstruction models.

    :param in_c: Input channels (image).
    :param out_c: Output channels (feature maps).
    :param kern_s: Kernel size.
    :param pad: Padding used in convolutions.
    :param dropout_p: Dropout probability.
    :return: Initialized encoding block.
    """
    return nn.Sequential(nn.Conv3d(in_c, out_c, kernel_size=kern_s,
                                   padding=pad),
                         nn.BatchNorm3d(out_c),
                         nn.ReLU(True),
                         nn.Conv3d(out_c, out_c, kernel_size=kern_s,
                                   padding=pad),
                         nn.BatchNorm3d(out_c),
                         nn.ReLU(True),
                         nn.Dropout3d(dropout_p))


def up_block_cr(in_c, out_c, kern_s_conv, kern_s_uconv, pad, stride_uc,
                dropout_p=0.5):
    """ Decoding block used in the skull reconstruction models.

    :param in_c: Input channels (image).
    :param out_c: Output channels (feature maps).
    :param kern_s_conv: Kernel size used in convolutions.
    :param kern_s_uconv: Kernel size used in upconvolutions.
    :param pad: Padding used in convolutions.
    :param stride_uc: Stride used in upconvolutions.
    :param dropout_p: Dropout probability.
    :return: Initialized encoding block.
    """
    return nn.Sequential(nn.ConvTranspose3d(in_c, in_c,
                                            kernel_size=kern_s_uconv,
                                            stride=stride_uc),
                         nn.Conv3d(in_c, out_c, kernel_size=kern_s_conv,
                                   padding=pad),
                         nn.BatchNorm3d(out_c),
                         nn.ReLU(True),
                         nn.Conv3d(out_c, out_c, kernel_size=kern_s_conv,
                                   padding=pad),
                         nn.BatchNorm3d(out_c),
                         nn.ReLU(True),
                         nn.Dropout3d(dropout_p))


class recAE_v2_fixed(nn.Module):
    """ U-Net model consisting in 4 encoding/decoding blocks.
    """

    def __init__(self, input_channels=1, kern_sz_conv=5, kern_sz_upconv=2,
                 stride_upconv=2, i_size=8, padding=2, dropout_p=0,
                 use_checkpoint=True):
        """ Constructor of the recAE_v2_fixed Class.

        This is a fully convolutional model, so it can be trained with any
        input size (be aware that since it produce 4 max-poolings, the size
        should be multiple of 16)

        :param input_channels: Amount of channels in the input (binary image by
        default).
        :param kern_sz_conv: Kernel size of the convolutions.
        :param kern_sz_upconv: Kernel size of the upconvolutions.
        :param stride_upconv: Stride of the upconvolutions.
        :param i_size: Initial amount of feature maps.
        :param padding: Padding used in up/down convolutions
        :param dropout_p: Dropout p.
        """
        super(recAE_v2_fixed, self).__init__()

        self.chk = use_checkpoint

        fms = [i_size * pow(2, n) for n in range(5)]  # Feature maps sizes

        self.mp = nn.MaxPool3d(kernel_size=2, padding=0, stride=2, dilation=1,
                               return_indices=True)

        self.dblock1 = down_block_cr(input_channels, fms[0],
                                     kern_s=kern_sz_conv, pad=padding,
                                     dropout_p=dropout_p)
        self.dblock2 = down_block_cr(fms[0], fms[1], kern_s=kern_sz_conv,
                                     pad=padding, dropout_p=dropout_p)
        self.dblock3 = down_block_cr(fms[1], fms[2], kern_s=kern_sz_conv,
                                     pad=padding, dropout_p=dropout_p)
        self.dblock4 = down_block_cr(fms[2], fms[3], kern_s=kern_sz_conv,
                                     pad=padding, dropout_p=dropout_p)

        self.cblock_center = nn.Sequential(nn.Conv3d(fms[3], fms[4],
                                                     kernel_size=kern_sz_conv,
                                                     padding=padding),
                                           nn.BatchNorm3d(fms[4]),
                                           nn.ReLU(True),
                                           nn.Conv3d(fms[4], fms[4],
                                                     kernel_size=kern_sz_conv,
                                                     padding=padding),
                                           nn.BatchNorm3d(fms[4]),
                                           nn.ReLU(True),
                                           nn.Dropout3d(dropout_p))

        self.ublock1 = up_block_cr(fms[4], fms[3], kern_sz_conv,
                                   kern_sz_upconv, padding, stride_upconv,
                                   dropout_p)
        self.ublock2 = up_block_cr(2 * fms[3], fms[2], kern_sz_conv,
                                   kern_sz_upconv, padding, stride_upconv,
                                   dropout_p)
        self.ublock3 = up_block_cr(2 * fms[2], fms[1], kern_sz_conv,
                                   kern_sz_upconv, padding, stride_upconv,
                                   dropout_p)
        self.ublock4 = up_block_cr(2 * fms[1], fms[0], kern_sz_conv,
                                   kern_sz_upconv, padding, stride_upconv,
                                   dropout_p)

        self.last_conv = nn.Conv3d(2 * fms[0], 2, kernel_size=1)

    def forward(self, x):
        down1 = checkpoint(self.dblock1, x) if self.chk else self.dblock1(x)
        down1_mp, _ = self.mp(down1)
        down2 = checkpoint(self.dblock2,
                           down1_mp) if self.chk else self.dblock2(down1_mp)
        down2_mp, _ = self.mp(down2)
        down3 = checkpoint(self.dblock3,
                           down2_mp) if self.chk else self.dblock3(down2_mp)
        down3_mp, _ = self.mp(down3)
        down4 = checkpoint(self.dblock4,
                           down3_mp) if self.chk else self.dblock4(down3_mp)
        down4_mp, _ = self.mp(down4)

        cblock = checkpoint(self.cblock_center,
                            down4_mp) if self.chk else self.cblock_center(
            down4_mp)

        up1 = checkpoint(self.ublock1, cblock) if self.chk else self.ublock1(
            cblock)
        up1 = cat((up1, down4), 1)
        up2 = checkpoint(self.ublock2, up1) if self.chk else self.ublock2(up1)
        up2 = cat((up2, down3), 1)
        up3 = checkpoint(self.ublock3, up2) if self.chk else self.ublock3(up2)
        up3 = cat((up3, down2), 1)
        up4 = checkpoint(self.ublock4, up3) if self.chk else self.ublock4(up3)
        up4 = cat((up4, down1), 1)
        lc = checkpoint(self.last_conv, up4) if self.chk else self.last_conv(
            up4)

        return F.softmax(lc, dim=1)


class UNet4_2IC(recAE_v2_fixed):
    def __init__(self):
        """ Constructor of the recAE_v2_fixed Class. This is a fully
        convolutional model, so it can be trained with any
        input size (be aware that since it produce 4 max-poolings, the size
        should be multiple of 16)

        :param input_channels: Amount of channels in the input (binary image by
        default).
        :param kern_sz_conv: Kernel size of the convolutions.
        :param kern_sz_upconv: Kernel size of the upconvolutions.
        :param stride_upconv: Stride of the upconvolutions.
        :param i_size: Initial amount of feature maps.
        :param padding: Padding used in up/down convolutions
        :param dropout_p: Dropout p.
        """
        super(UNet4_2IC, self).__init__(i_size=7, input_channels=2)

# class UNet4b2i3oFC(UNetSP):
#     """ Three-channel output UNet with Shape Priors.
#
#     This model implements the U-Net with Shape priors (2 input channels)
#     with 3 output channels, with a FC layer in the middle. It isn't used
#     directly, since it's thought for predicting both the missing flap and
#     recosntructing the full skull.
#
#     This model is maintained independently for making UNetSP easier to
#     understand, since this is a classic U-Net, with a given set of parameters.
#
#     """
#
#     def __init__(self):
#         super(UNet4b2i3oFC, self).__init__(i_size=3, input_channels=2,
#                                            out_channels=3,
#                                            chk=False,
#                                            fc_layer=True)
#
#
# class UNetSPFC(UNet4b2i3oFC):
#     """ Unet with Shape Priors with a FC Layer in the middle.
#
#     This model uses a standard U-Net with two input channels and three
#     output channels for allowing the prediction of both the missing bone
#     flap and the full skull.
#
#     The output of this model takes the output of the three-channel UNet,
#     and indirectly forces that the channels 1 and 2 are the flap and full
#     skull, respectively (the channel 0 is the full skull background). This
#     is enforced in the corresponding Loss function (see the ProblemHandler
#     FlapRecWithShapePriorDoubleOut.comp_losses_metrics).
#
#     """
#
#     def __init__(self):
#         super(UNetSPFC, self).__init__()
#
#     def forward(self, x):
#         # U-Net forward pass
#         backg_flap_fullsk = super(UNetSPFC, self).forward(x)
#         backg = backg_flap_fullsk[:, 0:1]
#         flap = backg_flap_fullsk[:, 1:2]
#         fullsk = backg_flap_fullsk[:, 2:3]
#
#         encoded_full_skull = ccat((backg,
#                                          flap + fullsk),
#                                         1)
#         encoded_flap = ccat((1 - flap,
#                                    flap),
#                                   1)
#         return encoded_full_skull, encoded_flap
#
#
# class UNet4b2i3oNC(UNet):
#     """ Three-channel output UNet with Shape Priors.
#
#     This model implements the U-Net with Shape priors (2 input channels)
#     with 3 output channels. It isn't used directly, since it's thought for
#     predicting both the missing flap and recosntructing the full skull. In
#     UNetSP, it will take the output of this model and
#
#     This model is maintained independently for making UNetSP easier to
#     understand, since this is a classic U-Net, with a given set of parameters.
#
#     """
#
#     def __init__(self):
#         super(UNet4b2i3oNC, self).__init__(i_size=7, input_channels=2,
#                                            out_channels=3,
#                                            chk=True,
#                                            cat=False)
#
#
# class UNet4b2i3oNC3c(UNet):
#     def __init__(self):
#         super(UNet4b2i3oNC3c, self).__init__(i_size=7, input_channels=2,
#                                              out_channels=3,
#                                              chk=True,
#                                              cat=False, padding=1,
#                                              kern_sz_conv=3)
#
#
# class UNetSPNC3c(UNet4b2i3oNC3c):
#     def __init__(self):
#         super(UNetSPNC3c, self).__init__()
#
#     def forward(self, x):
#         # U-Net forward pass
#         backg_flap_fullsk = super(UNetSPNC3c, self).forward(x)
#         backg = backg_flap_fullsk[:, 0:1]
#         flap = backg_flap_fullsk[:, 1:2]
#         fullsk = backg_flap_fullsk[:, 2:3]
#
#         encoded_full_skull = ccat((backg,
#                                          flap + fullsk),
#                                         1)
#         encoded_flap = ccat((1 - flap,
#                                    flap),
#                                   1)
#         return encoded_full_skull, encoded_flap
#
#
# class UNet4b2i3oNC3cR(UNet):
#     def __init__(self):
#         super(UNet4b2i3oNC3cR, self).__init__(i_size=4, input_channels=2,
#                                               out_channels=3,
#                                               chk=True,
#                                               cat=False, padding=1,
#                                               kern_sz_conv=3,
#                                               residual=True)
#
#
# class UNetSPNC3cR(UNet4b2i3oNC3cR):
#     def __init__(self):
#         super(UNetSPNC3cR, self).__init__()
#
#     def forward(self, x):
#         # U-Net forward pass
#         backg_flap_fullsk = super(UNetSPNC3cR, self).forward(x)
#         backg = backg_flap_fullsk[:, 0:1]
#         flap = backg_flap_fullsk[:, 1:2]
#         fullsk = backg_flap_fullsk[:, 2:3]
#
#         encoded_full_skull = ccat((backg,
#                                          flap + fullsk),
#                                         1)
#         encoded_flap = ccat((1 - flap,
#                                    flap),
#                                   1)
#         return encoded_full_skull, encoded_flap
