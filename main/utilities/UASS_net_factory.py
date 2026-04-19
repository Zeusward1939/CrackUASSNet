
from .UASS_unet import UNet, UNet_UASS


def net_factory(net_type="U-Net_UASS", in_chns=3, class_num=4):
    if net_type == "U-Net":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    
    elif net_type == "U-Net_UASS":
        net = UNet_UASS(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net
