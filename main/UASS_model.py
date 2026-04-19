import torch.nn as nn
from utilities.UASS_net_factory import net_factory

model = net_factory(net_type='U-Net_UASS', in_chns=3, class_num=2)
model = nn.DataParallel(model) 