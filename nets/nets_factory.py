from nets import DSAM_alexnet
from nets import DSAM_resnet


nets_map = {
    'DSAM_alexnet': DSAM_alexnet.DSAM_alexnet,
    'DSAM_resnet18': DSAM_resnet.DSAM_resnet18,
    'deepall_alexnet': DSAM_alexnet.deepall_alexnet,
    'deepall_resnet18': DSAM_resnet.deepall_resnet18
    }

def get_network(name):

    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(num_classes, **kwargs):
        return nets_map[name](num_classes, **kwargs)

    return get_network_fn
