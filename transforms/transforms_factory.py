from transforms import DSAM_transforms

def get_transforms(name):

    transforms_fn_map = {
        'DSAM_transforms': DSAM_transforms
    }

    if name not in transforms_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)

    def get_transforms_fn(output_size, **kwargs):
        return transforms_fn_map[name].get_transforms(output_size, **kwargs)

    return get_transforms_fn
