from trainings import DSAM_training

trainings_map = {
    'DSAM_training': DSAM_training
}

def get_training(name):
    if name not in trainings_map:
        raise ValueError('Name of training unknown %s' % name)

    def get_training_fn(model, **kwargs):
        return trainings_map[name].get_training(model, **kwargs)

    return get_training_fn
