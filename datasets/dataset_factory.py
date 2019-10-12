from datasets import PACS
from datasets import OfficeHome

datasets_map = {
    'PACS': PACS.get_dataset,
    'OfficeHome': OfficeHome.get_dataset,
}

def get_dataset(name):
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)

    def get_dataset_fn(dataset_dir, **kwargs):
        return datasets_map[name](dataset_dir, **kwargs)

    return get_dataset_fn
