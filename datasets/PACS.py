import os
from torchvision import datasets


PACS_domains = ['art_painting', 'cartoon', 'sketch', 'photo']
PACS_classes = ['dog', 'horse', 'house', 'person', 'guitar', 'gyraffe', 'elephant']


def make_PACS_train_val_splits(dataset_dir, domain):
    target = os.path.join(dataset_dir, 'pytorch', domain)
    os.mkdir(target)
    for d in ['train', 'val']:
        os.mkdir(os.path.join(target, d))

    for label in os.listdir(os.path.join(dataset_dir, domain)):

        for d in ['train', 'val']:
            os.mkdir(os.path.join(target, d, label))
        
        filenames = os.listdir(os.path.join(dataset_dir, domain, label))
        filenames = map(lambda x: os.path.join(dataset_dir, domain, label, x), filenames)

        for i, item in enumerate(filenames):
            if i % 10 == 0:
                os.symlink(item, os.path.join(target, 'val', label, item.split('/')[-1]))

            else:
                os.symlink(item, os.path.join(target, 'train', label, item.split('/')[-1]))


def make_PACS_splits(dataset_dir):
    os.mkdir(os.path.join(dataset_dir, 'pytorch'))
    for domain in PACS_domains:
        make_PACS_train_val_splits(dataset_dir, domain)


def get_dataset(dataset_dir, data_transforms, test_domain):

    assert test_domain in PACS_domains

    train_domains = list(filter(lambda x: x != test_domain, PACS_domains))
    assert len(train_domains)==3

    if not os.path.exists(os.path.join(dataset_dir, 'pytorch')):
        make_PACS_splits(dataset_dir)

    data_dirs = {
        'art_painting': {},
        'cartoon': {},
        'photo': {},
        'sketch': {}
    }
    
    for d in train_domains:
        for split in ['train','val']:
            data_dirs[d][split] = os.path.join(dataset_dir, 'pytorch', d, split)

    data_dirs[test_domain]['test'] = os.path.join(dataset_dir, test_domain)

    image_datasets = {x: [datasets.ImageFolder(data_dirs[d][x], data_transforms[x]) for d in domains]
                      for x, domains in zip(['train', 'val', 'test'], [train_domains, train_domains, [test_domain]])}

    return image_datasets, PACS_classes
