import os
from torchvision import datasets


OH_domains = ['Art', 'Clipart', 'Product', 'Real World']
OH_classes = ['Fan', 'Speaker', 'Folder', 'Ruler', 'Paper_Clip', 'Eraser',
              'Calendar', 'Exit_Sign', 'Spoon', 'Pencil', 'Batteries', 'Bed',
              'Drill', 'Bike', 'Push_Pin', 'Sneakers', 'Desk_Lamp', 'Chair',
              'Mop', 'Pan', 'Clipboards', 'Knives', 'Candles', 'Bottle',
              'Refrigerator', 'Screwdriver', 'Trash_Can', 'Oven', 'Telephone',
              'Mug', 'Sink', 'Shelf', 'ToothBrush', 'Pen', 'Table', 'TV',
              'Lamp_Shade', 'Bucket', 'Flipflops', 'Webcam', 'Hammer', 'Marker',
              'Computer', 'Soda', 'Calculator', 'Laptop', 'Printer',
              'File_Cabinet', 'Toys', 'Kettle', 'Notebook', 'Radio', 'Helmet',
              'Fork', 'Curtains', 'Keyboard', 'Alarm_Clock', 'Flowers', 'Couch',
              'Monitor', 'Postit_Notes', 'Backpack', 'Glasses', 'Mouse', 'Scissors']


def make_OH_train_val_splits(dataset_dir, domain):
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


def make_OH_splits(dataset_dir):
    os.mkdir(os.path.join(dataset_dir, 'pytorch'))
    for domain in OH_domains:
        make_OH_train_val_splits(dataset_dir, domain)


def get_dataset(dataset_dir, data_transforms, test_domain):

    assert test_domain in OH_domains

    train_domains = list(filter(lambda x: x != test_domain, OH_domains))

    if not os.path.exists(os.path.join(dataset_dir, 'pytorch')):
        make_OH_splits(dataset_dir)

    data_dirs = {
        'Art': {},
        'Clipart': {},
        'Product': {},
        'Real World': {}
    }
    
    for d in train_domains:
        for split in ['train','val']:
            data_dirs[d][split] = os.path.join(dataset_dir, 'pytorch', d, split)

    data_dirs[test_domain]['test'] = os.path.join(dataset_dir, test_domain)

    image_datasets = {x: [datasets.ImageFolder(data_dirs[d][x], data_transforms[x]) for d in domains]
                      for x, domains in zip(['train', 'val', 'test'], [train_domains, train_domains, [test_domain]])}

    return image_datasets, OH_classes
