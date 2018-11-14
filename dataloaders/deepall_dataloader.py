from torch.utils.data import DataLoader

def get_dataloader(image_datasets, batch_size, num_workers):

    dataloaders = {x: [DataLoader(image_dataset,
                                  batch_size=batch_size,
                                  shuffle=x=='train',
                                  num_workers=num_workers,
                                  drop_last=x=='train')
                       for image_dataset in image_datasets[x]]
                   for x in ['train', 'val', 'test']}

    return dataloaders
