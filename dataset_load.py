import torch
import torchvision
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# when sample_idx is given: (Selectively use only the indexed samples)
# The remaining samples are returned by valloader
def dataset_loader(batch_size=128):

    testloader = None
    classes = ()
    root_path = '../'

    # directory for data, don't include root.
    validation_dataroot = "data/test"

    # Threads for dataloader
    workers = 2

    # Image manipulation Variables

    # Spatial size of training images.
    image_size = 36
    # image crop, matches ResNet
    image_crop = 32

    batch_size = 1

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_crop),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) 

    testset = dset.ImageFolder(root=root_path+validation_dataroot,transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=workers)

    classes = ('COVID-19', 'normal', 'pneumonia')
    return (testloader, classes)


def dataset_length(dataset_name):
    return 13892
