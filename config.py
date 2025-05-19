import torch

gpu_name = 'cuda:6'
device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
root_dir = './data/HGG'

T = 4
num_epochs = 120
batch_size = 2
k_folds = 2
class_weights = torch.tensor([0.05, 0.3, 0.3, 0.35], dtype=torch.float32).to(device)

num_workers = 4

compute_hd = False

base_lr = 5e-4
num_warmup_epochs = 5
min_lr=1e-6