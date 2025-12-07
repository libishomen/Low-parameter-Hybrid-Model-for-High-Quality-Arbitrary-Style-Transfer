import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
# from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import itertools
from torchvision.utils import save_image
import net 
from sampler import InfiniteSamplerWrapper
from discrimination import MultiDiscriminator

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 该模型为完整的模型代码，非消融实验

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default="../image/content",
                    help='Directory path to COCO2014 data-set')
parser.add_argument('--style_dir', type=str, default="../image/style",
                    help='Directory path to Wikiart data-set')
parser.add_argument('--vgg', type=str, default='model_path/vgg_normalised.pth')

# training options
parser.add_argument('--training_mode', default='art',
                    help='Artistic or Photo-realistic')
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=320000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--ccp_weight', type=float, default=5.0)
parser.add_argument('--remd_weight', type=float, default=10.0)
parser.add_argument('--re_weight', type=float, default=50.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--tau', type=float, default=0.07)
parser.add_argument('--num_s', type=int, default=8, help='number of sampled anchor vectors')
parser.add_argument('--num_l', type=int, default=3, help='number of layers to calculate CCPL')
parser.add_argument('--gpu', type=int, default=1, help='which gpu to use')
args = parser.parse_args()

device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)

vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, args.training_mode)
net_D = MultiDiscriminator()
network.train()
network.to(device)
net_D.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(itertools.chain(network.style_encoder.parameters(), network.content_encoder.parameters(),
                                             network.decoder.parameters(), network.SCT.parameters(), network.mlp.parameters()), lr=args.lr)

optimizerD = torch.optim.Adam(net_D.parameters(), lr=args.lr)

valid = 1
fake = 0

parent_dir = "gener_style_image"

for i in tqdm(range(args.max_iter)):
    if i < 1e4:
        warmup_learning_rate(optimizer, iteration_count=i)  # 学习率的调整
    else:
        adjust_learning_rate(optimizer, iteration_count=i)

    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    gener, loss_c, loss_s, loss_ccp, loss_color = network(content_images, style_images, args.num_s, args.num_l)

    # 训练判别器
    optimizerD.zero_grad()
    d_loss = net_D.compute_loss(style_images, valid) + net_D.compute_loss(gener.detach(), fake)
    d_loss.backward()
    optimizerD.step()

    # 训练生成器
    optimizer.zero_grad()
    g_loss = net_D.compute_loss(gener, valid)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss_ccp = args.ccp_weight * loss_ccp

    loss = loss_c + loss_s + loss_ccp + g_loss + loss_color
    loss.backward()
    optimizer.step()

    if i % 200 == 0:
        output_name = '{:s}/{:s}{:s}'.format(
            parent_dir, str(i), ".jpg"
        )
        out = torch.cat((content_images, gener), 2)
        out = torch.cat((style_images, out), 2)
        save_image(out, output_name)

    # 模型保存
    save_dir = "experiments"
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(save_dir, i))

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.SCT.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/SCT_iter_{:d}.pth'.format(save_dir, i))

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.style_encoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/style_encoder_iter_{:d}.pth'.format(save_dir, i))

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.content_encoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/content_encoder_iter_{:d}.pth'.format(save_dir, i))
# writer.close()
