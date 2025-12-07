import argparse
from pathlib import Path

import time
import numpy as np
import thop
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from eval_artfid import compute_art_fid
import net as net
import torchvision.models as models
import traceback
from scipy.linalg import sqrtm
from assessment_criteria import losses

test_transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.ToTensor(),
])


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str, default="inputs/content",
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image')
parser.add_argument('--style_dir', type=str, default="inputs/style",
                    help='Directory path to a batch of style images')
parser.add_argument('--content_encoder', type=str, default='experiments/content_encoder_iter_249999.pth')
parser.add_argument('--style_encoder', type=str, default='experiments/style_encoder_iter_249999.pth')
# parser.add_argument('--modulator', type=str, default='experiments/modulator_iter_119999.pth')
parser.add_argument('--decoder', type=str, default='experiments/decoder_iter_249999.pth')
parser.add_argument('--style_transfer', type=str, default='experiments/SCT_iter_249999.pth')

# Additional options    Q
parser.add_argument('--content_size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=0,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='outputs',
                    help='Directory to save the output image(s)')
parser.add_argument('--gpu_id', type=int, default=0)

# Advanced options
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')

args = parser.parse_args()

device = torch.device('cuda:%d' % args.gpu_id)

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = [Path(args.style)]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

content_encoder = net.NextVit()
style_encoder = net.NextVit()
decoder = net.decoder()
style_transfer = net.style_transfer(256)

content_encoder.eval()
style_encoder.eval()
decoder.eval()

content_encoder.load_state_dict(torch.load(args.content_encoder))
style_encoder.load_state_dict(torch.load(args.style_encoder))
decoder.load_state_dict(torch.load(args.decoder))
style_transfer.load_state_dict(torch.load(args.style_transfer))
network = net.TestNet(content_encoder, style_encoder, style_transfer, decoder)
network.to(device)

content_tf = test_transform
style_tf = test_transform

pj = losses()
i = 0
Lc = 0
Ls = 0
lpips = 0
psnr = 0
ssim = 0
t = 0
f = 0.0
p = 0.0

import torch
import numpy as np
from scipy import linalg
from torchvision import models, transforms
from PIL import Image

# 加载InceptionV3模型并去掉最后一层分类层
model = models.inception_v3(pretrained=True, transform_input=False)
model.fc = torch.nn.Identity()  # 将最后的全连接层替换为恒等映射
model.eval()  # 设置为评估模式

# 图像预处理函数
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 提取单张图像的特征
def get_feature(img_path, model, device):
    model.to(device)
    with torch.no_grad():
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        feature = model(img)
    return feature.cpu().numpy().flatten()

# 计算均值和协方差
def calculate_statistics(feature):
    mu = feature
    sigma = np.outer(feature - mu, feature - mu)  # 协方差矩阵
    return mu, sigma

# 计算FID
def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for content_path in content_paths:
    for style_path in style_paths:
        try:
            content = content_tf(Image.open(str(content_path)).convert("RGB"))
            style = style_tf(Image.open(str(style_path)).convert("RGB"))

            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)

            with torch.no_grad():
                flops, params = thop.profile(network, inputs=(content, style))
                tic = time.time()
                output = network(content, style)
                end = time.time()
                t = t + (end - tic)
                c, s, l, _, sim = pj(content, style, output)
                Lc += c
                Ls += s
                lpips += l
                f += flops
                p += params
                i = i + 1

            output = output.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            save_image(output, str(output_name))

            # 获取每张图片的特征向量
            feature_style = get_feature(style_path, model, device)
            feature_generated = get_feature(output_name, model, device)

            # 计算两张图片的均值和协方差
            mu_style, sigma_style = calculate_statistics(feature_style)
            mu_generated, sigma_generated = calculate_statistics(feature_generated)

            # 计算FID分数
            fid_score = calculate_fid(mu_style, sigma_style, mu_generated, sigma_generated)
            print("fid:", fid_score)

        except:
            traceback.print_exc()

print("loss_c:", Lc/i)
print("loss_s:", Ls/i)
print("lpips:", lpips/i)
print("time:", t/i)
print("flops:", f/1e9/i)
print("params:", p/1e6/i)
print("finish")
