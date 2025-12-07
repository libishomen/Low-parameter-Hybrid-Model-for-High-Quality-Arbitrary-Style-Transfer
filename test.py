import argparse
from pathlib import Path

import time

import thop
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net as net

import traceback

from assessment_criteria import losses

test_transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.ToTensor(),
])


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str, default="trade/content",
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image')
parser.add_argument('--style_dir', type=str, default="trade/style",
                    help='Directory path to a batch of style images')
parser.add_argument('--content_encoder', type=str, default='experiments/content_encoder_iter_249999.pth')
parser.add_argument('--style_encoder', type=str, default='experiments/style_encoder_iter_249999.pth')
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
parser.add_argument('--output', type=str, default='trade_output/0_0',
                    help='Directory to save the output image(s)')
parser.add_argument('--gpu_id', type=int, default=1)

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

alpha = 0.0
for content_path in content_paths:
    for style_path in style_paths:
        try:
            content = content_tf(Image.open(str(content_path)).convert("RGB"))
            style = style_tf(Image.open(str(style_path)).convert("RGB"))

            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)

            with torch.no_grad():
                flops, params = thop.profile(network, inputs=(content, style))
                output = network(content, style)
                feat_cc = network(content, content)
                output = output * alpha + feat_cc * (1 - alpha)

            output = output.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            save_image(output, str(output_name))
        except:
            traceback.print_exc()

print("finish")