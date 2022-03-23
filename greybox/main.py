# UCF Video Label Extraction using SwinT Network
# Author: Vivek Mittal


from ast import parse
import torch
import torchvision
from torch import nn
from ucf_dataset import UCF101
from torch.utils.data import DataLoader
import argparse
from args import common_args
from swin_transformer.video_swin_transformer import SwinTransformer3D
import pickle as pkl


class SwinT(nn.Module):
    def __init__(self, swint_path) -> None:
        super(SwinT).__init__(self)
        self.model = SwinTransformer3D(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            patch_size=(2, 4, 4),
            window_size=(8, 7, 7),
            drop_path_rate=0.4,
            patch_norm=True,
        )
        self.model.load_state_dict(torch.load(swint_path))

    def forward(self, x):
        """
        :param x: shape (bs, channels, time, H, W)
        """
        out = self.model(x)
        return out


parser = argparse.ArgumentParser()
common_args(parser)
args = parser.parse_args()

ucf_video_path = ""
ucf_anno_path = ""
swint_path = ""

dataset = UCF101(
    ucf_video_path, ucf_anno_path, args.frames_per_clip, args.step_between_clips
)
dataloader = DataLoader(dataset, args.bs)
device = torch.cuda()

model = SwinT(swint_path)
model.to(device)


@torch.no_grad()
def main():
    swin_labels = []
    video_names = []
    for data in dataloader:
        video, names, _, _ = data
        video = video.to(device)
        output = model(video)
        output = output.cpu().numpy()
        swin_labels.append(output)
        video_names.append(names)

    with open("swin_ucf_labels.pkl", "wb") as f:
        pkl.dump({"names": video_names, "labels": swin_labels}, f)
