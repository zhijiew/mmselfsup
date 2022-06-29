import torch
import argparse
from mmseg.models.backbones import MobileNetV2

def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', type=str, help='checkpoint file')
    parser.add_argument('output', type=str, help='destination file name')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    assert args.output.endswith('.pth')

    src_checkpoint = torch.load(args.checkpoint)
    state_dict = src_checkpoint['state_dict']
    model = MobileNetV2()
    new_state_dict = {}
    
    for (key_src, key_tgt) in zip(state_dict.keys(), model.state_dict().keys()):
        new_state_dict[key_tgt] = state_dict[key_src]
        print('{} -> {}'.format(key_src, key_tgt))

    src_checkpoint['state_dict'] = new_state_dict

    torch.save(src_checkpoint, args.output)

if __name__ == '__main__':
    main()
