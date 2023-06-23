import pathlib
import torch

from data.dataloader import create_data_loader
from model.models import create_model
from utils.utils import load_configs, compute_angle_error


def test(model, test_loader, config):
    print('Started test')

    model.eval()

    device = torch.device(config['device'])

    predictions = []
    gts = []
    with torch.no_grad():
        for images, poses, gazes in enumerate(test_loader):
            images = images.to(device)
            # poses = poses.to(device)
            gazes = gazes.to(device)

            outputs = model(images)

            predictions.append(outputs.cpu())
            gts.append(gazes.cpu())

    predictions = torch.cat(predictions)
    gts = torch.cat(gts)
    angle_error = float(compute_angle_error(predictions, gts).mean())
    return predictions, gts, angle_error


def main():
    config = load_configs(is_train=False)

    test_loader = create_data_loader(config, is_train=False)
    print('Initialized test loader')
    model = create_model(config)
    checkpoint_dir = pathlib.Path(config['test']['checkpoint_dir'])
    checkpoint_name = config['test']['checkpoint']
    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f'Initialized model "{model.name}" with checkpoint "{checkpoint_name}"')

    predictions, gts, angle_error = test(model, test_loader, config)

    print(f'The mean angle error (deg): {angle_error:.2f}')


if __name__ == '__main__':
    main()
