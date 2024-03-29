import datetime
import pathlib
import torch
from fvcore.common.checkpoint import Checkpointer

from data.dataloader import create_data_loader
from model.models import create_model, create_loss, create_optimizer, create_scheduler
from utils.utils import load_configs, set_seeds, setup_cudnn, AverageMeter, compute_angle_error


def train(epoch, model, optimizer, loss_function, train_loader, config):
    print(f'Started training (epoch {epoch})')

    model.train()

    device = torch.device(config['device'])

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()

    for step, (images, poses, gazes) in enumerate(train_loader):
        images = images.to(device)
        # gazes are 2D gaze angle vectors
        gazes = gazes.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(outputs, gazes)
        loss.backward()

        optimizer.step()

        angle_error = compute_angle_error(outputs, gazes).mean()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        angle_error_meter.update(angle_error.item(), batch_size)

        if step % config['train']['log_period'] == 0:
            print(f'[Epoch {epoch}]'
                  f'[Step {step}/{len(train_loader)}]: '
                  f'loss {loss_meter.avg:.4f} '
                  f'angle error {angle_error_meter.avg:.2f}')


def validate(epoch, model, loss_function, val_loader, config):
    print(f'Started validation (epoch {epoch})')

    model.eval()

    device = torch.device(config['device'])

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()

    with torch.no_grad():
        for step, (images, poses, gazes) in enumerate(val_loader):
            images = images.to(device)
            gazes = gazes.to(device)

            outputs = model(images)

            loss = loss_function(outputs, gazes)
            angle_error = compute_angle_error(outputs, gazes).mean()

            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            angle_error_meter.update(angle_error.item(), batch_size)

    print(f'Epoch {epoch} --> loss {loss_meter.avg:.4f} angle error {angle_error_meter.avg:.2f}')


def main():
    config = load_configs(is_train=True)
    if config['device'] == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        config['device'] = device
    print('Running on device: {}'.format(config['device']))
    output_dir = pathlib.Path(config['output']['dir'])
    set_seeds()
    setup_cudnn()
    print(f'Using output dir: {output_dir}')

    train_loader, val_loader = create_data_loader(config, is_train=True)
    print('Initialized train and validation loaders')
    model = create_model(config)
    print(f'Initialized model: {model.name}')
    loss_function = create_loss(config)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    print(f'Initialized loss function, optimizer and scheduler')
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=str(output_dir),
                                save_to_disk=True)

    # validate(0, model, loss_function, val_loader, config)

    for epoch in range(1, config['scheduler']['epochs'] + 1):
        train(epoch, model, optimizer, loss_function, train_loader, config)
        validate(epoch, model, loss_function, val_loader, config)

        if epoch % config['train']['checkpoint_period'] == 0 or epoch == config['scheduler']['epochs']:
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
            checkpoint_name = f'trained_model_{timestamp}'
            checkpoint_config = {'epoch': epoch, 'config': config}
            checkpointer.save(checkpoint_name, **checkpoint_config)


if __name__ == '__main__':
    main()
