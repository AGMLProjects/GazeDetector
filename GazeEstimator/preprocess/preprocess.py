import pathlib
import h5py
import numpy as np
import tqdm


def save_subject(subject_id: str, dataset_dir: pathlib.Path, output_path: pathlib.Path) -> None:
    with h5py.File(dataset_dir / f'{subject_id}.mat', 'r') as input:
        images = input.get('Data/data')[()]
        labels = input.get('Data/label')[()][:, :4]

    images = images.transpose(0, 2, 3, 1).astype(np.uint8)
    poses = labels[:, 2:]
    gazes = labels[:, :2]

    with h5py.File(output_path, 'a') as output:
        for index, (image, gaze, pose) in tqdm.tqdm(enumerate(zip(images, gazes, poses)), leave=False):
            output.create_dataset(f'{subject_id}/image/{index:04}', data=image)
            output.create_dataset(f'{subject_id}/pose/{index:04}', data=pose)
            output.create_dataset(f'{subject_id}/gaze/{index:04}', data=gaze)


def main():
    print('Preprocessing started')
    output_dir = pathlib.Path('../dataset')
    output_path = output_dir / 'dataset.h5'
    print(f'Output path: {output_path}')
    if output_path.exists():
        raise ValueError(f'{output_path} already exists')

    dataset_dir = pathlib.Path('../dataset/MPIIFaceGaze')
    for subject_id in range(0, 15):
        subject_id = f'p{subject_id:02}'
        print(f'Saving subject {subject_id}')
        save_subject(subject_id, dataset_dir, output_path)


if __name__ == '__main__':
    main()
