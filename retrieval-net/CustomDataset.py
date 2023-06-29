import os

from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths, self.targets = self.get_image_paths_and_targets()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        target = self.targets[idx]
        return image_path, image, target

    def get_image_paths_and_targets(self):
        image_paths = []
        targets = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.jpg'):
                    image_paths.append(os.path.join(root, file))
                    targets.append((int(file.split('_')[1]), int(file.split('_')[0])))
        return image_paths, targets

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        # We could add some preprocessing steps if needed
        # For example:
        # image = image.resize((256, 256))  # Resize the image
        # image = some_other_preprocessing(image)
        return image
