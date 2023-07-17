import pathlib
import numpy as np

def merge_dataset() -> None:
    data_directory = pathlib.Path("/home/lorenzo/GazeDetection/GRA_Net/Output/train")
    image_count = len(list(data_directory.glob("*.npz")))

    output = None

    for file_path in data_directory.glob("*.npz"):
        try:
            data = np.load(file_path, allow_pickle = True)
            if output is None:
                output = data["arr_0"]
            else:
                output = np.concatenate((output, data["arr_0"]), axis = 0)
            data.close()
        except Exception as e:
            print(e)
            data.close()
            continue

    np.savez_compressed("/home/lorenzo/GazeDetection/GRA_Net/Output/Output", output)

def dataset_generator():
    with np.load("/home/lorenzo/GazeDetection/GRA_Net/Output/Output.npz", mmap_mode='r', allow_pickle = True) as data:
        for row in data["arr_0"]:
            yield row


if __name__ == "__main__":

    for row in dataset_generator():
        print(row)
        break
