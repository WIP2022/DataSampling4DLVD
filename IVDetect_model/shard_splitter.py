import os  # Used to do path manipulations
import shutil  # Used to copy files
import random


# Taken from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_abspath_files_in_directory(directory: str) -> list:
    """Takes in a directory and returns the abspath of all the files in the directory as a list

    Parameters
    ----------
    directory : str
        The directory to get the abspaths for

    Returns
    -------
    list
        A list of the abspaths of all the files in the directory
    """
    return [os.path.abspath(os.path.join(directory, path)) for path in os.listdir(directory)]


def split_to_subdirectories(file_paths: list, amount_per_folder: int):
    """Take in a list of file absolute paths, and copy them to folders

    Parameters
    ----------
    file_paths : list
        The list of abspaths to the file folders

    amount_per_folder : int
        The amount of files per folder to split the files into
    """
    file_paths = chunks(file_paths, amount_per_folder)

    for index, chunk in enumerate(file_paths):
        os.mkdir(str(index))  # Create a folder with a name of the current iteration index
        for file_path in chunk:
            file_name = file_path.split(os.sep)[-1]
            shutil.copy(file_path, os.path.join(str(index), file_name))


if __name__ == "__main__":
    file_paths = get_abspath_files_in_directory(
        "pyg_graph")  # Replace "original_folder" with the directory where your files are stored
    random.shuffle(file_paths)
    shard_size = int(len(file_paths) / 5)
    print(shard_size)
    split_to_subdirectories(file_paths, shard_size)