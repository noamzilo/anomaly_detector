from Utils.config.ConfigProvider import ConfigProvider
import os


def create_links():
    config = ConfigProvider.config()
    data_dir = config.data.folder

    full_path_list = []
    assert os.path.isdir(data_dir)
    for _, _, filenames in os.walk(data_dir):
        for filename in filenames:
            full_path_list.append(os.path.join(data_dir, filename))

    with open(config.data.links_file, 'w+') as links_file:
        for full_path in full_path_list:
            links_file.write(f"{full_path}\n")


def read_links_file_to_list():
    config = ConfigProvider.config()
    links_file_path = config.data.links_file
    if not os.path.isfile(links_file_path):
        raise RuntimeError("did you forget to create a file with links to images? Try using 'create_links()'")
    with open(links_file_path, 'r') as links_file:
        return links_file.readlines()


if __name__ == "__main__":
    create_links()
    for filename in read_links_file_to_list():
        print(filename)
