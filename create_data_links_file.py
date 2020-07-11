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
            links_file.write(full_path)


if __name__ == "__main__":
    create_links()
