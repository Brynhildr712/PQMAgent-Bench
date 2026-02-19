import os


def list_folders_and_files():
    """分开打印文件夹和文件"""
    print("=== 文件夹列表 ===")
    folders = []

    print("=== 文件列表 ===")
    files = []

    with os.scandir('.') as entries:
        for entry in entries:
            if entry.is_dir():
                folders.append(entry.name)
            elif entry.is_file():
                files.append(entry.name)

    # 打印文件夹
    if folders:
        for i, folder in enumerate(sorted(folders), 1):
            print(f"{folder}/")
    else:
        print("当前目录下没有文件夹")

    print("\n" + "=" * 30 + "\n")

    # 打印文件
    if files:
        for i, file in enumerate(sorted(files), 1):
            print(f"{file}")
    else:
        print("当前目录下没有文件")


if __name__ == "__main__":
    list_folders_and_files()