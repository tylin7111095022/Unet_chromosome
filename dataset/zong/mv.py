import shutil
import os
def main():
    target_dir = "train_mask"
    source_dir = "255"
    fs = os.listdir(source_dir)
    mv_fs = os.listdir("train2017")

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for f in mv_fs:
        shutil.move(os.path.join(source_dir, f), os.path.join(target_dir, f))


if __name__ == "__main__":
    main()