from pathlib import Path
import argparse
import mxnet as mx
from tqdm import tqdm
from PIL import Image
import bcolz
import pickle
import cv2
import numpy as np
from torchvision import transforms as trans
import os
import numbers
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


def save_rec_to_img_dir(dataset_dir, swap_color_channel=False, save_as_png=False):
    def save_img_one(imgrec, idx, lock, swap_color_channel, save_path):
        with lock:
            img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        if not isinstance(header.label, numbers.Number):
            label = int(header.label[0])
        else:
            label = int(header.label)

        label_path = save_path / str(label)
        if not label_path.exists():
            label_path.mkdir()
        img_save_path = label_path / "{}.jpg".format(idx)
        # if os.path.exists(img_save_path):
        #     return

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if swap_color_channel:
            # this option saves the image in the right color.
            # but the training code uses PIL (RGB)
            # and validation code uses Cv2 (BGR)
            # so we want to turn this off to deliberately swap the color channel order.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img)
        img.save(img_save_path, quality=100)


    save_path = os.path.join(dataset_dir, "imgs")
    if not os.path.exists(save_path):
        os.path.mkdir(save_path)
    imgrec = mx.recordio.MXIndexedRecordIO(
        os.path.join(dataset_dir, "train.idx"), 
        os.path.join(dataset_dir, "train.rec"), "r"
    )
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])

    executor = ThreadPoolExecutor()
    task_lis = []
    lock = Lock()
    for idx in tqdm(range(1, max_idx)):
        task = executor.submit(save_img_one, imgrec, idx, lock, swap_color_channel, save_path)
        task_lis.append(task)
    for i in tqdm(as_completed(task_lis), total=len(task_lis), desc='rec2img'):
        pass

        # save_img_one(imgrec, idx, lock, swap_color_channel, save_path)

    # for idx in tqdm(range(1, max_idx)):
    #     img_info = imgrec.read_idx(idx)
    #     header, img = mx.recordio.unpack_img(img_info)

def shuffle_image_rec(rec_path):
    imgrec = mx.recordio.MXIndexedRecordIO(
        os.path.join(rec_path, "train.idx"), 
        os.path.join(rec_path, "train.rec"), "r"
    )
    def repack_one(rec_in, i, idx, lock, rec_out):
        with lock[0]:
            img_info = rec_in.read_idx(idx)
            img_info.header.id2 = idx
        # with lock[1]:
            rec_out.write_idx(i, img_info)
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    chunk_size = 50000000
    chunk_num = max_idx // chunk_size + 1
    executor = ThreadPoolExecutor()
    task_lis = []
    lock_read = Lock()
    lock_lis = [[lock_read, Lock()]] * chunk_num
    rec_lis = [mx.recordio.MXIndexedRecordIO(
        os.path.join(rec_path, f'train{i}.idx'),
        os.path.join(rec_path, f'train{i}.rec'), 'w') for i in range(chunk_num)]

    idx_list = list(range(1, max_idx))
    np.random.shuffle(idx_list)
    for i, idx in tqdm(enumerate(idx_list), total=len(idx_list)):
        chunk_id = idx // chunk_size
        lock = lock_lis[chunk_id]
        rec = rec_lis[chunk_id]

        task = executor.submit(repack_one, imgrec, i, idx, lock, rec)
        task_lis.append(task)
    for i in tqdm(as_completed(task_lis), total=len(task_lis), desc='shuffle img rec'):
        pass
        # repack_one(imgrec, i, idx, lock, rec)


def get_all_files(root, extension_list=[".jpg", ".png", ".jpeg"]):
    all_files = list()
    for dirpath, dirnames, filenames in os.walk(root):
        all_files += [os.path.join(dirpath, file) for file in filenames]
    if extension_list is None:
        return all_files
    all_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, all_files))
    return all_files

def repack_rec(dataset_dir, lab_func=None, dataset_name='repack'):
    save_rec_to_img_dir(dataset_dir, swap_color_channel=False, save_as_png=False)
    if lab_func is None:
        lab_func = lambda x: int(x.split('/')[-2])
    image_paths = get_all_files(os.path.join(dataset_dir, 'imgs'))
    make_random_rec(image_paths, dataset_dir, lab_func, dataset_name)

def make_random_rec(img_paths, dataset_dir, lab_func, dataset_name, is_origin=False):
    np.random.shuffle(img_paths)
    make_rec(img_paths, dataset_dir, lab_func, dataset_name, is_origin)

def make_rec(img_paths, store_path, lab_func, dataset_name, is_origin=True):
    def encode_one(img_pth, i, lab_func, record, lock=None, origin=True):
        if lab_func is not None:
            label = lab_func(img_pth)
        else:
            label = i
        header = mx.recordio.IRHeader(0, label, i, 0)

        if origin:
            with open(img_pth, 'rb') as f:
                img = f.read()
            s = mx.recordio.pack(header, img)
        else:
            img = cv2.imread(img_pth).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            s = mx.recordio.pack_img(header, img, quality=100, img_fmt='.jpg')
        if lock is not None:
            with lock:
                record.write_idx(i, s)
        return s

    executor = ThreadPoolExecutor()
    task_lis = []
    lock = Lock()
    record = mx.recordio.MXIndexedRecordIO(
        os.path.join(store_path, f'{dataset_name}.idx'),
        os.path.join(store_path, f'{dataset_name}.rec'), 'w')

    for i, img_pth in tqdm(enumerate(img_paths), total=len(img_paths)):
        task = executor.submit(encode_one, img_pth, i, lab_func, record, lock, is_origin)
        task_lis.append(task)
    for i in tqdm(as_completed(task_lis), total=len(task_lis), desc='make rec'):
        pass
    record.close()

def load_bin(path, rootdir, save_imgs=False, image_size=[112, 112]):
    def decode_one(_bin, data, test_transform):
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if save_imgs:
            cv2.imwrite(os.path.join(rootdir, 'imgs', f'{i}.jpg'), img)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = test_transform(img)

    test_transform = trans.Compose(
        [trans.ToTensor(), trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, "rb"), encoding="bytes")
    data = bcolz.fill(
        [len(bins), 3, image_size[0], image_size[1]],
        dtype=np.float32,
        rootdir=rootdir,
        mode="w",
    )
    if save_imgs:
        if not os.path.exists(os.path.join(rootdir, 'imgs')):
            os.mkdir(os.path.join(rootdir, 'imgs'))
        for i in tqdm(range(len(bins))):
            decode_one(bins[i], data, test_transform)
    else:
        executor = ThreadPoolExecutor(max_workers=5)
        task_lis = []
        for i in tqdm(range(len(bins))):
            task = executor.submit(decode_one, bins[i], data, test_transform)
            task_lis.append(task)

        for i in tqdm(as_completed(task_lis), total=len(task_lis)):
            pass
    print(data.shape)
    np.save(str(rootdir) + "_list", np.array(issame_list))
    return data, issame_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="for face verification")
    parser.add_argument(
        "-r",
        "--rec_path",
        help="mxnet record file path",
        default="./faces_emore",
        type=str,
    )
    parser.add_argument("--make_image_files", action="store_true")
    parser.add_argument("--make_validation_memfiles", action="store_true")
    parser.add_argument("--swap_color_channel", action="store_true")

    args = parser.parse_args()
    rec_path = Path(args.rec_path)
    if args.make_image_files:
        # unfolds train.rec to image folders
        save_rec_to_img_dir(
            rec_path, swap_color_channel=args.swap_color_channel
        )

    if args.make_validation_memfiles:
        # for saving memory usage during training
        # bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
        bin_files = list(
            filter(
                lambda x: os.path.splitext(x)[1] in [".bin"],
                os.listdir(args.rec_path),
            )
        )
        bin_files = [i.split(".")[0] for i in bin_files]

        for i in range(len(bin_files)):
            print(f"Loading {bin_files[i]}")
            load_bin(
                rec_path / (bin_files[i] + ".bin"), rec_path / bin_files[i]
            )
