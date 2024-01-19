import sys
import os

sys.path.append("/workspace/QGFace")
from validation.validation_lq import (
    tinyface_helper,
    data_utils,
    validate_tinyface,
)
from validation.validation_mixed import validate_IJB_BC
from validation.validation_mixed import insightface_ijb_helper
import numpy as np
from functools import partial


def val_tiny(model, val_cfg):
    # profiler = Profiler()
    # profiler.start()
    model.to(f"cuda:{val_cfg.gpu_id}")
    tinyface_test = tinyface_helper.TinyFaceTest(
        tinyface_root=val_cfg.tinyface.data_root,
        alignment_dir_name=val_cfg.tinyface.aligned_dir,
    )
    # set save root
    save_path = os.path.join(
        "./tinyface_result", val_cfg.model, val_cfg.tinyface.fusion
    )
    img_paths = tinyface_test.image_paths

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("save_path: {}".format(save_path))
    print("total images : {}".format(len(img_paths)))
    dataloader = data_utils.prepare_dataloader(
        img_paths, val_cfg.batch_size, num_workers=val_cfg.num_workers
    )
    features, norms = validate_tinyface.infer(
        model,
        dataloader,
        gpu_id=val_cfg.gpu_id,
        use_flip_test=val_cfg.tinyface.use_flip_test,
        fusion_method=val_cfg.tinyface.fusion,
    )
    results = tinyface_test.test_identification(features, ranks=[1, 5, 20])
    print(results)


def val_IJB_subset(model, val_cfg, subset_name):
    save_path = f"./IJB_{subset_name}_result"
    os.makedirs(save_path, exist_ok=True)
    print("result save_path", save_path)
    model.to("cuda:{}".format(val_cfg.gpu_id))
    # get features and fuse

    img_paths, landmarks, faceness_scores = insightface_ijb_helper.dataloader.get_IJB_info(
        val_cfg.IJB.data_root, subset_name
    )
    dataloader = insightface_ijb_helper.dataloader.prepare_dataloader(
        img_paths,
        landmarks,
        val_cfg.batch_size,
        num_workers=val_cfg.num_workers,
        image_size=(112, 112),
    )

    features, norms = validate_tinyface.infer(
        model,
        dataloader,
        gpu_id=val_cfg.gpu_id,
        use_flip_test=val_cfg.IJB.use_flip_test,
        fusion_method=val_cfg.IJB.fusion,
    )

    print(
        "Feature Shape: ({} , {}) .".format(
            features.shape[0], features.shape[1]
        )
    )

    if val_cfg.IJB.fusion == "pre_norm_vector_add":
        features = features * norms

    # run protocol
    # identification(args.data_root, dataset_name, img_input_feats, save_path)
    validate_IJB_BC.verification(
        val_cfg.IJB.data_root, subset_name, features, save_path
    )


val_IJBB = partial(val_IJB_subset, subset_name="IJBB")
val_IJBC = partial(val_IJB_subset, subset_name="IJBC")