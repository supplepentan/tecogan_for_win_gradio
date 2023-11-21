import os
import os.path as osp

from codes.models import define_model
from codes.utils import (
    dist_utils,
    base_utils,
    data_utils,
    dataloader_utils,
)


def inference(opt):
    # infer and evaluate performance for each model
    for load_path in opt["model"]["generator"]["load_path_lst"]:
        # set model index
        model_idx = osp.splitext(osp.split(load_path)[-1])[0]

        # create model
        opt["model"]["generator"]["load_path"] = load_path
        model = define_model(opt)

        # for each test dataset
        for dataset_idx in sorted(opt["dataset"].keys()):
            # select testing dataset
            if "test" not in dataset_idx:
                continue

            ds_name = opt["dataset"][dataset_idx]["name"]
            base_utils.log_info(f"Testing on {ds_name} dataset")

            # create data loader
            test_loader = dataloader_utils.create_dataloader(
                opt, phase="test", idx=dataset_idx
            )
            test_dataset = test_loader.dataset
            num_seq = len(test_dataset)

            # create metric calculator
            # metric_calculator = create_metric_calculator(opt)

            # infer a sequence
            rank, world_size = dist_utils.get_dist_info()
            for idx in range(rank, num_seq, world_size):
                # fetch data
                data = test_dataset[idx]

                # prepare data
                model.prepare_inference_data(data)

                # infer
                hr_seq = model.infer()

                # save hr results
                if opt["test"]["save_res"]:
                    res_dir = osp.join(opt["test"]["res_dir"], ds_name)
                    res_seq_dir = osp.join(res_dir, data["seq_idx"])
                    data_utils.save_sequence(
                        res_dir, hr_seq, data["frm_idx"], to_bgr=True
                    )

            base_utils.log_info("-" * 40)
