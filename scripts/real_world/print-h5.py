from pathlib import Path

import cv2
import h5py


def print_hdf5_structure(file_path: Path) -> None:
    def print_attrs_and_shapes(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape = {obj.shape}, dtype = {obj.dtype}")
        else:
            print(f"{name}:")
        for key, val in obj.attrs.items():
            print(f"    {key}: {val}")

    with h5py.File(file_path, "r") as f:
        f.visititems(print_attrs_and_shapes)


# Usage
file_path = "/scr/thankyou/autom/demo-aug/data/real_world_recordings/Fri_May_31_22:39:48_2024/trajectory_im512.h5"
# file_path = "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/wine_glass_hanging/grasp/narrow/demo_01trials/2024-06-10/tks9xl4p/nullwine_glass_hanging_grasp_narrow_trials1_se3aug-dx0.0-0.0-dy0.0-0.0-dz0.0-0.0-dthetz0.0-0.0-biassampzrot_staug_joint-jointqposnoise0.06_defcurobocfg_CUROBO_tks9xl4p.hdf5"

file_path = Path(
    "/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-06-dummy/wine-glass-stowing-im128.hdf5"
)
file_path = Path(
    "/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-14-full-demo-w-kinks/Fri_Jun_14_14:41:20_2024/trajectory_im84_dp.hdf5"
)

# print_hdf5_structure(file_path)


def save_hdf5_images_to_mp4(file_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, "r") as f:
        datasets = [
            # "data/demo_0/obs/12391924_left_image",
            # "data/demo_0/obs/27432424_left_image"
            # "data/demo_0/observation/camera/image/hand_camera_image",
            # "data/demo_0/observation/camera/image/varied_camera_1_image",
            "data/demo_0/obs/agentview_image",
            "data/demo_0/obs/hand_camera_image",
        ]

        for dataset in datasets:
            import ipdb

            ipdb.set_trace()
            images = f[dataset][:]
            # convert from bgr to rgb
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

            output_file = output_dir / f"{dataset.split('/')[-1]}.mp4"
            height, width, layers = images[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(str(output_file), fourcc, 10, (width, height))

            for image in images:
                video.write(image)

            video.release()
            # SAVE AS PNG IMAGES
            for i, image in enumerate(images):
                cv2.imwrite(
                    str(output_dir / f"{dataset.split('/')[-1]}_{i}.png"), image
                )


def copy_demo_1_to_demo_0(file_path: Path) -> None:
    with h5py.File(file_path, "a") as f:
        if "data/demo_1" in f:
            f.copy("data/demo_1", "data/demo_0")
            print("Copied 'data/demo_1' to 'data/demo_0'")
        else:
            print("'data/demo_1' not found in the file")


copy_demo_1_to_demo_0(file_path)

# Usage
file_path = Path(
    "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/wine_glass_hanging/grasp/narrow/demo_01trials/2024-06-10/tks9xl4p/nullwine_glass_hanging_grasp_narrow_trials1_se3aug-dx0.0-0.0-dy0.0-0.0-dz0.0-0.0-dthetz0.0-0.0-biassampzrot_staug_joint-jointqposnoise0.06_defcurobocfg_CUROBO_tks9xl4p.hdf5"
)
file_path = Path(
    "/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-06-dummy/wine-glass-stowing-im128.hdf5"
)
file_path = Path(
    "/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-14-full-demo-w-kinks/Fri_Jun_14_14:41:20_2024/trajectory_im84_dp.hdf5"
)
file_path = Path(
    "/scr/thankyou/autom/diffusion_policy/data/robomimic/datasets/can/ph/1_demo.hdf5"
)

file_path = Path(
    "/scr/thankyou/autom/robosuite/robosuite/models/assets/demonstrations/1725256975_441713/demo.hdf5"
)
file_path = Path("../diffusion_policy/data/robomimic/datasets/can/ph/1_demo_v1.hdf5")
file_path = Path("/scr/thankyou/autom/robomimic/datasets/can/ph/image.hdf5")
file_path = Path(
    # "../diffusion_policy/data/robomimic/datasets/can/ph/1_demo_robomimic_backup_update_obs.hdf5"
    # "../diffusion_policy/data/robomimic/datasets/can/ph/1_demo_from_robomimic_backup.hdf5"
    "../diffusion_policy/data/robomimic/datasets/can/ph/1_demo.hdf5"
)
output_dir = file_path.parent
# copy_demo_1_to_demo_0(file_path)
print_hdf5_structure(file_path)

save_hdf5_images_to_mp4(file_path, output_dir)
print(f"Saved images to {output_dir}")
