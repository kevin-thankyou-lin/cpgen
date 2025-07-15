from typing import Tuple

import numpy as np

from demo_aug.configs.base_config import DemoAugConfig
from demo_aug.demo import Demo
from demo_aug.objects.reconstructor import ReconstructionManager, SegmentationMask


class SegmentationMaskAnnotator:
    """A class to parse the demonstration, constraints, reconstructions and annotate the reconstructions
    with segmentation masks.

    Currently, we query a human to annotate the constraint-relevant reconstructions with constraints.
    """

    @staticmethod
    def get_segmentation_masks(
        src_demo: Demo,
        demo_aug_cfg: DemoAugConfig,
        rec_manager: ReconstructionManager,
        rec_ts: Tuple[int],
        obj_name: str,
    ) -> SegmentationMask:
        # constraint info contains the object name and time right?
        # constraint info will tell us, for the given time, which source images to use for reconstructions
        # from the reconstruction times + object name, we can figure out the bounding box target
        # we might want to view the original nerf object from the rec_manager though?
        if (rec_ts, obj_name) == ((93,), "square_peg"):
            # bounding_box = np.array([[0.19, 0.06, 0.805], [0.27, 0.14, 0.955]])
            # bounding_box = np.array([[0.21, 0.09, 0.805], [0.26, 0.13, 0.94]])
            bounding_box = np.array([[0.21, 0.08, 0.74], [0.25, 0.12, 0.96]])
            obb_center = bounding_box.mean(axis=0)
            obb_scale = bounding_box[1] - bounding_box[0]
            obb_rotation = np.array([0.0, 0.0, 0.0])
        elif (
            (rec_ts, obj_name) == ((85,), "square_peg")
            or (rec_ts, obj_name) == ((105,), "square_peg")
            or (rec_ts, obj_name) == ((93,), "square_nut")
        ):
            # bounding_box = np.array([[0.19, 0.06, 0.805], [0.27, 0.14, 0.955]])
            # bounding_box = np.array([[0.21, 0.09, 0.805], [0.26, 0.13, 0.94]])
            # bounding_box = np.array([[0.17, 0.05, 0.805], [0.29, 0.15, 0.955]])
            bounding_box = np.array([[0.21, 0.08, 0.74], [0.25, 0.12, 0.96]])
            obb_center = bounding_box.mean(axis=0)
            obb_scale = bounding_box[1] - bounding_box[0]
            obb_rotation = np.array([0.0, 0.0, 0.0])
        elif (
            (rec_ts, obj_name) == ((85,), "square_nut")
            or (rec_ts, obj_name) == ((105,), "square_nut")
            or (rec_ts, obj_name) == ((93,), "square_nut")
        ):
            bounding_box = np.array(
                [
                    [0.07, 0.05, 0.945],
                    [0.27, 0.15, 0.995],
                ]
            )
            obb_center = np.array([0.20, 0.07, 0.97])
            obb_rotation = np.array([-0.04, 0.03, 0.1])
            obb_scale = np.array([0.16, 0.10, 0.020])

            # for removing gripper part to avoid drake collision checker saying collision btw gripper and object
            # obb_center = np.array([0.2, 0.07, 0.99])
            # obb_rotation = np.array([0.0, 0.0, 0.12])
            # obb_scale = np.array([0.1, 0.10, 0.06])

            bounding_box = np.array(
                [
                    obb_center - obb_scale / 2,
                    obb_center + obb_scale / 2,
                ]
            )
            segmentation_mask = SegmentationMask(
                oriented_bbox_pos=obb_center,
                oriented_bbox_rpy=obb_rotation,
                oriented_bbox_scale=obb_scale,
            )

            # for the purposes of collision checking between welded object and robot, we can just ignore welded object right?
            return segmentation_mask
        elif (rec_ts, obj_name) == ((35,), "square_nut") or (rec_ts, obj_name) == (
            (50,),
            "square_nut",
        ):
            # if both bb and obb info are provided, we use obb info
            obb_center = np.array(
                [-0.10999999940395355, 0.17000000178813934, 0.8299999833106995]
            )
            obb_rotation = np.array([0.0, 0.0, 0.0])
            obb_scale = np.array([0.14, 0.13, 0.02])
            bounding_box = np.array(
                [
                    obb_center - obb_scale / 2,
                    obb_center + obb_scale / 2,
                ]
            )
        elif (rec_ts, obj_name) == ((55,), "square_peg") or (rec_ts, obj_name) == (
            (5,),
            "square_peg",
        ):
            bounding_box = np.array([[0.214, 0.084, 0.75], [0.246, 0.116, 0.95]])
            obb_center = bounding_box.mean(axis=0)
            obb_scale = bounding_box[1] - bounding_box[0]
            obb_rotation = np.array([0.0, 0.0, 0.0])
        elif (rec_ts, obj_name) == ((93,), "door"):
            bounding_box = np.array(
                [
                    [-0.5, -0.6, 0.8],
                    [0.1, -0.2, 1.4],
                ]
            )
            obb_center = bounding_box.mean(axis=0)
            obb_scale = bounding_box[1] - bounding_box[0]
            obb_rotation = np.array([0.0, 0.0, 0.0])
        elif (rec_ts, obj_name) == ((20,), "red_cube"):
            obb_center = np.array([0.03, 0.03, 0.83])
            obb_scale = np.array([0.04, 0.05, 0.055])
            obb_rotation = np.array([0.04, 0.0, -0.5])
        elif (rec_ts, obj_name) == ((5,), "square_peg"):
            obb_center = np.array([0.23, 0.10, 0.89])
            obb_scale = np.array([0.03, 0.03, 0.16])
            obb_rotation = np.array([0.0, 0.0, 0.0])
        elif (rec_ts, obj_name) == ((33,), "can"):
            obb_center = np.array([0.12, -0.21, 0.86])
            obb_scale = np.array([0.05, 0.06, 0.09])
            obb_rotation = np.array([0.0, 0.0, 0.0])
            # for removing top of can b/c generated mesh is too large
            # obb_center = np.array([0.12, -0.21, 0.84])
            # obb_scale = np.array([0.05, 0.06, 0.09])
            # obb_rotation = np.array([0.0, 0.0, 0.0])
        elif (rec_ts, obj_name) == ((0,), "wine_glass"):
            obb_center = np.array([0.4, 0.17, 0.13])
            obb_scale = np.array([0.11, 0.12, 0.23])
            obb_rotation = np.array([0.00, 0.0, 0.0])
        elif (rec_ts, obj_name) == ((1,), "wine_glass_holder"):
            obb_center = np.array([0.53, -0.14, 0.185])
            # obb_center = np.array([0.53, -0.14, 0.17])
            obb_scale = np.array([0.27, 0.32, 0.33])
            obb_rotation = np.array([0.00, 0.0, -0.15])
        elif (rec_ts, obj_name) == ((2,), "background"):
            obb_center = np.array([0.2, 0, -0.1])
            obb_scale = np.array([0.001, 0.001, 0.001])
            obb_rotation = np.array(
                [0.00, 0.0, 0.0]
            )  # a bit awkward to specify b/c/ we can use hardcoded value for background in the xml? o maybe mesh is already fixed so al l good? man rendering takes a long time .. hmm
            obb_center = np.array([0.2, 0, -0.1])
            obb_scale = np.array([10, 10, 10])
            obb_rotation = np.array(
                [0.00, 0.0, 0.0]
            )  # a bit awkward to specify b/c/ we can use hardcoded value for background in the xml? o maybe mesh is already fixed so al l good? man rendering takes a long time .. hmm
        else:
            print(
                f"rec_ts: {rec_ts}, obj_name: {obj_name} not found in get_segmentation_masks."
            )
            import ipdb

            ipdb.set_trace()

        return SegmentationMask(obb_center, obb_rotation, obb_scale)
