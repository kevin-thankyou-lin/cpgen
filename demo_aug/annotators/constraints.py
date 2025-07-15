import copy
from typing import List, Tuple

import numpy as np

from demo_aug.configs.base_config import (
    AugmentationConfig,
    ConstraintInfo,
    DemoAugConfig,
    TaskRelevantObjectsConfig,
    load_constraint_infos,
)
from demo_aug.demo import Demo


class ConstraintAnnotator:
    """A class to parse the demonstration and annotates the demo with constraints.

    Currently, we query a human to annotate the demo with constraints.
    """

    @staticmethod
    def get_constraints(
        src_demo: Demo, demo_aug_cfg: DemoAugConfig
    ) -> List[ConstraintInfo]:
        """
        Queries the user for constraint data and saves constraint data to the hdf5 file.
        """
        constraint_infos: List[ConstraintInfo] = []
        constraint_infos = load_constraint_infos(
            str(demo_aug_cfg.demo_path), demo_aug_cfg.demo_name
        )
        if len(constraint_infos) > 0:
            # Ask user if they'd like to view and potentially overwrite the data
            # answer = input("Found existing data. Would you like to overwrite the data? (y/n) ")
            answer = "y"
            if answer.lower() == "n":
                print("Using loaded data")
                return
            else:
                print("Proceeding to view and potentially overwrite the data")

        # self.query_user_for_constraint_infos()
        # constraint_time_ranges = self.query_user_for_constraint_time_ranges()

        # currently can only provide info for 1x constraint_timerange from CLI
        # constraint_time_ranges: List[Tuple[int, int]] = [(97, 99)]  # [(50, 59)]
        # constraint_time_ranges: List[Tuple[int, int]] = [(190, 250)]  # [(50, 59)]
        # constraint_time_ranges: List[Tuple[int, int]] = [(230, 257)]  # [(50, 59)]
        # constraint_time_ranges: List[Tuple[int, int]] = [(230, 240)]  # [(50, 59)]
        # constraint_time_ranges: List[Tuple[int, int]] = [(50, 51)]  # [(50, 59)]
        # constraint_time_ranges: List[Tuple[int, int]] = [(135, 185)]  # [(50, 59)]
        constraint_time_ranges: List[Tuple[int, int]] = [(80, 80)]  # [(50, 59)]

        if demo_aug_cfg.task_name == "square":
            # constraint_time_ranges: List[Tuple[int, int]] = [(85, 86)]
            # constraint_time_ranges: List[Tuple[int, int]] = [(55, 63), (105, 120)]
            # constraint_time_ranges: List[Tuple[int, int]] = [(105, 120)]
            # constraint_time_ranges: List[Tuple[int, int]] = [(93, 120)]
            # grasping
            if demo_aug_cfg.subtask_name == "grasp":
                constraint_time_ranges: List[Tuple[int, int]] = [(55, 63)]
            elif demo_aug_cfg.subtask_name == "insert":
                constraint_time_ranges: List[Tuple[int, int]] = [(93, 120)]
            else:
                raise NotImplementedError("Subtask defaults not implemented yet")

        elif demo_aug_cfg.task_name == "can":
            if demo_aug_cfg.subtask_name == "grasp":
                constraint_time_ranges: List[Tuple[int, int]] = [(56, 69)]
            elif demo_aug_cfg.subtask_name == "drop":
                constraint_time_ranges: List[Tuple[int, int]] = [(105, 118)]

        elif demo_aug_cfg.task_name == "door":
            raise NotImplementedError("door task defaults not implemented yet")
        elif demo_aug_cfg.task_name == "lift":
            if demo_aug_cfg.method_name == "spartan":
                constraint_time_ranges: List[Tuple[int, int]] = []
                constraint_time_ranges = [
                    (start_t, 59) for start_t in np.random.randint(10, 40, 25)
                ]
            else:
                constraint_time_ranges: List[Tuple[int, int]] = [(50, 59)]
        elif demo_aug_cfg.task_name == "wine_glass_hanging":
            if demo_aug_cfg.subtask_name == "grasp":
                # TODO update this to only be grasping the wine glass (instead of basically both subtasks)
                constraint_time_ranges: List[Tuple[int, int]] = [(232, 265)]
            elif demo_aug_cfg.subtask_name == "insert":
                constraint_time_ranges: List[Tuple[int, int]] = [(300, 330)]
            else:
                raise NotImplementedError("Subtask defaults not implemented yet")
        else:
            raise NotImplementedError("task defaults not implemented yet")

        # TODO(klin): add out of bounds check
        # constraint_time_ranges: List[Tuple[int, int]] = demo_aug_cfg.constraint_infos.time_ranges

        for constraint_time_range in constraint_time_ranges:
            if demo_aug_cfg.task_name == "square":
                obj_names = ["square_peg", "square_nut"]
                # need to put this (welding) information into motion planning somewhere too
                constraint_time_range_aug_cfg: AugmentationConfig = copy.deepcopy(
                    demo_aug_cfg.aug_cfg
                )
                if constraint_time_range == (105, 120) or constraint_time_range == (
                    93,
                    120,
                ):
                    constraint_time_range_aug_cfg.start_aug_cfg.weld_obj_to_ee_for_reach_traj = True
                    constraint_time_range_aug_cfg.start_aug_cfg.set_gripper_action_close = True
                    constraint_time_range_aug_cfg.start_aug_cfg.only_use_tracking_start_gripper_qpos = True
                    constraint_time_range_aug_cfg.start_aug_cfg.end_with_tracking_start_gripper_qpos = True
                    # constraint_time_range_aug_cfg.se3_aug_cfg.use_abs_transform = False
                    # don't translate the peg for now; specify in run script first
                    t_to_task_relev_objs_cfg = {
                        # do we need per timestep per object config?
                        t: TaskRelevantObjectsConfig(
                            obj_names=obj_names,
                            obj_to_nerf_generation_image_ts={
                                # "square_peg": tuple([90]),
                                "square_peg": tuple([5]),
                                "square_nut": tuple(
                                    [85]
                                ),  # hoping no weird default value behaviors
                                # "square_nut": tuple([90]),  # hoping no weird default value behaviors
                            },
                            weld_obj_to_ee=True,
                            weld_obj_name="square_nut",
                            weld_t_src=85,  # for now, assume weld t is within constraint time;
                            # need to use these values (from base_config)
                            # range so that we have eef poses
                        )
                        for t in range(
                            constraint_time_range[0], constraint_time_range[1] + 1
                        )
                    }
                    constraint_info = ConstraintInfo(
                        time_range=constraint_time_range,
                        t_to_task_relev_objs_cfg=t_to_task_relev_objs_cfg,
                        aug_cfg=constraint_time_range_aug_cfg,
                        weld_obj_name="square_nut",
                        weld_t_src=85,  # for now, assume weld t is within
                        # constraint time range so that we have eef poses
                        weld_obj_to_ee=True,
                        weld_t_range=(93, 120),
                        se3_origin_obj_name="square_nut",  # only if we're also moving nut around ...
                        scale_origin_obj_name="square_peg",
                    )
                elif constraint_time_range == (55, 63):
                    constraint_time_range_aug_cfg.start_aug_cfg.weld_obj_to_ee_for_reach_traj = False
                    constraint_time_range_aug_cfg.start_aug_cfg.end_with_tracking_start_gripper_qpos = True

                    t_to_task_relev_objs_cfg = {
                        t: TaskRelevantObjectsConfig(
                            obj_names=obj_names,
                            obj_to_nerf_generation_image_ts={
                                "square_peg": tuple([5]),
                                "square_nut": tuple([50]),
                            },
                            weld_obj_to_ee=True,  # if True, assume the object is welded for the entire time range
                            weld_obj_name="square_nut",
                            weld_t_src=55,  # for now, assume weld t is within constraint time;
                        )
                        for t in range(
                            constraint_time_range[0], constraint_time_range[1] + 1
                        )
                    }
                    constraint_info = ConstraintInfo(
                        time_range=constraint_time_range,
                        t_to_task_relev_objs_cfg=t_to_task_relev_objs_cfg,
                        aug_cfg=constraint_time_range_aug_cfg,
                        weld_obj_name="square_nut",
                        weld_t_src=55,
                        weld_obj_to_ee=True,
                        weld_t_range=(55, 63),
                        se3_origin_obj_name="square_nut",  # only if we're also moving nut around ...
                        scale_origin_obj_name="square_nut",
                    )
                else:
                    t_to_task_relev_objs_cfg = {
                        t: TaskRelevantObjectsConfig(
                            obj_names=obj_names,
                            obj_to_nerf_generation_image_ts={
                                obj_name: list([t]) for obj_name in obj_names
                            },
                        )
                        for t in range(
                            constraint_time_range[0], constraint_time_range[1] + 1
                        )
                    }
                    constraint_info = ConstraintInfo(
                        time_range=constraint_time_range,
                        t_to_task_relev_objs_cfg=t_to_task_relev_objs_cfg,
                        aug_cfg=constraint_time_range_aug_cfg,
                        weld_obj_to_ee=False,
                    )

                constraint_infos.append(constraint_info)
            elif demo_aug_cfg.task_name == "lift":
                obj_names = ["red_cube"]
                constraint_time_range_aug_cfg: AugmentationConfig = copy.deepcopy(
                    demo_aug_cfg.aug_cfg
                )

                constraint_time_range_aug_cfg.start_aug_cfg.weld_obj_to_ee_for_reach_traj = False
                constraint_time_range_aug_cfg.start_aug_cfg.end_with_tracking_start_gripper_qpos = True

                t_to_task_relev_objs_cfg = {
                    t: TaskRelevantObjectsConfig(
                        obj_names=obj_names,
                        obj_to_nerf_generation_image_ts={"red_cube": tuple([20])},
                    )
                    for t in range(
                        constraint_time_range[0], constraint_time_range[1] + 1
                    )
                }
                constraint_info = ConstraintInfo(
                    time_range=constraint_time_range,
                    t_to_task_relev_objs_cfg=t_to_task_relev_objs_cfg,
                    aug_cfg=constraint_time_range_aug_cfg,
                    weld_obj_name="red_cube",
                    weld_t_src=50,
                    weld_obj_to_ee=True,
                    weld_t_range=(50, 59),
                    se3_origin_obj_name="red_cube",
                    scale_origin_obj_name="red_cube",
                )
                constraint_infos.append(constraint_info)
            elif demo_aug_cfg.task_name == "can":
                obj_names = ["can"]
                constraint_time_range_aug_cfg: AugmentationConfig = copy.deepcopy(
                    demo_aug_cfg.aug_cfg
                )

                if constraint_time_range == (56, 69):
                    constraint_time_range_aug_cfg.start_aug_cfg.weld_obj_to_ee_for_reach_traj = False
                    constraint_time_range_aug_cfg.start_aug_cfg.end_with_tracking_start_gripper_qpos = True

                    t_to_task_relev_objs_cfg = {
                        t: TaskRelevantObjectsConfig(
                            obj_names=obj_names,
                            obj_to_nerf_generation_image_ts={"can": tuple([33])},
                        )
                        for t in range(
                            constraint_time_range[0], constraint_time_range[1] + 1
                        )
                    }
                    constraint_info = ConstraintInfo(
                        time_range=constraint_time_range,
                        t_to_task_relev_objs_cfg=t_to_task_relev_objs_cfg,
                        aug_cfg=constraint_time_range_aug_cfg,
                        weld_obj_name="can",
                        weld_t_src=56,
                        weld_obj_to_ee=True,
                        weld_t_range=(56, 69),
                        se3_origin_obj_name="can",
                        scale_origin_obj_name="can",
                    )
                elif constraint_time_range == (105, 118):
                    constraint_time_range_aug_cfg.start_aug_cfg.weld_obj_to_ee_for_reach_traj = True
                    constraint_time_range_aug_cfg.start_aug_cfg.set_gripper_action_close = True
                    constraint_time_range_aug_cfg.start_aug_cfg.only_use_tracking_start_gripper_qpos = True
                    constraint_time_range_aug_cfg.start_aug_cfg.end_with_tracking_start_gripper_qpos = True

                    t_to_task_relev_objs_cfg = {
                        t: TaskRelevantObjectsConfig(
                            obj_names=obj_names,
                            obj_to_nerf_generation_image_ts={"can": tuple([33])},
                            weld_obj_to_ee=True,
                            weld_obj_name="can",
                            weld_t_src=56,
                        )
                        for t in range(
                            constraint_time_range[0], constraint_time_range[1] + 1
                        )
                    }
                    constraint_info = ConstraintInfo(
                        time_range=constraint_time_range,
                        t_to_task_relev_objs_cfg=t_to_task_relev_objs_cfg,
                        aug_cfg=constraint_time_range_aug_cfg,
                        weld_obj_name="can",
                        weld_t_src=56,
                        weld_obj_to_ee=True,
                        weld_t_range=(105, 118),
                        se3_origin_obj_name="can",
                        scale_origin_obj_name="can",
                    )
                else:
                    import ipdb

                    ipdb.set_trace()
                constraint_infos.append(constraint_info)
            elif demo_aug_cfg.task_name == "wine_glass_hanging":
                # obj_names = ["wine_glass", "wine_glass_holder", "background"]
                obj_names = ["wine_glass", "background"]
                constraint_time_range_aug_cfg: AugmentationConfig = copy.deepcopy(
                    demo_aug_cfg.aug_cfg
                )
                if demo_aug_cfg.subtask_name == "grasp":
                    constraint_time_range_aug_cfg.start_aug_cfg.weld_obj_to_ee_for_reach_traj = False
                    constraint_time_range_aug_cfg.start_aug_cfg.end_with_tracking_start_gripper_qpos = True

                    t_to_task_relev_objs_cfg = {
                        t: TaskRelevantObjectsConfig(
                            obj_names=obj_names,
                            obj_to_nerf_generation_image_ts={
                                "wine_glass": tuple([0]),
                                # "wine_glass_holder": tuple([1]),
                                "background": tuple([2]),
                            },
                        )
                        for t in range(
                            constraint_time_range[0], constraint_time_range[1] + 1
                        )
                    }
                    constraint_info = ConstraintInfo(
                        time_range=constraint_time_range,
                        t_to_task_relev_objs_cfg=t_to_task_relev_objs_cfg,
                        aug_cfg=constraint_time_range_aug_cfg,
                        weld_obj_name="wine_glass",
                        weld_t_src=232,
                        weld_obj_to_ee=True,
                        weld_t_range=(232, 265),
                        se3_origin_obj_name="wine_glass",
                        scale_origin_obj_name="wine_glass",
                    )
                elif demo_aug_cfg.subtask_name == "insert":
                    raise NotImplementedError(
                        "insert subtask doesn't work for now, b/cassume weld t is assumed to be within constraint time."
                    )
                    constraint_time_range_aug_cfg.start_aug_cfg.weld_obj_to_ee_for_reach_traj = True
                    constraint_time_range_aug_cfg.start_aug_cfg.set_gripper_action_close = True
                    constraint_time_range_aug_cfg.start_aug_cfg.only_use_tracking_start_gripper_qpos = True
                    constraint_time_range_aug_cfg.start_aug_cfg.end_with_tracking_start_gripper_qpos = True

                    t_to_task_relev_objs_cfg = {
                        t: TaskRelevantObjectsConfig(
                            obj_names=obj_names,
                            obj_to_nerf_generation_image_ts={
                                "wine_glass": tuple([0]),
                                "wine_glass_holder": tuple([1]),
                            },
                        )
                        for t in range(
                            constraint_time_range[0], constraint_time_range[1] + 1
                        )
                    }
                    constraint_info = ConstraintInfo(
                        time_range=constraint_time_range,
                        t_to_task_relev_objs_cfg=t_to_task_relev_objs_cfg,
                        aug_cfg=constraint_time_range_aug_cfg,
                        weld_obj_name="wine_glass",
                        weld_t_src=250,
                        weld_obj_to_ee=True,
                        weld_t_range=(
                            300,
                            330,
                        ),  # for now, assume weld t is within constraint time;
                        se3_origin_obj_name="wine_glass",
                        scale_origin_obj_name="wine_glass_holder",
                    )
                constraint_infos.append(constraint_info)

            elif demo_aug_cfg.task_name == "door":
                raise NotImplementedError("door task defaults not implemented yet")
            else:
                raise NotImplementedError("task defaults not implemented yet")

        return constraint_infos

    @staticmethod
    def _query_user(question: str) -> bool:
        """Query the user for a yes or no answer."""
        while True:
            answer = input(f"{question} (y/n): ")
            if answer == "y":
                return True
            elif answer == "n":
                return False
            else:
                print("Please answer with y or n.")

    @staticmethod
    def query_user_for_constraint_time_ranges(src_demo: Demo) -> List[int]:
        """Given a demo, query the user to identify the constraint timesteps.

        Feed the demo to the user and ask them to identify the constraint timesteps.
        """
        constraint_time_ranges = []
        for i, timestep_data in enumerate(src_demo.timestep_data):
            from PIL import Image

            if i > 9:
                img = Image.fromarray(
                    (timestep_data.obs["agentview_image"]).astype(np.uint8)
                )
                img.show()
                if ConstraintAnnotator._query_user("Is this a constraint timestep?"):
                    constraint_time_ranges.append(i)

        constraint_time_ranges.append((50, 59))
        return constraint_time_ranges
