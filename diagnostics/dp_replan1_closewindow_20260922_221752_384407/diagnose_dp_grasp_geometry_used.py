#!/usr/bin/env python3
"""Causal diagnostic: one-step replanning + clean observation history + non-working-arm masking.

Run from ~/RoboTwin with the existing robotwin_dp environment:
    CUDA_VISIBLE_DEVICES=0 python diagnose_dp_failure.py

No repository source files or checkpoints are overwritten. Two fresh child
processes test exactly one requested scene each. Test count, start seed and
result directory are changed ONLY in the in-memory evaluation entry point.
The working-arm DP actions, TOPP, observation history and success criteria remain unchanged.\nOnly the task-irrelevant arm is held at its current commanded state as a diagnostic intervention. Requires the user's existing RoboTwin dependencies.

The hooks are based on the local code pasted into the supplied audit document;
this script has not been run on the user's SAPIEN/CUDA installation.
"""
from __future__ import annotations

import argparse
import ast
from collections import deque
from datetime import datetime
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import types
import zipfile

np = None  # Loaded in the worker, using the user's existing environment.


class AuditStop(BaseException):
    """Not caught by eval_policy's broad 'except Exception: continue'."""


def clean(value):
    if np is not None:
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding="utf-8")


def max_abs(value):
    arr = np.asarray(value, dtype=float)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def array_info(value) -> dict:
    arr = np.asarray(value)
    result = {"shape": list(arr.shape), "dtype": str(arr.dtype), "size": int(arr.size)}
    if arr.size and np.issubdtype(arr.dtype, np.number):
        result["all_finite"] = bool(np.isfinite(arr).all())
        if result["all_finite"]:
            result.update(min=float(arr.min()), max=float(arr.max()), mean=float(arr.mean()))
    return result


def fingerprint(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()[:16]


def try_read(fn):
    try:
        return clean(fn())
    except Exception as exc:
        return {"read_error": repr(exc)}


def robot_state(env) -> dict:
    """Only getters: no rendering, get_obs, RNG draw or physics stepping."""
    robot = env.robot
    result = {}
    for arm in ("left", "right"):
        cmd = np.asarray(getattr(robot, f"get_{arm}_arm_jointState")(), dtype=float)
        actual = np.asarray(getattr(robot, f"get_{arm}_arm_real_jointState")(), dtype=float)
        entity = getattr(robot, f"{arm}_entity")
        qpos = np.asarray(entity.get_qpos(), dtype=float)
        active = list(entity.get_active_joints())
        gripper_joints = getattr(robot, f"{arm}_gripper", [])
        # *_real_jointState() still appends the commanded gripper value in this repo.
        # Physical gripper positions are logged separately below.
        real_gripper = try_read(lambda: [float(qpos[active.index(item[0])]) for item in gripper_joints])
        result[arm] = {
            "q_cmd": cmd[:-1], "q_real": actual[:-1],
            "gripper_cmd_normalized": float(cmd[-1]),
            "gripper_real_joint_positions": real_gripper,
            "tracking_error_max_rad": max_abs(cmd[:-1] - actual[:-1]),
        }
    result["cup_pose"] = try_read(lambda: list(env.cup.get_pose().p) + list(env.cup.get_pose().q))
    result["coaster_pose"] = try_read(lambda: list(env.coaster.get_pose().p) + list(env.coaster.get_pose().q))

    # Grasp-geometry diagnostics.
    # For place_empty_cup, the expert uses cup contact_point_id=0 for the RIGHT arm.
    try:
        cup_center = np.asarray(env.cup.get_pose().p, dtype=float)
        cup_contact0 = np.asarray(env.cup.get_contact_point(0, "pose").p, dtype=float)
        left_tcp = np.asarray(robot.get_left_tcp_pose()[:3], dtype=float)
        right_tcp = np.asarray(robot.get_right_tcp_pose()[:3], dtype=float)

        contact_positions = env.get_gripper_actor_contact_position(env.cup.get_name())
        result["grasp_geometry"] = {
            "cup_center": cup_center,
            "cup_contact0": cup_contact0,
            "left_tcp": left_tcp,
            "right_tcp": right_tcp,
            "left_tcp_to_cup_center_m": float(np.linalg.norm(left_tcp - cup_center)),
            "right_tcp_to_cup_center_m": float(np.linalg.norm(right_tcp - cup_center)),
            "left_tcp_to_contact0_m": float(np.linalg.norm(left_tcp - cup_contact0)),
            "right_tcp_to_contact0_m": float(np.linalg.norm(right_tcp - cup_contact0)),
            "cup_z_m": float(cup_center[2]),
            "left_gripper_cmd": float(robot.get_left_gripper_val()),
            "right_gripper_cmd": float(robot.get_right_gripper_val()),
            "cup_gripper_contact_count": int(len(contact_positions)),
        }
    except Exception as exc:
        result["grasp_geometry"] = {"read_error": repr(exc)}

    try:
        c = np.asarray(env.cup.get_functional_point(0, "pose").p, dtype=float)
        t = np.asarray(env.coaster.get_functional_point(0, "pose").p, dtype=float)
        result["placement"] = {
            "cup_functional_point": c, "coaster_functional_point": t,
            "xy_error_m": float(np.linalg.norm(c[:2] - t[:2])),
            "z_error_m": float(abs(c[2] - t[2])),
            "left_gripper_open": bool(env.is_left_gripper_open()),
            "right_gripper_open": bool(env.is_right_gripper_open()),
        }
    except Exception as exc:
        result["placement"] = {"read_error": repr(exc)}
    return result


class Recorder:
    def __init__(self, folder: Path, requested_seed: int, max_actions: int = 180):
        self.folder = folder
        self.seed = requested_seed
        self.max_actions = int(max_actions)
        self.first_right_close_step = None
        self.stop_after_close_actions = 15
        self.stream = (folder / "events.jsonl").open("w", encoding="utf-8", buffering=1)
        self.phase = "not_started"
        self.setup_count = 0
        self.actions = []
        self.initial_states = []
        self.obs_count = 0
        self.predict_count = 0
        self.current_prediction = 0
        self.last_head_hash = None
        self.head_change_count = 0
        self.expert_success = None
        self.eval_success = None
        self.nominal_arm = None
        self.last_state = None
        self.status = "started"
        self.error = None
        self.limit_arrays = {}

    def event(self, event: str, **fields):
        row = {"event": event, "eval_seed": self.seed, "phase": self.phase, **fields}
        self.stream.write(json.dumps(clean(row), ensure_ascii=False) + "\n")

    def finish(self):
        per_arm = {}
        for arm in ("left", "right"):
            entries = [r[arm] for r in self.actions if arm in r]
            tops = [t for r in self.actions for t in r.get("topp", []) if t["arm"] == arm]
            per_arm[arm] = {
                "topp_calls": len(tops),
                "topp_exceptions": sum(t.get("exception") is not None for t in tops),
                "topp_zero_sample_returns": sum(t.get("n_samples") == 0 for t in tops),
                "topp_one_sample_returns": sum(t.get("n_samples") == 1 for t in tops),
                "actions_without_arm_command": sum(r.get("set_calls", 0) == 0 for r in entries),
                "max_target_minus_cmd_rad": max((r.get("target_minus_cmd_max_rad", 0) for r in entries), default=None),
                "max_target_minus_real_rad": max((r.get("target_minus_real_max_rad", 0) for r in entries), default=None),
                "max_real_motion_per_action_rad": max((r.get("real_motion_max_rad", 0) for r in entries), default=None),
                "max_tracking_error_rad": max((r.get("post_tracking_error_max_rad", 0) for r in entries), default=None),
            }
        # Save one compact grasp-geometry row per executed action.
        grasp_rows = []
        for r in self.actions:
            pre_g = (r.get("pre") or {}).get("grasp_geometry", {})
            post_g = (r.get("post") or {}).get("grasp_geometry", {})
            grasp_rows.append({
                "action_step": r.get("action_step"),
                "prediction": r.get("prediction"),
                "pre_right_tcp_to_contact0_m": pre_g.get("right_tcp_to_contact0_m"),
                "post_right_tcp_to_contact0_m": post_g.get("right_tcp_to_contact0_m"),
                "pre_right_tcp_to_cup_center_m": pre_g.get("right_tcp_to_cup_center_m"),
                "post_right_tcp_to_cup_center_m": post_g.get("right_tcp_to_cup_center_m"),
                "pre_cup_z_m": pre_g.get("cup_z_m"),
                "post_cup_z_m": post_g.get("cup_z_m"),
                "pre_right_gripper_cmd": pre_g.get("right_gripper_cmd"),
                "post_right_gripper_cmd": post_g.get("right_gripper_cmd"),
                "pre_contact_count": pre_g.get("cup_gripper_contact_count"),
                "post_contact_count": post_g.get("cup_gripper_contact_count"),
            })

        if grasp_rows:
            import csv
            csv_path = self.folder / "grasp_geometry.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(grasp_rows[0].keys()))
                writer.writeheader()
                writer.writerows(grasp_rows)

        def finite_values(key):
            vals = []
            for row in grasp_rows:
                v = row.get(key)
                if isinstance(v, (int, float)):
                    try:
                        if np.isfinite(float(v)):
                            vals.append(float(v))
                    except Exception:
                        pass
            return vals

        geom_summary = {}
        if grasp_rows:
            contact_d = finite_values("post_right_tcp_to_contact0_m")
            center_d = finite_values("post_right_tcp_to_cup_center_m")
            cup_zs = finite_values("post_cup_z_m")

            geom_summary["min_right_tcp_to_contact0_m"] = min(contact_d) if contact_d else None
            geom_summary["min_right_tcp_to_cup_center_m"] = min(center_d) if center_d else None

            initial_z = None
            first_pre_z = grasp_rows[0].get("pre_cup_z_m")
            if isinstance(first_pre_z, (int, float)):
                initial_z = float(first_pre_z)
            geom_summary["initial_cup_z_m"] = initial_z
            geom_summary["max_cup_z_m"] = max(cup_zs) if cup_zs else None
            geom_summary["max_cup_lift_m"] = (
                max(cup_zs) - initial_z if cup_zs and initial_z is not None else None
            )

            first_close = None
            first_contact = None
            first_lift_5mm = None
            for row in grasp_rows:
                step = row.get("action_step")
                grip = row.get("post_right_gripper_cmd")
                ccount = row.get("post_contact_count")
                z = row.get("post_cup_z_m")
                if first_close is None and isinstance(grip, (int, float)) and float(grip) < 0.2:
                    first_close = step
                if first_contact is None and isinstance(ccount, (int, float)) and int(ccount) > 0:
                    first_contact = step
                if (
                    first_lift_5mm is None
                    and initial_z is not None
                    and isinstance(z, (int, float))
                    and float(z) > initial_z + 0.005
                ):
                    first_lift_5mm = step

            geom_summary["first_right_gripper_close_step_lt_0p2"] = first_close
            geom_summary["first_cup_gripper_contact_step"] = first_contact
            geom_summary["first_cup_lift_5mm_step"] = first_lift_5mm

        summary = {
            "status": self.status, "error": self.error, "eval_seed": self.seed,
            "grasp_geometry_summary": geom_summary,
            "expert_success_before_close": self.expert_success,
            "policy_success": self.eval_success,
            "nominal_working_arm_from_initial_cup_x": self.nominal_arm,
            "executed_take_action_calls": len(self.actions),
            "policy_predictions": self.predict_count,
            "policy_get_obs_calls": self.obs_count,
            "head_image_hash_changes": self.head_change_count,
            "per_arm": per_arm, "final_state": self.last_state,
            "note": "Counts are observations, not a root-cause verdict. Idle-arm TOPP errors must not be equated with task failure.",
        }
        write_json(self.folder / "summary.json", summary)
        self.stream.close()
        return clean(summary)

    def attach(self, env, model):
        self.event("model", policy_type=type(model.policy).__name__,
                   policy_source=inspect.getfile(type(model.policy)),
                   runner_source=inspect.getfile(type(model.runner)),
                   wrapper_source=inspect.getfile(type(model)),
                   runner_n_obs_steps=model.runner.n_obs_steps,
                   policy_n_obs_steps=model.policy.n_obs_steps,
                   runner_n_action_steps=model.runner.n_action_steps,
                   policy_n_action_steps=model.policy.n_action_steps,
                   horizon=model.policy.horizon,
                   rgb_keys=list(model.policy.obs_encoder.rgb_keys))
        if model.runner.n_obs_steps != model.policy.n_obs_steps:
            raise AuditStop("Runner 与 checkpoint 的 n_obs_steps 不一致；已记录，未自动修正。")
        if model.runner.n_action_steps != model.policy.n_action_steps:
            self.event("configuration_warning", message="Runner 与 checkpoint 的 n_action_steps 不同，保留原配置运行。")

        original_setup = env.setup_demo
        @functools.wraps(original_setup)
        def setup(*args, **kwargs):
            actual_seed = kwargs.get("seed")
            if actual_seed != self.seed:
                raise AuditStop(f"请求 seed={self.seed} 未通过专家筛选或发生异常；原入口准备跳到 {actual_seed}，诊断已停止，未替换场景。")
            self.setup_count += 1
            self.phase = "expert" if self.setup_count == 1 else "policy"
            self.event("setup_begin", setup_count=self.setup_count)
            result = original_setup(*args, **kwargs)
            state = robot_state(env)
            self.initial_states.append(clean(state))
            self.event("setup_complete", state=state)
            if self.phase == "policy":
                self.last_state = state
                cup_pose = state["cup_pose"]
                if isinstance(cup_pose, list):
                    self.nominal_arm = "right" if cup_pose[0] > 0 else "left"
                for arm in ("left", "right"):
                    joints = getattr(env.robot, f"{arm}_arm_joints")
                    limits = try_read(lambda: [j.get_limits()[0].tolist() for j in joints])
                    if isinstance(limits, list):
                        try:
                            self.limit_arrays[arm] = np.asarray(limits, dtype=float)
                        except (TypeError, ValueError):
                            pass
                    self.event("joint_model", arm=arm, names=[j.get_name() for j in joints], limits=limits,
                               topp_owner=repr(type(getattr(env.robot, f"{arm}_mplib_planner").planner)))
                if len(self.initial_states) >= 2:
                    self.event("initial_scene_comparison", expert=self.initial_states[0], policy=self.initial_states[1])
            return result
        env.setup_demo = setup

        original_play = env.play_once
        @functools.wraps(original_play)
        def play(*args, **kwargs):
            try:
                result = original_play(*args, **kwargs)
            except BaseException as exc:
                self.event("expert_exception", error=repr(exc))
                raise
            self.expert_success = bool(env.plan_success and env.check_success())
            self.event("expert_result", plan_success=env.plan_success, check_success=env.check_success())
            print(f"[DIAG] seed={self.seed} expert_success={self.expert_success}", flush=True)
            return result
        env.play_once = play

        original_obs = env.get_obs
        @functools.wraps(original_obs)
        def get_obs(*args, **kwargs):
            result = original_obs(*args, **kwargs)
            if self.phase == "policy":
                self.obs_count += 1
                head = result["observation"]["head_camera"]["rgb"]
                h = fingerprint(head)
                if self.last_head_hash is not None and self.last_head_hash != h:
                    self.head_change_count += 1
                self.last_head_hash = h
                self.event("observation", obs_number=self.obs_count,
                           action_step=env.take_action_cnt, head=array_info(head), head_hash=h,
                           agent_pos=result["joint_action"]["vector"])
                if self.obs_count == 1:
                    np.save(self.folder / "first_head_rgb.npy", np.asarray(head))
            return result
        env.get_obs = get_obs

        original_stack = model.runner.get_n_steps_obs
        @functools.wraps(original_stack)
        def stack(*args, **kwargs):
            result = original_stack(*args, **kwargs)
            self.event("policy_history", prediction=self.current_prediction,
                       agent_pos=result.get("agent_pos"),
                       head_hashes=[fingerprint(f) for f in result.get("head_cam", [])],
                       obs_shapes={k: list(np.asarray(v).shape) for k, v in result.items()})
            return result
        model.runner.get_n_steps_obs = stack

        original_prediction = model.get_action
        @functools.wraps(original_prediction)
        def prediction(*args, **kwargs):
            self.predict_count += 1
            self.current_prediction = self.predict_count
            result = original_prediction(*args, **kwargs)
            self.event("policy_actions", prediction=self.current_prediction,
                       actions=result, properties=array_info(result))
            return result
        model.get_action = prediction

        original_action = env.take_action
        @functools.wraps(original_action)
        def take_action(action, *args, **kwargs):
            if self.phase == "policy":
                current_step = int(env.take_action_cnt)

                # Preferred stop: 15 actions after the right gripper first becomes truly closed.
                if (
                    self.first_right_close_step is not None
                    and current_step >= self.first_right_close_step + self.stop_after_close_actions
                ):
                    self.event(
                        "diagnostic_post_close_limit_reached",
                        action_step=current_step,
                        first_right_close_step=self.first_right_close_step,
                        stop_after_close_actions=self.stop_after_close_actions,
                    )
                    print(
                        f"[DIAG] seed={self.seed} collected {self.stop_after_close_actions} "
                        f"actions after first right-gripper close; stopping.",
                        flush=True,
                    )
                    raise AuditStop("抓取诊断已采集到闭爪后窗口；这不是任务执行错误。")

                # Safety cap if the policy never closes the gripper.
                if current_step >= self.max_actions:
                    self.event(
                        "diagnostic_action_limit_reached",
                        action_step=current_step,
                        diagnostic_max_actions=self.max_actions,
                    )
                    print(
                        f"[DIAG] seed={self.seed} reached safety cap={self.max_actions}; stopping.",
                        flush=True,
                    )
                    raise AuditStop("达到抓取诊断最大 action 数；这不是任务执行错误。")

                if current_step > 0 and current_step % 5 == 0:
                    print(
                        f"[DIAG] seed={self.seed} policy_action={current_step} "
                        f"first_close={self.first_right_close_step}",
                        flush=True,
                    )

            if env.eval_success or env.take_action_cnt >= env.step_lim:
                return original_action(action, *args, **kwargs)
            mode = kwargs.get("action_type", args[0] if args else "qpos")
            if mode != "qpos":
                raise AuditStop(f"诊断脚本针对当前 14D qpos 接口；实际 action_type={mode}，未改变执行逻辑。")
            arr = np.asarray(action, dtype=float).copy()
            raw_arr = arr.copy()
            pre = robot_state(env)
            robot = env.robot
            n_left = len(pre["left"]["q_cmd"])
            n_right = len(pre["right"]["q_cmd"])
            if arr.shape != (n_left + n_right + 2,) or not np.isfinite(arr).all():
                self.event("invalid_action", action=arr, expected_dim=n_left + n_right + 2)
                raise AuditStop("发现形状错误或 NaN/Inf 动作，已记录并停止，没有送入仿真。")

            # Causal diagnostic intervention:
            # freeze the task-irrelevant arm at its CURRENT commanded state.
            # This does not repair or modify the working-arm DP output.
            # It is only used to test whether cross-arm action interference is causal.
            if self.nominal_arm == "right":
                arr[:n_left] = np.asarray(pre["left"]["q_cmd"], dtype=float)
                arr[n_left] = float(pre["left"]["gripper_cmd_normalized"])
                masked_arm = "left"
            elif self.nominal_arm == "left":
                r0 = n_left + 1
                arr[r0:r0 + n_right] = np.asarray(pre["right"]["q_cmd"], dtype=float)
                arr[r0 + n_right] = float(pre["right"]["gripper_cmd_normalized"])
                masked_arm = "right"
            else:
                raise AuditStop(f"无法确定 nominal working arm: {self.nominal_arm}")

            self.event(
                "arm_mask",
                action_step=int(env.take_action_cnt) + 1,
                nominal_working_arm=self.nominal_arm,
                masked_arm=masked_arm,
                raw_action=raw_arr,
                masked_action=arr,
            )

            row = {"action_step": int(env.take_action_cnt) + 1, "prediction": self.current_prediction,
                   "raw_action": raw_arr.copy(), "action": arr.copy(), "pre": pre, "topp": []}
            self.event("action_begin", action_step=row["action_step"],
                       raw_action=raw_arr, action=arr, state=pre)
            restored = []
            set_calls = {"left": 0, "right": 0}
            try:
                for arm in ("left", "right"):
                    planner = getattr(robot, f"{arm}_mplib_planner")
                    original_topp = planner.TOPP
                    def make_topp(fn, side):
                        @functools.wraps(fn)
                        def call(path, *targs, **tkwargs):
                            item = {"arm": side, "path": np.asarray(path).copy()}
                            row["topp"].append(item)
                            try:
                                output = fn(path, *targs, **tkwargs)
                            except BaseException as exc:
                                item["exception"] = repr(exc)
                                item["traceback"] = traceback.format_exc()
                                self.event("topp_exception", action_step=row["action_step"], **item)
                                raise  # Original take_action decides how to handle it.
                            try:
                                pos = np.asarray(output[1])
                                item.update(n_samples=int(pos.shape[0]), positions=array_info(pos),
                                            duration=float(output[4]))
                                if pos.shape[0]:
                                    item["first_position"] = pos[0].copy()
                                    item["last_position"] = pos[-1].copy()
                            except Exception as exc:
                                item["diagnostic_parse_error"] = repr(exc)
                            self.event("topp_return", action_step=row["action_step"], **item)
                            return output  # Same object; do not alter trajectory.
                        return call
                    planner.TOPP = make_topp(original_topp, arm)
                    restored.append((planner, "TOPP", original_topp))
                original_set = robot.set_arm_joints
                @functools.wraps(original_set)
                def set_arm(position, velocity, arm_tag):
                    key = str(arm_tag)
                    if key in set_calls:
                        set_calls[key] += 1
                    return original_set(position, velocity, arm_tag)
                robot.set_arm_joints = set_arm
                restored.append((robot, "set_arm_joints", original_set))
                result = original_action(arr, *args, **kwargs)
            except BaseException as exc:
                row["raised"] = repr(exc)
                raise
            finally:
                for obj, name, old in reversed(restored):
                    setattr(obj, name, old)
                post = robot_state(env)
                self.last_state = post
                row["post"] = post
                row["eval_success"] = bool(env.eval_success)

                # Record the first action after which the RIGHT gripper command is truly closed.
                try:
                    post_grip = float(post["right"]["gripper_cmd_normalized"])
                    if self.first_right_close_step is None and post_grip < 0.2:
                        self.first_right_close_step = int(row["action_step"])
                        self.event(
                            "first_right_gripper_close",
                            action_step=self.first_right_close_step,
                            right_gripper_cmd=post_grip,
                        )
                        print(
                            f"[DIAG] seed={self.seed} first right-gripper close "
                            f"at action={self.first_right_close_step}",
                            flush=True,
                        )
                except Exception:
                    pass
                targets = {"left": arr[:n_left], "right": arr[n_left + 1:n_left + 1 + n_right]}
                for arm in ("left", "right"):
                    target = targets[arm]
                    row[arm] = {
                        "set_calls": set_calls[arm],
                        "target_minus_cmd_max_rad": max_abs(target - pre[arm]["q_cmd"]),
                        "target_minus_real_max_rad": max_abs(target - pre[arm]["q_real"]),
                        "real_motion_max_rad": max_abs(post[arm]["q_real"] - pre[arm]["q_real"]),
                        "post_tracking_error_max_rad": post[arm]["tracking_error_max_rad"],
                    }
                    limits = self.limit_arrays.get(arm)
                    if limits is not None and limits.shape == (target.size, 2):
                        outside = (target < limits[:, 0]) | (target > limits[:, 1])
                        row[arm]["target_outside_joint_limits_indices"] = np.where(outside)[0].tolist()
                self.actions.append(clean(row))
                self.event("action_end", **row)
                if len(self.actions) == 1 or len(self.actions) % 50 == 0 or env.eval_success:
                    print(f"[DIAG] seed={self.seed} action={row['action_step']} "
                          f"set_calls L/R={set_calls['left']}/{set_calls['right']} "
                          f"real_motion L/R={row['left']['real_motion_max_rad']:.3g}/"
                          f"{row['right']['real_motion_max_rad']:.3g} rad success={env.eval_success}", flush=True)
            return result
        env.take_action = take_action


def load_evaluator(root: Path, case: Path, seed: int):
    """Override ONLY main's test count/start/output path, in memory, fail closed."""
    path = root / "script/eval_policy.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    main = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    if main is None:
        raise RuntimeError("未找到 eval_policy.py::main；请提供新的本地入口代码。")
    counts = {"test_num": 0, "st_seed": 0, "save_dir": 0}
    for node in ast.walk(main):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name == "test_num":
            node.value = ast.Constant(value=1)
            counts[name] += 1
        elif name == "st_seed":
            node.value = ast.Constant(value=int(seed))
            counts[name] += 1
        elif name == "save_dir" and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "Path":
            node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                  args=[ast.Constant(str(case / "eval"))], keywords=[])
            counts[name] += 1
    if counts != {"test_num": 1, "st_seed": 1, "save_dir": 1}:
        raise RuntimeError(f"入口结构与上传代码不同，停止自动诊断，未修改源码：{counts}")
    ast.fix_missing_locations(tree)
    module = types.ModuleType("robotwin_dp_audit_eval")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    write_json(case / "entry_overrides.json", {
        "test_num": 1, "st_seed": seed, "save_dir": str(case / "eval"),
        "source_modified_on_disk": False,
        "original_eval_sha256": hashlib.sha256(text.encode()).hexdigest(),
    })
    return module


def worker(opts) -> int:
    global np
    import numpy as numpy_module
    np = numpy_module
    import yaml
    root = Path(opts.root).expanduser().resolve()
    case = Path(opts.case_dir).resolve()
    os.chdir(root)
    # Match `python script/eval_policy.py`, rather than force-importing the
    # local diffusion_policy package and accidentally hiding a path conflict.
    # eval_policy.py / dp_model.py perform their own original sys.path updates.
    sys.path[0] = str(root / "script")
    rec = Recorder(case, opts.worker_seed, max_actions=opts.diag_max_actions)
    code = 0
    try:
        # Same rendering smoke test as the original __main__ entry point.
        from test_render import Sapien_TEST
        Sapien_TEST()
        module = load_evaluator(root, case, opts.worker_seed)
        print(
            f"[DIAG] seed={opts.worker_seed} evaluator loaded; "
            f"next step loads checkpoint/model ({opts.checkpoint}.ckpt).",
            flush=True,
        )
        original_eval = module.eval_policy
        signature = inspect.signature(original_eval)
        @functools.wraps(original_eval)
        def monitored_eval(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            env, model = bound.arguments["TASK_ENV"], bound.arguments["model"]
            rec.attach(env, model)

            # Causal test: one-action receding-horizon deployment.
            # Keep the checkpoint/policy unchanged. The policy still predicts its
            # original 6-step action sequence, but the runner executes only the
            # first action and replans from the next observation.
            model.runner.n_action_steps = 1

            # deploy_policy.eval() normally appends the post-action observation
            # through model.update_obs(), and the next get_action(obs) appends the
            # same observation again. With one-action replanning that would duplicate
            # every frame. Disable that extra append so get_action() alone advances
            # the history exactly once per physical action.
            def _no_post_action_append(_observation):
                return None
            model.update_obs = _no_post_action_append

            rec.event(
                "deployment_intervention",
                runner_n_action_steps=1,
                post_action_update_obs_disabled=True,
                note="execute 1 action, then replan; avoid duplicate obs append"
            )
            print("[DIAG] intervention: n_action_steps=1, clean obs history", flush=True)

            try:
                result = original_eval(*args, **kwargs)
            finally:
                rec.eval_success = bool(getattr(env, "eval_success", False)) if rec.phase == "policy" else None
            rec.event("eval_result", next_seed=result[0], successes=result[1], evaluated_episodes=1)
            return result
        module.eval_policy = monitored_eval
        cfg = yaml.safe_load((root / "policy/DP/deploy_policy.yml").read_text(encoding="utf-8"))
        cfg.update(policy_name="DP", task_name=opts.task_name, task_config=opts.task_config,
                   ckpt_setting=opts.ckpt_setting, expert_data_num=opts.expert_data_num,
                   seed=opts.train_seed, checkpoint_num=opts.checkpoint)
        write_json(case / "deploy_args.json", cfg)
        module.main(cfg)
        rec.status = "completed"
    except AuditStop as exc:
        rec.status = "stopped_for_diagnostic_condition"
        rec.error = str(exc)
        rec.event("stopped", message=str(exc))
        print(f"[DIAG] STOP: {exc}", flush=True)
        code = 2
    except BaseException as exc:
        rec.status = "error"
        rec.error = repr(exc)
        rec.event("fatal", error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
        code = 1
    finally:
        summary = rec.finish()
        print(f"[DIAG] saved {case / 'summary.json'} status={summary['status']}", flush=True)
    return code


def git_text(root: Path, *args) -> str:
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, errors="replace")
    if p.returncode:
        raise RuntimeError(p.stderr.strip() or "git command failed")
    return p.stdout.strip()


def parent(opts) -> int:
    root = Path(opts.root).expanduser().resolve()
    needed = ["script/eval_policy.py", "policy/DP/deploy_policy.yml", "envs/_base_task.py"]
    for name in needed:
        if not (root / name).is_file():
            raise FileNotFoundError(f"缺少 {root / name}；请先 cd ~/RoboTwin 再运行。")
    branch = git_text(root, "branch", "--show-current")
    if branch != opts.expected_branch:
        raise RuntimeError(f"当前分支是 {branch!r}，预期 {opts.expected_branch!r}。停止运行；脚本不会自动切分支或丢弃修改。")
    checkpoint = root / f"policy/DP/checkpoints/{opts.task_name}-{opts.ckpt_setting}-{opts.expert_data_num}-{opts.train_seed}/{opts.checkpoint}.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint}")
    run = root / "diagnostics" / ("dp_replan1_closewindow_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in ("numpy", "torch", "torchvision", "sapien", "mplib", "toppra", "diffusers", "numba", "zarr"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not registered in package metadata"
    write_json(run / "metadata.json", {
        "root": root, "branch": branch, "commit": git_text(root, "rev-parse", "HEAD"),
        "git_status": git_text(root, "status", "--short"), "python": sys.executable,
        "python_version": sys.version, "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "versions": versions,
        "checkpoint": checkpoint, "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_mtime": checkpoint.stat().st_mtime,
        "arguments": vars(opts),
    })
    files = [
        "script/eval_policy.py", "policy/DP/deploy_policy.py", "policy/DP/dp_model.py",
        "policy/DP/diffusion_policy/env_runner/dp_runner.py", "envs/_base_task.py",
        "envs/robot/robot.py", "envs/robot/planner.py", "envs/place_empty_cup.py",
        "policy/DP/diffusion_policy/config/robot_dp_14.yaml",
        "policy/DP/diffusion_policy/config/task/default_task_14.yaml",
        "policy/DP/diffusion_policy/workspace/robotworkspace.py", "policy/DP/train.sh",
        f"task_config/{opts.task_config}.yml", "task_config/_eval_step_limit.yml",
    ]
    hashes = {}
    for name in files:
        path = root / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(run / "source_hashes_before.json", hashes)
    shutil.copyfile(Path(__file__).resolve(), run / "diagnose_dp_grasp_geometry_used.py")
    print(f"诊断目录：{run}\n分支：{branch}；checkpoint：{checkpoint.name}\n"
          "每个 seed 一个独立进程；只测指定场景，不重训、不覆盖源码。", flush=True)
    results = []
    exit_code = 0
    try:
        for seed in opts.eval_seeds:
            case = run / f"seed_{seed}"
            case.mkdir()
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                       "--root", str(root), "--case-dir", str(case), "--worker-seed", str(seed),
                       "--task-name", opts.task_name, "--task-config", opts.task_config,
                       "--ckpt-setting", opts.ckpt_setting, "--expert-data-num", str(opts.expert_data_num),
                       "--train-seed", str(opts.train_seed), "--checkpoint", str(opts.checkpoint),
                       "--diag-max-actions", str(opts.diag_max_actions)]
            print(f"\n开始 seed={seed}。完整控制台输出保存至 {case / 'console.log'}", flush=True)
            tail = deque(maxlen=20)
            with (case / "console.log").open("w", encoding="utf-8", buffering=1) as log:
                proc = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, errors="replace", bufsize=1)
                try:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        log.write(line)
                        tail.append(line.rstrip())
                        if "[DIAG]" in line:
                            print(line.rstrip(), flush=True)
                    rc = proc.wait()
                except KeyboardInterrupt:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise
            summary_path = case / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"status": "process_exited_without_summary"}
            summary["returncode"] = rc
            summary["eval_seed"] = seed
            results.append(summary)
            if rc:
                exit_code = 1
                print(f"seed={seed} 未正常完成（返回码 {rc}），日志末尾：\n" + "\n".join(tail), flush=True)
            else:
                print(f"seed={seed} 完成：policy_success={summary.get('policy_success')}", flush=True)
    except KeyboardInterrupt:
        exit_code = 130
        print("\n已中断，将打包已获得的日志。", flush=True)
    finally:
        write_json(run / "combined_summary.json", results)
        after = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                 for name in hashes if (root / name).is_file()}
        write_json(run / "source_hashes_after.json", after)
        write_json(run / "source_unchanged.json", {"unchanged": hashes == after})
        archive = run.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(run.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(run.parent))
        print(f"\n诊断包：{archive}\n请上传这个 ZIP；其中含 grasp_geometry.csv、完整日志、逐步动作/关节记录和原流程视频。", flush=True)
    return exit_code


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".")
    p.add_argument("--checkpoint", type=int, default=1000)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[100000, 100001])
    p.add_argument("--task-name", default="place_empty_cup")
    p.add_argument("--task-config", default="dp_test_demo")
    p.add_argument("--ckpt-setting", default="dp_test_demo")
    p.add_argument("--expert-data-num", type=int, default=50)
    p.add_argument("--expected-branch", default="robotwin_dp")
    p.add_argument("--diag-max-actions", type=int, default=120,
                   help="Safety cap; preferred stop is 15 actions after the first right-gripper close.")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--case-dir", help=argparse.SUPPRESS)
    p.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    return p.parse_args()


if __name__ == "__main__":
    options = parse_args()
    try:
        raise SystemExit(worker(options) if options.worker else parent(options))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
