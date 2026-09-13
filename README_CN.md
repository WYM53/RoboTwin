<h1 align="center">RoboTwin 2.0</h1>

<p align="center"><b>面向双臂机器人操作的数据生成与评测平台</b></p>

<p align="center">
  <a href="https://robotwin-platform.github.io/">项目主页</a> ·
  <a href="https://robotwin-platform.github.io/doc/">使用文档</a> ·
  <a href="https://arxiv.org/abs/2506.18088">论文</a> ·
  <a href="https://robotwin-platform.github.io/leaderboard">排行榜</a> ·
  <a href="https://robotwin-platform.github.io/doc/community/index.html">社区</a>
</p>

<p align="center"><a href="./README.md">English</a> | <b>简体中文</b></p>

https://private-user-images.githubusercontent.com/88101805/463126988-e3ba1575-4411-4a36-ad65-f0b2f49890c3.mp4

## 项目简介

RoboTwin 是一个面向双臂机器人操作研究的仿真平台。它将任务构建、专家轨迹生成、域随机化、策略训练与统一评测整合在同一套工作流中，可用于研究视觉鲁棒性、多任务学习、语言条件控制以及跨机器人本体泛化。

当前仓库对应 **RoboTwin 2.0**，内置 50 个操作任务，并提供超过 10 万条预采集轨迹以及多种主流策略基线。平台支持自由组合任务、相机、机器人本体和数据模态，适合快速构建不同规模的双臂操作实验。

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%" alt="RoboTwin 2.0 任务展示">
</p>

## 核心能力

- **自动生成专家数据**：先搜索能够成功完成任务的随机种子，再回放轨迹并保存训练数据。
- **强域随机化**：支持背景、光照、桌面高度、相机位置和桌面杂物等随机化设置。
- **多机器人本体**：可配置同构双臂或由两个不同机械臂组成的异构双臂系统。
- **多模态观测**：支持 RGB、深度、点云、关节位置、末端位姿及分割信息。
- **统一策略评测**：仓库集成多种模仿学习和视觉—语言—动作策略，并提供自定义策略接口。
- **语言指令生成**：可为采集到的 episode 生成多样化任务指令。

## 快速开始

### 1. 获取代码

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
```

推荐在独立的 Conda 环境中安装。完整的环境要求和安装说明请先阅读[官方安装文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)。仓库提供的基础安装脚本会安装 Python 依赖、PyTorch3D 和 CuRobo，并对部分依赖做兼容性调整：

```bash
bash script/_install.sh
```

### 2. 下载仿真资源

任务运行依赖背景纹理、物体模型和机器人本体资源：

```bash
bash script/_download_assets.sh
```

下载完成后，脚本会解压资源并更新机器人本体配置中的本地路径。

### 3. 检查渲染环境

```bash
python script/test_render.py
```

### 4. 采集第一组数据

```bash
bash collect_data.sh beat_block_hammer demo_randomized 0
```

三个参数依次表示：

1. `beat_block_hammer`：任务名称，对应 `envs/beat_block_hammer.py`；
2. `demo_randomized`：配置名称，对应 `task_config/demo_randomized.yml`；
3. `0`：使用的 GPU 编号。

程序会先搜索足够数量的成功随机种子，然后回放对应轨迹并写入：

```text
data/<任务名称>/<配置名称>/
```

## 任务与配置

### 内置任务

本仓库包含 50 个双臂操作任务，任务实现位于 `envs/`。部分示例如下：

| 类型 | 示例任务 |
| --- | --- |
| 抓取与放置 | `place_object_basket`、`place_phone_stand`、`place_bread_skillet` |
| 双臂协作 | `handover_block`、`handover_mic`、`pick_dual_bottles` |
| 堆叠与排序 | `stack_blocks_three`、`stack_bowls_two`、`blocks_ranking_rgb` |
| 工具与设备操作 | `beat_block_hammer`、`open_laptop`、`open_microwave` |
| 精细操作 | `rotate_qrcode`、`stamp_seal`、`turn_switch` |

完整任务说明和成功条件请查看[任务文档](https://robotwin-platform.github.io/doc/tasks/index.html)。

### 默认配置

| 配置 | 用途 |
| --- | --- |
| `demo_clean.yml` | 干净场景，不启用背景、光照和桌面杂物随机化 |
| `demo_randomized.yml` | 开启背景、光照、桌面高度和杂物等域随机化 |

创建自定义配置：

```bash
bash task_config/create_task_config.sh my_config
```

随后编辑 `task_config/my_config.yml`。常用字段包括：

```yaml
episode_num: 50
embodiment: [aloha-agilex]
language_num: 100

domain_randomization:
  random_background: true
  cluttered_table: true
  random_table_height: 0.03
  random_light: true

camera:
  head_camera_type: D435
  wrist_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: true

data_type:
  rgb: true
  depth: false
  pointcloud: false
  endpose: true
  qpos: true
```

所有配置项的含义请参考[任务配置文档](https://robotwin-platform.github.io/doc/usage/configurations.html)。

## 数据集

官方发布的 [RoboTwin 2.0 数据集](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset)包含超过 10 万条预采集轨迹，可直接用于策略训练和基准复现。

由于任务、机器人本体、相机和域随机化均可自由配置，如果公开数据与实验设置不完全一致，建议使用本仓库自行采集数据。

<p align="center">
  <img src="./assets/files/domain_randomization.png" width="100%" alt="RoboTwin 域随机化效果">
</p>

## 策略基线

当前仓库包含以下策略实现或适配代码：

- [Diffusion Policy](https://robotwin-platform.github.io/doc/usage/DP.html)
- [ACT](https://robotwin-platform.github.io/doc/usage/ACT.html)
- [DP3](https://robotwin-platform.github.io/doc/usage/DP3.html)
- [RDT](https://robotwin-platform.github.io/doc/usage/RDT.html)
- [π0](https://robotwin-platform.github.io/doc/usage/Pi0.html) 与 π0.5
- [OpenVLA-OFT](https://robotwin-platform.github.io/doc/usage/OpenVLA-oft.html)
- [TinyVLA](https://robotwin-platform.github.io/doc/usage/TinyVLA.html)
- [DexVLA](https://robotwin-platform.github.io/doc/usage/DexVLA.html)
- [LLaVA-VLA](https://robotwin-platform.github.io/doc/usage/LLaVA-VLA.html)
- [GO-1](https://robotwin-platform.github.io/doc/usage/GO1.html)

不同策略的依赖、数据预处理、训练和评测命令并不相同，请进入对应的 `policy/<策略名称>/` 目录并阅读其文档。如需接入自己的模型，可从 `policy/Your_Policy/` 开始，并参考[自定义策略部署指南](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)。

## 推荐评测方向

RoboTwin 2.0 的统一设置可用于比较：

- 单任务微调能力；
- 视觉变化下的鲁棒性；
- 不同语言表达下的指令遵循能力；
- 多任务联合学习能力；
- 跨机器人本体的迁移与泛化性能。

评测协议、提交方法和最新结果见 [RoboTwin 2.0 排行榜](https://robotwin-platform.github.io/leaderboard)。

## 项目结构

```text
RoboTwin/
├── assets/             # 机器人、物体和背景等仿真资源
├── envs/               # 任务环境、机器人、相机与通用组件
├── task_config/        # 相机、本体和数据采集配置
├── script/             # 安装、采集、评测和数据处理脚本
├── policy/             # 策略基线及自定义策略模板
├── description/        # 物体描述与语言指令生成工具
├── code_gen/           # 任务代码生成相关工具
├── data/               # 默认的数据输出目录
└── collect_data.sh     # 数据采集入口
```

## 相关版本

| 版本或用途 | 分支 |
| --- | --- |
| RoboTwin 2.0 | [`main`](https://github.com/RoboTwin-Platform/RoboTwin/tree/main) |
| RoboTwin 1.0 | [`RoboTwin-1.0`](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0) |
| IsaacLab-Arena 支持 | [`IsaacLab-Arena`](https://github.com/RoboTwin-Platform/RoboTwin/tree/IsaacLab-Arena) |
| RLinf 支持 | [`RLinf_support`](https://github.com/RoboTwin-Platform/RoboTwin/tree/RLinf_support) |
| WBCD 2026 | [`WBCD-2026`](https://github.com/RoboTwin-Platform/RoboTwin/tree/WBCD-2026) |
| 早期版本 | [`early_version`](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version) |

## 引用

如果 RoboTwin 对您的研究有所帮助，请引用：

```bibtex
@article{chen2025robotwin,
  title={Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Li, Zixuan and Liang, Qiwei and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

```bibtex
@InProceedings{Mu_2025_CVPR,
  author    = {Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and Lan, Zhiqian and Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and Zou, Yude and Xu, Mingkun and Lin, Lunkai and Xie, Zhiqiang and Ding, Mingyu and Luo, Ping},
  title     = {RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {27649-27660}
}
```

## 致谢

感谢 D-Robotics 提供软件支持、AgileX Robotics 提供硬件支持，以及 Deemos 提供 AIGC 支持。

如有问题或建议，请通过 [RoboTwin 社区](https://robotwin-platform.github.io/doc/community/index.html)交流，或联系 [Tianxing Chen](https://tianxingchen.github.io)。

## 开源协议

本项目基于 [MIT License](./LICENSE) 开源。
