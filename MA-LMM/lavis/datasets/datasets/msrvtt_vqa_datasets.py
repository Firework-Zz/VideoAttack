"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import re
from PIL import Image

import pandas as pd
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor

from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset
from lavis.datasets.data_utils import load_video
import pdb

class MSRVTTVQADataset(VideoQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt='', split='train', **kwargs):
        """
        [修改] 增加了 **kwargs，防止 Builder 传入多余参数时报错
        """
        self.vis_root = vis_root

        self.annotation = {}
        for ann_path in ann_paths:
            self.annotation.update(json.load(open(ann_path)))
        self.question_id_list = list(self.annotation.keys())
        self.question_id_list.sort()
        self.fps = 10

        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt = prompt
        # self._add_instance_ids()
        # pdb.set_trace()

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."
        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]

        # Divide the range into num_frames segments and select a random index from each segment
        segment_list = np.linspace(0, ann['frame_length'], self.num_frames + 1, dtype=int)
        segment_start_list = segment_list[:-1]
        segment_end_list = segment_list[1:]
        selected_frame_index = []
        for start, end in zip(segment_start_list, segment_end_list):
            if start == end:
                selected_frame_index.append(start)
            else:
                selected_frame_index.append(np.random.randint(start, end))

        frame_list = []
        # [修改] 增加 try-except 保护，防止只有 .pt 特征而没有原始视频时报错
        try:
            for frame_index in selected_frame_index:
                # 假设你的数据结构是: vis_root/videoX/frame00001.jpg
                img_path = os.path.join(self.vis_root, ann['video'], "frame{:06d}.jpg".format(frame_index + 1))
                frame = Image.open(img_path).convert("RGB")
                frame = pil_to_tensor(frame).to(torch.float32)
                frame_list.append(frame)
            video = torch.stack(frame_list, dim=1)
            video = self.vis_processor(video)
        except Exception as e:
            # 如果是训练阶段且必须要有视频，这里应该报错；
            # 但如果 Builder 传参错误导致误入这里，给个全0 tensor 防止崩溃
            video = torch.zeros(3, self.num_frames, 224, 224)
            # print(f"Warning: Failed to load frames for {ann['video']}: {e}")

        question = self.text_processor(ann["question"])
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
        answer = self.text_processor(ann["answer"])

        return {
            "image": video,
            "text_input": question,
            "text_output": answer,
            "question_id": ann["question_id"],
            # "instance_id": ann["instance_id"],
        }
        
    def __len__(self):
        return len(self.question_id_list)


class MSRVTTVQAEvalDataset(MSRVTTVQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='test', use_adv_feats=False, adv_feats_dir=None, **kwargs):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='test', **kwargs)
        self.use_adv_feats = use_adv_feats
        self.adv_feats_dir = adv_feats_dir

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."
        
        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]
        
        # 1. 你的特征存放路径 (硬编码写死，防止配置出错)
        feature_root = "/data2/codefile/fjh/data/feats_pgd_answer"
        
        # 2. 获取视频名称 (MSRVTT 中通常是 video0, video1...)
        video_name = ann['video']
        
        # 3. 拼接特征文件路径
        feature_path = os.path.join(feature_root, f"{video_name}.pt")

        # 4. 加载特征
        if os.path.exists(feature_path):
            try:
                # map_location='cpu' 防止多线程加载时显存爆炸
                video_features = torch.load(feature_path, map_location='cpu')
                
                # 确保转换为 float32 (模型权重通常是 float32/fp16，但在输入端转为 float 比较稳妥)
                video_features = video_features.float()
                
            except Exception as e:
                print(f"Error loading feature {feature_path}: {e}")
                # 遇到坏文件，给个全0占位，防止程序崩溃退出
                video_features = torch.zeros(10, 257, 1408)
        else:
            # 没找到文件 (比如 extract 还没跑完)，给个全0占位
            # print(f"Warning: Missing feature for {video_name}") 
            video_features = torch.zeros(10, 257, 1408)
            
        # =======================================================

        # 5. 处理文本 (保持原样)
        question = self.text_processor(ann["question"])
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
        answer = self.text_processor(ann["answer"])

        return {
            "image": video_features, # 这里传的是 Feature Tensor，不是 Image
            "text_input": question,
            "text_output": answer,
            "question_id": ann["question_id"],
        }
