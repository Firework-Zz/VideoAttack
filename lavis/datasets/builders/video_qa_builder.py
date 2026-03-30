"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import lavis.common.utils as utils
import os
import warnings # 【修复】添加了 warnings 模块导入

from lavis.common.registry import registry
from lavis.common.utils import get_cache_path
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset
from lavis.datasets.datasets.msvd_vqa_datasets import MSVDVQADataset, MSVDVQAEvalDataset
from lavis.datasets.datasets.msrvtt_vqa_datasets import MSRVTTVQADataset, MSRVTTVQAEvalDataset
from lavis.datasets.datasets.activitynet_vqa_datasets import ActivityNetVQADataset, ActivityNetVQAEvalDataset

class VideoQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoQADataset
    eval_dataset_cls = VideoQADataset

    def build(self):
        datasets = super().build()

        # MSRVTT 等数据集通常需要 ans2label
        ans2label = self.config.build_info.annotations.get("ans2label")
        if ans2label is None:
            # 如果是某些不需要 ans2label 的数据集，这里可以改为 pass 或 log warning
            # 但 MSRVTT-QA 是需要的
            pass 
        else:
            ans2label = get_cache_path(ans2label.storage)
            for split in datasets:
                if hasattr(datasets[split], '_build_class_labels'):
                    datasets[split]._build_class_labels(ans2label)

        return datasets


@registry.register_builder("msrvtt_qa")
class MSRVTTQABuilder(VideoQABuilder):
    train_dataset_cls = MSRVTTVQADataset
    eval_dataset_cls = MSRVTTVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_qa.yaml",
    }
    
    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        # LAVIS 中 MSRVTT 的 data_type 默认通常是 "images"
        vis_info = build_info.get(self.data_type) 

        ans2label = ann_info.get("ans2label")
        if ans2label is None:
             # 为了避免没配 ans2label 报错，先给个警告，或者确保 YAML 里配了
            warnings.warn("ans2label is not specified in build_info.")
            ans2label_path = None
        else:
            ans2label_path = get_cache_path(ans2label.storage)

        # 【核心修改】从 self.config 获取攻击配置（对应 YAML 中的 datasets.msrvtt_qa 下的参数）
        use_adv_feats = self.config.get("use_adv_feats", False)
        adv_feats_dir = self.config.get("adv_feats_dir", "")
            
        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )
            
            # 获取 prompt
            # 注意：需确保 YAML 的 text_processor 下有 prompt 字段
            tp_cfg = self.config.get("text_processor")
            if is_train:
                prompt = tp_cfg.get('train').get('prompt', "Question: {} Short answer:")
            else:
                prompt = tp_cfg.get('eval').get('prompt', "")

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                vis_path = utils.get_cache_path(vis_path)

            # 【修复】这里不再强行报错，只是警告
            # 因为如果是 adv 模式，vis_path 其实是用不到的
            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist. (If you are using adv_feats, please ignore this)".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                num_frames=self.config.num_frames,
                prompt=prompt,
                # 【逻辑】只在 eval/test 阶段且 use_adv_feats=True 时启用攻击特征
                use_adv_feats=(use_adv_feats and not is_train),
                adv_feats_dir=adv_feats_dir if (use_adv_feats and not is_train) else None,
            )
            
            if ans2label_path:
                datasets[split]._build_class_labels(ans2label_path)

        return datasets

@registry.register_builder("msvd_qa")
class MSVDQABuilder(VideoQABuilder):
    train_dataset_cls = MSVDVQADataset
    eval_dataset_cls = MSVDVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_qa.yaml",
    }

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        ans2label = ann_info.get("ans2label")
        if ans2label is None:
             # MSVD 也可以选择性警告
             pass
        ans2label_path = get_cache_path(ans2label.storage) if ans2label else None
            
        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            vis_processor = self.vis_processors["train"] if is_train else self.vis_processors["eval"]
            text_processor = self.text_processors["train"] if is_train else self.text_processors["eval"]
            
            tp_cfg = self.config.get("text_processor")
            if is_train:
                prompt = tp_cfg.get('train').get('prompt', "Question: {} Short answer:")
            else:
                prompt = tp_cfg.get('eval').get('prompt', "")

            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]
            
            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            vis_path = vis_info.storage
            if not os.path.isabs(vis_path):
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                num_frames=self.config.num_frames,
                prompt=prompt,
            )
            if ans2label_path:
                datasets[split]._build_class_labels(ans2label_path)

        return datasets

@registry.register_builder("activitynet_qa")
class ActivityNetQABuilder(VideoQABuilder):
    train_dataset_cls = ActivityNetVQADataset
    eval_dataset_cls = ActivityNetVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/activitynet/defaults_qa.yaml",
    }
    
    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            vis_processor = self.vis_processors["train"] if is_train else self.vis_processors["eval"]
            text_processor = self.text_processors["train"] if is_train else self.text_processors["eval"]
            
            tp_cfg = self.config.get("text_processor")
            if is_train:
                prompt = tp_cfg.get('train').get('prompt')
            else:
                prompt = tp_cfg.get('eval').get('prompt')

            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]
            
            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            vis_path = vis_info.storage
            if not os.path.isabs(vis_path):
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                num_frames=self.config.num_frames,
                prompt=prompt,
            )

        return datasets
