import glob
import os.path
import pickle
import random
from multiprocessing import Pool
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import KernelPCA

from src.self_logger import logger

class ProbioticDataset(Dataset):
    def __init__(
            self,
            dest_path: str,
            tokenizer: PreTrainedTokenizerBase,
            dataset_name: str = "FB",
            split: str = "train",
            max_length: int = 512,
            seed: int = 42,
            **kwargs
    ):
        super(ProbioticDataset, self).__init__()
        np.random.seed(seed)
        random.seed(seed)
        dataset_name = str(dataset_name)
        
        data_path = os.path.join(dest_path, dataset_name, split + ".txt")
        assert os.path.exists(data_path), f"Data path {data_path} does not exist."

        with open(data_path, "r") as f:
            data = f.read().split("\n")

        # sample.txt
        self.data_path = data_path
        # pkl path list
        self.sequences = data
        self.max_length = max_length
        self.labels = None
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self._data_collator = DataCollatorWithPadding(self.tokenizer)

    def __len__(self): 
        return len(self.sequences)

    def __getitem__(self, idx):
        with open(self.sequences[idx], "rb") as f:
            data = pickle.load(f)
        try:
            seq = data["Seq"][:self.max_length]
            label = data["Label"]
        except:
            seq = data["seq"][:self.max_length]
            label = data["label"]
        return {    
            'labels': label,
            "input_ids": self.tokenizer(seq)["input_ids"],
        }

    def data_collator(self, features: List[Dict[str, Any]]):
        input_ids, batch_label, attention_mask = [], [], []
        max_length = max([len(s['input_ids']) for s in features])

        for sample in features:
            input_id = np.zeros(max_length, dtype=np.float32)
            sample_id = sample['input_ids']
            mask = [1] * len(sample_id) + [0] * (max_length - len(sample_id))
            attention_mask.append(mask)
            input_id[:len(sample_id)] = sample_id
            input_ids.append(input_id)
            batch_label.append(sample['labels'])
        input_ids, batch_label, attention_mask = np.array(input_ids), np.array(batch_label), np.array(attention_mask)
        input_ids = torch.tensor(input_ids).long()
        batch_label = torch.tensor(batch_label).long()
        attention_mask = torch.tensor(attention_mask).long()

        return {"input_ids": input_ids, "labels": batch_label, "attention_mask": attention_mask}

class ProbioticEnhanceRepresentationDataset(Dataset):
    def __init__(
            self,
            dest_path: str,
            _dest_path: str,
            dataset_name: str = "",
            split: str = "train",
            seed: int = 42,
            only_features: bool = False,
            **kwargs
    ):
        super(ProbioticEnhanceRepresentationDataset, self).__init__()
        np.random.seed(seed)
        random.seed(seed)
        split = str(split)

        embedding_txt_path = os.path.join(dest_path, dataset_name, split + ".txt")
        assert os.path.exists(embedding_txt_path), f"Data path {embedding_txt_path} does not exist."
        with open(embedding_txt_path, "r") as f:
            embedding_paths = f.read().split("\n")
        if embedding_paths[-1] == "":
            embedding_paths = embedding_paths[:-1]

        if os.path.exists(os.path.join(_dest_path, dataset_name, split + ".txt")) or os.path.exists(os.path.join(_dest_path, dataset_name, "all.txt")):
            if os.path.exists(os.path.join(_dest_path, dataset_name, split + ".txt")):
                manual_feature_txt_path = os.path.join(_dest_path, dataset_name, split + ".txt")
            else:
                manual_feature_txt_path = os.path.join(_dest_path, dataset_name, "all.txt")
            with open(manual_feature_txt_path, "r") as f:
                manual_feature_paths = f.read().split("\n")

        self.embedding_paths = embedding_paths
        self.manual_feature_paths = manual_feature_paths
        self.dest_path = dest_path
        self._dest_path = _dest_path
        self.only_features = only_features

    def __len__(self):
        return len(self.embedding_paths)

    def __getitem__(self, idx):
        with open(self.embedding_paths[idx], "rb") as f:
            embedding_dict = pickle.load(f)

        for path in self.manual_feature_paths:
            if path.endswith(self.embedding_paths[idx].split("/")[-1]):
                with open(path, "rb") as f:
                    manual_feature_dict = pickle.load(f)
                    break

        sample_name = embedding_dict['sample_name']
        seqs_paths = embedding_dict['seqs_paths']
        seqs_labels = embedding_dict['seqs_labels'][0] if isinstance(embedding_dict['seqs_labels'], list) else embedding_dict['seqs_labels']
        embedding = embedding_dict['model_predict']['embedding']
        manual_feature = []
        # Some orf fragments are filtered, so we need to align them here
        for path in seqs_paths:
            try:
                index = manual_feature_dict['seqs_paths'].index(path)
                manual_feature.append(manual_feature_dict['model_predict']['embedding'][index])
            except:
                pass
        if len(manual_feature) != len(embedding):
            raise ValueError(f"manual_feature and embedding length not equal, manual_feature: {len(manual_feature)}, embedding: {len(embedding)}")
        if self.only_features:
            return {
                "seqs_labels": seqs_labels,
                'manual_feature': manual_feature,
                "embedding": embedding,
            }
        return {
            "sample_name": sample_name,
            'seqs_paths': seqs_paths,
            "seqs_labels": seqs_labels,
            'manual_feature': manual_feature,
            "embedding": embedding,
        }
    def data_collator(self, features: List[Dict[str, Any]]):
        # 如果除了input_ids、token_type_ids、attention_mask、labels外，还有其他的自定义的input，就需要自定义data_collator
        # 官方的 datacollector没法对其他字段做padding
        max_length = max([len(s['embedding']) for s in features])
        feature_dim = len(features[0]['embedding'][1])

        batch_label, manual_feature, embedding = [], [], []

        for sample in features:
            # 处理 manual_feature
            sample_manual_feature = sample['manual_feature']
            max_manual_feature = np.zeros((max_length, feature_dim), dtype=np.float32)
            max_manual_feature[:len(sample_manual_feature), :] = sample_manual_feature
            manual_feature.append(max_manual_feature)

            # 处理 embedding
            sample_embedding = sample['embedding']
            max_embedding = np.zeros((max_length, feature_dim), dtype=np.float32)
            max_embedding[:len(sample_embedding), :] = sample_embedding
            embedding.append(max_embedding)

            # 添加标签
            batch_label.append(sample['seqs_labels'])


        batch_label, manual_feature, embedding = np.array(batch_label), np.array(manual_feature), np.array(embedding)
        batch_label = torch.tensor(batch_label).long()
        manual_feature = torch.tensor(manual_feature)
        embedding = torch.tensor(embedding)

        return {"seqs_labels": batch_label, "manual_feature": manual_feature, "embedding": embedding}


class ProbioticAllSplitFeatureFusionDataset(Dataset):
    def __init__(
            self,
            dest_path: str,
            _dest_path: str = "",
            dataset_name: str = "",
            split: str = "train",
            seed: int = 42,
            # only_features: bool = False,
            **kwargs
    ): 
        super(ProbioticAllSplitFeatureFusionDataset, self).__init__()
        np.random.seed(seed)
        random.seed(seed)
        split = str(split)

        # 第一个特征需要有txt文件
        embedding_txt_path = os.path.join(dest_path, dataset_name, split + ".txt")
        assert os.path.exists(embedding_txt_path), f"Data path {embedding_txt_path} does not exist."
        with open(embedding_txt_path, "r") as f:
            embedding_paths = f.read().split("\n")
        if embedding_paths[-1] == "":
            embedding_paths = embedding_paths[:-1]
        # # 只测试两个样本
        # embedding_paths = embedding_paths[:2]

        if _dest_path != "":
            # 第二个特征，先检查有没有train.txt/val.txt/test.txt/all.txt
            if os.path.exists(os.path.join(_dest_path, dataset_name, split + ".txt")) or os.path.exists(os.path.join(_dest_path, dataset_name, "all.txt")):
                if os.path.exists(os.path.join(_dest_path, dataset_name, split + ".txt")):
                    manual_feature_txt_path = os.path.join(_dest_path, dataset_name, split + ".txt")
                else:
                    manual_feature_txt_path = os.path.join(_dest_path, dataset_name, "all.txt")
                with open(manual_feature_txt_path, "r") as f:
                    manual_feature_paths = f.read().split("\n")
            else:
                # 如果没有txt文件，说明是最原始的人工特征
                manual_feature_paths = []
                species_names = os.listdir(_dest_path)
                for path in embedding_paths:
                    for species_name in species_names:
                        if os.path.exists(f"{_dest_path.rstrip('/')}/{species_name}/{path.split('/')[-1]}"):
                            manual_feature_paths.append(f"{_dest_path.rstrip('/')}/{species_name}/{path.split('/')[-1]}")
            
            self.manual_feature_paths = manual_feature_paths
            self.manual_feature = []

        self._dest_path = _dest_path
        self.embedding_paths = embedding_paths
        self.dest_path = dest_path
        # self.only_features = only_features

        self.seqs_labels = []
        self.embedding = []
        # 读取self.embedding_paths的所有样本的所有片段
        for idx in range(len(self.embedding_paths)):
            with open(self.embedding_paths[idx], "rb") as f:
                embedding_dict = pickle.load(f)
            embedding = embedding_dict['model_predict']['embedding']
            seqs_paths = embedding_dict['seqs_paths']

            if _dest_path != "":
                # 找到一个与self.embedding_paths[idx]对应的样本
                for path in self.manual_feature_paths:
                    if path.endswith(self.embedding_paths[idx].split("/")[-1]):
                        with open(path, "rb") as f:
                            manual_feature_dict = pickle.load(f)
                            break
                # 有些orf片段被过滤了，此处对齐
                manual_feature = []
                for path in seqs_paths:
                    try:
                        index = manual_feature_dict['seqs_paths'].index(path)
                        manual_feature.append(manual_feature_dict['model_predict']['embedding'][index])
                    except:
                        pass
                if len(manual_feature) != len(embedding):
                    raise ValueError(f"manual_feature and embedding length not equal, manual_feature: {len(manual_feature)}, embedding: {len(embedding)}")
                self.manual_feature.extend(manual_feature)
            if len(embedding_dict['seqs_labels'])==1:
                self.seqs_labels.extend([embedding_dict['seqs_labels']]*len(embedding))
            else:
                self.seqs_labels.extend(embedding_dict['seqs_labels'])
            self.embedding.extend(embedding)
        # 输出self.embedding的长度
        print(f"embedding length: {len(self.embedding)}")

    # 定义一个方法，用于返回当前对象的长度
    def __len__(self):
        # 返回当前对象的embedding属性的长度
        return len(self.embedding)

    def __getitem__(self, idx):
        if self._dest_path != "":
            return {
                "labels": self.seqs_labels[idx],
                # 此处是为了在输入cross attn之前在第0维增加1个维度
                'manual_feature': [self.manual_feature[idx]],
                "embedding": [self.embedding[idx]],
            }
        else:
            return {
                "labels": self.seqs_labels[idx],
                "embedding": self.embedding[idx],
            }
            

class ProbioticSplitEnhanceRepresentationDataset(Dataset):
    def __init__(
            self,
            seqs_labels: int,
            manual_feature,
            embedding,
            seed: int = 42,
            **kwargs
    ):
        super(ProbioticSplitEnhanceRepresentationDataset, self).__init__()
        np.random.seed(seed)
        random.seed(seed)   

        self.seqs_labels = seqs_labels
        self.manual_feature = manual_feature
        self.embedding = embedding
        self.pca = KernelPCA(n_components=132, kernel='rbf')
 
        # LLM representation and engineering feature global normalization
        embed_min = np.min(self.embedding)
        embed_max = np.max(self.embedding)
        self.embedding = (self.embedding - embed_min) / (embed_max - embed_min)
        mf_min = np.min(self.manual_feature)
        mf_max = np.max(self.manual_feature)
        self.manual_feature = (self.manual_feature - mf_min) / (mf_max - mf_min)
        
        # LLM representation dimensionality reduction
        self.embedding = self.pca.fit_transform(self.embedding)

        # LLM representation global normalization
        embed_min = np.min(self.embedding)
        embed_max = np.max(self.embedding)
        self.embedding = (self.embedding - embed_min) / (embed_max - embed_min)

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        # transform to tensor and add a dimension at 0
        embedding = torch.tensor(self.embedding[idx]).unsqueeze(0)
        manual_feature = torch.tensor(self.manual_feature[idx]).unsqueeze(0)

        return {    
            'labels': self.seqs_labels[0],
            "embedding": embedding,
            "manual_feature": manual_feature,
        }

class ProbioticSplitFeatureFusionDataset(Dataset):
    def __init__(
            self,
            seqs_labels: int,
            manual_feature,
            embedding,
            fusion_strategy : str = 'finetune',
            seed: int = 42,
            **kwargs
    ):
        super(ProbioticSplitFeatureFusionDataset, self).__init__()
        np.random.seed(seed)
        random.seed(seed)   

        self.seqs_labels = seqs_labels
        self.manual_feature = manual_feature
        self.embedding = embedding
        self.fusion_strategy = fusion_strategy
        self.pca = KernelPCA(n_components=100, kernel='rbf')

        if fusion_strategy not in ["concat", "finetune"]:
            # 全局归一化 
            embed_min = np.min(self.embedding)
            embed_max = np.max(self.embedding)
            self.embedding = (self.embedding - embed_min) / (embed_max - embed_min)
            mf_min = np.min(self.manual_feature)
            mf_max = np.max(self.manual_feature)
            self.manual_feature = (self.manual_feature - mf_min) / (mf_max - mf_min)
            
            # 使用KernelPCA将embedding降维到100维度
            self.embedding = self.pca.fit_transform(self.embedding)

            # 使用RF进行特征排序并选择前100个特征
            best_rf = RandomForestClassifier()
            best_rf.fit(self.manual_feature, [self.seqs_labels]*len(self.manual_feature))
            importances = best_rf.feature_importances_
            indices = np.argsort(importances)[::-1][:100]
            self.manual_feature = self.manual_feature[:, indices]


            # 为了避免KernelPCA之后特征消失，再次进行全局归一化
            embed_min = np.min(self.embedding)
            embed_max = np.max(self.embedding)
            self.embedding = (self.embedding - embed_min) / (embed_max - embed_min)
            mf_min = np.min(self.manual_feature)
            mf_max = np.max(self.manual_feature)
            self.manual_feature = (self.manual_feature - mf_min) / (mf_max - mf_min)
        else:
            pass

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        if self.fusion_strategy == "cross_attention" or self.fusion_strategy == "finetune":
            # 将列表转换为tensor，并在第0维增加一个维度
            embedding = torch.tensor(self.embedding[idx]).unsqueeze(0)
            manual_feature = torch.tensor(self.manual_feature[idx]).unsqueeze(0)
        elif self.fusion_strategy == "first" or self.fusion_strategy == "second" or self.fusion_strategy == "concat":
            embedding = self.embedding[idx]
            manual_feature = self.manual_feature[idx]
        else:
            raise ValueError(f"fusion_strategy {self.fusion_strategy} is not supported.")
        return {    
            'labels': self.seqs_labels,
            "embedding": embedding,
            "manual_feature": manual_feature,
        }

    
class ProbioticsDataProcess:
    def __init__(self, seed=42):
        self.seed = seed
        self._set_random_seed(self.seed)

    @staticmethod
    def _set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)


    '''
    output:
    /RCtrlPGB_GPB/sample_task/GPB.txt
    /RCtrlPGB_GPB/sample_task/PGB.txt
    /RCtrlPGB_GPB/sample_task/RCtrlB.txt
    '''
    def probiotics_data_process(
            self,
            dest_path: str = "/home/share/huadjyin/home/sunhaotong/trash/Growth-promoting_Bacteria/",
            save_path: str = "/home/share/huadjyin/home/sunhaotong/trash",
            name: str = "*",
            num_bound: Optional[Tuple] = (3000, 7000),
            len_bound: Optional[Tuple] = (100, 2500),
            filter_sample_name: List[str] = []
    ):
        """每个类型各取nums"""
        assert os.path.exists(dest_path), f"{dest_path} is not exits"
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        species_name_dir = {
            "Growth-promoting_Bacteria":"GPB",
            "Low-scoring_Bacteria":"RCtrlB",
            "High-low-scoring_Bacteria":"test_manual",
            "Wet-test_Bacteria":"test_wet",
            "Pathogenic_Bacteria":"PGB",
            'Bacillus': 'Bacillus',
            'Non_Bacillus': 'Non_Bacillus',
        }
        # 某物种的所有样本
        file_dirs = glob.glob(os.path.join(dest_path, name))
        logger.info(f"find {len(file_dirs)} samples")
        # 遍历所有样本
        filtered_pkl_paths = []  # 创建一个新列表来存储一个物种过滤后的pkl文件路径
        for file_dir in tqdm.tqdm(file_dirs):
            # 按照细菌维度过滤样本
            if os.path.basename(file_dir) in filter_sample_name:
                continue
            pkl_path = glob.glob(os.path.join(f"{file_dir}", "*.pkl"))
            # filter sample by orf nums
            if num_bound is not None:
                if len(pkl_path) < num_bound[0] or len(pkl_path) > num_bound[1]:
                    continue
            if len_bound is not None:
                pool = Pool(10)  # use multi-process to speed up
                valid_pkl = []
                for pkl in pkl_path:
                    valid_pkl.append(pool.apply_async(self.filter_pkl, args=(pkl, len_bound[0], len_bound[1],)))
                pool.close()
                pool.join()
                pkl_path = [pkl.get() for pkl in valid_pkl if pkl.get() is not None]
            else:
                pkl_path = pkl_path
            filtered_pkl_paths.extend(pkl_path)
        species_name = os.path.basename(dest_path.rstrip("/"))
        save_name = species_name_dir[species_name] + ".txt"
        logger.info(f"find {len(filtered_pkl_paths)} samples from {species_name} after filtering")
        self.write_txt(filtered_pkl_paths, save_dir=save_path, name=save_name)

    '''
    output:
    /RCtrlPGB_GPB/sample_task/GPB.txt
    /RCtrlPGB_GPB/sample_task/PGB.txt
    /RCtrlPGB_GPB/sample_task/RCtrlB.txt
    '''
    def get_species_txt_from_sample_txt(
            self,
            sample_txt_path: str,
            species_txt_path: str,
    ):
        species_txt_list = []
        sample_txt_list = glob.glob(os.path.join(sample_txt_path, '*.txt'))
        for sample_txt in sample_txt_list:
            txt_path = os.path.join(sample_txt_path, sample_txt)
            with open(txt_path, "r") as f:
                for line in f:
                    species_txt_list.append(line.strip())
        name = os.path.basename(sample_txt_path.rstrip("/")) + ".txt"
        self.write_txt(data=species_txt_list, save_dir=species_txt_path, name=name)
    '''
    output:
    /RCtrlPGB_GPB/sample_task/GPB/*.txt
    /RCtrlPGB_GPB/sample_task/PGB/*.txt
    '''
    def probiotics_single_sample_data(
            self,
            dest_path: str,
            save_path: str,
            sample_num: int = -1,
            name: str = "*",
            num_bound: Optional[Tuple] = (3000, 7000),
            len_bound: Optional[Tuple] = (100, 2500),
            seed: int = 42,
            short_genomes: List[str] = []

    ):

        random.seed(seed)

        assert os.path.exists(dest_path), f"{dest_path} is not exits"
        dest_path = os.path.normpath(dest_path)
        os.makedirs(save_path, exist_ok=True)

        file_dirs = glob.glob(os.path.join(dest_path, name))
        if sample_num != -1:
            random.shuffle(file_dirs)
            count = 0
        # logger.info(f"find {len(file_dirs)} samples")
        for file_dir in tqdm.tqdm(file_dirs):
            # filter sample by sample length
            if os.path.basename(file_dir) in short_genomes or os.path.basename(file_dir) == "logs":
                # print(f"skip {file_dir}")
                continue
            pkl_path = glob.glob(os.path.join(f"{file_dir}", "*.pkl"))
            # filter sample by orf nums
            if num_bound is not None:
                if len(pkl_path) < num_bound[0] or len(pkl_path) > num_bound[1]:
                    continue
            if len_bound is not None:
                pool = Pool(10)  # use multi-process to speed up
                valid_pkl = []
                for pkl in pkl_path:
                    valid_pkl.append(pool.apply_async(self.filter_pkl, args=(pkl, len_bound[0], len_bound[1],)))
                pool.close()
                pool.join()
                pkl_path = [pkl.get() for pkl in valid_pkl if pkl.get() is not None]
            else:
                pkl_path = pkl_path
            if sample_num != -1:
                count = count + 1
                if count == sample_num:
                    break
            save_name = os.path.basename(file_dir).rstrip("/") + ".txt"
            self.write_txt(pkl_path, save_dir=save_path, name=save_name)

    @staticmethod
    def write_txt(data: list, save_dir: str, name: str):
        if data:
            assert os.path.exists(save_dir), f"{save_dir} is not exit."
            path = os.path.join(save_dir, name)
            with open(path, "w", encoding="utf8") as f:
                f.write("\n".join(data))
        else:
            logger.info(f"{name} is empty. data nums: {len(data)}")

    '''
    output:
    /RCtrlPGB_GPB/sample_task/train
    /RCtrlPGB_GPB/sample_task/val
    /RCtrlPGB_GPB/sample_task/test
    /RCtrlPGB_GPB/seq_task/train.txt
    /RCtrlPGB_GPB/seq_task/val.txt
    /RCtrlPGB_GPB/seq_task/test.txt
    '''
    def split_data_with_seq_sample(
            self,
            dir_path: str,
            positive_groups: list,
            negative_groups: list,
            save_pkl_dir: str,
            save_sample_dir: str,
            max_val_nums: int = 25000,
            seed: int = 42,
            choice_num = None,
    ):
        """split data to train: val: test=8: 1: 1 by sample level. if group is *.txt, then read the file, else read all txt in the dir"""
        os.path.exists(dir_path), f"{dir_path} is not exit."
        os.path.exists(save_pkl_dir), f"{save_pkl_dir} is not exit."
        os.path.exists(save_sample_dir), f"{save_sample_dir} is not exit."
        random.seed(seed)
        # positive groups
        df_positives, df_negatives = pd.DataFrame(), pd.DataFrame()
        for name in positive_groups:
            assert os.path.exists(os.path.join(dir_path, name)), f"{name} is not exit."
            if name.split(".")[-1] == "txt":
                group_paths = os.path.join(dir_path, name)
            else:
                group_paths = glob.glob(os.path.join(dir_path, name, "*.txt"))
            logger.info(f"find {len(group_paths)} positive groups")
            for group_path in group_paths:
                with open(group_path) as f:
                    positive_pkls = f.read().split("\n")
                    positive_names = [os.path.basename(os.path.dirname(i)) for i in positive_pkls]
                    df_positive = pd.DataFrame({"pkl": positive_pkls, "name": positive_names})
                    df_positives = pd.concat([df_positives, df_positive], axis=0)
        # negative groups
        for name in negative_groups:
            assert os.path.exists(os.path.join(dir_path, name)), f"{name} is not exit."
            if name.split(".")[-1] == "txt":
                group_paths = os.path.join(dir_path, name)
            else:
                group_paths = glob.glob(os.path.join(dir_path, name, "*.txt"))
            logger.info(f"find {len(group_paths)} negative groups")
            for group_path in group_paths:
                with open(group_path) as f:
                    negative_pkls = f.read().split("\n")
                    negatives_names = [os.path.basename(os.path.dirname(i)) for i in negative_pkls]
                    df_negative = pd.DataFrame({"pkl": negative_pkls, "name": negatives_names})
                    df_negatives = pd.concat([df_negatives, df_negative], axis=0)
        # 正负样本各抽样sample_pos_uni_name个
        # df_positives = df_positives.sample(frac=1, random_state=self.seed, replace=False)
        # df_negatives = df_negatives.sample(frac=1, random_state=self.seed, replace=False)
        pos_uni_name = list(set(df_positives["name"]))
        neg_uni_name = list(set(df_negatives["name"]))
        if choice_num is not None:
            sample_pos_uni_name = random.sample(pos_uni_name, choice_num)
            df_positives = df_positives[df_positives["name"].isin(sample_pos_uni_name)]
            sample_neg_uni_name = random.sample(neg_uni_name, choice_num)
            df_negatives = df_negatives[df_negatives["name"].isin(sample_neg_uni_name)]

        df_merge = pd.concat([df_positives, df_negatives], axis=0)

        pst_uni_name = list(set(df_positives["name"]))
        ngt_uni_name = list(set(df_negatives["name"]))

        # NOTE: Try to balance the positive and negative samples
        # sample_nums = int((len(pst_uni_name) + len(ngt_uni_name)) / 2)
        # test_nums = int(sample_nums * 0.1)
        # test_names = pst_uni_name[:test_nums] + ngt_uni_name[:test_nums]
        # val_nums = min(int(sample_nums * 0.1), max_val_nums)  # 避免验证集过大，导致训练时间过长, 与康博讨论过
        # val_names = pst_uni_name[test_nums:test_nums + val_nums] + ngt_uni_name[test_nums:test_nums + val_nums]
        # train_names = pst_uni_name[test_nums + val_nums:] + ngt_uni_name[test_nums + val_nums:]

        # positive sample
        pos_test_nums = int(len(pst_uni_name) * 0.1)
        pos_test_names = pst_uni_name[:pos_test_nums]
        pos_val_nums = min(int(len(pst_uni_name) * 0.1), max_val_nums)
        pos_val_names = pst_uni_name[pos_test_nums:pos_test_nums + pos_val_nums]
        pos_train_names = pst_uni_name[pos_test_nums + pos_val_nums:]
        # negitive sample
        neg_test_nums = int(len(ngt_uni_name) * 0.1)
        neg_test_names = ngt_uni_name[:neg_test_nums]
        neg_val_nums = min(int(len(ngt_uni_name) * 0.1), max_val_nums)
        neg_val_names = ngt_uni_name[neg_test_nums:neg_test_nums + neg_val_nums]
        neg_train_names = ngt_uni_name[neg_test_nums + neg_val_nums:]

        # test_nums = pos_test_nums + neg_test_nums
        test_names = pos_test_names + neg_test_names
        # val_nums = pos_val_nums + neg_val_nums
        val_names = pos_val_names + neg_val_names
        # train_nums = pos_train_names + neg_train_names
        train_names = pos_train_names + neg_train_names

        # save pkl level
        test = df_merge.loc[df_merge["name"].isin(test_names)].to_dict(orient="list")["pkl"]
        self.write_txt(test, save_pkl_dir, name="test.txt")
        logger.info(f"test_sample: {len(test_names)}; test_nums: {len(test)}, save in {save_pkl_dir}")

        val = df_merge.loc[df_merge["name"].isin(val_names)].to_dict(orient="list")["pkl"]
        self.write_txt(val, save_pkl_dir, name="val.txt")
        logger.info(f"val_sample: {len(val_names)}; val_nums: {len(val)}, save in {save_pkl_dir}")

        train = df_merge.loc[df_merge["name"].isin(train_names)].to_dict(orient="list")["pkl"]
        self.write_txt(train, save_pkl_dir, name="train.txt")
        logger.info(f"train_sample: {len(train_names)}; train_nums: {len(train)}, save in {save_pkl_dir}")

        # save sample level
        test_save_sample_dir = os.path.join(save_sample_dir, "test")
        self.save_sample_data(test_names, df_merge, test_save_sample_dir)

        val_save_sample_dir = os.path.join(save_sample_dir, "val")
        self.save_sample_data(val_names, df_merge, val_save_sample_dir)

        train_save_sample_dir = os.path.join(save_sample_dir, "train")
        self.save_sample_data(train_names, df_merge, train_save_sample_dir)

    def save_sample_data(self, names: List[str], df_merge: pd.DataFrame, save_sample_dir: str):
        os.makedirs(save_sample_dir, exist_ok=True)
        logger.info(f"save {len(names)} sample data to {save_sample_dir}")
        for name in tqdm.tqdm(names):
            merge_bs = df_merge.loc[df_merge["name"] == name]["pkl"].tolist()
            self.write_txt(merge_bs, save_sample_dir, name=f"{name}.txt")

    @staticmethod
    def filter_pkl(pkl: str, len_lower_bound: int = 100, len_upper_bound: int = 2500):
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        flag = len(data["Seq"]) < len_lower_bound or len(data["Seq"]) > len_upper_bound
        if flag:
            return None
        else:
            return pkl


    def get_predict_output_txt(self, predict_output_path: str, save_path: str):
        for datasets in os.listdir(predict_output_path):
            datasets_path = os.path.join(predict_output_path, datasets)
            # 如果是文件夹
            if os.path.isdir(datasets_path):
                sample_pkl_path = glob.glob(os.path.join(datasets_path, "pickles", "*.pkl"))
                if datasets.startswith('test_manual_') or datasets.startswith('test_wet_') or datasets.startswith('seq_task') or datasets.startswith('unbalanced_test') or datasets.startswith('High-low-scoring_Bacteria') or datasets.startswith('Wet-test_Bacteria'):
                    self.write_txt(
                        sample_pkl_path,
                        save_dir=save_path,
                        name=datasets.split('_')[0] + "_" + datasets.split('_')[1] + ".txt"
                    )
                else:
                    self.write_txt(
                        sample_pkl_path,
                        save_dir=save_path,
                        name=f"{datasets.split('_')[0]}.txt"
                    )
                logger.info(f"{datasets} nums: {len(sample_pkl_path)}")
    def probiotics_check_sample_len(
            self,
            data_path: str,
            min_len: int = 100,
            max_len: int = 2500
    ):
        short_genomes = []
        for file_name in tqdm.tqdm(os.listdir(data_path)):
            if file_name.endswith('.fna') or file_name.endswith('.fasta'):
                file_path = os.path.join(data_path, file_name)
                total_length = 0
                for record in SeqIO.parse(file_path, "fasta"):
                    total_length += len(record.seq)
                if total_length < min_len or total_length > max_len:
                    short_genomes.append(file_name)
        return short_genomes


    def probiotics_get_pickles_txt(
            self,
            dir: str,
    ):
        with open(dir+'/pickles.txt', "w") as f:
            for path in glob.glob(f"{dir}/pickles/*.pkl"):
                f.write(path + "\n")
        # logger.info(f"All results saved in {dir+'/pickles.txt'}")
    
    def probiotics_get_all_txt(
            self,
            pkl_path: str,
    ):
        with open(pkl_path + '/all.txt', "w") as f:
            for path in glob.glob(f"{pkl_path}/*.pkl"):
                f.write(path + "\n")
        # logger.info(f"All results saved in {pkl_path}/all.txt")


if __name__ == '__main__':
    pbt_dp = ProbioticsDataProcess()
