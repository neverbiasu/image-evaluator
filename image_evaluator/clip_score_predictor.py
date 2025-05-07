# Taken from https://github.com/Taited/clip-score/blob/master/src/clip_score/clip_score.py

import os
import torch
import os.path as osp
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor, AutoTokenizer


class DummyDataset(Dataset):

    FLAGS = ['img', 'txt']

    def __init__(
        self,
        real_path,
        fake_path,
        real_flag: str = 'img',
        fake_flag: str = 'txt',
        transform=None,
        tokenizer=None,
    ) -> None:
        super().__init__()
        if real_flag not in self.FLAGS or fake_flag not in self.FLAGS:
            raise TypeError(
                'CLIP Score only support modality of {}. '
                'However, get {} and {}'.format(self.FLAGS, real_flag, fake_flag)
            )
        self.real_folder = self._combine_without_prefix(real_path)
        self.real_flag = real_flag
        self.fake_folder = self._combine_without_prefix(fake_path)
        self.fake_flag = fake_flag
        self.transform = transform
        self.tokenizer = tokenizer
        # assert self._check()

    def __len__(self):
        real_folder_length = len(self.real_folder) if isinstance(self.real_folder, list) else 1
        fake_folder_length = len(self.fake_folder) if isinstance(self.fake_folder, list) else 1
        return max(real_folder_length, fake_folder_length)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        
        # 处理real_folder的索引边界
        if isinstance(self.real_folder, list):
            # 使用取模操作确保索引在有效范围内
            real_index = index % len(self.real_folder)
            real_path = self.real_folder[real_index]
        else:
            real_path = self.real_folder
            
        # 处理fake_folder的索引边界
        if isinstance(self.fake_folder, list):
            # 使用取模操作确保索引在有效范围内
            fake_index = index % len(self.fake_folder)
            fake_path = self.fake_folder[fake_index]
        else:
            fake_path = self.fake_folder
            
        real_data = self._load_modality(real_path, self.real_flag)
        fake_data = self._load_modality(fake_path, self.fake_flag)

        sample = dict(real=real_data, fake=fake_data)
        return sample

    def _load_modality(self, path, modality):
        if modality == 'img':
            data = self._load_img(path)
        elif modality == 'txt':
            data = self._load_txt(path)
        else:
            raise TypeError('Got unexpected modality: {}'.format(modality))
        return data

    def _load_img(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(text=None, images=img)
            img['pixel_values'] = img['pixel_values'][0]
        return img

    def _load_txt(self, path):
        if osp.exists(path):
            try:
                # 首先尝试使用UTF-8编码
                with open(path, 'r', encoding='utf-8') as fp:
                    data = fp.read()
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试使用latin-1编码（可以处理任何字节序列）
                with open(path, 'r', encoding='latin-1') as fp:
                    data = fp.read()
        else:
            data = path
        if self.transform is not None:
            # 添加truncation=True参数以截断过长文本，设置max_length=77匹配CLIP模型限制
            data = self.tokenizer(data, padding=True, truncation=True, max_length=77, return_tensors='pt')
            for key in data:
                data[key] = data[key].squeeze()
        return data

    def _check(self):
        for idx in range(len(self)):
            real_name = self.real_folder[idx].split('.')
            fake_name = self.fake_folder[idx].split('.')
            if fake_name != real_name:
                return False
        return True

    def _combine_without_prefix(self, folder_path, prefix='.'):
        if not osp.exists(folder_path):
            return folder_path
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(osp.join(folder_path, name))
        folder.sort()
        return folder


class ClipScorePredictor:
    def __init__(self, clip_model='openai/clip-vit-base-patch32', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print('Loading CLIP model: {}'.format(clip_model))
        self.model = AutoModel.from_pretrained(clip_model).to(self.device)
        self.processor = AutoProcessor.from_pretrained(clip_model)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model)

    def evaluate_clip_score(
        self,
        real_path,
        fake_path,
        real_flag='img',
        fake_flag='txt',
        batch_size=50,
        num_workers=None,
    ):
        """Evaluate CLIP score between images and text

        Supports both single files and folders evaluation.

        Args:
            real_path: Path to real image or folder
            fake_path: Path to text prompt or folder, or text string directly
            real_flag: Type of real input modality, 'img' or 'txt'
            fake_flag: Type of fake input modality, 'img' or 'txt'
            batch_size: Batch size
            num_workers: Number of workers for data loader

        Returns:
            float: CLIP score
        """
        # Check if it's a single file
        if os.path.isfile(real_path) and (
            not os.path.exists(fake_path) or os.path.isfile(fake_path)
        ):
            return self._evaluate_single_file(
                real_path, fake_path, real_flag, fake_flag
            )
        else:
            return self.evaluate_folder_clip_score(
                real_path, fake_path, real_flag, fake_flag, batch_size, num_workers
            )

    def evaluate_folder_clip_score(
        self,
        real_path,
        fake_path,
        real_flag='img',
        fake_flag='txt',
        batch_size=50,
        num_workers=None,
    ):
        """Evaluate CLIP score between multiple files in folders

        Args:
            real_path: Path to folder containing real inputs
            fake_path: Path to folder containing fake inputs
            real_flag: Type of real input modality, 'img' or 'txt'
            fake_flag: Type of fake input modality, 'img' or 'txt'
            batch_size: Batch size
            num_workers: Number of workers for data loader

        Returns:
            float: CLIP score
        """
        # 强制将批大小设置为1，避免合并不同大小的张量
        batch_size = 1
        # 禁用多进程加载以避免工作进程中的错误
        num_workers = 0

        dataset = DummyDataset(
            real_path,
            fake_path,
            real_flag,
            fake_flag,
            transform=self.processor,
            tokenizer=self.tokenizer,
        )
        dataloader = DataLoader(
            dataset, batch_size, num_workers=num_workers, pin_memory=True
        )

        print('Calculating CLIP Score:')
        score_acc = 0.0
        sample_num = 0.0
        for batch_data in tqdm(dataloader):
            real = batch_data['real']
            real_features = self._forward_modality(real, real_flag)
            fake = batch_data['fake']
            fake_features = self._forward_modality(fake, fake_flag)

            # normalize features
            real_features = real_features / real_features.norm(dim=1, keepdim=True).to(
                torch.float32
            )
            fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(
                torch.float32
            )

            # calculate scores
            score = (fake_features * real_features).sum()
            score_acc += score
            sample_num += real_features.shape[0]

        clip_score = score_acc / sample_num
        return clip_score.cpu().item()

    def _evaluate_single_file(
        self, image_path, text_path_or_string, image_flag='img', text_flag='txt'
    ):
        """Evaluate CLIP score between a single image file and text

        Args:
            image_path: Path to image file
            text_path_or_string: Path to text file or text string directly
            image_flag: Type of image input modality, default is 'img'
            text_flag: Type of text input modality, default is 'txt'

        Returns:
            float: CLIP score
        """
        # Determine which is image and which is text
        if image_flag == 'img' and text_flag == 'txt':
            img_path, txt_path = image_path, text_path_or_string
            img_flag, txt_flag = image_flag, text_flag
        elif image_flag == 'txt' and text_flag == 'img':
            img_path, txt_path = text_path_or_string, image_path
            img_flag, txt_flag = text_flag, image_flag
        else:
            raise ValueError("Must specify one 'img' and one 'txt' modality")

        # Create a single sample dataset
        dataset = DummyDataset(
            img_path,
            txt_path,
            img_flag,
            txt_flag,
            transform=self.processor,
            tokenizer=self.tokenizer,
        )

        # Get data
        sample = dataset[0]
        img_data = sample['real'] if img_flag == 'real_flag' else sample['fake']
        txt_data = sample['fake'] if txt_flag == 'txt' else sample['real']

        # Compute features
        img_features = self._forward_modality(img_data, 'img')
        txt_features = self._forward_modality(txt_data, 'txt')

        # Normalize features
        img_features = img_features / img_features.norm(dim=1, keepdim=True).to(
            torch.float32
        )
        txt_features = txt_features / txt_features.norm(dim=1, keepdim=True).to(
            torch.float32
        )

        # Compute score
        score = (img_features * txt_features).sum()

        return score.cpu().item()

    def _forward_modality(self, data, flag):
        device = self.device
        for key in data:
            data[key] = data[key].to(device)
        if flag == 'img':
            features = self.model.get_image_features(**data)
        elif flag == 'txt':
            features = self.model.get_text_features(**data)
        else:
            raise TypeError(f'Got unexpected modality: {flag}')
        return features
