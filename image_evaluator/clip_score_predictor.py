# Taken from https://github.com/Taited/clip-score/blob/master/src/clip_score/clip_score.py

import os
import os.path as osp

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer


class DummyDataset(Dataset):

    FLAGS = ["img", "txt"]

    def __init__(
        self,
        real_path,
        fake_path,
        real_flag: str = "img",
        fake_flag: str = "txt",
        transform=None,
        tokenizer=None,
    ) -> None:
        super().__init__()
        if real_flag not in self.FLAGS or fake_flag not in self.FLAGS:
            raise TypeError(
                "CLIP Score only support modality of {}. "
                "However, get {} and {}".format(
                    self.FLAGS, real_flag, fake_flag
                )
            )
        from image_evaluator._stem_pairing import (
            allowed_exts_for_flag,
            collect_flat_dir,
            pair_dirs,
        )

        self.real_flag = real_flag
        self.fake_flag = fake_flag
        self.transform = transform
        self.tokenizer = tokenizer
        real_is_dir = isinstance(real_path, str) and osp.isdir(real_path)
        fake_is_dir = isinstance(fake_path, str) and osp.isdir(fake_path)
        if real_is_dir and fake_is_dir:
            self._mode = "paired"
            self._pairs = pair_dirs(
                real_path,
                fake_path,
                allowed_exts_for_flag(real_flag),
                allowed_exts_for_flag(fake_flag),
            )
            self.real_folder = [r for r, _ in self._pairs]
            self.fake_folder = [f for _, f in self._pairs]
        elif real_is_dir:
            mapping = collect_flat_dir(
                real_path, allowed_exts_for_flag(real_flag)
            )
            self._real_list = [mapping[s] for s in sorted(mapping)]
            if isinstance(fake_path, list):
                if len(fake_path) != len(self._real_list):
                    raise ValueError(
                        f"Number of prompts ({len(fake_path)}) does not "
                        f"match number of images ({len(self._real_list)}) "
                        f"in '{real_path}'"
                    )
                self._mode = "paired_list"
                self._fake_list = fake_path
                self.real_folder = list(self._real_list)
                self.fake_folder = list(fake_path)
            elif (
                isinstance(fake_path, (str, os.PathLike))
                and osp.isfile(fake_path)
            ):
                if not os.access(fake_path, os.R_OK):
                    raise OSError(f"Unreadable file: {fake_path}")
                try:
                    with open(fake_path, "r", encoding="utf-8") as f:
                        lines = [
                            line.strip() for line in f if line.strip()
                        ]
                except UnicodeDecodeError:
                    with open(fake_path, "r", encoding="latin-1") as f:
                        lines = [
                            line.strip() for line in f if line.strip()
                        ]
                if len(lines) == len(self._real_list):
                    self._mode = "paired_list"
                    self._fake_list = lines
                    self.real_folder = list(self._real_list)
                    self.fake_folder = list(lines)
                else:
                    self._mode = "broadcast_fake"
                    self._fake_scalar = (
                        " ".join(lines) if lines else ""
                    )
                    self.real_folder = list(self._real_list)
                    self.fake_folder = self._fake_scalar
            else:
                self._mode = "broadcast_fake"
                self._fake_scalar = fake_path
                self.real_folder = list(self._real_list)
                self.fake_folder = fake_path
        elif fake_is_dir:
            self._mode = "broadcast_real"
            mapping = collect_flat_dir(
                fake_path, allowed_exts_for_flag(fake_flag)
            )
            self._fake_list = [mapping[s] for s in sorted(mapping)]
            self._real_scalar = real_path
            if isinstance(real_path, str) and osp.isfile(real_path):
                if not os.access(real_path, os.R_OK):
                    raise OSError(f"Unreadable file: {real_path}")
            self.real_folder = real_path
            self.fake_folder = list(self._fake_list)
        else:
            self._mode = "scalar"
            self.real_folder = real_path
            self.fake_folder = fake_path
        # assert self._check()

    def __len__(self):
        if self._mode in ("paired", "paired_list"):
            return len(self.real_folder)
        if self._mode == "broadcast_fake":
            return len(self._real_list)
        if self._mode == "broadcast_real":
            return len(self._fake_list)
        return 1

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        if self._mode == "paired":
            real_path, fake_path = self._pairs[index]
        elif self._mode == "paired_list":
            real_path = self._real_list[index]
            fake_path = self._fake_list[index]
        elif self._mode == "broadcast_fake":
            real_path = self._real_list[index]
            fake_path = self._fake_scalar
        elif self._mode == "broadcast_real":
            real_path = self._real_scalar
            fake_path = self._fake_list[index]
        else:
            real_path = self.real_folder
            fake_path = self.fake_folder

        real_data = self._load_modality(real_path, self.real_flag)
        fake_data = self._load_modality(fake_path, self.fake_flag)

        sample = dict(real=real_data, fake=fake_data)
        return sample

    def _load_modality(self, path, modality):
        if modality == "img":
            data = self._load_img(path)
        elif modality == "txt":
            data = self._load_txt(path)
        else:
            raise TypeError("Got unexpected modality: {}".format(modality))
        return data

    def _load_img(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(text=None, images=img)
            img["pixel_values"] = img["pixel_values"][0]
        return img

    def _load_txt(self, path):
        if isinstance(path, list):
            data = path[0] if path else ""
        elif isinstance(path, (str, os.PathLike)) and osp.exists(path):
            try:
                # 首先尝试使用UTF-8编码
                with open(path, "r", encoding="utf-8") as fp:
                    data = fp.read()
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试使用latin-1编码（可以处理任何字节序列）
                with open(path, "r", encoding="latin-1") as fp:
                    data = fp.read()
        else:
            data = str(path)
        if self.transform is not None:
            # Truncate long text to fit CLIP max token length.
            data = self.tokenizer(
                data,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            for key in data:
                data[key] = data[key].squeeze()
        return data

    def _check(self):
        for idx in range(len(self)):
            real_name = self.real_folder[idx].split(".")
            fake_name = self.fake_folder[idx].split(".")
            if fake_name != real_name:
                return False
        return True

    def _combine_without_prefix(self, folder_path, prefix="."):
        if not osp.isdir(folder_path):
            return folder_path
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(osp.join(folder_path, name))
        folder.sort()
        return folder


class ClipScorePredictor:
    def __init__(self, clip_model="openai/clip-vit-base-patch32", device=None):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        print("Loading CLIP model: {}".format(clip_model))
        self.model = AutoModel.from_pretrained(clip_model).to(self.device)
        self.processor = AutoProcessor.from_pretrained(clip_model)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model)

    def evaluate_clip_score(
        self,
        real_path,
        fake_path,
        real_flag="img",
        fake_flag="txt",
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
        # Check if it's a single file or in-memory image
        is_mem = not isinstance(real_path, (str, os.PathLike))
        is_single_real = is_mem or (
            isinstance(real_path, (str, os.PathLike))
            and osp.isfile(real_path)
        )
        if is_single_real:
            return self._evaluate_single_file(
                real_path, fake_path, real_flag, fake_flag
            )
        else:
            return self.evaluate_folder_clip_score(
                real_path,
                fake_path,
                real_flag,
                fake_flag,
                batch_size,
                num_workers,
            )

    def evaluate_folder_clip_score(
        self,
        real_path,
        fake_path,
        real_flag="img",
        fake_flag="txt",
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

        print("Calculating CLIP Score:")
        score_acc = 0.0
        sample_num = 0.0
        for batch_data in tqdm(dataloader):
            real = batch_data["real"]
            real_features = self._forward_modality(real, real_flag)
            fake = batch_data["fake"]
            fake_features = self._forward_modality(fake, fake_flag)

            # normalize features
            real_features = real_features / real_features.norm(
                dim=1, keepdim=True
            ).to(torch.float32)
            fake_features = fake_features / fake_features.norm(
                dim=1, keepdim=True
            ).to(torch.float32)

            # calculate scores
            score = (fake_features * real_features).sum()
            score_acc += score
            sample_num += real_features.shape[0]

        clip_score = score_acc / sample_num
        return clip_score.cpu().item()

    def _evaluate_single_file(
        self,
        image_path,
        text_path_or_string,
        image_flag="img",
        text_flag="txt",
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
        if image_flag == "img" and text_flag == "txt":
            img_path, txt_path = image_path, text_path_or_string
            img_flag, txt_flag = image_flag, text_flag
        elif image_flag == "txt" and text_flag == "img":
            img_path, txt_path = text_path_or_string, image_path
            img_flag, txt_flag = text_flag, image_flag
        else:
            raise ValueError("Must specify one 'img' and one 'txt' modality")

        if not isinstance(img_path, (str, os.PathLike)):
            from image_evaluator._input_adapters import to_pil_image

            pil_img = to_pil_image(img_path)
            img_data = self.processor(images=pil_img, return_tensors="pt")
            if isinstance(txt_path, list):
                txt_content = txt_path[0] if txt_path else ""
            elif (
                isinstance(txt_path, (str, os.PathLike))
                and not os.path.exists(txt_path)
            ):
                txt_content = str(txt_path)
            elif isinstance(txt_path, (str, os.PathLike)):
                with open(txt_path, "r", encoding="utf-8") as f:
                    txt_content = f.read().strip()
            else:
                txt_content = str(txt_path)
            txt_data = self.tokenizer(
                txt_content,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            for key in txt_data:
                txt_data[key] = txt_data[key].squeeze()
        else:
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
            img_data = sample["real"]
            txt_data = sample["fake"]

        # Compute features
        img_features = self._forward_modality(img_data, "img")
        txt_features = self._forward_modality(txt_data, "txt")

        # Normalize features
        img_features = img_features / img_features.norm(
            dim=1, keepdim=True
        ).to(torch.float32)
        txt_features = txt_features / txt_features.norm(
            dim=1, keepdim=True
        ).to(torch.float32)

        # Compute score
        score = (img_features * txt_features).sum()

        return score.cpu().item()

    def _forward_modality(self, data, flag):
        device = self.device
        for key in data:
            data[key] = data[key].to(device)
        if flag == "img":
            if (
                "pixel_values" in data
                and isinstance(data["pixel_values"], torch.Tensor)
                and data["pixel_values"].ndim == 3
            ):
                data["pixel_values"] = data["pixel_values"].unsqueeze(0)
            features = self.model.get_image_features(**data)
        elif flag == "txt":
            features = self.model.get_text_features(**data)
        else:
            raise TypeError(f"Got unexpected modality: {flag}")
        if isinstance(features, torch.Tensor):
            return features
        pooler_output = getattr(features, "pooler_output", None)
        if isinstance(pooler_output, torch.Tensor):
            return pooler_output
        return features
