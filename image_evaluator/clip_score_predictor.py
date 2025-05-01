# taken from https://github.com/Taited/clip-score/blob/master/src/clip_score/clip_score.py
import os
import torch
import os.path as osp
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor, AutoTokenizer

class DummyDataset(Dataset):

    FLAGS = ['img', 'txt']

    def __init__(self,
                 real_path,
                 fake_path,
                 real_flag: str = 'img',
                 fake_flag: str = 'txt',
                 transform=None,
                 tokenizer=None) -> None:
        super().__init__()
        if real_flag not in self.FLAGS or fake_flag not in self.FLAGS:
            raise TypeError('CLIP Score only support modality of {}. '
                            'However, get {} and {}'.format(
                                self.FLAGS, real_flag, fake_flag))
        self.real_folder = self._combine_without_prefix(real_path)
        self.real_flag = real_flag
        self.fake_folder = self._combine_without_prefix(fake_path)
        self.fake_flag = fake_flag
        self.transform = transform
        self.tokenizer = tokenizer
        # assert self._check()

    def __len__(self):
        if isinstance(self.real_folder, list):
            real_folder_lenghth = len(self.real_folder)
        else:
            real_folder_lenghth = 1
        if isinstance(self.fake_folder, list):
            fake_folder_lenghth = len(self.fake_folder)
        else:
            fake_folder_lenghth = 1
        return max(real_folder_lenghth, fake_folder_lenghth)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        if isinstance(self.real_folder, list):
            real_path = self.real_folder[index]
        else:
            real_path = self.real_folder
        if isinstance(self.fake_folder, list):
            fake_path = self.fake_folder[index]
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
            with open(path, 'r') as fp:
                data = fp.read()
                fp.close()
        else:
            data = path
        if self.transform is not None:
            data = self.tokenizer(data, padding=True, return_tensors='pt')
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

    def evaluate_clip_score(self, real_path, fake_path, real_flag='img', fake_flag='txt', batch_size=50, num_workers=None):
        if num_workers is None:
            try:
                num_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                num_cpus = os.cpu_count()
            num_workers = min(num_cpus, 8) if num_cpus is not None else 0

        dataset = DummyDataset(real_path, fake_path, real_flag, fake_flag, transform=self.processor, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)

        print('Calculating CLIP Score:')
        score_acc = 0.
        sample_num = 0.
        for batch_data in tqdm(dataloader):
            real = batch_data['real']
            real_features = self._forward_modality(real, real_flag)
            fake = batch_data['fake']
            fake_features = self._forward_modality(fake, fake_flag)

            # normalize features
            real_features = real_features / real_features.norm(dim=1, keepdim=True).to(torch.float32)
            fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(torch.float32)

            # calculate scores
            score = (fake_features * real_features).sum()
            score_acc += score
            sample_num += real_features.shape[0]

        clip_score = score_acc / sample_num
        return clip_score.cpu().item()

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
