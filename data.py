from datasets import load_dataset
from pathlib import Path
import logging
from collections import Counter
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from PIL import Image as PILImage

import cv2
import numpy as np
import requests
from tqdm import tqdm



logging.basicConfig(
    level=print,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)  # Get a logger for this file


def get_artist_names(artist_file):
    artists = []
    with open(artist_file, 'r') as f:
        for line in f:
            artists.append(line.strip().lower())
    return artists


def build_dataset():
    print("Loading WikiArt dataset...")
    ds = load_dataset("matrixglitch/wikiart-215k", split='train') # .select(range(TEST_SIZE))  # small subset for testing
    
    artist_names = get_artist_names('artists.txt')
    
    def extract_artist(example):
        example['artist'] = Path(example['file_link']).parts[-2].replace('-', ' ')
        return example
    
    ds = ds.map(extract_artist)
    ds = ds.filter(lambda x: x['artist'] in artist_names)
    
    print(f"\nTotal images: {len(ds)}")
    found_artists = list(set(ds['artist']))
    print(f"Found artists: {len(found_artists)}")
    
    return ds, found_artists

class ArtworkDataset(Dataset):
    def __init__(self, hf_dataset, artist_names, transform=None, max_samples_per_artist=None):
        self.transform = transform
        self.artist_to_idx = {name: idx for idx, name in enumerate(artist_names)}
        self.idx_to_artist = {idx: name for name, idx in self.artist_to_idx.items()}
        
        # Build examples list
        self.examples = []
        artist_counts = defaultdict(int)
        failed_loads = 0
        
        print("Building dataset...")
        for item in tqdm(hf_dataset):
            artist = item['artist']
            if max_samples_per_artist is None or artist_counts[artist] < max_samples_per_artist:
                try:
                    # Download and convert to numpy array
                    response = requests.get(item['file_link'], stream=True)
                    arr = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        # Resize if too large
                        if max(img.shape) > 2048:
                            scale = 2048 / max(img.shape)
                            new_size = tuple(int(dim * scale) for dim in reversed(img.shape))
                            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
                        
                        self.examples.append({
                            'image': img,
                            'artist': artist,
                            'label': self.artist_to_idx[artist]
                        })
                        artist_counts[artist] += 1
                except Exception as e:
                    failed_loads += 1
                    logging.warning(f"Failed to load image from {item['file_link']}: {str(e)}")
                    continue
        
        # Calculate weights for balanced sampling
        label_counts = Counter(ex['label'] for ex in self.examples)
        total_samples = len(self.examples)
        self.class_weights = []
        for i in range(len(artist_names)):
            if i in label_counts and label_counts[i] > 0:
                weight = total_samples / (len(label_counts) * label_counts[i])
            else:
                weight = 0.0
            self.class_weights.append(weight)
            
        print(f"Dataset created with {len(self.examples)} images")
        print(f"Failed to load {failed_loads} images")
        print("Images per artist:")
        for artist, count in artist_counts.items():
            print(f"- {artist}: {count}")
    
    def get_sample_weights(self):
        return [self.class_weights[ex['label']] for ex in self.examples]
    
    def get_artist_name(self, idx):
        return self.idx_to_artist[idx]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        image = item['image']
        
        # Convert numpy array to PIL Image for transforms
        image = PILImage.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, item['label']

def create_dataset(args):
    # Load dataset and get artist distribution
    dataset, found_artists = build_dataset()
    
    # Create train/val split
    splits = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_transform, val_transform = create_transforms()
    
    # Create datasets
    train_dataset = ArtworkDataset(
        splits['train'], 
        found_artists,
        transform=train_transform,
        max_samples_per_artist=args.max_samples_per_artist
    )
    val_dataset = ArtworkDataset(
        splits['test'], 
        found_artists,
        transform=val_transform,
        max_samples_per_artist=args.max_samples_per_artist
    )
    
    return train_dataset, val_dataset, found_artists

def create_transforms():
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
        transforms.RandomRotation(5),
        transforms.RandomAdjustSharpness(0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    return train_transform, val_transform



def create_weighted_sampler(dataset):
    # Create samplers for balanced training
   return WeightedRandomSampler(
        weights=dataset.get_sample_weights(),
        num_samples=len(dataset),
        replacement=True
    )


def create_dataloaders(args, train_dataset, val_dataset):
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=create_weighted_sampler(train_dataset),
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    return train_loader, val_loader
