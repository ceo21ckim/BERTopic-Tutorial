import os, pandas as pd, numpy as np 
import datasets 
from typing import Tuple 
import argparse 

parser = argparse.ArgumentParser(description='load/save datasets')
parser.add_argument('--save', '-s', action='store_true')

args = parser.parse_args()

class DataLoader:
    def __init__(self, source="daekeun-ml/naver-news-summarization-ko", save=True):
        self.source = source; self.verbose = getattr(self, 'verbose', True); self.save = save
        
        if self.verbose: print(f"Loading Datasets source: {self.source} 🚀")
        try: self.datasets = self._data_loader()
        except Exception as e: print(f"Error > Invalid value [{self.source}]\n{e}")

        self.train_set, self.valid_set, self.test_set = self._train_valid_test_split()

        if self.save:
            if self.verbose: print(f"\nSaving Datasets! ✔")
            try:self._save_data()
            except Exception as e: print(f"Error self._save_data\n{e}")         
        
    def _data_loader(self) -> datasets.dataset_dict.DatasetDict:
        from datasets import load_dataset 
        return load_dataset(self.source)
    
    def _train_valid_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train, valid, test = self.datasets['train'].to_pandas(), self.datasets['validation'].to_pandas(), self.datasets['test'].to_pandas()
        self.features = np.array([*self.datasets['train'].features], dtype='str')
        
        if self.verbose:
            print(f'train dataset shape:        {train.shape}')
            print(f'validation dataset shape:   {valid.shape}')
            print(f'test dataset shape:         {test.shape}\n')
            print(f'features names are {self.features}\n\n')
            print(f'data samples:\n')
            for col in self.features:
                samples = train[col].sample(1)
                print(f'{col}: {samples.item()}')
                                    
        return (train, valid, test)

    def _save_data(self):
        self.data_path = os.path.join(os.path.dirname(__file__), 'data')
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print(f'Make data folder [PATH: {self.data_path}]')
        
        for src in ['train', 'valid', 'test']:
            path = os.path.join(self.data_path, f'{src}_set.csv')
            getattr(self, f'{src}_set').to_csv(path, encoding='utf-8-sig', index=False)
            print(f"Success Saving datasets: [PATH: {path}]")
                
if __name__ == '__main__':
    data_loader = DataLoader(save=args.save)
    train, valid, test = data_loader.train_set, data_loader.valid_set, data_loader.test_set
