from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from torch.utils.data import Dataset

from .builder import DATASETS
from .collate_fn import collate_to_dict_of_list


@DATASETS.register_module()
class IBAPromptsDataset(Dataset):

    def __init__(
            self, csv_path: str, case_range: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()

        self.df = pd.read_csv(csv_path)
        self.df.dropna(axis=0, inplace=True)
        if case_range is not None:
            self.df = self.df[(case_range[0] <= self.df['case_id'])
                              & (self.df['case_id'] < case_range[1])]

        self.df['case_id'] = self.df['case_id'].astype(int)
        self.df['seed'] = self.df['seed'].astype(int)
        self.df['class_label'] = self.df['class_label'].astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.df.loc[index]
        case_id = sample['case_id']
        prompt = sample['prompt']
        seed = int(sample['seed'])
        class_label = sample['class_label']

        result = {
            'case_id': case_id,
            'prompt': prompt,
            'seed': seed,
            'class_label': class_label
        }
        return result

    @staticmethod
    def get_collate_fn() -> Callable[[List], Dict[str, List]]:
        return collate_to_dict_of_list