from typing import Dict, List


def collate_to_dict_of_list(batch: List[Dict]) -> Dict[str, List]:
    collated_batch = {k: [x[k] for x in batch] for k in batch[0].keys()}
    return collated_batch