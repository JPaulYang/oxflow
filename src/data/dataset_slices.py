import torch
from torch.utils.data import Dataset
from typing import Optional, List


class LongitudinalCTDataset(Dataset):
    """
    Dataset for longitudinal CT scans.

    Each item corresponds to one (patient_id, slice_position) and holds the CT
    slice and tumor mask at every available timepoint.

    Two modes:
      - 'train': __getitem__ returns (ct, mask) for one randomly chosen timepoint
      - 'eval':  __getitem__ returns all timepoints for that slice position
    """

    def __init__(self, tensor_path, mode='train', filter_no_tumor=False):
        """
        Args:
            tensor_path: Path to the longitudinal_data.pt file
            mode: 'train' or 'eval'
            filter_no_tumor: If True, keep only slice positions where at least one
                             timepoint contains tumor. If False, keep everything
                             (more data for training the generative model).
        """
        all_data = torch.load(tensor_path)  # List of dicts
        self.mode = mode

        if filter_no_tumor:
            self.data = [item for item in all_data if item.get('has_tumor', True)]
            print(f"Filtered dataset: {len(self.data)}/{len(all_data)} slices have tumor")
        else:
            self.data = all_data
            print(f"Unfiltered dataset: {len(self.data)} total slices")

        # Index for fast lookup: (patient_id, slice_idx) -> position in self.data
        self.patient_slice_index = {
            (item['patient_id'], item['slice_idx']): idx
            for idx, item in enumerate(self.data)
        }

        self.patient_ids = sorted({item['patient_id'] for item in self.data})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.mode == 'train':
            n_timepoints = item['ct_slices'].shape[0]
            t_idx = torch.randint(0, n_timepoints, (1,)).item()
            ct = item['ct_slices'][t_idx]               # (1, H, W)
            mask = (item['masks'][t_idx] > 0).float()   # (1, H, W)
            return ct, mask

        elif self.mode == 'eval':
            return self._as_series(item)

        raise ValueError(f'Invalid mode: {self.mode}. Choose "train" or "eval".')

    @staticmethod
    def _as_series(item) -> dict:
        return {
            'ct_series': item['ct_slices'],              # (T, 1, H, W)
            'mask_series': (item['masks'] > 0).float(),  # (T, 1, H, W)
            'timepoints': item['timepoints'],            # List[int]
            'patient_id': item['patient_id'],
            'slice_idx': item['slice_idx'],
        }

    def query_by_patient_slice(self, patient_id: int, slice_idx: int) -> Optional[dict]:
        """Return all timepoints for one (patient, slice position), or None."""
        key = (patient_id, slice_idx)
        if key not in self.patient_slice_index:
            return None
        return self._as_series(self.data[self.patient_slice_index[key]])

    def get_patient_slices(self, patient_id: int) -> List[dict]:
        """Return all slice positions of one patient, sorted by slice_idx."""
        results = [self._as_series(item) for item in self.data
                   if item['patient_id'] == patient_id]
        return sorted(results, key=lambda x: x['slice_idx'])

    def get_available_patients(self) -> List[int]:
        """Return the list of patient IDs in the dataset."""
        return self.patient_ids.copy()
