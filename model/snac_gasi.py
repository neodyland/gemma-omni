from torch import nn
from snac import SNAC
import torch


class SnacGasi(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.size = self.model.codebook_size
        self.sr = 24000

    def encode(self, ids_bef: torch.Tensor):
        bs = ids_bef.size(0)
        ids = self.model.encode(ids_bef)
        ids_new_all = []
        for b in range(bs):
            ids_new = []
            for i in range(len(ids[0][b])):
                ids_new.extend(ids[0][b][i : i + 1].tolist())
                ids_new.extend(ids[1][b][i * 2 : i * 2 + 2].tolist())
                ids_new.extend(ids[2][b][i * 4 : i * 4 + 4].tolist())
            ids_new_all.append(ids_new)
        return torch.tensor(ids_new_all).to(ids_bef.device, non_blocking=True)

    def decode(self, ids: torch.Tensor):
        bs = len(ids)
        ids_old_all_1 = []
        ids_old_all_2 = []
        ids_old_all_3 = []
        for b in range(bs):
            ids_old_1 = []
            ids_old_2 = []
            ids_old_3 = []
            for i in range(0, len(ids[b]), 7):
                ids_old_1.extend(
                    [ids[b][i]],
                )
                ids_old_2.extend(
                    [ids[b][i + 1], ids[b][i + 2]],
                )
                ids_old_3.extend(
                    [ids[b][i + 3], ids[b][i + 4], ids[b][i + 5], ids[b][i + 6]],
                )
            ids_old_all_1.append(ids_old_1)
            ids_old_all_2.append(ids_old_2)
            ids_old_all_3.append(ids_old_3)
        return self.model.decode(
            [
                torch.tensor(x).to(ids.device, dtype=ids.dtype, non_blocking=True)
                for x in [ids_old_all_1, ids_old_all_2, ids_old_all_3]
            ]
        )
