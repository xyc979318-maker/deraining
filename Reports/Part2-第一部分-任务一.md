完整版见python文件，主要修改如下：

#### 一、

```python
parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "unet","DerainNet","Mymodel"], help="Model to use")
```

#### 二、

```python
import torch
import torch.nn.functional as F
from torch import nn


class DerainNet(nn.Module):
    def __init__(self):
        super(DerainNet, self).__init__()
        pass
    def forward(self, x):
        pass
```

```python
from models.baseline_net import BaselineNet
from models.unet import UNet
from models.Mymodel import Mymodel
from models.DerainNet import DerainNet
from losses.perceptual_loss import PerceptualLoss
```

#### 三、

```python
def main():
    args = get_args()
    device = "mps" if torch.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.model == "baseline":
        model = BaselineNet().to(device)
    elif args.model == "unet":
        model = UNet().to(device)
    elif args.model == "DerainNet":
        model = DerainNet().to(device)
    elif args.model == "Mymodel":
        model = Mymodel().to(device)
```

### 