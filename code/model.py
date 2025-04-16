import torch
import torch.nn.functional as F
import torch.nn.init as init
from momentfm import MOMENTPipeline
from torch import nn


class ExpandedClassificationHead(nn.Module):
    def __init__(self, input_features=1024, hidden_dim=64, num_classes=2, dropout_prob=0.1):
        super(ExpandedClassificationHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, input_mask):
        return self.head(x)


class EncoderComparisonModel(nn.Module):
    def __init__(self, encoder_name, encoder, encoding_dim, hidden_dim, num_classes, bnorm=True, actual_window=512):
        """
        :param encoder: Pre-trained encoder that takes 1xFxT input and outputs an encoding
        :param encoding_dim: Dimensionality of the output from the encoder
        :param hidden_dim: Dimensionality of the hidden layer in the MLP
        :param output_dim: Dimensionality of the final output (e.g., binary classification = 1, or multi-class = number of classes)
        """
        super(EncoderComparisonModel, self).__init__()
        self.encoder_name = encoder_name
        self.encoder = encoder
        self.actual_window = actual_window  # mask out some of the input to test smaller input windows
        if bnorm:
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),  # Additional dropout
                nn.Linear(hidden_dim, num_classes),  # Output logits for each class
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),  # Additional dropout
                nn.Linear(hidden_dim, num_classes),  # Output logits for each class
            )
        self.compress = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape Bx2xFxT
        """
        B, pair, F, T = x.shape
        assert pair == 2, "Input must contain pairs of sequences."

        # Split the input pairs
        x1 = x[:, 0, :, :].squeeze(1)  # Shape: Bx1xFxT
        x2 = x[:, 1, :, :].squeeze(1)  # Shape: Bx1xFxT

        # put back second dim if missing
        if len(x1.shape) == 2:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)

        # Process each 1xFxT slice through the pre-trained encoder
        if self.encoder_name == "MOMENT":
            if self.actual_window < 512:
                # zero out signal
                x1[:, :, self.actual_window :] = 0
                x2[:, :, self.actual_window :] = 0
                encoding1 = self.encoder(x_enc=x1).embeddings  # Shape: BxEncodingDim
                encoding2 = self.encoder(x_enc=x2).embeddings  # Shape: BxEncodingDim

            else:
                # import IPython; IPython.embed()
                encoding1 = self.encoder(x_enc=x1).embeddings  # Shape: BxEncodingDim
                encoding2 = self.encoder(x_enc=x2).embeddings  # Shape: BxEncodingDim

        elif self.encoder_name == "CNN":
            encoding1 = self.encoder(x1)
            encoding2 = self.encoder(x2)

        encoding1 = self.compress(encoding1)
        encoding2 = self.compress(encoding2)

        # Diff the encodings from both pairs
        combined_encoding = torch.cat([encoding1 - encoding2], dim=1)

        # Pass the concatenated encoding through the fully connected MLP
        logits = self.fc(combined_encoding)  # Shape: BxNumClasses
        # import IPython; IPython.embed()
        return logits


class CNNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, t_pool_size=32):
        super(CNNEncoder, self).__init__()

        # Initial layers that operate on W only
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)  # W -> W, D -> 64
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)  # W -> W, D -> 128
        self.bn2 = nn.BatchNorm1d(128)

        # Residual block that operates on W only
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, padding=3)  # W -> W, D stays the same
        self.bn3 = nn.BatchNorm1d(128)

        # Aggregating across both D and W
        self.conv4 = nn.Conv1d(128, 256, kernel_size=7, padding=3)  # W -> W, D -> 256
        self.bn4 = nn.BatchNorm1d(256)

        # Downsample and aggregate to reach receptive field
        self.pool = nn.AdaptiveMaxPool1d(output_size=t_pool_size)

        # Residual block that operates across D and W
        self.conv5 = nn.Conv1d(256, 256, kernel_size=7, padding=0)  # W -> W, D stays the same
        self.conv6 = nn.Conv1d(256, 256, kernel_size=7, padding=0)  # W -> W, D stays the same
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(256)

        self.conv2d = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(256, 3))
        self.bn7 = nn.BatchNorm2d(256)

        # Final transformation to output shape BxM
        self.fc = nn.Linear(4608, output_dim)

    def forward(self, x):
        # Initial convolution layers (operate on W only)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        
        # Aggregating across both D and W
        out = F.relu(self.bn4(self.conv4(out)))

        # Downsampling with pooling to reach the desired receptive field
        out = self.pool(out)

        # Reshape input for Conv2d
        out = out.unsqueeze(1)  # Now x is B x 1 x F x W

        # Apply Conv2d across the F dimension
        out = F.relu(
            self.bn7(self.conv2d(out))
        )  # Output will be B x 64 x 1 x (W-2) if kernel_size=3, padding might be needed
        
        # Remove the singleton dimension if desired
        out = out.squeeze(2)  # Output will be B x 64 x (W-2)
        
        # Additional residual block operating on D and W
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        
        # Flatten and project to output
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)  # B x M
        
        return out


def kaiming_init_recursive(module):
    """Recursively applies Kaiming initialization to all layers with 2+ dimensional weights."""
    for submodule in module.children():
        # Only apply to layers with a weight attribute of 2+ dimensions
        if hasattr(submodule, "weight") and isinstance(submodule.weight, torch.Tensor):
            if submodule.weight.dim() >= 2:
                torch.nn.init.kaiming_normal_(submodule.weight, nonlinearity="relu")
            if hasattr(submodule, "bias") and submodule.bias is not None:
                torch.nn.init.constant_(submodule.bias, 0)

        # Recursively apply to submodules
        kaiming_init_recursive(submodule)


def initialize_weights(m):
    """Initialize weights after model creation."""
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)  # Kaiming uniform initialization
        if m.bias is not None:
            init.zeros_(m.bias)  # Initialize biases to zero


def load_model(cfg):  # device, dataset, weights_path, args):
    if cfg.model.name.startswith("MOMENT"):
        moment_model = MOMENTPipeline.from_pretrained(
            f"AutonLab/{cfg.model.name}",
            model_kwargs={
                "task_name": "embedding",
                "n_channels": len(cfg.data.data_keys),
                "dropout_prob": 0.0,
            },
        )
        moment_model.init()
        if "actual_window_obs" not in cfg.data:
            actual_window_obs = 512
        else:
            actual_window_obs = cfg.data.actual_window_obs
        if cfg.model.name.endswith("small"):
            model = EncoderComparisonModel("MOMENT", moment_model, 512, 32, 2, actual_window=actual_window_obs)  # SMALL
        elif cfg.model.name.endswith("base"):
            model = EncoderComparisonModel("MOMENT", moment_model, 768, 256, 2, actual_window=actual_window_obs)
        elif cfg.model.name.endswith("large"):
            model = EncoderComparisonModel(
                "MOMENT", moment_model, 1024, 256, 2, actual_window=actual_window_obs
            )  # LARGE
        else:
            raise NotImplementedError
        # import IPython; IPython.embed()
        model.fc.apply(initialize_weights)
        model.compress.apply(initialize_weights)

    elif cfg.model.name.startswith("CNN"):
        CNNencoder = CNNEncoder(input_dim=4, output_dim=512)
        model = EncoderComparisonModel("CNN", CNNencoder, 512, 32, 2, bnorm=True)
        model.apply(kaiming_init_recursive)

    else:
        raise NotImplementedError

    return model