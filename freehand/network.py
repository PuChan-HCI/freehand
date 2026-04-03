
import torch


def build_model(model_function, in_frames, out_dim):
    """Build a torchvision backbone adapted to stacked ultrasound frames.

    The training pipeline stacks multiple grayscale frames along the channel
    dimension, so the first convolution is replaced to accept ``in_frames``
    channels. The final classification layer is also replaced so the network
    predicts a vector of length ``out_dim``.
    """
    model_name = model_function.__name__

    if model_name[:12] == "efficientnet":
        model = model_function(weights=None)
        stem = model.features[0][0]

        # EfficientNet starts with a Conv2d inside features[0][0].
        model.features[0][0] = torch.nn.Conv2d(
            in_channels  = in_frames, 
            out_channels = stem.out_channels, 
            kernel_size  = stem.kernel_size, 
            stride       = stem.stride, 
            padding      = stem.padding, 
            bias         = stem.bias is not None
        )

        # Swap the classifier head so its output matches the prediction target.
        model.classifier[1] = torch.nn.Linear(
            in_features   = model.classifier[1].in_features,
            out_features  = out_dim
        )
    elif model_name[:6] == "resnet":
        model = model_function(weights=None)

        # ResNet exposes the input stem as conv1 rather than via features.
        stem = model.conv1
        model.conv1 = torch.nn.Conv2d(
            in_channels  = in_frames, 
            out_channels = stem.out_channels, 
            kernel_size  = stem.kernel_size, 
            stride       = stem.stride, 
            padding      = stem.padding, 
            bias         = stem.bias is not None
        )
        model.fc = torch.nn.Linear(
            in_features   = model.fc.in_features,
            out_features  = out_dim
        )
    else:
        raise ValueError(f"Unknown model family: {model_name}")
    
    return model