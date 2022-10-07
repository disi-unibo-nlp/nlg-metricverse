import torch
from transformers import AutoModel

from nlgmetricverse.utils.common import cache_scibert


def get_model(model_type, num_layers, all_layers=None):
    if model_type.startswith("scibert"):
        model = AutoModel.from_pretrained(cache_scibert(model_type))
    elif "t5" in model_type:
        from transformers import T5EncoderModel

        model = T5EncoderModel.from_pretrained(model_type)
    else:
        model = AutoModel.from_pretrained(model_type)
    model.eval()

    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        model = model.encoder

    # drop unused layers
    if not all_layers:
        if hasattr(model, "n_layers"):  # xlm
            assert (
                    0 <= num_layers <= model.n_layers
            ), f"Invalid num_layers: num_layers should be between 0 and {model.n_layers} for {model_type}"
            model.n_layers = num_layers
        elif hasattr(model, "layer"):  # xlnet
            assert (
                    0 <= num_layers <= len(model.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layer)} for {model_type}"
            model.layer = torch.nn.ModuleList([layer for layer in model.layer[:num_layers]])
        elif hasattr(model, "encoder"):  # albert
            if hasattr(model.encoder, "albert_layer_groups"):
                assert (
                        0 <= num_layers <= model.encoder.config.num_hidden_layers
                ), f"Invalid num_layers: num_layers should be between 0 and {model.encoder.config.num_hidden_layers} for {model_type}"
                model.encoder.config.num_hidden_layers = num_layers
            elif hasattr(model.encoder, "block"):  # t5
                assert (
                        0 <= num_layers <= len(model.encoder.block)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.block)} for {model_type}"
                model.encoder.block = torch.nn.ModuleList([layer for layer in model.encoder.block[:num_layers]])
            else:  # bert, roberta
                assert (
                        0 <= num_layers <= len(model.encoder.layer)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.layer)} for {model_type}"
                model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])
        elif hasattr(model, "transformer"):  # bert, roberta
            assert (
                    0 <= num_layers <= len(model.transformer.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.transformer.layer)} for {model_type}"
            model.transformer.layer = torch.nn.ModuleList([layer for layer in model.transformer.layer[:num_layers]])
        elif hasattr(model, "layers"):  # bart
            assert (
                    0 <= num_layers <= len(model.layers)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layers)} for {model_type}"
            model.layers = torch.nn.ModuleList([layer for layer in model.layers[:num_layers]])
        else:
            raise ValueError("Not supported")
    else:
        if hasattr(model, "output_hidden_states"):
            model.output_hidden_states = True
        elif hasattr(model, "encoder"):
            model.encoder.output_hidden_states = True
        elif hasattr(model, "transformer"):
            model.transformer.output_hidden_states = True
        # else:
        #     raise ValueError(f"Not supported model architecture: {model_type}")

    return model
