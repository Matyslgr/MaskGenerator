##
## EPITECH PROJECT, 2025
## MaskGeneratorScaleway
## File description:
## qat_utils
##

import torch
import logging
import torch.quantization as tq
from torch.ao.quantization.observer import MinMaxObserver
from mask_generator.models.my_unet import MyUNet
import mask_generator.settings as settings

logger = logging.getLogger(settings.logger_name)

def get_default_qat_qconfig() -> tq.QConfig:
    """
    Returns the default QConfig for Quantization Aware Training (QAT).
    Args:
        backend (str): The quantization backend to use. Default is "fbgemm".
        Returns:
        tq.QConfig: The default QConfig for QAT.
    """
    activation_observer = MinMaxObserver.with_args(
        quant_min=0, quant_max=255, dtype=torch.quint8,
        qscheme=torch.per_tensor_affine
    )
    weight_observer = MinMaxObserver.with_args(
        quant_min=-128, quant_max=127, dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric
    )
    return tq.QConfig(activation=activation_observer, weight=weight_observer)

def prepare_qat_model(model: MyUNet, backend: str = "fbgemm") -> MyUNet:
    """
    Prepares a model for Quantization Aware Training (QAT).
    Args:
        model (MyUNet): The model to prepare for QAT.
        backend (str): The quantization backend to use. Default is "fbgemm".
    Returns:
        MyUNet: The model prepared for QAT.
    """
    if not isinstance(model, MyUNet):
        raise TypeError("model must be an instance of MyUNet")

    torch.backends.quantized.engine = backend
    model.eval()
    model.fuse_model()
    model.train()
    model.qconfig = get_default_qat_qconfig()
    tq.prepare_qat(model, inplace=True)
    logger.info(f"Model prepared for QAT with backend: {backend}")
    return model

def convert_qat_to_quantized(model: MyUNet) -> MyUNet:
    """
    Converts a Quantization Aware Training (QAT) model to a quantized model.
    Args:
        model (MyUNet): The QAT model to convert.
    Returns:
        MyUNet: The quantized model.
    """
    if not isinstance(model, MyUNet):
        raise TypeError("model must be an instance of MyUNet")

    model.eval()
    quantized_model = tq.convert(model, inplace=False)
    logger.info("Converted QAT model to quantized model")
    return quantized_model

def export_to_onnx(model: MyUNet, onnx_path: str, input_shape: tuple = (1, 3, 256, 256)) -> None:
    """
    Exports the model to ONNX format.
    Args:
        model (MyUNet): The model to export.
        onnx_path (str): The path where the ONNX model will be saved.
        input_shape (tuple): The shape of the input tensor. Default is (1, 3, 256, 256).
    """
    if not isinstance(model, MyUNet):
        raise TypeError("model must be an instance of MyUNet")

    quantized_model = convert_qat_to_quantized(model)

    device = torch.device("cpu")

    quantized_model.to(device)
    quantized_model.eval()

    dummy_input = torch.randn(*input_shape)

    export_kwargs = {
        "model": quantized_model,
        "args": dummy_input,
        "f": onnx_path,
        "opset_version": 11,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        }
    }

    torch.onnx.export(**export_kwargs)
    logger.info(f"Exporting model to ONNX format at {onnx_path} with input shape {input_shape}")
