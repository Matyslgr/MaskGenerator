##
## EPITECH PROJECT, 2025
## MaskGeneratorScaleway
## File description:
## qat_utils
##

import copy
import torch
import logging
import torch.ao.quantization as tq
import torch.ao.quantization.quantize_fx as quantize_fx
import mask_generator.settings as settings
from mask_generator.models.my_unet import MyUNet

logger = logging.getLogger(settings.logger_name)

def get_custom_qconfig_mapping() -> tq.QConfigMapping:
    """
    Returns the default QConfigMapping for Quantization Aware Training (QAT).
    This mapping is used to specify how different layers in the model should be quantized.
    """
    weight_observer = tq.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    activation_observer = tq.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    qconfig = tq.QConfig(
        activation=activation_observer,
        weight=weight_observer
    )

    return tq.QConfigMapping().set_global(qconfig)

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

    model_to_quantize = copy.deepcopy(model)

    model_to_quantize.eval()
    model_to_quantize.fuse_model()
    model_to_quantize.train()

    qconfig_mapping = get_custom_qconfig_mapping()

    print("QConfigMapping global:", qconfig_mapping.global_qconfig)

    example_inputs = (torch.randn(1, 3, 256, 256),)
    model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_mapping, example_inputs)

    logger.info(f"Model prepared for QAT with backend: {backend}")
    return model_prepared

def convert_qat_to_quantized(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Converts a Quantization Aware Training (QAT) model to a quantized model.
    Args:
        model (torch.fx.GraphModule): The QAT model to convert.
    Returns:
        torch.fx.GraphModule: The quantized model.
    """
    if not isinstance(model, torch.fx.GraphModule):
        raise TypeError("model must be an instance of torch.fx.GraphModule")

    model.eval()

    # Check if the model is on cpu
    if next(model.parameters()).device.type != 'cpu':
        device = torch.device("cpu")
        model.to(device)

    qconfig_mapping = get_custom_qconfig_mapping()

    quantized_model = quantize_fx.convert_fx(model, qconfig_mapping=qconfig_mapping)
    logger.info("Converted QAT model to quantized model")

    for name, module in quantized_model.named_modules():
        print(f"Module: {name}, Type: {module.__class__.__name__}")

    print("\nOperations dans le graph :")
    quantized_model.graph.print_tabular()

    return quantized_model

def export_to_onnx(model: torch.fx.GraphModule, onnx_path: str, input_shape: tuple = (1, 3, 256, 256)):
    """
    Exports the model to ONNX format.
    Args:
        model (torch.fx.GraphModule): The model to export.
        onnx_path (str): The path where the ONNX model will be saved.
        input_shape (tuple): The shape of the input tensor. Default is (1, 3, 256, 256).
    """
    model.eval()
    device = torch.device("cpu")

    # Check if the model is on cpu
    if next(model.parameters()).device.type != 'cpu':
        model.to(device)

    dummy_input = torch.randn(*input_shape, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        f=onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        },
        training=torch.onnx.TrainingMode.EVAL,
    )
    logger.info(f"Exporting model to ONNX format at {onnx_path} with input shape {input_shape}")
