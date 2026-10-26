"""
Builds the PatchCore feature extractor (ONNX, CPU / onnxruntime) from ImageNet-pretrained Keras
weights, without PyTorch or TensorFlow (numpy + h5py + onnx). Batch norm is folded into the convs.

    python tools/build_patchcore_onnx.py mobilenet_v2  <weights.h5> models/patchcore_mobilenetv2.onnx
    python tools/build_patchcore_onnx.py resnet50      <weights.h5> models/patchcore_resnet50.onnx

Weights (download once on a PC with internet; the built .onnx is committed to the repository so
the production PCs never download anything):
    mobilenet_v2: https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/
                  mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
    resnet50:     https://storage.googleapis.com/tensorflow/keras-applications/resnet/
                  resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

Outputs ("feat_s8", "feat_s16"): the two mid-level feature maps PatchCore uses, at stride 8 and 16.
    mobilenet_v2: block 6 / block 13 expansion (192 / 576 channels), input x / 127.5 - 1
    resnet50:     conv3 (layer2) / conv4 (layer3) block outputs (512 / 1024 channels),
                  input BGR - [103.939, 116.779, 123.68]
Input: (N, 3, H, W) float32, already preprocessed as above.
"""
import os
import sys

import h5py
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


class Builder:
    def __init__(self, h5_path, eps):
        self.f = h5py.File(h5_path, "r")
        self.eps = eps
        self.nodes, self.inits = [], []

    def w(self, layer, name):
        return np.array(self.f[layer][f"{layer}/{name}:0"])

    def conv_bn(self, x, conv, bn, stride, pads, act, out, depthwise=False, bias=True):
        k = self.w(conv, "depthwise_kernel" if depthwise else "kernel")
        b = self.w(conv, "bias") if bias else 0.0
        g, beta = self.w(bn, "gamma"), self.w(bn, "beta")
        mean, var = self.w(bn, "moving_mean"), self.w(bn, "moving_variance")
        s = g / np.sqrt(var + self.eps)
        if depthwise: # (kh, kw, C, 1) -> (C, 1, kh, kw)
            W = (k[:, :, :, 0] * s).transpose(2, 0, 1)[:, None].astype(np.float32)
            group = k.shape[2]
        else: # (kh, kw, in, out) -> (out, in, kh, kw)
            W = (k * s).transpose(3, 2, 0, 1).astype(np.float32)
            group = 1
        B = ((b - mean) * s + beta).astype(np.float32)
        self.inits += [numpy_helper.from_array(W, conv + "_W"), numpy_helper.from_array(B, conv + "_B")]
        y = out + "_pre" if act else out
        self.nodes.append(helper.make_node("Conv", [x, conv + "_W", conv + "_B"], [y], strides=[stride, stride],
                                           pads=pads, kernel_shape=list(k.shape[:2]), group=group))
        if act == "relu":
            self.nodes.append(helper.make_node("Relu", [y], [out]))
        elif act == "relu6":
            if "c0" not in [i.name for i in self.inits]:
                self.inits += [numpy_helper.from_array(np.array(0, np.float32), "c0"),
                               numpy_helper.from_array(np.array(6, np.float32), "c6")]
            self.nodes.append(helper.make_node("Clip", [y, "c0", "c6"], [out]))
        return out

    def save(self, outputs, channels, path):
        graph = helper.make_graph(
            self.nodes, "patchcore_features",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])],
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, ["N", c, f"h{i}", f"w{i}"])
             for i, (n, c) in enumerate(zip(outputs, channels))], self.inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        onnx.save(model, path)
        print(f"saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def mobilenet_v2(h5, out):
    b = Builder(h5, 1e-3)
    same2 = [0, 0, 1, 1] # TF 'same' padding, stride 2, even size: extra row / column at the end
    x = b.conv_bn("input", "Conv1", "bn_Conv1", 2, same2, "relu6", "conv1", bias=False)
    # block 0 (no expansion)
    x = b.conv_bn(x, "mobl0_conv_0_depthwise", "bn0_conv_0_bn_depthwise", 1, [1, 1, 1, 1], "relu6", "b0_dw",
                  depthwise=True, bias=False)
    x = b.conv_bn(x, "mobl0_conv_0_project", "bn0_conv_0_bn_project", 1, [0, 0, 0, 0], None, "b0_out", bias=False)
    stride2 = {1, 3, 6, 13}
    residual = {2, 4, 5, 7, 8, 9, 11, 12, 14, 15}
    outputs = {}
    for i in range(1, 14):
        e = b.conv_bn(x, f"mobl{i}_conv_{i}_expand", f"bn{i}_conv_{i}_bn_expand", 1, [0, 0, 0, 0], "relu6",
                      f"b{i}_expand", bias=False)
        if i == 6:
            outputs["feat_s8"] = e # stride 8, 192 channels
        if i == 13:
            outputs["feat_s16"] = e # stride 16, 576 channels
            break
        s = 2 if i in stride2 else 1
        d = b.conv_bn(e, f"mobl{i}_conv_{i}_depthwise", f"bn{i}_conv_{i}_bn_depthwise", s,
                      same2 if s == 2 else [1, 1, 1, 1], "relu6", f"b{i}_dw", depthwise=True, bias=False)
        pj = b.conv_bn(d, f"mobl{i}_conv_{i}_project", f"bn{i}_conv_{i}_bn_project", 1, [0, 0, 0, 0], None,
                       f"b{i}_pj", bias=False)
        if i in residual:
            b.nodes.append(helper.make_node("Add", [pj, x], [f"b{i}_out"]))
            x = f"b{i}_out"
        else:
            x = pj
    for name, src in outputs.items():
        b.nodes.append(helper.make_node("Identity", [src], [name]))
    b.save(["feat_s8", "feat_s16"], [192, 576], out)


def resnet50(h5, out):
    b = Builder(h5, 1.001e-5)
    x = b.conv_bn("input", "conv1_conv", "conv1_bn", 2, [3, 3, 3, 3], "relu", "conv1")
    b.nodes.append(helper.make_node("MaxPool", [x], ["pool1"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]))
    x = "pool1"
    for stage, blocks, first_stride in ((2, 3, 1), (3, 4, 2), (4, 6, 2)):
        for i in range(1, blocks + 1):
            n = f"conv{stage}_block{i}"
            s = first_stride if i == 1 else 1
            sc = b.conv_bn(x, n + "_0_conv", n + "_0_bn", s, [0, 0, 0, 0], None, n + "_sc") if i == 1 else x
            y = b.conv_bn(x, n + "_1_conv", n + "_1_bn", s, [0, 0, 0, 0], "relu", n + "_a")
            y = b.conv_bn(y, n + "_2_conv", n + "_2_bn", 1, [1, 1, 1, 1], "relu", n + "_b")
            y = b.conv_bn(y, n + "_3_conv", n + "_3_bn", 1, [0, 0, 0, 0], None, n + "_c")
            b.nodes.append(helper.make_node("Add", [y, sc], [n + "_sum"]))
            b.nodes.append(helper.make_node("Relu", [n + "_sum"], [n + "_out"]))
            x = n + "_out"
    b.nodes.append(helper.make_node("Identity", ["conv3_block4_out"], ["feat_s8"]))
    b.nodes.append(helper.make_node("Identity", ["conv4_block6_out"], ["feat_s16"]))
    b.save(["feat_s8", "feat_s16"], [512, 1024], out)


if __name__ == "__main__":
    arch, h5, out = sys.argv[1:4]
    {"mobilenet_v2": mobilenet_v2, "resnet50": resnet50}[arch](h5, out)
