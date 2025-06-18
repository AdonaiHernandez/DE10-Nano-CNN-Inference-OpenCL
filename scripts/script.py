# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(input: R.Tensor((1, 3, 240, 320), dtype="float32")) -> R.Tuple(R.Tensor((1, 4420, 2), dtype="float32"), R.Tensor((1, 4420, 4), dtype="float32")):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            lv: R.Tensor((1, 16, 120, 160), dtype="float32") = R.nn.conv2d(input, metadata["relax.expr.Constant"][0], strides=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv1: R.Tuple(R.Tensor((1, 16, 120, 160), dtype="float32"), R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")) = R.nn.batch_norm(lv, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2], metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv2: R.Tensor((1, 16, 120, 160), dtype="float32") = lv1[0]
            lv3: R.Tensor((16,), dtype="float32") = lv1[1]
            lv4: R.Tensor((16,), dtype="float32") = lv1[2]
            lv5: R.Tensor((1, 16, 120, 160), dtype="float32") = R.nn.relu(lv2)
            lv6: R.Tensor((1, 16, 120, 160), dtype="float32") = R.nn.conv2d(lv5, metadata["relax.expr.Constant"][5], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=16, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv7: R.Tuple(R.Tensor((1, 16, 120, 160), dtype="float32"), R.Tensor((16,), dtype="float32"), R.Tensor((16,), dtype="float32")) = R.nn.batch_norm(lv6, metadata["relax.expr.Constant"][6], metadata["relax.expr.Constant"][7], metadata["relax.expr.Constant"][8], metadata["relax.expr.Constant"][9], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv8: R.Tensor((1, 16, 120, 160), dtype="float32") = lv7[0]
            lv9: R.Tensor((16,), dtype="float32") = lv7[1]
            lv10: R.Tensor((16,), dtype="float32") = lv7[2]
            lv11: R.Tensor((1, 16, 120, 160), dtype="float32") = R.nn.relu(lv8)
            lv12: R.Tensor((1, 32, 120, 160), dtype="float32") = R.nn.conv2d(lv11, metadata["relax.expr.Constant"][10], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv13: R.Tuple(R.Tensor((1, 32, 120, 160), dtype="float32"), R.Tensor((32,), dtype="float32"), R.Tensor((32,), dtype="float32")) = R.nn.batch_norm(lv12, metadata["relax.expr.Constant"][11], metadata["relax.expr.Constant"][12], metadata["relax.expr.Constant"][13], metadata["relax.expr.Constant"][14], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv14: R.Tensor((1, 32, 120, 160), dtype="float32") = lv13[0]
            lv15: R.Tensor((32,), dtype="float32") = lv13[1]
            lv16: R.Tensor((32,), dtype="float32") = lv13[2]
            lv17: R.Tensor((1, 32, 120, 160), dtype="float32") = R.nn.relu(lv14)
            lv18: R.Tensor((1, 32, 60, 80), dtype="float32") = R.nn.conv2d(lv17, metadata["relax.expr.Constant"][15], strides=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=32, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv19: R.Tuple(R.Tensor((1, 32, 60, 80), dtype="float32"), R.Tensor((32,), dtype="float32"), R.Tensor((32,), dtype="float32")) = R.nn.batch_norm(lv18, metadata["relax.expr.Constant"][16], metadata["relax.expr.Constant"][17], metadata["relax.expr.Constant"][18], metadata["relax.expr.Constant"][19], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv20: R.Tensor((1, 32, 60, 80), dtype="float32") = lv19[0]
            lv21: R.Tensor((32,), dtype="float32") = lv19[1]
            lv22: R.Tensor((32,), dtype="float32") = lv19[2]
            lv23: R.Tensor((1, 32, 60, 80), dtype="float32") = R.nn.relu(lv20)
            lv24: R.Tensor((1, 32, 60, 80), dtype="float32") = R.nn.conv2d(lv23, metadata["relax.expr.Constant"][20], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv25: R.Tuple(R.Tensor((1, 32, 60, 80), dtype="float32"), R.Tensor((32,), dtype="float32"), R.Tensor((32,), dtype="float32")) = R.nn.batch_norm(lv24, metadata["relax.expr.Constant"][21], metadata["relax.expr.Constant"][22], metadata["relax.expr.Constant"][23], metadata["relax.expr.Constant"][24], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv26: R.Tensor((1, 32, 60, 80), dtype="float32") = lv25[0]
            lv27: R.Tensor((32,), dtype="float32") = lv25[1]
            lv28: R.Tensor((32,), dtype="float32") = lv25[2]
            lv29: R.Tensor((1, 32, 60, 80), dtype="float32") = R.nn.relu(lv26)
            lv30: R.Tensor((1, 32, 60, 80), dtype="float32") = R.nn.conv2d(lv29, metadata["relax.expr.Constant"][25], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=32, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv31: R.Tuple(R.Tensor((1, 32, 60, 80), dtype="float32"), R.Tensor((32,), dtype="float32"), R.Tensor((32,), dtype="float32")) = R.nn.batch_norm(lv30, metadata["relax.expr.Constant"][26], metadata["relax.expr.Constant"][27], metadata["relax.expr.Constant"][28], metadata["relax.expr.Constant"][29], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv32: R.Tensor((1, 32, 60, 80), dtype="float32") = lv31[0]
            lv33: R.Tensor((32,), dtype="float32") = lv31[1]
            lv34: R.Tensor((32,), dtype="float32") = lv31[2]
            lv35: R.Tensor((1, 32, 60, 80), dtype="float32") = R.nn.relu(lv32)
            lv36: R.Tensor((1, 32, 60, 80), dtype="float32") = R.nn.conv2d(lv35, metadata["relax.expr.Constant"][30], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv37: R.Tuple(R.Tensor((1, 32, 60, 80), dtype="float32"), R.Tensor((32,), dtype="float32"), R.Tensor((32,), dtype="float32")) = R.nn.batch_norm(lv36, metadata["relax.expr.Constant"][31], metadata["relax.expr.Constant"][32], metadata["relax.expr.Constant"][33], metadata["relax.expr.Constant"][34], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv38: R.Tensor((1, 32, 60, 80), dtype="float32") = lv37[0]
            lv39: R.Tensor((32,), dtype="float32") = lv37[1]
            lv40: R.Tensor((32,), dtype="float32") = lv37[2]
            lv41: R.Tensor((1, 32, 60, 80), dtype="float32") = R.nn.relu(lv38)
            lv42: R.Tensor((1, 32, 30, 40), dtype="float32") = R.nn.conv2d(lv41, metadata["relax.expr.Constant"][35], strides=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=32, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv43: R.Tuple(R.Tensor((1, 32, 30, 40), dtype="float32"), R.Tensor((32,), dtype="float32"), R.Tensor((32,), dtype="float32")) = R.nn.batch_norm(lv42, metadata["relax.expr.Constant"][36], metadata["relax.expr.Constant"][37], metadata["relax.expr.Constant"][38], metadata["relax.expr.Constant"][39], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv44: R.Tensor((1, 32, 30, 40), dtype="float32") = lv43[0]
            lv45: R.Tensor((32,), dtype="float32") = lv43[1]
            lv46: R.Tensor((32,), dtype="float32") = lv43[2]
            lv47: R.Tensor((1, 32, 30, 40), dtype="float32") = R.nn.relu(lv44)
            lv48: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.conv2d(lv47, metadata["relax.expr.Constant"][40], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv49: R.Tuple(R.Tensor((1, 64, 30, 40), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = R.nn.batch_norm(lv48, metadata["relax.expr.Constant"][41], metadata["relax.expr.Constant"][42], metadata["relax.expr.Constant"][43], metadata["relax.expr.Constant"][44], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv50: R.Tensor((1, 64, 30, 40), dtype="float32") = lv49[0]
            lv51: R.Tensor((64,), dtype="float32") = lv49[1]
            lv52: R.Tensor((64,), dtype="float32") = lv49[2]
            lv53: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.relu(lv50)
            lv54: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.conv2d(lv53, metadata["relax.expr.Constant"][45], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=64, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv55: R.Tuple(R.Tensor((1, 64, 30, 40), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = R.nn.batch_norm(lv54, metadata["relax.expr.Constant"][46], metadata["relax.expr.Constant"][47], metadata["relax.expr.Constant"][48], metadata["relax.expr.Constant"][49], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv56: R.Tensor((1, 64, 30, 40), dtype="float32") = lv55[0]
            lv57: R.Tensor((64,), dtype="float32") = lv55[1]
            lv58: R.Tensor((64,), dtype="float32") = lv55[2]
            lv59: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.relu(lv56)
            lv60: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.conv2d(lv59, metadata["relax.expr.Constant"][50], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv61: R.Tuple(R.Tensor((1, 64, 30, 40), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = R.nn.batch_norm(lv60, metadata["relax.expr.Constant"][51], metadata["relax.expr.Constant"][52], metadata["relax.expr.Constant"][53], metadata["relax.expr.Constant"][54], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv62: R.Tensor((1, 64, 30, 40), dtype="float32") = lv61[0]
            lv63: R.Tensor((64,), dtype="float32") = lv61[1]
            lv64: R.Tensor((64,), dtype="float32") = lv61[2]
            lv65: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.relu(lv62)
            lv66: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.conv2d(lv65, metadata["relax.expr.Constant"][55], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=64, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv67: R.Tuple(R.Tensor((1, 64, 30, 40), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = R.nn.batch_norm(lv66, metadata["relax.expr.Constant"][56], metadata["relax.expr.Constant"][57], metadata["relax.expr.Constant"][58], metadata["relax.expr.Constant"][59], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv68: R.Tensor((1, 64, 30, 40), dtype="float32") = lv67[0]
            lv69: R.Tensor((64,), dtype="float32") = lv67[1]
            lv70: R.Tensor((64,), dtype="float32") = lv67[2]
            lv71: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.relu(lv68)
            lv72: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.conv2d(lv71, metadata["relax.expr.Constant"][60], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv73: R.Tuple(R.Tensor((1, 64, 30, 40), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = R.nn.batch_norm(lv72, metadata["relax.expr.Constant"][61], metadata["relax.expr.Constant"][62], metadata["relax.expr.Constant"][63], metadata["relax.expr.Constant"][64], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv74: R.Tensor((1, 64, 30, 40), dtype="float32") = lv73[0]
            lv75: R.Tensor((64,), dtype="float32") = lv73[1]
            lv76: R.Tensor((64,), dtype="float32") = lv73[2]
            lv77: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.relu(lv74)
            lv78: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.conv2d(lv77, metadata["relax.expr.Constant"][65], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=64, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv79: R.Tuple(R.Tensor((1, 64, 30, 40), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = R.nn.batch_norm(lv78, metadata["relax.expr.Constant"][66], metadata["relax.expr.Constant"][67], metadata["relax.expr.Constant"][68], metadata["relax.expr.Constant"][69], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv80: R.Tensor((1, 64, 30, 40), dtype="float32") = lv79[0]
            lv81: R.Tensor((64,), dtype="float32") = lv79[1]
            lv82: R.Tensor((64,), dtype="float32") = lv79[2]
            lv83: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.relu(lv80)
            lv84: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.conv2d(lv83, metadata["relax.expr.Constant"][70], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv85: R.Tuple(R.Tensor((1, 64, 30, 40), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = R.nn.batch_norm(lv84, metadata["relax.expr.Constant"][71], metadata["relax.expr.Constant"][72], metadata["relax.expr.Constant"][73], metadata["relax.expr.Constant"][74], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv86: R.Tensor((1, 64, 30, 40), dtype="float32") = lv85[0]
            lv87: R.Tensor((64,), dtype="float32") = lv85[1]
            lv88: R.Tensor((64,), dtype="float32") = lv85[2]
            lv89: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.relu(lv86)
            lv90: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.conv2d(lv89, metadata["relax.expr.Constant"][75], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=64, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv91: R.Tensor((1, 64, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][76], R.shape([1, 64, 1, 1]))
            lv92: R.Tensor((1, 64, 30, 40), dtype="float32") = R.add(lv90, lv91)
            lv93: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.relu(lv92)
            lv94: R.Tensor((1, 6, 30, 40), dtype="float32") = R.nn.conv2d(lv93, metadata["relax.expr.Constant"][77], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv95: R.Tensor((1, 6, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][78], R.shape([1, 6, 1, 1]))
            lv96: R.Tensor((1, 6, 30, 40), dtype="float32") = R.add(lv94, lv95)
            lv97: R.Tensor((1, 30, 40, 6), dtype="float32") = R.permute_dims(lv96, axes=[0, 2, 3, 1])
            lv98: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.conv2d(lv89, metadata["relax.expr.Constant"][79], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=64, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv99: R.Tensor((1, 64, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][80], R.shape([1, 64, 1, 1]))
            lv100: R.Tensor((1, 64, 30, 40), dtype="float32") = R.add(lv98, lv99)
            lv101: R.Tensor((1, 64, 30, 40), dtype="float32") = R.nn.relu(lv100)
            lv102: R.Tensor((1, 12, 30, 40), dtype="float32") = R.nn.conv2d(lv101, metadata["relax.expr.Constant"][81], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv103: R.Tensor((1, 12, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][82], R.shape([1, 12, 1, 1]))
            lv104: R.Tensor((1, 12, 30, 40), dtype="float32") = R.add(lv102, lv103)
            lv105: R.Tensor((1, 30, 40, 12), dtype="float32") = R.permute_dims(lv104, axes=[0, 2, 3, 1])
            lv106: R.Tensor((1, 64, 15, 20), dtype="float32") = R.nn.conv2d(lv89, metadata["relax.expr.Constant"][83], strides=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=64, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv107: R.Tuple(R.Tensor((1, 64, 15, 20), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = R.nn.batch_norm(lv106, metadata["relax.expr.Constant"][84], metadata["relax.expr.Constant"][85], metadata["relax.expr.Constant"][86], metadata["relax.expr.Constant"][87], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv108: R.Tensor((1, 64, 15, 20), dtype="float32") = lv107[0]
            lv109: R.Tensor((64,), dtype="float32") = lv107[1]
            lv110: R.Tensor((64,), dtype="float32") = lv107[2]
            lv111: R.Tensor((1, 64, 15, 20), dtype="float32") = R.nn.relu(lv108)
            lv112: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.conv2d(lv111, metadata["relax.expr.Constant"][88], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv113: R.Tuple(R.Tensor((1, 128, 15, 20), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = R.nn.batch_norm(lv112, metadata["relax.expr.Constant"][89], metadata["relax.expr.Constant"][90], metadata["relax.expr.Constant"][91], metadata["relax.expr.Constant"][92], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv114: R.Tensor((1, 128, 15, 20), dtype="float32") = lv113[0]
            lv115: R.Tensor((128,), dtype="float32") = lv113[1]
            lv116: R.Tensor((128,), dtype="float32") = lv113[2]
            lv117: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.relu(lv114)
            lv118: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.conv2d(lv117, metadata["relax.expr.Constant"][93], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=128, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv119: R.Tuple(R.Tensor((1, 128, 15, 20), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = R.nn.batch_norm(lv118, metadata["relax.expr.Constant"][94], metadata["relax.expr.Constant"][95], metadata["relax.expr.Constant"][96], metadata["relax.expr.Constant"][97], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv120: R.Tensor((1, 128, 15, 20), dtype="float32") = lv119[0]
            lv121: R.Tensor((128,), dtype="float32") = lv119[1]
            lv122: R.Tensor((128,), dtype="float32") = lv119[2]
            lv123: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.relu(lv120)
            lv124: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.conv2d(lv123, metadata["relax.expr.Constant"][98], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv125: R.Tuple(R.Tensor((1, 128, 15, 20), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = R.nn.batch_norm(lv124, metadata["relax.expr.Constant"][99], metadata["relax.expr.Constant"][100], metadata["relax.expr.Constant"][101], metadata["relax.expr.Constant"][102], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv126: R.Tensor((1, 128, 15, 20), dtype="float32") = lv125[0]
            lv127: R.Tensor((128,), dtype="float32") = lv125[1]
            lv128: R.Tensor((128,), dtype="float32") = lv125[2]
            lv129: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.relu(lv126)
            lv130: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.conv2d(lv129, metadata["relax.expr.Constant"][103], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=128, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv131: R.Tuple(R.Tensor((1, 128, 15, 20), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = R.nn.batch_norm(lv130, metadata["relax.expr.Constant"][104], metadata["relax.expr.Constant"][105], metadata["relax.expr.Constant"][106], metadata["relax.expr.Constant"][107], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv132: R.Tensor((1, 128, 15, 20), dtype="float32") = lv131[0]
            lv133: R.Tensor((128,), dtype="float32") = lv131[1]
            lv134: R.Tensor((128,), dtype="float32") = lv131[2]
            lv135: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.relu(lv132)
            lv136: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.conv2d(lv135, metadata["relax.expr.Constant"][108], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv137: R.Tuple(R.Tensor((1, 128, 15, 20), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = R.nn.batch_norm(lv136, metadata["relax.expr.Constant"][109], metadata["relax.expr.Constant"][110], metadata["relax.expr.Constant"][111], metadata["relax.expr.Constant"][112], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv138: R.Tensor((1, 128, 15, 20), dtype="float32") = lv137[0]
            lv139: R.Tensor((128,), dtype="float32") = lv137[1]
            lv140: R.Tensor((128,), dtype="float32") = lv137[2]
            lv141: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.relu(lv138)
            lv142: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.conv2d(lv141, metadata["relax.expr.Constant"][113], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=128, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv143: R.Tensor((1, 128, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][114], R.shape([1, 128, 1, 1]))
            lv144: R.Tensor((1, 128, 15, 20), dtype="float32") = R.add(lv142, lv143)
            lv145: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.relu(lv144)
            lv146: R.Tensor((1, 4, 15, 20), dtype="float32") = R.nn.conv2d(lv145, metadata["relax.expr.Constant"][115], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv147: R.Tensor((1, 4, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][116], R.shape([1, 4, 1, 1]))
            lv148: R.Tensor((1, 4, 15, 20), dtype="float32") = R.add(lv146, lv147)
            lv149: R.Tensor((1, 15, 20, 4), dtype="float32") = R.permute_dims(lv148, axes=[0, 2, 3, 1])
            lv150: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.conv2d(lv141, metadata["relax.expr.Constant"][117], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=128, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv151: R.Tensor((1, 128, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][118], R.shape([1, 128, 1, 1]))
            lv152: R.Tensor((1, 128, 15, 20), dtype="float32") = R.add(lv150, lv151)
            lv153: R.Tensor((1, 128, 15, 20), dtype="float32") = R.nn.relu(lv152)
            lv154: R.Tensor((1, 8, 15, 20), dtype="float32") = R.nn.conv2d(lv153, metadata["relax.expr.Constant"][119], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv155: R.Tensor((1, 8, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][120], R.shape([1, 8, 1, 1]))
            lv156: R.Tensor((1, 8, 15, 20), dtype="float32") = R.add(lv154, lv155)
            lv157: R.Tensor((1, 15, 20, 8), dtype="float32") = R.permute_dims(lv156, axes=[0, 2, 3, 1])
            lv158: R.Tensor((1, 128, 8, 10), dtype="float32") = R.nn.conv2d(lv141, metadata["relax.expr.Constant"][121], strides=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=128, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv159: R.Tuple(R.Tensor((1, 128, 8, 10), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = R.nn.batch_norm(lv158, metadata["relax.expr.Constant"][122], metadata["relax.expr.Constant"][123], metadata["relax.expr.Constant"][124], metadata["relax.expr.Constant"][125], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv160: R.Tensor((1, 128, 8, 10), dtype="float32") = lv159[0]
            lv161: R.Tensor((128,), dtype="float32") = lv159[1]
            lv162: R.Tensor((128,), dtype="float32") = lv159[2]
            lv163: R.Tensor((1, 128, 8, 10), dtype="float32") = R.nn.relu(lv160)
            lv164: R.Tensor((1, 256, 8, 10), dtype="float32") = R.nn.conv2d(lv163, metadata["relax.expr.Constant"][126], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv165: R.Tuple(R.Tensor((1, 256, 8, 10), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = R.nn.batch_norm(lv164, metadata["relax.expr.Constant"][127], metadata["relax.expr.Constant"][128], metadata["relax.expr.Constant"][129], metadata["relax.expr.Constant"][130], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv166: R.Tensor((1, 256, 8, 10), dtype="float32") = lv165[0]
            lv167: R.Tensor((256,), dtype="float32") = lv165[1]
            lv168: R.Tensor((256,), dtype="float32") = lv165[2]
            lv169: R.Tensor((1, 256, 8, 10), dtype="float32") = R.nn.relu(lv166)
            lv170: R.Tensor((1, 256, 8, 10), dtype="float32") = R.nn.conv2d(lv169, metadata["relax.expr.Constant"][131], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=256, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv171: R.Tuple(R.Tensor((1, 256, 8, 10), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = R.nn.batch_norm(lv170, metadata["relax.expr.Constant"][132], metadata["relax.expr.Constant"][133], metadata["relax.expr.Constant"][134], metadata["relax.expr.Constant"][135], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv172: R.Tensor((1, 256, 8, 10), dtype="float32") = lv171[0]
            lv173: R.Tensor((256,), dtype="float32") = lv171[1]
            lv174: R.Tensor((256,), dtype="float32") = lv171[2]
            lv175: R.Tensor((1, 256, 8, 10), dtype="float32") = R.nn.relu(lv172)
            lv176: R.Tensor((1, 256, 8, 10), dtype="float32") = R.nn.conv2d(lv175, metadata["relax.expr.Constant"][136], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv177: R.Tuple(R.Tensor((1, 256, 8, 10), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = R.nn.batch_norm(lv176, metadata["relax.expr.Constant"][137], metadata["relax.expr.Constant"][138], metadata["relax.expr.Constant"][139], metadata["relax.expr.Constant"][140], axis=1, epsilon=9.9999997473787516e-06, center=True, scale=True, momentum=0.10000000000000001, training=True)
            lv178: R.Tensor((1, 256, 8, 10), dtype="float32") = lv177[0]
            lv179: R.Tensor((256,), dtype="float32") = lv177[1]
            lv180: R.Tensor((256,), dtype="float32") = lv177[2]
            lv181: R.Tensor((1, 256, 8, 10), dtype="float32") = R.nn.relu(lv178)
            lv182: R.Tensor((1, 256, 8, 10), dtype="float32") = R.nn.conv2d(lv181, metadata["relax.expr.Constant"][141], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=256, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv183: R.Tensor((1, 256, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][142], R.shape([1, 256, 1, 1]))
            lv184: R.Tensor((1, 256, 8, 10), dtype="float32") = R.add(lv182, lv183)
            lv185: R.Tensor((1, 256, 8, 10), dtype="float32") = R.nn.relu(lv184)
            lv186: R.Tensor((1, 4, 8, 10), dtype="float32") = R.nn.conv2d(lv185, metadata["relax.expr.Constant"][143], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv187: R.Tensor((1, 4, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][144], R.shape([1, 4, 1, 1]))
            lv188: R.Tensor((1, 4, 8, 10), dtype="float32") = R.add(lv186, lv187)
            lv189: R.Tensor((1, 8, 10, 4), dtype="float32") = R.permute_dims(lv188, axes=[0, 2, 3, 1])
            lv190: R.Tensor((1, 256, 8, 10), dtype="float32") = R.nn.conv2d(lv181, metadata["relax.expr.Constant"][145], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=256, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv191: R.Tensor((1, 256, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][146], R.shape([1, 256, 1, 1]))
            lv192: R.Tensor((1, 256, 8, 10), dtype="float32") = R.add(lv190, lv191)
            lv193: R.Tensor((1, 256, 8, 10), dtype="float32") = R.nn.relu(lv192)
            lv194: R.Tensor((1, 8, 8, 10), dtype="float32") = R.nn.conv2d(lv193, metadata["relax.expr.Constant"][147], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv195: R.Tensor((1, 8, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][148], R.shape([1, 8, 1, 1]))
            lv196: R.Tensor((1, 8, 8, 10), dtype="float32") = R.add(lv194, lv195)
            lv197: R.Tensor((1, 8, 10, 8), dtype="float32") = R.permute_dims(lv196, axes=[0, 2, 3, 1])
            lv198: R.Tensor((1, 64, 8, 10), dtype="float32") = R.nn.conv2d(lv181, metadata["relax.expr.Constant"][149], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv199: R.Tensor((1, 64, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][150], R.shape([1, 64, 1, 1]))
            lv200: R.Tensor((1, 64, 8, 10), dtype="float32") = R.add(lv198, lv199)
            lv201: R.Tensor((1, 64, 8, 10), dtype="float32") = R.nn.relu(lv200)
            lv202: R.Tensor((1, 64, 4, 5), dtype="float32") = R.nn.conv2d(lv201, metadata["relax.expr.Constant"][151], strides=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=64, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv203: R.Tensor((1, 64, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][152], R.shape([1, 64, 1, 1]))
            lv204: R.Tensor((1, 64, 4, 5), dtype="float32") = R.add(lv202, lv203)
            lv205: R.Tensor((1, 64, 4, 5), dtype="float32") = R.nn.relu(lv204)
            lv206: R.Tensor((1, 256, 4, 5), dtype="float32") = R.nn.conv2d(lv205, metadata["relax.expr.Constant"][153], strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv207: R.Tensor((1, 256, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][154], R.shape([1, 256, 1, 1]))
            lv208: R.Tensor((1, 256, 4, 5), dtype="float32") = R.add(lv206, lv207)
            lv209: R.Tensor((1, 256, 4, 5), dtype="float32") = R.nn.relu(lv208)
            lv210: R.Tensor((1, 6, 4, 5), dtype="float32") = R.nn.conv2d(lv209, metadata["relax.expr.Constant"][155], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv211: R.Tensor((1, 6, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][156], R.shape([1, 6, 1, 1]))
            lv212: R.Tensor((1, 6, 4, 5), dtype="float32") = R.add(lv210, lv211)
            lv213: R.Tensor((1, 4, 5, 6), dtype="float32") = R.permute_dims(lv212, axes=[0, 2, 3, 1])
            lv214: R.Tensor((1, 12, 4, 5), dtype="float32") = R.nn.conv2d(lv209, metadata["relax.expr.Constant"][157], strides=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype="void")
            lv215: R.Tensor((1, 12, 1, 1), dtype="float32") = R.reshape(metadata["relax.expr.Constant"][158], R.shape([1, 12, 1, 1]))
            lv216: R.Tensor((1, 12, 4, 5), dtype="float32") = R.add(lv214, lv215)
            lv217: R.Tensor((1, 4, 5, 12), dtype="float32") = R.permute_dims(lv216, axes=[0, 2, 3, 1])
            lv218: R.Tensor((1, 3600, 2), dtype="float32") = R.reshape(lv97, R.shape([1, 3600, 2]))
            lv219: R.Tensor((1, 600, 2), dtype="float32") = R.reshape(lv149, R.shape([1, 600, 2]))
            lv220: R.Tensor((1, 160, 2), dtype="float32") = R.reshape(lv189, R.shape([1, 160, 2]))
            lv221: R.Tensor((1, 60, 2), dtype="float32") = R.reshape(lv213, R.shape([1, 60, 2]))
            lv222: R.Tensor((1, 3600, 4), dtype="float32") = R.reshape(lv105, R.shape([1, 3600, 4]))
            lv223: R.Tensor((1, 600, 4), dtype="float32") = R.reshape(lv157, R.shape([1, 600, 4]))
            lv224: R.Tensor((1, 160, 4), dtype="float32") = R.reshape(lv197, R.shape([1, 160, 4]))
            lv225: R.Tensor((1, 60, 4), dtype="float32") = R.reshape(lv217, R.shape([1, 60, 4]))
            lv226: R.Tensor((1, 4420, 2), dtype="float32") = R.concat((lv218, lv219, lv220, lv221), axis=1)
            lv227: R.Tensor((1, 4420, 2), dtype="float32") = R.nn.softmax(lv226, axis=2)
            lv228: R.Tensor((1, 4420, 4), dtype="float32") = R.concat((lv222, lv223, lv224, lv225), axis=1)
            gv: R.Tuple(R.Tensor((1, 4420, 2), dtype="float32"), R.Tensor((1, 4420, 4), dtype="float32")) = lv227, lv228
            R.output(gv)
        return gv

# Metadata omitted. Use show_meta=True in script() method to show it.