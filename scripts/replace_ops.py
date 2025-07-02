import onnx
from onnx import helper

def replace_first_n_conv3x3_with_custom_op(input_model_path, output_model_path, n=6,
                                           custom_op_name="MyCustomConv", custom_op_domain="com.yourcompany.fpga"):
    print(f"Cargando modelo ONNX desde: {input_model_path}")
    model = onnx.load(input_model_path)
    graph = model.graph

    new_nodes = []
    replaced_count = 0

    for node in graph.node:
        if node.op_type == "Conv" and replaced_count < n:
            # Verifica si el kernel es 3x3
            kernel_shape = None
            strides = [1, 1]
            pads = [0, 0, 0, 0]
            groups = 1

            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_shape = attr.ints
                elif attr.name == "strides":
                    strides = attr.ints
                elif attr.name == "pads":
                    pads = attr.ints
                elif attr.name == "group":
                    groups = attr.i

            if kernel_shape != [3, 3]:
                new_nodes.append(node)
                continue

            print(f"Reemplazando Conv 3x3: {node.name}")
            replaced_count += 1

            custom_conv_node = helper.make_node(
                op_type=custom_op_name,
                inputs=list(node.input),
                outputs=list(node.output),
                name=f"{node.name}_custom_{replaced_count}",
                domain=custom_op_domain,
                kernel_shape=kernel_shape,
                strides=strides,
                pads=[pads[0], pads[1]],  # Asumimos padding simétrico
                group=groups
            )

            new_nodes.append(custom_conv_node)
        else:
            new_nodes.append(node)

    if replaced_count == 0:
        print("No se encontraron capas Conv 3x3 para reemplazar.")

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=graph.input,
        outputs=graph.output,
        initializer=graph.initializer,
        value_info=graph.value_info
    )

    new_model = helper.make_model(new_graph, producer_name=model.producer_name, opset_imports=model.opset_import)
    new_model.ir_version = 8

    if not any(opset.domain == custom_op_domain for opset in new_model.opset_import):
        new_model.opset_import.append(helper.make_opsetid(custom_op_domain, 1))

    onnx.save(new_model, output_model_path)
    print(f"Modelo guardado en: {output_model_path}")

    try:
        onnx.checker.check_model(new_model)
        print("Modelo verificado correctamente.")
    except Exception as e:
        print(f"Error de verificación: {e}")

if __name__ == "__main__":
    replace_first_n_conv3x3_with_custom_op("modelo.onnx", "modelo_custom_op.onnx")
