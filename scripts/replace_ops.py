import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto

def replace_first_conv_with_custom_op(input_model_path, output_model_path, custom_op_name="MyCustomConv", custom_op_domain="com.yourcompany.fpga"):
    """
    Carga un modelo ONNX, encuentra la primera operación Conv, y la reemplaza
    por una operación personalizada con el nombre y dominio especificados.

    Args:
        input_model_path (str): Ruta al modelo ONNX de entrada.
        output_model_path (str): Ruta donde se guardará el modelo ONNX modificado.
        custom_op_name (str): El nombre de tu operador personalizado (debe coincidir con GetOpName()).
        custom_op_domain (str): El dominio de tu operador personalizado.
    """
    print(f"Cargando modelo ONNX desde: {input_model_path}")
    model = onnx.load(input_model_path)
    graph = model.graph

    # Clonamos el grafo para evitar modificarlo mientras lo iteramos
    new_nodes = []
    found_first_conv = False

    for node in graph.node:
        if node.op_type == "Conv" and not found_first_conv:
            print(f"Encontrada la primera capa Conv: {node.name}")
            found_first_conv = True

            # Extraer atributos de la capa Conv original
            kernel_shape = None
            strides = [1, 1]  # Default ONNX Conv stride
            pads = [0, 0, 0, 0] # Default ONNX Conv pads (top, left, bottom, right)

            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_shape = attr.ints
                elif attr.name == "strides":
                    strides = attr.ints
                elif attr.name == "pads":
                    pads = attr.ints
                # Puedes añadir más atributos si tu Custom Op los necesita (ej. dilations, group)

            if kernel_shape is None:
                raise ValueError(f"La capa Conv '{node.name}' no tiene el atributo 'kernel_shape' necesario.")
            if len(pads) != 4:
                 raise ValueError(f"La capa Conv '{node.name}' pads debe tener 4 valores (top, left, bottom, right).")


            # Crear el nuevo nodo de operador personalizado
            # Las entradas y salidas son las mismas que la capa Conv original
            custom_conv_node = helper.make_node(
                op_type=custom_op_name,
                inputs=list(node.input),  # Las mismas entradas que la Conv original (X, W, B si existe)
                outputs=list(node.output), # Las mismas salidas que la Conv original
                name=f"{node.name}_custom", # Nuevo nombre para evitar conflictos
                domain=custom_op_domain,   # Tu dominio personalizado
                # Atributos de tu Custom Op
                kernel_shape=kernel_shape,
                strides=strides,
                pads=[pads[0], pads[1]] # Asumimos pads[0] y pads[1] son pad_h y pad_w respectivamente en tu kernel
                                         # Ojo: tu kernel OpenCL espera pad_h, pad_w. ONNX pads es [H_start, W_start, H_end, W_end].
                                         # Necesitas asegurar que tu Custom Op interprete esto correctamente.
                                         # Aquí estamos asumiendo padding simétrico y tomando solo los primeros dos.
                                         # Podrías necesitar un atributo como 'pad_h', 'pad_w' en tu Custom Op si ONNX Conv pads es complejo.
            )
            # Copiar otros atributos relevantes si existen en la Conv original y los usas en tu Custom Op
            # for attr in node.attribute:
            #     if attr.name not in ["kernel_shape", "strides", "pads"]:
            #         custom_conv_node.attribute.append(attr)

            new_nodes.append(custom_conv_node)
        else:
            # Si no es la primera Conv, o ya la encontramos, simplemente añade el nodo original
            new_nodes.append(node)

    if not found_first_conv:
        print("Advertencia: No se encontró ninguna capa Conv en el modelo.")
        # Podrías decidir salir o lanzar un error aquí si el modelo debe tener convoluciones.

    # Reconstruir el grafo con los nuevos nodos
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=graph.input,
        outputs=graph.output,
        initializer=graph.initializer,
        value_info=graph.value_info,
        # Mantén solo esta línea para el doc_string
        doc_string=str(graph.doc_string) if graph.doc_string else "",
    )

    # Reconstruir el modelo
    new_model = helper.make_model(new_graph, producer_name=model.producer_name, opset_imports=model.opset_import)
    new_model.ir_version = 8
    # Añadir o modificar la definición de opset_import para incluir tu dominio personalizado
    # Esto le dice a ONNX Runtime que hay un nuevo dominio de operador disponible
    found_custom_domain_opset = False
    for opset in new_model.opset_import:
        if opset.domain == custom_op_domain:
            opset.version = 1 # Puedes establecer la version de tu dominio si la tienes
            found_custom_domain_opset = True
            break
    if not found_custom_domain_opset:
        new_model.opset_import.append(helper.make_opsetid(custom_op_domain, 1)) # Añade tu dominio con una version (ej. 1)

    onnx.save(new_model, output_model_path)
    print(f"Modelo modificado guardado en: {output_model_path}")
    print("¡Recuerda que debes tener tu librería de Custom Ops cargada en ONNX Runtime para que esto funcione!")

if __name__ == "__main__":
    # Define la ruta a tu modelo ONNX original (ej. un modelo de PyTorch/TensorFlow exportado)
    # y el nombre del modelo de salida.
    input_onnx_model = "modelo.onnx"
    output_onnx_model = "modelo_custom_op.onnx"

    # --- EJEMPLO DE USO ---
    # Para probar esto, podrías exportar un modelo simple de PyTorch con una convolución:
    # import torch
    # import torch.nn as nn
    # class SimpleConvNet(nn.Module):
    #     def __init__(self):
    #         super(SimpleConvNet, self).__init__()
    #         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    #         self.relu = nn.ReLU()
    #         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    #     def forward(self, x):
    #         x = self.conv1(x)
    #         x = self.relu(x)
    #         x = self.pool(x)
    #         return x
    # model = SimpleConvNet()
    # dummy_input = torch.randn(1, 3, 32, 32)
    # torch.onnx.export(model, dummy_input, input_onnx_model, opset_version=11,
    #                   input_names=['input'], output_names=['output'])
    # print(f"Modelo de ejemplo '{input_onnx_model}' creado.")
    # ---------------------

    replace_first_conv_with_custom_op(input_onnx_model, output_onnx_model)

    # Opcional: Verificación del modelo modificado
    try:
        onnx.checker.check_model(output_onnx_model)
        print("El modelo modificado es válido.")
    except Exception as e:
        print(f"Error en la verificación del modelo modificado: {e}")