__kernel void generic_convolution_2d_full(
    __global const float* input,          // [N, H_in, W_in, C_in] (o [N, C_in, H_in, W_in])
    __global const float* weights,        // [C_out, C_in, K_h, K_w]
    __global float* output,               // [N, H_out, W_out, C_out] (o [N, C_out, H_out, W_out])
    const int input_height,
    const int input_width,
    const int input_channels,             // Nuevo
    const int output_channels,            // Nuevo
    const int kernel_height,
    const int kernel_width,
    const int stride_height,
    const int stride_width,
    const int pad_height,
    const int pad_width
) {
    // out_x, out_y, out_channel_idx, batch_idx
    const int out_x = get_global_id(0);
    const int out_y = get_global_id(1);
    const int out_c = get_global_id(2); // Nuevo: índice del canal de salida
    // Si procesas batch, necesitarías get_global_id(3) para el batch_idx

    float sum = 0.0f;

    // Bucle sobre los canales de entrada
    for (int ic = 0; ic < input_channels; ++ic) {
        // Bucle sobre las dimensiones del filtro
        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                int in_y = out_y * stride_height + ky - pad_height;
                int in_x = out_x * stride_width + kx - pad_width;

                if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                    // Acceso a input: [H_in, W_in, C_in] (para un solo elemento del batch)
                    int input_idx = (in_y * input_width + in_x) * input_channels + ic;
                    // Acceso a weights: [C_out, C_in, K_h, K_w]
                    int weight_idx = (((out_c * input_channels + ic) * kernel_height + ky) * kernel_width + kx);

                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    // Acceso a output: [H_out, W_out, C_out]
    int output_idx = (out_y * get_global_size(0) + out_x) * output_channels + out_c;
    output[output_idx] = sum;
}