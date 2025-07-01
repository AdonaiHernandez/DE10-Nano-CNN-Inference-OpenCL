#include "onnxruntime_cxx_api.h"
#include <iostream>
#include "aocl_utils.h"
#include <CL/cl.h>
#include "MyCustomConvOpKernel.h"

using namespace aocl_utils;

cl_uint num_platforms;
cl_platform_id platform;
cl_uint num_devices;
cl_device_id device;
cl_context clContext;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;
cl_int ret;

std::string deviceInfo;

void cleanup()
{
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(clContext);
    
}

void initOpenCL(){
	cl_int status;

    platform = findPlatform("Intel(R) FPGA");
    if (platform == NULL) {
        cleanup();
        std::cerr << "Error: could not find Intel FPGA platform." << std::endl;
        return;
    }
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    checkError (status, "Error: could not query devices");
    num_devices = 1; // always only using one device
    char info[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(info), info, NULL);
    deviceInfo = info;

    clContext = clCreateContext(0, num_devices, &device, NULL, NULL, &status);
    checkError(status, "Error: could not create OpenCL context");

    queue = clCreateCommandQueue(clContext, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Error: could not create command queue");
    std::string binary_file = getBoardBinaryFile("conv", device);
    std::cout << "Using AOCX: " << binary_file << "\n";
    program = createProgramFromBinary(clContext, binary_file.c_str(), &device, 1);

    status = clBuildProgram(program, num_devices, &device, "", NULL, NULL);
    checkError(status, "Error: could not build program");
	
	kernel = clCreateKernel(program, "conv", &status);
    checkError(status, "Error: could not create conv kernel"); 
	
}

MyCustomKernel::MyCustomKernel(const OrtApi& ort_api, const OrtKernelInfo* /*info*/)
      : ort_(ort_api){
		  initOpenCL();
	  }

void MyCustomKernel::Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);

    // Obtener tensores de entrada como Ort::Value
    const Ort::ConstValue  input_tensor = ctx.GetInput(0);
    const Ort::ConstValue  weight_tensor = ctx.GetInput(1);

    // Obtener punteros a datos float (const)
    const float* input_data = input_tensor.GetTensorData<float>();
    const float* weight_data = weight_tensor.GetTensorData<float>();

    // El bias puede ser opcional, así que verifica dimensiones
    const float* bias_data = nullptr;

    // Obtener dimensiones de entrada y pesos
    Ort::TensorTypeAndShapeInfo input_info = input_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = input_info.GetShape();

    Ort::TensorTypeAndShapeInfo weight_info = weight_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> weight_dims = weight_info.GetShape();

    // Asumiendo formato NCHW para input: [batch, channels_in, height, width]
    int batch = (int)input_dims[0];
    int in_channels = (int)input_dims[1];
    int in_height = (int)input_dims[2];
    int in_width = (int)input_dims[3];

    int out_channels = (int)weight_dims[0];
    int kernel_h = (int)weight_dims[2];
    int kernel_w = (int)weight_dims[3];

    int stride_h = 2;
    int stride_w = 2;

    int pad_h = 1;
    int pad_w = 1;

    int out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    // Crear tensor de salida
    std::vector<int64_t> output_dims = {batch, out_channels, out_height, out_width};
	
    Ort::UnownedValue output_tensor = ctx.GetOutput(0, output_dims);

    float* output_data = output_tensor.GetTensorMutableData<float>();

    // --- Aquí va tu código OpenCL ---

    size_t input_size_bytes = in_channels * in_height * in_width * sizeof(float);
    size_t output_size_bytes = out_channels * out_height * out_width * sizeof(float);
    size_t weight_size_bytes = weight_dims[0] * weight_dims[1] * weight_dims[2] * weight_dims[3] * sizeof(float);
    size_t bias_size_bytes = bias_data ? out_channels * sizeof(float) : 0;

    cl_int err;
    cl_mem input_buf = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_size_bytes, (void*)input_data, &err);
    checkError(err, "Error creando input_buf");

    cl_mem output_buf = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, output_size_bytes, nullptr, &err);
    checkError(err, "Error creando output_buf");

    cl_mem weight_buf = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weight_size_bytes, (void*)weight_data, &err);
    checkError(err, "Error creando weight_buf");

    cl_mem bias_buf = nullptr;
    if (bias_data != nullptr && bias_size_bytes > 0) {
        bias_buf = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bias_size_bytes, (void*)bias_data, &err);
        checkError(err, "Failed to create bias buffer");
    }

    // Argumentos
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
	checkError(err, "Set arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &weight_buf);
	checkError(err, "Set arg 1");

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buf);
	checkError(err, "Set arg 3");
    err = clSetKernelArg(kernel, 3, sizeof(int), &in_channels);
	checkError(err, "Set arg 5");
    err = clSetKernelArg(kernel, 4, sizeof(int), &in_height);
	checkError(err, "Set arg 6");
    err = clSetKernelArg(kernel, 5, sizeof(int), &in_width);
	checkError(err, "Set arg 7");
    err = clSetKernelArg(kernel, 6, sizeof(int), &out_channels);
	checkError(err, "Set arg 8");
    err = clSetKernelArg(kernel, 7, sizeof(int), &out_height);
	checkError(err, "Set arg 9");
    err = clSetKernelArg(kernel, 8, sizeof(int), &out_width);
	checkError(err, "Set arg 10");
    err = clSetKernelArg(kernel, 9, sizeof(int), &stride_h);
	checkError(err, "Set arg 11");
    err = clSetKernelArg(kernel, 10, sizeof(int), &stride_w);
	checkError(err, "Set arg 12");
    err = clSetKernelArg(kernel, 11, sizeof(int), &pad_h);
	checkError(err, "Set arg 13");
    err = clSetKernelArg(kernel, 12, sizeof(int), &pad_w);
	checkError(err, "Set arg 14");
	int groups = 1;
    err = clSetKernelArg(kernel, 13, sizeof(int), &groups);
	checkError(err, "Set arg 15");

    size_t global_work_size[3] = {out_channels, out_height, out_width};

    err = clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    checkError(err, "Failed to execute kernel");
    clFinish(queue);

    err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, output_size_bytes, output_data, 0, nullptr, nullptr);
    checkError(err, "Failed read output from kernel");

    clReleaseMemObject(input_buf);
    clReleaseMemObject(weight_buf);
    clReleaseMemObject(output_buf);
 }
