#pragma once

#include <onnxruntime_cxx_api.h>  // API C++ de ONNX Runtime

struct MyCustomKernel {
  MyCustomKernel(const OrtApi& ort_api, const OrtKernelInfo* /*info*/);

  void Compute(OrtKernelContext* context);

 private:
  const OrtApi& ort_;
};

struct MyCustomOp : Ort::CustomOpBase<MyCustomOp, MyCustomKernel> {
  explicit MyCustomOp(const char* provider) : provider_(provider) {}
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { return new MyCustomKernel(api, info); };
  const char* GetName() const { return "MyCustomConv"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    // Both the inputs need to be necessarily of float type
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

 private:
  const char* provider_{"CPUExecutionProvider"};
};