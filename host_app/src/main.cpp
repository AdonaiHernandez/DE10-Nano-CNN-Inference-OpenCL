#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // For std::min and std::max
#include <cmath>     // For std::exp, std::log
#include <chrono>

// OpenCV Headers
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/dnn.hpp> // For NMSBoxes

// ONNX Runtime Headers
#define ORT_API_VERSION 14 
#include <onnxruntime_c_api.h>

// Include the new prior_boxes header
#include "prior_boxes.h"

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

int main() {
    // 1. Initialize ONNX Runtime environment
    auto start_time_0 = std::chrono::steady_clock::now();
    OrtEnv* env;
    g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "onnx_inference_face_detection", &env);

    // 2. Session options
    OrtSessionOptions* session_options;
    g_ort->CreateSessionOptions(&session_options);
    g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED); 

    // 3. Create ONNX session
    const char* model_path = "modelo.onnx"; 
    OrtSession* session;
    if (g_ort->CreateSession(env, model_path, session_options, &session) != nullptr) {
        std::cerr << "ERROR: Failed to load model: " << model_path << std::endl;
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return -1;
    }
    std::cout << "Model '" << model_path << "' loaded successfully." << std::endl;

    // 4. Get default allocator
    OrtAllocator* allocator;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);

    // 5. Get input tensor name
    char* input_name_ptr;
    g_ort->SessionGetInputName(session, 0, allocator, &input_name_ptr);
    std::cout << "Input name: " << input_name_ptr << std::endl;

    // --- 6. Load and Preprocess the image with OpenCV ---
    const char* image_path = "img.jpg"; // <--- REMEMBER TO PLACE YOUR TEST IMAGE HERE!
    cv::Mat img = cv::imread(image_path); 
    auto start_time_1 = std::chrono::steady_clock::now();
    if (img.empty()) {
        std::cerr << "ERROR: Could not load image from " << image_path << ". Make sure it exists and is valid." << std::endl;
        g_ort->AllocatorFree(allocator, input_name_ptr);
        g_ort->ReleaseSession(session);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return -1;
    }
    int orig_w = img.cols;
    int orig_h = img.rows;
    std::cout << "Image '" << image_path << "' loaded successfully. Original dimensions: " 
              << orig_w << "x" << orig_h << std::endl;

    cv::Mat processed_image;
    double minVal, maxVal; 

    const int model_input_height = 240; 
    const int model_input_width = 320;  

    // --- PREPROCESSING STEPS (MATCHING PYTHON SCRIPT EXACTLY) ---
    cv::resize(img, processed_image, cv::Size(model_input_width, model_input_height));
    cv::minMaxLoc(processed_image, &minVal, &maxVal);
    //std::cout << "1. After Resize to " << model_input_width << "x" << model_input_height 
    //          << " (uchar, 0-255): Min=" << minVal << ", Max=" << maxVal << std::endl;

    cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2RGB);
    cv::minMaxLoc(processed_image, &minVal, &maxVal);
    //std::cout << "2. After BGR2RGB (uchar, 0-255): Min=" << minVal << ", Max=" << maxVal << std::endl;

    processed_image.convertTo(processed_image, CV_32F);
    cv::minMaxLoc(processed_image, &minVal, &maxVal);
    //std::cout << "3. After ConvertTo CV_32F (float, 0.0-255.0): Min=" << minVal << ", Max=" << maxVal << std::endl;

    processed_image = (processed_image - 127.0f) / 128.0f; 
    cv::minMaxLoc(processed_image, &minVal, &maxVal);
    //std::cout << "4. After Normalization (float, -1.0-1.0): Min=" << minVal << ", Max=" << maxVal << std::endl;

    std::vector<int64_t> input_dims_fixed = {1, 3, model_input_height, model_input_width};
    size_t input_tensor_size = input_dims_fixed[0] * input_dims_fixed[1] * input_dims_fixed[2] * input_dims_fixed[3];
    std::vector<float> input_tensor_values(input_tensor_size);

    for (int c = 0; c < processed_image.channels(); ++c) {
        for (int h = 0; h < processed_image.rows; ++h) {
            for (int w = 0; w < processed_image.cols; ++w) {
                input_tensor_values[c * processed_image.rows * processed_image.cols + h * processed_image.cols + w] =
                    processed_image.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    //std::cout << "5. Preprocessed image copied to input_tensor_values. Input tensor dimensions: [";
    /*for (size_t i = 0; i < input_dims_fixed.size(); ++i) {
        std::cout << input_dims_fixed[i] << (i + 1 == input_dims_fixed.size() ? "" : ", ");
    }
    std::cout << "]" << std::endl;*/

    auto end_time_1 = std::chrono::steady_clock::now();
    // 7. Create input tensor for ONNX Runtime
    OrtMemoryInfo* memory_info;
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    OrtValue* input_tensor = nullptr;
    g_ort->CreateTensorWithDataAsOrtValue(memory_info,
        input_tensor_values.data(),
        input_tensor_size * sizeof(float),
        input_dims_fixed.data(),
        input_dims_fixed.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);

    if (!input_tensor) {
        std::cerr << "ERROR: Failed to create input tensor." << std::endl;
        // ... (resource release) ...
        return -1;
    }

    // 8. Prepare output names
    size_t num_outputs = 0;
    g_ort->SessionGetOutputCount(session, &num_outputs);
    //std::cout << "\nNumber of outputs from the model: " << num_outputs << std::endl;

    std::vector<const char*> output_names_ptrs(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        char* output_name = nullptr;
        g_ort->SessionGetOutputName(session, i, allocator, &output_name);
        output_names_ptrs[i] = output_name;
        //std::cout << "Output " << i << " name: " << output_name << std::endl;
    }

    // 9. Run inference
    std::vector<OrtValue*> output_tensors(num_outputs, nullptr);
    auto start_time_2 = std::chrono::steady_clock::now();
    g_ort->Run(session,
        nullptr, 
        (const char* const*)&input_name_ptr, &input_tensor, 1, 
        output_names_ptrs.data(), num_outputs, 
        output_tensors.data());
    std::cout << "\nInference completed." << std::endl;
    auto end_time_2 = std::chrono::steady_clock::now();    
    // 10. Process Raw Output Tensors
    if (num_outputs < 2) { // Expecting at least 2 outputs: scores and boxes
        std::cerr << "ERROR: Expected at least 2 output tensors (scores and boxes)." << std::endl;
        // ... (resource release) ...
        return -1;
    }

    // --- Get Scores Data (Tensor 0) ---
    OrtTensorTypeAndShapeInfo* scores_info;
    g_ort->GetTensorTypeAndShape(output_tensors[0], &scores_info);
    float* scores_data;
    g_ort->GetTensorMutableData(output_tensors[0], (void**)&scores_data);
    
    size_t scores_dim_count;
    g_ort->GetDimensionsCount(scores_info, &scores_dim_count);
    std::vector<int64_t> scores_dims(scores_dim_count);
    g_ort->GetDimensions(scores_info, scores_dims.data(), scores_dim_count);
    size_t num_prior_boxes = scores_dims[1]; // Should be 4420
    size_t num_classes = scores_dims[2];    // Should be 2 (background, face)
    std::cout << "\nScores Tensor Dimensions: [";
    for(int64_t dim : scores_dims) std::cout << dim << ",";
    std::cout << "]. Number of prior boxes: " << num_prior_boxes << ", Classes per box: " << num_classes << std::endl;
    g_ort->ReleaseTensorTypeAndShapeInfo(scores_info);

    // --- Get Boxes Data (Tensor 1) ---
    OrtTensorTypeAndShapeInfo* boxes_info;
    g_ort->GetTensorTypeAndShape(output_tensors[1], &boxes_info);
    float* boxes_data;
    g_ort->GetTensorMutableData(output_tensors[1], (void**)&boxes_data);
    
    size_t boxes_dim_count;
    g_ort->GetDimensionsCount(boxes_info, &boxes_dim_count);
    std::vector<int64_t> boxes_dims(boxes_dim_count);
    g_ort->GetDimensions(boxes_info, boxes_dims.data(), boxes_dim_count);
    std::cout << "Boxes Tensor Dimensions: [";
    for(int64_t dim : boxes_dims) std::cout << dim << ",";
    std::cout << "]. Expected number of prior boxes: " << num_prior_boxes << ", Coords per box: 4" << std::endl;
    g_ort->ReleaseTensorTypeAndShapeInfo(boxes_info);

    // --- GENERATE PRIOR BOXES ---
    // These are fixed for the model's architecture
    std::cout << "\nGenerating prior boxes..." << std::endl;
    std::vector<PriorBox> prior_boxes = generatePriorBoxes(model_input_width, model_input_height);
    if (prior_boxes.size() != num_prior_boxes) {
        std::cerr << "WARNING: Mismatch between generated prior boxes (" << prior_boxes.size() 
                  << ") and model output prior boxes (" << num_prior_boxes << ")!" << std::endl;
    } else {
        std::cout << "Generated " << prior_boxes.size() << " prior boxes successfully." << std::endl;
    }

    // --- DECODING CONSTANTS (specific to this model) ---
    // These are often called 'variances' in SSD models
    const float center_variance = 0.1f; 
    const float size_variance = 0.2f;   

    float score_threshold = 0.5f; // Threshold for face probability (class 1)
    float nms_threshold = 0.4f;   // NMS threshold

    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> indices; 

    //std::cout << "\nDecoding and filtering detections..." << std::endl;
    for (size_t i = 0; i < num_prior_boxes; ++i) {
        // Score for 'face' class (class 1)
        float score = scores_data[i * num_classes + 1]; 
        
        if (score > score_threshold) {
            // Raw deltas from model output
            float dx = boxes_data[i * 4 + 0];
            float dy = boxes_data[i * 4 + 1];
            float dw = boxes_data[i * 4 + 2];
            float dh = boxes_data[i * 4 + 3];

            // Get corresponding prior box
            const PriorBox& prior = prior_boxes[i];

            // Decode to center_x, center_y, width, height (normalized to 0-1 range)
            float decoded_cx = dx * center_variance * prior.w + prior.cx;
            float decoded_cy = dy * center_variance * prior.h + prior.cy;
            float decoded_w = std::exp(dw * size_variance) * prior.w;
            float decoded_h = std::exp(dh * size_variance) * prior.h;

            // Convert (cx, cy, w, h) to (x1, y1, x2, y2) (normalized 0-1 range)
            float x1_norm = decoded_cx - decoded_w / 2.0f;
            float y1_norm = decoded_cy - decoded_h / 2.0f;
            float x2_norm = decoded_cx + decoded_w / 2.0f;
            float y2_norm = decoded_cy + decoded_h / 2.0f;

            // Scale to original image dimensions
            int x1 = static_cast<int>(x1_norm * orig_w);
            int y1 = static_cast<int>(y1_norm * orig_h);
            int x2 = static_cast<int>(x2_norm * orig_w);
            int y2 = static_cast<int>(y2_norm * orig_h);

            // Clamp coordinates to image boundaries (essential!)
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(orig_w - 1, x2); 
            y2 = std::min(orig_h - 1, y2);

            int width = x2 - x1;
            int height = y2 - y1;

            if (width <= 0 || height <= 0) {
                // std::cout << "  Skipping invalid box (width=" << width << ", height=" << height << ") after adjustment." << std::endl;
                continue; 
            }

            bboxes.push_back(cv::Rect(x1, y1, width, height));
            confidences.push_back(score);
        }
    }

    // Apply Non-Maximum Suppression (NMS)
   // cv::dnn::NMSBoxes(bboxes, confidences, score_threshold, nms_threshold, indices); 
    // If you uncommented NMS, this 'indices' vector contains the final filtered box indices.

    // --- Draw Bounding Boxes on the original image (`img`) ---
    //std::cout << "\nFaces detected after NMS and decoding (" << indices.size() << "):" << std::endl;
    for (size_t idx = 0; idx < bboxes.size(); ++idx) { // Iterate using indices from NMS
        cv::Rect box = bboxes[idx];
        float score = confidences[idx];

        //std::cout << "  Final Box: (" << box.x << ", " << box.y << ", " << box.width << ", " << box.height << ") Score: " << score << std::endl;

        cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2); // Green
        std::string label = cv::format("%.2f", score); 
        cv::Point text_origin(box.x, box.y > 10 ? box.y - 5 : box.y + 15);
        cv::putText(img, label, text_origin,
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }

    // 11. Save the modified image
    std::string output_image_path = "imagen_resultado.jpeg";
    cv::imwrite(output_image_path, img); 
    std::cout << "\nProcessed image saved to: " << output_image_path << std::endl;

    // 12. Release ONNX Runtime memory
    g_ort->ReleaseValue(input_tensor);
    for (size_t i = 0; i < num_outputs; ++i) {
        g_ort->ReleaseValue(output_tensors[i]);
        g_ort->AllocatorFree(allocator, (void*)output_names_ptrs[i]);
    }
    g_ort->AllocatorFree(allocator, (void*)input_name_ptr); 
    
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    std::cout << "Face detection inference completed." << std::endl;
    auto end_time_0 = std::chrono::steady_clock::now();

    auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_1 - start_time_1);
    auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_2 - start_time_2);
    auto duration_0 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_0 - start_time_0);

     std::cout << "Tiempo de Preprocesamiento: " << duration_1.count() << " ms" << std::endl;
     std::cout << "Tiempo de Inferencia: " << duration_2.count() << " ms" << std::endl;
     std::cout << "Tiempo de Total: " << duration_0.count() << " ms" << std::endl;

    return 0;
}