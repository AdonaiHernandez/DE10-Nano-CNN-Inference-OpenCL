#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
#include "MyCustomConvOpKernel.h"

static MyCustomOp custom_op{"CPUExecutionProvider"}; 

struct Box {
    float xmin, ymin, xmax, ymax;
};

std::vector<std::array<float,4>> generate_priors(int input_width=320, int input_height=240) {
    std::vector<std::array<float,4>> prior_boxes;
    std::vector<std::pair<int,int>> feature_maps = {{30,40},{15,20},{8,10},{4,5}};
    std::vector<std::vector<int>> min_sizes = {
        {10,16,24}, {32,48}, {64,96}, {128,192,256}
    };
    std::vector<int> steps = {8,16,32,64};

    for (size_t k = 0; k < feature_maps.size(); ++k) {
        int fm_h = feature_maps[k].first;
        int fm_w = feature_maps[k].second;
        int step = steps[k];

        for (int i = 0; i < fm_h; ++i) {
            for (int j = 0; j < fm_w; ++j) {
                for (int min_size : min_sizes[k]) {
                    float cx = (j + 0.5f) * step / input_width;
                    float cy = (i + 0.5f) * step / input_height;
                    float box_w = (float)min_size / input_width;
                    float box_h = (float)min_size / input_height;
                    prior_boxes.push_back({cx, cy, box_w, box_h});
                }
            }
        }
    }
    return prior_boxes;
}

std::vector<Box> decode_boxes(const std::vector<std::array<float,4>>& loc, 
                              const std::vector<std::array<float,4>>& priors, 
                              float variance_center=0.1f, float variance_size=0.2f) {
    std::vector<Box> boxes(loc.size());
    for (size_t i = 0; i < loc.size(); ++i) {
        float cx = priors[i][0] + loc[i][0] * variance_center * priors[i][2];
        float cy = priors[i][1] + loc[i][1] * variance_center * priors[i][3];
        float w = priors[i][2] * std::exp(loc[i][2] * variance_size);
        float h = priors[i][3] * std::exp(loc[i][3] * variance_size);

        boxes[i].xmin = cx - w / 2.0f;
        boxes[i].ymin = cy - h / 2.0f;
        boxes[i].xmax = cx + w / 2.0f;
        boxes[i].ymax = cy + h / 2.0f;
    }
    return boxes;
}

std::vector<int> non_max_suppression(const std::vector<Box>& boxes, const std::vector<float>& scores,
                                     float iou_threshold = 0.3f, float score_threshold = 0.6f) {
    std::vector<int> indices;
    std::vector<int> keep;

    // Filtrar boxes con score < threshold
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] >= score_threshold) indices.push_back((int)i);
    }

    // Ordenar indices por score descendente
    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });

    while (!indices.empty()) {
        int current = indices[0];
        keep.push_back(current);
        std::vector<int> rest;

        for (size_t i = 1; i < indices.size(); ++i) {
            int idx = indices[i];
            // Calcular IoU
            float xx1 = std::max(boxes[current].xmin, boxes[idx].xmin);
            float yy1 = std::max(boxes[current].ymin, boxes[idx].ymin);
            float xx2 = std::min(boxes[current].xmax, boxes[idx].xmax);
            float yy2 = std::min(boxes[current].ymax, boxes[idx].ymax);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float inter = w * h;

            float area1 = (boxes[current].xmax - boxes[current].xmin) * (boxes[current].ymax - boxes[current].ymin);
            float area2 = (boxes[idx].xmax - boxes[idx].xmin) * (boxes[idx].ymax - boxes[idx].ymin);
            float ovr = inter / (area1 + area2 - inter);

            if (ovr <= iou_threshold)
                rest.push_back(idx);
        }
        indices = rest;
    }
    return keep;
}

std::vector<float> preprocess(const cv::Mat& img, int input_width=320, int input_height=240) {
    cv::Mat resized, rgb_float;
    cv::resize(img, resized, cv::Size(input_width, input_height));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(rgb_float, CV_32FC3, 1.0f/255.0f);
    std::vector<float> chw(input_width * input_height * 3);
    int channel_size = input_width * input_height;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < input_height; ++i) {
            for (int j = 0; j < input_width; ++j) {
                chw[c * channel_size + i * input_width + j] = rgb_float.at<cv::Vec3f>(i,j)[c];
            }
        }
    }
    return chw;
}

int main() {
    const int input_width = 320;
    const int input_height = 240;


    Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "UltraLightFace");
    Ort::SessionOptions session_options;
	
	std::cout << "Iniciamos onnx" << std::endl;
	

    // Crear dominio personalizado y registrar el operador
    Ort::CustomOpDomain custom_domain("com.yourcompany.fpga");
	std::cout << "Se mete el customop" << std::endl;
    custom_domain.Add(&custom_op);  // aquí se añade la instancia
	std::cout << "se mete en las opciones" << std::endl;
    session_options.Add(custom_domain);
	
    session_options.SetIntraOpNumThreads(1);
	std::cout << "cargamos modelo" << std::endl;
    Ort::Session session(*env, "modelo.onnx", session_options);
	
	std::cout << "Creada session" << std::endl;

    const char* input_name = "input";
    const char* output_names[] = {"scores", "boxes"};

    std::vector<std::array<float,4>> priors = generate_priors(input_width, input_height);
	
	std::cout << "Generadas prior box" << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir la cámara\n";
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
	std::cout << "Iniciamos capturas" << std::endl;
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        frame_count++;
        auto chw = preprocess(frame);
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            chw.data(),
            chw.size(),
            std::vector<int64_t>{1, 3, input_height, input_width}.data(),
            4
        );
		
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                          &input_name, &input_tensor, 1, 
                                          output_names, 2);                                

        // Obtener punteros y tamaños de salida
        float* conf_ptr = output_tensors[0].GetTensorMutableData<float>();
        float* loc_ptr = output_tensors[1].GetTensorMutableData<float>();

        size_t num_boxes = priors.size();

        // Leer conf y loc a vectores C++
        std::vector<std::array<float,2>> conf(num_boxes);
        std::vector<std::array<float,4>> loc(num_boxes);
        for (size_t i = 0; i < num_boxes; ++i) {
            conf[i][0] = conf_ptr[i*2 + 0];
            conf[i][1] = conf_ptr[i*2 + 1];
            loc[i][0] = loc_ptr[i*4 + 0];
            loc[i][1] = loc_ptr[i*4 + 1];
            loc[i][2] = loc_ptr[i*4 + 2];
            loc[i][3] = loc_ptr[i*4 + 3];
        }

        // Decodificar cajas
        std::vector<Box> boxes = decode_boxes(loc, priors);

        // Scores clase "cara" es índice 1
        std::vector<float> scores(num_boxes);
        for (size_t i = 0; i < num_boxes; ++i) {
            scores[i] = conf[i][1];
        }

        // NMS
        std::vector<int> keep = non_max_suppression(boxes, scores, 0.3f, 0.6f);

        // Dibujar cajas filtradas
        float scale_w = (float)frame.cols / input_width;
        float scale_h = (float)frame.rows / input_height;

        for (int idx : keep) {
            int xmin = (int)(boxes[idx].xmin * input_width * scale_w);
            int ymin = (int)(boxes[idx].ymin * input_height * scale_h);
            int xmax = (int)(boxes[idx].xmax * input_width * scale_w);
            int ymax = (int)(boxes[idx].ymax * input_height * scale_h);

            cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0,255,0), 2);
            cv::putText(frame, cv::format("%.2f", scores[idx]), cv::Point(xmin, ymin - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
        }

        if (frame_count % 30 == 0) { // cada 30 frames
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            double fps = frame_count / diff.count();
            std::cout << "FPS: " << fps << std::endl;

            // Reiniciar
            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        cv::imshow("Ultra Light Face Detector", frame);
        if (cv::waitKey(1) == 27) break;  // ESC para salir
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
