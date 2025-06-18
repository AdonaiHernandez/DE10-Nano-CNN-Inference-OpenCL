#include "prior_boxes.h"
#include <vector>
#include <cmath>
#include <algorithm> // For std::min, std::max

// Helper function to calculate intersection over union (IoU)
// Not directly used for prior box generation, but useful for context/NMS later
float calculate_iou(const PriorBox& box1, const PriorBox& box2) {
    float x1_1 = box1.cx - box1.w / 2.0f;
    float y1_1 = box1.cy - box1.h / 2.0f;
    float x2_1 = box1.cx + box1.w / 2.0f;
    float y2_1 = box1.cy + box1.h / 2.0f;

    float x1_2 = box2.cx - box2.w / 2.0f;
    float y1_2 = box2.cy - box2.h / 2.0f;
    float x2_2 = box2.cx + box2.w / 2.0f;
    float y2_2 = box2.cy + box2.h / 2.0f;

    float intersect_x1 = std::max(x1_1, x1_2);
    float intersect_y1 = std::max(y1_1, y1_2);
    float intersect_x2 = std::min(x2_1, x2_2);
    float intersect_y2 = std::min(y2_1, y2_2);

    float intersect_w = std::max(0.0f, intersect_x2 - intersect_x1);
    float intersect_h = std::max(0.0f, intersect_y2 - intersect_y1);

    float intersect_area = intersect_w * intersect_h;
    float area1 = box1.w * box1.h;
    float area2 = box2.w * box2.h;

    return intersect_area / (area1 + area2 - intersect_area + 1e-6f); // Add epsilon to avoid division by zero
}


std::vector<PriorBox> generatePriorBoxes(int input_width, int input_height) {
    std::vector<PriorBox> prior_boxes;

    // These values are specific to the Ultra-Light-Fast-Generic-Face-Detector-1MB model (320x240 input)
    // They define the feature map sizes, min_sizes, and aspect_ratios for each detection layer.
    
    // Feature map sizes (h, w)
    std::vector<std::vector<int>> feature_maps = {
        {30, 40}, // From layer 1 (8x8 stride)
        {15, 20}, // From layer 2 (16x16 stride)
        {8, 10},  // From layer 3 (32x32 stride)
        {4, 5}    // From layer 4 (64x64 stride)
    };

    // Minimum box sizes for each feature map
    std::vector<std::vector<int>> min_sizes = {
        {10, 16, 24}, // Layer 1
        {32, 48},     // Layer 2
        {64, 96},     // Layer 3
        {128, 192, 256} // Layer 4
    };

    // Strides for each feature map
    std::vector<int> steps = {8, 16, 32, 64}; // Corresponding strides (e.g., 8px per cell)

    // Variances used in the decoding process (fixed for this model)
    const float variance_center = 0.1f; // For cx, cy
    const float variance_size = 0.2f;   // For w, h

    for (size_t i = 0; i < feature_maps.size(); ++i) {
        int fm_h = feature_maps[i][0];
        int fm_w = feature_maps[i][1];
        int step = steps[i];

        for (int h = 0; h < fm_h; ++h) {
            for (int w = 0; w < fm_w; ++w) {
                for (int min_size : min_sizes[i]) {
                    // Calculate center (cx, cy) of the prior box relative to the input image
                    // Normalize by input_width/height
                    float cx = (w + 0.5f) * step / input_width;
                    float cy = (h + 0.5f) * step / input_height;

                    // Calculate width and height of the prior box
                    // Normalize by input_width/height
                    float box_w = (float)min_size / input_width;
                    float box_h = (float)min_size / input_height; // Assuming aspect ratio 1:1 for initial min_size

                    // For this model, there's only one aspect ratio (1:1) per min_size.
                    // If the model supported multiple aspect ratios, you'd loop through them here.
                    
                    prior_boxes.push_back({cx, cy, box_w, box_h});
                }
            }
        }
    }
    return prior_boxes;
}