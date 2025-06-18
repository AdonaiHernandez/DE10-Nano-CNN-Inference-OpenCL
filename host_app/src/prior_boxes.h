#pragma once

#include <vector>

// Structure to hold a prior box: center_x, center_y, width, height
struct PriorBox {
    float cx;
    float cy;
    float w;
    float h;
};

// Function to generate the prior boxes for the Ultra-Light-Fast-Generic-Face-Detector-1MB
// These are specific to the model's architecture (input size 320x240 and feature map sizes)
// This function needs to be precisely implemented based on the model's original Python code or paper.
// For this model, the prior boxes are generated based on feature map sizes and aspect ratios.
std::vector<PriorBox> generatePriorBoxes(int input_width, int input_height);
