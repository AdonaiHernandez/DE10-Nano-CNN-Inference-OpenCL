#include <opencv2/opencv.hpp>
#include <iostream>

// ... (tus includes de OpenCL y definiciones si las necesitas)
// Para este ejemplo de mostrar la ventana, no son estrictamente necesarios,
// pero los mantendría si tu objetivo es integrarlo con tu pipeline FPGA.

int main() {
    cv::VideoCapture cap;
    if (!cap.open(0)) {
        std::cerr << "Error: No se pudo abrir la webcam." << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);

    cv::Mat frame_cpu;
    std::cout << "Mostrando video de la webcam. Presiona 'q' en la ventana para salir." << std::endl;

    // Bucle principal para capturar y mostrar frames
    while (true) {
        cap >> frame_cpu; // Captura un frame
        if (frame_cpu.empty()) {
            std::cerr << "Error: Frame vacío/perdido. Reintentando captura..." << std::endl;
            continue;
        }

        // --- Parte de mostrar el frame ---
        cv::imshow("Webcam Feed", frame_cpu); // Muestra el frame en una ventana llamada "Webcam Feed"

        // Espera una tecla por 1 milisegundo. Si se presiona 'q', sale.
        // cv::waitKey() también maneja los eventos de la GUI (dibujar la ventana).
        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 27) { // 27 es la tecla ESC
            std::cout << "Tecla 'q' o 'ESC' presionada. Saliendo." << std::endl;
            break;
        }

        // Opcional: Si no usas cv::waitKey(>0) y quieres un retardo
        // std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    cap.release(); // Libera la cámara
    cv::destroyAllWindows(); // Cierra todas las ventanas de OpenCV
    std::cout << "Aplicación terminada." << std::endl;
    return 0;
}