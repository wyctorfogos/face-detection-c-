#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace dnn;

int main() {
    // Carregar o modelo pré-treinado
    std::string modelFile = "deploy.prototxt";  // Caminho para o arquivo .prototxt
    std::string weightsFile = "res10_300x300_ssd_iter_140000.caffemodel"; // Caminho para o arquivo .caffemodel

    Net net = readNetFromCaffe(modelFile, weightsFile);

    // Ler a imagem
    Mat image = imread("/app/src/data/images/59559b5c-ba40-4eae-9250-e2c3e6d7fc1c-960x605.jpg");
    if (image.empty()) {
        printf("Erro ao carregar a imagem!");
        return -1;
    }

    // Preprocessar a imagem
    Mat blob;
    blobFromImage(image, blob, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);
    net.setInput(blob);

    // Fazer a detecção de faces
    Mat detections = net.forward();

    // Processar as detecções
    for (int i = 0; i < detections.size[2]; i++) {
        float confidence = detections.at<float>(0, 0, i, 2);

        if (confidence > 0.05) {  // Apenas se a confiança for maior que 0.5
            int x1 = static_cast<int>(detections.at<float>(0, 0, i, 3) * image.cols);
            int y1 = static_cast<int>(detections.at<float>(0, 0, i, 4) * image.rows);
            int x2 = static_cast<int>(detections.at<float>(0, 0, i, 5) * image.cols);
            int y2 = static_cast<int>(detections.at<float>(0, 0, i, 6) * image.rows);

            // Desenhar a caixa ao redor da face detectada
            rectangle(image, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);
        }
    }

    // Salvar a imagem com as detecções
    imwrite("/app/src/data/output/dnn_face_detection_result.jpg", image);
    std::cout << "Detecção de faces completa! Resultado salvo em /app/src/data/output/dnn_face_detection_result.jpg" << std::endl;

    return 0;
}
