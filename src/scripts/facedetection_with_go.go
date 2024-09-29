package main

import (
    "fmt"
    "gocv.io/x/gocv"
    "gocv.io/x/gocv/dnn"
)

func main() {
    // Carregar o modelo MTCNN em formato ONNX ou TensorFlow (.pb)
    net := gocv.ReadNet("mtcnn.pb", "")
    if net.Empty() {
        fmt.Println("Erro ao carregar o modelo MTCNN!")
        return
    }

    // Abrir a câmera
    webcam, err := gocv.VideoCaptureDevice(0)
    if err != nil {
        fmt.Printf("Erro ao abrir a câmera: %v\n", err)
        return
    }
    defer webcam.Close()

    // Criar uma janela para exibir o vídeo
    window := gocv.NewWindow("Detecção de Faces com MTCNN")
    defer window.Close()

    // Capturar frames da câmera
    img := gocv.NewMat()
    defer img.Close()

    for {
        if ok := webcam.Read(&img); !ok || img.Empty() {
            fmt.Println("Erro ao capturar frame!")
            break
        }

        // Preprocessar a imagem (criar um blob)
        blob := gocv.BlobFromImage(img, 1.0, gocv.NewSize(300, 300), gocv.NewScalar(0, 0, 0, 0), true, false)
        net.SetInput(blob, "")

        // Fazer a inferência (detectar faces)
        detections := net.Forward("")

        // Processar os resultados
        rows := detections.Rows()
        cols := detections.Cols()
        for i := 0; i < rows; i++ {
            for j := 0; j < cols; j++ {
                confidence := detections.GetFloatAt(i, j+2) // Confiança da detecção
                if confidence > 0.5 {                      // Limite de confiança
                    x1 := int(detections.GetFloatAt(i, j+3) * float32(img.Cols()))
                    y1 := int(detections.GetFloatAt(i, j+4) * float32(img.Rows()))
                    x2 := int(detections.GetFloatAt(i, j+5) * float32(img.Cols()))
                    y2 := int(detections.GetFloatAt(i, j+6) * float32(img.Rows()))

                    // Desenhar o retângulo ao redor da face detectada
                    gocv.Rectangle(&img, gocv.NewRect(x1, y1, x2-x1, y2-y1), gocv.NewScalar(0, 255, 0, 0), 2)
                }
            }
        }

        // Mostrar a imagem com as detecções
        window.IMShow(img)

        // Pressione 'q' para sair
        if window.WaitKey(1) == 'q' {
            break
        }
    }
}
