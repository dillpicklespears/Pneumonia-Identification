from GUI import gui
import predict

if __name__ == "__main__":
    prediction = predict.Predicter()
    prediction_function = prediction.predict_image
    window = gui.ProgramInterface(prediction_function) 