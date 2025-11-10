import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk

class ProgramInterface:

    def __init__(self, prediction_function):
        self.root = tk.Tk()
        self.root.geometry("600x600")
        self.root.title("DegenerativeAI Diagnosis")
        self.root.config(background='grey')
        self.image_label = tk.Label(self.root, text="No Image Selected", font=("Arial", 23), bg="black", fg="white")
        self.result_label = tk.Label(self.root, text="Results will be displayed here", font=("Arial", 23), bg="black", fg="white")
        self.image_label.grid_propagate(False)
        self.result_label.grid_propagate(False)

        self.current_image = None
        self.photo_image = None
        self.prediction_function = prediction_function
        self.filename = None
        self.result = None

        self.root.grid_rowconfigure(0, weight=3)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        button_explore = tk.Button(self.root, text='Upload Image', command=self.browse_files)
        button_exit = tk.Button(self.root, text='Exit', command=exit)
        button_predict = tk.Button(self.root, text='Predict', command=self.prediction_call)

        self.image_label.grid(column=0, row=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        self.result_label.grid(column=0, row=2, columnspan=3, padx=10, pady=10, sticky="nsew")
        button_explore.grid(column=0, row=3, padx=10, pady=20, sticky="sw")
        button_exit.grid(column=2, row=3, padx=10, pady=20, sticky="se")
        button_predict.grid(column=1, row=3, padx=10, pady=20, sticky="s")
        
        self.root.mainloop()

    def prediction_call(self):
        if self.filename != None:
            self.result = self.prediction_function(self.filename)
            print(self.result)
            self.result_label.configure(text="Result: " + str(self.result['class_name']) + ", Confidence: " + str(round(self.result['confidence'], 3)))
        else:
            print("No image file detected") 


    def browse_files(self):
        starting_dir = os.path.expanduser("~")
        self.filename = filedialog.askopenfilename(initialdir=starting_dir, title="Select a File", filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"),))

        if self.filename:
            self.display_image(self.filename)
            
    def display_image(self, image_path):
        try:
            self.current_image = Image.open(image_path)

            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()

            if label_width < 10:
                label_width = 600
                label_height = 400

            image_ratio = self.current_image.width / self.current_image.height
            label_ratio = label_width / label_height

            if image_ratio > label_ratio:
                new_width = label_width
                new_height = int(label_width / image_ratio)
            else:
                new_height = label_height
                new_width = int(label_height * image_ratio)

            resized_image = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(resized_image)
            self.image_label.configure(image=self.photo_image, text="")
            self.result_label.configure(text="Results will be displayed here.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    ProgramInterface()

if __name__ == '__main__':
    main()
