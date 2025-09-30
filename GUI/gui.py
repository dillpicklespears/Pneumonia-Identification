import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk

class ProgramInterface:

    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("600x400")
        self.root.title("DegenerativeAI Diagnosis")
        self.root.config(background='grey')
        self.image_label = tk.Label(self.root, text="No Image Selected", bg="black", fg="white", width=300, height=300)
        self.image_label.grid_propagate(False)

        self.current_image = None
        self.photo_image = None

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        button_explore = tk.Button(self.root, text='Upload Image', command=self.browse_files)
        button_exit = tk.Button(self.root, text='Exit', command=exit)

        self.image_label.grid(column=0, row=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        button_explore.grid(column=0, row=2, padx=10, pady=20, sticky="sw")
        button_exit.grid(column=1, row=2, padx=10, pady=20, sticky="se")
        
        self.root.mainloop()

    def browse_files(self):
        starting_dir = os.path.expanduser("~")
        filename = filedialog.askopenfilename(initialdir=starting_dir, title="Select a File", filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"),))

        if filename:
            self.display_image(filename)
            
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
        except Exception as e:
            print(f"Error: {e}")


def main():
    ProgramInterface()

if __name__ == '__main__':
    main()
