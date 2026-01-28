from loaddata import *
from model import * 

MODELPATH = 'model_complex_with_weight.pth'

# This is the class that connects everything together 
# after instatiating the class, you can run predict_image with the image path

# this will return a dictionary, it contains: 

# class_id: 
#       not terribly important for what we're doing
# class_name: 
#       the class it has decided it is, so either PNEUMONIA or NORMAL
# confidence: 
#       decimal value of how confident it is in that class, between 0.01 and 0.99
# probabilities: 
#       all class probabilities, stored as a list, [0] is normal, [1] is pneumonia

# access these like VARIABLE_STORING_RESULT['class_id/class_name/confidence/probabilities']
# you can access the probabilities with result['probabilities'][0] or result['probabilities'][1]

class Predicter:
    def __init__(self):
        # Select device safely (Windows, Mac, Linux)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model onto selected device
        self.model = torch.load(
            MODELPATH,
            map_location=self.device,
            weights_only=False
        )
        self.model.to(self.device)
        self.model.eval()

        self.loaddata = LoadData()


    def predict_image(self, image_path):
        """Loads a single image, preprocesses it, and returns model prediction details."""
        try:
            # Load the image using PIL
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        # Preprocess: Apply validation/test transforms, add batch dimension, move to device
        image_tensor = self.loaddata.val_test_transforms(image).unsqueeze(0).to(self.device)

        # Make prediction
        self.model.eval() # Ensure model is in evaluation mode
        with torch.no_grad(): # Disable gradient calculations
            output = self.model(image_tensor) # Output raw logits
            probabilities = F.softmax(output, dim=1) # Probabilities
            # Get the highest probability score and the corresponding class index
            confidence, predicted_class_idx = torch.max(probabilities, 1)

        # Extract results
        class_idx = predicted_class_idx.item()
        class_name = self.loaddata.class_names[class_idx] # Map index to class name
        confidence_score = confidence.item()

        # Return results as a dictionary
        return {
            'class_id': class_idx,
            'class_name': class_name,
            'confidence': confidence_score,
            'probabilities': probabilities[0].cpu().numpy() # All class probabilities
        }

if __name__ == "__main__":
    predicter = Predicter()
    
    print("------------------------")

    try:
        normal_dir = os.path.join(predicter.loaddata.test_dir, "NORMAL") # Target the NORMAL directory
        # Get only .jpeg files from the directory
        normal_test_files = [f for f in os.listdir(normal_dir) if f.lower().endswith('.jpeg')]

        if not normal_test_files:
            print(f"No NORMAL test images found in {normal_dir}.")
        else:
            # Select a random image file
            random_filename = random.choice(normal_test_files)
            test_image_path = os.path.join(normal_dir, random_filename)
            print(f"\nPredicting on random NORMAL image: {random_filename}")

            # Get prediction using the function
            result = predicter.predict_image(test_image_path)

            if result:
                # Display the prediction details
                print(f"  Actual class: NORMAL") # State the true class
                print(f"  Predicted class: {result['class_name']}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print(f"  Class probabilities: Normal={result['probabilities'][0]:.4f}, Pneumonia={result['probabilities'][1]:.4f}")

                # Visualize the image with prediction
                try:
                    img = Image.open(test_image_path)
                    plt.figure(figsize=(6, 6))
                    plt.imshow(img, cmap='gray')
                    # Include TRUE label in title for clarity
                    plt.title(f"True: NORMAL | Prediction: {result['class_name']} ({result['confidence']:.4f})")
                    plt.axis('off')
                    plt.show()
                except Exception as e:
                    print(f"Error displaying image {test_image_path}: {e}")

    except FileNotFoundError:
        print(f"Error: Directory {normal_dir} not found.")
    except Exception as e:
        print(f"An error occurred during prediction example: {e}")