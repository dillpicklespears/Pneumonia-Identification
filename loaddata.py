from dataset import *

class LoadData:
    def __init__(self):
        # Define base directories relative to your notebook/script location
        self.data_dir = "chest_xray"
        self.train_dir = os.path.join(self.data_dir, "train")
        self.test_dir = os.path.join(self.data_dir, "test")
        # self.val_dir = os.path.join(self.data_dir, "val")

        # Define the classes based on the subfolder names
        self.class_names = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.val_test_transforms = None

        self.__load__()


    # Helper function to scan directories, filter JPEG images, and collect paths/labels
    def get_image_paths_and_labels(self, data_dir):
        image_paths = []
        labels = []
        #print(f"Scanning directory: {data_dir}")
        for label_name in self.class_names:
            class_dir = os.path.join(data_dir, label_name)
            count = 0
            # List files in the class directory
            for filename in os.listdir(class_dir):
                # Keep only files ending with .jpeg (case-insensitive)
                if filename.lower().endswith('.jpeg'):
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(self.class_to_idx[label_name])
                    count += 1
            #print(f"  Found {count} '.jpeg' images for class '{label_name}'")
        return image_paths, labels

    def __load__(self):
        # Get paths and labels for the training set
        all_train_paths, all_train_labels = self.get_image_paths_and_labels(self.train_dir) # these varibles don't need to be accessed outside this class
        train_counts = collections.Counter(all_train_labels)
        total_train_images = len(all_train_paths)

        print(f"\nTraining Set Counts:")
        print(f"  NORMAL (Class 0): {train_counts[self.class_to_idx['NORMAL']]}")
        print(f"  PNEUMONIA (Class 1): {train_counts[self.class_to_idx['PNEUMONIA']]}")
        print(f"  Total Training Samples: {total_train_images}")

        # Get paths and labels for the test set
        all_test_paths, all_test_labels = self.get_image_paths_and_labels(self.test_dir)
        test_counts = collections.Counter(all_test_labels)
        total_test_images = len(all_test_paths)

        print(f"\nTest Set Counts:")
        print(f"  NORMAL (Class 0): {test_counts[self.class_to_idx['NORMAL']]}")
        print(f"  PNEUMONIA (Class 1): {test_counts[self.class_to_idx['PNEUMONIA']]}")
        print(f"  Total Test Samples: {total_test_images}")

        # NO LONGER USING VAL SET
        # 
        # Get paths and labels for the val set
        # all_val_paths, all_val_labels = self.get_image_paths_and_labels(self.val_dir)
        # val_counts = collections.Counter(all_val_labels)
        # total_val_images = len(all_val_paths)

        # print(f"\nVal Set Counts:")
        # print(f"  NORMAL (Class 0): {val_counts[self.class_to_idx['NORMAL']]}")
        # print(f"  PNEUMONIA (Class 1): {val_counts[self.class_to_idx['PNEUMONIA']]}")
        # print(f"  Total Val Samples: {total_val_images}")


        # SPLITTING TRAINING SET FOR VAL SET
        # 
        # Define proportion for validation set
        val_split_ratio = 0.2 
        SEED = 42

        # Perform stratified split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_train_paths,
            all_train_labels,
            test_size=val_split_ratio,
            stratify=all_train_labels,
            random_state=SEED
        )

        # Print the number of samples in each resulting set
        print(f"Original training image count: {len(all_train_paths)}")
        print(f"--> Split into {len(train_paths)} training samples")
        print(f"--> Split into {len(val_paths)} validation samples")

        # TRANSFORMING THE DATA SET
        # 
        # Transformations for the training set (including augmentation)
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()  # Converts to tensor AND scales to [0, 1]
        ])

        # Transformations for the validation and test sets (NO augmentation)
        self.val_test_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()  # Converts to tensor AND scales to [0, 1]
        ])

        print("Transformation pipelines defined.")

        # CREATING THE DATASETS
        #
        # Instantiate the custom Dataset for each split
        train_dataset = XRayDataset(
            image_paths=train_paths,
            labels=train_labels,
            transform=train_transforms     # Apply training transforms (incl. augmentation)
        )

        val_dataset = XRayDataset(
            image_paths=val_paths,
            labels=val_labels,
            transform=self.val_test_transforms  # Apply validation transforms (no augmentation)
        )

        test_dataset = XRayDataset(
            image_paths=all_test_paths,    # Using all_test_paths from verification step
            labels=all_test_labels,        # Using all_test_labels from verification step
            transform=self.val_test_transforms  # Apply validation/test transforms
        )

        # Print dataset sizes to confirm
        print("\nFinal Dataset objects created:")
        print(f"  Training dataset size:   {len(train_dataset)}")
        print(f"  Validation dataset size: {len(val_dataset)}")
        print(f"  Test dataset size:       {len(test_dataset)}")

        # LOADING THE DATA TO DATALOADER
        #
        # Define batch size (can be tuned depending on GPU memory)
        batch_size = 32

        # Create DataLoader for the training set
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,        # Shuffle data each epoch for training
            num_workers=2,       # Number of subprocesses to use for data loading (adjust based on system)
            pin_memory=True      # Speeds up CPU-GPU transfer if using CUDA
        )

        # Create DataLoader for the validation set
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,       # No need to shuffle validation data
            num_workers=2,
            pin_memory=True
        )

        # Create DataLoader for the test set
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,       # No need to shuffle test data
            num_workers=2,
            pin_memory=True
        )

        print(f"\nDataLoaders created with batch size {batch_size}.")

if __name__ == "__main__":
    loaddata = LoadData()
    loaddata.__load__()
    