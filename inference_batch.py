import tensorflow as tf
import numpy as np
import pandas as pd
import argparse

def main():
    # 1. Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run batch inference on new dataset using a saved Keras model."
    )
    parser.add_argument("--model_path", type=str, default="models/best_keras_model.keras",
                        help="Path to the saved Keras model.")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the CSV file containing new data.")
    parser.add_argument("--output_csv", type=str, default="data/predictions.csv",
                        help="Path to save the output predictions.")
    args = parser.parse_args()

    # 2. Load the CSV with new records
    print(f"Reading new data from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Data shape: {df.shape}")

    # 3. Load the Keras model
    print(f"Loading model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    # 4. Convert df to NumPy array for inference
    #    Ensure the columns are in the same order as the model expects
    #    For Iris example: [sepal_length, sepal_width, petal_length, petal_width]
    sample_array = df.to_numpy()  # shape (N, 4)

    # 5. Predict probabilities
    print("Running model predictions on the entire dataset...")
    probs = model.predict(sample_array)  # shape (N, num_classes)
    # If you have 3 classes (Iris), you'll get shape (N,3)

    # 6. (Optional) Map class index -> class name
    class_names = ["Setosa", "Versicolor", "Virginica"]
    pred_class_indices = np.argmax(probs, axis=1)
    pred_class_names = [class_names[idx] for idx in pred_class_indices]

    # 7. Attach predictions to original DataFrame
    #    If you only want discrete labels, do:
    df["predicted_class"] = pred_class_names

    #    If you also want probabilities, show them as columns or combined
    #    We'll do separate columns for each class probability:
    for i, cname in enumerate(class_names):
        df[f"prob_{cname}"] = probs[:, i]

    # 8. Save output to CSV
    print(f"Saving predictions to: {args.output_csv}")
    df.to_csv(args.output_csv, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
