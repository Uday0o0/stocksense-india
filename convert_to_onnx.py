import tensorflow as tf
import subprocess
import os

STOCKS = {
    "Reliance": "Reliance_model",
    "TCS": "TCS_model",
    "Infosys": "Infosys_model",
    "HDFC_Bank": "HDFC_Bank_model",
    "ICICI_Bank": "ICICI_Bank_model",
    "Wipro": "Wipro_model",
    "Bajaj_Finance": "Bajaj_Finance_model",
    "Bharti_Airtel": "Bharti_Airtel_model",
    "LT": "LT_model",
    "Asian_Paints": "Asian_Paints_model",
}

SAVE_DIR = r"C:\Users\LENOVO\stock price predictor"
PYTHON   = r"C:\Users\LENOVO\tf_env\Scripts\python.exe"

for name, model_file in STOCKS.items():
    model_path = os.path.join(SAVE_DIR, f"{model_file}.keras")
    saved_path = os.path.join(SAVE_DIR, f"{name}_saved")
    onnx_path  = os.path.join(SAVE_DIR, f"{name}.onnx")

    print(f"\nConverting {name}...")

    # Load and export as SavedModel
    model = tf.keras.models.load_model(model_path)
    model.export(saved_path)
    print(f"  Exported SavedModel")

    # Convert using subprocess (handles spaces in paths correctly)
    result = subprocess.run(
        [PYTHON, "-m", "tf2onnx.convert",
         "--saved-model", saved_path,
         "--output", onnx_path,
         "--opset", "13"],
        capture_output=True, text=True
    )

    if os.path.exists(onnx_path):
        size = os.path.getsize(onnx_path)
        print(f"  ✅ {name}.onnx saved! ({size // 1024} KB)")
    else:
        print(f"  ❌ Failed!")
        print(result.stderr[-500:] if result.stderr else "No error output")

print("\n=== All done! ===")
