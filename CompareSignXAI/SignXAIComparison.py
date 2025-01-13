import sys
import numpy as np
import torch
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import vgg16
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions

# Import both SignXAI implementations
from signxai.methods.wrappers import calculate_relevancemap as tf_calculate_relevancemap
from signxai_torch.methods.wrapper import calculate_relevancemap as torch_calculate_relevancemap


def debug_class_predictions(tf_model, torch_model, tf_img, torch_img):
    """Debug function to compare model predictions"""
    # Get TensorFlow prediction
    tf_pred = tf_model.predict(np.expand_dims(tf_img, 0), verbose=0)
    tf_top5 = decode_predictions(tf_pred)[0]
    print("\nTensorFlow top 5 predictions:")
    for i, (class_id, name, prob) in enumerate(tf_top5):
        print(f"{i + 1}. {name}: {prob:.3f}")

    # Get PyTorch prediction
    with torch.no_grad():
        torch_pred = torch.nn.functional.softmax(torch_model(torch_img), dim=1)
        torch_top5 = torch.topk(torch_pred, 5)
        print("\nPyTorch top 5 predictions:")
        for i, (prob, idx) in enumerate(zip(torch_top5.values[0], torch_top5.indices[0])):
            print(f"{i + 1}. Class {idx.item()}: {prob.item():.3f}")

    return tf_top5[0], (torch_top5.values[0][0].item(), torch_top5.indices[0][0].item())


def load_and_preprocess_image(image_path, size=(224, 224)):
    """Load and preprocess image for both implementations"""
    img = Image.open(image_path)
    img = img.resize(size)

    # Convert to numpy array
    img_array = np.array(img)
    print(f"Original image range: [{img_array.min()}, {img_array.max()}]")

    # For TensorFlow/original SignXAI
    tf_img = tf.keras.applications.vgg16.preprocess_input(img_array)
    print(f"TensorFlow preprocessed range: [{tf_img.min()}, {tf_img.max()}]")

    # Make a contiguous copy for PyTorch
    tf_img_copy = np.ascontiguousarray(tf_img)

    # For PyTorch/your implementation
    torch_img = torch.from_numpy(tf_img_copy).permute(2, 0, 1).unsqueeze(0).float()
    print(f"PyTorch preprocessed range: [{torch_img.min().item()}, {torch_img.max().item()}]")

    return img, tf_img, torch_img


def normalize_relevance_map(relevance_map, debug_prefix=""):
    """Normalize relevance map to [-1, 1] range"""
    print(f"\n{debug_prefix} Before normalization:")
    print(f"Shape: {relevance_map.shape}")
    print(f"Type: {type(relevance_map)}")
    print(f"Range: [{np.min(relevance_map) if isinstance(relevance_map, np.ndarray) else relevance_map.min()}, "
          f"{np.max(relevance_map) if isinstance(relevance_map, np.ndarray) else relevance_map.max()}]")

    if isinstance(relevance_map, torch.Tensor):
        relevance_map = relevance_map.detach().cpu().numpy()
        relevance_map = np.squeeze(relevance_map)
        relevance_map = np.transpose(relevance_map, (1, 2, 0))

    if len(relevance_map.shape) == 3:
        # Print channel-wise statistics before summation
        print(f"Channel ranges:")
        for i in range(relevance_map.shape[-1]):
            channel = relevance_map[..., i]
            print(f"Channel {i}: [{channel.min():.6f}, {channel.max():.6f}] "
                  f"(mean: {channel.mean():.6f}, std: {channel.std():.6f})")

        relevance_map = np.sum(relevance_map, axis=-1)

    abs_max = np.max(np.abs(relevance_map))
    if abs_max > 0:
        relevance_map = relevance_map / abs_max

    print(f"{debug_prefix} After normalization:")
    print(f"Shape: {relevance_map.shape}")
    print(f"Range: [{relevance_map.min():.6f}, {relevance_map.max():.6f}]")
    print(f"Mean: {relevance_map.mean():.6f}, Std: {relevance_map.std():.6f}")

    return relevance_map


if __name__ == "__main__":
    # Print versions for debugging
    print("Python version:", sys.version)
    print("TensorFlow version:", tf.__version__)
    print("PyTorch version:", torch.__version__)
    print("NumPy version:", np.__version__)

    image_path = "testimage.jpg"
    img, tf_img, torch_img = load_and_preprocess_image(image_path)

    print("\nInitializing models...")
    tf_model = VGG16(weights='imagenet')
    torch_model = vgg16(pretrained=True)

    # Get detailed predictions
    tf_best, torch_best = debug_class_predictions(tf_model, torch_model, tf_img, torch_img)
    print(f"\nBest predictions:")
    print(f"TensorFlow: {tf_best[1]} (class {tf_best[0]}, prob {tf_best[2]:.3f})")
    print(f"PyTorch: class {torch_best[1]} (prob {torch_best[0]:.3f})")

    # Prepare models
    print("\nPreparing models...")
    tf_model.layers[-1].activation = None
    torch_model.eval()

    method = 'gradient_x_input'
    print(f"\nTesting method: {method}")

    try:
        print("\nTesting TensorFlow implementation...")
        tf_result = tf_calculate_relevancemap(method, tf_img, tf_model)

        print("\nTesting PyTorch implementation...")
        torch_result = torch_calculate_relevancemap(method, torch_img, torch_model)

        if tf_result is not None and torch_result is not None:
            print("\nNormalizing and visualizing results...")
            tf_normalized = normalize_relevance_map(tf_result, "TensorFlow")
            torch_normalized = normalize_relevance_map(torch_result, "PyTorch")

            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.title('Original Image')
            plt.imshow(img)
            plt.axis('off')

            plt.subplot(132)
            plt.title(f'TensorFlow ({method})')
            plt.imshow(tf_normalized, cmap='seismic', vmin=-1, vmax=1)
            plt.axis('off')

            plt.subplot(133)
            plt.title(f'PyTorch ({method})')
            plt.imshow(torch_normalized, cmap='seismic', vmin=-1, vmax=1)
            plt.axis('off')

            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback

        print(traceback.format_exc())