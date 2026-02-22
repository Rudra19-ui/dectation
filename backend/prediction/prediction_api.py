#!/usr/bin/env python3
"""
Simple API for Breast Cancer Detection Prediction
Easy-to-use functions for integration
"""

import os

from predict_model import load_model, predict_image, preprocess_image


class BreastCancerPredictor:
    """Simple predictor class for breast cancer detection"""

    def __init__(self, model_path=None):
        """Initialize the predictor with a trained model"""
        self.model = load_model(model_path)
        if self.model is None:
            raise ValueError("Failed to load model")

    def predict(self, image_path):
        """
        Predict class for a mammogram image

        Args:
            image_path (str): Path to the mammogram image

        Returns:
            dict: Prediction results with keys:
                - predicted_class (str): 'Normal', 'Benign', or 'Malignant'
                - confidence_score (float): Confidence between 0 and 1
                - all_probabilities (list): Probabilities for all classes
                - success (bool): Whether prediction was successful
        """
        try:
            result = predict_image(self.model, image_path)
            if result:
                result["success"] = True
                return result
            else:
                return {
                    "predicted_class": None,
                    "confidence_score": 0.0,
                    "all_probabilities": [0.0, 0.0, 0.0],
                    "success": False,
                    "error": "Failed to process image",
                }
        except Exception as e:
            return {
                "predicted_class": None,
                "confidence_score": 0.0,
                "all_probabilities": [0.0, 0.0, 0.0],
                "success": False,
                "error": str(e),
            }

    def predict_batch(self, image_paths):
        """
        Predict classes for multiple mammogram images

        Args:
            image_paths (list): List of image paths

        Returns:
            list: List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result["image_path"] = image_path
            results.append(result)
        return results


def quick_predict(image_path, model_path=None):
    """
    Quick prediction function for single image

    Args:
        image_path (str): Path to mammogram image
        model_path (str, optional): Path to model file

    Returns:
        dict: Prediction results
    """
    try:
        predictor = BreastCancerPredictor(model_path)
        return predictor.predict(image_path)
    except Exception as e:
        return {
            "predicted_class": None,
            "confidence_score": 0.0,
            "all_probabilities": [0.0, 0.0, 0.0],
            "success": False,
            "error": str(e),
        }


def format_prediction_result(result):
    """
    Format prediction result for easy reading

    Args:
        result (dict): Prediction result from predict() or quick_predict()

    Returns:
        str: Formatted result string
    """
    if not result.get("success", False):
        return f"❌ Prediction failed: {result.get('error', 'Unknown error')}"

    class_name = result["predicted_class"]
    confidence = result["confidence_score"]
    confidence_pct = confidence * 100

    # Determine confidence level
    if confidence > 0.8:
        level = "High"
        emoji = "🟢"
    elif confidence > 0.6:
        level = "Medium"
        emoji = "🟡"
    else:
        level = "Low"
        emoji = "🔴"

    return (
        f"{emoji} Predicted: {class_name} | Confidence: {confidence_pct:.1f}% ({level})"
    )


# Example usage functions
def example_usage():
    """Example of how to use the prediction API"""
    print("🏥 Breast Cancer Detection API - Example Usage")
    print("=" * 50)

    # Example 1: Quick prediction
    print("\n📋 Example 1: Quick prediction")
    print("result = quick_predict('path/to/mammogram.jpg')")
    print("print(format_prediction_result(result))")

    # Example 2: Using predictor class
    print("\n📋 Example 2: Using predictor class")
    print("predictor = BreastCancerPredictor()")
    print("result = predictor.predict('path/to/mammogram.jpg')")
    print("print(result['predicted_class'], result['confidence_score'])")

    # Example 3: Batch prediction
    print("\n📋 Example 3: Batch prediction")
    print("predictor = BreastCancerPredictor()")
    print("results = predictor.predict_batch(['img1.jpg', 'img2.jpg'])")
    print("for result in results:")
    print("    print(format_prediction_result(result))")


if __name__ == "__main__":
    example_usage()
