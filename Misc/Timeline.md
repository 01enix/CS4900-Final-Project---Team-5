# Projected Timeline

## Week 8 (Biweekly Check-in): Model Architecture Design & Data Preparation
### Primary Goals:
- Design the CNN model architecture for CIFAR-100 classification.
- Load and preprocess CIFAR-100 dataset (normalization, transformations, splitting into training and validation sets).
- Implement data loaders for mini-batches.
- Dataset should be fully ready for training.

### Deliverables for Check-in:
- CNN model architecture defined.
- Dataset preparation (splitting into training, validation).
- Data loading pipeline set up (with mini-batches).
- Code committed to GitHub: Initial commit for model and data preparation.

## Week 9: Model Training & Basic Evaluation
### Primary Goals:
- Start training the CNN model on CIFAR-100 dataset.
- Train for 10-20 epochs initially, visualize training progress with TensorBoard (loss vs. epochs plot).
- Implement model saving functionality.
- Basic evaluation: check accuracy and loss trends.

### Deliverables for Check-in:
- Initial training script for CNN.
- First loss vs. epochs plot using TensorBoard.
- Trained model saved and committed to GitHub.

## Week 10 (Biweekly Check-in): Model Evaluation & Performance Metrics
### Primary Goals:
- Test the model: load the saved model, run the evaluation script.
- Compute performance metrics: accuracy, precision, recall, F1 score (per-class and macro).
- Train the model using super-class ground truth (20 classes) and test.

### Deliverables for Check-in:
- Test script for evaluating the model.
- Model performance metrics (accuracy, precision, recall, F1 score).
- Training and testing with both class and super-class ground truths.
- Code committed to GitHub: Evaluation code.

## Week 11: Learning Rate Experimentation & Epochs Tuning (B-level Requirement)
### Primary Goals:
- Experiment with learning rates (high, medium, low).
- Visualize loss vs. epochs for each learning rate.
- Refine number of epochs based on loss behavior.
- Implement command-line arguments for training configurations (learning rate, epochs).

### Deliverables for Check-in:
- Learning rate experiments (3 plots for different learning rates).
- Epoch selection process based on loss trends.
- Training script with argparse or docopt for configurations (learning rate, epochs).
- Code committed to GitHub: Learning rate experiments, training configurations.

## Week 12: Linear Layer Model & Further Refinement
### Primary Goals:
- Implement a linear layers model (no convolutional layers) for comparison.
- Train and evaluate both models: CNN and linear layer model.
- Refactor code to integrate training configurations (learning rate, epochs).

### Deliverables for Check-in:
- Linear layers model implemented and evaluated.
- Refined code with modular functions and arguments for configurations.
- Code committed to GitHub: Linear model and refactoring.

## Week 13: Model Comparison & Testing Enhancements
### Primary Goals:
- Compare CNN vs. linear model performance on both class and super-class ground truths.
- Implement testing metrics: accuracy, precision, recall, F1-score.
- Refactor code to ensure clean, modular structure (functions, docstrings).

### Deliverables for Check-in:
- Model comparison: CNN vs. linear model evaluation.
- Testing script with additional performance metrics (macro-average, precision, recall, F1).
- Code committed to GitHub: Model comparison and testing scripts.

## Week 14: GUI Development & Integration
### Primary Goals:
- Develop the GUI using Tkinter to load, display, and predict images using the trained model.
- Integrate model predictions with GUI (e.g., show class predictions or super-class results).
- Ensure smooth user interaction with the app.

### Deliverables for Check-in:
- Basic GUI implemented: ability to load an image, display it, and predict the class.
- Model integration with GUI: Show results (predicted class and super-class).
- Code committed to GitHub: GUI code.

## Week 15: GUI Finalization & Testing
### Primary Goals:
- Finalize GUI: polish interface, ensure error handling, and improve user experience.
- Test GUI with real data and ensure predictions are correct and displayed properly.
- Final evaluation of models (CNN and linear) using the trained models on both class and super-class data.

### Deliverables for Check-in:
- Finalized GUI ready for demo.
- Final testing of models on CIFAR-100 and display results on the GUI.
- Code committed to GitHub: Final GUI and model integration.

## Week 16 (Submission Deadline): Final Review & Project Submission
### Primary Goals:
- Conduct final review: ensure the app, model, and evaluation metrics are functioning as expected.
- Prepare and finalize documentation: README with setup instructions, code explanations.
- Submit the project on GitHub and BlazeView.
- Final submission to GitHub and BlazeView by end of the week.

### Deliverables for Submission:
- Finalized README with instructions and explanations.
- Code committed to GitHub: Final code with all required functionalities.
- Project submitted on GitHub and BlazeView.

## Week 17 (Final Presentation): Presentation and Demonstration
### Primary Goals:
- Prepare for the final presentation: finalize slides, demo materials, and app for presentation.
- In-class demo: showcase the Tkinter GUI, trained model, and performance metrics.

### Deliverables:
- Presentation slides with an overview of the project, model architecture, training process, and performance results.
- Final demonstration of the app, including predictions and evaluation metrics.
