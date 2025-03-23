# Paper Outline

## Prompt:
Write a detailed analysis and observation of the development of the CIFAR-100 image classification app.

Please describe the motivation of your project and why it’s worth doing.

## Problem Definition:
- What is the problem that you are addressing?**  
  Provide details that specify whether you are fulfilling A, B, or C requirements.
- Who is the end user, and what is the end user profile?**

- Refer to a bit of information from the slides to talk about classification, why we want to do it automatically, why CNNs are used for this problem, and mention some classic CNNs used. Use References to make citations like it is done in papers. 

## Introduction:
- Introduce the need of the app.
- Continue by addressing the capability of making an app that can propose a solution to the need.
- Admit that our group has been learning to develop such an app and therefore are going to be documenting the process, observations, challenges, and conclusions.

## Theory:
- Expand on the problem and how it can affect potential end users (without mentioning the app or declaring these are end users).
- Propose the solution again, going more in depth on how the app could potentially help solve this problem.
- Note the reality of developing the app with the tools we have at our disposal:
  - **Tkinter** for the GUI
  - **CIFAR-100** for the dataset
  - **Pytorch** and **Tensorboard** for training/testing/calculation
  - **Github Repo**

## GUI Development:
### Building with Tkinter:
- Basics with Tkinter
- Applicable Tkinter code
- Initial functions and methods for front-end development
  - Text
  - Buttons
  - etc.

## Training:
- Use and understanding of provided CNN model
- Hyperparameters for:
  - Batch
  - Epoch
  - Learning rate
- First uses and observations from using Pytorch and Tensorboard
- Evaluation of the training and understanding our loss calculations

## Testing:
- Go over our testing process
- Aim to discuss the testing integration of the entire dataset and implementation into the GUI
- Evaluate the accuracy and efficiency; Identify any errors
- Discuss how end user needs could be met at this point

## Post-Development:
- Discuss the architecture, correctness, and dependability of the trained model.
- Go over any post-training adjustments
- Final GUI detailing
- Detail how the model's performance was validated

## Conclusion:
- Recap the development, our learning, and team-building process
- Go over successes, failures, what we anticipated vs experienced
- Reflect on how this may improve our future as developers, knowledge-wise and team-wise
- Make sure to bring it back around to the overall usefulness of the app and project as a whole
