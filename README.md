***Diffusion Models***\\
Implemented state of the art diffusion model technique for image generation and denoising task. Due to limited availability to compute resources only part of the dataset was used for training. Hence the quality of the generated images may not meet expectations.

**Dataset**\\
The dataset is the Animal Faces-HQ dataset [AFHQ](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq). It consists of 15,000 highquality
images at 512 × 512 resolution. The dataset includes three domains of cat, dog, and wildlife, but I used only cats to reduce computation complexity.

**File Description**\\
1) _data_: Contains the AFHQ dataset.
2) _diffusion.py:_ Constructs the diffusion model, including the forward process, backward process, and scheduler, which you will implement.
3) _main.py_: Serves as the main entry point for training and evaluating your diffusion model. Run it using the command python main.py. Append flags to this command to adjust the diffusion model’s configuration.
4) _requirements.txt_: Lists the packages that need to be installed for this assignment.
5)_ run_in_colab.ipynb_: Provides command lines to train and evaluate your diffusion model in Google Colab.
6) _trainer.py_: Provides code for training and evaluating the diffusion model.
