***Diffusion Models***\
Implemented state of the art diffusion model technique for image generation and denoising task. Due to limited availability to compute resources only part of the dataset was used for training. Hence the quality of the generated images may not meet expectations.

**Dataset**\
The dataset is the Animal Faces-HQ dataset [AFHQ](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq). It consists of 15,000 highquality
images at 512 × 512 resolution. The dataset includes three domains of cat, dog, and wildlife, but I used only cats to reduce computation complexity.

**File Description**
1) _data_: Contains the AFHQ dataset.
2) _diffusion.py:_ Constructs the diffusion model, including the forward process, backward process, and scheduler, which you will implement.
3) _main.py_: Serves as the main entry point for training and evaluating your diffusion model. Run it using the command python main.py. Append flags to this command to adjust the diffusion model’s configuration.
4) _requirements.txt_: Lists the packages that need to be installed for this assignment.
5)_ run_in_colab.ipynb_: Provides command lines to train and evaluate your diffusion model in Google Colab.
6) _trainer.py_: Provides code for training and evaluating the diffusion model.

**Flags:**\
| Configuration Parameters  | Example Flag Usage |
| ------------- | ------------- |
| Model image size | --image_size 32 |
| Model batch size | --batch_size 32 |
| Model data domain of AFHQ dataset |--data_class cat |
| Directory where the model is stored | --save_folder ./results/ |
| Path of a trained model | --load_path ./results/model.pt |
| Directory from which to load dataset | --data_path ./data/train/ |
| Number of iterations to train the model  | --train_steps 10000 |
| Number of steps of diffusion process, T | --time_steps 300 |
| Number of output channels of the first layer in U-Net | --unet_dim 16 |
| Learning rate in the training | --learning_rate 1e-3 |
| Frequency of periodic save, sample and (optionally) FID calculation | --save_and_sample_every 1000 |
| Enable FID calculation | --fid |
|Enable visualization | --visualize |

**Sample Results**\
![5_1 (1)](https://github.com/vigkneshvr/Diffusion-Model/assets/48051034/0edcb4aa-d6a2-4411-b532-4b3da35415d4)

![5_2 (1)](https://github.com/vigkneshvr/Diffusion-Model/assets/48051034/1f8ae7df-f2f4-4f4f-98b1-27cd4eab796e)

<img width="641" alt="5_3 (1)" src="https://github.com/vigkneshvr/Diffusion-Model/assets/48051034/86a761d9-3254-4bf2-8939-65573dff98f1">
<img width="641" alt="5_4 (1)" src="https://github.com/vigkneshvr/Diffusion-Model/assets/48051034/f75b2438-e1f2-41d9-ae53-7e27a567fb3b">

Final Generated Images\
![5_5 (1)](https://github.com/vigkneshvr/Diffusion-Model/assets/48051034/cbe5c891-d256-46a9-9d60-2466193c9d4a)

**References**
1) https://www.cs.cmu.edu/~mgormley/courses/10423/



