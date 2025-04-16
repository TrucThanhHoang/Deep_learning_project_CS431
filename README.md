# DisenDiff

FILE PDF: [DisenDiff](https://arxiv.org/abs/2403.18551) 

<div>
<p align="center">
<img src='assets/first_figure.jpg' align="center" width=900>
</p>
</div>

##Update (Cập nhật)
Update các file src/retrieve.py (API của lation không còn hoạt động), run.sh. 

## Datasets
The training images are located in `datasets/images`, the test prompts are located in `datasets/prompts`, and the processed images for evaluating image-alignment can be found in `datasets/data_eval`.

## Key modules
The crucial constraints for optimization are implemented in the function `p_losses` within `src/model.py`.

## Results
<div>
<p align="center">
<img src='assets/results_github.jpg' align="center" width=900>
</p>
</div>

## Getting Started
```
conda env create -f environment.yml
conda activate ldm
git clone https://github.com/CompVis/stable-diffusion.git
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers
pip install google-search-results
mkdir -p checkpoints  
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt -P checkpoints/

mkdir -p real_reg/samples_cat_dog/images
python src/retrieve.py --class_prompt "<new1> cat and <new2> dog" --class_data_dir "real_reg/samples_cat_dog" --num_class_images 5 --api_key ""
```

## Fine-tuning
```
## run training
bash run.sh

## sample and evaluate
bash eval.sh
```
The `run.sh` and `eval.sh` scripts include several hyperparameters such as `classes` in the input image,`data_path`, `save_path`, training `caption`, random `seed`, and more. Please modify these executable files to suit your specific requirements.

