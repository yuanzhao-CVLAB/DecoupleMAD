
<h1 align="center"> DecoupleMAD: Boosting Sensitivity in Multimodal Anomaly Detection via Representation Decoupling </h1> 



<h2 id="file_cabinet"> :file_cabinet: Datasets </h2>

In our experiments, we employed two datasets featuring rgb images and point clouds: [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) and [Eyecandies](https://eyecan-ai.github.io/eyecandies/). You can preprocess them with the scripts contained in `data/processing_eyecandies.py`(or processing_mvtec_SN). Then, you need to replace the  "data_root" with your path in the config.json.



### :hammer_and_wrench: Setup Instructions

**Dependencies**: Ensure that you have installed all the necessary dependencies. The list of dependencies can be found in the `./requirements.txt` file.




### :rocket: Inference DecoupleMAD

The main.py script tests DecoupleMAD using either Python or the accelerate launch command (for multiple GPUs). You can select the classes to be tested by modifying the datasets_classes dictionary in data/ader_dataset.py.

```bash
python main.py --mode eval --resume /your/checkpoint/path
```


### :rocket: Train DecoupleMAD

The main.py script trains DecoupleMAD using either Python or the accelerate launch command (for multiple GPUs). You can select the classes to be trained by modifying the datasets_classes dictionary in data/ader_dataset.py.

```bash
python main.py --mode train  
```

You can configure the following options:
   - `--print_bar_step`:  Evaluates the model every print_bar_step epochs.
   - `--img_size`:  Specifies the input image size.
   - `--EPOCHs`: Number of epochs for DecoupleMAD optimization.
   - `--batch_size`: Number of samples per batch for DecoupleMAD optimization.



## :pray: Contacts

For questions, please send an email to alex.costanzino@unibo.it or pierluigi.zama@unibo.it.
