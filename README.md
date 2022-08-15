# Setups

1. [Pytorch Installation on Conda](https://pytorch.org/)
: We use following command to install pytorch on Ubuntu 20.04 using Conda with python 3.7.6

```bash
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c Pytorch
```

2. [PyG Installation on Conda](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
: We use Pytorch 1.9.0 Linux Conda CUDA 10.2

```bash
conda install pyg -c pyg
```

3. Install Dependencies using <code> conda install --file requirements.txt </code>

4. Run the code with following commandline/parameters
```bash
 python main.py --use_user true --use_date true --use_entity true 
```

5. [Downloading UPFD dataset](https://github.com/safe-graph/GNN-FakeNews)
- The UPFD dataset will be automatically downloaded to <code> data/{dataset-name}/raw/</code> when runnig <code> main.py </code>

