
# autofit

Automated end-to-end data preprocessing, model training, and evaluation pipeline for transformer-based text classifiers.

## run

`autofit` is written in `tensorflow`. It includes declarative architecture search using `ray tune` and experiment management using `sacred`.

### install

- `pip install tensorflow numpy pandas tqdm joblib bpemb ray sacred`
- Experiment management also requires `mongodb` and `omniboard`.

### operations

- Set tune.run parameter
    - config.py: config dict definieren
      - vgl. https://docs.ray.io/en/latest/tune/api_docs/search_space.html#random-distributions-api
    - run.py: num_samples, EXPERIMENT_NAME, bpemb (select built-in or custom bpe embeddings)

- Start experiment
    - `export CUDA_VISIBLE_DEVICES="0,1"`
    - `export TUNE_DISABLE_STRICT_METRIC_CHECKING="1"`
    - `python3 run_de.py`

- Connect omniboard
    - to local: `omniboard`
    - to remote host: `omniboard -m 192.168.0.8:27017:sacred`

- Manage experiments
    - add metric columns and sort by scores
    - tag candidate models

- Backup candidates
    - using `datalake/sacred-sync.py`: find and copy all sacred experiments matching a key=value (e.g. tag) to another (i.e. local) MongoDB instance.

## good to know

- Branch `dataload` integrates with a complete MLOps infrastructure: a data loader for various text classification data sets (`dataload`), optionally backed by GridFS (`datalake`).

- In the multi-class setting, training data can be limited and prone to class imbalances. Therefore, the training pipeline in the `dataload` branch also uses a data augmentation system (`augment`). It registers all data transformations (`backlog.json`), so datasets for every training run are fully accounted for.
