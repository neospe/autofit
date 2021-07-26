
from ray import tune

# transformer_num_heads needs to be divisor of emb_dim
# see e.g. big_sweep_1
def range_divisors(start, end, emb_dim):
    divisors = []
    for i in range(start, end):
        if emb_dim % i == 0:
            divisors.append(i)
    return divisors


# API: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#random-distributions-api

# OTHER VALUES
# randint: braucht lower bound = 0
# optimizer: nadam, adam, sgd, rmsprop
# custom_seed: int, False

# ATTENTION
# pooling_on: False -> erzeugt architekturen, die crashen


# latest
default_model_4 = {
    "transformer_on": False,
    "transformer_num_heads": 10,
    "transformer_ff_dim": 2,
    "transformer_dropout": 0.15,
    "dense_0_size": 100,
    "dropout_1_on": True,
    "dropout_1": 0.17,
    "dense_1_on": False,
    "dense_1_size": 100,
    "dropout_2_on": False,
    "dropout_2": 0.24,
    "dense_2_on": False,
    "dense_2_size": 4,
    "dropout_3_on": True,
    "dropout_3": 0.68,
    "dense_3_on": False,
    "dense_3_size": 0,
    "pooling_on": True,
    "batch_norm_on": True, 
    "batch_size": 1280,
    "optimizer": 'nadam',
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
    "epochs": tune.choice([1000]),
    "custom_seed": False,
    "preprocessing": True
}

big_sweep_1 = {
    "transformer_on": tune.choice([True, False]),
    "transformer_num_heads": tune.choice(range_divisors(1, 20, 300)),
    "transformer_ff_dim": tune.randint(0, 40),
    "transformer_dropout": tune.uniform(0, 1),
    "dense_0_size": tune.randint(0, 200),
    "dropout_1_on": tune.choice([True, False]),
    "dropout_1": tune.uniform(0, 1),
    "dense_1_on": tune.choice([True, False]),
    "dense_1_size": tune.randint(0, 200),
    "dropout_2_on": tune.choice([True, False]),
    "dropout_2": tune.uniform(0, 1),
    "dense_2_on": tune.choice([True, False]),
    "dense_2_size": tune.randint(0, 100),
    "dropout_3_on": tune.choice([True, False]),
    "dropout_3": tune.uniform(0, 1),
    "dense_3_on": tune.choice([True, False]),
    "dense_3_size": tune.randint(0, 100),
    "pooling_on": tune.choice([True, False]),
    "batch_norm_on": tune.choice([True, False]),
    "batch_size": tune.choice([1280]),
    "optimizer": tune.choice(['rmsprop', 'nadam']),
    "learning_rate": tune.choice([0.001, 0.002]),
    "early_stopping_patience": tune.choice([5]),
    "epochs": tune.choice([1000]),
    "custom_seed": False,
    "preprocessing": True
}

default_model_3 = {
    "transformer_num_heads": 10,
    "transformer_ff_dim": 2,
    "transformer_dropout": 0.15,
    "dropout_1_on": True,
    "dropout_1": 0.39,
    "dense_1_on": True,
    "dense_1_size": 100,
    "dropout_2_on": True,
    "dropout_2": 0.24,
    "dense_2_on": True,
    "dense_2_size": 4,
    "dropout_3_on": False,
    "dropout_3": 0.0,
    "dense_3_on": False,
    "dense_3_size": 0,
    "pooling_on": True,
    "batch_norm_on": False, 
    "batch_size": 1280,
    "optimizer": 'rmsprop',
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
    "epochs": tune.choice([1000]),
    "custom_seed": False,
    "preprocessing": True
}

default_model_2 = {
    "transformer_num_heads": 2,
    "transformer_ff_dim": 100,
    "transformer_dropout": 0.8,
    "dropout_1_on": True,
    "dropout_1": 0.6,
    "dense_1_on": False,
    "dense_1_size": 100,
    "dropout_2_on": True,
    "dropout_2": 0.8,
    "dense_2_on": True,
    "dense_2_size": 20,
    "dropout_3_on": True,
    "dropout_3": 0.6,
    "dense_3_on": False,
    "dense_3_size": 20,
    "pooling_on": True,
    "batch_norm_on": True, 
    "batch_size": 1280,
    "optimizer": 'nadam',
    "learning_rate": 0.005,
    "early_stopping_patience": 10,
    "epochs": tune.choice([1000]),
    "custom_seed": False,
    "preprocessing": True
}

small_sweep_1_bestmodel = {
    "transformer_num_heads": 5,
    "transformer_ff_dim": 25,
    "transformer_dropout": 0.0,
    "dropout_1_on": True,
    "dropout_1": 0.4,
    "dense_1_on": True,
    "dense_1_size": 200,
    "dropout_2_on": True,
    "dropout_2": 0.8,
    "dense_2_on": True,
    "dense_2_size": 20,
    "dropout_3_on": True,
    "dropout_3": 0.4,
    "dense_3_on": True,
    "dense_3_size": 20,
    "pooling_on": True,
    "batch_norm_on": False,
    "batch_size": 1280,
    "optimizer": 'nadam',
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
    "epochs": tune.choice([1000]),
    "custom_seed": False,  #131931278
    "preprocessing": True
}

small_sweep_1_modified = {
    "transformer_num_heads": tune.choice([3, 6, 12]),
    "transformer_ff_dim": tune.choice([6, 12, 24, 32]),
    "transformer_dropout": tune.choice([0.0, 0.1, 0.2]),
    "dropout_1_on": tune.choice([True]),
    "dropout_1": tune.choice([0.4]),
    "dense_1_on": tune.choice([True]),
    "dense_1_size": tune.choice([200]),
    "dropout_2_on": tune.choice([True]),
    "dropout_2": tune.choice([0.8]),
    "dense_2_on": tune.choice([True]),
    "dense_2_size": tune.choice([20]),
    "dropout_3_on": tune.choice([True, False]),
    "dropout_3": tune.choice([0.4]),
    "dense_3_on": tune.choice([True, False]),
    "dense_3_size": tune.choice([20]),
    "pooling_on": tune.choice([True]),
    "batch_norm_on": tune.choice([True, False]),
    "batch_size": tune.choice([1280]),
    "optimizer": tune.choice(['rmsprop', 'nadam']),
    "learning_rate": tune.choice([0.001, 0.002]),
    "early_stopping_patience": tune.choice([2]),
    "epochs": tune.choice([1000]),
    "custom_seed": False,
    "preprocessing": False
}

default_model_1_modified = {
    "transformer_num_heads": 12,
    "transformer_ff_dim": 32,
    "transformer_dropout": 0.2,
    "dropout_1_on": True,
    "dropout_1": 0.4,
    "dense_1_on": True,
    "dense_1_size": 200,
    "dropout_2_on": True,
    "dropout_2": 0.8,
    "dense_2_on": True,
    "dense_2_size": 20,
    "dropout_3_on": True,
    "dropout_3": 0.4,
    "dense_3_on": True,
    "dense_3_size": 20,
    "pooling_on": True,
    "batch_norm_on": True,
    "batch_size": 1280,
    "optimizer": 'nadam',
    "learning_rate": 0.002,
    "early_stopping_patience": 10,
    "epochs": tune.choice([1000]),
    "custom_seed": False,
    "preprocessing": False
}

big_sweep_1_bestmodel = {
    "transformer_num_heads": 6,
    "transformer_ff_dim": 12,
    "transformer_dropout": 0.2,
    "dropout_1_on": True,
    "dropout_1": 0.5,
    "dense_1_on": True,
    "dense_1_size": 136,
    "dropout_2_on": False,
    "dropout_2": 0.0,
    "dense_2_on": False,
    "dense_2_size": 0,
    "dropout_3_on": True,
    "dropout_3": 0.5,
    "dense_3_on": True,
    "dense_3_size": 22,
    "pooling_on": True,
    "batch_norm_on": False,
    "batch_size": 1280,
    "optimizer": 'nadam',
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
    "epochs": tune.choice([1000]),
    "custom_seed": False,
    "preprocessing": True
}

small_sweep_1 = {
    "transformer_num_heads": tune.choice([6, 12]),
    "transformer_ff_dim": tune.choice([32]),
    "transformer_dropout": tune.choice([0.1, 0.2]),
    "dropout_1_on": tune.choice([True]),
    "dropout_1": tune.choice([0.4]),
    "dense_1_on": tune.choice([True]),
    "dense_1_size": tune.choice([200]),
    "dropout_2_on": tune.choice([True]),
    "dropout_2": tune.choice([0.8]),
    "dense_2_on": tune.choice([True]),
    "dense_2_size": tune.choice([20]),
    "dropout_3_on": tune.choice([True, False]),
    "dropout_3": tune.choice([0.4]),
    "dense_3_on": tune.choice([True, False]),
    "dense_3_size": tune.choice([20]),
    "pooling_on": tune.choice([True]),
    "batch_norm_on": tune.choice([True, False]),
    "batch_size": tune.choice([1280]),
    "optimizer": tune.choice(['rmsprop', 'nadam']),
    "learning_rate": tune.choice([0.001, 0.002]),
    "early_stopping_patience": tune.choice([5]),
    "epochs": tune.choice([1000]),
    "custom_seed": False,
    "preprocessing": True
}

default_model_1 = {
    "transformer_num_heads": 12,
    "transformer_ff_dim": 32,
    "transformer_dropout": 0.0,
    "dropout_1_on": True,
    "dropout_1": 0.4,
    "dense_1_on": True,
    "dense_1_size": 200,
    "dropout_2_on": True,
    "dropout_2": 0.8,
    "dense_2_on": True,
    "dense_2_size": 20,
    "dropout_3_on": True,
    "dropout_3": 0.4,
    "dense_3_on": True,
    "dense_3_size": 20,
    "pooling_on": True,
    "batch_norm_on": True,
    "batch_size": 1280,
    "optimizer": 'rmsprop',
    "learning_rate": 0.001,
    "early_stopping_patience": 5,
    "epochs": tune.choice([1000]),
    "custom_seed": False,
    "preprocessing": False
}