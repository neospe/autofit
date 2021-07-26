
from ray import tune

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
# transformer_num_heads: needs to be divisor of emb_dim -> see e.g. big_sweep_1


default_4 = {
    "transformer_on": False,
    "transformer_num_heads": 1,
    "transformer_ff_dim": 1,
    "transformer_dropout": 0.1,
    "dense_0_size": 400,
    "dropout_1_on": False,
    "dropout_1": 0.0,
    "dense_1_on": True,
    "dense_1_size": 116,
    "dropout_2_on": False,
    "dropout_2": 0.0,
    "dense_2_on": False,
    "dense_2_size": 87,
    "dropout_3_on": False,
    "dropout_3": 0.0,
    "dense_3_on": True,
    "dense_3_size": 38,
    "pooling_on": True,
    "batch_norm_on": False,
    "batch_size": 1280,
    "optimizer": 'nadam',
    "learning_rate": 0.002,
    "early_stopping_patience": 20,
    "epochs": tune.choice([1000]),
    "custom_seed": False,  #393893510
    "preprocessing": False
}

big_sweep_2 = {
    "transformer_on": False,
    "transformer_num_heads": 1,
    "transformer_ff_dim": 1,
    "transformer_dropout": 0.1,
    "dense_0_size": 400,
    "dropout_1_on": tune.choice([True, False]),
    "dropout_1": tune.uniform(0, 1),
    "dense_1_on": tune.choice([True, False]),
    "dense_1_size": tune.randint(100, 200),
    "dropout_2_on": tune.choice([True, False]),
    "dropout_2": tune.uniform(0, 1),
    "dense_2_on": tune.choice([True, False]),
    "dense_2_size": tune.randint(50, 100),
    "dropout_3_on": tune.choice([True, False]),
    "dropout_3": tune.uniform(0, 1),
    "dense_3_on": tune.choice([True, False]),
    "dense_3_size": tune.randint(0, 50),
    "pooling_on": True,
    "batch_norm_on": tune.choice([True, False]),
    "batch_size": tune.choice([1280]),
    "optimizer": tune.choice(['rmsprop', 'nadam']),
    "learning_rate": tune.choice([0.001, 0.002]),
    "early_stopping_patience": tune.choice([5]),
    "epochs": tune.choice([1000]),
    "custom_seed": False,
    "preprocessing": False
}

default_model_3a = {
    "transformer_num_heads": 5,
    "transformer_ff_dim": 25,
    "transformer_dropout": 0.4,
    "dropout_1_on": True,
    "dropout_1": 0.4,
    "dense_1_on": True,
    "dense_1_size": 100, #200
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
    "learning_rate": 0.005,
    "early_stopping_patience": 10,
    "epochs": tune.choice([1000]),
    "custom_seed": False,  #474877480, 131931278, False
    "preprocessing": True
}

# vorlage: DE-default_model_2
default_model_3 = {
    "transformer_num_heads": 2,
    "transformer_ff_dim": 10,
    "transformer_dropout": 0.4,
    "dropout_1_on": True,
    "dropout_1": 0.6,
    "dense_1_on": False,
    "dense_1_size": 200,
    "dropout_2_on": True,
    "dropout_2": 0.8,
    "dense_2_on": True,
    "dense_2_size": 20,
    "dropout_3_on": True,
    "dropout_3": 0.6,
    "dense_3_on": False,
    "dense_3_size": 20,
    "pooling_on": True,
    "batch_norm_on": False, 
    "batch_size": 1280,
    "optimizer": 'nadam',
    "learning_rate": 0.005,
    "early_stopping_patience": 10,
    "epochs": tune.choice([1000]),
    "custom_seed": False,
    "preprocessing": False
}


big_sweep_1 = {
    "transformer_num_heads": tune.choice(range_divisors(1, 20, 300)),
    "transformer_ff_dim": tune.randint(0, 40),
    "transformer_dropout": tune.uniform(0, 1),
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
    "pooling_on": True,
    "batch_norm_on": tune.choice([True, False]),
    "batch_size": tune.choice([1280]),
    "optimizer": tune.choice(['rmsprop', 'nadam']),
    "learning_rate": tune.choice([0.001, 0.002]),
    "early_stopping_patience": tune.choice([5]),
    "epochs": tune.choice([1000]),
    "custom_seed": False,
    "preprocessing": False
}


# vorlage: EN-big_sweep_1 (Id: 685)
default_model_2 = {
    "transformer_num_heads": 10, #12
    "transformer_ff_dim": 32,
    "transformer_dropout": 0.4,
    "dropout_1_on": True,
    "dropout_1": 0.26,
    "dense_1_on": False,
    "dense_1_size": 200,
    "dropout_2_on": True,
    "dropout_2": 0.68,
    "dense_2_on": True,
    "dense_2_size": 65,
    "dropout_3_on": True,
    "dropout_3": 0.34,
    "dense_3_on": False,
    "dense_3_size": 20,
    "pooling_on": True,
    "batch_norm_on": tune.choice([False]),  #True, 
    "batch_size": 1280,
    "optimizer": tune.choice(['rmsprop']),  #'nadam'
    "learning_rate": tune.choice([0.002]),  #0.0005, 0.001, 
    "early_stopping_patience": 5,
    "epochs": tune.choice([1000]),
    "custom_seed": False,
    "preprocessing": False
}


# entspricht bestem modell aus model-2020-06 (nn_skizze9_english_klein copy 6.py)

default_model_1 = {
    "transformer_num_heads": 12,
    "transformer_ff_dim": 32,
    "transformer_dropout": 0.0,
    "dropout_1_on": True,
    "dropout_1": 0.8,
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
    "epochs": 1000,
    "custom_seed": False,
    "preprocessing": False
}
