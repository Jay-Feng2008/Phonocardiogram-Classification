logs1: pre_train=1, num_heads=[128, 32]
logs2: pre_train=4, num_heads=[32, 32]
logs3: pre_train=4, num_heads[32, 32], lr * 2
logs4: pre_train=4, num_heads=[32, 32], changed residual connection, lr * 1, (best result)
logs5: deleted residual connection, rest remains the same.
logs6: added residual connections between two probsparse and conv layers.
logs7: reduce model_dim to 64 (achieved 100% acc once)
logs8: reduce head_size to 16, model_dim=64
logs9: head_size=[64, 32], model_dim=64, (achieved 100% acc 19 times) 