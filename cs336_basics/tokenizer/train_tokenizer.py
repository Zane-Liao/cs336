"""
Use tokenizers for comparison
"""
# from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# tokenizer = Tokenizer(models.BPE())

# tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()


# trainer = trainers.BpeTrainer(
#     vocab_size=10000,
#     special_tokens=["<|endoftext|>"]
# )


# files = ["../../data/owt_train.txt"]

# tokenizer.train(files, trainer)

# tokenizer.save("vocab_train1.json")