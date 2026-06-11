import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
# test with dummy audio
dummy = tf.zeros([16000])  # 1 second of silence
scores, embeddings, log_mel = yamnet(dummy)
print("YAMNet output shape:", embeddings.shape)  # should be (1, 1024)
print("✅ YAMNet working!")