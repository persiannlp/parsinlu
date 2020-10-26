#!/usr/bin/env python
# coding: utf-8

# In[1]:


import transformers


# In[2]:


import tensorflow as tf


# In[3]:


from transformers import TFGPT2LMHeadModel


# In[4]:


model = TFGPT2LMHeadModel.from_pretrained('gpt2')


# In[5]:


input_spec = tf.TensorSpec([1, 64], tf.int32)
model._set_inputs(input_spec, training=False)
print(model.inputs)
print(model.outputs)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# For FP16 quantization:
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]


# In[ ]:


tflite_model = converter.convert()
open("gpt2-64.tflite", "wb").write(tflite_model)


# In[ ]:




