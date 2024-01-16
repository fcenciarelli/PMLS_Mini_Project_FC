

# On-device training on the MAX78000 ultra-low-power CNN accelerator

The project was inspired by previous work done in this repo as a Master Project (https://github.com/Gio200023/Continual-Learning-on-Max78000). 
The author of that repo however did not created a complete update script. So the aim of this project was to build on his work and create a complete system. To do so most of the functions were rewritten from sctach in C to improve speed of execution and improve formatting.

## Repository Structure


- /training_scripts = Training scripts containing the model architecture used and a custom train file made to test fine tuning. At the same time contains the trained model file. To run the files please follow https://github.com/MaximIntegratedAI/ai8x-training and https://github.com/MaximIntegratedAI/ai8x-synthesis where you can install two virtual environments.

- /python_scripts = Python Scripts do decode weights arrangement and at the same time create samples in C that will be accepted by MAX78000

- /mnist_training = C compiled project code

Contains the compiled C code to carry out training on the device.

-cnn.h and cnn.c= contains instruction to control the CNN accelerator
```c
int cnn_unload_frozen_layer(uint32_t *out_buf);
int cnn_init_layer(int layer_num);
int cnn_config_layer(int layer_num);
int cnn_start_layer(int layer_num);
int cnn_stop_SMs();
int get_next_OS_layer(int layer_count);
int get_last_OS_layer();
```

-backpropagation.h and backpropagation.c= contains functions to handle the weights and biases and carry out calculations

```c
void output_layer3(const int32_t out, uint8_t *final);

int mod2(int val);

void get_bias(int8_t biases[10]);
void set_bias(int8_t biases[10]);
void get_weights(uint8_t  weights[10][192]);
void set_weights(uint8_t  weights[10][192]);
float cross_entropy_loss(float* softmax_output, int* lables, int output_size, int batch_size);

```

-main.c contains the main script with the training loop


