

#include "backpropagation.h"
#include "stdint.h"
#include "math.h"
#include "stdio.h"
#include "weights.h"
#include "math.h"

void array_substraction(int *res, q15_t *arr1, int *arr2, int size)
{
    // ml_softmax-true_output
    int i = 0;
    double tmp;
    for (i = 0; i < size; i++)
    {
        tmp = ((double)*(arr1 + i) / (double)0x7fff);
        *(res + i) = (round(tmp)) - *(arr2 + i);
    }
}

int mod2(int val)
{
    return -((val >> 7) & 0b1) * pow(2, 7) + ((val >> 6) & 0b1) * pow(2, 6) + ((val >> 5) & 0b1) * pow(2, 5) + ((val >> 4) & 0b1) * pow(2, 4) + ((val >> 3) & 0b1) * pow(2, 3) + ((val >> 2) & 0b1) * pow(2, 2) + ((val >> 1) & 0b1) * pow(2, 1) + ((val >> 0) & 0b1) * pow(2, 0);
}


#define NUM_SEGMENTS 4
#define SEGMENT_SIZE 16

void output_layer_3(const int32_t *out, uint8_t *final)
{
	if (out == NULL || final == NULL) {
		printf("Null pointer passed to output_layer_10.\n");
	    return;
	}

	uint32_t first[SEGMENT_SIZE], second[SEGMENT_SIZE], third[SEGMENT_SIZE], fourth[SEGMENT_SIZE];
	size_t try = 0, i = 0;
	size_t out_length = CNN_NUM_OUTPUTS_FROZEN_LAYER / sizeof(uint32_t);


	while (*out != 0 && try < CNN_NUM_OUTPUTS_FROZEN_LAYER / (NUM_SEGMENTS * SEGMENT_SIZE))
	    {
	        for (i = 0; i < SEGMENT_SIZE; i++)
	        {
	            if (out_length <= 0) {
	                printf("Insufficient data in the input buffer.\n");
	                return;
	            }

	            first[i] = mod2(*out >> 0);
	            second[i] = mod2(*out >> 8);
	            third[i] = mod2(*out >> 16);
	            fourth[i] = mod2(*out >> 24);

	            out++; // Move to next uint32_t
	            out_length--;
	        }

	        size_t base_index = try * SEGMENT_SIZE * NUM_SEGMENTS;
	        for (i = 0; i < SEGMENT_SIZE * NUM_SEGMENTS; i++)
	        {
	            size_t index = base_index + i;
	            if (index < CNN_NUM_OUTPUTS_FROZEN_LAYER)
	            {
	                if (i < SEGMENT_SIZE)
	                    final[index] = first[i];
	                else if (i < SEGMENT_SIZE * 2)
	                    final[index] = second[i - SEGMENT_SIZE];
	                else if (i < SEGMENT_SIZE * 3)
	                    final[index] = third[i - SEGMENT_SIZE * 2];
	                else
	                    final[index] = fourth[i - SEGMENT_SIZE * 3];
	            }
	            else {
	                printf("Index out of bounds in output buffer.\n");
	                return;
	            }
	        }

	        try++;
	    }

}

uint16_t append[16];
int counter = 0;
static int update_place = 0;
static int spot = 0;



void add_kernel(unsigned int *kernels_s[], unsigned int kernel[], int *kernel_size, int *kernels_s_size)
{
	unsigned int *new_kernel = malloc((*kernel_size) * sizeof(unsigned int));
	for (int i = 0; i < *kernel_size; i++)
	{
		new_kernel[i] = kernel[i];
	}
	kernels_s[*kernels_s_size] = new_kernel;
	(*kernels_s_size)++;

}

void extract_bytes(unsigned int num, char values[][3], int *values_size)
{
	unsigned int bytes[4];
	bytes[0] = (num >> 24) & 0xFF;
	bytes[1] = (num >> 16) & 0xFF;
	bytes[2] = (num >> 8) & 0xFF;
	bytes[3] = num & 0xFF;

	for (int i = 0; i < 4; i++)
	{
		printf(values[*values_size], "%02x", bytes[i]);
		(*values_size)++;
	}
}


#define MAX_KERNELS 100
#define MAX_KERNEL_SIZE 400
#define MAX_VALUES 400




void set_weights( uint8_t weights[10][192])
{

	uint8_t subarray[12][10][16] = {0};

	int index = 0, j = 0;

	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 12; j++)
		{
			for (int l = 0; l < 16; l++)
			{
				subarray[j][i][l] = weights[i][index + l];
			}
			index = index + 16;
		}
		index = 0;
	}

	uint8_t flattened[12][162] = {0};

	index = 0;

	for (int i= 0; i < 12; i++)
	{
		for (int j=0; j < 10; j++)
		{
			for (int l =0; l < 16; l++)
			{
				flattened[i][index+l] = subarray[i][j][l];
			}
			index = index + 16;
		}
		index = 0;
	}

	uint8_t sets[12][18][9] = {0};

	index = 0;
	for (int i= 0; i < 12; i++)
		{
			for (int j=0; j < 18; j++)
			{
				for (int l =0; l < 9; l++)
				{
					sets[i][j][l] = flattened[i][index+l];

					// REVERSE HERE
				}
				index = index + 9;
			}

			for (int j = 0; j < 18; ++j)
			{
				for (int l = 0; l < 9/2; ++l)
				{
					uint8_t temp = sets[i][j][l];
					sets[i][j][l] = sets[i][j][9-1-l];
					sets[i][j][9-1-l] = temp;

				}
			}



			index = 0;
		}

	uint8_t values[12][164] = {0};

	index = 0;
	for (int i= 0; i < 12; i++)
	{
		for (int j = 0; j <18; j++)
		{
			for (int l = 0; l< 9; l++)
			{
				values[i][index+l] = sets[i][j][l];
			}
			index = index +9;
		}
		// values[i][index] = (uint8_t) 0;
		// values[i][index+1] = (uint8_t) 0;
		index = 0;
	}

		/*
	for (int i = 0; i < 12; i++)
	{
		for (int j = 0; j < 164; j++)
		{
			printf("%02x ", values[i][j]);
		}
		printf("\n\n");
	}*/


	int active = 0;
	int active2 = 0;
	int counter = 0;
	int r = -1;
	int c = 0;

	int d = 0;

	for (int index = 0; index < 17930; index++)
	{
		// uint32_t element = model_weights[index];


		if (active ==  1)
		{


			model_weights[index] = (values[r][c] << 24) | (values[r][c+1] << 16) | (values[r][c+2] << 8) | values[r][c+3];

			// if (r == 0) printf("%08x ", model_weights[index] );
			c = c+ 4;

		}

		if (c == 164 )
		{
			c = 0;
			active = 0;
		}

		if (active2 == 1)
				{
					d++;
					if (d == 288)
					{
						active = 1;
						d =0;
						active2 = 0;
					}
				}
		if ((model_weights[index] >> 16) == 0x0 && (model_weights[index - 1] >> 24) == 0x50 && (model_weights[index] == 41 || model_weights[index] == 329))
		{
			if (model_weights[index] == 329)
			{
				active2 = 1;
				r = r + 1;
				c = 0;
			}
			else
			{
				active = 1;
				r = r + 1;
				c = 0;
			}

		}

	}

}


void set_bias(int8_t biases[10])
{
	for (int i = 0; i < 10; i++)
	{
		bias[i] = biases[i];
		printf("%d ", mod2((uint8_t) bias[i]));
	}

}



void get_bias(int8_t biases[10])
{
	for (int i = 0; i < 10; i++)
	{
		biases[i] = bias[i];
		printf("%d ", mod2((uint8_t) bias[i]));
	}

}

void get_weights( uint8_t weights[10][192])
{

	uint8_t s_k[12][164] = {0};
	int active = 0;
	int active2 = 0;
	int counter = 0;

	int r = -1;
	int c = 0;
	int d = 0;

	for (int index = 0; index < 17930; index++)
	{
		uint32_t element = model_weights[index];


		if (active ==  1)
		{
			for (int j = 24; j >= 0; j = j - 8)
			{

				s_k[r][c] = (element >> j) & 0xFF;
				c++;
			}
		}

		if (c == 164 )
		{
			c = 0;
			active = 0;
		}

		if (active2 == 1)
				{
					d++;
					if (d == 288)
					{
						active = 1;
						d =0;
						active2 = 0;
					}
				}

		if ((model_weights[index] >> 16) == 0x0 && (model_weights[index - 1] >> 24) == 0x50 && (model_weights[index] == 41 || model_weights[index] == 329))
		{
			if (model_weights[index] == 329)
			{
				active2 = 1;
				r = r + 1;
				c = 0;
			}
			else {
				active = 1;
				r = r + 1;
				c = 0;
			}

		}

	}


	for (int i = 0;  i < 12; i++)
	{
		for (int j = 0; j < 164; j++)
		{
			// printf("%02X ", s_k[i][j]);
		}
		// printf("\n\n");
	}

	uint8_t subarrays[18][9];
	// uint8_t largeArray[192][10] = {0};
	uint8_t finalLargeArray[192][10] = {0};

	for (int channel = 0; channel < 12; channel++)
	{

		for (int i = 0; i < 18; i++)
		{
			for (int j = 0; j < 9; j++)
			{
				int index = i * 9 + j;
                subarrays[i][j] = (index < 160) ? s_k[channel][index] : 0;
			}
		}

		for (int i = 0; i < 18; i++) {
					// printf("Sub-array %d: ", i);
					for (int j = 0; j < 9; j++) {
						//printf("%02x ", subarrays[i][j]);
					}
					// printf("\n");
				}


		for (int i=0; i < 18; ++i)
		{
			for (int j = 0; j < 9/2; ++j)
			{
				uint8_t temp = subarrays[i][j];
				subarrays[i][j] = subarrays[i][9-1-j];
				subarrays[i][9-1-j] = temp;
			}
		}

		for (int i = 0; i < 18; i++) {
			//printf("Sub-array %d: ", i);
			for (int j = 0; j < 9; j++) {
				// printf("%02x ", subarrays[i][j]);
			}
			// printf("\n");
		}

		uint8_t flattened[164];

		for (int i = 0; i < 18; i++) {
			                for (int j = 0; j < 9; j++) {
			                    flattened[i * 9 + j] = subarrays[i][j];
			                }
			            }

		for (int i = 0; i < 164; i++) {
			//printf("%02x ", flattened[i]);
		}


		uint8_t subarrays_new[10][16] = {0};

		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 16; j++) {
				int index = i * 16 + j;
			    if (index < 160) {
			    	subarrays_new[i][j] = flattened[index];
			        }
			    }
		}

		for (int i = 0; i < 10; ++i) {
			for (int j = 0; j < 16; ++j) {
				weights[i][j+(channel*16)] = subarrays_new[i][j];
			}
		}

		for (int i = 0; i < 10; ++i) {
			//printf("Row %d: ", i);
			for (int j = 0; j < 16; ++j) {
				//printf("%d ", mod2(largeArray[j+(channel*16)][i]));
			}
			//printf("\n");
		}
	}

	for (int i = 0; i < 10; ++i) {
				//printf("Row %d: ", i);
				for (int j = 0; j < 192; ++j) {
					//printf("%d ", mod2(largeArray[j][i]));
				}
				// printf("\n");
			}

}


float cross_entropy_loss(float* softmax_output, int* labels, int output_size, int batch_size)
{
	float loss = 0.0;
	for (int i =0; i < batch_size; i++ )
	{
		int label = labels[i];
		loss += -log(softmax_output[i * output_size + label]);
	}


	return loss / batch_size;
}


