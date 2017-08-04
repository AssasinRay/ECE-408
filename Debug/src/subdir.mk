################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/main.cu \
../src/neuralNetwork.cu \
../src/neuralNetworkTrainer.cu 

CPP_SRCS += \
../src/dataReader.cpp 

OBJS += \
./src/dataReader.o \
./src/main.o \
./src/neuralNetwork.o \
./src/neuralNetworkTrainer.o 

CU_DEPS += \
./src/main.d \
./src/neuralNetwork.d \
./src/neuralNetworkTrainer.d 

CPP_DEPS += \
./src/dataReader.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I"/Developer/NVIDIA/CUDA-7.5/samples/0_Simple" -I"/Developer/NVIDIA/CUDA-7.5/samples/common/inc" -I"/Users/Zhuangyiwei/Desktop/NN_CUDA_NEW/NN_CUDA_NEW" -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I"/Developer/NVIDIA/CUDA-7.5/samples/0_Simple" -I"/Developer/NVIDIA/CUDA-7.5/samples/common/inc" -I"/Users/Zhuangyiwei/Desktop/NN_CUDA_NEW/NN_CUDA_NEW" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I"/Developer/NVIDIA/CUDA-7.5/samples/0_Simple" -I"/Developer/NVIDIA/CUDA-7.5/samples/common/inc" -I"/Users/Zhuangyiwei/Desktop/NN_CUDA_NEW/NN_CUDA_NEW" -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -I"/Developer/NVIDIA/CUDA-7.5/samples/0_Simple" -I"/Developer/NVIDIA/CUDA-7.5/samples/common/inc" -I"/Users/Zhuangyiwei/Desktop/NN_CUDA_NEW/NN_CUDA_NEW" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


