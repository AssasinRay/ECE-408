//
//  nerualNetworkTranier.cpp
//  N_Network
//
//  Created by Hector Zeng on 11/23/15.
//  Copyright © 2015 Mac Book Pro. All rights reserved.
//
#include <cuda.h>
//standard includes
#include <iostream>
#include <fstream>
#include <math.h>

//include definition file
#include "neuralNetworkTrainer.h"
#include "neuralNetwork.h"

using namespace std;
int i = 0;
__device__ double device_getOutputErrorGradient(double desiredValue, double outputValue)
{
    return outputValue * ( 1 - outputValue ) * ( desiredValue - outputValue );
}

__global__ void MatAdd_device(double *A, double *B, double *C, int colMax, int rowMax, int useBatch){
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    if(col < colMax && row < rowMax){

        C[col + row * colMax] = A[col + row * colMax] + B[col + row * colMax];
       // printf("current COL:%d ||current ROW: %d ||result: %f\n", col, row,  C[col + row * colMax]);
        if(useBatch){
            A[col + row * colMax] = 0;  //clear deltaInputHidden and deltaHiddenOutput
        }
    }
}



//__global__ void device_backprogate(double* output, double* input, int outsize, int insize,  double* error, double* NN, double learningRate,int size)
//{
////	int idx = threadIdx.x + blockDim.x * blockIdx.x;
////	if(idx>=size) return;
////	int out = idx/insize;
////	int in = idx% insize;
////	__shared__ float p[32];
////	int t = threadIdx.x;
////	int b = blockDim.x;
////	int start = 2*blockIdx.x*16;
////	if( start+t < size) 		p[t] = input[start + t] + learningRate * NN[in] * error[out] ;
////	else		p[t] =0 ;
////	if(start+t+16<size) 		p[t+16] = input[start+t+16] + learningRate * NN[in] * error[out] ;
////	else             		p[t+16] = 0;
////	for(unsigned int stride = 16; stride>=1;stride>>1)
////	{
////		__syncthreads();
////		       if (t < stride)
////		          p[t] += p[t+stride];
////	}
////	if(t==0) output[blockIdx.x] = p[0];
//
//	int idx = threadIdx.x + blockDim.x * blockIdx.x;
//	//if(idx<10) printf("output = %f \n", output[idx]);
//
//	if(idx>=size) return ;
//	//if(idx < 32 ) printf("outputi[%d] = %f \n",idx, output[idx]);
//	int out = idx/insize;
//	int in = idx% insize;
//     output[idx]  = output[idx] + learningRate * NN[in] * error[out];
//     __syncthreads();
//
//	//atomicAdd(&output[idx], learningRate * NN[in] * error[out]);
//    //if(idx < 3 ) printf("outputf[%d] = %f \n",idx, output[idx]);
//}
__global__ void device_backprogate(double* output, double* input, int outsize, int insize,  double* error, double* NN, double learningRate,int size)
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int blocksize = blockDim.x;
    int ix = tx+ bx * blocksize ;
    int iy = ty + by * blocksize;
    //if(ix * insize + iy>32) return;
    if(ix<outsize && iy <insize)
    {
        double sum = input[ix * insize + iy];
       //printf("old(%d) sum = %f NN[%d]=%f err[%d]=%f,lR=%f \n",ix * insize + iy,sum,iy,NN[iy],ix,error[ix],learningRate); //y in x out

        sum += learningRate * NN[iy] * error[ix];
        output[ix * insize + iy] = sum;

       printf("new(%d) sum = %f \n",ix * insize + iy,sum);
    }

//    int tx = threadIdx.x;
//
//       int ty = threadIdx.y;
//       int bx = blockIdx.x;
//       int by = blockIdx.y;
//       int blocksize = blockDim.x;
//       int ix = tx+ bx * blocksize ;
//       int iy = ty + by * blocksize;
//       //if(iy * insize + ix>32) return;
//       if(iy<outsize && ix <insize)
//       {
//           double sum = input[iy * insize + ix];
//          printf("old(%d) sum = %f NN[%d]=%f err[%d]=%f \n",iy * insize + ix,sum,ix,NN[ix],ix,error[iy]);
//           __syncthreads();
//           sum += learningRate * NN[ix] * error[iy];
//           output[iy * insize + ix] = sum;
//
//           printf("new(%d) sum = %f \n",iy * insize + ix,sum);
//       }



}
/*******************************************************************
 * constructor
 ********************************************************************/
neuralNetworkTrainer::neuralNetworkTrainer( neuralNetwork *nn )	:	NN(nn),
epoch(0),
learningRate(LEARNING_RATE),
momentum(MOMENTUM),
maxEpochs(MAX_EPOCHS),
desiredAccuracy(DESIRED_ACCURACY),
useBatch(false),
trainingSetAccuracy(0),
validationSetAccuracy(0),
generalizationSetAccuracy(0),
trainingSetMSE(0),
validationSetMSE(0),
generalizationSetMSE(0)
{

	    cudaMalloc((void**)&d_deltaInputHidden, sizeof(double) * (NN->nInput+1) * NN->nHidden);
	    cudaMalloc((void**)&d_deltaHiddenOutput, sizeof(double) * (NN->nOutput) * (NN->nHidden+1));
	    cudaMalloc((void**)&d_outputErrorGradients,sizeof(double)*(NN->nHidden+1));
	    cudaMalloc((void**)&d_hiddenErrorGradients,sizeof(double)*(NN->nOutput+1));
    //create delta lists
    //--------------------------------------------------------------------------------------------------------
    deltaInputHidden = new( double*[NN->nInput + 1] );
    for ( int i=0; i <= NN->nInput; i++ )
    {
        deltaInputHidden[i] = new (double[NN->nHidden]);
        for ( int j=0; j < NN->nHidden; j++ ) deltaInputHidden[i][j] = 0;
    }

    deltaHiddenOutput = new( double*[NN->nHidden + 1] );
    for ( int i=0; i <= NN->nHidden; i++ )
    {
        deltaHiddenOutput[i] = new (double[NN->nOutput]);
        for ( int j=0; j < NN->nOutput; j++ ) deltaHiddenOutput[i][j] = 0;
    }

    //create error gradient storage
    //--------------------------------------------------------------------------------------------------------
    hiddenErrorGradients = new( double[NN->nHidden + 1] );
    for ( int i=0; i <= NN->nHidden; i++ ) hiddenErrorGradients[i] = 0;

    outputErrorGradients = new( double[NN->nOutput + 1] );
    for ( int i=0; i <= NN->nOutput; i++ ) outputErrorGradients[i] = 0;
}
/*******************************************************************
 * Destructor
 ********************************************************************/
neuralNetworkTrainer::~neuralNetworkTrainer()
{
    //delete neurons
    delete[] hiddenErrorGradients;
    delete[] outputErrorGradients;

    //delete weight storage
    for (int i=0; i <= NN->nInput; i++) delete[] deltaInputHidden[i];
    delete[] deltaInputHidden;

    for (int j=0; j <= NN->nHidden; j++) delete[] deltaHiddenOutput[j];
    delete[] deltaHiddenOutput;

    //free cuda memory
    cudaFree(d_deltaInputHidden);
    cudaFree(d_deltaHiddenOutput);
    cudaFree(d_outputErrorGradients);
    cudaFree(d_hiddenErrorGradients);


}
/*******************************************************************
 * Set training parameters
 ********************************************************************/
void neuralNetworkTrainer::setTrainingParameters( double lR, double m, bool batch )
{
    learningRate = lR;
    momentum = m;
    useBatch = batch;
}
/*******************************************************************
 * Set stopping parameters
 ********************************************************************/
void neuralNetworkTrainer::setStoppingConditions( int mEpochs, double dAccuracy )
{
    maxEpochs = mEpochs;
    desiredAccuracy = dAccuracy;
}
/*******************************************************************
 * Enable training logging
 ********************************************************************/
void neuralNetworkTrainer::enableLogging(const char* filename, int resolution = 1)
{
    //create log file
    if ( ! logFile.is_open() )
    {
        logFile.open(filename, ios::out);

        if ( logFile.is_open() )
        {
            //write log file header
            logFile << "Epoch,Training Set Accuracy, Generalization Set Accuracy,Training Set MSE, Generalization Set MSE" << endl;

            //enable logging
            loggingEnabled = true;

            //resolution setting;
            logResolution = resolution;
            lastEpochLogged = -resolution;
        }
    }
}
/*******************************************************************
 * calculate output error gradient
 ********************************************************************/
inline double neuralNetworkTrainer::getOutputErrorGradient( double desiredValue, double outputValue)
{
    //return error gradient
    return outputValue * ( 1 - outputValue ) * ( desiredValue - outputValue );
}

/*******************************************************************
 * calculate input error gradient
 ********************************************************************/
double neuralNetworkTrainer::getHiddenErrorGradient( int j )
{
    //get sum of hidden->output weights * output error gradients
    double weightedSum = 0;
    for( int k = 0; k < NN->nOutput; k++ ) weightedSum += NN->wHiddenOutput[j][k] * outputErrorGradients[k];

    //return error gradient
    return NN->hiddenNeurons[j] * ( 1 - NN->hiddenNeurons[j] ) * weightedSum;
}
/*******************************************************************
 * Train the NN using gradient descent
 ********************************************************************/
void neuralNetworkTrainer::trainNetwork( trainingDataSet* tSet )
{
    cout	<< endl << " Neural Network Training Starting: " << endl
    << "==========================================================================" << endl
    << " LR: " << learningRate << ", Momentum: " << momentum << ", Max Epochs: " << maxEpochs << endl
    << " " << NN->nInput << " Input Neurons, " << NN->nHidden << " Hidden Neurons, " << NN->nOutput << " Output Neurons" << endl
    << "==========================================================================" << endl << endl;

    //reset epoch and log counters
    epoch = 0;
    lastEpochLogged = -logResolution;

    //train network using training dataset for training and generalization dataset for testing
    //--------------------------------------------------------------------------------------------------------
    while (	( trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy ) && epoch < maxEpochs )
    {
        //store previous accuracy
        double previousTAccuracy = trainingSetAccuracy;
        double previousGAccuracy = generalizationSetAccuracy;

        //use training set to train network
        runTrainingEpoch( tSet->trainingSet );

        //get generalization set accuracy and MSE
        generalizationSetAccuracy = NN->getSetAccuracy( tSet->generalizationSet );
        generalizationSetMSE = NN->getSetMSE( tSet->generalizationSet );

        //Log Training results
        if ( loggingEnabled && logFile.is_open() && ( epoch - lastEpochLogged == logResolution ) )
        {
            logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl;
            lastEpochLogged = epoch;
        }

        //print out change in training /generalization accuracy (only if a change is greater than a percent)
        if ( ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy) )
        {
            cout << "Epoch :" << epoch;
            cout << " TSet Acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE ;
            cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << endl;
        }

        //once training set is complete increment epoch
        epoch++;

    }//end while

    //get validation set accuracy and MSE
    validationSetAccuracy = NN->getSetAccuracy(tSet->validationSet);
    validationSetMSE = NN->getSetMSE(tSet->validationSet);

    //log end
    logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl << endl;
    logFile << "Training Complete!!! - > Elapsed Epochs: " << epoch << " Validation Set Accuracy: " << validationSetAccuracy << " Validation Set MSE: " << validationSetMSE << endl;

    //out validation accuracy and MSE
    cout << endl << "Training Complete!!! - > Elapsed Epochs: " << epoch << endl;
    cout << " Validation Set Accuracy: " << validationSetAccuracy << endl;
    cout << " Validation Set MSE: " << validationSetMSE << endl << endl;
}
/*******************************************************************
 * Run a single training epoch
 ********************************************************************/
void neuralNetworkTrainer::runTrainingEpoch( vector<dataEntry*> trainingSet )
{
    //incorrect patterns
    double incorrectPatterns = 0;
    double mse = 0;

    //for every training pattern
    for ( int tp = 0; tp < (int) trainingSet.size(); tp++)
    {
        //feed inputs through network and backpropagate errors
        NN->feedForward( trainingSet[tp]->pattern );
        backpropagate( trainingSet[tp]->target );

        //pattern correct flag
        bool patternCorrect = true;

        //check all outputs from neural network against desired values
        for ( int k = 0; k < NN->nOutput; k++ )
        {
            //pattern incorrect if desired and output differ
            if ( NN->clampOutput( NN->outputNeurons[k] ) != trainingSet[tp]->target[k] ) patternCorrect = false;

            //calculate MSE
            mse += pow(( NN->outputNeurons[k] - trainingSet[tp]->target[k] ), 2);
        }

        //if pattern is incorrect add to incorrect count
        if ( !patternCorrect ) incorrectPatterns++;

    }//end for

    //if using batch learning - update the weights
    if ( useBatch ) updateWeights();

    //update training accuracy and MSE
    trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);
    trainingSetMSE = mse / ( NN->nOutput * trainingSet.size() );
}
/*******************************************************************
 * Propagate errors back through NN and calculate delta values
 ********************************************************************/
void neuralNetworkTrainer::backpropagate( double* desiredOutputs )
{
    //modify deltas between hidden and output layers
    //--------------------------------------------------------------------------------------------------------
	//useBatch =  0;
	//i++;
	if(!useBatch)
	{
		for (int k = 0; k < NN->nOutput; k++)
		    {
		        //get error gradient for every output node
		        outputErrorGradients[k] = getOutputErrorGradient( desiredOutputs[k], NN->outputNeurons[k] );

		        //for all nodes in hidden layer and bias neuron
		        for (int j = 0; j <= NN->nHidden; j++)
		        {
		        	//printf("output[%d] = %f \n", (j+k*(NN->nHidden + 1)),deltaHiddenOutput[j][k]);
		            //calculate change in weight
		            deltaHiddenOutput[j][k] = learningRate * NN->hiddenNeurons[j] * outputErrorGradients[k] + momentum * deltaHiddenOutput[j][k];
		            //printf("i = % d  and deltaHiddenOutput[%d][%d] = %f \n",i, j,k,deltaHiddenOutput[j][k]);
		            //printf("outputf[%d] = %f \n", (j+k*(NN->nHidden + 1)),deltaHiddenOutput[j][k]);
		        }
		    }

		  for (int j = 0; j < NN->nHidden; j++)
		    {
		        //get error gradient for every hidden node
		        hiddenErrorGradients[j] = getHiddenErrorGradient( j );

		        //for all nodes in input layer and bias neuron
		        for (int i = 0; i <= NN->nInput; i++)
		        {
		            //calculate change in weight
		            deltaInputHidden[i][j] = learningRate * NN->inputNeurons[i] * hiddenErrorGradients[j] + momentum * deltaInputHidden[i][j];
		           // printf("deltaInputHidden[%d][%d]= %d\n",i,j,deltaInputHidden[i][j]);
		        }
		    }

		    //if using stochastic learning update the weights immediately
		   updateWeights();
	}
 else{
//

	 	        double* d_hidden;
	 	        double* d_input;



	 	        int len1 = NN->nOutput * (NN->nHidden+1);
	 	        int len2 = NN->nHidden * (NN->nInput+1);
	 	        int smallsize1 = sizeof(double)*(NN->nHidden+1);
	 	        int smallsize2 = sizeof(double)*(NN->nInput+1);
	 	        int medsize1   = sizeof(double)* (NN->nOutput);
	 	        int medsize2  = sizeof(double)*(NN->nHidden);
	 	        int bigsize1  = sizeof(double)* NN->nOutput * (NN->nHidden+1);
	 	        //printf("bigsize1 = %d \n",bigsize1);
	 	        int bigsize2  = sizeof(double)* NN->nHidden * (NN->nInput+1);

	 	        double deltaO[bigsize1];
	 	        double deltaI[bigsize2];
	 	        int   blocksize = 4;

	 	        for(int k = 0; k < NN->nOutput; k++)
	 	        {
	 	            for(int j = 0; j <= NN->nHidden; j++)
	 	            {
	 	                deltaO[j+ k*(NN->nHidden+1)] = deltaHiddenOutput[j][k];
	 	                printf("i = % d  and deltaHiddenOutput[%d][%d] = %f idx= %d \n",i, j,k,deltaHiddenOutput[j][k],j+ k*(NN->nHidden+1));
	 	            }
	 	        }
	 	        cudaMemcpy(d_deltaHiddenOutput,deltaO,bigsize1,cudaMemcpyHostToDevice);

	 	        for (int k = 0; k < NN->nHidden; k++)
	 	        {

	 	            for (int j = 0; j <= NN->nInput; j++)
	 	            {
	 	               deltaI[j+ k*(NN->nInput+1)]   = deltaInputHidden[j][k];
	 	               //printf("i = % d  and deltaInputHidden[%d][%d]; = %f idx= %d \n",i, j,k,deltaInputHidden[j][k], k*(NN->nHidden+1));
	 	            }
	 	        }
	 	        cudaMemcpy(d_deltaInputHidden,deltaI,bigsize2,cudaMemcpyHostToDevice);

	 	        for(int k = 0 ; k<NN->nOutput;k++)
	 	        {
	 	            outputErrorGradients[k] = getOutputErrorGradient( desiredOutputs[k], NN->outputNeurons[k] );

	 	        }
	 	        cudaMemcpy(d_outputErrorGradients,outputErrorGradients,medsize1,cudaMemcpyHostToDevice);

	 	        for(int k = 0; k<NN->nHidden;k++)
	 	        {
	 	            hiddenErrorGradients[k] = getHiddenErrorGradient( k );
	 	            //printf("hiddenErrorGradients[%d] = %f \n",k,hiddenErrorGradients[k]);
	 	        }
	 	        cudaMemcpy(d_hiddenErrorGradients,hiddenErrorGradients,medsize2,cudaMemcpyHostToDevice);




	 	        cudaMalloc((void**)&d_hidden,smallsize1);
	 	        cudaMemcpy(d_hidden,NN->hiddenNeurons,smallsize1,cudaMemcpyHostToDevice);
	 	        cudaMalloc((void**)&d_input,smallsize2);
	 	        cudaMemcpy(d_input,NN->inputNeurons,smallsize2,cudaMemcpyHostToDevice);

	 	        dim3 block1(9,16,1);
	 	        dim3 grid1((NN->nOutput-1)/blocksize+1,NN->nHidden/blocksize+1,1);
	 	        dim3 grid2((NN->nHidden-1)/blocksize+1,NN->nInput/blocksize+1,1);

	 	        double *outputI;
	 	        double *outputO;
	 	        cudaMalloc((void**)&outputI,bigsize1);
	 	        cudaMalloc((void**)&outputO,bigsize2);

	 	        //printf("403");

	 	        device_backprogate<<<grid1,block1>>>(outputO,d_deltaHiddenOutput,NN->nOutput,NN->nHidden+1,d_outputErrorGradients,d_hidden,learningRate,len1);
	 	        cudaDeviceSynchronize();
	 	        cudaMemcpy(deltaO,outputO,bigsize1,cudaMemcpyDeviceToHost);
	 	        for(int k = 0; k < NN->nOutput; k++)
	 	       	           {
	 	       	            for(int j = 0; j <= NN->nHidden; j++)
	 	       	            {
	 	       	             //printf("do= %d\n", deltaO[j+ k* (NN->nHidden+1)]);
	 	       	                deltaHiddenOutput[j][k]= deltaO[j+ k*(NN->nHidden+1)] ;
	 	       	                printf("deltaHiddenOutput[%d][%d]= %f\n",j,k,deltaHiddenOutput[j][k]);
	 	       	            }
	 	       	           }
	 	       // printf("demacia\n");
	 //	        device_backprogate<<<grid2,block1>>>(outputI,d_deltaInputHidden,NN->nHidden,NN->nInput+1,d_hiddenErrorGradients,d_input,learningRate,len2);
	 //	        cudaDeviceSynchronize();
	 //
	 //	        cudaMemcpy(deltaI,outputI,bigsize2,cudaMemcpyDeviceToHost);
	 //
	 //
	 //
	 //
	 //	        for (int k = 0; k < NN->nHidden; k++)
	 //	        {
	 //
	 //	            for (int j = 0; j <= NN->nInput; j++)
	 //	            {
	 //	                deltaInputHidden[j][k] =  deltaI[j+ k*(NN->nInput+1)]   ;
	 //	            }
	 //	        }
	 	        for (int j = 0; j < NN->nHidden; j++)
	 	            {
	 	                //get error gradient for every hidden node
	 	                hiddenErrorGradients[j] = getHiddenErrorGradient( j );

	 	                //for all nodes in input layer and bias neuron
	 	                for (int i = 0; i <= NN->nInput; i++)
	 	                {
	 	                    //calculate change in weight
	 	                    deltaInputHidden[i][j] += learningRate * NN->inputNeurons[i] * hiddenErrorGradients[j];

	 	                }
	 	            }
	 	        cudaFree(d_hidden);
	 	        cudaFree(d_input);


 }
}
/*******************************************************************
 * Update weights using delta values
 ********************************************************************/
void neuralNetworkTrainer::updateWeights()
{


    int data_size_0 = (NN->nInput+1) * NN->nHidden;
    //cout<<"data size:"<<data_size_0<<endl;
    int data_size_1 = (NN->nHidden+1) * NN->nOutput;
/*
    cudaMemcpy(d_deltaInputHidden, deltaInputHidden[0], data_size_0*sizeof(double), cudaMemcpyHostToDevice);//A
    cudaMemcpy(NN->d_wInputHidden, NN->wInputHidden[0], data_size_0*sizeof(double), cudaMemcpyHostToDevice);//B
    dim3 BlockSize(16,16);
    dim3 GridSize(((NN->nHidden-1)/16+1), (NN->nInput/16+1));
    MatAdd_device<<<GridSize, BlockSize>>>(d_deltaInputHidden,NN->d_wInputHidden,NN->d_wInputHidden,NN->nHidden, (NN->nInput+1), useBatch);

    cudaMemcpy(NN->wInputHidden[0], NN->d_wInputHidden,data_size_0*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(deltaInputHidden[0], d_deltaInputHidden,data_size_0*sizeof(double),cudaMemcpyDeviceToHost);
*/    //input -> hidden weights
    //--------------------------------------------------------------------------------------------------------
    for (int i = 0; i <= NN->nInput; i++)
    {
        for (int j = 0; j < NN->nHidden; j++)
        {
            //update weight
            NN->wInputHidden[i][j] += deltaInputHidden[i][j];

            //clear delta only if using batch (previous delta is needed for momentum
            if (useBatch) deltaInputHidden[i][j] = 0;
        }
    }

    //hidden -> output weights
    //--------------------------------------------------------------------------------------------------------

        cudaMemcpy(d_deltaHiddenOutput, deltaHiddenOutput[0], data_size_1*sizeof(double), cudaMemcpyHostToDevice);//A
        cudaMemcpy(NN->d_wHiddenOutput, NN->wHiddenOutput[0], data_size_1*sizeof(double), cudaMemcpyHostToDevice);//B
        dim3 BlockSize_1(16,16);
        dim3 GridSize_1(((NN->nOutput-1)/16+1), (NN->nHidden/16+1));
        MatAdd_device<<<GridSize_1, BlockSize_1>>>(d_deltaHiddenOutput,NN->d_wHiddenOutput,NN->d_wHiddenOutput,NN->nOutput, (NN->nHidden+1), useBatch);

        cudaMemcpy(NN->wHiddenOutput[0], NN->d_wHiddenOutput,data_size_1*sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(deltaHiddenOutput[0], d_deltaHiddenOutput,data_size_1*sizeof(double),cudaMemcpyDeviceToHost);
   /*
    for (int j = 0; j <= NN->nHidden; j++)
    {
        for (int k = 0; k < NN->nOutput; k++)
        {
            //update weight
            NN->wHiddenOutput[j][k] += deltaHiddenOutput[j][k];

            //clear delta only if using batch (previous delta is needed for momentum)
            if (useBatch)deltaHiddenOutput[j][k] = 0;
        }
    }
    */
}
