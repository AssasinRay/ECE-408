


//
//  nerualNetwork.cpp
//  N_Network
//
//  Created by Hector Zeng on 11/23/15.
//  Copyright © 2015 Mac Book Pro. All rights reserved.
//

#include <stdio.h>
#include <cuda.h>
//standard includes
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

//include definition file
#include "neuralNetwork.h"

using namespace std;
#define BLOCK_SIZE 512
#define BLOCK_SIZE_S 32

void print_last_CUDA_error()
/* just run cudaGetLastError() and print the error message if its return value is not cudaSuccess */
{
  cudaError_t cudaError;

  cudaError = cudaGetLastError();
  if(cudaError != cudaSuccess)
  {
    printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
  }
}
__global__ void getSetMSE_device(double * target, double * input, double * output, int len){
//printf("reach!!!\n");
 __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len)
       partialSum[t] = pow((input[start + t] - target[start+t]),2) ;
    else
       partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len)
       partialSum[BLOCK_SIZE + t] = pow((input[start + BLOCK_SIZE+t] - target[start+BLOCK_SIZE+t]),2);
    else
       partialSum[BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
    if (t == 0){
       output[blockIdx.x] = partialSum[0];
     //printf("partial sum: %f \n", partialSum[0]);
    }
}


__global__ void feedFoward_device(double * wInput, double * input, double * hiddenNeurons, int nHidden, int nPara){

  int i = threadIdx.x + blockDim.x * blockIdx.x; //i is the width -- nPara(nInput/nOutput)
  int j = threadIdx.y + blockDim.y * blockIdx.y; //j is the height -- nHidden

  __shared__ double partialSum[BLOCK_SIZE_S][ 2 * BLOCK_SIZE_S];

  unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE_S;
  if (start + t < nPara && j < nHidden){
       partialSum[j][t] = (input[start + t] * wInput[start+t + j* nPara]);
   //   printf("wInput[%d] for j[%d]: %f \n", t, j, wInput[start+t + j* nPara]);
    }

    else
       partialSum[j][t] = 0;

    if (start + BLOCK_SIZE_S + t < nPara && j < nHidden)
       partialSum[j][BLOCK_SIZE_S + t] = (input[start +BLOCK_SIZE_S+t] * wInput[start+BLOCK_SIZE_S+t+j*nPara]);
    else
       partialSum[j][BLOCK_SIZE_S+ t] = 0;
    __syncthreads();


  if( i < nPara && j < nHidden){

  if(t == 0)
	  hiddenNeurons[j] = 0;


    __syncthreads();
    //printf("partial sum[%d] for j[%d]: %f \n", t, j,partialSum[BLOCK_SIZE_S*nPara + t+ j*nPara] );

    for (unsigned int stride = BLOCK_SIZE_S; stride >= 1; stride >>= 1) {
       __syncthreads();

       if (t < stride)
          partialSum[j][t+ j*nPara] += partialSum[j][t+stride+ j*nPara];
    //   printf("partialsum[%d]: %f: ||stride: %d \n", j, partialSum[t+j*nPara] , stride);

    }


     if (t == 0){
       hiddenNeurons[j] = partialSum[j][0];
  //   printf("J:[%d]||after: %f \n", j, hiddenNeurons[j]);
      hiddenNeurons[j] = 1/(1+exp(-( hiddenNeurons[j])));
     // hiddenNeurons[j] = 0;

    }
  }

}

/*******************************************************************
 * Constructor
 ********************************************************************/
neuralNetwork::neuralNetwork(int nI, int nH, int nO) : nInput(nI), nHidden(nH), nOutput(nO)
{
	//allocate cuda memory
    cudaMalloc((void**)&deviceInput, sizeof(double) * nOutput);
    cudaMalloc((void**)&deviceTarget, sizeof(double) * nOutput);
    cudaMalloc((void**)&deviceOutput, sizeof(double) * nOutput);

    cudaMalloc((void**)&d_wInputHidden, sizeof(double) * (nInput+1) * nHidden);
    cudaMalloc((void**)&d_wHiddenOutput, sizeof(double) * (nOutput) * (nHidden+1));
    cudaMalloc((void**)&d_inputNeurons, sizeof(double) * (nInput+1));
    cudaMalloc((void**)&d_hiddenNeurons, sizeof(double) * (nHidden+1));
    cudaMalloc((void**)&d_outputNeurons, sizeof(double) * nOutput);

    //create neuron lists
    //--------------------------------------------------------------------------------------------------------
    inputNeurons = new( double[nInput + 1] );
    for ( int i=0; i < nInput; i++ ) inputNeurons[i] = 0;

    //create input bias neuron
    inputNeurons[nInput] = -1;

    hiddenNeurons = new( double[nHidden + 1] );
    for ( int i=0; i < nHidden; i++ ) hiddenNeurons[i] = 0;

    //create hidden bias neuron
    hiddenNeurons[nHidden] = -1;

    outputNeurons = new( double[nOutput] );
    for ( int i=0; i < nOutput; i++ ) outputNeurons[i] = 0;

    //create weight lists (include bias neuron weights)
    //--------------------------------------------------------------------------------------------------------
    wInputHidden = new( double*[nInput + 1] );
    for ( int i=0; i <= nInput; i++ )
    {
        wInputHidden[i] = new (double[nHidden]);
        for ( int j=0; j < nHidden; j++ ) wInputHidden[i][j] = 0;
    }

    wHiddenOutput = new( double*[nHidden + 1] );
    for ( int i=0; i <= nHidden; i++ )
    {
        wHiddenOutput[i] = new (double[nOutput]);
        for ( int j=0; j < nOutput; j++ ) wHiddenOutput[i][j] = 0;
    }

    //initialize weights
    //--------------------------------------------------------------------------------------------------------
    initializeWeights();
}

/*******************************************************************
 * Destructor
 ********************************************************************/
neuralNetwork::~neuralNetwork()
{
    //delete neurons
    delete[] inputNeurons;
    delete[] hiddenNeurons;
    delete[] outputNeurons;

    //delete weight storage
    for (int i=0; i <= nInput; i++) delete[] wInputHidden[i];
    delete[] wInputHidden;

    for (int j=0; j <= nHidden; j++) delete[] wHiddenOutput[j];
    delete[] wHiddenOutput;

    //free cuda memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceTarget);

    cudaFree(d_inputNeurons);
    cudaFree(d_wInputHidden);
    cudaFree(d_wHiddenOutput);
    cudaFree(d_hiddenNeurons);
    cudaFree(d_outputNeurons);


}
/*******************************************************************
 * Load Neuron Weights
 ********************************************************************/
bool neuralNetwork::loadWeights(char* filename)
{
    //open file for reading
    fstream inputFile;
    inputFile.open(filename, ios::in);

    if ( inputFile.is_open() )
    {
        vector<double> weights;
        string line = "";

        //read data
        while ( !inputFile.eof() )
        {
            getline(inputFile, line);

            //process line
            if (line.length() > 2 )
            {
                //store inputs
                char* cstr = new char[line.size()+1];
                char* t;
                strcpy(cstr, line.c_str());

                //tokenise
                int i = 0;
                t=strtok (cstr,",");

                while ( t!=NULL )
                {
                    weights.push_back( atof(t) );

                    //move token onwards
                    t = strtok(NULL,",");
                    i++;
                }

                //free memory
                delete[] cstr;
            }
        }

        //check if sufficient weights were loaded
        if ( weights.size() != ( (nInput + 1) * nHidden + (nHidden +  1) * nOutput ) )
        {
            cout << endl << "Error - Incorrect number of weights in input file: " << filename << endl;

            //close file
            inputFile.close();

            return false;
        }
        else
        {
            //set weights
            int pos = 0;

            for ( int i=0; i <= nInput; i++ )
            {
                for ( int j=0; j < nHidden; j++ )
                {
                    wInputHidden[i][j] = weights[pos++];
                }
            }

            for ( int i=0; i <= nHidden; i++ )
            {
                for ( int j=0; j < nOutput; j++ )
                {
                    wHiddenOutput[i][j] = weights[pos++];
                }
            }

            //print success
            cout << endl << "Neuron weights loaded successfuly from '" << filename << "'" << endl;

            //close file
            inputFile.close();

            return true;
        }
    }
    else
    {
        cout << endl << "Error - Weight input file '" << filename << "' could not be opened: " << endl;
        return false;
    }
}
/*******************************************************************
 * Save Neuron Weights
 ********************************************************************/
bool neuralNetwork::saveWeights(char* filename)
{
    //open file for reading
    fstream outputFile;
    outputFile.open(filename, ios::out);

    if ( outputFile.is_open() )
    {
        outputFile.precision(50);

        //output weights
        for ( int i=0; i <= nInput; i++ )
        {
            for ( int j=0; j < nHidden; j++ )
            {
                outputFile << wInputHidden[i][j] << ",";
            }
        }

        for ( int i=0; i <= nHidden; i++ )
        {
            for ( int j=0; j < nOutput; j++ )
            {
                outputFile << wHiddenOutput[i][j];
                if ( i * nOutput + j + 1 != (nHidden + 1) * nOutput ) outputFile << ",";
            }
        }

        //print success
        cout << endl << "Neuron weights saved to '" << filename << "'" << endl;

        //close file
        outputFile.close();

        return true;
    }
    else
    {
        cout << endl << "Error - Weight output file '" << filename << "' could not be created: " << endl;
        return false;
    }
}
/*******************************************************************
 * Feed pattern through network and return results
 ********************************************************************/
int* neuralNetwork::feedForwardPattern(double *pattern)
{
    feedForward(pattern);

    //create copy of output results
    int* results = new int[nOutput];
    for (int i=0; i < nOutput; i++ ) results[i] = clampOutput(outputNeurons[i]);

    return results;
}
/*******************************************************************
 * Return the NN accuracy on the set
 ********************************************************************/
double neuralNetwork::getSetAccuracy( std::vector<dataEntry*>& set )
{
    double incorrectResults = 0;

    //for every training input array
    for ( int tp = 0; tp < (int) set.size(); tp++)
    {
        //feed inputs through network and backpropagate errors
        feedForward( set[tp]->pattern );

        //correct pattern flag
        bool correctResult = true;

        //check all outputs against desired output values
        for ( int k = 0; k < nOutput; k++ )
        {
            //set flag to false if desired and output differ
            if ( clampOutput(outputNeurons[k]) != set[tp]->target[k] ) correctResult = false;
        }

        //inc training error for a incorrect result
        if ( !correctResult ) incorrectResults++;

    }//end for

    //calculate error and return as percentage
    return 100 - (incorrectResults/set.size() * 100);
}
/*******************************************************************
 * Return the NN mean squared error on the set
 ********************************************************************/

double neuralNetwork::getSetMSE( std::vector<dataEntry*>& set )
{
    double mse = 0;

    //for every training input array
    for ( int tp = 0; tp < (int) set.size(); tp++)
    {
        //feed inputs through network and backpropagate errors
        feedForward( set[tp]->pattern );

        //check all outputs against desired output values
        for ( int k = 0; k < nOutput; k++ )
        {
            //sum all the MSEs together
            mse += pow((outputNeurons[k] - set[tp]->target[k]), 2);
        }

    }//end for

    //calculate error and return as percentage
    return mse/(nOutput * set.size());
}


/*
 double neuralNetwork::getSetMSE( std::vector<dataEntry*>& set )
{
    double mse = 0;


          cudaMemcpy(deviceInput, outputNeurons, sizeof(double) * nOutput, cudaMemcpyHostToDevice);
          dim3 dimGrid(nOutput, 1, 1);
          dim3 dimBlock(BLOCK_SIZE, 1, 1);
    //for every training input array
    for ( int tp = 0; tp < (int) set.size(); tp++)
    {
        //feed inputs through network and backpropagate errors
        feedForward( set[tp]->pattern );
        double target[nOutput];
        double temp[nOutput];
        //check all outputs against desired output values
        for ( int k = 0; k < nOutput; k++ )
        {
            //sum all the MSEs together
        	target[k] = (outputNeurons[k] - set[tp]->target[k]);
           // mse += pow((outputNeurons[k] - set[tp]->target[k]), 2);
        }


        cudaMemcpy(deviceTarget, target, sizeof(double) * nOutput, cudaMemcpyHostToDevice);

        getSetMSE_device<<<dimGrid, dimBlock>>>(deviceTarget, deviceInput, deviceOutput, nOutput);
        cudaDeviceSynchronize();

        cudaMemcpy(temp, deviceOutput, sizeof(double) * nOutput, cudaMemcpyDeviceToHost);
        for (int i = 1; i < nOutput; i++) {
                temp[0] += temp[i];
            }
        mse += temp[0];
       // cout<<"//////////"<<*temp<<endl;

    }//end for


    //calculate error and return as percentage
    return mse/(nOutput * set.size());
}*/
/*******************************************************************
 * Initialize Neuron Weights
 ********************************************************************/
void neuralNetwork::initializeWeights()
{
    //set range
    double rH = 1/sqrt( (double) nInput);
    double rO = 1/sqrt( (double) nHidden);

    //set weights between input and hidden
    //--------------------------------------------------------------------------------------------------------
    for(int i = 0; i <= nInput; i++)
    {
        for(int j = 0; j < nHidden; j++)
        {
            //set weights to random values
            wInputHidden[i][j] = ( ( (double)(rand()%100)+1)/100  * 2 * rH ) - rH;
        }
    }

    //set weights between input and hidden
    //--------------------------------------------------------------------------------------------------------
    for(int i = 0; i <= nHidden; i++)
    {
        for(int j = 0; j < nOutput; j++)
        {
            //set weights to random values
            wHiddenOutput[i][j] = ( ( (double)(rand()%100)+1)/100 * 2 * rO ) - rO;
        }
    }
}
/*******************************************************************
 * Activation Function
 ********************************************************************/
inline double neuralNetwork::activationFunction( double x )
{
    //sigmoid function
    return 1/(1+exp(-x));
}
/*******************************************************************
 * Output Clamping
 ********************************************************************/
inline int neuralNetwork::clampOutput( double x )
{
    if ( x < 0.1 ) return 0;
    else if ( x > 0.9 ) return 1;
    else return -1;
}
/*******************************************************************
 * Feed Forward Operation
 ********************************************************************/
void neuralNetwork::feedForward(double* pattern)
{

    //set input neurons to input values
    for(int i = 0; i < nInput; i++) {inputNeurons[i] = pattern[i];}

    double temp[(nInput+1) * nHidden];
    for(int j=0; j < nHidden; j++){
    	for(int i = 0; i <= nInput; i++){
    		temp[i + j * (nInput+1)] = wInputHidden[i][j];

    	}
    }

    //Calculate Hidden Layer values - include bias neuron
    //--------------------------------------------------------------------------------------------------------
    double bias;
      bias = hiddenNeurons[nHidden];
      dim3 dimBlock1(32, 32, 1);
      dim3 dimGrid1((nInput )/32 + 1, (nHidden - 1)/32 + 1, 1);
      cudaMemcpy(d_inputNeurons, inputNeurons, sizeof(double) * (nInput+1), cudaMemcpyHostToDevice);
      cudaMemcpy(d_wInputHidden, temp, sizeof(double) * ((nInput+1)*nHidden), cudaMemcpyHostToDevice);

      feedFoward_device<<<dimGrid1, dimBlock1>>>(d_wInputHidden, d_inputNeurons, d_hiddenNeurons, nHidden, (nInput+1));

      cudaDeviceSynchronize();

      cudaMemcpy(hiddenNeurons, d_hiddenNeurons, sizeof(double) * (nHidden+1), cudaMemcpyDeviceToHost);
      hiddenNeurons[nHidden] = bias;
      //print_last_CUDA_error();

 //cout<<"output[0]: <by CUDA>:"<<hiddenNeurons[0]<<endl;
/*
   for(int j=0; j < nHidden; j++)
    {
        //clear value
        hiddenNeurons[j] = 0;

        //get weighted sum of pattern and bias neuron
        for( int i=0; i <= nInput; i++ ) hiddenNeurons[j] += inputNeurons[i] * wInputHidden[i][j];

        //set to result of sigmoid
        hiddenNeurons[j] = activationFunction( hiddenNeurons[j] );
    }
 //cout<<"output[0]: <by SER>:"<<hiddenNeurons[0]<<endl<<endl;
*/

    //Calculating Output Layer values - include bias neuron
    //--------------------------------------------------------------------------------------------------------
 /*
    for(int j=0; j < nOutput; j++){
       	for(int i = 0; i <= nHidden; i++){
       		temp[i + j * nHidden] = wHiddenOutput[i][ j ];
       	}
       }

        dim3 dimBlock2(32, 32, 1);
        dim3 dimGrid2((nHidden - 1)/32 + 1, (nOutput - 1)/32 + 1, 1);
        cudaMemcpy(d_hiddenNeurons, hiddenNeurons, sizeof(double) * nHidden, cudaMemcpyHostToDevice);
        cudaMemcpy(d_wInputHidden, wHiddenOutput[0], sizeof(double) * (nInput*nHidden), cudaMemcpyHostToDevice);

        feedFoward_device<<<dimGrid2, dimBlock2>>>(d_wInputHidden, d_hiddenNeurons, d_outputNeurons, nOutput, nHidden);
        cudaDeviceSynchronize();
        cudaMemcpy(outputNeurons, d_outputNeurons, sizeof(double) * nOutput, cudaMemcpyDeviceToHost);
 //       cout<<"output[2]: <by CUDA>:"<<outputNeurons[2]<<endl;
*/
   for(int k=0; k < nOutput; k++)
    {
        //clear value
        outputNeurons[k] = 0;

        //get weighted sum of pattern and bias neuron
        for( int j=0; j <= nHidden; j++ ) outputNeurons[k] += hiddenNeurons[j] * wHiddenOutput[j][k];

        //set to result of sigmoid
        outputNeurons[k] = activationFunction( outputNeurons[k] );
    }
 // cout<<"output[2]: <by serial>:"<<outputNeurons[2]<<endl<<endl;


}
