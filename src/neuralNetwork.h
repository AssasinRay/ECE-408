


//
//  neuralNetwork.h
//  N_Network
//
//  Created by Hector Zeng on 11/23/15.
//  Copyright © 2015 Mac Book Pro. All rights reserved.
//

/*******************************************************************
 * Basic Feed Forward Neural Network Class
 * ------------------------------------------------------------------
 * Bobby Anguelov - takinginitiative.wordpress.com (2008)
 * MSN & email: banguelov@cs.up.ac.za
 ********************************************************************/

#ifndef NNetwork
#define NNetwork

#include "dataReader.h"

class neuralNetworkTrainer;

class neuralNetwork
{
    //class members
    //--------------------------------------------------------------------------------------------
public:

    //number of neurons
    int nInput, nHidden, nOutput;

    //neurons
    double* inputNeurons;
    double* hiddenNeurons;
    double* outputNeurons;

    //weights
    double** wInputHidden;
    double** wHiddenOutput;

    //device counter parts
    double *deviceInput;
    double *deviceTarget;
    double *deviceOutput;
//--------------------------
    double *d_wInputHidden;
    double *d_wHiddenOutput;

    double *d_inputNeurons;
    double *d_hiddenNeurons;
    double *d_outputNeurons;
    //double *deviceHN_fw;

    //Friends
    //--------------------------------------------------------------------------------------------
    friend neuralNetworkTrainer;

    //public methods
    //--------------------------------------------------------------------------------------------



    //constructor & destructor
    neuralNetwork(int numInput, int numHidden, int numOutput);
    ~neuralNetwork();

    //weight operations
    bool loadWeights(char* inputFilename);
    bool saveWeights(char* outputFilename);
    int* feedForwardPattern( double* pattern );
    double getSetAccuracy( std::vector<dataEntry*>& set );
    double getSetMSE( std::vector<dataEntry*>& set );


    //private methods
    //--------------------------------------------------------------------------------------------

private:

    void initializeWeights();
    inline double activationFunction( double x );
    inline int clampOutput( double x );
    void feedForward( double* pattern );

};

#endif
