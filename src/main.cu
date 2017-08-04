//
//  main.cpp
//  N_Network
//
//  Created by Hector Zeng on 11/23/15.
//  Copyright © 2015 Mac Book Pro. All rights reserved.
//


/*******************************************************************
 * Neural Network Training Example
 * ------------------------------------------------------------------
 * Bobby Anguelov - takinginitiative.wordpress.com (2008)
 * MSN & email: banguelov@cs.up.ac.za
 *********************************************************************/

//standard libraries
#include <iostream>
#include <ctime>
#include <cuda.h>

//custom includes
#include "neuralNetwork.h"
#include "neuralNetworkTrainer.h"

//use standard namespace
using namespace std;

int main()
{
    //seed random number generator
    srand( (unsigned int) time(0) );
    int mode;
    cout<< "Please enter the mode:"<<endl;
    scanf("%d", &mode);
    cudaDeviceReset();
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (mode == 1) {
    	clock_t t;
    	        cout<<"Current MODE: TRAINNING"<<endl<<endl;
    	    //create data set reader and load data file
    	    dataReader d;
    	    d.loadDataFile("letter-recognition-2.csv",16,13);
    	    d.setCreationApproach( STATIC, 10);
    	    t = clock();
    	    //create neural network
    	    neuralNetwork nn(16,10,3);

    	    //create neural network trainer
    	    neuralNetworkTrainer nT( &nn );
    	    nT.setTrainingParameters(0.001, 0.9, false);
    	    nT.setStoppingConditions(100, 90);
    	    nT.enableLogging("log.csv", 5);

    	    //train neural network on data sets
    	    for (int i=0; i < d.getNumTrainingSets(); i++ )
    	    {
    	        nT.trainNetwork( d.getTrainingDataSet() );
    	    }
    	    t = clock() - t;
    	    printf ("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);

    	    //save the weights
    	    nn.saveWeights("weights.csv");


    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    else{
    //use the network weights
        cout<<"Current MODE: OUTPUT"<<endl<<endl;
    //create neural network
    neuralNetwork nn(16,10,3);
    nn.loadWeights("weights.csv");
    double input[16] = { 2,3,3,1,1,8,9,3,4,10,4,6,4,9,1,7};
    double * pattern;
    pattern = input;
    int * result = nn.feedForwardPattern(pattern);


    cout<< endl<<"The Output from the trained Neural Network:"<<result[0]<<result[1]<<result[2]<<endl<<endl;
    }


    cout << endl << endl << "-- END OF PROGRAM --" << endl;
    char c; cin >> c;

}
