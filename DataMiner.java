import java.util.Random;
import java.util.Scanner;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class DataMiner {
	
	
	public static void main(String[] args) {
	   String train = "irisTrainData.arff";
	   String test = "irisTestData.arff";
	   //String total = "iris.arff";
	   
	   String conTrain = "Contraceptive Data Train.arff";
	   String conTest = "Contraceptive Data Test.arff";
	   //String conTotal = "Contraceptive Data.arff";
	   
	   System.out.print("Would you like to mine with the J48 or JRip (RIPPER) Algorithm? ");
	   Scanner decision = new Scanner(System.in);
	   if (decision.next().toLowerCase().contains("j48")) {
		        System.out.println("\n============= Iris Data =============\n");
	   			j48Model(train, test, true);
	   			System.out.println("\n============= Contraception Data =============\n");
	   			j48Model(conTrain, conTest, false);
	   		}
	   		else {
	   			System.out.println("\n============= Iris Data =============\n");
	   			JRipModel(train, test, true);
	   			System.out.println("\n============= Contraception Data =============\n");
	   			JRipModel(conTrain,conTest, false);
	   		}

	}
	/**
	 * Runs RIPPER algorithm on 
	 * @param train filename for training data
	 * @param test filename for testing data
	 * @param iris boolean to determine which data set to print out
	 */
	
	public static void JRipModel(String train, String test, boolean iris) {
		JRip rip = new JRip();
		
		Instances trainData = null;
		Instances testData = null;
 
		try {
			
			DataSource source = new DataSource(train);
	    	trainData = source.getDataSet();
			int num_Attributes = trainData.numAttributes() - 1;
		    trainData.setClassIndex(num_Attributes);
	
			rip.buildClassifier(trainData);
			Evaluation eval = new Evaluation(trainData);
			
		    //Read the test data
		    DataSource testSource = new DataSource(test);
		    testData = testSource.getDataSet();
		    testData.setClassIndex(num_Attributes);
			
		    //Evaluate the test data using the RIPPER algorithm
		    eval.evaluateModel(rip, testData);
		    
		    //Print out the results
		    System.out.println(eval.toSummaryString("\nJRIP Results\n======\n", false));
		    System.out.println(rip);
		    	//System.out.println(eval.toMatrixString());
		    
		    int num_Instances = (int) eval.numInstances();
		    
		    //Print out each entry of the test data and what its predicted class was
		    if (iris) {
		    	for (int index = 0; index < num_Instances; index++ ) {
			    	Instance current = testData.instance(index);
			    	String predicted = "";
			    	
			    	try {
			    		if (rip.classifyInstance(current) == 0) {
			    			predicted = "Iris-setosa";
			    		}
			    		if (rip.classifyInstance(current) == 1) {
			    			predicted = "Iris-versicolor";
			    		}
			    		if (rip.classifyInstance(current) == 2) {
			    			predicted = "Iris-virginica";
			    		}
						System.out.println(current + "---" + predicted);
					} catch (Exception e) {
						e.printStackTrace();
					}
			    }
		    }
		    else {
			    for (int index = 0; index < num_Instances; index++ ) {
			    	Instance current = testData.instance(index);
			    	String predicted = "";
			    	
			    	try {
			    		if (rip.classifyInstance(current) == 0) {
			    			predicted = "No-use";
			    		}
			    		if (rip.classifyInstance(current) == 1) {
			    			predicted = "Long-term";
			    		}
			    		if (rip.classifyInstance(current) == 2) {
			    			predicted = "Short-term";
			    		}
						System.out.println(current + "---" + predicted);
					} catch (Exception e) {
						e.printStackTrace();
					}
			    }
		    }
		   
		    
		    eval.crossValidateModel(rip, testData, 10, new Random());
			   		 
		//Catches various errors that may come up during the reading of the file  
		} catch (Exception e) {
			e.printStackTrace();
		}


	}
	
	/**
	 * Given data, use J48 algorithm to produce train-test model and cross-validate model
	 * @param train - location of train data
	 * @param test - location of test data
	 * @param total - location of train+test data
	 */
	public static void j48Model(String train, String test, boolean iris) {
		   Instances trainData = null;
		   Instances testData = null;
		   
		  //Try reading the file; if it does not exist, let user know 
		    try {
		    	//Build a j48 model using the train data
		    	DataSource source = new DataSource(train);
		    	trainData = source.getDataSet();
				int num_Attributes = trainData.numAttributes() - 1;
			    trainData.setClassIndex(num_Attributes);
			    J48 j = new J48();
			    j.buildClassifier(trainData);
			    Evaluation eval = new Evaluation(trainData);
			    
			    
			    //Read the test data
			    DataSource testSource = new DataSource(test);
			    testData = testSource.getDataSet();
			    testData.setClassIndex(num_Attributes);
			    
			    int num_Instances = testData.numInstances();
			    
			    //Evaluate the test data using the j48 model
			    eval.evaluateModel(j, testData);
			    
			    //Print out the results
			    	//System.out.println(eval.toMatrixString());
			    System.out.println(eval.toSummaryString("\nJ48 Results\n======\n", false)); 
			    System.out.println(j);

			    
			    //Print out each entry of the test data and what its predicted class was
			    if (iris) {
			    	for (int index = 0; index < num_Instances; index++ ) {
				    	Instance current = testData.instance(index);
				    	String predicted = "";
				    	
				    	try {
				    		if (j.classifyInstance(current) == 0) {
				    			predicted = "Iris-setosa";
				    		}
				    		if (j.classifyInstance(current) == 1) {
				    			predicted = "Iris-versicolor";
				    		}
				    		if (j.classifyInstance(current) == 2) {
				    			predicted = "Iris-virginica";
				    		}
							System.out.println(current + "---" + predicted);
						} catch (Exception e) {
							e.printStackTrace();
						}
				    }
			    }
			    else {
				    for (int index = 0; index < num_Instances; index++ ) {
				    	Instance current = testData.instance(index);
				    	String predicted = "";
				    	
				    	try {
				    		if (j.classifyInstance(current) == 0) {
				    			predicted = "No-use";
				    		}
				    		if (j.classifyInstance(current) == 1) {
				    			predicted = "Long-term";
				    		}
				    		if (j.classifyInstance(current) == 2) {
				    			predicted = "Short-term";
				    		}
							System.out.println(current + "---" + predicted);
						} catch (Exception e) {
							e.printStackTrace();
						}
				    }
			    }
			    

			//Catches various errors that may come up during the reading of the file     
			} catch (Exception e) {
				e.printStackTrace();
			}
		    

			
	}
	
}
