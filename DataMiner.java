import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class DataMiner {
	
	
	public static void main(String[] args) {
	   String train = "irisTrainData.arff";
	   String test = "irisTestData.arff";
	   String total = "iris.arff";
	   j48Model(train, test, total);
	   
	   String conTrain = "Contraceptive Data Train.arff";
	   String conTest = "Contraceptive Data Test.arff";
	   String conTotal = "Contraceptive Data.arff";
	   j48Model(conTrain, conTest, conTotal);
	   
	   PARTModel(conTrain,conTest,conTotal);
	   PARTModel(train, test, total);
	   	   

	}
	
	
	public static void PARTModel(String train, String test, String total) {
		
		  Instances trainData = null;
		  Instances testData = null;
		  Instances totalData = null;
		  PART pa = new PART();
		  
		try {
			DataSource source = new DataSource(train);
	    	trainData = source.getDataSet();
			int num_Attributes = trainData.numAttributes() - 1;
		    trainData.setClassIndex(num_Attributes);
			//apri.buildAssociations(trainData);
			pa.buildClassifier(trainData);
			Evaluation eval = new Evaluation(trainData);
			
		    //Read the test data
		    DataSource testSource = new DataSource(test);
		    testData = testSource.getDataSet();
		    testData.setClassIndex(num_Attributes);
			//System.out.println(pa);
			
		    //Evaluate the test data using the PART algorithm
		    eval.evaluateModel(pa, testData);
		    System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		   
		    
		    eval.crossValidateModel(pa, testData, 10, new Random());
		    System.out.println(eval.toSummaryString("\nResults\n======\n", false));
			   		 
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
	public static void j48Model(String train, String test, String total) {
		   Instances trainData = null;
		   Instances testData = null;
		   Instances totalData = null;
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
			    
			    //Evaluate the test data using the j48 model
			    eval.evaluateModel(j, testData);
			    System.out.println(eval.toSummaryString("\nResults\n======\n", false));
			    
			    //Cross-validate instead of train-test model
			    DataSource bigSource = new DataSource(total);
			    totalData = bigSource.getDataSet();
			    totalData.setClassIndex(num_Attributes);
			    Evaluation eval2 = new Evaluation(totalData);
			    eval2.crossValidateModel(j, totalData, 10, new Random());
			    System.out.println(eval2.toSummaryString("\nResults\n======\n", false));
			//Catches various errors that may come up during the reading of the file     
			} catch (Exception e) {
				e.printStackTrace();
			}

			
	}
	
}
