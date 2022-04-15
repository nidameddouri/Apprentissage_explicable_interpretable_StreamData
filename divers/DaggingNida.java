 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * Dagging.java
 * Copyright (C) 2005 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.meta;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.RandomizableSingleClassifierEnhancer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;

/**
 <!-- globalinfo-start -->
 * This meta classifier creates a number of disjoint, stratified folds out of the data and feeds each chunk of data to a copy of the supplied base classifier. Predictions are made via majority vote, since all the generated base classifiers are put into the Vote meta classifier. <br/>
 * Useful for base classifiers that are quadratic or worse in time behavior, regarding number of instances in the training data. <br/>
 * <br/>
 * For more information, see: <br/>
 * Ting, K. M., Witten, I. H.: Stacking Bagged and Dagged Models. In: Fourteenth international Conference on Machine Learning, San Francisco, CA, 367-375, 1997.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{Ting1997,
 *    address = {San Francisco, CA},
 *    author = {Ting, K. M. and Witten, I. H.},
 *    booktitle = {Fourteenth international Conference on Machine Learning},
 *    editor = {D. H. Fisher},
 *    pages = {367-375},
 *    publisher = {Morgan Kaufmann Publishers},
 *    title = {Stacking Bagged and Dagged Models},
 *    year = {1997}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -F &lt;folds&gt;
 *  The number of folds for splitting the training set into
 *  smaller chunks for the base classifier.
 *  (default 10)</pre>
 * 
 * <pre> -verbose
 *  Whether to print some more information during building the
 *  classifier.
 *  (default is off)</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.functions.SMO)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.functions.SMO:
 * </pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -no-checks
 *  Turns off all checks - use with caution!
 *  Turning them off assumes that data is purely numeric, doesn't
 *  contain any missing values, and has a nominal class. Turning them
 *  off also means that no header information will be stored if the
 *  machine is linear. Finally, it also assumes that no instance has
 *  a weight equal to 0.
 *  (default: checks on)</pre>
 * 
 * <pre> -C &lt;double&gt;
 *  The complexity constant C. (default 1)</pre>
 * 
 * <pre> -N
 *  Whether to 0=normalize/1=standardize/2=neither. (default 0=normalize)</pre>
 * 
 * <pre> -L &lt;double&gt;
 *  The tolerance parameter. (default 1.0e-3)</pre>
 * 
 * <pre> -P &lt;double&gt;
 *  The epsilon for round-off error. (default 1.0e-12)</pre>
 * 
 * <pre> -M
 *  Fit logistic models to SVM outputs. </pre>
 * 
 * <pre> -V &lt;double&gt;
 *  The number of folds for the internal
 *  cross-validation. (default -1, use training data)</pre>
 * 
 * <pre> -W &lt;double&gt;
 *  The random number seed. (default 1)</pre>
 * 
 * <pre> -K &lt;classname and parameters&gt;
 *  The Kernel to use.
 *  (default: weka.classifiers.functions.supportVector.PolyKernel)</pre>
 * 
 * <pre> 
 * Options specific to kernel weka.classifiers.functions.supportVector.PolyKernel:
 * </pre>
 * 
 * <pre> -D
 *  Enables debugging output (if available) to be printed.
 *  (default: off)</pre>
 * 
 * <pre> -no-checks
 *  Turns off all checks - use with caution!
 *  (default: checks on)</pre>
 * 
 * <pre> -C &lt;num&gt;
 *  The size of the cache (a prime number), 0 for full cache and 
 *  -1 to turn it off.
 *  (default: 250007)</pre>
 * 
 * <pre> -E &lt;num&gt;
 *  The Exponent to use.
 *  (default: 1.0)</pre>
 * 
 * <pre> -L
 *  Use lower-order terms.
 *  (default: no)</pre>
 * 
 <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p/>
 *
 * @author Bernhard Pfahringer (bernhard at cs dot waikato dot ac dot nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5305 $
 * @see       Vote
 */
public class DaggingNida
  extends RandomizableSingleClassifierEnhancer
  implements TechnicalInformationHandler {
  
  /** for serialization */
  static final long serialVersionUID = 4560165876570074309L;

  /** the number of folds to use to split the training data */
  //protected int m_NumFolds = 10; // Default value
  protected int m_NumFolds = 11; // Ideal to use CNC

  /** the classifier used for voting */
  protected Vote m_Vote = null;

  /** whether to output some progress information during building */
  protected boolean m_Verbose = false;
    
  /** Pour un affichage avancè */
  DecimalFormat df = new DecimalFormat("000.0000");
  
  /** Vecteur contenant les prédictions de chaque classifieur faible 
   * pour tout les instances du contexte (nb_it*nb_inst) 
   */ 
  public static ArrayList <ArrayList <Integer>> prediction_byWeak_it;
  
  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return 
     "This meta classifier creates a number of disjoint, stratified folds out "
     + "of the data and feeds each chunk of data to a copy of the supplied "
     + "base classifier. Predictions are made via averaging, since all the "
     + "generated base classifiers are put into the Vote meta classifier. \n"
     + "Useful for base classifiers that are quadratic or worse in time "
     + "behavior, regarding number of instances in the training data. \n"
     + "\n"
     + "For more information, see: \n"
     + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.INPROCEEDINGS);
    result.setValue(Field.AUTHOR, "Ting, K. M. and Witten, I. H.");
    result.setValue(Field.TITLE, "Stacking Bagged and Dagged Models");
    result.setValue(Field.BOOKTITLE, "Fourteenth international Conference on Machine Learning");
    result.setValue(Field.EDITOR, "D. H. Fisher");
    result.setValue(Field.YEAR, "1997");
    result.setValue(Field.PAGES, "367-375");
    result.setValue(Field.PUBLISHER, "Morgan Kaufmann Publishers");
    result.setValue(Field.ADDRESS, "San Francisco, CA");
    
    return result;
  }
    
  /**
   * Constructor.
   */
  public DaggingNida() {
    m_Classifier = new weka.classifiers.fca.CNC();
    //m_Classifier = new weka.classifiers.functions.SMO();
  }

  /**
   * String describing default classifier.
   * 
   * @return the default classifier classname
   */
  protected String defaultClassifierString() {
    //return weka.classifiers.functions.SMO.class.getName();
	return weka.classifiers.fca.CNC.class.getName();
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {
    Vector result = new Vector();
    
    result.addElement(new Option(
        "\tThe number of folds for splitting the training set into\n"
        + "\tsmaller chunks for the base classifier.\n"
        + "\t(default 10)",
        "F", 1, "-F <folds>"));
    
    result.addElement(new Option(
        "\tWhether to print some more information during building the\n"
        + "\tclassifier.\n"
        + "\t(default is off)",
        "verbose", 0, "-verbose"));
    
    Enumeration en = super.listOptions();
    while (en.hasMoreElements())
      result.addElement(en.nextElement());
      
    return result.elements();
  }


  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -F &lt;folds&gt;
   *  The number of folds for splitting the training set into
   *  smaller chunks for the base classifier.
   *  (default 10)</pre>
   * 
   * <pre> -verbose
   *  Whether to print some more information during building the
   *  classifier.
   *  (default is off)</pre>
   * 
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.functions.SMO)</pre>
   * 
   * <pre> 
   * Options specific to classifier weka.classifiers.functions.SMO:
   * </pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   * <pre> -no-checks
   *  Turns off all checks - use with caution!
   *  Turning them off assumes that data is purely numeric, doesn't
   *  contain any missing values, and has a nominal class. Turning them
   *  off also means that no header information will be stored if the
   *  machine is linear. Finally, it also assumes that no instance has
   *  a weight equal to 0.
   *  (default: checks on)</pre>
   * 
   * <pre> -C &lt;double&gt;
   *  The complexity constant C. (default 1)</pre>
   * 
   * <pre> -N
   *  Whether to 0=normalize/1=standardize/2=neither. (default 0=normalize)</pre>
   * 
   * <pre> -L &lt;double&gt;
   *  The tolerance parameter. (default 1.0e-3)</pre>
   * 
   * <pre> -P &lt;double&gt;
   *  The epsilon for round-off error. (default 1.0e-12)</pre>
   * 
   * <pre> -M
   *  Fit logistic models to SVM outputs. </pre>
   * 
   * <pre> -V &lt;double&gt;
   *  The number of folds for the internal
   *  cross-validation. (default -1, use training data)</pre>
   * 
   * <pre> -W &lt;double&gt;
   *  The random number seed. (default 1)</pre>
   * 
   * <pre> -K &lt;classname and parameters&gt;
   *  The Kernel to use.
   *  (default: weka.classifiers.functions.supportVector.PolyKernel)</pre>
   * 
   * <pre> 
   * Options specific to kernel weka.classifiers.functions.supportVector.PolyKernel:
   * </pre>
   * 
   * <pre> -D
   *  Enables debugging output (if available) to be printed.
   *  (default: off)</pre>
   * 
   * <pre> -no-checks
   *  Turns off all checks - use with caution!
   *  (default: checks on)</pre>
   * 
   * <pre> -C &lt;num&gt;
   *  The size of the cache (a prime number), 0 for full cache and 
   *  -1 to turn it off.
   *  (default: 250007)</pre>
   * 
   * <pre> -E &lt;num&gt;
   *  The Exponent to use.
   *  (default: 1.0)</pre>
   * 
   * <pre> -L
   *  Use lower-order terms.
   *  (default: no)</pre>
   * 
   <!-- options-end -->
   *
   * Options after -- are passed to the designated classifier.<p>
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    String        tmpStr;

    tmpStr = Utils.getOption('F', options);
    if (tmpStr.length() != 0)
      setNumFolds(Integer.parseInt(tmpStr));
    else
      setNumFolds(10);
    
    setVerbose(Utils.getFlag("verbose", options));
    
    super.setOptions(options);
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String[] getOptions() {
    Vector        result;
    String[]      options;
    int           i;
    
    result  = new Vector();

    result.add("-F");
    result.add("" + getNumFolds());
    
    if (getVerbose())
      result.add("-verbose");
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    return (String[]) result.toArray(new String[result.size()]);
  }

  /**
   * Gets the number of folds to use for splitting the training set.
   *
   * @return the number of folds
   */
  public int getNumFolds() {
    return m_NumFolds;
  }
  
  /**
   * Sets the number of folds to use for splitting the training set.
   *
   * @param value     the new number of folds
   */
  public void setNumFolds(int value) {
    if (value > 0)
      m_NumFolds = value;
    else
      System.out.println(
          "At least 1 fold is necessary (provided: " + value + ")!");
  }

  /**
   * Returns the tip text for this property
   *
   * @return tip text for this property suitable for
   *         displaying in the explorer/experimenter gui
   */
  public String numFoldsTipText() {
    return "The number of folds to use for splitting the training set into smaller chunks for the base classifier.";
  }
  
  /**
   * Set the verbose state.
   *
   * @param value the verbose state
   */
  public void setVerbose(boolean value) {
    m_Verbose = value;
  }
  
  /**
   * Gets the verbose state
   *
   * @return the verbose state
   */
  public boolean getVerbose() {
    return m_Verbose;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String verboseTipText() {
    return "Whether to ouput some additional information during building.";
  }

  /**
   * Bagging method.
   *
   * @param data the training data to be used for generating the
   * bagged classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances data) throws Exception {

	if (m_Debug)
		System.err.println("\n\nDagging: Build Classifier "+ this.getClassifierSpec() + " on "+ data.relationName());
	  	  
    Classifier[]        base;
    int                 i;
    int                 n;
    int                 fromIndex;
    int                 toIndex;
    Instances           train;
    double              chunkSize;

    
    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    m_Vote    = new Vote();
    base      = new Classifier[getNumFolds()];
    chunkSize = (double) data.numInstances() / (double) getNumFolds();
    
    // stratify data
    if (getNumFolds() > 1) {
      data.randomize(data.getRandomNumberGenerator(getSeed()));
      data.stratify(getNumFolds());
    }
    
    // Nida Meddouri: 
    Evaluation evaluation;
    // Vecteur contenant les prédictions de chaque classifieur faible pour tout les instances du contexte d'apprentissage
    if (m_Debug){
    	prediction_byWeak_it = new ArrayList <ArrayList <Integer>> ();
    }

    // generate <folds> classifiers
    for (i = 0; i < getNumFolds(); i++) 
    {
      base[i] = makeCopy(getClassifier());

      // generate training data
      if (getNumFolds() > 1) 
      {
        // some progress information
        if (getVerbose())
          System.out.print(".");
        
        train     = data.testCV(getNumFolds(), i);
      }
      else 
      {
        train = data;
      }

      // train classifier
      base[i].buildClassifier(train);
      
      if (m_Debug)
      {
    	  evaluation = new Evaluation(data);
          evaluation.evaluateModel(base[i],data);
          
          //System.err.println(m_Classifiers[j].toString());
          
	      // Les predictions du classifieur faible courant pour chaque instance du contexte d'apprentissage
	      ArrayList <Integer> prediction_inst_byWeak = new ArrayList <Integer>();
	  	  prediction_inst_byWeak.clear();
	      for(int k=0; k<data.numInstances();k++)
	      {
	    	  if(data.instance(k).classValue() == evaluation.evaluateModelOnceAndRecordPrediction(base[i], data.instance(k)))
	    		  prediction_inst_byWeak.add(1); 
	    	  else
	    		  prediction_inst_byWeak.add(0);
	      }
      
	      prediction_byWeak_it.add(prediction_inst_byWeak);
      	  //System.out.println("Training classifier " + (j+1) + " Classification OutPut: "+prediction_inst_byWeak.toString());
      
	      ArrayList <ArrayList<Double>> abcd = new ArrayList <ArrayList<Double>>();
	  	  abcd = DiversityMeasure(prediction_byWeak_it);
  		  System.out.println((i+1) +"\t[DagData: "+ train.numInstances()+"]\t"
  				+"evaluation.errorRate(): "+df.format(evaluation.errorRate()*100)+"\t"
  				+"evaluation.pctCorrect(): "+df.format(evaluation.pctCorrect())+"\t"
  	    		+"Q: "+ df.format(Q_DivMeas(abcd))
  	    		+"\tCC: "+ df.format(CC_DivMeas(abcd))
  	    		+"\tkp: "+ df.format(kp_DivMeas(abcd))
  	    		//+DataSetSpecification(train)
  	    		);

  		  /*if(i == base.length-1)
  		  System.out.println(
  				  "\n\nDagging of "+ i +": Build Classifier "+ this.getClassifierSpec() + " on "+ data.relationName()
  				    +"\n[ "+ i +"% of DagData: "+ train.numInstances()+"]\t"
  		    		+"evaluation.errorRate(): "+df.format(evaluation.errorRate()*100)
  		    		+"\tQ: "+ df.format(Q_DivMeas(abcd))
  		    		+"\tCC: "+ df.format(CC_DivMeas(abcd))
  		    		+"\tkp: "+ df.format(kp_DivMeas(abcd))
  		    		//+DataSetSpecification(bagData)
  		    		);
  		  */
      }
    }
    
    // init vote
    m_Vote.setClassifiers(base);
    
    if (getVerbose())
      System.out.println();  
  }

  /**
   * Calculates the class membership probabilities for the given test
   * instance.
   *
   * @param instance the instance to be classified
   * @return predicted class probability distribution
   * @throws Exception if distribution can't be computed successfully 
   */
  public double[] distributionForInstance(Instance instance) throws Exception {
    return m_Vote.distributionForInstance(instance);
  }

  /**
   * Returns description of the classifier.
   *
   * @return description of the classifier as a string
   */
  public String toString() {
    if (m_Vote == null)
      return this.getClass().getName().replaceAll(".*\\.", "") 
             + ": No model built yet.";
    else
      return m_Vote.toString();
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 201510615 $");
  }

  /**
   * Main method for testing this class.
   *
   * @param args the options
   */
  public static void main(String[] args) {
    runClassifier(new DaggingNida(), args);
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  public final ArrayList <ArrayList <Double>> DiversityMeasure(ArrayList <ArrayList <Integer>> All_result_classif)
  {
	  ArrayList <ArrayList <Double>> abcd = new ArrayList <ArrayList <Double>>();
	  for (int i=0; i<(int) All_result_classif.size();i++)
		  for(int j=i+1; j<(int) All_result_classif.size();j++)
		  {
			  ArrayList <Double> temp_abcd = new ArrayList <Double>();
			  temp_abcd = pairwiseABCD(All_result_classif.get(i), All_result_classif.get(j));
			  abcd.add(temp_abcd);
		  }
	  return abcd;	  
  }
  
  public final ArrayList <Double> Calcul_OccuracyClassif(ArrayList <ArrayList <Integer>> All_result_classif)
  {
	  ArrayList <Double> each_occuracy_classif = new ArrayList <Double>();
	  int Compteur_Classif = 0;
	  while(Compteur_Classif < All_result_classif.get(0).size())
	  {
		  Integer Correctly_Classif = 0;
		  for(int i=0; i<(int) All_result_classif.size();i++)
		  {
			  if (All_result_classif.get(i).get(Compteur_Classif)==1)
				  Correctly_Classif ++;			  
		  }
		  Double occuracy = (Double) (Correctly_Classif / (double) All_result_classif.get(0).size());
		  each_occuracy_classif.add((Double)occuracy*100);
		  Compteur_Classif++;		  
	  }
	  
	  for(int i=0; i<each_occuracy_classif.size(); i++)
		  System.out.println(i+" * "+each_occuracy_classif.get(i));

	  ArrayList <Double> occuracy_classif = new ArrayList <Double>();
	  for (int i=0; i<(int) each_occuracy_classif.size();i++)
		  for(int j=i+1; j<(int) each_occuracy_classif.size();j++)
		  {
			  Double temp_occuracy_classif = (Double) ((each_occuracy_classif.get(i) + each_occuracy_classif.get(j))/2);			  
			  occuracy_classif.add(temp_occuracy_classif);
		  }
	  
//	  for(int i=0; i<occuracy_classif.size(); i++)
//		  System.out.println(i+" *** "+occuracy_classif.get(i));
	  
	  return occuracy_classif;	  
  }
   
  /**
   * Returns value of Q statistic if class is nominal.
   *
   * @return the value of the Q statistic
   */

  public final Double Q_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumQ2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumQ2+=Q2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumQ2/abcd.size());			  
  }

  public final Double Q2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double ad = (Double) abcdIJ.get(0) * abcdIJ.get(3);
	  Double bc = (Double) abcdIJ.get(1) * abcdIJ.get(2);
	  Double tempQ2;
	  if(ad==0.00000000 && bc==0.00000000)
		  tempQ2 = new Double(0);
	  else
	  	  tempQ2 = (Double)(ad-bc)/(ad+bc);

	  return tempQ2;
  }
  
  /**
   * Returns value of the Correlation coefficient if class is nominal.
   *
   * @return the value of the Correlation coefficient
   */

  public final Double CC_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumCC2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumCC2+=CC2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumCC2/abcd.size());			  
  }

  public final Double CC2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempCC2;
	  Double ad = (Double) abcdIJ.get(0) * abcdIJ.get(3);
	  Double bc = (Double) abcdIJ.get(1) * abcdIJ.get(2);
	  Double aPLUSb = (Double) abcdIJ.get(0) + abcdIJ.get(1);
	  Double cPLUSd = (Double) abcdIJ.get(2) + abcdIJ.get(3);
	  Double aPLUSc = (Double) abcdIJ.get(0) + abcdIJ.get(2);
	  Double bPLUSd = (Double) abcdIJ.get(1) + abcdIJ.get(3);
	  Double multiplication = (Double) aPLUSb * cPLUSd * aPLUSc * bPLUSd;
	  if(multiplication==0.00000000)
		  tempCC2 = new Double(0);
	  else
		  tempCC2 = (Double) (ad-bc)/Math.sqrt(multiplication);
//	  System.out.println(multiplication+"  --  "+ Math.sqrt(multiplication) +"  --  "+ (Math.sqrt(multiplication)*Math.sqrt(multiplication)) );
	  return tempCC2;
  }

  /**
   * Returns value of Disagreement if class is nominal.
   *
   * @return the value of the Disagreement
   */

  public final Double Disagreement_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumDisagreement2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumDisagreement2+=CC2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumDisagreement2/abcd.size());			  
  }

  public final Double Disagreement2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempDisagreement2;
	  tempDisagreement2 = (Double) abcdIJ.get(1)+abcdIJ.get(2);
	  return tempDisagreement2;
  }
  
  /**
   * Returns value of Double-fault if class is nominal.
   *
   * @return the value of the Double-fault
   */

  public final Double DoubleFault_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  int cptDF=0;
	  Double SumDF2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumDF2+=abcd.get(i).get(3);
		  cptDF++;			  
	  }
	  return ((Double) SumDF2/cptDF);			  
  }

  /**
   * Returns value of Kohavi-Wolpert variance if class is nominal.
   *
   * @return the value of the Kohavi-Wolpert variance
   */

  public final Double KW_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumKW2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumKW2+=KW2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumKW2/abcd.size());			  
  }

  public final Double KW2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempKW2;
	  tempKW2 = (Double) (abcdIJ.get(1)+abcdIJ.get(2))/4 ;
	  return tempKW2;
  }

  /**
   * Returns value of the kappa if class is nominal.
   *
   * @return the value of the kappa
   */

  public final Double kappa_DivMeas(ArrayList <ArrayList <Double>> abcd, ArrayList <Double> occuracy_classif) {	  	  
	  Double Sumkappa=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
//		  System.out.print("occuracy_classif: "+occuracy_classif.get(i));
		  Sumkappa+=kappa2_DivMeas(abcd.get(i),occuracy_classif.get(i));
	  }
	  return ((Double) Sumkappa/abcd.size());			  
  }

  public final Double kappa2_DivMeas(ArrayList <Double> abcdIJ, Double occuracy_classif_IJ)
  {
	  Double tempkappa2;
	  Double bPLUSc = (Double) abcdIJ.get(1) + abcdIJ.get(2);
//	  System.out.println(bPLUSc);
	  
	  Double multiplication = (Double) (2 * occuracy_classif_IJ * (1 - occuracy_classif_IJ));
//	  System.out.print("    multiplication: "+multiplication);
	  if(multiplication==0.00000000)
		  tempkappa2 = new Double(1);
	  else
		  tempkappa2 = (Double) (1 - ((bPLUSc)/multiplication));
//	  System.out.println("     kappa: "+tempkappa2);
	  return tempkappa2;
  }
  
  /**
   * Returns value of the Measurement of interrater agreement if class is nominal.
   *
   * @return the value of the Measurement of interrater agreement
   */

  public final Double kp_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double Sumkp2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  Sumkp2+=kp2_DivMeas(abcd.get(i));
	  }
	  return ((Double) Sumkp2/abcd.size());			  
  }

  public final Double kp2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempkp2;
	  Double ad = (Double) abcdIJ.get(0) * abcdIJ.get(3);
	  Double bc = (Double) abcdIJ.get(1) * abcdIJ.get(2);
	  Double aPLUSb = (Double) abcdIJ.get(0) + abcdIJ.get(1);
	  Double cPLUSd = (Double) abcdIJ.get(2) + abcdIJ.get(3);
	  Double aPLUSc = (Double) abcdIJ.get(0) + abcdIJ.get(2);
	  Double bPLUSd = (Double) abcdIJ.get(1) + abcdIJ.get(3);
	  Double multiplication = (Double) aPLUSb * cPLUSd * aPLUSc * bPLUSd;
	  if(multiplication==0.00000000)
		  tempkp2 = new Double(0);
	  else
		  tempkp2 = (Double) (2*(ad-bc))/multiplication;
	  return tempkp2;
  }
  
  /**
   * Returns value of Entropy if class is nominal.
   *
   * @return the value of the Entropy
   */

  public final Double Entropy_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumEntropy2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumEntropy2+=Entropy2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumEntropy2/abcd.size());			  
  }

  public final Double Entropy2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempEntropy2 = (Double) (abcdIJ.get(1)+abcdIJ.get(2))/2;
	  return tempEntropy2;
  }
  
  /**
   * Returns value of the Difficulty if class is nominal.
   *
   * @return the value of the Difficulty
   */

  public final Double Difficulty_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumDifficulty2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumDifficulty2+=Difficulty2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumDifficulty2/abcd.size());			  
  }

  public final Double Difficulty2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempDifficulty2;
	  Double ad = (Double) abcdIJ.get(0) * abcdIJ.get(3);
	  Double aPLUSd = (Double) abcdIJ.get(0) + abcdIJ.get(3);
	  Double bPLUSc = (Double) abcdIJ.get(1) + abcdIJ.get(2);
	  tempDifficulty2 = (Double) (ad+((aPLUSd*bPLUSc)/4));
	  return tempDifficulty2;
  }
  
  /**
   * Returns value of the Generalized diversity if class is nominal.
   *
   * @return the value of the Generalized diversity
   */

  public final Double GD_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumGD2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumGD2+=GD2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumGD2/abcd.size());			  
  }

  public final Double GD2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempGD2;
	  Double bPLUSc = (Double) abcdIJ.get(1) + abcdIJ.get(2);
	  Double aPLUSd = (Double) abcdIJ.get(0) + abcdIJ.get(3);
	  Double quotion = (Double) (1 - aPLUSd);
	  if(quotion == 0.0000000)
		  tempGD2 = new Double (0);
	  else
		  tempGD2 = (Double) (bPLUSc/quotion);
	  return tempGD2;
  }
  
  /**
   * Returns value of the Coincident Failure diversity if class is nominal.
   *
   * @return the value of the Coincident Failure diversity
   */

  public final Double CFD_DivMeas(ArrayList <ArrayList <Double>> abcd) {
	  Double SumCFD2=new Double(0);
	  for (int i=0; i<(int) abcd.size();i++)
	  {
		  SumCFD2+=CFD2_DivMeas(abcd.get(i));
	  }
	  return ((Double) SumCFD2/abcd.size());			  
  }

  public final Double CFD2_DivMeas(ArrayList <Double> abcdIJ)
  {
	  Double tempCFD2;
	  Double bPLUSc = (Double) abcdIJ.get(1) + abcdIJ.get(2);
	  Double quotion = (Double) (1 - abcdIJ.get(0));
	  if(quotion == 0.0000000)
		  tempCFD2 = new Double (0);
	  else
		  tempCFD2 = (Double) (bPLUSc/quotion);
	  return tempCFD2;
  }  

  public final ArrayList <Double> pairwiseABCD(ArrayList <Integer> OPbinCLASS1, ArrayList <Integer> OPbinCLASS2)
  {
	  Integer a=0, b=0, c=0, d=0;
	  for(int i =0; i<(int) OPbinCLASS1.size();i++)
	  {
		  if (OPbinCLASS1.get(i)==OPbinCLASS2.get(i))
			  if(OPbinCLASS1.get(i)== 1)
				  a++;
			  else 
				  d++;
		  else 
			  if(OPbinCLASS1.get(i)== 1) 
				  b++; 
			  else 
				  c++;
	  }
	  ArrayList <Double> abcd = new ArrayList <Double>();
	  abcd.add((double) a/OPbinCLASS1.size());
	  abcd.add((double) b/OPbinCLASS1.size());
	  abcd.add((double) c/OPbinCLASS1.size());
	  abcd.add((double) d/OPbinCLASS1.size());
	  
	  return abcd;
	  
  }
   
}
