package moa.classifiers.meta;

import java.util.Random;

import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.rules.core.voting.Vote;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;
import moa.classifiers.Classifier;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

public class Dagging extends AbstractClassifier implements MultiClassClassifier, CapabilitiesHandler {

	/**
	 * Default Serial Version
	 */
	private static final long serialVersionUID = 1L;

	/* the number of folds to use to split the training data */
	protected static int m_numFolds = 10; // nombre de sous classifiers
	protected static Classifier[] m_ensemble; // list contenant les sous classifiers
	private static Instances instances; // l'instances utilisé par le classifieur
	private static int m_batch = 100; // nombre d'instance dans chaque Instances
	private static boolean m_Debug = false; // affiche ou non la trace
	private static Classifier baseLearner; // type de classifier selectionné
	private static boolean m_entrainement = false; // indicateur si

	@Override
	public String getPurposeString() {
		return "Dagging for evolving data streams.";
	}

	// le learner selectionné
	public ClassOption baseLearnerOption = new ClassOption("BaseLearner", 'l', "Classifier to train.", Classifier.class,
			"fca.CNC");

	// taille du Bag
	public IntOption numFoldOption = new IntOption("NumFolds", 'B',
			"The number of folds to use for splitting the training set into smaller chunks for the base classifier.",
			11, 5, Integer.MAX_VALUE);

	// taille de l'instances
	public IntOption batchSizeOption = new IntOption("BatchSize", 'b',
			"The preferred number of instances to process if batch prediction is being performed.", 1000, 0,
			Integer.MAX_VALUE);

	// debug option
	public MultiChoiceOption debugOption = new MultiChoiceOption("Debug", 'd', "debug option.",
			new String[] { "False", "True" }, new String[] { "False", "True" }, 0);

	@Override
	public void resetLearningImpl() {
		m_Debug = Boolean.valueOf(this.debugOption.getChosenLabel());
		m_batch = this.batchSizeOption.getValue();
		m_numFolds = this.numFoldOption.getValue();
		m_ensemble = new Classifier[m_numFolds];
		baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
		baseLearner.resetLearning();

		if (m_Debug) {
			System.out.println(baseLearner.getModel().toString());
		}
		for (int i = 0; i < m_ensemble.length; i++) {
			m_ensemble[i] = baseLearner.copy();
		}

		m_entrainement = false;
		count = 0;
	}
	private static int count = 0;

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		//count++;
		//System.out.println(count);
		if (instances == null) {
			// construction de l'instances
			instances = new Instances(inst.dataset(), m_batch);
		} else {
			instances.add(inst);
		}
		if (instances.size() == m_batch) {
			m_entrainement = true;
			if (m_Debug)
				System.out.println(
						java.time.LocalDateTime.now() + " - Dagging: Classification " + baseLearner.getClass());

			Instances train = null;
			if (m_numFolds > 1) {
				instances.randomize(new Random());
				instances.stratify(m_numFolds);
			}

			for (int i = 0; i < m_numFolds; i++) {
				if (m_numFolds > 1) {
					train = instances.testCV(m_numFolds, i);
				} else {
					train = instances;
				}
				if (m_Debug)
					System.out.println("Nombre d'instance du Fold (" + i + ") : " + train.numInstances());
				for (int j = 0; j < train.size(); j++) {
					m_ensemble[i].trainOnInstance(train.get(j));
				}
			}
			if (m_Debug)
				System.out.println();
			instances = null;
		}
		if (m_Debug) {
			if (!m_entrainement) {
				System.out.println(" *** ensemble pas encore entrainé *** ");
			}
		}

	}

	// Afficher l'instance recuperer par le vote et sa décsision
	@Override
	public double[] getVotesForInstance(Instance inst) {
		if (m_Debug) {
			System.out.println(java.time.LocalDateTime.now() + " Instance a Tester : " + inst.toString());
		}
		DoubleVector vote = null;
		for (int i = 0; i < m_ensemble.length; i++) {
			vote = new DoubleVector(m_ensemble[i].getVotesForInstance(inst));
			if (m_Debug) {
				System.out.println("Prediction fold (" + i + ") : " + vote);
			}
		}
		if (m_Debug) {
			System.out.println();
		}
		return vote.getArrayRef();
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub

	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public ImmutableCapabilities defineImmutableCapabilities() {
		if (this.getClass() == Dagging.class)
			return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
		else
			return new ImmutableCapabilities(Capability.VIEW_STANDARD);
	}

}