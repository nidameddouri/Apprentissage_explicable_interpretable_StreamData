package moa.classifiers.fca;

import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
import moa.core.Utils;
import weka.core.ContingencyTables;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.Attribute;

import java.util.Calendar;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;

public class CNC extends AbstractClassifier implements MultiClassClassifier {

	Calendar calendar;
	SimpleDateFormat sdf = new SimpleDateFormat("E MM/dd/yyyy HH:mm:ss.SSS");

	private static final long serialVersionUID = 1L;
	// fermetures
	public static final int CONCEPT_LEARNING_Best_Value_Best_Attribute = 0; // Default: Fermeture de la meilleur valeur
																			// Nominal du meilleur attribut
	public static final int CONCEPT_LEARNING_All_Value_Best_Attribute = 1; // fermeture de toutes les valeurs du
																			// meilleur attribute
	public static final int CONCEPT_LEARNING_Best_Value_All_Attribute = 2; // fermeture de la meilleur valeur de tous
																			// les attributs
	public static final int CONCEPT_LEARNING_All_Value_All_Attribute = 3; // fermeture de toutes les valeurs de tous les
																			// attributs
	public static final int CONCEPT_LEARNING_Best_Value_All_Active_Attribute = 4; // fermeture de la meilleur valeur de
																					// tous les attributs actif (mesure
																					// > 0)
	public static final int CONCEPT_LEARNING_All_Value_All_Active_Attribute = 5; // fermeture de toutes les valeurs de
																					// tous les attributs actif (mesure
																					// > 0)
	private static int NominalConceptLearning = CONCEPT_LEARNING_Best_Value_Best_Attribute;

	// mesures
	public static final int PERTINENCE_MEASURE_Gain_Info = 0; // Default: Mesure qui maximise le gain informationnel
	public static final int PERTINENCE_MEASURE_Gain_Ratio = 1; // Mesure qui maximise le gain rationnel
	public static final int PERTINENCE_MEASURE_Correlation = 2; // Mesure qui maximise la correlation attribut - eval
	public static final int PERTINENCE_MEASURE_HRatio = 3; // Mesure qui maximise le gain H-Ratio
	public static final int PERTINENCE_MEASURE_Info_Mutuelle = 4; // Mesure qui maximise le Info_Mutuelle
	public static final int PERTINENCE_MEASURE_Symmetrical = 5; // les valeurs nominales qui maximise de symmetrical
	public static final int PERTINENCE_MEASURE_Entropy = 6; // Mesure qui minimise l'entropy
	private static int PertinenceMeasure = PERTINENCE_MEASURE_Gain_Info;

	// votes
	public static final int Vote_Maj = 0; // Default: Vote majoritaire
	public static final int Vote_Plur = 1; // Vote pluralité
	private static int VoteMethods = Vote_Maj;

	// Variables générales
	private static ArrayList<ArrayList<Classification_Rule>> m_FinalClassifierNC = new ArrayList<>(); // liste contenant
																										// toutes les

	private static Instances instances; // L'instances utilisé par le classifieur
	private static boolean needFilter = false;
	private static boolean m_Debug = false; // affiche ou non la trace
	private static boolean m_DebugTrain = false;
	private static boolean m_DebugDisplay = false;

	private static int m_Batch = 100; // nombre d'instance dans chaque Instances
										// chaque class
  
	private static MOADiscretize m_Filter = null;

	/*
	 * Description de la méthode
	 */
	public String getPurposeString() {
		return "CNC: Classifier Nominal Concepts to discover classification Rules.";
	}

	/*
	 * Options paramétrable
	 */
	public MultiChoiceOption Pertinence_MeasureOption = new MultiChoiceOption("PertinenceMeasure", 'F',
			"Pertinence Measure for Best Value Prediction.",
			new String[] { "Gain Info", "Gain Ratio", "Correlation", "H-RATIO", "Info Mutuelle", "Symmetrical",
					"Entropy" },
			new String[] { "PERTINENCE_MEASURE_Gain_Info", "PERTINENCE_MEASURE_Gain_Ratio",
					"PERTINENCE_MEASURE_Correlation", "PERTINENCE_MEASURE_HRatio", "PERTINENCE_MEASURE_Info_Mutuelle",
					"PERTINENCE_MEASURE_Symmetrical", "PERTINENCE_MEASURE_Entropy" },
			0);

	// choix des fermetures
	public MultiChoiceOption conceptLearningOption = new MultiChoiceOption("ConceptLearning", 'c',
			"Closure of best nominal attribut.",
			new String[] { "closure best value of pertinent attribute", "closure all values of pertinent attribute",
					"closure best value of all attributes", "closure all values of all attributes",
					"closure best value of all active attributes", "closure all values of all active attributes" },
			new String[] { "CONCEPT_LEARNING_Best_Value_Best_Attribute", "CONCEPT_LEARNING_All_Value_Best_Attribute",
					"CONCEPT_LEARNING_Best_Value_All_Attribute", "CONCEPT_LEARNING_All_Value_All_Attribute",
					"CONCEPT_LEARNING_Best_Value_All_Active_Attribute",
					"CONCEPT_LEARNING_All_Value_All_Active_Attribute" },
			0);

	// taille de l'instances
	public IntOption batchSizeOption = new IntOption("InstancesSize", 'b',
			"Prefer to set it to the same size as sampleFrequency", 100, 0, Integer.MAX_VALUE);
	// debug option
	public MultiChoiceOption debugOption = new MultiChoiceOption("Debug", 'd', "debug option.",
			new String[] { "False", "True", "TrueWithGetVotes", "TrueWithDisplayRules" },
			new String[] { "False", "True", "TrueWithGetVotes", "TrueWithDisplayRules" }, 0);
	// choix du vote
	public MultiChoiceOption vote_MethodsOption = new MultiChoiceOption("Vote_Methods", 'v', "vote_Methods option.",
			new String[] { "Majority Vote", "Plurality Vote" }, new String[] { "Majority Vote", "Plurality Vote" }, 0);
	
	// discretize option
		public MultiChoiceOption filterOption = new MultiChoiceOption("Discretize", 'z', "Discretize option.",
				new String[] { "False", "True" }, new String[] { "False", "True" }, 0);

	/*
	 * reset / initialise l'environnement du classifier
	 */
	@Override
	public void resetLearningImpl() {
		PertinenceMeasure = this.Pertinence_MeasureOption.getChosenIndex();
		m_Batch = this.batchSizeOption.getValue();
		NominalConceptLearning = this.conceptLearningOption.getChosenIndex();
		
		if (this.debugOption.getChosenIndex() > 0) {
			m_Debug = true;
		} else {
			m_Debug = false;
		}
		if (this.debugOption.getChosenIndex() == 2) {
			m_DebugTrain = true;
		} else {
			m_DebugTrain = false;
		}
		if (this.debugOption.getChosenIndex() == 3) {
			m_DebugDisplay = true;
		} else {
			m_DebugDisplay = false;
		}
		VoteMethods = this.vote_MethodsOption.getChosenIndex();
		needFilter = Boolean.valueOf(this.filterOption.getChosenLabel());
		instances = null;
		m_FinalClassifierNC.clear();
	}

	/*
	 * Realise une prediction sur les règles généré
	 */
	@Override
	public double[] getVotesForInstance(Instance inst) {
		if (needFilter) {
			if (m_Filter == null) {
				return new double[0];
			} else {
				try {
					inst = m_Filter.instanceFilter(inst);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}

		ArrayList<ArrayList<Classification_Rule>> regles = m_FinalClassifierNC;
		ArrayList<Classification_Rule> classRuls = new ArrayList<>(); // contient toutes les régles
		Map<Double, ArrayList<Classification_Rule>> regleGenererr = new HashMap<>(); // map qui contient les regles qui
																						// satisfaient l'instance

		// Récupérer les régles dans une liste
		for (int i = 0; i < regles.size(); i++) {
			for (int j = 0; j < regles.get(i).size(); j++) {
				classRuls.add(regles.get(i).get(j));
			}
		}

		// Parcourir les régles
		for (int i = 0; i < classRuls.size(); i++) {
			for (int j = 0; j < classRuls.get(i).getRule_Attr().size(); j++) {

				if (((classRuls.get(i).getRule_Attr().get(j) == "-null-")
						|| classRuls.get(i).getRule_Attr().get(j) == inst.attribute(j).value((int) inst.value(j)))) {
					if (j == classRuls.get(i).getRule_Attr().size() - 1) { // Dernier attribut de la régle
						if (!regleGenererr.containsKey(classRuls.get(i).getRule_indClassMaj())) {

							ArrayList<Classification_Rule> regleMap = new ArrayList<>();
							regleMap.add((classRuls.get(i)));
							regleGenererr.put(classRuls.get(i).Rule_indClassMaj, regleMap);
						} else {
							regleGenererr.get(classRuls.get(i).Rule_indClassMaj).add(classRuls.get(i));

						}
					}
				} else {
					break;
				}
			}
		}

		// Extraction d'une map qui contient nos class et la taille des listes qui
		// convient
		// a ces classes
		Map<Double, Integer> map = new HashMap<>();
		for (Map.Entry mapentry : regleGenererr.entrySet()) {
			ArrayList<Classification_Rule> tmp = new ArrayList<>();
			tmp.addAll((Collection<? extends Classification_Rule>) mapentry.getValue());
			map.put((Double) mapentry.getKey(), tmp.size());
		}

		double Classmajority = -1.0;
		int majority = 0;
		int pularity = 0;
		for (Map.Entry mapentry : map.entrySet()) {
			if ((int) mapentry.getValue() >= majority) {
				majority = (int) mapentry.getValue();
				Classmajority = (double) mapentry.getKey();
			}
			pularity += (int) mapentry.getValue();
		}
		if (majority == 0) {
			Classmajority = -1.0;
		}

		// affichage des regles dans la trace
		if (m_DebugTrain) {
			afficheInfoInst(inst);
			System.out.println();
			afficheRegle(regleGenererr);
			System.out.print("La class majoritaire correspondante : ");
		}
		switch (VoteMethods) {
		case 0:
			// majority
			if (m_DebugTrain) {
				System.out.println("\t" + Classmajority);
			}
			break;
		case 1:
			// plurality
			if ((pularity / 2) + 1 >= majority && majority != 0) {
				if (m_DebugTrain) {
					System.out.println("\t" + Classmajority);
				}
			} else {
				Classmajority = -1.0;
				if (m_DebugTrain) {
					System.out.println("\t" + Classmajority);
				}
			}
			break;
		default:
			break;
		}
		if (m_DebugTrain) {
			System.out.println();
		}
		
		double[] rsl = null;
		if (Classmajority == -1.0) {
			rsl = new double[0];
		} else {
			rsl = new double[inst.numClasses()];
			rsl[(int) Classmajority] = 1;
		}
		return rsl;
	}

	// fonction qui affiche les information de l'instance
	public void afficheInfoInst(Instance inst) {
		System.out.println(java.time.LocalTime.now() + " Instance a étiqueté : ");
		String chaine = "	";
		for (int i = 0; i < inst.numAttributes(); i++) {
			chaine += " " + inst.attribute(i).value((int) inst.value(i));
		}
		chaine += ".";
		System.out.println(chaine);
	}

	// fonction qui affiche les information des regles qui satisfaient l'instance
	public void afficheRegle(Map<Double, ArrayList<Classification_Rule>> regleGenererr) {
		ArrayList<Classification_Rule> tmp = new ArrayList<>();
		for (Map.Entry mapentry : regleGenererr.entrySet()) {
			tmp.addAll((Collection<? extends Classification_Rule>) mapentry.getValue());
		}
		if (tmp.size() > 0) {
			System.out.println("Les régles qui satisfaient l'instance :");

			for (int i = 0; i < tmp.size(); i++) {
				System.out.print("	" + java.time.LocalTime.now());
				System.out.println("	Régle " + i + " Attributs : " + tmp.get(i).getRule_Attr() + " : Class = "
						+ tmp.get(i).getRule_indClassMaj());
			}
		} else {
			System.out.println("Pas de regles qui satisfaient cette instance ");
		}

	}

	/*
	 * Appeler par MOA pour générer des règles
	 */

	@Override
	public void trainOnInstanceImpl(Instance inst) {

		if (instances == null) {
			// construction de l'instances
			instances = new Instances(inst.dataset(), m_Batch);
		} else {
			instances.add(inst);
		}
		if (instances.size() == m_Batch) {
			if (needFilter) {
				m_Filter = new MOADiscretize(instances);
				instances = m_Filter.getInstances();
			}
			
			// extraction de règles
			ArrayList<Classification_Rule> ERF = ExtraireRegleFermNom(instances, PertinenceMeasure);
			if (m_Debug) {
				System.out.println();
				System.out.println("----- Règles généré à : " + java.time.LocalTime.now() + " -----");
				for (int i = 0; i < ERF.size(); i++) {
					System.out.println("Affichage de la règle d'index : " + i);
					System.out.println("\tClasse Majoritaire : " + ERF.get(i).getRule_ClassMaj());
					System.out.println("\tAttributs : " + ERF.get(i).getRule_Attr()); // affichage de la liste des attr
																						// formant la règle
					writeGetRuleAttr(ERF.get(i), inst); // écriture d'une règle sous la forme "SI ... ET ... ALORS..."
				}
			}
			m_FinalClassifierNC.add(ERF);// toutes les règles sont rangé dans cette structure (m_FinalClassifierNC)
			instances = null;

			if (m_DebugDisplay) {
				System.out.println("--- --- --- ---");
				for (int i = 0; i < m_FinalClassifierNC.size(); i++) {
					for (int j = 0; j < m_FinalClassifierNC.get(i).size(); j++) {
						Classification_Rule regle = m_FinalClassifierNC.get(i).get(j);
						System.out.println(regle.getRule_Attr() + " : " + regle.Rule_ClassMaj);
					}
				}
				System.out.println();
			}
		}

	}

	// fonction bonus d'écriture d'une règle pour l'afficher en console
	private void writeGetRuleAttr(Classification_Rule classRule, Instance inst) {
		ArrayList<String> listeAttr = classRule.getRule_Attr();
		String chaine = "IF ";
		Boolean before = false;
		for (int i = 0; i < listeAttr.size(); i++) {
			if (!listeAttr.get(i).equals("-null-")) {
				if (i != 0 && before) {
					chaine += " AND ";
				}
				before = true;
				chaine += inst.attribute(i).name() + ":" + listeAttr.get(i);
			}
		}
		chaine += " THEN " + inst.classAttribute().name() + ":" + classRule.getRule_ClassMaj();
		System.out.println("\t" + chaine);
		System.out.println();
	}

	// méthode d'extraction de règles
	public ArrayList<Classification_Rule> ExtraireRegleFermNom(Instances inst, int critere) {
		if (m_Debug) {
			System.out.println(
					"****************************** Nouvelle Classification **********************************");
			System.out.println("Instances informations :");
			System.out.println("\tNombre instance : " + inst.size());
			System.out.print("\tListe des attributs :");
			for (int i = 0; i < inst.numAttributes(); i++) {
				System.out.print(
						"(" + (i + 1) + ")" + inst.attribute(i).name() + "[" + inst.attribute(i).numValues() + "] ");
			}
			System.out.println();
		}

		double[] measure = null;
		if (NominalConceptLearning == 2 || NominalConceptLearning == 3) {

		} else {
			switch (PertinenceMeasure) {
			case 0:
				// Compute attribute with maximum information gain
				measure = computeInfoGains(inst);
				break;

			case 1:
				// Compute attribute with maximum gain ratio
				measure = new double[inst.numAttributes()];
				for (int i = 0; i < inst.numAttributes() - 1; i++) {
					measure[i] = computeGainRatio(inst, i);
				}
				break;

			case 2:
				// Compute attribute with maximum correlation
				measure = computeCorrelation(inst);
				break;

			case 3:
				// Compute attribute with maximum HRATIO
				measure = computeHRatio(inst);
				break;

			case 4:
				// Compute attribute with maximum Info_Mutuelle
				measure = computeInfoMutuelle(inst);
				break;

			case 5:
				// Compute attribute with maximum Symmetrical
				measure = new double[inst.numAttributes()];
				for (int i = 0; i < inst.numAttributes() - 1; i++) {
					measure[i] = computeSymmetrical(inst, i);
				}
				break;

			case 6:
				// Compute attribute with minimum Entropy
				measure = new double[inst.numAttributes()];
				for (int i = 0; i < inst.numAttributes() - 1; i++) {
					measure[i] = computeEntropy(inst, i);
				}

				break;

			default:
				if (m_Debug)
					System.out.println("\npertinence measure default : No measure selected");
				break;
			}

			int index = 0;
			if (PertinenceMeasure == 6) {
				double[] newMeasure = new double[measure.length - 1];
				for (int i = 0; i < measure.length - 1; i++) {
					newMeasure[i] = measure[i];
				}
				measure = newMeasure;
				index = Utils.minIndex(measure);// selection de l'index de l'attribut qui minimise la mesure
			} else {
				index = Utils.maxIndex(measure);// selection de l'index de l'attribut qui maximise la mesure
			}
			Attribute m_Attribute = inst.attribute(index);// selection de l'attribut correspondant à l'index

			if (m_Debug) {
				System.out.println("\n" + java.time.LocalDateTime.now());
				System.out.println("Calcul des mesures de chaque attribut de ce context");
				for (int i = 0; i < inst.numAttributes() - 1; i++)
					System.out.println("\tMesure de l'attribut " + inst.attribute(i).name() + ": " + measure[i]);
			}
			if (m_Debug) {
				if (NominalConceptLearning < 2) {
					if (PertinenceMeasure == 6) {
						System.out.println("L'attribut retenu qui minimise la mesure de pertinence ("
								+ this.Pertinence_MeasureOption.getChosenLabel() + ") : " + m_Attribute.name());

					} else {
						System.out.println("L'attribut retenu qui maximise la mesure de pertinence ("
								+ this.Pertinence_MeasureOption.getChosenLabel() + ") : " + m_Attribute.name());
					}
					System.out.println("\tIndice de l'Attribut : " + index);
					System.out.println("\tNombre des differents valeurs possibles: " + numDistinctValues(inst, index));
				}
				System.out.println();
			}
		}

		/*
		 * créer une liste vide de classification
		 **/
		ArrayList<Classification_Rule> classifierNC = new ArrayList<Classification_Rule>();

		switch (NominalConceptLearning) {
		case 0:
			// best value best attribut
			classifierNC = conceptLearningBestValueBestAttribute(inst, measure);
			break;

		case 1:
			// all value best attribut
			classifierNC = conceptLearningAllValueBestAttribute(inst, measure);
			break;

		case 2:
			// best value all attribut
			classifierNC = conceptLearningBestValueAllAttribute(inst, measure);
			break;

		case 3:
			// all value all attribut
			classifierNC = conceptLearningAllValueAllAttribute(inst, measure);
			break;

		case 4:
			// best value all active attribut
			classifierNC = conceptLearningBestValueAllActiveAttribute(inst, measure);
			break;

		case 5:
			// all value all active attribut
			classifierNC = conceptLearningAllValueAllActiveAttribute(inst, measure);
			break;

		default:
			if (m_Debug)
				System.out.println("concept learning default : No concept learning selected");
			break;
		}
		return classifierNC;
	}

	// --------------------- FERMETURES

	// fermeture de la meilleur valeur du meilleur attribut
	public ArrayList<Classification_Rule> conceptLearningBestValueBestAttribute(Instances inst, double[] measure) {
		int index = 0;
		if (PertinenceMeasure == 6) {
			index = Utils.minIndex(measure);
		} else {
			index = Utils.maxIndex(measure);
		}
		Attribute m_Attribute = inst.attribute(index);

		ArrayList<Classification_Rule> classifierNC = new ArrayList<Classification_Rule>();
		int supportDistVal = 0;
		int indexBestDistVal = 0;
		int suppBestDistVal = 0;

		ArrayList<Integer> listCountDistVal = new ArrayList<Integer>();

		// parcours des valeurs distinctes
		for (int i = 0; i < numDistinctValues(inst, index); i++) {
			// Calcul du support de cette DistinctValue
			ArrayList<Integer> instDistVal = new ArrayList<Integer>();
			instDistVal.clear();

			supportDistVal = 0;
			for (int j = 0; j < inst.numInstances(); j++) {
				if (inst.instance(j).value(index) == inst.instance(i).value(index)) {
					supportDistVal++;
					instDistVal.add(j);
				}
			}
			if (suppBestDistVal <= supportDistVal) {
				suppBestDistVal = supportDistVal;
				indexBestDistVal = (int) inst.instance(i).value(index);
			}
			listCountDistVal.add(supportDistVal);
		}

		// réaliser le support pour chaque valeurs de notre nominal retenu
		if (m_Debug) {
			System.out.println("\n" + java.time.LocalDateTime.now() + " - Recherche de la meilleur valeur distincte :");
			System.out.print("\tLes occurences : ");
			for (int i = 0; i < listCountDistVal.size(); i++)
				System.out.print(listCountDistVal.get(i) + " - ");
			System.out.println();
			System.out.println(java.time.LocalDateTime.now() + " - Meilleur DistinctValue: ( "
					+ m_Attribute.value(indexBestDistVal) + " ) de l'attribut ( " + m_Attribute.name()
					+ " ) avec un support de: " + suppBestDistVal);
		}

		// Extraires les exemples associés a cet attribut
		ArrayList<Integer> FERM_exe = new ArrayList<Integer>();
		ArrayList<String> FERM_att = new ArrayList<String>();

		// Liste des instances verifiant la fermeture
		for (int i = 0; i < inst.numInstances(); i++) {
			if (m_Attribute.value((int) inst.instance(i).value(index)) == m_Attribute.value(indexBestDistVal)) {
				FERM_exe.add(i);
			}
		}
		int nbrInstFer = FERM_exe.size();
		if (m_Debug) {
			System.out.println(java.time.LocalDateTime.now() + " - Fermeture des instances: ");
			System.out.print("\tLes indices : ");
			for (int i = 0; i < nbrInstFer; i++)
				System.out.print(FERM_exe.get(i) + " - ");
			System.out.println();

//			for (int i = 0; i < nbrInstFer; i++)
//				System.out.println("\t\t" + FERM_exe.get(i) + " : " + inst.instance(FERM_exe.get(i)).toString());
		}
		// Liste des attributs associés à la fermeture
		String nl = "-null-";
		for (int i = 0; i < (int) inst.numAttributes() - 1; i++) {
			int cmpt = 0;
			double FirstDistVal = inst.instance(FERM_exe.get(0)).value(i);
			for (int j = 0; j < nbrInstFer;) {
				if (inst.instance(FERM_exe.get(j)).value(i) == FirstDistVal) {
					cmpt++;
					if (cmpt == nbrInstFer) {
						FERM_att.add(inst.attribute(i).value((int) inst.instance(FERM_exe.get(0)).value(i)));
					}
					j++;
				} else {
					j = nbrInstFer;
					FERM_att.add(nl);
				}
			}
		}
		/////////////// Extraire la classe majoritaire associée////////////////////
		int[] nbClasse = new int[inst.numClasses() + 1];
		for (int k = 0; k < inst.numClasses(); k++) {
			nbClasse[k] = 0;
		}

		// Parcourir les exemples associée à ce concept
		for (int j = 0; j < nbrInstFer; j++) {
			nbClasse[(int) inst.instance(FERM_exe.get(j)).classValue()]++;
		}

		// Detertminer l'indice de la classe associée
		int indiceMax = 0;
		for (int i = 0; i < inst.numClasses(); i++) {
			if (nbClasse[i] > nbClasse[indiceMax]) {
				indiceMax = i;
			}
		}
		// On retourne le concept Pertinent comme un vecteur de String
		ArrayList<String> CP = new ArrayList<String>();
		for (int i = 0; i < (int) inst.numAttributes() - 1; i++) {
			CP.add(FERM_att.get(i));
		}
		if (m_Debug)// modifier
			System.out.println("Fermeture pertinent retenu : " + CP);
		Classification_Rule r = new Classification_Rule(CP, indiceMax,
				inst.attribute(inst.classIndex()).value(indiceMax), measure[index]);
		classifierNC.add(r);
		return classifierNC;
	}

	// fermeture de toutes les valeurs du meilleur attribut
	public ArrayList<Classification_Rule> conceptLearningAllValueBestAttribute(Instances inst, double[] measure) {

		ArrayList<Classification_Rule> classifierNC = new ArrayList<Classification_Rule>();

		int index = 0;
		if (PertinenceMeasure == 6) {
			index = Utils.minIndex(measure);
		} else {
			index = Utils.maxIndex(measure);
		}
		Attribute m_Attribute = inst.attribute(index);

		ArrayList<Integer> instDistVal = new ArrayList<Integer>();
		// parcours sur les valeurs distinctes
		for (int indDistVal = 0; indDistVal < numDistinctValues(inst, index); indDistVal++) {
			instDistVal.clear();

			for (int j = 0; j < inst.numInstances(); j++) {
				if (m_Attribute.value((int) inst.instance(j).value(index)) == inst.attribute(index).value(indDistVal)) {
					instDistVal.add((int) inst.instance(j).value(index));
				}
			}

			if (instDistVal.size() != 0) {
				// Extraires les exemples associés a cet attribut
				ArrayList<Integer> FERM_exe = new ArrayList<Integer>();
				ArrayList<String> FERM_att = new ArrayList<String>();

				// Liste des instances verifiant la fermeture
				for (int i = 0; i < inst.numInstances(); i++) {
					if (m_Attribute.value((int) inst.instance(i).value(index)) == m_Attribute.value(indDistVal)) {
						FERM_exe.add(i);
					}
				}
				int nbrInstFer = FERM_exe.size();
				if (m_Debug) {
					System.out.println("\n" + java.time.LocalDateTime.now() + " - Fermeture des instances: ");
					System.out.print("\tLes indices : ");
					for (int i = 0; i < nbrInstFer; i++)
						System.out.print(FERM_exe.get(i) + " - ");
					System.out.println();

//					for (int i = 0; i < nbrInstFer; i++)
//						System.out.println("\t\t" + FERM_exe.get(i) + " : " + inst.instance(FERM_exe.get(i)).toString());
				}
				String nl = "-null-";
				for (int i = 0; i < (int) inst.numAttributes() - 1; i++) {
					int cmpt = 0;
					Double FirstDistVal = inst.instance(FERM_exe.get(0)).value(i);
					for (int j = 0; j < nbrInstFer;) {
						if (inst.instance(FERM_exe.get(j)).value(i) == FirstDistVal) {
							cmpt++;
							if (cmpt == nbrInstFer) {
								FERM_att.add(inst.attribute(i).value((int) inst.instance(FERM_exe.get(0)).value(i)));
							}
							j++;
						} else {
							j = nbrInstFer;
							FERM_att.add(nl);
						}
					}
				}

				// Extraire la classe majoritaire associée//
				int[] nbClasse = new int[inst.numClasses()];
				for (int k = 0; k < inst.numClasses(); k++) {
					nbClasse[k] = 0;
				}

				// Parcourir les exemples associée à ce concept
				for (int j = 0; j < nbrInstFer; j++) {
					nbClasse[(int) inst.instance(FERM_exe.get(j)).classValue()]++;
				}

				// Detertminer l'indice de la classe associé
				int indiceMax = 0;
				for (int i = 0; i < inst.numClasses(); i++) {
					if (nbClasse[i] > nbClasse[indiceMax]) {
						indiceMax = i;
					}
				}

				// On retourne le concept Pertinent comme un vecteur
				ArrayList<String> CP = new ArrayList<String>();
				for (int i = 0; i < (int) inst.numAttributes() - 1; i++) {
					CP.add(FERM_att.get(i));
				}

				if (m_Debug)
					System.out.println("concept pertinent retenu : " + CP);
				Classification_Rule r = new Classification_Rule(CP, indiceMax,
						inst.attribute(inst.classIndex()).value(indiceMax));
				classifierNC.add(r);
			}
		}
		return classifierNC;
	}

	// fermeture de la meilleur valeur de tous les attributs
	public ArrayList<Classification_Rule> conceptLearningBestValueAllAttribute(Instances inst, double[] measure) {

		ArrayList<Classification_Rule> classifierNC = new ArrayList<Classification_Rule>();
		int index;
		Attribute m_Attribute;
		// parcours de tous les attributs
		for (int ind = 0; ind < inst.numAttributes() - 1; ind++) {
			index = ind;
			m_Attribute = inst.attribute(index);

			int supportDistVal = 0;
			int indexBestDistVal = 0;
			int suppBestDistVal = 0;

			ArrayList<Integer> listCountDistVal = new ArrayList<Integer>();
			// Parcourir les differentes valeurs du 'm_Attribute'
			// numDistinctValues)
			for (int i = 0; i < numDistinctValues(inst, index); i++) {
				// Calcul du support de cette DistinctValue
				ArrayList<Integer> instDistVal = new ArrayList<Integer>();
				instDistVal.clear();

				supportDistVal = 0;
				for (int j = 0; j < inst.numInstances(); j++) {
					if (inst.instance(j).value(index) == inst.instance(i).value(index)) {
						supportDistVal++;
						instDistVal.add(j);
					}
				}
				if (suppBestDistVal <= supportDistVal) {
					suppBestDistVal = supportDistVal;
					indexBestDistVal = (int) inst.instance(i).value(index);
				}
				listCountDistVal.add(supportDistVal);
			}
			if (m_Debug) {
				System.out.println(
						"\n" + java.time.LocalDateTime.now() + " - Recherche de la meilleur valeur distincte :");
				System.out.print("\tLes occurences : ");
				for (int i = 0; i < listCountDistVal.size(); i++)
					System.out.print(listCountDistVal.get(i) + " - ");
				System.out.println();
				System.out.println(java.time.LocalDateTime.now() + " - Meilleur DistinctValue: ( "
						+ m_Attribute.value(indexBestDistVal) + " ) de l'attribut ( " + m_Attribute.name()
						+ " ) avec un support de: " + suppBestDistVal);
			}

			// Extraires les exemples associés a cet attribut
			ArrayList<Integer> FERM_exe = new ArrayList<Integer>();
			ArrayList<String> FERM_att = new ArrayList<String>();

			// Liste des instances verifiant la fermeture
			for (int i = 0; i < inst.numInstances(); i++) {
				if (m_Attribute.value((int) inst.instance(i).value(index)) == m_Attribute.value(indexBestDistVal)) {
					FERM_exe.add(i);
				}
			}
			int nbrInstFer = FERM_exe.size();
			if (m_Debug) {
				System.out.println(java.time.LocalDateTime.now() + " - Fermeture des instances: ");
				System.out.print("\tLes indices : ");
				for (int i = 0; i < nbrInstFer; i++)
					System.out.print(FERM_exe.get(i) + " - ");
				System.out.println();

//				for (int i = 0; i < nbrInstFer; i++)
//					System.out.println("\t\t" + FERM_exe.get(i) + " : " + inst.instance(FERM_exe.get(i)).toString());
			}
			// Liste des attributs associés à la fermeture
			String nl = "-null-";
			for (int i = 0; i < (int) inst.numAttributes() - 1; i++) {
				int cmpt = 0;
				double FirstDistVal = inst.instance(FERM_exe.get(0)).value(i);
				for (int j = 0; j < nbrInstFer;) {
					if (inst.instance(FERM_exe.get(j)).value(i) == FirstDistVal) {
						cmpt++;
						if (cmpt == nbrInstFer) {
							FERM_att.add(inst.attribute(i).value((int) inst.instance(FERM_exe.get(0)).value(i)));
						}
						j++;
					} else {
						j = nbrInstFer;
						FERM_att.add(nl);
					}
				}
			}
			/////////////// Extraire la classe majoritaire associée////////////////////
			int[] nbClasse = new int[inst.numClasses()];
			for (int k = 0; k < inst.numClasses(); k++) {
				nbClasse[k] = 0;
			}

			// Parcourir les exemples associée à ce concept
			for (int j = 0; j < nbrInstFer; j++) {
				nbClasse[(int) inst.instance(FERM_exe.get(j)).classValue()]++;
			}

			// Detertminer l'indice de la classe associée
			int indiceMax = 0;
			for (int i = 0; i < inst.numClasses(); i++) {
				if (nbClasse[i] > nbClasse[indiceMax]) {
					indiceMax = i;
				}
			}
			// On retourne le concept Pertinent comme un vecteur de String
			ArrayList<String> CP = new ArrayList<String>();
			for (int i = 0; i < (int) inst.numAttributes() - 1; i++) {
				CP.add(FERM_att.get(i));
			}

			if (m_Debug)
				System.out.println("concept pertinent retenu : " + CP);
			Classification_Rule r = new Classification_Rule(CP, indiceMax,
					inst.attribute(inst.classIndex()).value(indiceMax));

			classifierNC.add(r);

		}
		return classifierNC;
	}

	// fermeture de toutes les valeurs de tous les attributs
	public ArrayList<Classification_Rule> conceptLearningAllValueAllAttribute(Instances inst, double[] measure) {

		ArrayList<Classification_Rule> classifierNC = new ArrayList<Classification_Rule>();

		int index;
		Attribute m_Attribute;
		// parcours des attributs
		for (int ind = 0; ind < inst.numAttributes() - 1; ind++) {
			index = ind;
			m_Attribute = inst.attribute(index);
			ArrayList<Integer> instDistVal = new ArrayList<Integer>();

			// parcours des valeurs distinctes
			for (int indDistVal = 0; indDistVal < numDistinctValues(inst, index); indDistVal++) {
				instDistVal.clear();

				for (int j = 0; j < inst.numInstances(); j++) {
					if (m_Attribute.value((int) inst.instance(j).value(index)) == inst.attribute(index)
							.value(indDistVal)) {
						instDistVal.add((int) inst.instance(j).value(index));
					}
				}

				if (instDistVal.size() != 0) {
					// Extraires les exemples associés a cet attribut
					ArrayList<Integer> FERM_exe = new ArrayList<Integer>();
					ArrayList<String> FERM_att = new ArrayList<String>();

					// Liste des instances verifiant la fermeture
					for (int i = 0; i < inst.numInstances(); i++) {
						if (m_Attribute.value((int) inst.instance(i).value(index)) == m_Attribute.value(indDistVal)) {
							FERM_exe.add(i);
						}
					}
					int nbrInstFer = FERM_exe.size();
					if (m_Debug) {
						System.out.println("\n" + java.time.LocalDateTime.now() + " - Fermeture des instances: ");
						System.out.print("\tLes indices : ");
						for (int i = 0; i < nbrInstFer; i++)
							System.out.print(FERM_exe.get(i) + " - ");
						System.out.println();

//						for (int i = 0; i < nbrInstFer; i++)
//							System.out.println("\t\t" + FERM_exe.get(i) + " : " + inst.instance(FERM_exe.get(i)).toString());
					}
					String nl = "-null-";
					for (int i = 0; i < (int) inst.numAttributes() - 1; i++) {
						int cmpt = 0;
						Double FirstDistVal = inst.instance(FERM_exe.get(0)).value(i);
						for (int j = 0; j < nbrInstFer;) {
							if (inst.instance(FERM_exe.get(j)).value(i) == FirstDistVal) {
								cmpt++;
								if (cmpt == nbrInstFer) {
									FERM_att.add(
											inst.attribute(i).value((int) inst.instance(FERM_exe.get(0)).value(i)));
								}
								j++;
							} else {
								j = nbrInstFer;
								FERM_att.add(nl);
							}
						}
					}

					// Extraire la classe majoritaire associée//
					int[] nbClasse = new int[inst.numClasses()];
					for (int k = 0; k < inst.numClasses(); k++) {
						nbClasse[k] = 0;
					}

					// Parcourir les exemples associée à ce concept
					for (int j = 0; j < nbrInstFer; j++) {
						nbClasse[(int) inst.instance(FERM_exe.get(j)).classValue()]++;
					}

					// Detertminer l'indice de la classe associé
					int indiceMax = 0;
					for (int i = 0; i < inst.numClasses(); i++) {
						if (nbClasse[i] > nbClasse[indiceMax]) {
							indiceMax = i;
						}
					}

					// On retourne le concept Pertinent comme un vecteur
					ArrayList<String> CP = new ArrayList<String>();
					for (int i = 0; i < (int) inst.numAttributes() - 1; i++) {
						CP.add(FERM_att.get(i));
					}

					if (m_Debug)
						System.out.println("concept pertinent retenu : " + CP);
					Classification_Rule r = new Classification_Rule(CP, indiceMax,
							inst.attribute(inst.classIndex()).value(indiceMax));

					classifierNC.add(r);
				}
			}
		}
		return classifierNC;
	}

	// fermeture de la meilleur valeur de tout les attributs actif
	public ArrayList<Classification_Rule> conceptLearningBestValueAllActiveAttribute(Instances inst, double[] measure) {
		ArrayList<Classification_Rule> classifierNC = new ArrayList<Classification_Rule>();
		int index;
		Attribute m_Attribute;

		if (PertinenceMeasure == 6) {
			if (m_Debug) {
				System.out.println("Attribut actif retenu : ");
				for (int i = 0; i < inst.numAttributes() - 1; i++) {
					System.out.println("\tAttribut " + inst.attribute(i).name() + ": " + measure[i]);
				}
			}
		} else {
			if (m_Debug) {
				System.out.println("Attribut actif retenu : ");
				for (int i = 0; i < inst.numAttributes() - 1; i++) {
					if (measure[i] > 0) {
						System.out.println("\tAttribut " + inst.attribute(i).name() + ": " + measure[i]);
					}
				}
				boolean test = false;
				for (int i = 0; i < inst.numAttributes() - 1; i++) {
					if (measure[i] == 0) {
						test = true;
					}
				}
				if (test) {
					System.out.println("Attribut passif non retenu : ");
					for (int i = 0; i < inst.numAttributes() - 1; i++) {
						if (measure[i] == 0) {
							System.out.println("\tAttribut " + inst.attribute(i).name() + ": " + measure[i]);
						}
					}
				}
			}
		}

		// parcours des attributs
		for (int ind = 0; ind < inst.numAttributes() - 1; ind++) {
			index = ind;
			m_Attribute = inst.attribute(index);

			// on ne fait la fermeture que des attributs dont la mesure est strictement
			// supérieur à 0
			boolean vrai = true;
			if (PertinenceMeasure == 6) {
				vrai = true;
			} else {
				if (measure[index] > 0) {
					vrai = true;
				} else {
					vrai = false;
				}
			}

			if (vrai) {
				int supportDistVal = 0;
				int indexBestDistVal = 0;
				int suppBestDistVal = 0;

				ArrayList<Integer> listCountDistVal = new ArrayList<Integer>();
				// parcours des valeurs distinctes
				for (int i = 0; i < numDistinctValues(inst, index); i++) {
					// Calcul du support de cette DistinctValue
					ArrayList<Integer> instDistVal = new ArrayList<Integer>();
					instDistVal.clear();

					supportDistVal = 0;
					for (int j = 0; j < inst.numInstances(); j++) {
						if (inst.instance(j).value(index) == inst.instance(i).value(index)) {
							supportDistVal++;
							instDistVal.add(j);
						}
					}
					if (suppBestDistVal <= supportDistVal) {
						suppBestDistVal = supportDistVal;
						indexBestDistVal = (int) inst.instance(i).value(index);
					}
					listCountDistVal.add(supportDistVal);
				}
				if (m_Debug) {
					System.out.println(
							"\n" + java.time.LocalDateTime.now() + " - Recherche de la meilleur valeur distincte :");
					System.out.print("\tLes occurences : ");
					for (int i = 0; i < listCountDistVal.size(); i++)
						System.out.print(listCountDistVal.get(i) + " - ");
					System.out.println();
					System.out.println(java.time.LocalDateTime.now() + " - Meilleur DistinctValue: ( "
							+ m_Attribute.value(indexBestDistVal) + " ) de l'attribut ( " + m_Attribute.name()
							+ " ) avec un support de: " + suppBestDistVal);
				}

				// Extraires les exemples associés a cet attribut
				ArrayList<Integer> FERM_exe = new ArrayList<Integer>();
				ArrayList<String> FERM_att = new ArrayList<String>();

				// Liste des instances verifiant la fermeture
				for (int i = 0; i < inst.numInstances(); i++) {
					if (m_Attribute.value((int) inst.instance(i).value(index)) == m_Attribute.value(indexBestDistVal)) {
						FERM_exe.add(i);
					}
				}
				int nbrInstFer = FERM_exe.size();
				if (m_Debug) {
					System.out.println(java.time.LocalDateTime.now() + " - Fermeture des instances: ");
					System.out.print("\tLes indices : ");
					for (int i = 0; i < nbrInstFer; i++)
						System.out.print(FERM_exe.get(i) + " - ");
					System.out.println();

//					for (int i = 0; i < nbrInstFer; i++)
//						System.out.println("\t\t" + FERM_exe.get(i) + " : " + inst.instance(FERM_exe.get(i)).toString());
				}
				// Liste des attributs associés à la fermeture
				String nl = "-null-";
				for (int i = 0; i < (int) inst.numAttributes() - 1; i++) {
					int cmpt = 0;
					double FirstDistVal = inst.instance(FERM_exe.get(0)).value(i);
					for (int j = 0; j < nbrInstFer;) {
						if (inst.instance(FERM_exe.get(j)).value(i) == FirstDistVal && measure[i] > 0) {
							cmpt++;
							if (cmpt == nbrInstFer) {
								FERM_att.add(inst.attribute(i).value((int) inst.instance(FERM_exe.get(0)).value(i)));
							}
							j++;
						} else {
							j = nbrInstFer;
							FERM_att.add(nl);
						}
					}
				}
				/////////////// Extraire la classe majoritaire associée////////////////////
				int[] nbClasse = new int[inst.numClasses()];
				for (int k = 0; k < inst.numClasses(); k++) {
					nbClasse[k] = 0;
				}

				// Parcourir les exemples associée à ce concept
				for (int j = 0; j < nbrInstFer; j++) {
					nbClasse[(int) inst.instance(FERM_exe.get(j)).classValue()]++;
				}

				// Detertminer l'indice de la classe associée
				int indiceMax = 0;
				for (int i = 0; i < inst.numClasses(); i++) {
					if (nbClasse[i] > nbClasse[indiceMax]) {
						indiceMax = i;
					}
				}
				// On retourne le concept Pertinent comme un vecteur de String
				ArrayList<String> CP = new ArrayList<String>();
				for (int i = 0; i < (int) inst.numAttributes() - 1; i++) {
					CP.add(FERM_att.get(i));
				}
				if (m_Debug)
					System.out.println("concept pertinent retenu : " + CP);
				Classification_Rule r = new Classification_Rule(CP, indiceMax,
						inst.attribute(inst.classIndex()).value(indiceMax), measure[index]);
				classifierNC.add(r);
			}
		}
		return classifierNC;
	}

	// fermeture de toutes les valeurs de tous les attributs actif
	public ArrayList<Classification_Rule> conceptLearningAllValueAllActiveAttribute(Instances inst, double[] measure) {
		ArrayList<Classification_Rule> classifierNC = new ArrayList<Classification_Rule>();

		int index;
		Attribute m_Attribute;

		if (m_Debug) {
			System.out.println("Attribut actif retenu : ");
			for (int i = 0; i < inst.numAttributes() - 1; i++) {
				if (measure[i] > 0) {
					System.out.println("\tAttribut " + inst.attribute(i).name() + ": " + measure[i]);
				}
			}

			boolean test = false;
			for (int i = 0; i < inst.numAttributes() - 1; i++) {
				if (measure[i] == 0) {
					test = true;
				}
			}
			if (test) {
				System.out.println("Attribut passif non retenu : ");
				for (int i = 0; i < inst.numAttributes() - 1; i++) {
					if (measure[i] == 0) {
						System.out.println("\tAttribut " + inst.attribute(i).name() + ": " + measure[i]);
					}
				}
			}
		}
		// parcours des attributs
		for (int ind = 0; ind < inst.numAttributes() - 1; ind++) {
			index = ind;
			m_Attribute = inst.attribute(index);
			// on ne garde que les attributs dont la mesures est > 0
			if (measure[index] > 0) {
				ArrayList<Integer> instDistVal = new ArrayList<Integer>();

				// parcours des valeurs distinctes
				for (int indDistVal = 0; indDistVal < numDistinctValues(inst, index); indDistVal++) {
					instDistVal.clear();

					for (int j = 0; j < inst.numInstances(); j++) {
						if (m_Attribute.value((int) inst.instance(j).value(index)) == inst.attribute(index)
								.value(indDistVal)) {
							instDistVal.add((int) inst.instance(j).value(index));
						}
					}

					if (instDistVal.size() != 0) {
						// Extraires les exemples associés a cet attribut
						ArrayList<Integer> FERM_exe = new ArrayList<Integer>();
						ArrayList<String> FERM_att = new ArrayList<String>();

						// Liste des instances verifiant la fermeture
						for (int i = 0; i < inst.numInstances(); i++) {
							if (m_Attribute.value((int) inst.instance(i).value(index)) == m_Attribute
									.value(indDistVal)) {
								FERM_exe.add(i);
							}
						}
						int nbrInstFer = FERM_exe.size();
						if (m_Debug) {
							System.out.println("\n" + java.time.LocalDateTime.now() + " - Fermeture des instances: ");
							System.out.print("\tLes indices : ");
							for (int i = 0; i < nbrInstFer; i++)
								System.out.print(FERM_exe.get(i) + " - ");
							System.out.println();

//							for (int i = 0; i < nbrInstFer; i++)
//								System.out.println("\t\t" + FERM_exe.get(i) + " : " + inst.instance(FERM_exe.get(i)).toString());
						}
						String nl = "-null-";
						for (int i = 0; i < (int) inst.numAttributes() - 1; i++) {
							int cmpt = 0;
							Double FirstDistVal = inst.instance(FERM_exe.get(0)).value(i);
							for (int j = 0; j < nbrInstFer;) {
								if (inst.instance(FERM_exe.get(j)).value(i) == FirstDistVal && measure[i] > 0) {
									cmpt++;
									if (cmpt == nbrInstFer && measure[i] > 0) {
										FERM_att.add(
												inst.attribute(i).value((int) inst.instance(FERM_exe.get(0)).value(i)));
									}
									j++;
								} else {
									j = nbrInstFer;
									FERM_att.add(nl);
								}
							}
						}

						// Extraire la classe majoritaire associée//
						int[] nbClasse = new int[inst.numClasses()];
						for (int k = 0; k < inst.numClasses(); k++) {
							nbClasse[k] = 0;
						}

						// Parcourir les exemples associée à ce concept
						for (int j = 0; j < nbrInstFer; j++) {
							nbClasse[(int) inst.instance(FERM_exe.get(j)).classValue()]++;
						}

						// Detertminer l'indice de la classe associé
						int indiceMax = 0;
						for (int i = 0; i < inst.numClasses(); i++) {
							if (nbClasse[i] > nbClasse[indiceMax]) {
								indiceMax = i;
							}
						}

						// On retourne le concept Pertinent comme un vecteur
						ArrayList<String> CP = new ArrayList<String>();
						for (int i = 0; i < (int) inst.numAttributes() - 1; i++) {
							CP.add(FERM_att.get(i));
						}

						if (m_Debug)
							System.out.println("concept pertinent retenu : " + CP);
						Classification_Rule r = new Classification_Rule(CP, indiceMax,
								inst.attribute(inst.classIndex()).value(indiceMax));

						classifierNC.add(r);
					}
				}
			}
		}
		return classifierNC;
	}

	// ----------------------------MESURE

	/**
	 * @param inst  container of instance
	 * @param index index of the attribut selected
	 * @return count of dinstinct value
	 */
	private int numDistinctValues(Instances inst, int index) {
		ArrayList<Double> distinctValues = new ArrayList<>();
		for (int i = 0; i < inst.numInstances(); i++) {
			double key = inst.get(i).value(index);
			if (!Utils.isMissingValue(key) && !distinctValues.contains(key)) {
				distinctValues.add(key);
			}
		}
		return distinctValues.size();
	}

	/**
	 * @param data container of instance
	 * @return double[] measurement of all atributs
	 */
	public double[] computeInfoGains(Instances data) {
		double[] m_InfoGains;
		int classIndex = data.classIndex();
		int numInstances = data.numInstances();

		int numClasses = data.attribute(classIndex).numValues();
		// Reserve space and initialize counters
		double[][][] counts = new double[data.numAttributes()][][];
		for (int k = 0; k < data.numAttributes(); k++) {
			if (k != classIndex) {
				int numValues = data.attribute(k).numValues();
				counts[k] = new double[(int) (numValues + 2)][(int) (numClasses + 1)];
			}
		}

		// Initialize counters
		double[] temp = new double[numClasses + 1];
		for (int k = 0; k < numInstances; k++) {
			Instance inst = data.instance(k);
			if (inst.classIsMissing()) {
				temp[numClasses] += inst.weight();
			} else {
				temp[(int) inst.classValue()] += inst.weight();
			}
		}
		for (int k = 0; k < counts.length; k++) {
			if (k != classIndex) {
				for (int i = 0; i < temp.length; i++) {
					counts[k][0][i] = temp[i];
				}
			}
		}
		// Get counts
		for (int k = 0; k < numInstances; k++) {
			Instance inst = data.instance(k);
			for (int i = 0; i < inst.numValues(); i++) {
				if (inst.index(i) != classIndex) {
					if (inst.isMissingSparse(i) || inst.classIsMissing()) {
						if (!inst.isMissingSparse(i)) {
							counts[inst.index(i)][(int) inst.valueSparse(i)][numClasses] += inst.weight();
							counts[inst.index(i)][0][numClasses] -= inst.weight();
						} else if (!inst.classIsMissing()) {
							counts[inst.index(i)][data.attribute(inst.index(i)).numValues()][(int) inst
									.classValue()] += inst.weight();
							counts[inst.index(i)][0][(int) inst.classValue()] -= inst.weight();
						} else {
							counts[inst.index(i)][data.attribute(inst.index(i)).numValues()][numClasses] += inst
									.weight();
							counts[inst.index(i)][0][numClasses] -= inst.weight();
						}
					} else {
						counts[(int) inst.index(i)][(int) inst.valueSparse(i)][(int) inst.classValue()] += inst
								.weight();
						counts[(int) inst.index(i)][0][(int) inst.classValue()] -= inst.weight();
					}
				}
			}
		}

		// Compute info gains
		m_InfoGains = new double[data.numAttributes()];
		for (int i = 0; i < data.numAttributes(); i++) {
			if (i != classIndex) {
				m_InfoGains[i] = (ContingencyTables.entropyOverColumns(counts[i])
						- ContingencyTables.entropyConditionedOnRows(counts[i]));
			}
		}
		return m_InfoGains;
	}

	/**
	 * @param data      container of instance
	 * @param attribute index of the selected attribut
	 * @return double measurement of the selected attribut
	 */
	public double computeGainRatio(Instances data, int attribute) {
		Instances m_trainInstances = data;
		int m_classIndex = m_trainInstances.classIndex();
		int m_numInstances = m_trainInstances.numInstances();
		int m_numClasses = m_trainInstances.attribute(m_classIndex).numValues();
		boolean m_missing_merge = true;

		int i, j, ii, jj;
		int ni, nj;
		double sum = 0.0;
		ni = m_trainInstances.attribute(attribute).numValues() + 1;
		nj = m_numClasses + 1;
		double[] sumi, sumj;
		Instance inst;
		double temp = 0.0;
		sumi = new double[ni];
		sumj = new double[nj];
		double[][] counts = new double[ni][nj];
		sumi = new double[ni];
		sumj = new double[nj];

		for (i = 0; i < ni; i++) {
			sumi[i] = 0.0;

			for (j = 0; j < nj; j++) {
				sumj[j] = 0.0;
				counts[i][j] = 0.0;
			}
		}

		// Fill the contingency table
		for (i = 0; i < m_numInstances; i++) {
			inst = m_trainInstances.instance(i);

			if (inst.isMissing(attribute)) {
				ii = ni - 1;
			} else {
				ii = (int) inst.value(attribute);
			}

			if (inst.isMissing(m_classIndex)) {
				jj = nj - 1;
			} else {
				jj = (int) inst.value(m_classIndex);
			}

			counts[ii][jj] += inst.weight();
		}

		// get the row totals
		for (i = 0; i < ni; i++) {
			sumi[i] = 0.0;

			for (j = 0; j < nj; j++) {
				sumi[i] += counts[i][j];
				sum += counts[i][j];
			}
		}

		// get the column totals
		for (j = 0; j < nj; j++) {
			sumj[j] = 0.0;

			for (i = 0; i < ni; i++) {
				sumj[j] += counts[i][j];
			}
		}

		// distribute missing counts
		if (m_missing_merge && (sumi[ni - 1] < sum) && (sumj[nj - 1] < sum)) {
			double[] i_copy = new double[sumi.length];
			double[] j_copy = new double[sumj.length];
			double[][] counts_copy = new double[sumi.length][sumj.length];

			for (i = 0; i < ni; i++) {
				System.arraycopy(counts[i], 0, counts_copy[i], 0, sumj.length);
			}

			System.arraycopy(sumi, 0, i_copy, 0, sumi.length);
			System.arraycopy(sumj, 0, j_copy, 0, sumj.length);
			double total_missing = (sumi[ni - 1] + sumj[nj - 1] - counts[ni - 1][nj - 1]);

			// do the missing i's
			if (sumi[ni - 1] > 0.0) {
				for (j = 0; j < nj - 1; j++) {
					if (counts[ni - 1][j] > 0.0) {
						for (i = 0; i < ni - 1; i++) {
							temp = ((i_copy[i] / (sum - i_copy[ni - 1])) * counts[ni - 1][j]);
							counts[i][j] += temp;
							sumi[i] += temp;
						}

						counts[ni - 1][j] = 0.0;
					}
				}
			}

			sumi[ni - 1] = 0.0;

			// do the missing j's
			if (sumj[nj - 1] > 0.0) {
				for (i = 0; i < ni - 1; i++) {
					if (counts[i][nj - 1] > 0.0) {
						for (j = 0; j < nj - 1; j++) {
							temp = ((j_copy[j] / (sum - j_copy[nj - 1])) * counts[i][nj - 1]);
							counts[i][j] += temp;
							sumj[j] += temp;
						}

						counts[i][nj - 1] = 0.0;
					}
				}
			}

			sumj[nj - 1] = 0.0;

			// do the both missing
			if (counts[ni - 1][nj - 1] > 0.0 && total_missing < sum) {
				for (i = 0; i < ni - 1; i++) {
					for (j = 0; j < nj - 1; j++) {
						temp = (counts_copy[i][j] / (sum - total_missing)) * counts_copy[ni - 1][nj - 1];
						counts[i][j] += temp;
						sumi[i] += temp;
						sumj[j] += temp;
					}
				}

				counts[ni - 1][nj - 1] = 0.0;
			}
		}

		return ContingencyTables.gainRatio(counts);
	}

	/**
	 * @param data container of instance
	 * @return double[] measurement of all attributs
	 */
	public double[] computeCorrelation(Instances data) {

		int numClasses = data.classAttribute().numValues();
		int classIndex = data.classIndex();
		int numInstances = data.numInstances();
		double[] m_correlations = new double[data.numAttributes()];
		/*
		 * boolean hasNominals = false; boolean hasNumerics = false;
		 */
		List<Integer> numericIndexes = new ArrayList<Integer>();
		List<Integer> nominalIndexes = new ArrayList<Integer>();
		StringBuffer m_detailedOutputBuff = new StringBuffer();
		boolean m_detailedOutput = false;

		// add another dimension just before the last [2] (0 for 0/1 binary vector
		// and
		// 1 for corresponding instance weights for the 1's)
		double[][][] nomAtts = new double[data.numAttributes()][][];
		for (int i = 0; i < data.numAttributes(); i++) {
			if (data.attribute(i).isNominal() && i != classIndex) {
				nomAtts[i] = new double[data.attribute(i).numValues()][data.numInstances()];
				Arrays.fill(nomAtts[i][0], 1.0); // set zero index for this att to all
													// 1's
				nominalIndexes.add(i);
			} else if (data.attribute(i).isNumeric() && i != classIndex) {
				numericIndexes.add(i);
			}
		}

		// do the nominal attributes
		if (nominalIndexes.size() > 0) {
			for (int i = 0; i < data.numInstances(); i++) {
				Instance current = data.instance(i);
				for (int j = 0; j < current.numValues(); j++) {
					if (current.attribute(current.index(j)).isNominal() && current.index(j) != classIndex) {
						// Will need to check for zero in case this isn't a sparse
						// instance (unless we add 1 and subtract 1)
						nomAtts[current.index(j)][(int) current.valueSparse(j)][i] += 1;
						nomAtts[current.index(j)][0][i] -= 1;
					}
				}
			}
		}

		if (data.classAttribute().isNumeric()) {
			double[] classVals = attributeToDoubleArray(data, classIndex);

			// do the numeric attributes
			for (Integer i : numericIndexes) {
				double[] numAttVals = attributeToDoubleArray(data, i);
				m_correlations[i] = Utils.correlation(numAttVals, classVals, numAttVals.length);

				if (m_correlations[i] == 1.0) {
					// check for zero variance (useless numeric attribute)
					if (Utils.variance(numAttVals) == 0) {
						m_correlations[i] = 0;
					}
				}
			}

			// do the nominal attributes
			if (nominalIndexes.size() > 0) {
				// now compute the correlations for the binarized nominal attributes
				for (Integer i : nominalIndexes) {
					double sum = 0;
					double corr = 0;
					double sumCorr = 0;
					double sumForValue = 0;

					if (m_detailedOutput) {
						m_detailedOutputBuff.append("\n\n").append(data.attribute(i).name());
					}

					for (int j = 0; j < data.attribute(i).numValues(); j++) {
						sumForValue = Utils.sum(nomAtts[i][j]);
						corr = Utils.correlation(nomAtts[i][j], classVals, classVals.length);

						// useless attribute - all instances have the same value
						if (sumForValue == numInstances || sumForValue == 0) {
							corr = 0;
						}
						if (corr < 0.0) {
							corr = -corr;
						}
						sumCorr += sumForValue * corr;
						sum += sumForValue;

						if (m_detailedOutput) {
							m_detailedOutputBuff.append("\n\t").append(data.attribute(i).value(j)).append(": ");
							m_detailedOutputBuff.append(Utils.doubleToString(corr, 6));
						}
					}
					m_correlations[i] = (sum > 0) ? sumCorr / sum : 0;
				}
			}
		} else {
			// class is nominal
			double[][] binarizedClasses = new double[data.classAttribute().numValues()][data.numInstances()];

			// this is equal to the number of instances for all inst weights = 1
			double[] classValCounts = new double[data.classAttribute().numValues()];

			for (int i = 0; i < data.numInstances(); i++) {
				Instance current = data.instance(i);
				binarizedClasses[(int) current.classValue()][i] = 1;
			}
			for (int i = 0; i < data.classAttribute().numValues(); i++) {
				classValCounts[i] = Utils.sum(binarizedClasses[i]);
			}

			double sumClass = Utils.sum(classValCounts);

			// do numeric attributes first
			if (numericIndexes.size() > 0) {
				for (Integer i : numericIndexes) {
					double[] numAttVals = attributeToDoubleArray(data, i);
					double corr = 0;
					double sumCorr = 0;

					for (int j = 0; j < data.classAttribute().numValues(); j++) {
						corr = Utils.correlation(numAttVals, binarizedClasses[j], numAttVals.length);
						if (corr < 0.0) {
							corr = -corr;
						}

						if (corr == 1.0) {
							// check for zero variance (useless numeric attribute)
							if (Utils.variance(numAttVals) == 0) {
								corr = 0;
							}
						}

						sumCorr += classValCounts[j] * corr;
					}
					m_correlations[i] = sumCorr / sumClass;
				}
			}

			if (nominalIndexes.size() > 0) {
				for (Integer i : nominalIndexes) {
					if (m_detailedOutput) {
						m_detailedOutputBuff.append("\n\n").append(data.attribute(i).name());
					}

					double sumForAtt = 0;
					double corrForAtt = 0;
					for (int j = 0; j < data.attribute(i).numValues(); j++) {
						double sumForValue = Utils.sum(nomAtts[i][j]);
						double corr = 0;
						double sumCorr = 0;
						double avgCorrForValue = 0;

						sumForAtt += sumForValue;
						for (int k = 0; k < numClasses; k++) {

							// corr between value j and class k
							corr = Utils.correlation(nomAtts[i][j], binarizedClasses[k], binarizedClasses[k].length);

							// useless attribute - all instances have the same value
							if (sumForValue == numInstances || sumForValue == 0) {
								corr = 0;
							}
							if (corr < 0.0) {
								corr = -corr;
							}
							sumCorr += classValCounts[k] * corr;
						}
						avgCorrForValue = sumCorr / sumClass;
						corrForAtt += sumForValue * avgCorrForValue;

						if (m_detailedOutput) {
							m_detailedOutputBuff.append("\n\t").append(data.attribute(i).value(j)).append(": ");
							m_detailedOutputBuff.append(Utils.doubleToString(avgCorrForValue, 6));
						}
					}

					// the weighted average corr for att i as
					// a whole (wighted by value frequencies)
					m_correlations[i] = (sumForAtt > 0) ? corrForAtt / sumForAtt : 0;
				}
			}
		}

		if (m_detailedOutputBuff != null && m_detailedOutputBuff.length() > 0) {
			m_detailedOutputBuff.append("\n");
		}
		return m_correlations;
	}

	/**
	 * @param data container of instance
	 * @return double[] measurement of all attributs
	 */
	public double[] computeHRatio(Instances data) {
		int classIndex = data.classIndex();
		int numInstances = data.numInstances();

		int numClasses = data.attribute(classIndex).numValues();

		// Reserve space and initialize counters
		double[][][] counts = new double[data.numAttributes()][][];
		for (int k = 0; k < data.numAttributes(); k++) {
			if (k != classIndex) {
				int numValues = data.attribute(k).numValues();
				counts[k] = new double[numValues + 1][numClasses + 1];
			}
		}

		// Initialize counters
		double[] temp = new double[numClasses + 1];
		for (int k = 0; k < numInstances; k++) {
			Instance inst = data.instance(k);
			if (inst.classIsMissing()) {
				temp[numClasses] += inst.weight();
			} else {
				temp[(int) inst.classValue()] += inst.weight();
			}
		}
		for (int k = 0; k < counts.length; k++) {
			if (k != classIndex) {
				for (int i = 0; i < temp.length; i++) {
					counts[k][0][i] = temp[i];
				}
			}
		}

		// Get counts
		for (int k = 0; k < numInstances; k++) {
			Instance inst = data.instance(k);
			for (int i = 0; i < inst.numValues(); i++) {
				if (inst.index(i) != classIndex) {
					if (inst.isMissingSparse(i) || inst.classIsMissing()) {
						if (!inst.isMissingSparse(i)) {
							counts[inst.index(i)][(int) inst.valueSparse(i)][numClasses] += inst.weight();
							counts[inst.index(i)][0][numClasses] -= inst.weight();
						} else if (!inst.classIsMissing()) {
							counts[inst.index(i)][data.attribute(inst.index(i)).numValues()][(int) inst
									.classValue()] += inst.weight();
							counts[inst.index(i)][0][(int) inst.classValue()] -= inst.weight();
						} else {
							counts[inst.index(i)][data.attribute(inst.index(i)).numValues()][numClasses] += inst
									.weight();
							counts[inst.index(i)][0][numClasses] -= inst.weight();
						}
					} else {
						counts[inst.index(i)][(int) inst.valueSparse(i)][(int) inst.classValue()] += inst.weight();
						counts[inst.index(i)][0][(int) inst.classValue()] -= inst.weight();
					}
				}
			}

		}

		boolean m_missing_merge = true;
		// distribute missing counts if required
		if (m_missing_merge) {

			for (int k = 0; k < data.numAttributes(); k++) {
				if (k != classIndex) {
					int numValues = data.attribute(k).numValues();

					// Compute marginals
					double[] rowSums = new double[numValues];
					double[] columnSums = new double[numClasses];
					double sum = 0;
					for (int i = 0; i < numValues; i++) {
						for (int j = 0; j < numClasses; j++) {
							rowSums[i] += counts[k][i][j];
							columnSums[j] += counts[k][i][j];
						}
						sum += rowSums[i];
					}

					if (Utils.gr(sum, 0)) {
						double[][] additions = new double[numValues][numClasses];

						// Compute what needs to be added to each row
						for (int i = 0; i < numValues; i++) {
							for (int j = 0; j < numClasses; j++) {
								additions[i][j] = (rowSums[i] / sum) * counts[k][numValues][j];
							}
						}

						// Compute what needs to be added to each column
						for (int i = 0; i < numClasses; i++) {
							for (int j = 0; j < numValues; j++) {
								additions[j][i] += (columnSums[i] / sum) * counts[k][j][numClasses];
							}
						}

						// Compute what needs to be added to each cell
						for (int i = 0; i < numClasses; i++) {
							for (int j = 0; j < numValues; j++) {
								additions[j][i] += (counts[k][j][i] / sum) * counts[k][numValues][numClasses];
							}
						}

						// Make new contingency table
						double[][] newTable = new double[numValues][numClasses];
						for (int i = 0; i < numValues; i++) {
							for (int j = 0; j < numClasses; j++) {
								newTable[i][j] = counts[k][i][j] + additions[i][j];
							}
						}
						counts[k] = newTable;
					}
				}
			}
		}

		// Compute info gains
		double[] m_InfoGains = new double[data.numAttributes()];
		for (int i = 0; i < data.numAttributes(); i++) {
			if (i != classIndex) {
				m_InfoGains[i] = (ContingencyTables.entropyConditionedOnRows(counts[i])
						+ ContingencyTables.entropyOverRows(counts[i]))
						/ ContingencyTables.entropyOverColumns(counts[i]);
			}
		}
		return m_InfoGains;

	}

	/**
	 * @param data container of instance
	 * @return double[] measurement of all atributs
	 */
	public double[] computeInfoMutuelle(Instances data) {

		int classIndex = data.classIndex();
		int numInstances = data.numInstances();
		int numClasses = data.attribute(classIndex).numValues();

		// Reserve space and initialize counters
		double[][][] counts = new double[data.numAttributes()][][];
		for (int k = 0; k < data.numAttributes(); k++) {
			if (k != classIndex) {
				int numValues = data.attribute(k).numValues();
				counts[k] = new double[numValues + 1][numClasses + 1];
			}
		}

		// Initialize counters
		double[] temp = new double[numClasses + 1];
		for (int k = 0; k < numInstances; k++) {
			Instance inst = data.instance(k);
			if (inst.classIsMissing()) {
				temp[numClasses] += inst.weight();
			} else {
				temp[(int) inst.classValue()] += inst.weight();
			}
		}
		for (int k = 0; k < counts.length; k++) {
			if (k != classIndex) {
				for (int i = 0; i < temp.length; i++) {
					counts[k][0][i] = temp[i];
				}
			}
		}

		// Get counts
		for (int k = 0; k < numInstances; k++) {
			Instance inst = data.instance(k);
			for (int i = 0; i < inst.numValues(); i++) {
				if (inst.index(i) != classIndex) {
					if (inst.isMissingSparse(i) || inst.classIsMissing()) {
						if (!inst.isMissingSparse(i)) {
							counts[inst.index(i)][(int) inst.valueSparse(i)][numClasses] += inst.weight();
							counts[inst.index(i)][0][numClasses] -= inst.weight();
						} else if (!inst.classIsMissing()) {
							counts[inst.index(i)][data.attribute(inst.index(i)).numValues()][(int) inst
									.classValue()] += inst.weight();
							counts[inst.index(i)][0][(int) inst.classValue()] -= inst.weight();
						} else {
							counts[inst.index(i)][data.attribute(inst.index(i)).numValues()][numClasses] += inst
									.weight();
							counts[inst.index(i)][0][numClasses] -= inst.weight();
						}
					} else {
						counts[inst.index(i)][(int) inst.valueSparse(i)][(int) inst.classValue()] += inst.weight();
						counts[inst.index(i)][0][(int) inst.classValue()] -= inst.weight();
					}
				}
			}
		}

		boolean m_missing_merge = true;
		// distribute missing counts if required
		if (m_missing_merge) {

			for (int k = 0; k < data.numAttributes(); k++) {
				if (k != classIndex) {
					int numValues = data.attribute(k).numValues();

					// Compute marginals
					double[] rowSums = new double[numValues];
					double[] columnSums = new double[numClasses];
					double sum = 0;
					for (int i = 0; i < numValues; i++) {
						for (int j = 0; j < numClasses; j++) {
							rowSums[i] += counts[k][i][j];
							columnSums[j] += counts[k][i][j];
						}
						sum += rowSums[i];
					}

					if (Utils.gr(sum, 0)) {
						double[][] additions = new double[numValues][numClasses];

						// Compute what needs to be added to each row
						for (int i = 0; i < numValues; i++) {
							for (int j = 0; j < numClasses; j++) {
								additions[i][j] = (rowSums[i] / sum) * counts[k][numValues][j];
							}
						}

						// Compute what needs to be added to each column
						for (int i = 0; i < numClasses; i++) {
							for (int j = 0; j < numValues; j++) {
								additions[j][i] += (columnSums[i] / sum) * counts[k][j][numClasses];
							}
						}

						// Compute what needs to be added to each cell
						for (int i = 0; i < numClasses; i++) {
							for (int j = 0; j < numValues; j++) {
								additions[j][i] += (counts[k][j][i] / sum) * counts[k][numValues][numClasses];
							}
						}

						// Make new contingency table
						double[][] newTable = new double[numValues][numClasses];
						for (int i = 0; i < numValues; i++) {
							for (int j = 0; j < numClasses; j++) {
								newTable[i][j] = counts[k][i][j] + additions[i][j];
							}
						}
						counts[k] = newTable;
					}
				}
			}
		}

		// Compute info gains
		double[] m_InfoGains = new double[data.numAttributes()];
		for (int i = 0; i < data.numAttributes(); i++) {
			if (i != classIndex) {
				m_InfoGains[i] = (ContingencyTables.entropyOverColumns(counts[i])
						+ ContingencyTables.entropyOverRows(counts[i])
						- ContingencyTables.entropyConditionedOnRows(counts[i]));
			}
		}
		return m_InfoGains;
	}

	/**
	 * @param data      container of instance
	 * @param attribute index of the selected attribut
	 * @return double measurement of the attribut
	 */
	public double computeSymmetrical(Instances data, int attribute) {
		Instances m_trainInstances = data;
		int m_classIndex = m_trainInstances.classIndex();
		int m_numInstances = m_trainInstances.numInstances();
		int m_numClasses = m_trainInstances.attribute(m_classIndex).numValues();

		int i, j, ii, jj;
		int ni, nj;
		double sum = 0.0;
		ni = m_trainInstances.attribute(attribute).numValues() + 1;
		nj = m_numClasses + 1;
		double[] sumi, sumj;
		Instance inst;
		double temp = 0.0;
		sumi = new double[ni];
		sumj = new double[nj];
		double[][] counts = new double[ni][nj];
		sumi = new double[ni];
		sumj = new double[nj];

		for (i = 0; i < ni; i++) {
			sumi[i] = 0.0;

			for (j = 0; j < nj; j++) {
				sumj[j] = 0.0;
				counts[i][j] = 0.0;
			}
		}

		// Fill the contingency table
		for (i = 0; i < m_numInstances; i++) {
			inst = m_trainInstances.instance(i);

			if (inst.isMissing(attribute)) {
				ii = ni - 1;
			} else {
				ii = (int) inst.value(attribute);
			}

			if (inst.isMissing(m_classIndex)) {
				jj = nj - 1;
			} else {
				jj = (int) inst.value(m_classIndex);
			}

			counts[ii][jj]++;
		}

		// get the row totals
		for (i = 0; i < ni; i++) {
			sumi[i] = 0.0;

			for (j = 0; j < nj; j++) {
				sumi[i] += counts[i][j];
				sum += counts[i][j];
			}
		}

		// get the column totals
		for (j = 0; j < nj; j++) {
			sumj[j] = 0.0;

			for (i = 0; i < ni; i++) {
				sumj[j] += counts[i][j];
			}
		}

		boolean m_missing_merge = true;
		// distribute missing counts
		if (m_missing_merge && (sumi[ni - 1] < m_numInstances) && (sumj[nj - 1] < m_numInstances)) {
			double[] i_copy = new double[sumi.length];
			double[] j_copy = new double[sumj.length];
			double[][] counts_copy = new double[sumi.length][sumj.length];

			for (i = 0; i < ni; i++) {
				System.arraycopy(counts[i], 0, counts_copy[i], 0, sumj.length);
			}

			System.arraycopy(sumi, 0, i_copy, 0, sumi.length);
			System.arraycopy(sumj, 0, j_copy, 0, sumj.length);
			double total_missing = (sumi[ni - 1] + sumj[nj - 1] - counts[ni - 1][nj - 1]);

			// do the missing i's
			if (sumi[ni - 1] > 0.0) {
				for (j = 0; j < nj - 1; j++) {
					if (counts[ni - 1][j] > 0.0) {
						for (i = 0; i < ni - 1; i++) {
							temp = ((i_copy[i] / (sum - i_copy[ni - 1])) * counts[ni - 1][j]);
							counts[i][j] += temp;
							sumi[i] += temp;
						}

						counts[ni - 1][j] = 0.0;
					}
				}
			}

			sumi[ni - 1] = 0.0;

			// do the missing j's
			if (sumj[nj - 1] > 0.0) {
				for (i = 0; i < ni - 1; i++) {
					if (counts[i][nj - 1] > 0.0) {
						for (j = 0; j < nj - 1; j++) {
							temp = ((j_copy[j] / (sum - j_copy[nj - 1])) * counts[i][nj - 1]);
							counts[i][j] += temp;
							sumj[j] += temp;
						}

						counts[i][nj - 1] = 0.0;
					}
				}
			}

			sumj[nj - 1] = 0.0;

			// do the both missing
			if (counts[ni - 1][nj - 1] > 0.0 && total_missing != sum) {
				for (i = 0; i < ni - 1; i++) {
					for (j = 0; j < nj - 1; j++) {
						temp = (counts_copy[i][j] / (sum - total_missing)) * counts_copy[ni - 1][nj - 1];
						counts[i][j] += temp;
						sumi[i] += temp;
						sumj[j] += temp;
					}
				}

				counts[ni - 1][nj - 1] = 0.0;
			}
		}

		return ContingencyTables.symmetricalUncertainty(counts);
	}

	/**
	 * @param data      container of instance
	 * @param attribute index of the selected attribut
	 * @return double measurement of the attribut
	 */
	public double computeEntropy(Instances data, int attribute) {
		double[] count = new double[data.numInstances()];
		for (int i = 0; i < count.length; i++) {
			count[i] = data.get(i).value(attribute);
		}
		return ContingencyTables.entropy(count);
	}
  
	/**
	 * @param inst  container of instance
	 * @param index index of the selected attribut
	 * @return double[] list of values for the choosen attribut
	 */
	private double[] attributeToDoubleArray(Instances inst, int index) {

		double[] result = new double[inst.numInstances()];
		for (int i = 0; i < result.length; i++) {
			result[i] = inst.get(i).value(index);
		}
		return result;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public ImmutableCapabilities defineImmutableCapabilities() {
		if (this.getClass() == CNC.class)
			return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
		else
			return new ImmutableCapabilities(Capability.VIEW_STANDARD);
	}
}