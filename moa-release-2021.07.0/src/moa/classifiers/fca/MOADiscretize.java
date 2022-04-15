package moa.classifiers.fca;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;

import moa.core.Utils;
import weka.core.OptionHandler;
import weka.filters.Filter;

public class MOADiscretize {
	// Outils utilisé pour la discretization
	private static SamoaToWekaInstanceConverter converterMoaToWeika = new SamoaToWekaInstanceConverter();
	private static WekaToSamoaInstanceConverter converterWeikaToSamoa = new WekaToSamoaInstanceConverter();
	protected static Filter m_Filter = new weka.filters.supervised.attribute.Discretize();

	private static weka.core.Instances wekaInstances;
	public MOADiscretize(Instances inst) {
		wekaInstances = converterMoaToWeika.wekaInstances(inst);

		try {
			m_Filter.setInputFormat(wekaInstances);
			wekaInstances = Filter.useFilter(wekaInstances, m_Filter);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}  
	}
	
	public Instance instanceFilter(Instance inst) throws Exception {
		weka.core.Instance newInstance = converterMoaToWeika.wekaInstance(inst);
		
	    if (m_Filter.numPendingOutput() > 0) 
	      throw new Exception("Filter output queue not empty!");
	    
	    if (!m_Filter.input(newInstance))
	      throw new Exception("Filter didn't make the test instance immediately available!");

	    m_Filter.batchFinished();
	    newInstance = m_Filter.output();
	    
		inst = converterWeikaToSamoa.samoaInstance(newInstance);
		
		return inst;
	}


	public Instances getInstances() {
		return converterWeikaToSamoa.samoaInstances(wekaInstances);
	}
	
	public Filter getFilter() {
		return m_Filter;
	}
	protected String getFilterSpec() {
		Filter c = getFilter();
		if (c instanceof OptionHandler) {
			return c.getClass().getName() + " " + Utils.joinOptions(((OptionHandler) c).getOptions());
		}
		return c.getClass().getName();
	}
	
}
