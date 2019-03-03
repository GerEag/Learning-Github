

import org.apache.edgent.function.Function;
import org.apache.edgent.function.Functions;
import org.apache.edgent.providers.direct.DirectProvider;
import org.apache.edgent.topology.TStream;
import org.apache.edgent.topology.TWindow;
import org.apache.edgent.topology.Topology; // This is important for streaming data

import java.util.List;
import java.util.concurrent.TimeUnit;

// This script just keeps running until you give an interupt of some kind

// Must assign data type for each variable when you define it

/**
 * Hello Edgent Topology sample.
 *
 */
public class HelloEdgent {

    /**
     * Print "Hello Edgent!" as two tuples.
     * @param args command arguments
     * @throws Exception on failure
     */
    public static void main(String[] args) throws Exception {

        int sensorID=441413; // sensor ID
        String APIKey = "a32bb5e652bb1bbd93e6c7d9facf9fff";


        ReadSensor sensor = new ReadSensor(sensorID); // Program that reads temperature sensor data, returns all the attributes associated with the sensor
        WeatherSensor wsensor = new WeatherSensor(APIKey);
        DirectProvider dp = new DirectProvider();
        Topology topology = dp.newTopology(); // topology makes a stream

	// To produce/sample a new stream
        TStream<Sensor> tempReadings = topology.poll(sensor, 1, TimeUnit.MINUTES); // request sensor inform every x time units
        // refer https://docs.oracle.com/javase/8/docs/api/java/util/concurrent/TimeUnit.html?is-external=true
        // https://edgent.incubator.apache.org/javadoc/latest/org/apache/edgent/topology/Topology.html#poll-org.apache.edgent.function.Supplier-long-java.util.concurrent.TimeUnit-

        TStream<WeatherInfo> weatherReadings = topology.poll(wsensor, 1, TimeUnit.MINUTES); 
        
        
        
        TStream<Sensor> filteredReadings = tempReadings.filter(reading -> reading.getTempValue() < 50 || reading.getTempValue() > 50);
        TWindow<Sensor,Integer> window = tempReadings.last(10, Functions.unpartitioned());
        TStream<Double> averageReadings = window.aggregate((List<Sensor> list, Integer partition) -> {
              double avg = 0.0;
              for (Sensor s : list) avg += s.getTempValue();
              if (list.size() > 0) avg /= list.size();
              return avg;
            });


        // filter operation https://edgent.incubator.apache.org/javadoc/latest/org/apache/edgent/topology/TStream.html#filter-org.apache.edgent.function.Predicate-
        // reader is a temporary variable


        // perform a join operation on two streams
        //https://edgent.incubator.apache.org/javadoc/latest/org/apache/edgent/topology/TStream.html#filter-org.apache.edgent.function.Predicate-
        
        //Average Temperature from last 10 readings
        averageReadings.print();
        //filteredReadings.print();
        
        
        
        //extract Temperature from sensor and weather streams
        
        //Temperature from Sensor
        TStream<Double> sensorTemperature = tempReadings.map(t ->{
        	double d =0.0;
        	d = t.getTempValue();
        	return d;
        	});
        
        // Temperature from Weather API
        TStream<Double> weatherTemperature = weatherReadings.map(t ->{
        	double d =0.0;
        	d = t.getTemp();
        	return d;
        	});
        
        
        sensorTemperature.print();
        weatherTemperature.print();
        
        dp.submit(topology);
    }
}


