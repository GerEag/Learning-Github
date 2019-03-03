import org.apache.edgent.providers.direct.DirectProvider;
import org.apache.edgent.topology.TStream;
import org.apache.edgent.topology.Topology;



/**
 * Hello Edgent Topology sample.
 *
 */
public class HelloWorld {

    /**
     * Print "Hello Edgent!" as two tuples.
     * @param args command arguments
     * @throws Exception on failure
     */
    public static void main(String[] args) throws Exception {

        DirectProvider dp = new DirectProvider();

        Topology top = dp.newTopology();

        TStream<String> helloStream = top.strings("Hello", "Edgent!");

        helloStream.print();

        dp.submit(top);
    }
}
