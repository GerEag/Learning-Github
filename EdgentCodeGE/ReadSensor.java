import org.apache.edgent.function.Supplier;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Date;

import org.json.JSONObject;


import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSession;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;
import java.security.cert.X509Certificate;


public class ReadSensor implements Supplier<Sensor> {


    String token = "SU9UQ2xhc3M6Q1AkNTg4NiE=";
    String requestURL = "https://www.imonnit.com/json/SensorGet/";
    int sensorID = 441413;
    String completeRequest = "";
    
    
    public ReadSensor(int sensorID) {
    	this.sensorID = sensorID;
    	this.completeRequest = requestURL+token+"?sensorID="+Integer.toString(sensorID);
    }

	Sensor getData(){
        Sensor s = null;
        try {

		TrustManager[] trustAllCerts = new TrustManager[] {new X509TrustManager() {
	                public java.security.cert.X509Certificate[] getAcceptedIssuers() {
	                    return null;
	                }
	                public void checkClientTrusted(X509Certificate[] certs, String authType) {
	                }
	                public void checkServerTrusted(X509Certificate[] certs, String authType) {
	                }
	            }
	        };
	 
	        // Install the all-trusting trust manager
	        SSLContext sc = SSLContext.getInstance("SSL");
	        sc.init(null, trustAllCerts, new java.security.SecureRandom());
	        HttpsURLConnection.setDefaultSSLSocketFactory(sc.getSocketFactory());
	 
	        // Create all-trusting host name verifier
	        HostnameVerifier allHostsValid = new HostnameVerifier() {
	            public boolean verify(String hostname, SSLSession session) {
	                return true;
	            }
	        };
	 
	        // Install the all-trusting host verifier
        	HttpsURLConnection.setDefaultHostnameVerifier(allHostsValid);

            URL obj = new URL(completeRequest);
            HttpURLConnection con = (HttpURLConnection) obj.openConnection();
            //con.setRequestMethod("GET");
            //con.setRequestProperty("User-Agent", "Mozilla/5.0");
            BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
            String inputLine;
            StringBuffer response = new StringBuffer();
            while ((inputLine = in.readLine()) != null) {
                response.append(inputLine);
            }
            in.close();
            JSONObject myResponse = new JSONObject(response.toString());
            String tempString = myResponse.getJSONObject("Result").getString("CurrentReading");

            int sensorID = myResponse.getJSONObject("Result").getInt("SensorID");
            int applicationID = myResponse.getJSONObject("Result").getInt("ApplicationID");
            int csnetID = myResponse.getJSONObject("Result").getInt("CSNetID");
            String sensorName = myResponse.getJSONObject("Result").getString("SensorName");
            String lastCommDate = myResponse.getJSONObject("Result").getString("LastCommunicationDate");
            String nextCommDate = myResponse.getJSONObject("Result").getString("NextCommunicationDate");
            String lastMsgGUID = myResponse.getJSONObject("Result").getString("LastDataMessageMessageGUID");
            int powerSrcId = myResponse.getJSONObject("Result").getInt("PowerSourceID");
            int status = myResponse.getJSONObject("Result").getInt("Status");
            boolean canUpdate = myResponse.getJSONObject("Result").getBoolean("CanUpdate");
            double batteryLevel = myResponse.getJSONObject("Result").getDouble("BatteryLevel");
            double signalStrength = myResponse.getJSONObject("Result").getDouble("SignalStrength");
            boolean alertsActive = myResponse.getJSONObject("Result").getBoolean("AlertsActive");
            String checkDigit = myResponse.getJSONObject("Result").getString("CheckDigit");
            int AccountID = myResponse.getJSONObject("Result").getInt("AccountID");
            int monnitAppID = myResponse.getJSONObject("Result").getInt("MonnitApplicationID");

            
            String humidityString = tempString.substring(0,tempString.indexOf("%"));
            double humidityValue = Double.parseDouble(humidityString.trim());
            
            tempString = tempString.substring(tempString.indexOf("@")+2,tempString.indexOf("F")-2);
            double tempValue = Double.parseDouble(tempString);
            
            
            s = new Sensor(sensorID, applicationID, csnetID, sensorName, lastCommDate, nextCommDate, lastMsgGUID, powerSrcId, status, canUpdate, tempValue, humidityValue, batteryLevel, signalStrength, alertsActive, checkDigit, AccountID, monnitAppID);
        }
        catch (Exception E){
            E.printStackTrace();
        }
        return s;
    }

    public Sensor get(){
        Sensor val = getData();
        return val;
    }


}
