import org.apache.edgent.function.Supplier;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Date;

import org.json.JSONObject;

public class WeatherSensor implements Supplier<WeatherInfo>{
	
	String sensorAPI = "";
	String requestURL = "http://api.openweathermap.org/data/2.5/weather?q=70503,us&APPID=";
	String completeURL = requestURL+sensorAPI;
	
	public WeatherSensor(String sensorAPI) {
		this.sensorAPI = sensorAPI;
		this.completeURL = requestURL+sensorAPI;
	
	}

	
	public WeatherInfo getData() {
		WeatherInfo w = null;
		try {
			URL obj = new URL(completeURL);
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
            JSONObject mainData = myResponse.getJSONObject("main");
            double temperature = mainData.getDouble("temp");
            temperature = (((temperature - 273) * 9/5) + 32);
            double humidity = mainData.getDouble("humidity");
            w = new WeatherInfo(temperature, humidity);			
		}
		catch(Exception E) {
			E.printStackTrace();
		}
		return w;
	}
	@Override
	public WeatherInfo get() {
		WeatherInfo w = getData();
		return w;
	}
	
}